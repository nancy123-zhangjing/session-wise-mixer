"""CPU-only, one-time recovery preparation after proven terminal platform Job.

Preserve the old pipeline. Retain completed results and resume interrupted states
only with the same frozen training code/data/config/seed. The new supervisor is
separately versioned and does not replace any training implementation file.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "previous", "output", "evidence", "deployment-manifest"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root, previous, output, evidence = map(lambda s: Path(s).resolve(), (args.root, args.previous, args.output, args.evidence))
    assert previous.parent == output.parent == root / "outputs/stage06_continuous"
    assert previous.name == "pipeline_20261016_precision_v2" and not output.exists()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from scripts.continuous.campaign import verify_result
    from scripts.continuous.checkpoints import load_checkpoint, repair_committed
    from scripts.continuous.model import ContinuousRecommender
    from scripts.continuous.protocol import atomic_json
    from scripts.continuous.trainer import model_config, source_identity
    torch.set_num_threads(2)
    assert source_identity(root)["source_sha256"] == SOURCE
    events, logs = read(evidence / "platform_events.json"), read(evidence / "platform_logs.json")
    status = read(evidence / "platform_status.json")
    assert status["success"] and status["data"]["status"] == "FAILED"
    assert status["data"]["name"] == "nancy-stage06-20261016-8h100-precision-v2"
    assert events["success"] and logs["success"] and not logs["data"]["truncated"]
    items = events["data"]["items"]
    assert any(e["reason"] == "PyTorchJobFailed" for e in items)
    assert any(e["reason"] == "SuccessfulDeletePod" for e in items)
    assert any("Worker failed: search__esasrec_26__seed2024" in row["message"] for row in logs["data"]["logs"])
    old = read(previous / "campaign/queue.json")
    assert old["source_sha256"] == SOURCE and old["status"] == "FAILED"
    assert old["workers"]["4"]["identity"]["command"] == ""
    assert not old["top3"] and not old["selected_configs"] and not (previous / "campaign/FROZEN_SELECTION.json").exists()
    assert len(old["tasks"]) == 64
    completed = [t for t in old["tasks"] if t["status"] == "succeeded"]
    assert len(completed) == 38
    ledger = read(previous.parents[3] / "delivery/checkpoint_payload_audits/ledger.json")
    assert ledger["source_sha256"] == SOURCE and set(ledger["trials"]) == {t["task_id"] for t in completed}
    deployment = read(args.deployment_manifest)
    assert deployment["training_source_sha256"] == SOURCE
    state = copy.deepcopy(old)
    new_campaign = output / "campaign"
    new_campaign.mkdir(parents=True)
    (new_campaign / "tasks").mkdir()
    before, recovered = {}, []
    worker_outputs = {w["task_id"]: Path(w["output"]) for w in old["workers"].values()}
    for task in state["tasks"]:
        if task["status"] == "pending":
            continue
        source = Path(task.get("output") or worker_outputs[task["task_id"]])
        assert source.resolve().is_relative_to(previous / "campaign/runs") and not source.is_symlink()
        before[task["task_id"]] = {name: sha(source/name) for name in ("run.json", "CURRENT.json", "curves.csv", "best.pt", "last.pt")}
        if task["status"] == "succeeded":
            assert verify_result(task, source, SOURCE) == task["result"]
            target = new_campaign / "runs" / task["task_id"] / "attempt_0"
            # Completed payloads are immutable; hard links retain original bytes.
            shutil.copytree(source, target, copy_function=os.link)
            task["output"] = str(target)
            continue
        assert task["task_id"] in worker_outputs and task["attempt"] == 0
        target = new_campaign / "runs" / task["task_id"] / "attempt_1"
        # Mutable CSV/JSON are independent copies. Immutable checkpoint payloads
        # are linked and are never edited in place by the frozen trainer.
        def safe_copy(src, dst):
            return os.link(src, dst) if Path(src).suffix == ".pt" else shutil.copyfile(src, dst)
        shutil.copytree(source, target, copy_function=safe_copy)
        pointer = repair_committed(target)
        model = ContinuousRecommender(model_config(task["config"])).cpu()
        checks = {}
        for label in ("best", "last"):
            saved = load_checkpoint(target / (label + ".pt"), "cpu")
            assert saved["binding"]["source_sha256"] == SOURCE
            assert saved["binding"]["config_sha256"] == task["config_sha256"] and saved["binding"]["seed"] == task["seed"]
            assert saved["config"] == task["config"]
            model.load_state_dict(saved["model"], strict=True)
            assert all(torch.isfinite(t).all().item() for t in saved["model"].values())
            optimizer = torch.optim.Adam(model.parameters(), lr=task["config"]["lr"], betas=(.9,.98), weight_decay=task["config"]["weight_decay"])
            optimizer.load_state_dict(saved["optimizer"])
            for parameter, values in optimizer.state.items():
                assert values["exp_avg"].shape == values["exp_avg_sq"].shape == parameter.shape
                assert float(values["step"]) == saved["updates"]
                assert all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all().item() for v in values.values())
            assert saved["scheduler"]["last_epoch"] == saved["epoch"]
            random.Random().setstate(saved["random_state"]["python"])
            np.random.RandomState().set_state(saved["random_state"]["numpy"])
            torch.Generator().set_state(saved["random_state"]["torch"].cpu())
            assert len(saved["random_state"]["cuda"]) == 1 and saved["random_state"]["cuda"][0].dtype == torch.uint8
            assert sha(target / (label + ".pt")) == before[task["task_id"]][label + ".pt"]
            checks[label] = {"epoch": saved["epoch"], "updates": saved["updates"], "sha256": sha(target/(label+".pt")), "cpu_state_acceptance": "PASS", "cuda_rng_bytes_present": True}
            del saved, optimizer
        task.update(status="pending", output=str(target), attempt=1,
                    recovery_kind="supervisor_defect_same_training_implementation", resume_from_attempt=str(source))
        task.pop("exit_code", None)
        recovered.append({"task_id": task["task_id"], "source": str(source), "destination": str(target), "checks": checks})
        del model
    assert len(recovered) == 8
    state.update(status="PREPARED_SUPERVISOR_RECOVERY", workers={}, owner=None)
    state["events"].append({"type": "supervisor_defect_recovery", "time": time.time(),
        "previous_queue_sha256": sha(previous / "campaign/queue.json"), "evidence": str(evidence),
        "retained_completed": 38, "resumed_tasks": [r["task_id"] for r in recovered],
        "training_source_unchanged": SOURCE, "deployment_manifest_sha256": sha(args.deployment_manifest)})
    atomic_json(new_campaign / "queue.json", state)
    shutil.copytree(previous / "initial_benchmarks", output / "initial_benchmarks")
    shutil.copytree(previous / "resource_attempts", output / "resource_attempts")
    for task_id, hashes in before.items():
        old_task = next(t for t in old["tasks"] if t["task_id"] == task_id)
        source = Path(old_task.get("output") or worker_outputs[task_id])
        assert all(sha(source / name) == digest for name, digest in hashes.items())
    assert source_identity(root)["source_sha256"] == SOURCE
    proof = {"status": "PASS", "cpu_only": True, "training_source_sha256": SOURCE,
        "previous_pipeline": str(previous), "output": str(output), "retained_completed": 38,
        "resumed_tasks": recovered, "pending_unstarted": 18, "required_total_runs": 89,
        "previous_artifacts_unchanged": True, "test_unlocked": False,
        "deployment_manifest_sha256": sha(args.deployment_manifest), "prepare_script_sha256": sha(__file__)}
    atomic_json(output / "RECOVERY_PREPARED.json", proof)
    print(json.dumps({k:v for k,v in proof.items() if k != "resumed_tasks"}), flush=True)


if __name__ == "__main__":
    main()
