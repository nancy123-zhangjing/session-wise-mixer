"""Cumulative independent audit of hash-chained precision-v2 metadata batches."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.checks.verify_continuous_campaign import audit_curves, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifests = sorted(HERE.glob("completed_metadata_precision_v2_batch[0-9][0-9]/manifest.json"))
    assert manifests
    expected = {t["trial_id"]: t for t in read(ROOT / "configs/06_continuous_baseline/search_manifest.json")["trials"]}
    data_path = ROOT / "outputs/stage06_continuous/data_20261015_a/manifest.json"
    assert sha(data_path) == "6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa"
    data = read(data_path)["files"]
    gpu = read(HERE / "gpu_precision_v2_20261016_01/smoke/GPU_GATE.json")
    source = gpu["source_sha256"]
    assert source == "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
    gpu_ids = {r["uuid"] for r in gpu["inventory"]}
    recovered_gate = HERE / "scheduler_failure_20261016_01/gpu_gate_supervisor_v3/smoke/GPU_GATE.json"
    if recovered_gate.exists():
        extra = read(recovered_gate)
        assert extra["status"] == "PASS" and extra["source_sha256"] == source
        assert len(extra["inventory"]) == 8 and len({r["uuid"] for r in extra["inventory"]}) == 8
        gpu_ids.update(r["uuid"] for r in extra["inventory"])
    records, bindings, previous_manifest, verified_files = [], {}, None, 0
    for index, manifest_path in enumerate(manifests, 1):
        batch = read(manifest_path)
        assert batch["source_sha256"] == source
        if previous_manifest:
            assert batch["previous_batch_sha256"] == sha(previous_manifest)
        archive = ROOT / f"archive/server_sync/stage06_continuous_precision_v2_20261016/completed_metadata_batch{index:02d}"
        for relative, entry in batch["files"].items():
            payload = (archive / relative).read_bytes()
            assert len(payload) == entry["bytes"] and hashlib.sha256(payload).hexdigest() == entry["sha256"]
        verified_files += len(batch["files"])
        relative = batch.get("pipeline_relative", "code/outputs/stage06_continuous/pipeline_20261016_precision_v2")
        assert relative in {"code/outputs/stage06_continuous/pipeline_20261016_precision_v2",
                            "code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3"}
        campaign = archive / relative / "campaign"
        queue = read(campaign / "queue.json")
        assert queue["source_sha256"] == source and len(queue["tasks"]) <= 89
        completed = {t["task_id"]: t for t in queue["tasks"] if t["status"] == "succeeded"}
        new_ids = batch["completed_task_ids"]
        assert len(new_ids) == len(set(new_ids)) and not set(new_ids) & bindings.keys()
        assert set(completed) == set(bindings) | set(new_ids)
        for old_id, result_hash in bindings.items():
            assert fingerprint(completed[old_id]["result"]) == result_hash
        for task_id in new_ids:
            task = completed[task_id]
            config = task["config"]
            run = campaign / "runs" / task_id / f"attempt_{task['attempt']}"
            meta, result, current = read(run / "run.json"), read(run / "COMPLETE.json"), read(run / "CURRENT.json")
            assert config == meta["config"] == expected[task["trial_id"]]["config"]
            assert fingerprint(config) == task["config_sha256"] == expected[task["trial_id"]]["config_sha256"]
            seed_sets = {"search": {2024}, "confirmation": {2025, 2026},
                         "formal": set(range(3031, 3036)), "legacy": {3031, 3032, 3033}}
            assert task["seed"] in seed_sets[task["phase"]]
            protocol = "legacy_prefix_v1" if task["phase"] == "legacy" else "full_train_prefix_v1"
            assert task["protocol"] == protocol
            if task["phase"] == "legacy":
                assert config["model"] == "sasrec_fullce"
            binding = {"config_sha256": task["config_sha256"], "seed": task["seed"], "protocol": protocol,
                       "train_sha256": data[protocol + "/train.jsonl"]["sha256"],
                       "validation_sha256": data[protocol + "/validation.jsonl"]["sha256"],
                       "source_sha256": source, "precision": "bf16", "smoke": False}
            assert result == task["result"] and result["status"] == "SUCCEEDED" and not result["smoke"]
            assert result["binding"] == meta["binding"] == binding
            assert meta["source"]["git_head"] == "bbaa45b515a15912a04665721e0a5c43d64dc9e9"
            assert meta["attention_precision_policy"] == "fp32_math" and meta["device"] == "cuda:0"
            assert meta["torch_version"] == "2.6.0+cu118" and meta["numpy_version"] == "2.2.6"
            assert meta["cuda_visible_devices"] == meta["gpu_uuid"] in gpu_ids and "H100" in meta["gpu"]
            effective_uuid = meta["gpu_uuid"]
            if task["attempt"]:
                resume = [json.loads(line) for line in (run / "resume_events.jsonl").read_text().splitlines()]
                assert resume and resume[-1]["cuda_visible_devices"] == resume[-1]["gpu_uuid"]
                effective_uuid = resume[-1]["gpu_uuid"]
            assert effective_uuid in gpu_ids
            if relative.endswith("_supervisor_v3"):
                deployment_path = HERE / "scheduler_failure_20261016_01/DEPLOYMENT_MANIFEST.json"
                assert queue["deployment_manifest_sha256"] == sha(deployment_path)
                assert queue["supervisor_sha256"] == read(deployment_path)["supervisor_sha256"]
                assert effective_uuid == queue["gpu_inventory"][task["gpu"]]["uuid"]
            n_train = data[protocol + "/train.jsonl"]["records"]
            rows = list(csv.DictReader((run / "curves.csv").open()))
            curve = audit_curves(config, result, rows, n_train, 22363)
            if result["epochs"] < config["max_epochs"]:
                assert result["epochs"] - result["best_epoch"] == config["patience"]
            norms = [float(r["grad_norm_last"]) for r in rows]
            assert all(math.isfinite(n) and n >= 0 for n in norms)
            for label in ("best", "last"):
                assert result[label + "_sha256"] == read(run / (label + ".receipt.json"))["sha256"] == current[label]["sha256"]
            bindings[task_id] = fingerprint(result)
            records.append({"task_id": task_id, "phase": task["phase"], "seed": task["seed"], "protocol": protocol,
                            **curve, "maximum_logged_epoch_last_gradient_norm": max(norms),
                            "bindings_and_checkpoint_receipts": "PASS"})
        previous_manifest = manifest_path
    report = {"status": "PASS", "source_sha256": source, "batches": len(manifests),
              "completed_trials": len(records), "verified_metadata_files": verified_files,
              "latest_snapshot_counts": dict(Counter(t["status"] for t in queue["tasks"])),
              "trials": records, "test_metrics_read": False,
              "scope": "Metadata and epoch curves only; gradient samples are epoch-last only; full payload tensor audits are recorded separately under checkpoint_payload_precision_v2",
              "audit_script_sha256": sha(__file__)}
    destination = manifests[-1].parent / "AUDIT.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}))


if __name__ == "__main__":
    main()
