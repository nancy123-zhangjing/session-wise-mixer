"""Bounded CPU-only full-checkpoint audit of completed precision-v2 trials."""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time

import numpy as np
import torch

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    assert 1 <= args.max_runs <= 16 and 1 <= args.max_seconds <= 300
    assert args.batch_id.isalnum()
    base = Path(args.base).resolve()
    code = base / "code"
    pipeline = code / "outputs/stage06_continuous/pipeline_20261016_precision_v2"
    sys.path.insert(0, str(code))
    from scripts.continuous.model import ContinuousRecommender
    from scripts.continuous.trainer import model_config, source_identity
    from scripts.continuous.protocol import canonical_hash
    from scripts.checks.verify_continuous_campaign import audit_curves
    torch.set_num_threads(2)
    assert source_identity(code)["source_sha256"] == SOURCE
    destination = base / "delivery/checkpoint_payload_audits"
    destination.mkdir(exist_ok=True)
    batch_path = destination / ("batch_" + args.batch_id + ".json")
    assert not batch_path.exists(), "Audit batch already exists; inspect its receipt"
    ledger_path = destination / "ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"source_sha256": SOURCE, "trials": {}}
    assert ledger["source_sha256"] == SOURCE
    queue = read(pipeline / "campaign/queue.json")
    assert queue["source_sha256"] == SOURCE
    manifest = read(code / "configs/06_continuous_baseline/search_manifest.json")
    expected = {t["trial_id"]: t for t in manifest["trials"]}
    prepared = base.with_name("goal_20261016") / "inputs/prepared/manifest.json"
    assert sha(prepared) == "6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa"
    data = read(prepared)["files"]
    pending = [t for t in queue["tasks"] if t["status"] == "succeeded" and t["task_id"] not in ledger["trials"]]
    groups = {m: [t for t in pending if t["config"]["model"] == m] for m in ("sasrec_fullce", "esasrec")}
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    records, started = [], time.monotonic()
    for task in ordered[:args.max_runs]:
        if time.monotonic() - started > args.max_seconds:
            break
        output = Path(task["output"])
        assert output.is_relative_to(pipeline / "campaign/runs")
        config = task["config"]
        meta, complete, current = read(output / "run.json"), read(output / "COMPLETE.json"), read(output / "CURRENT.json")
        assert complete == task["result"] and complete["status"] == "SUCCEEDED" and not complete["smoke"]
        assert config == expected[task["trial_id"]]["config"] == meta["config"]
        assert canonical_hash(config) == task["config_sha256"] == expected[task["trial_id"]]["config_sha256"]
        protocol = task["protocol"]
        binding = {"config_sha256": task["config_sha256"], "seed": task["seed"], "protocol": protocol,
                   "train_sha256": data[protocol + "/train.jsonl"]["sha256"],
                   "validation_sha256": data[protocol + "/validation.jsonl"]["sha256"],
                   "source_sha256": SOURCE, "precision": "bf16", "smoke": False}
        assert complete["binding"] == meta["binding"] == binding
        assert meta["attention_precision_policy"] == "fp32_math" and meta["device"] == "cuda:0"
        assert meta["cuda_visible_devices"] == meta["gpu_uuid"] and "H100" in meta["gpu"]
        assert meta["source"]["git_head"] == "bbaa45b515a15912a04665721e0a5c43d64dc9e9"
        n_train = data[protocol + "/train.jsonl"]["records"]
        rows = list(csv.DictReader((output / "curves.csv").open()))
        audit_curves(config, complete, rows, n_train, 22363)
        checks = {}
        model = ContinuousRecommender(model_config(config)).cpu()
        assert sum(p.numel() for p in model.parameters()) == complete["parameter_count"]
        payload_hashes = {}
        for label in ("best", "last"):
            alias = output / (label + ".pt")
            snapshot = output / current[label]["file"]
            assert snapshot.resolve().is_relative_to((output / "checkpoint_epochs").resolve())
            inode = (snapshot.stat().st_dev, snapshot.stat().st_ino)
            if inode not in payload_hashes:
                payload_hashes[inode] = sha(snapshot)
            actual_sha = payload_hashes[inode]
            assert actual_sha == current[label]["sha256"] == complete[label + "_sha256"]
            assert read(alias.with_suffix(".receipt.json"))["sha256"] == actual_sha
            assert alias.samefile(snapshot) or sha(alias) == actual_sha
            state = torch.load(snapshot, map_location="cpu", weights_only=False)
            assert state["binding"] == binding and state["config"] == config
            epoch = complete["best_epoch"] if label == "best" else complete["epochs"]
            assert state["epoch"] == epoch and state["updates"] == epoch * math.ceil(n_train / config["batch_size"])
            assert len(state["curves"]) == epoch
            for saved, raw in zip(state["curves"], rows[:epoch]):
                assert saved.keys() == raw.keys()
                assert all(math.isfinite(float(v)) and float(v) == float(raw[k]) for k, v in saved.items())
            assert state["bad_epochs"] == epoch - state["best_epoch"]
            assert all(torch.isfinite(t).all().item() for t in state["model"].values())
            model.load_state_dict(state["model"], strict=True)
            assert torch.equal(model.item_embedding.weight[0], torch.zeros_like(model.item_embedding.weight[0]))
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                         betas=tuple(config["adam_betas"]), weight_decay=config["weight_decay"])
            optimizer.load_state_dict(state["optimizer"])
            assert len(optimizer.state) == len(list(model.parameters()))
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert group["lr"] == config["lr"] and group["betas"] == tuple(config["adam_betas"])
            assert group["weight_decay"] == config["weight_decay"]
            for parameter, values in optimizer.state.items():
                assert values["exp_avg"].shape == values["exp_avg_sq"].shape == parameter.shape
                assert float(values["step"]) == state["updates"]
                assert all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all().item() for v in values.values())
            assert state["scheduler"]["last_epoch"] == epoch
            assert state["scheduler"]["_last_lr"] == [config["lr"]]
            rng = state["random_state"]
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator(device="cpu").set_state(rng["torch"].cpu())
            assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8 and rng["cuda"][0].numel() > 0
            checks[label] = {"sha256": actual_sha, "epoch": epoch, "updates": state["updates"],
                             "model_parameter_shapes_and_pad": "PASS", "model_optimizer_tensors_finite": True,
                             "optimizer_shapes_steps_hyperparameters": "PASS", "scheduler_epoch_lr": "PASS",
                             "checkpoint_curves_match_csv": True, "python_numpy_cpu_rng_state_acceptance": "PASS",
                             "cuda_rng_byte_state_present": True}
            del state, optimizer
            gc.collect()
        del model
        gc.collect()
        record = {"status": "PASS", "task_id": task["task_id"], "phase": task["phase"], "seed": task["seed"],
                  "protocol": protocol, "config_sha256": task["config_sha256"], "source_sha256": SOURCE,
                  "gpu_uuid_at_training": meta["gpu_uuid"], "checks": checks}
        ledger["trials"][task["task_id"]] = record
        ledger["updated_unix"] = time.time()
        atomic(ledger_path, ledger)
        records.append(record)
        print(json.dumps({"audited_task": task["task_id"], "cumulative_trials": len(ledger["trials"])}), flush=True)
    assert source_identity(code)["source_sha256"] == SOURCE
    proof = {"status": "PASS", "source_sha256": SOURCE, "new_trials": len(records),
             "cumulative_trials": len(ledger["trials"]), "records": records,
             "elapsed_seconds": time.monotonic() - started, "max_runs": args.max_runs,
             "max_seconds_before_next_trial": args.max_seconds, "audit_script_sha256": sha(__file__),
             "scope": "Read-only completed best/last payloads on CPU; no optimizer updates, validation or test forwards; CUDA RNG checked as bytes, not replayed on GPU"}
    atomic(batch_path, proof)
    print(json.dumps({k: v for k, v in proof.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
