"""Independent stdlib recheck of archived completed search metadata."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.checks.verify_continuous_campaign import audit_curves, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    archive = ROOT / "archive/server_sync/stage06_continuous_precision_v2_20261016/completed_metadata_batch01"
    manifest = read(ROOT / ".runtime/continuous_goal_20261015/completed_metadata_precision_v2_batch01.json")
    for relative, entry in manifest["files"].items():
        data = (archive / relative).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    campaign = archive / "code/outputs/stage06_continuous/pipeline_20261016_precision_v2/campaign"
    queue = read(campaign / "queue.json")
    expected = {t["trial_id"]: t for t in read(ROOT / "configs/06_continuous_baseline/search_manifest.json")["trials"]}
    inventory = read(HERE.parent / "gpu_precision_v2_20261016_01/smoke/GPU_GATE.json")["inventory"]
    completed = [t for t in queue["tasks"] if t["status"] == "succeeded"]
    assert {t["task_id"] for t in completed} == set(manifest["completed_task_ids"])
    records = []
    for task in completed:
        run = campaign / "runs" / task["task_id"] / "attempt_0"
        config = task["config"]
        meta, result, current = read(run / "run.json"), read(run / "COMPLETE.json"), read(run / "CURRENT.json")
        rows = list(csv.DictReader((run / "curves.csv").open()))
        assert config == expected[task["trial_id"]]["config"] == meta["config"]
        assert task["config_sha256"] == fingerprint(config) == expected[task["trial_id"]]["config_sha256"]
        assert task["seed"] == 2024 and task["phase"] == "search" and task["protocol"] == "full_train_prefix_v1"
        assert result == task["result"] and result["status"] == "SUCCEEDED" and not result["smoke"]
        assert meta["binding"] == result["binding"]
        assert meta["binding"]["config_sha256"] == task["config_sha256"]
        assert meta["binding"]["seed"] == 2024 and meta["binding"]["protocol"] == "full_train_prefix_v1"
        assert meta["binding"]["train_sha256"] == "894bed89c76a08907ef428c86e252a57076917c893b877b899cfd4f42ec3f78f"
        assert meta["binding"]["validation_sha256"] == "4400827739f741e61f7dfc48f7e7221a1d0b2a278d6138adbf360e1c7729027f"
        assert not meta["binding"]["smoke"]
        assert meta["binding"]["source_sha256"] == queue["source_sha256"] == manifest["source_sha256"]
        assert meta["binding"]["source_sha256"] == "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
        assert meta["source"]["git_head"] == "bbaa45b515a15912a04665721e0a5c43d64dc9e9"
        assert meta["attention_precision_policy"] == "fp32_math" and meta["binding"]["precision"] == "bf16"
        assert meta["torch_version"] == "2.6.0+cu118" and meta["numpy_version"] == "2.2.6"
        assert meta["cuda_visible_devices"] == meta["gpu_uuid"] in {r["uuid"] for r in inventory}
        curve = audit_curves(config, result, rows, 131413, 22363)
        if result["epochs"] < 300:
            assert result["epochs"] - result["best_epoch"] == 50
        norms = [float(r["grad_norm_last"]) for r in rows]
        assert all(math.isfinite(n) and n >= 0 for n in norms)
        for label in ("best", "last"):
            receipt = read(run / (label + ".receipt.json"))
            assert result[label + "_sha256"] == receipt["sha256"] == current[label]["sha256"]
        records.append({"task_id": task["task_id"], **curve,
                        "maximum_logged_epoch_last_gradient_norm": max(norms),
                        "config_data_source_device_binding": "PASS", "checkpoint_receipt_agreement": "PASS"})
    report = {"status": "PASS", "scope": "Archived metadata and epoch curves; full checkpoint tensors audited separately for first completed trial",
              "source_sha256": queue["source_sha256"], "verified_metadata_files": len(manifest["files"]),
              "completed_trials": len(records), "trials": records,
              "test_metrics_read": False,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    shutil.copyfile(ROOT / ".runtime/continuous_goal_20261015/completed_metadata_precision_v2_batch01.json", HERE / "manifest.json")
    (HERE / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
