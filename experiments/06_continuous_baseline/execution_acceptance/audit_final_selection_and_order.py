"""Local independent validation-only selection, ordering and raw-curve audit.

Reads the immutable evidence archive; imports no training or selection helpers.
Archive verification here covers evidence only. Full payload backup has its own audit.
"""
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARTS = ROOT / "outputs/stage06_continuous/final_delivery_20261016/backup_parts"
OUT = HERE / "final_delivery_20261016"
SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), (a, b)


def main():
    manifest_bytes = (PARTS / "FINAL_BACKUP_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    record = next(r for r in manifest["archives"] if r["file"] == "evidence-01.tar.gz")
    path = PARTS / record["file"]
    assert path.stat().st_size == record["bytes"] and digest(path.read_bytes()) == record["sha256"]
    files = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            assert member.isfile() and member.name not in files
            content = archive.extractfile(member).read()
            expected = manifest["files"][member.name]
            assert expected["archive"] == record["file"]
            assert len(content) == expected["bytes"] and digest(content) == expected["sha256"]
            files[member.name] = content
    assert set(files) == {n for n, r in manifest["files"].items() if r["archive"] == record["file"]}
    read = lambda n: json.loads(files[n])
    queue = read("campaign/queue.json")
    frozen = read("campaign/FROZEN_SELECTION.json")
    search = read("source/configs/06_continuous_baseline/search_manifest.json")
    tasks = queue["tasks"]
    assert len(tasks) == 89 and not queue["workers"] and queue["status"] == "TRAINING_COMPLETE"
    assert Counter(t["phase"] for t in tasks) == {"search": 64, "confirmation": 12, "formal": 10, "legacy": 3}
    assert all(t["status"] == "succeeded" and t["result"]["binding"]["source_sha256"] == SOURCE for t in tasks)
    assert frozen["source_sha256"] == queue["source_sha256"] == manifest["source_sha256"] == SOURCE
    assert canonical(search) == queue["manifest_sha256"]
    assert canonical([{"task_id": t["task_id"], "result": t["result"]} for t in tasks]) == frozen["training_results_sha256"]
    formal_details, selections = [], {}
    for model in ("sasrec_fullce", "esasrec"):
        candidates = [t for t in tasks if t["phase"] == "search" and t["config"]["model"] == model]
        assert len(candidates) == 32
        declared = {t["trial_id"]: t for t in search["trials"] if t["config"]["model"] == model}
        assert len(declared) == 32
        for task in candidates:
            assert task["seed"] == 2024 and task["config"] == declared[task["trial_id"]]["config"]
        top3 = sorted(candidates, key=lambda t: (-t["result"]["best_validation"]["NDCG@10"],
                      -t["result"]["best_validation"]["HR@10"], t["result"]["parameter_count"], t["trial_id"]))[:3]
        assert [t["trial_id"] for t in top3] == queue["top3"][model]
        averages = []
        for first in top3:
            group = [t for t in tasks if t["trial_id"] == first["trial_id"] and t["phase"] in {"search", "confirmation"}]
            assert len(group) == 3 and {t["seed"] for t in group} == {2024, 2025, 2026}
            assert all(t["config"] == first["config"] for t in group)
            averages.append({"trial_id": first["trial_id"], "NDCG@10": statistics.mean(t["result"]["best_validation"]["NDCG@10"] for t in group),
                             "HR@10": statistics.mean(t["result"]["best_validation"]["HR@10"] for t in group),
                             "parameters": first["result"]["parameter_count"]})
        chosen = min(averages, key=lambda r: (-r["NDCG@10"], -r["HR@10"], r["parameters"], r["trial_id"]))
        locked = frozen["models"][model]
        assert chosen["trial_id"] == locked["trial_id"] == queue["selected_configs"][model]["trial_id"]
        for metric in ("NDCG@10", "HR@10"):
            close(chosen[metric], locked["confirmation_mean"][metric])
        formal = [t for t in tasks if t["phase"] == "formal" and t["config"]["model"] == model]
        assert len(formal) == 5 and {t["seed"] for t in formal} == set(range(3031, 3036))
        assert all(t["trial_id"] == chosen["trial_id"] and t["config"] == locked["config"] for t in formal)
        means = {metric: statistics.mean(t["result"]["best_validation"][metric] for t in formal) for metric in ("NDCG@10", "HR@10")}
        for metric, value in means.items():
            close(value, locked["formal_validation_mean"][metric])
        selections[model] = {"trial_id": chosen["trial_id"], "confirmation_means": averages,
                             "formal_validation_mean": means, "parameters": chosen["parameters"]}
    selected = min(selections, key=lambda m: (-selections[m]["formal_validation_mean"]["NDCG@10"],
                   -selections[m]["formal_validation_mean"]["HR@10"], selections[m]["parameters"], m))
    assert selected == frozen["selected_model"] == "sasrec_fullce"
    for task in tasks:
        prefix = f"campaign/runs/{task['task_id']}/attempt_{task['attempt']}"
        curves = list(csv.DictReader(io.StringIO(files[prefix + "/curves.csv"].decode())))
        result = task["result"]
        assert len(curves) == result["epochs"] and int(curves[-1]["updates"]) == result["updates"]
        best = next(r for r in curves if int(r["epoch"]) == result["best_epoch"])
        close(float(best["validation_NDCG@10"]), result["best_validation"]["NDCG@10"])
        close(float(best["validation_NLL"]), result["best_validation"]["NLL"])
        assert all(math.isfinite(float(r[k])) for r in curves for k in ("train_loss", "validation_NLL", "grad_norm_last"))
        if task["phase"] in {"formal", "legacy"}:
            formal_details.append({"task_id": task["task_id"], "model": task["config"]["model"], "phase": task["phase"], "seed": task["seed"],
                "epochs": result["epochs"], "best_epoch": result["best_epoch"], "updates": result["updates"],
                "training_seconds": result["elapsed_seconds"], "best_epoch_train_loss": float(best["train_loss"]),
                "best_validation_NLL": float(best["validation_NLL"]), "last_epoch_train_loss": float(curves[-1]["train_loss"]),
                "last_validation_NLL": float(curves[-1]["validation_NLL"]), "source_curve_sha256": digest(files[prefix + "/curves.csv"])})
    events = [json.loads(line) for line in files["pipeline/pipeline_events.jsonl"].splitlines()]
    phases = {e["phase"]: e["updated_unix"] for e in events}
    assert all(a["updated_unix"] < b["updated_unix"] for a, b in zip(events, events[1:]))
    frozen_time = manifest["files"]["campaign/FROZEN_SELECTION.json"]["source_mtime_ns"] / 1e9
    finished = max(t["finished_at"] for t in tasks)
    assert finished <= frozen_time <= phases["final_test"] < phases["final_exclusive_benchmark"] < phases["gpu_work_complete"]
    evaluations = read("campaign/evaluation_index.json")
    assert len(evaluations) == 13
    times = []
    for identity in evaluations:
        p = "campaign/evaluations/" + identity + "/result.json"
        epoch = manifest["files"][p]["source_mtime_ns"] / 1e9
        assert phases["final_test"] < epoch < phases["final_exclusive_benchmark"]
        times.append(epoch)
    benchmark_times = {m: manifest["files"][f"campaign/final_benchmarks/{m}/result.json"]["source_mtime_ns"] / 1e9 for m in selections}
    assert phases["final_exclusive_benchmark"] < benchmark_times["sasrec_fullce"] < benchmark_times["esasrec"] < phases["gpu_work_complete"]
    source = files["orchestration_v3/scripts/train/continuous_queue_supervisor.py"]
    assert digest(source) == "578fdcaed74a47ac470abc393af62fa29e9bc2bd676fbe97533e1a49ad98dd91"
    assert b'checkpoints, inventory[0]["uuid"]' in source
    for model in selections:
        bench = read(f"campaign/final_benchmarks/{model}/result.json")
        assert bench["binding"]["seed"] == 3031 and bench["binding"]["config_sha256"] == frozen["models"][model]["config_sha256"]
    detail_path = OUT / "FORMAL_TRAINING_DETAILS.csv"
    with detail_path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(formal_details[0]))
        writer.writeheader()
        writer.writerows(formal_details)
    proof = {"status": "PASS", "source_sha256": SOURCE, "verified_training_runs": 89, "raw_curves": 89,
        "validation_only_selected_model": selected, "selection": selections, "last_training_finished_unix": finished,
        "frozen_selection_original_mtime_unix": frozen_time, "pipeline_phase_times": phases,
        "latest_test_result_original_mtime_unix": max(times), "benchmark_result_original_mtimes_unix": benchmark_times,
        "configured_benchmark_gpu_uuid": queue["gpu_inventory"][0]["uuid"], "workers_at_queue_completion": 0,
        "evidence_archive_sha256": record["sha256"], "backup_manifest_sha256": digest(manifest_bytes),
        "frozen_selection_sha256": digest(files["campaign/FROZEN_SELECTION.json"]), "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "formal_training_details_sha256": digest(detail_path.read_bytes()),
        "scope": "Recomputed from validation results without training helper imports. Original server mtimes establish ordering; frozen source binds sequential GPU0 benchmark calls after queue drain and synchronous evaluations. No continuous process trace or kernel profile was collected, so per-kernel dtype and absence of external loads are not independently proven."}
    with (OUT / "FINAL_SELECTION_AND_ORDER_AUDIT.json").open("x", encoding="utf8") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
