"""Recompute the complete search and both real Top-3 populations from archived validation results."""
from pathlib import Path
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
read = lambda path: json.loads(path.read_text(encoding="utf8"))
manifest = read(HERE / "manifest.json")
archive = ROOT / "archive/server_sync/stage06_continuous_precision_v2_20261016/completed_metadata_batch16"
queue_path = archive / manifest["pipeline_relative"] / "campaign/queue.json"
queue_sha = hashlib.sha256(queue_path.read_bytes()).hexdigest()
assert queue_sha == manifest["files"][queue_path.relative_to(archive).as_posix()]["sha256"]
state = read(queue_path)
assert state["source_sha256"] == manifest["source_sha256"]
assert len(state["tasks"]) == len({task["task_id"] for task in state["tasks"]}) <= 89
assert read(HERE / "AUDIT.json")["completed_trials"] == 70
assert not any(task["status"] in {"failed", "interrupted"} for task in state["tasks"])
families = {}
for model in ("sasrec_fullce", "esasrec"):
    tasks = [task for task in state["tasks"] if task["config"]["model"] == model]
    search = [task for task in tasks if task["phase"] == "search"]
    confirmations = [task for task in tasks if task["phase"] == "confirmation"]
    formal = [task for task in tasks if task["phase"] == "formal"]
    legacy = [task for task in tasks if task["phase"] == "legacy"]
    assert len(search) == 32 and {task["seed"] for task in search} == {2024}
    completed = [task for task in search if task["status"] == "succeeded"]
    result = {"completed_searches": len(completed)}
    families[model] = result
    if len(completed) < 32:
        assert model not in state["top3"] and model not in state["selected_configs"]
        assert not confirmations and not formal and not legacy
        result["dependencies_correctly_locked"] = True
        continue
    ordered = sorted(completed, key=lambda task: (
        -task["result"]["best_validation"]["NDCG@10"],
        -task["result"]["best_validation"]["HR@10"],
        task["result"]["parameter_count"], task["trial_id"]))[:3]
    top3 = [task["trial_id"] for task in ordered]
    assert state["top3"][model] == top3
    bases = {task["trial_id"]: task for task in ordered}
    assert len(confirmations) == 6
    assert {(task["trial_id"], task["seed"]) for task in confirmations} == {
        (trial_id, seed) for trial_id in top3 for seed in (2025, 2026)}
    assert all(task["config"] == bases[task["trial_id"]]["config"] and
               task["config_sha256"] == bases[task["trial_id"]]["config_sha256"] and
               task["protocol"] == "full_train_prefix_v1" for task in confirmations)
    result["independent_top3"] = top3
    result["completed_confirmations"] = sum(task["status"] == "succeeded" for task in confirmations)
    if result["completed_confirmations"] < 6:
        assert model not in state["selected_configs"] and not formal and not legacy
        continue
    candidates = []
    for trial_id in top3:
        group = [task for task in search + confirmations if task["trial_id"] == trial_id]
        assert len(group) == 3 and {task["seed"] for task in group} == {2024, 2025, 2026}
        averages = {metric: statistics.mean(task["result"]["best_validation"][metric] for task in group)
                    for metric in ("NDCG@10", "HR@10")}
        candidates.append({"trial_id": trial_id, "metrics": averages})
    winner = min(candidates, key=lambda candidate: (
        -candidate["metrics"]["NDCG@10"], -candidate["metrics"]["HR@10"],
        bases[candidate["trial_id"]]["result"]["parameter_count"], candidate["trial_id"]))
    selected = state["selected_configs"][model]
    base = bases[winner["trial_id"]]
    assert selected == {"trial_id": winner["trial_id"], "config": base["config"],
                        "config_sha256": base["config_sha256"],
                        "confirmation_mean": winner["metrics"], "candidates": candidates}
    assert len(formal) == 5 and {task["seed"] for task in formal} == set(range(3031, 3036))
    assert all(task["config"] == base["config"] and task["config_sha256"] == base["config_sha256"]
               and task["protocol"] == "full_train_prefix_v1" for task in formal)
    if model == "esasrec":
        assert not legacy
    else:
        assert len(legacy) == 3 and {task["seed"] for task in legacy} == {3031, 3032, 3033}
        assert all(task["config"] == base["config"] and task["protocol"] == "legacy_prefix_v1"
                   for task in legacy)
    result.update({"independent_selected_trial": winner["trial_id"],
                   "three_seed_validation_candidates": candidates,
                   "formal_config_and_seed_bindings": "PASS", "formal_trials": len(formal)})
assert len(state["tasks"]) == 81 and set(state["selected_configs"]) == {"esasrec"}
proof = {"status": "PASS", "families": families, "queue_sha256": queue_sha,
         "test_metrics_read": False,
         "scope": "All 64 searches and both Top-3 populations; first configuration lock; no final family selection or test results",
         "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (HERE / "DEPENDENCY_AUDIT.json").open("x", encoding="utf8", newline="\n") as stream:
    json.dump(proof, stream, indent=2)
    stream.write("\n")
print(json.dumps(proof))
