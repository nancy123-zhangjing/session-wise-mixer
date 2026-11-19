"""Independent Top-3 recheck of the first real confirmation-stage snapshot."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
read = lambda path: json.loads(path.read_text(encoding="utf8"))
manifest = read(HERE / "manifest.json")
archive = ROOT / "archive/server_sync/stage06_continuous_precision_v2_20261016/completed_metadata_batch13"
queue_path = archive / manifest["pipeline_relative"] / "campaign/queue.json"
assert hashlib.sha256(queue_path.read_bytes()).hexdigest() == manifest["files"][queue_path.relative_to(archive).as_posix()]["sha256"]
state = read(queue_path)
assert len(state["tasks"]) == 70 and state["source_sha256"] == manifest["source_sha256"]
families = {}
for model in ("sasrec_fullce", "esasrec"):
    searches = [task for task in state["tasks"] if task["phase"] == "search" and task["config"]["model"] == model]
    confirmations = [task for task in state["tasks"] if task["phase"] == "confirmation" and task["config"]["model"] == model]
    assert len(searches) == 32 and {task["seed"] for task in searches} == {2024}
    completed = [task for task in searches if task["status"] == "succeeded"]
    if len(completed) < 32:
        assert model not in state["top3"] and not confirmations
        families[model] = {"completed_searches": len(completed), "top3_not_unlocked": True}
        continue
    ordered = sorted(completed, key=lambda task: (-task["result"]["best_validation"]["NDCG@10"],
        -task["result"]["best_validation"]["HR@10"], task["result"]["parameter_count"], task["trial_id"]))[:3]
    assert state["top3"][model] == [task["trial_id"] for task in ordered]
    assert len(confirmations) == 6
    assert {(task["trial_id"], task["seed"]) for task in confirmations} == {
        (task["trial_id"], seed) for task in ordered for seed in (2025, 2026)}
    bases = {task["trial_id"]: task for task in ordered}
    assert all(task["config"] == bases[task["trial_id"]]["config"] and
               task["config_sha256"] == bases[task["trial_id"]]["config_sha256"] and
               task["protocol"] == "full_train_prefix_v1" for task in confirmations)
    families[model] = {"completed_searches": 32, "independent_top3": state["top3"][model],
                       "confirmation_trials": 6, "exact_configs_and_seeds": True}
assert not state["selected_configs"]
assert not any(task["phase"] in {"formal", "legacy"} for task in state["tasks"])
curves = read(HERE / "AUDIT.json")
proof = {"status": "PASS", "families": families,
         "runs_at_300_epoch_cap": [task for task in curves["trials"] if task["epochs"] == 300],
         "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
         "test_metrics_read": False, "scope": "First real Top-3 confirmation boundary only; no configuration or final model selected",
         "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (HERE / "DEPENDENCY_AUDIT.json").open("x", encoding="utf8", newline="\n") as stream:
    json.dump(proof, stream, indent=2)
    stream.write("\n")
print(json.dumps(proof))
