"""Read existing logs only; no model imports, training, or evaluation."""
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
result = {"observed_at": datetime.now(timezone.utc).isoformat(), "arms": {}}
for arm in ("B0", "B1", "B2"):
    entry = {}
    for phase, path in (("pretrain", root / (arm + "_rq")), ("joint", root / arm)):
        events = []
        log = path / "events.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Concurrent append can leave an unfinished last line.
        entry[phase] = {"latest_by_kind": {}}
        for event in events:
            entry[phase]["latest_by_kind"][event["kind"]] = event
        for name in ("COMPLETE.json", "TRAINING_COMPLETE.json", "TEST_COMPLETE.json", "BEST.json", "ERROR.json"):
            file = path / name
            if file.exists():
                entry[phase][name] = json.loads(file.read_text())
        if phase == "pretrain" and "COMPLETE.json" in entry[phase]:
            checks = [e for e in events if e["kind"] == "pretrain_validation"]
            selected = min(checks, key=lambda e: e["encoding"]["collision_redundancy"])
            entry[phase]["selection_audit"] = {
                "validation_count": len(checks),
                "all_200_epochs_present": [e["epoch"] for e in checks] == list(range(50, 10001, 50)),
                "first_minimum_epoch": selected["epoch"],
                "matches_receipt": selected["epoch"] == entry[phase]["COMPLETE.json"]["best_collision_epoch"],
                "selected_encoding": selected["encoding"],
            }
        if phase == "joint":
            entry[phase]["test"] = {
                p.parent.name: json.loads(p.read_text())
                for p in sorted((path / "test").glob("*/summary.json"))
            }
            entry[phase]["test_progress"] = [
                {k: v for k, v in json.loads(p.read_text()).items() if k != "identity"} | {"node": p.parent.name}
                for p in sorted((path / "test").glob("*/progress.json"))
                if not (p.parent / "summary.json").exists()
            ]
            entry[phase]["validation"] = [json.loads(p.read_text()) for p in sorted((path / "validation").glob("R*/summary.json"))]
            entry[phase]["validation_progress"] = [
                {k: v for k, v in json.loads(p.read_text()).items() if k != "identity"} | {"node": p.parent.name}
                for p in sorted((path / "validation").glob("R*/progress.json"))
                if not (p.parent / "summary.json").exists()
            ]
    result["arms"][arm] = entry
print(json.dumps(result))
