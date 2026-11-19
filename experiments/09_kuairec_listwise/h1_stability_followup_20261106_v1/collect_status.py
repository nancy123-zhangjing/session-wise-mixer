"""Read bounded progress/receipts only. No model load, updates or inference."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("r26_hold", "r26_low", "release20_freeze", "release20_linear")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail(path, size=32768):
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace").splitlines()


def collect():
    rows = {}
    for arm in ARMS:
        output, jobs = ROOT / "runs" / arm, ROOT / "jobs" / arm
        row = dict(output_exists=output.exists(), job_result=read_json(jobs / "result.json"),
                   training_complete=read_json(output / "complete.json"),
                   tests=read_json(output / "test_predictions_complete.json"),
                   restore=read_json(output / "restore_verified.json"), validation=[])
        for path in sorted((output / "validation_records").glob("R*.json")):
            value = read_json(path)
            if value:
                row["validation"].append({k: value.get(k) for k in ("epoch", "ndcg5", "r_updates", "t_updates", "kind")})
        lines = tail(output / "training.jsonl")
        for line in reversed(lines):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            row["latest_update"] = {k: value.get(k) for k in (
                "epoch", "kind", "phase_index", "update", "r_updates", "t_updates", "loss", "seconds", "increment_alpha_used", "mixer_gate")}
            break
        row["training_mtime_utc"] = (datetime.datetime.fromtimestamp(
            (output / "training.jsonl").stat().st_mtime, datetime.timezone.utc).isoformat()
            if (output / "training.jsonl").exists() else None)
        for stage in ("train", "test", "verify"):
            lines = tail(jobs / f"{stage}.log", 8192)
            if lines:
                row[f"{stage}_tail"] = lines[-6:]
        rows[arm] = row
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), arms=rows)


if __name__ == "__main__":
    print(json.dumps(collect(), ensure_ascii=False))
