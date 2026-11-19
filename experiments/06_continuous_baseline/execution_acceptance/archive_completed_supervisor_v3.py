"""CPU-only immutable metadata backup for the live, frozen stage-06 campaign.

Each invocation creates the next numbered archive only when new runs completed.
It never changes queue state, training code, checkpoints or selection decisions.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
DEPLOYMENT = "229b9948c63571d65f08e75d1a16a40ec37bf6ec104d1a9c7fb7ddea5e55b0a6"
PIPELINE = "code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    assert 7 <= args.batch <= 99
    base = Path(args.base).resolve()
    assert base.name == "goal_20261016_precision_v2"
    delivery = base / "delivery"
    import fcntl
    with (delivery / "metadata_archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deployment_path = base / "orchestration_v3/DEPLOYMENT_MANIFEST.json"
        assert digest(deployment_path.read_bytes()) == DEPLOYMENT
        deployment = read(deployment_path)
        assert deployment["training_source_sha256"] == SOURCE
        assert digest((base / "orchestration_v3/scripts/train/continuous_queue_supervisor.py").read_bytes()) == deployment["supervisor_sha256"]
        assert all(digest((base / "code" / name).read_bytes()) == expected
                   for name, expected in deployment["training_source_files"].items())
        previous = delivery / f"completed_metadata_precision_v2_batch{args.batch-1:02d}.json"
        earlier = read(previous)
        seen = set(earlier["all_completed_task_ids"])
        assert earlier["source_sha256"] == SOURCE
        pipeline = base / PIPELINE
        queue_bytes = (pipeline / "campaign/queue.json").read_bytes()
        queue = json.loads(queue_bytes)
        assert queue["source_sha256"] == SOURCE and len(queue["tasks"]) <= 89
        assert queue["deployment_manifest_sha256"] == DEPLOYMENT
        assert queue["supervisor_sha256"] == deployment["supervisor_sha256"]
        completed = [t for t in queue["tasks"] if t["status"] == "succeeded"]
        assert seen <= {t["task_id"] for t in completed}
        fresh = [t for t in completed if t["task_id"] not in seen]
        archive = delivery / f"completed_metadata_precision_v2_batch{args.batch:02d}.tar.gz"
        manifest = archive.with_name(f"completed_metadata_precision_v2_batch{args.batch:02d}.json")
        assert not archive.exists() and not manifest.exists(), "Inspect existing batch before retrying"
        if not fresh:
            print(json.dumps({"status": "NO_NEW_COMPLETIONS", "next_batch": args.batch}), flush=True)
            return
        entries = {PIPELINE + "/campaign/queue.json": queue_bytes}
        for task in fresh:
            output = Path(task["output"]).resolve()
            assert output.is_relative_to(pipeline / "campaign/runs")
            assert read(output / "COMPLETE.json") == task["result"]
            assert task["result"]["binding"]["source_sha256"] == SOURCE
            names = ["run.json", "COMPLETE.json", "CURRENT.json", "curves.csv", "best_validation.json",
                     "best.receipt.json", "last.receipt.json", "performance.jsonl"]
            if (output / "resume_events.jsonl").exists():
                names.append("resume_events.jsonl")
            for name in names:
                path = output / name
                entries[path.relative_to(base).as_posix()] = path.read_bytes()
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        assert not temporary.exists(), "Unfinished archive must be inspected"
        with tarfile.open(temporary, "w:gz") as stream:
            for name, data in entries.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
        temporary.rename(archive)
        result = {"status": "COMPLETED_METADATA_ARCHIVED", "snapshot_unix": time.time(),
            "source_sha256": SOURCE, "pipeline_relative": PIPELINE,
            "supervisor_sha256": queue["supervisor_sha256"],
            "counts": dict(Counter(t["status"] for t in queue["tasks"])),
            "completed_task_ids": [t["task_id"] for t in fresh],
            "all_completed_task_ids": [t["task_id"] for t in completed],
            "previous_batch_sha256": digest(previous.read_bytes()),
            "archive_sha256": digest(archive.read_bytes()),
            "archive_producer_sha256": digest(Path(__file__).read_bytes()),
            "scope": "Immutable completed metadata only; same frozen training implementation; no test metric reads",
            "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in entries.items()}}
        with manifest.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in {"files", "all_completed_task_ids"}}), flush=True)


if __name__ == "__main__":
    main()
