"""CPU-only, source-bound multipart backup of the completed corrected campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
PIPELINE = "code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-gib", type=float, default=2)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    assert base.name == "goal_20261016_precision_v2"
    output = Path(args.output).resolve()
    assert output.parent == base / "delivery" and not output.exists()
    assert 0.5 <= args.part_gib <= 4
    pipe = base / PIPELINE
    campaign = pipe / "campaign"
    queue = read(campaign / "queue.json")
    assert len(queue["tasks"]) == 89 and not queue["workers"]
    assert all(task["status"] == "succeeded" for task in queue["tasks"])
    complete = read(pipe / "CPU_DELIVERY_PASS.json")
    assert complete["status"] == "PASS" and complete["verified_training_runs"] == 89
    assert complete["test_evaluations"] == complete["independent_metric_audits"] == 13
    assert complete["formal_exports"] == 10 and queue["source_sha256"] == SOURCE
    for name in ("FINAL_BENCHMARK_AUDIT_20261016_01.json", "FINAL_EXPORT_AUDIT_20261016_01.json"):
        assert read(base / "delivery" / name)["status"] == "PASS"
    gate = read(base / "code/outputs/stage06_continuous/remote_cpu_gate_20261016_precision_v2/CPU_GATE.json")

    def verify_source():
        assert gate["source"]["source_sha256"] == SOURCE
        assert all(sha(base / "code" / name) == value for name, value in gate["source"]["files"].items())

    verify_source()
    entries = {}

    def add(source, name, group="evidence", expected=None):
        source = Path(source)
        assert source.is_file() and not source.is_symlink(), str(source)
        assert not name.startswith("/") and ".." not in Path(name).parts and "\\" not in name
        record = {"source": source, "name": name, "group": group, "bytes": source.stat().st_size,
                  "mtime_ns": source.stat().st_mtime_ns, "expected": expected}
        assert name not in entries, name
        entries[name] = record

    def tree(directory, prefix, group="evidence", skip_weights=False):
        assert directory.is_dir(), str(directory)
        for path in sorted(directory.rglob("*")):
            if {".git", "__pycache__", ".pytest_cache"}.intersection(path.relative_to(directory).parts):
                continue
            if path.is_dir():
                continue
            if path.suffix in {".pyc", ".lock", ".tmp"}:
                continue
            if skip_weights and path.suffix in {".pt", ".npy"}:
                continue
            assert path.is_file() and not path.is_symlink(), str(path)
            add(path, prefix + "/" + path.relative_to(directory).as_posix(), group)

    for task in queue["tasks"]:
        run = Path(task["output"]).resolve()
        assert run.is_relative_to(campaign / "runs")
        prefix = "campaign/" + run.relative_to(campaign).as_posix()
        names = ["run.json", "COMPLETE.json", "CURRENT.json", "curves.csv", "best_validation.json",
                 "best.receipt.json", "last.receipt.json", "performance.jsonl"]
        if (run / "resume_events.jsonl").exists():
            names.append("resume_events.jsonl")
        for name in names:
            add(run / name, prefix + "/" + name)
        for kind in ("best", "last"):
            add(run / (kind + ".pt"), prefix + "/" + kind + ".pt", "checkpoints",
                task["result"][kind + "_sha256"])
    exports = read(campaign / "export_index.json")
    assert len(exports) == 10
    for identity, entry in exports.items():
        directory = campaign / "exports" / identity
        for path in sorted(directory.iterdir()):
            assert path.is_file()
            add(path, "campaign/exports/" + identity + "/" + path.name, "exports", entry["files"].get(path.name))
    for name in ("delivery", "evaluations", "final_benchmarks"):
        tree(campaign / name, "campaign/" + name)
    if (campaign / "logs").is_dir():
        tree(campaign / "logs", "campaign/logs")
    for name in ("queue.json", "FROZEN_SELECTION.json", "evaluation_index.json", "export_index.json"):
        add(campaign / name, "campaign/" + name)
    for name in ("PIPELINE.json", "GPU_WORK_COMPLETE.json", "CPU_DELIVERY_PASS.json", "pipeline_events.jsonl"):
        add(pipe / name, "pipeline/" + name)
    for name in ("initial_benchmarks", "resource_attempts", "smoke"):
        tree(pipe / name, "pipeline/" + name, skip_weights=True)
    inputs = base.with_name("goal_20261016") / "inputs"
    tree(inputs / "prepared", "inputs/prepared")
    tree(inputs / "cache", "inputs/cache")
    for name, expected in gate["source"]["files"].items():
        add(base / "code" / name, "source/" + name, expected=expected)
    add(base / "code/outputs/stage06_continuous/remote_cpu_gate_20261016_precision_v2/CPU_GATE.json",
        "audits/REMOTE_CPU_GATE.json")
    tree(base / "orchestration_v3", "orchestration_v3", skip_weights=True)
    tree(base / "delivery/checkpoint_payload_audits", "audits/checkpoint_payload_audits")
    for name in ("FINAL_BENCHMARK_AUDIT_20261016_01.json", "FINAL_EXPORT_AUDIT_20261016_01.json"):
        add(base / "delivery" / name, "audits/" + name)
    for name in ("archive_final_delivery.py", "audit_formal_exports.py", "audit_benchmark_measurements.py"):
        add(base / "delivery" / name, "audits/tools/" + name)
    output.mkdir()
    parts, files = [], {}
    maximum = int(args.part_gib * 1024 ** 3)
    for group in ("evidence", "exports", "checkpoints"):
        group_entries = sorted((entry for entry in entries.values() if entry["group"] == group), key=lambda r: r["name"])
        batches, batch, size = [], [], 0
        for entry in group_entries:
            assert entry["bytes"] <= maximum
            if batch and size + entry["bytes"] > maximum:
                batches.append(batch)
                batch, size = [], 0
            batch.append(entry)
            size += entry["bytes"]
        if batch:
            batches.append(batch)
        for number, batch in enumerate(batches, 1):
            name = f"{group}-{number:02d}.tar.gz"
            path = output / name
            temporary = path.with_name(name + ".tmp")
            print(json.dumps({"part_start": name, "files": len(batch)}), flush=True)
            with temporary.open("xb") as raw:
                with tarfile.open(fileobj=raw, mode="w:gz", compresslevel=1, format=tarfile.PAX_FORMAT) as archive:
                    for entry in batch:
                        source = entry["source"]
                        assert source.stat().st_size == entry["bytes"] and source.stat().st_mtime_ns == entry["mtime_ns"]
                        info = tarfile.TarInfo(entry["name"])
                        info.size, info.mode, info.mtime = entry["bytes"], 0o644, entry["mtime_ns"] // 10 ** 9
                        with source.open("rb") as stream:
                            reader = HashedReader(stream)
                            archive.addfile(info, reader)
                        digest = reader.digest.hexdigest()
                        assert entry["expected"] is None or digest == entry["expected"], entry["name"]
                        assert source.stat().st_size == entry["bytes"] and source.stat().st_mtime_ns == entry["mtime_ns"]
                        files[entry["name"]] = {"archive": name, "bytes": entry["bytes"], "sha256": digest,
                                                 "source_mtime_ns": entry["mtime_ns"]}
                raw.flush()
                os.fsync(raw.fileno())
            temporary.rename(path)
            part = {"file": name, "bytes": path.stat().st_size, "sha256": sha(path),
                    "files": len(batch), "uncompressed_bytes": sum(entry["bytes"] for entry in batch)}
            parts.append(part)
            print(json.dumps({"part_complete": part}), flush=True)
    verify_source()
    assert len(files) == len(entries)
    proof = {"status": "BACKUP_ARCHIVES_CREATED", "created_unix": time.time(), "source_sha256": SOURCE,
             "training_runs": 89, "checkpoint_states": 178, "formal_exports": 10, "test_evaluations": 13,
             "archives": parts, "files": files, "archive_script_sha256": sha(__file__),
             "scope": "Corrected 89-run campaign, best/last payloads, all exports/evaluations/curves, locked inputs and frozen sources. Prior failed-version outputs remain separately preserved, not valid training runs. Archive creation is not local download verification."}
    with (output / "FINAL_BACKUP_MANIFEST.json").open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in proof.items() if key not in {"files", "archives"}}), flush=True)


if __name__ == "__main__":
    main()
