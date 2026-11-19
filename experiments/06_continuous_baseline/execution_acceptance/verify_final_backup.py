"""Independent streaming verification, then guarded extraction of final deliverables."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def safe_name(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
    assert str(path) == name and name not in {"", "."}
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--extract", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = read(manifest_path)
    assert manifest["status"] == "BACKUP_ARCHIVES_CREATED" and manifest["source_sha256"] == SOURCE
    assert manifest["training_runs"] == 89 and manifest["checkpoint_states"] == 178
    assert manifest["formal_exports"] == 10 and manifest["test_evaluations"] == 13
    assert manifest["archive_script_sha256"] == sha(HERE / "archive_final_delivery.py")
    files, parts = manifest["files"], manifest["archives"]
    assert len({name.casefold() for name in files}) == len(files)
    assert len({part["file"] for part in parts}) == len(parts)
    for name in files:
        safe_name(name)
    queue_root = ROOT / "archive/server_sync/stage06_continuous_precision_v2_20261016/completed_metadata_batch20"
    queue = read(queue_root / "code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/campaign/queue.json")
    ledger = read(HERE / "checkpoint_payload_precision_v2/ledger_snapshot_22.json")
    assert len(queue["tasks"]) == len(ledger["trials"]) == 89 and not queue["workers"]
    assert all(task["status"] == "succeeded" for task in queue["tasks"])
    assert Counter(task["phase"] for task in queue["tasks"]) == {"search": 64, "confirmation": 12, "formal": 10, "legacy": 3}
    checkpoint_names = set()
    for task in queue["tasks"]:
        prefix = f"campaign/runs/{task['task_id']}/attempt_{task['attempt']}"
        for label in ("best", "last"):
            name = prefix + "/" + label + ".pt"
            assert files[name]["sha256"] == task["result"][label + "_sha256"] == ledger["trials"][task["task_id"]]["checks"][label]["sha256"]
            checkpoint_names.add(name)
    actual_checkpoints = {name for name in files if name.startswith("campaign/runs/") and name.endswith(("/best.pt", "/last.pt"))}
    assert actual_checkpoints == checkpoint_names and len(checkpoint_names) == 178
    archives = Path(args.archives).resolve()
    seen, verified_parts, total_bytes = set(), [], 0
    for part in parts:
        name = part["file"]
        assert safe_name(name).name == name and name.endswith(".tar.gz")
        path = archives / name
        assert path.stat().st_size == part["bytes"] and sha(path) == part["sha256"]
        expected = {name for name, record in files.items() if record["archive"] == part["file"]}
        current = set()
        with tarfile.open(path, "r|gz") as stream:
            for member in stream:
                safe_name(member.name)
                assert member.isfile() and member.name in expected and member.name not in seen
                record = files[member.name]
                assert member.size == record["bytes"]
                source = stream.extractfile(member)
                digest, size = hashlib.sha256(), 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                assert size == record["bytes"] and digest.hexdigest() == record["sha256"], member.name
                current.add(member.name)
                seen.add(member.name)
                total_bytes += size
        assert current == expected and len(current) == part["files"]
        assert sum(files[name]["bytes"] for name in current) == part["uncompressed_bytes"]
        verified_parts.append(part)
        print(json.dumps({"verified_archive": name, "files": len(current)}), flush=True)
    assert seen == files.keys()
    destination = Path(args.extract).resolve()
    assert destination.is_relative_to(ROOT / "outputs/stage06_continuous") and not destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    selected_checkpoint = "campaign/runs/formal__sasrec_fullce_13__seed3031/attempt_0/"
    wanted = {name for name in files if not name.startswith("campaign/runs/") or name.startswith(selected_checkpoint)}
    extracted = set()
    for part in parts:
        selected = {name for name in wanted if files[name]["archive"] == part["file"]}
        if not selected:
            continue
        with tarfile.open(archives / part["file"], "r:gz") as stream:
            for member in stream:
                if member.name not in selected:
                    continue
                assert member.isfile()
                target = destination.joinpath(*safe_name(member.name).parts)
                assert target.resolve().is_relative_to(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    source = stream.extractfile(member)
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                assert target.stat().st_size == files[member.name]["bytes"] and sha(target) == files[member.name]["sha256"]
                extracted.add(member.name)
    assert extracted == wanted
    evaluations = read(destination / "campaign/evaluation_index.json")
    assert len(evaluations) == 13
    for identity, entry in evaluations.items():
        result = read(destination / "campaign/evaluations" / identity / "result.json")
        assert result == entry["result"] and result["samples"] == 22363
        assert result["prediction_sha256"] == sha(destination / "campaign/evaluations" / identity / "predictions.jsonl")
    exports = read(destination / "campaign/export_index.json")
    assert len(exports) == 10
    for identity, entry in exports.items():
        for name, expected in entry["files"].items():
            assert sha(destination / "campaign/exports" / identity / name) == expected
    proof = {"status": "PASS", "source_sha256": SOURCE, "training_runs": 89, "checkpoint_states": 178,
             "formal_exports": 10, "test_evaluations": 13, "verified_archive_parts": len(parts),
             "verified_files": len(seen), "verified_uncompressed_bytes": total_bytes,
             "extracted_deliverable_files": len(extracted), "extraction_root": str(destination),
             "manifest_sha256": sha(manifest_path), "verifier_sha256": sha(__file__),
             "scope": "Every downloaded archive and regular member checked by size/SHA256; all 178 checkpoint digests matched to independent CPU payload ledger; exports/predictions extracted and rehashed; optimizer states for other seeds remain in verified archives, not all extracted."}
    with Path(args.report).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
