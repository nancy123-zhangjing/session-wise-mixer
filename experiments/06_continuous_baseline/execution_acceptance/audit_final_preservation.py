"""Verify original files/engines and the frozen training source without writing them."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, directory=ROOT):
    return subprocess.check_output(["git", "-C", str(directory), *args], encoding="utf8").strip()


def main():
    initial_path = ROOT / ".runtime/continuous_goal_20261015/initial_state.json"
    initial = json.loads(initial_path.read_text(encoding="utf8"))
    changed = {}
    for name, before in initial["tracked_hashes"].items():
        path = ROOT / name
        assert path.is_file(), name
        after = sha(path)
        if before != after:
            changed[name] = {"before": before, "after": after}
    assert set(changed) == {".gitattributes", "INSPIRE.md"}, changed
    engines = {}
    for name, original in initial["engines"].items():
        head = git("rev-parse", "HEAD", directory=ROOT / name)
        status = git("status", "--short", directory=ROOT / name)
        assert head == original["head"] and status == original["status"] == ""
        engines[name] = {"head": head, "status": status}
    parts = ROOT / "outputs/stage06_continuous/final_delivery_20261016/backup_parts"
    manifest_path = parts / "FINAL_BACKUP_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    evidence = next(r for r in manifest["archives"] if r["file"] == "evidence-01.tar.gz")
    assert sha(parts / evidence["file"]) == evidence["sha256"]
    with tarfile.open(parts / evidence["file"], "r:gz") as archive:
        run = json.load(archive.extractfile("campaign/runs/formal__sasrec_fullce_13__seed3031/attempt_0/run.json"))
    source = run["source"]
    for name, expected in source["files"].items():
        assert sha(ROOT / name) == expected == manifest["files"]["source/" + name]["sha256"]
    canonical = hashlib.sha256(json.dumps(source["files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert canonical == source["source_sha256"] == "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
    assert source["git_head"] == "bbaa45b515a15912a04665721e0a5c43d64dc9e9" and source["git_status"] == ""
    for name in ("README.md", "docs/guides/README.md"):
        assert sha(ROOT / name) == initial["tracked_hashes"][name]
    proof = {"status": "PASS", "original_tracked_files_checked": len(initial["tracked_hashes"]),
        "unchanged_original_tracked_files": len(initial["tracked_hashes"]) - len(changed),
        "authorized_original_file_changes": changed, "engines": engines,
        "training_source_files_checked": len(source["files"]), "training_source_sha256": canonical,
        "training_commit": source["git_head"], "initial_state_sha256": sha(initial_path),
        "backup_manifest_sha256": sha(manifest_path), "audit_script_sha256": sha(__file__),
        "protected_root_readmes_unchanged": True,
        "scope": "Original 908 file bytes and two nested engine repositories checked against the initial workspace snapshot; the 27 training source files match frozen remote run/backup hashes. Current staging and unrelated untracked drafts require a separate explicit-staging audit."}
    target = HERE / "final_delivery_20261016/FINAL_PRESERVATION_AUDIT.json"
    with target.open("x", encoding="utf8") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
