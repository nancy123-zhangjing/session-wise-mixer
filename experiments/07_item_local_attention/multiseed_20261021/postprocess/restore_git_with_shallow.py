"""Restore original bundle bytes with independently captured shallow metadata.

The original package remains unchanged. SOURCE_SHALLOW.json and engine.shallow
must come from a read-only capture of the actual frozen engine repository.
This restores that repository's shallow history, not complete upstream history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multiseed_backup as original

need, read, sha, write = original.need, original.read, original.sha, original.write
SCHEMA = "stage07_source_shallow_supplement_v1"


def record(path):
    path = Path(path).resolve(strict=True)
    need(path.is_file(), "Required ordinary file missing: " + str(path))
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def git(repository, *arguments):
    # Local settings persist after restoration; command settings also cover init.
    command = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "-C", str(repository), *arguments]
    result = subprocess.run(command, env=original.git_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(json.dumps({"git_arguments": list(arguments), "repository": str(repository),
                           "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False))
    return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def load_inputs(verified_root, package_dir, supplement):
    root, package, supplement = [Path(p).resolve(strict=True) for p in (verified_root, package_dir, supplement)]
    manifest, entries = original.verified_manifest(root, package)

    def checked(member):
        need(member in entries, "Required original archive member missing: " + member)
        path = original.byte.inside(root, original.byte.safe_name(member))
        need(original.byte.file_record(member, path) == entries[member], "Original extracted bytes changed: " + member)
        return path

    source_receipt = supplement / "SOURCE_SHALLOW.json"
    capture = read(source_receipt)
    shallow_record = capture["shallow_file"]
    need(shallow_record["member"] == "engine.shallow", "Expected a single explicit shallow sidecar filename")
    shallow_path = original.byte.inside(supplement, shallow_record["member"])
    need(original.byte.file_record("engine.shallow", shallow_path) == shallow_record, "Captured shallow bytes/SHA differ")
    shallow_bytes = shallow_path.read_bytes()
    shallow_commits = shallow_bytes.decode("ascii").splitlines()
    need(shallow_commits and len(shallow_commits) == len(set(shallow_commits))
         and all(re.fullmatch(r"[0-9a-f]{40}", commit) for commit in shallow_commits), "Malformed actual shallow boundaries")
    engine_bundle = checked("source/engine.bundle")
    engine_audit_path = checked("source/engine_git_audit.json")
    engine_audit = read(engine_audit_path)
    goal_root = manifest["original_goal_root"].rstrip("/")
    expected_engine = manifest["source_identity"]["engine"]["commit"]
    need(capture["schema"] == SCHEMA and capture["campaign_id"] == original.CAMPAIGN
         and capture["repository"] == "engine" and capture["capture_mode"] == "read_only"
         and capture["source_goal_root"].rstrip("/") == goal_root
         and capture["source_repository"].rstrip("/") == goal_root + "/engine"
         and capture["source_commit"] == expected_engine == engine_audit["commit"]
         and capture["source_is_shallow"] is True and capture["source_status_porcelain"] == ""
         and capture["shallow_commits"] == shallow_commits and bool(capture["captured_utc"]),
         "Shallow receipt is not a read-only capture of this frozen source")
    need(capture["original_package_sha256"] == sha(package / "PACKAGE.json")
         and capture["original_manifest_sha256"] == sha(root / "MANIFEST.json")
         and capture["original_engine_bundle_sha256"] == sha(engine_bundle) == engine_audit["bundle_sha256"]
         and capture["original_engine_audit_sha256"] == sha(engine_audit_path),
         "Shallow supplement does not bind the original verified package/source bytes")
    sources = {}
    for name in ("code", "engine"):
        bundle = checked(f"source/{name}.bundle")
        audit_path = checked(f"source/{name}_git_audit.json")
        audit = read(audit_path)
        expected = manifest["source_identity"][name]["commit"]
        need(re.fullmatch(r"[0-9a-f]{40}", expected) and audit["commit"] == expected and audit["clean"] is True
             and audit["bundle_sha256"] == sha(bundle), "Original repository identity/audit differs")
        sources[name] = {"bundle": bundle, "commit": expected, "audit": record(audit_path)}
    evidence = [record(path) for path in (source_receipt, shallow_path, package / "PACKAGE.json", package / "PACK_COMPLETE.json",
                root / "MANIFEST.json", root / "BYTE_VERIFICATION.json", *(r["bundle"] for r in sources.values()),
                *(root / f"source/{name}_git_audit.json" for name in sources))]
    return {"root": root, "package": package, "supplement": supplement, "capture": capture,
            "sources": sources, "shallow_bytes": shallow_bytes, "shallow_commits": shallow_commits, "evidence": evidence}


def inspect_raw_parents(repository, captured_boundaries):
    """Check actual stored commits, including objects unreachable after grafting.

    fsck alone can accept dangling incomplete history after a wrong HEAD graft.
    Inspecting raw commit headers prevents such a graft from hiding missing
    parents at a different, uncaptured boundary.
    """
    lines = git(repository, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")["stdout"].splitlines()
    objects = dict(line.split() for line in lines)
    need(all(objects.get(boundary) == "commit" for boundary in captured_boundaries),
         "Captured shallow boundary is not an actual bundled commit")
    missing = []
    commits = [oid for oid, kind in objects.items() if kind == "commit"]
    for oid in commits:
        header = git(repository, "cat-file", "-p", oid)["stdout"].split("\n\n", 1)[0]
        parents = [line[7:] for line in header.splitlines() if line.startswith("parent ")]
        for parent in parents:
            need(re.fullmatch(r"[0-9a-f]{40}", parent), "Malformed raw commit parent")
            if objects.get(parent) != "commit":
                need(oid in captured_boundaries, "Missing parent belongs to an uncaptured shallow boundary: " + oid)
                missing.append({"boundary_commit": oid, "missing_parent": parent})
    return {"stored_commit_count": len(commits), "all_missing_parent_edges_covered_by_actual_capture": True,
            "missing_parent_edges": missing}


def restore(verified_root, package_dir, supplement, output):
    need(not os.path.lexists(output), "Use a wholly new output directory; preserve all prior attempts")
    inputs = load_inputs(verified_root, package_dir, supplement)
    output = Path(output).resolve()
    need(not any(output.is_relative_to(inputs[key]) for key in ("root", "package", "supplement")),
         "Keep restoration outside original evidence and supplement")
    output.mkdir(parents=True, exist_ok=False)
    repositories = {}
    for name, source in inputs["sources"].items():
        repository = output / name
        git(output, "init", "--quiet", str(repository))
        git(repository, "config", "--local", "core.longpaths", "true")
        git(repository, "config", "--local", "core.autocrlf", "false")
        need(git(repository, "bundle", "list-heads", str(source["bundle"]))["stdout"] == source["commit"] + " HEAD",
             "Bundle must advertise exactly the frozen HEAD")
        # unbundle writes objects without prematurely traversing the missing
        # shallow parents. Only actual captured boundaries are then installed.
        imported = git(repository, "bundle", "unbundle", str(source["bundle"]))
        need(imported["stdout"] == source["commit"] + " HEAD", "Imported bundle HEAD differs")
        parent_check = inspect_raw_parents(repository, inputs["shallow_commits"] if name == "engine" else [])
        if name == "engine":
            with (repository / ".git/shallow").open("xb") as handle:
                handle.write(inputs["shallow_bytes"])
            for boundary in inputs["shallow_commits"]:
                git(repository, "merge-base", "--is-ancestor", boundary, source["commit"])
        git(repository, "checkout", "--detach", source["commit"])
        head = git(repository, "rev-parse", "HEAD")["stdout"]
        status = git(repository, "status", "--porcelain", "--untracked-files=all")["stdout"]
        need(head == source["commit"] and not status, "Independent checkout is not exact and clean")
        git(repository, "diff", "--exit-code", "HEAD", "--")
        fsck = git(repository, "fsck", "--full", "--strict")
        is_shallow = git(repository, "rev-parse", "--is-shallow-repository")["stdout"]
        need(is_shallow == ("true" if name == "engine" else "false"), "Restored source shallow status differs")
        need(not (repository / ".git/objects/info/alternates").exists(), "Recovery must not borrow another object store")
        shallow_identity = record(repository / ".git/shallow") if name == "engine" else None
        if shallow_identity:
            need(shallow_identity["sha256"] == inputs["capture"]["shallow_file"]["sha256"], "Restored shallow bytes changed")
        repositories[name] = {"path": str(repository), "commit": head, "bundle": record(source["bundle"]),
                              "clean": True, "git_fsck": fsck, "is_shallow_repository": name == "engine",
                              "raw_parent_integrity": parent_check,
                              "shallow_file": shallow_identity, "shallow_commits": inputs["shallow_commits"] if name == "engine" else [],
                              "reachable_commit_count": int(git(repository, "rev-list", "--count", "HEAD")["stdout"]),
                              "repository_local_core_longpaths": git(repository, "config", "--local", "--get", "core.longpaths")["stdout"],
                              "repository_local_core_autocrlf": git(repository, "config", "--local", "--get", "core.autocrlf")["stdout"]}
    need(all(record(row["path"]) == row for row in inputs["evidence"]), "Original package/supplement evidence changed during recovery")
    result = {"status": "INDEPENDENT_SOURCE_RESTORE_WITH_CAPTURED_SHALLOW_PASSED", "campaign_id": original.CAMPAIGN,
              "verified_utc": datetime.now(timezone.utc).isoformat(), "evidence": inputs["evidence"], "repositories": repositories,
              "source_supplement_schema": SCHEMA, "original_archive_modified": False, "GPU_used": False,
              "source_audit_correction": {
                  "applies_to": "source/engine_git_audit.json and manifest.source_identity.engine",
                  "original_engine_audit_sha256": inputs["capture"]["original_engine_audit_sha256"],
                  "original_claim_bundle_has_no_external_prerequisites": True,
                  "original_empty_repository_bundle_verify_was_insufficient": True,
                  "engine_bundle_alone_independently_restorable": False,
                  "engine_bundle_with_captured_shallow_independently_restored": True,
                  "complete_upstream_engine_history_included": False,
                  "restored_scope": "Exact source tree and the original source repository's available shallow history, with its actual captured boundary; not complete upstream history"},
              "tool": record(__file__), "original_restore_helper": record(HERE / "multiseed_backup.py"),
              "relocated_training_resume_exercised": False, "goal_complete": False}
    write(output / "GIT_RESTORE_WITH_SHALLOW.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--source-supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_existed = os.path.lexists(args.output)
    try:
        result = restore(args.verified_root, args.package_dir, args.source_supplement, args.output)
        print(json.dumps({"status": result["status"], "receipt": str(args.output / "GIT_RESTORE_WITH_SHALLOW.json")}))
        return 0
    except Exception as error:
        result = {"status": "INDEPENDENT_SOURCE_RESTORE_WITH_CAPTURED_SHALLOW_FAILED", "error_type": type(error).__name__,
                  "error": str(error), "original_archive_modified": False, "GPU_used": False, "goal_complete": False}
        if not output_existed and args.output.is_dir() and not (args.output / "RESTORE_FAILED.json").exists():
            write(args.output / "RESTORE_FAILED.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
