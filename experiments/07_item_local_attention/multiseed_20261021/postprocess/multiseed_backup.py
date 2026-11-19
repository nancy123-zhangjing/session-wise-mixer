"""Full CPU byte backup, safe extraction, and independent Git restoration.

pack --goal-root G --output NEW_PACKAGE_DIR [--part-mib 512]
verify --package-dir P --output NEW_EXTRACT_DIR
restore-git --verified-root E --package-dir P --output NEW_GIT_RESTORE_DIR

The two seed trees and goal delivery are included in full, including private
inputs, best weights and last optimizer/RNG states. Shared environments and
account directories are outside scope. Symlinks are never followed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import archive_bytes as byte

CAMPAIGN = byte.CAMPAIGN
ENGINE_COMMIT = "1c879252f6395d6e64462511823c3929b9c42c00"
TOOLS = ("multiseed_backup.py", "archive_bytes.py", "verify_item_local_archive.py",
         "audit_multiseed_checkpoints.py", "checkpoint_payload_checks.py", "test_multiseed_backup.py",
         "test_checkpoint_readback.py", "README.md")
FORBIDDEN_COMPONENTS = {".git", ".ssh", ".aws", ".azure", ".codex", ".kube", ".venv", "conda-envs", "miniconda3",
                        ".env", ".netrc", "credentials", "credentials.json", "token", "id_rsa", "id_ed25519"}
need, read, sha, write = byte.need, byte.read, byte.sha, byte.write


def import_module(path):
    spec = importlib.util.spec_from_file_location("frozen_multiseed_backup_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regular_tree(root):
    """Strict nonfollowing walk; no suffix exclusion can omit a state file."""
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "Required ordinary directory is missing: " + str(root))
    found = []
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                need(not any(part.casefold() in FORBIDDEN_COMPONENTS for part in path.relative_to(root).parts),
                     "Account/environment-looking entry requires inspection: " + str(path))
                mode = entry.stat(follow_symlinks=False).st_mode
                need(not stat.S_ISLNK(mode), "Symlink found; no external target will be followed: " + str(path))
                if stat.S_ISDIR(mode):
                    stack.append(path)
                else:
                    need(stat.S_ISREG(mode), "Special file is not safe to archive: " + str(path))
                    byte.safe_name(relative)
                    found.append(path)
    return sorted(found)


def source_inventory(goal):
    sources = {}
    for relative in ("seeds/seed2025", "seeds/seed2026", "delivery"):
        for path in regular_tree(goal / relative):
            member = path.relative_to(goal).as_posix()
            need(member not in sources, "Duplicate source member")
            sources[member] = path
    return sources


def git_env():
    env = os.environ.copy()
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    return env


def git(repository, *arguments):
    return subprocess.check_output(["git", "-c", "core.autocrlf=false", "-C", str(repository), *arguments],
                                   env=git_env(), text=True, stderr=subprocess.STDOUT).strip()


def repository_snapshot(goal, name, expected, temporary):
    repository = goal / name
    status = git(repository, "status", "--porcelain", "--untracked-files=all")
    need(git(repository, "rev-parse", "HEAD") == expected and not status,
         "Source must match its exact clean deployment commit: " + name)
    ignored = git(repository, "ls-files", "--others", "--ignored", "--exclude-standard", "-z").split("\0")
    ignored_audit = []
    for relative in filter(None, ignored):
        target = repository / relative
        row = {"path": relative, "included": False, "reason": "ignored file outside tracked source snapshot"}
        if target.is_symlink():
            resolved = target.resolve()
            need(name == "code" and relative == "external/DiffGRM" and resolved == goal / "engine",
                 "Unexpected ignored source symlink: " + str(target))
            row.update(kind="symlink", link_text=os.readlink(target), captured_separately="engine Git snapshot")
        else:
            row["kind"] = "directory" if target.is_dir() else "file"
        ignored_audit.append(row)
    bundle, archive = temporary / f"{name}.bundle", temporary / f"{name}.tar"
    subprocess.run(["git", "-C", str(repository), "bundle", "create", str(bundle), "HEAD"],
                   check=True, env=git_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", "--output", str(archive), expected],
                   check=True, env=git_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # An empty repository proves the bundle has no missing prerequisites.
    bare = temporary / f"{name}_empty_bundle_check.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(bare)], check=True, env=git_env())
    bundle_verification = git(bare, "bundle", "verify", str(bundle))
    need(not git(repository, "status", "--porcelain", "--untracked-files=all")
         and git(repository, "rev-parse", "HEAD") == expected, "Repository changed during snapshot")
    audit = {"commit": expected, "status_porcelain": status, "clean": True,
             "bundle_sha256": sha(bundle), "archive_sha256": sha(archive),
             "bundle_has_no_external_prerequisites": True, "bundle_verification": bundle_verification,
             "ignored_entries": ignored_audit, "git_config_and_account_settings_included": False}
    audit_path = temporary / f"{name}_git_audit.json"
    write(audit_path, audit)
    return audit, {f"source/{name}.bundle": bundle, f"source/{name}.tar": archive,
                   f"source/{name}_git_audit.json": audit_path}


def audit_campaign(goal):
    wrapper = import_module(goal / "code/scripts/train/item_local_multiseed_gpu_job.py")
    lock_path = goal / "delivery/DEPLOYMENT_LOCK.json"
    lock = wrapper.source_gate(goal, lock_path)
    wrapper.cpu_gate(goal, goal / "delivery/CPU_GATE.json", lock_path)
    results = {}
    for seed in wrapper.SEEDS:
        seed_root = goal / "seeds" / f"seed{seed}"
        need(not (seed_root / ".active_gpu_wrapper").exists(), "Paired smoke wrapper is still active")
        pairs = {}
        receipts = [(p, read(p)) for p in (seed_root / "delivery").glob("*/receipt.json")]
        smoke = [(p, row) for p, row in receipts if row.get("status") == "smoke_completed"]
        need(len(smoke) == 1, "Need exactly one successful paired smoke receipt per seed")
        wrapper.smoke_gate(goal, smoke[0][0], seed, lock_path)
        for variant in wrapper.VARIANTS:
            run = wrapper.run_path(goal, seed, variant)
            need(not (run / ".active_gpu_wrapper").exists(), "Formal wrapper is still active")
            training = wrapper.training_result(goal, seed, variant, "formal")
            evaluation = wrapper.evaluation_result(run, seed, variant, training)
            matching = [(p, row) for p, row in receipts if row.get("status") == "formal_completed"
                        and row.get("variant") == variant and row.get("seed") == seed]
            need(len(matching) == 1, "Need exactly one successful formal receipt for each arm")
            completed = matching[0][1]
            need(completed["training"] == training and completed["evaluation"] == evaluation
                 and completed["owned_process_groups_alive_after_cleanup"] == [], "Formal wrapper/result binding differs")
            for mode, work in (("formal", run), ("smoke", run / "smoke")):
                records = read(work / "input_identities.json")["artifacts"]
                for item in records:
                    # Every immutable and private copy is included, not just identified.
                    bases = [seed_root / "inputs", work / "workspace"]
                    if mode == "formal":
                        bases.append(run / "validation/shared/workspace")
                    for base in bases:
                        target = byte.inside(base.resolve(), byte.safe_name(item["path"]))
                        need(byte.file_record(item["path"], target) == {"member": item["path"],
                             "sha256": item["sha256"], "size_bytes": item["size_bytes"]},
                             "Locked input bytes differ before full backup")
            pairs[variant] = {"training": training, "validation": evaluation,
                              "wrapper_receipt": wrapper.record(matching[0][0])}
        for field in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256", "input_artifacts"):
            need(pairs["control"]["training"][field] == pairs["item_local"]["training"][field],
                 "Same-seed A/B start differs before backup: " + field)
        results[str(seed)] = pairs
    return {"status": "COMPLETED_ARTIFACT_BINDINGS_VERIFIED", "deployment_lock": wrapper.record(lock_path),
            "source_identity": {name: lock[name + "_commit"] for name in ("code", "engine")},
            "results": results, "platform_gpu_release_audited_here": False}


def pack(goal, output, part_mib):
    need(not os.path.lexists(output), "Choose a new package directory")
    goal, output = goal.resolve(), output.resolve()
    need(not output.is_relative_to(goal), "Keep the package outside the complete campaign tree")
    acceptance = audit_campaign(goal)
    sources = source_inventory(goal)
    original_names = set(sources)
    raw_bytes = sum(path.stat().st_size for path in sources.values())
    print(json.dumps({"scope_files": len(sources), "scope_raw_bytes_before_git": raw_bytes,
                      "compression_size_known_only_after_pack": True}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="snapshot_", dir=output) as temp:
        temporary = Path(temp)
        identities = {}
        for name, commit in acceptance["source_identity"].items():
            identities[name], additions = repository_snapshot(goal, name, commit, temporary)
            sources.update(additions)
        audit_path = temporary / "artifact_acceptance.json"
        write(audit_path, acceptance)
        sources["artifact_acceptance.json"] = audit_path
        for name in TOOLS:
            need((HERE / name).is_file(), "Missing packaged verification tool: " + name)
            sources["backup_tools/" + name] = HERE / name
        metadata = {"source_identity": identities, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "Full private seed inputs/runs/delivery and goal delivery; exact source Git bundles/archives; no shared environment or account data",
                    "all_private_input_copies_included": True, "symlinks_followed": False,
                    "original_goal_root": str(goal), "scope_raw_bytes_before_git": raw_bytes,
                    "relocated_resume_exercised": False, "platform_gpu_release_audited_here": False,
                    "tool_sha256": sha(Path(__file__))}
        receipt = byte.archive_files(sources, output, metadata, part_mib * 1024 * 1024)
        need(set(source_inventory(goal)) == original_names, "Campaign file inventory changed while packaging")
        need(audit_campaign(goal) == acceptance, "Completed campaign bindings changed while packaging")
        for name, commit in acceptance["source_identity"].items():
            need(git(goal / name, "rev-parse", "HEAD") == commit
                 and not git(goal / name, "status", "--porcelain", "--untracked-files=all"),
                 "Frozen repository changed while packaging")
    write(output / "PACK_COMPLETE.json", {"status": "FULL_MULTISEED_PACKAGE_COMPLETED", "campaign_id": CAMPAIGN,
          "package_sha256": sha(output / "PACKAGE.json"), "archive_sha256": receipt["archive_sha256"],
          "manifest_sha256": receipt["manifest_sha256"], "full_source_byte_recheck": True,
          "final_completed_artifact_binding_recheck": True, "source_file_inventory_unchanged": True})
    return receipt


def package_identity(package):
    receipt, completed = read(package / "PACKAGE.json"), read(package / "PACK_COMPLETE.json")
    need(receipt["format"] == byte.FORMAT and receipt["campaign_id"] == completed["campaign_id"] == CAMPAIGN
         and completed["status"] == "FULL_MULTISEED_PACKAGE_COMPLETED"
         and completed["package_sha256"] == sha(package / "PACKAGE.json")
         and completed["archive_sha256"] == receipt["archive_sha256"]
         and completed["manifest_sha256"] == receipt["manifest_sha256"]
         and completed["full_source_byte_recheck"] is True
         and completed["final_completed_artifact_binding_recheck"] is True
         and completed["source_file_inventory_unchanged"] is True,
         "Full package completion does not bind this package receipt")
    return receipt


def verified_manifest(root, package):
    root = root.resolve()
    receipt = package_identity(package.resolve())
    byte_receipt, manifest = read(root / "BYTE_VERIFICATION.json"), read(root / "MANIFEST.json")
    need(byte_receipt["status"] == "BYTE_RESTORATION_VERIFIED" and manifest["format"] == byte.FORMAT
         and manifest["campaign_id"] == CAMPAIGN
         and byte_receipt["archive_sha256"] == receipt["archive_sha256"]
         and sha(root / "MANIFEST.json") == receipt["manifest_sha256"],
         "Full byte extraction must pass and match the external package receipt")
    entries = {row["member"]: row for row in manifest["files"]}
    need(len(entries) == len({name.casefold() for name in entries}) == len(manifest["files"])
         == byte_receipt["files"] == receipt["files"], "Verified member count differs")
    for member in entries:
        byte.safe_name(member)
    return manifest, entries


def restore_git(root, package, output):
    need(not os.path.lexists(output), "Choose a new Git restoration directory")
    root, output = root.resolve(), output.resolve()
    need(not output.is_relative_to(root), "Keep independent Git restoration outside extracted evidence")
    manifest, entries = verified_manifest(root, package)
    output.mkdir(parents=True, exist_ok=False)
    result = {}
    for name in ("code", "engine"):
        member = f"source/{name}.bundle"
        bundle = byte.inside(root, member)
        need(byte.file_record(member, bundle) == entries[member], "Verified Git bundle bytes changed")
        expected = manifest["source_identity"][name]["commit"]
        repository = output / name
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True, env=git_env())
        verification = git(repository, "bundle", "verify", str(bundle))
        git(repository, "fetch", "--no-tags", str(bundle), "HEAD")
        need(git(repository, "rev-parse", "FETCH_HEAD") == expected, "Bundle HEAD differs from frozen commit")
        git(repository, "checkout", "--detach", expected)
        need(git(repository, "rev-parse", "HEAD") == expected and not git(repository, "status", "--porcelain"),
             "Independent checkout did not restore cleanly")
        fsck = git(repository, "fsck", "--full", "--strict")
        result[name] = {"commit": expected, "bundle_sha256": sha(bundle), "clean": True,
                        "bundle_verify": verification, "git_fsck": fsck}
    receipt = {"status": "INDEPENDENT_GIT_RESTORE_PASSED", "campaign_id": CAMPAIGN,
               "source_manifest_sha256": sha(root / "MANIFEST.json"), "repositories": result,
               "GPU_used": False, "relocated_training_resume_exercised": False}
    write(output / "GIT_RESTORE.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("pack", "verify", "restore-git"))
    parser.add_argument("--goal-root", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=512)
    args = parser.parse_args()
    need(args.part_mib > 0, "Part size must be positive")
    if args.mode == "pack":
        need(args.goal_root is not None, "pack requires --goal-root")
        result = pack(args.goal_root, args.output, args.part_mib)
    elif args.mode == "verify":
        need(args.package_dir is not None, "verify requires --package-dir")
        package_identity(args.package_dir)
        result = byte.verify(args.package_dir, args.output)
    else:
        need(args.verified_root is not None and args.package_dir is not None,
             "restore-git requires --verified-root and --package-dir")
        result = restore_git(args.verified_root, args.package_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
