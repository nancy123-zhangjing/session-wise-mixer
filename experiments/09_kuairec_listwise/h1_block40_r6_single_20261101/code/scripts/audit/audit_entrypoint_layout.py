"""B2a migration audit: relocate tests/docs without rewriting research evidence.

This audit records path-only changes and explicitly reviewed control edits.
It never trains, restores assets, deletes sources, or reseals experiment packages.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess

from audit_layout_migration import digest, read, write_new


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reproducibility/layout_migrations/20261008_b2a_tests_docs"
TEST_NAMES = (
    "test_ar_variable_codebooks.py", "test_collision_aware_eval.py",
    "test_cross_pse_module.py", "test_joint_cross_pse.py",
    "test_diff_crosspse_diger.py", "test_diff_refinedpse_diger.py",
    "test_asset_locations.py", "test_daily_repro.py", "test_legacy_import.py",
    "test_project_layout.py", "test_project_repro.py", "test_stage05_repro.py",
)
DOC_GROUPS = {
    "guides": ("EXPERIMENT_WORKFLOW.md", "PROJECT_RESTRUCTURE_PLAN.md"),
    "methods": (
        "diffgrm_pipeline.md", "diffgrm_repo_reproduction_v6.md",
        "diffgrm_rqkmeans_v6.md", "diffgrm_rqvae_v6.md",
        "refinedpse_crosspse_v3.md", "diffgrm_crosspse_frozen_control.md",
        "diffgrm_refinedpse_joint_control.md", "diffgrm_refinedpse_frozen_control.md",
    ),
    "history": (
        "START_HERE.md", "FIRST_STAGE.md", "FORMAL_3X256.md",
        "diffgrm_repo_reproduction_v5.md", "diffgrm_crosspse_v1.md",
        "diffgrm_crosspse_v2.md", "diff_pilot_seed2024_warmup_misaligned_20260810.md",
        "diffgrm_crosspse_joint_sia.md", "diffgrm_crosspse_diger.md",
    ),
}
MOVES = {f"scripts/{name}": f"tests/{name}" for name in TEST_NAMES}
MOVES.update({f"docs/{name}": f"docs/{group}/{name}"
              for group, names in DOC_GROUPS.items() for name in names})
CALLER_NAMES = (
    "preflight_diff_crosspse_diger.py", "preflight_diff_crosspse_diger_v2.py",
    "preflight_diff_crosspse_joint.py", "preflight_diff_formal.py",
    "preflight_diff_refinedpse_joint.py", "run_ar_smoke_v2.ps1", "run_ar_smoke_v2.sh",
    "run_diff_crosspse_diger.sh", "run_diff_crosspse_diger_v2.sh",
    "run_diff_crosspse_frozen.sh", "run_diff_crosspse_joint.sh",
    "run_diff_crosspse_v1.sh", "run_diff_crosspse_v2.sh", "run_diff_pse_refinement.sh",
    "run_diff_refinedpse_frozen.sh", "run_diff_refinedpse_joint.sh",
)
HELPERS = {"scripts/project_repro.py", "scripts/stage05_repro.py"}
EDITABLE = HELPERS | {f"scripts/{name}" for name in CALLER_NAMES} | {
    ".gitattributes", "README.md", "scripts/README.md", "docs/README.md",
    "docs/guides/EXPERIMENT_WORKFLOW.md", "docs/guides/PROJECT_RESTRUCTURE_PLAN.md",
    "reproducibility/README.md", "reproducibility/WORKFLOW_B1.md",
    "experiments/05_crosspse_joint/reproducibility/README.md",
    "tests/test_asset_locations.py", "tests/test_legacy_import.py",
    "tests/test_project_layout.py", "tests/test_project_repro.py", "tests/test_stage05_repro.py",
}
NEW_TEST_FILES = {"tests/README.md", "tests/run_repro_checks.py", "tests/test_entrypoint_layout.py"}
NEW_DOC_FILES = {f"docs/{group}/README.md" for group in DOC_GROUPS}


def capture():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not head.startswith("5f4f433"):
        raise ValueError("Capture requires the reviewed B1 commit 5f4f433")
    for old, new in MOVES.items():
        if not (ROOT / old).is_file() or (ROOT / new).exists():
            raise ValueError(f"Expected unmoved source and absent destination: {old} -> {new}")
    paths = sorted(filter(None, subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")))
    write_new(AUDIT / "before.json", {
        "baseline_commit": head,
        "mapping": MOVES,
        "files": [{"before": p, "after": MOVES.get(p, p), "sha256": digest(ROOT / p)} for p in paths],
        "seals": {p: read(ROOT / p) for p in paths
                  if p.startswith("experiments/") and p.endswith("/package_integrity.json")},
    })
    print(f"Captured {len(paths)} existing files, {len(MOVES)} planned moves")


def check_seal(old, current):
    expected = copy.deepcopy(old)
    for key in ("shared", "helpers"):
        for row in expected.get(key, []):
            original_path = row["path"]
            row["path"] = MOVES.get(original_path, original_path)
            if original_path in HELPERS or original_path in MOVES:
                row["sha256"] = digest(ROOT / row["path"])
    if expected != current:
        raise ValueError("Seal changed beyond reviewed helper/test locations and hashes")


def verify(write_receipt=False, check_receipt=False):
    before = read(AUDIT / "before.json")
    if before["mapping"] != MOVES:
        raise ValueError("Migration mapping changed after capture")
    for old in MOVES:
        if (ROOT / old).exists():
            raise ValueError(f"Old entry point still present: {old}")
    records = before["files"]
    package_paths = {r["after"] for r in records if r["before"].startswith("experiments/")}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*") if p.is_file()}
    if package_paths != actual:
        raise ValueError(f"Package inventory changed: {sorted(package_paths ^ actual)}")
    for directory, additions in (("tests", NEW_TEST_FILES), ("docs", NEW_DOC_FILES)):
        expected = {r["after"] for r in records if r["after"].startswith(directory + "/")} | additions
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        if expected != actual:
            raise ValueError(f"Unexpected {directory} inventory: {sorted(expected ^ actual)}")
    unchanged, changed, moved = [], [], []
    for record in records:
        current_hash = digest(ROOT / record["after"])
        if record["before"] != record["after"]:
            moved.append({"before": record["before"], "after": record["after"]})
        if current_hash == record["sha256"]:
            unchanged.append(record["after"])
        elif record["after"] not in EDITABLE and record["before"] not in before["seals"]:
            raise ValueError(f"Protected content changed: {record['after']}")
        else:
            changed.append({**record, "after_sha256": current_hash})
    for path, seal in before["seals"].items():
        check_seal(seal, read(ROOT / path))
    result = {
        "before_manifest_sha256": digest(AUDIT / "before.json"),
        "baseline_commit": before["baseline_commit"],
        "checked_existing_files": len(records), "byte_identical_files": len(unchanged),
        "moves": moved, "changed_files": changed, "package_file_count": len(package_paths),
        "protected_method_history_docs": 17,
        "scientific_files_unchanged": True, "training_performed": False,
        "old_materials_deleted": False, "runtime_config_locations_changed": False,
    }
    if write_receipt:
        write_new(AUDIT / "receipt.json", result)
    if check_receipt and read(AUDIT / "receipt.json") != result:
        raise ValueError("Current migration state differs from the recorded B2a receipt")
    print(f"B2a verified: {len(records)} checked, {len(unchanged)} byte-identical, "
          f"{len(changed)} approved edits, {len(moved)} relocations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    receipts = parser.add_mutually_exclusive_group()
    receipts.add_argument("--write-receipt", action="store_true")
    receipts.add_argument("--check-receipt", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    else:
        verify(args.write_receipt, args.check_receipt)
