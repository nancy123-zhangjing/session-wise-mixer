"""B2c script categorization audit. Does not train or rebuild experiment packages."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reproducibility/layout_migrations/20261008_b2c_scripts"
BASELINE = "3d41254"
GROUPS = {
    "repro": "project_repro.py project_replay.py stage05_repro.py asset_locations.py".split(),
    "audit": "audit_layout_migration.py audit_repro_workflow.py audit_entrypoint_layout.py audit_config_layout.py".split(),
    "prepare": "bootstrap_upstreams.sh bootstrap_upstreams.ps1 prepare_formal_embeddings.py prepare_diff_crosspse_fit_mask.py prepare_diff_rqvae_inputs.py".split(),
    "tokenizers": "train_cross_pse_tokenizer.py train_rqvae_tokenizer.py inspect_crosspse_tokenizer.py".split(),
    "reports": "build_crosspse_stage_report.py build_stage1_pdf_report.py build_stage1_readme_assets.py render_stage1_pdf.py".split(),
    "analysis": "analyze_ar_self_attention.py analyze_diff_cpd_attention.py analyze_sid_collisions.py analyze_tokenizer_geometry.py compare_diff_tokenizers.py compare_pse_refinement_tokenizers.py summarize_formal_results.py summarize_diff_formal.py summarize_diff_attention.py summarize_ar_self_attention.py summarize_ar_internal_study.py summarize_ar_internal_attention.py plot_ar_internal_study.py plot_position_loss.py analyze_ar_smoke_v2.sh run_formal_attention.sh".split(),
    "checks": "preflight_diff_crosspse_diger_v2.py preflight_diff_crosspse_diger.py preflight_diff_crosspse_frozen.py preflight_diff_crosspse_joint.py preflight_diff_formal.py preflight_diff_refinedpse_frozen.py preflight_diff_refinedpse_joint.py preflight_formal.py check_diff_crosspse_diger_smoke.py check_diff_crosspse_frozen_smoke.py check_diff_refinedpse_frozen_smoke.py check_diff_refinedpse_joint_smoke.py verify_diff_crosspse_artifact.py verify_diff_rqvae_artifact.py find_diff_checkpoint.py".split(),
    "train": "profile_formal.sh run_ar_internal_study.sh run_ar_smoke_v2.ps1 run_ar_smoke_v2.sh run_ar_smoke.ps1 run_ar_smoke.sh run_diff_crosspse_diger_v2.sh run_diff_crosspse_diger.sh run_diff_crosspse_frozen.sh run_diff_crosspse_joint.sh run_diff_crosspse_v1.sh run_diff_crosspse_v2.sh run_diff_crosspse_v3_catalog.sh run_diff_crosspse_v3.sh run_diff_formal.sh run_diff_pse_refinement.sh run_diff_refinedpse_frozen.sh run_diff_refinedpse_joint.sh run_diff_refinedpse_v1.sh run_diff_refinedpse_v2_catalog.sh run_diff_rqkmeans_v6.sh run_diff_rqvae_v6.sh run_formal_pair.sh".split(),
    "legacy": ["archive_diff_pilot.py"],
}
MOVES = {f"scripts/{name}": f"scripts/{group}/{name}"
         for group, names in GROUPS.items() for name in names}
STRICT_TESTS = {
    "tests/test_asset_locations.py", "tests/test_daily_repro.py", "tests/test_legacy_import.py",
    "tests/test_project_layout.py", "tests/test_project_repro.py", "tests/test_stage05_repro.py",
}
LAYOUT_TESTS = {"tests/test_entrypoint_layout.py", "tests/test_config_layout.py"}
LEGACY_IMPORT = "scripts/legacy/import_experiments.py"
SEALED_HELPERS = {f"scripts/{name}" for name in GROUPS["repro"]} | {
    "tests/test_project_repro.py", "tests/test_stage05_repro.py",
}
NAVIGATION = {
    ".gitattributes", ".gitignore", "README.md", "scripts/README.md", "tests/README.md",
    "docs/README.md", "docs/guides/README.md", "docs/guides/EXPERIMENT_WORKFLOW.md",
    "docs/guides/PROJECT_RESTRUCTURE_PLAN.md", "configs/README.md",
    "reproducibility/README.md", "reproducibility/WORKFLOW_B1.md", "scripts/legacy/README.md",
} | {f"experiments/{stage}/reproducibility/README.md" for stage in (
    "01_ar_baseline_3x256", "02_ar_capacity_ablation", "03_diffgrm_tokenizer_comparison",
    "04_crosspse_offline", "05_crosspse_joint",
)}
ADDITIONS = {
    "scripts/audit/audit_script_layout.py", "tests/test_script_layout.py",
    "docs/guides/PROJECT_MAP.md",
    "reproducibility/layout_migrations/20261008_b2c_scripts/before.json",
    "reproducibility/layout_migrations/20261008_b2c_scripts/README.md",
    "reproducibility/layout_migrations/20261008_b2c_scripts/receipt.json",
} | {f"scripts/{group}/README.md" for group in GROUPS if group != "legacy"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalized_sha(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def baseline():
    """Immutable B2b snapshot; not a substitute for checking current files."""
    return read(AUDIT / "before.json")


def spellings(old, new):
    yield old, new
    yield old.replace("/", "\\"), new.replace("/", "\\")
    for quote in ('"', "'"):
        yield " / ".join(quote + part + quote for part in old.split("/")), \
              " / ".join(quote + part + quote for part in new.split("/"))


def relocate_text(text, reverse=False):
    """Exact spelling conversion for audit only; not a runtime path resolver."""
    pairs = [pair for old, new in MOVES.items() for pair in spellings(old, new)]
    for old, new in sorted(pairs, key=lambda pair: len(pair[1] if reverse else pair[0]), reverse=True):
        text = text.replace(new, old) if reverse else text.replace(old, new)
    return text


def restore_script_text(old, text):
    """Undo only the reviewed path, root-depth and import bootstrap changes.

    Historical auditors are deliberately excluded from path substitution: their
    old mappings, helper constants and receipts describe past migrations.
    """
    if old == LEGACY_IMPORT:
        return text.replace('str(ROOT / "scripts" / "repro")', 'str(ROOT / "scripts")')
    if old not in MOVES:
        raise ValueError(f"Not a captured script: {old}")
    if MOVES[old].startswith("scripts/audit/"):
        return text.replace("Path(__file__).resolve().parents[2]", "Path(__file__).resolve().parents[1]")
    text = relocate_text(text, reverse=True)
    text = text.replace("Path(__file__).resolve().parents[2]", "Path(__file__).resolve().parents[1]")
    text = text.replace('$(dirname "${BASH_SOURCE[0]}")/../..', '$(dirname "${BASH_SOURCE[0]}")/..')
    text = text.replace(
        "$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)",
        "$ProjectRoot = Split-Path -Parent $PSScriptRoot")
    if old == "scripts/preflight_diff_formal.py":
        text = text.replace('root / "scripts" / "train" / runner_name', 'root / "scripts" / runner_name')
    return text


def byte_hash_candidates(text):
    """Accept checkout newline translation, not text or whitespace edits."""
    lf = text.replace("\r\n", "\n")
    return {hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in (text, lf, lf.replace("\n", "\r\n"))}


def check_script(old, text, before=None):
    before = baseline() if before is None else before
    restored = restore_script_text(old, text)
    if old in MOVES:
        if normalized_sha(restored) != before["script_normalized_sha256"][old]:
            raise ValueError(f"Script changed beyond reviewed paths/root/import: {old}")
        if "Path(__file__).resolve().parents[1]" in text:
            raise ValueError(f"Script root depth was not relocated: {old}")
        if ('$(dirname "${BASH_SOURCE[0]}")/.."' in text or
                "$ProjectRoot = Split-Path -Parent $PSScriptRoot" in text):
            raise ValueError(f"Shell root depth was not relocated: {old}")
        if not MOVES[old].startswith("scripts/audit/"):
            if any(original in text for source, target in MOVES.items()
                   for original, _ in spellings(source, target)):
                raise ValueError(f"Live script still refers to a flat entry: {old}")
        if old == "scripts/preflight_diff_formal.py" and 'root / "scripts" / runner_name' in text:
            raise ValueError("Dynamic training runner directory was not relocated")
    else:
        record = next(row for row in before["files"] if row["before"] == old)
        if record["sha256"] not in byte_hash_candidates(restored):
            raise ValueError(f"Legacy loader changed beyond its import bootstrap: {old}")
        if 'str(ROOT / "scripts" / "repro")' not in text:
            raise ValueError("Legacy loader import bootstrap was not relocated")
    return restored


def historical_text(old):
    """Verified current source projected back to the B2b layout for old tests."""
    return check_script(old, (ROOT / MOVES.get(old, old)).read_text(encoding="utf-8"))


def snapshot_digest(path):
    """Read B2b snapshot hashes only when exercising historical seal rules.

    Current seals are checked independently by check_seal; old audit logic must
    not be weakened to accept newer paths or refreshed scientific hashes.
    """
    relative = Path(path).relative_to(ROOT).as_posix()
    return next(row["sha256"] for row in baseline()["files"] if row["before"] == relative)


def restore_test_text(path, text, before=None):
    if path not in STRICT_TESTS:
        raise ValueError(f"Not a path-only test adapter: {path}")
    before = baseline() if before is None else before
    text = text.replace("\r\n", "\n")
    if 'str(ROOT / "scripts" / "repro")' not in text:
        raise ValueError(f"Test import bootstrap was not relocated: {path}")
    if path == "tests/test_project_layout.py" and (
            'str(ROOT / "scripts" / "audit")' not in text or
            'ROOT / "scripts" / "reports" / name' not in text):
        raise ValueError("Layout test audit/report locations were not relocated")
    if path == "tests/test_daily_repro.py" and 'ROOT / "scripts/project_repro.py"' in text:
        raise ValueError("Daily plan CLI path was not relocated")
    text = relocate_text(text, reverse=True)
    if path == "tests/test_project_layout.py":
        text = text.replace(
            'sys.path.insert(0, str(ROOT / "scripts" / "repro"))\n'
            'sys.path.insert(0, str(ROOT / "scripts" / "audit"))',
            'sys.path.insert(0, str(ROOT / "scripts"))')
        text = text.replace('ROOT / "scripts" / "reports" / name', 'ROOT / "scripts" / name')
    else:
        text = text.replace('str(ROOT / "scripts" / "repro")', 'str(ROOT / "scripts")')
    record = next(row for row in before["files"] if row["before"] == path)
    if record["sha256"] not in byte_hash_candidates(text):
        raise ValueError(f"Test changed beyond path/import bootstrap: {path}")
    return text


def check_seal(old, current):
    expected = copy.deepcopy(old)
    for key in ("shared", "helpers"):
        for row in expected.get(key, []):
            original = row["path"]
            if original in SEALED_HELPERS:
                row["path"] = MOVES.get(original, original)
                row["sha256"] = digest(ROOT / row["path"])
    if expected != current:
        raise ValueError("Seal changed beyond six reviewed helper/test path/hash bindings")


def file_inventory(directory):
    return {path.relative_to(ROOT).as_posix() for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.relative_to(ROOT).as_posix() != "reproducibility/asset_locations.local.json"}


def verify(write_receipt=False, check_receipt=False):
    before = baseline()
    if (before["mapping"] != MOVES or len(MOVES) != 75 or
            not before["baseline_commit"].startswith(BASELINE) or len(before["seals"]) != 5):
        raise ValueError("Captured migration scope changed")
    records = before["files"]
    if len(records) != 732 or len({row["before"] for row in records}) != 732:
        raise ValueError("Captured file scope changed")
    required = ADDITIONS - {"reproducibility/layout_migrations/20261008_b2c_scripts/receipt.json"}
    if any(not (ROOT / path).is_file() for path in required):
        raise ValueError("Required B2c audit/test/navigation file is missing")
    if any((ROOT / old).exists() for old in MOVES):
        raise ValueError("Old flat script entry or compatibility copy still exists")
    expected_code = set(MOVES.values()) | {LEGACY_IMPORT, "scripts/audit/audit_script_layout.py"}
    actual_code = {path for path in file_inventory("scripts") if Path(path).suffix in {".py", ".sh", ".ps1"}}
    if actual_code != expected_code:
        raise ValueError(f"Unexpected script inventory: {sorted(actual_code ^ expected_code)}")
    for directory in ("scripts", "tests", "docs", "configs", "experiments", "reproducibility"):
        expected = {row["after"] for row in records if row["after"].startswith(directory + "/")}
        actual = file_inventory(directory)
        allowed = {path for path in ADDITIONS if path.startswith(directory + "/")}
        if not expected <= actual or actual - expected - allowed:
            raise ValueError(f"Unexpected {directory} inventory: {sorted((expected - actual) | (actual - expected - allowed))}")
    changed, unchanged = [], []
    for row in records:
        old, new = row["before"], row["after"]
        current = digest(ROOT / new)
        if old in MOVES or old == LEGACY_IMPORT:
            check_script(old, (ROOT / new).read_text(encoding="utf-8"), before)
        elif old in STRICT_TESTS:
            restore_test_text(old, (ROOT / new).read_text(encoding="utf-8"), before)
        elif old in before["seals"]:
            check_seal(before["seals"][old], read(ROOT / new))
        elif current != row["sha256"] and old not in NAVIGATION | LAYOUT_TESTS:
            raise ValueError(f"Protected scientific/historical content changed: {old}")
        if current == row["sha256"]:
            unchanged.append(new)
        else:
            changed.append({**row, "after_sha256": current})
    result = {
        "before_manifest_sha256": digest(AUDIT / "before.json"), "baseline_commit": before["baseline_commit"],
        "checked_existing_files": len(records), "byte_identical_files": len(unchanged),
        "script_moves": len(MOVES), "path_only_script_checks": len(MOVES) + 1,
        "strict_test_adapters": len(STRICT_TESTS), "changed_files": changed,
        "package_file_count": len(file_inventory("experiments")), "integrity_seals_checked": 5,
        "new_files": {path: digest(ROOT / path) for path in sorted(ADDITIONS)
                      if (ROOT / path).is_file() and not path.endswith("/receipt.json")},
        "scientific_files_unchanged": True, "training_performed": False, "old_materials_deleted": False,
    }
    if write_receipt:
        write_new(AUDIT / "receipt.json", result)
    if check_receipt and read(AUDIT / "receipt.json") != result:
        raise ValueError("Current state differs from the B2c receipt")
    print(f"B2c verified: {len(records)} checked, {len(unchanged)} byte-identical, "
          f"{len(changed)} reviewed edits; 75 categorized scripts with path-only checks")
    return result


def capture():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not head.startswith(BASELINE):
        raise ValueError("Capture requires committed B2b baseline")
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").iterdir()
              if p.is_file() and p.suffix in {".py", ".sh", ".ps1"}}
    if actual != set(MOVES) or any((ROOT / path).exists() for path in MOVES.values()):
        raise ValueError(f"Unexpected source/destination inventory: {actual ^ set(MOVES)}")
    paths = sorted(filter(None, subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")))
    seals = {path: read(ROOT / path) for path in paths if path.endswith("/package_integrity.json")}
    if len(seals) != 5:
        raise ValueError("Expected exactly five preserved integrity seals")
    write_new(AUDIT / "before.json", {
        "baseline_commit": head, "mapping": MOVES,
        "files": [{"before": path, "after": MOVES.get(path, path), "sha256": digest(ROOT / path)} for path in paths],
        "seals": seals,
        "script_normalized_sha256": {path: normalized_sha((ROOT / path).read_text(encoding="utf-8")) for path in MOVES},
    })
    print(f"Captured {len(paths)} tracked files, {len(MOVES)} script moves, {len(seals)} seals")


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
