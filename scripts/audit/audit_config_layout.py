"""B2b configuration relocation audit; no training, restoration or resealing."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess

from audit_layout_migration import digest, read, write_new


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reproducibility/layout_migrations/20261008_b2b_configs"
CONFIG_GROUPS = {
    "shared": (
        "ar_formal_common.yaml", "diff_formal_common.yaml", "diff_formal_repo_v6.yaml",
        "diff_smoke.yaml", "diff_profile.yaml",
    ),
    "01_ar_baseline_3x256": (
        "ar_formal_rqkmeans.yaml", "ar_formal_rqvae.yaml", "ar_smoke_6gb.yaml", "ar_smoke_v2_unique.yaml",
    ),
    "02_ar_capacity_ablation": (
        "ar_formal_pse.yaml", "ar_formal_rqkmeans_128_256_512.yaml", "ar_formal_rqkmeans_512_256_128.yaml",
    ),
    "03_diffgrm_tokenizer_comparison": (
        "diff_formal_pse.yaml", "diff_formal_rqkmeans_v6.yaml", "diff_formal_rqvae_v6.yaml",
        "diff_formal_bf16_v3.yaml", "diff_formal_bf16_v4.yaml", "diff_formal_repo_v5.yaml",
    ),
    "04_crosspse_offline": (
        "diff_formal_crosspse_v1.yaml", "diff_formal_crosspse_v2.yaml", "diff_formal_crosspse_v3.yaml",
        "diff_formal_crosspse_v3_catalog.yaml", "diff_formal_refinedpse_v1.yaml", "diff_formal_refinedpse_v2_catalog.yaml",
    ),
    "05_crosspse_joint": (
        "diff_formal_crosspse_joint_sia.yaml", "diff_formal_crosspse_joint_diger.yaml",
        "diff_formal_crosspse_joint_diger_v2.yaml", "diff_formal_crosspse_frozen_decoder_v1.yaml",
        "diff_formal_refinedpse_joint_diger_v1.yaml", "diff_formal_refinedpse_frozen_decoder_v1.yaml",
        "diff_joint_smoke.yaml", "diff_diger_smoke.yaml", "diff_frozen_crosspse_smoke.yaml",
        "diff_frozen_refinedpse_smoke.yaml",
    ),
}
MOVES = {f"configs/{name}": f"configs/{group}/{name}"
         for group, names in CONFIG_GROUPS.items() for name in names}
PYTHON_CALLERS = (
    "analyze_ar_self_attention.py", "analyze_diff_cpd_attention.py", "preflight_formal.py",
    "preflight_diff_formal.py", "preflight_diff_crosspse_joint.py", "preflight_diff_crosspse_diger.py",
    "preflight_diff_crosspse_diger_v2.py", "preflight_diff_crosspse_frozen.py",
    "preflight_diff_refinedpse_joint.py", "preflight_diff_refinedpse_frozen.py",
)
SHELL_CALLERS = (
    "profile_formal.sh", "run_formal_pair.sh", "run_ar_internal_study.sh",
    "run_ar_smoke.sh", "run_ar_smoke.ps1", "run_ar_smoke_v2.sh", "run_ar_smoke_v2.ps1",
    "run_diff_formal.sh", "run_diff_rqkmeans_v6.sh", "run_diff_rqvae_v6.sh",
    "run_diff_crosspse_v1.sh", "run_diff_crosspse_v2.sh", "run_diff_pse_refinement.sh",
    "run_diff_crosspse_joint.sh", "run_diff_crosspse_diger.sh", "run_diff_crosspse_diger_v2.sh",
    "run_diff_crosspse_frozen.sh", "run_diff_refinedpse_joint.sh", "run_diff_refinedpse_frozen.sh",
)
CALLERS = {f"scripts/{name}" for name in PYTHON_CALLERS + SHELL_CALLERS}
NAVIGATION = {
    ".gitattributes", "README.md", "configs/README.md", "scripts/README.md", "tests/README.md",
    "docs/README.md", "docs/guides/README.md", "docs/guides/EXPERIMENT_WORKFLOW.md",
    "docs/guides/PROJECT_RESTRUCTURE_PLAN.md", "reproducibility/README.md",
    "experiments/05_crosspse_joint/reproducibility/README.md",
    "tests/test_entrypoint_layout.py",
}


def spellings(old, new):
    """Exact slash, Windows and pathlib spellings; never rewrites YAML."""
    yield old, new
    yield old.replace("/", "\\"), new.replace("/", "\\")
    for quote in ('"', "'"):
        yield " / ".join(quote + part + quote for part in old.split("/")), \
              " / ".join(quote + part + quote for part in new.split("/"))


def relocate_text(text, reverse=False):
    """Audit/test utility only, not a runtime config resolver."""
    for old, new in MOVES.items():
        for original, relocated in spellings(old, new):
            text = text.replace(relocated, original) if reverse else text.replace(original, relocated)
    return text


def normalized_sha(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def references(text, relocated=False):
    matches = []
    for old, new in MOVES.items():
        for original, target in spellings(old, new):
            pattern = target if relocated else original
            matches.extend((match.start(), old) for match in re.finditer(re.escape(pattern), text))
    return [old for _, old in sorted(matches)]


def capture():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not head.startswith("07d6a83"):
        raise ValueError("Capture requires the reviewed B2a commit 07d6a83")
    if {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").glob("*.yaml")} != set(MOVES):
        raise ValueError("Unexpected original configuration inventory")
    if any((ROOT / new).exists() for new in MOVES.values()):
        raise ValueError("Configuration destinations must not already exist")
    paths = sorted(filter(None, subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")))
    callers = {}
    for path in sorted(CALLERS):
        text = (ROOT / path).read_text(encoding="utf-8")
        callers[path] = {"normalized_sha256": normalized_sha(text), "config_sequence": references(text)}
        if not callers[path]["config_sequence"]:
            raise ValueError(f"Caller does not reference a mapped configuration: {path}")
    write_new(AUDIT / "before.json", {
        "baseline_commit": head, "mapping": MOVES,
        "files": [{"before": path, "after": MOVES.get(path, path), "sha256": digest(ROOT / path)} for path in paths],
        "callers": callers,
    })
    print(f"Captured {len(paths)} files, {len(MOVES)} configs, {len(callers)} callers")


def check_caller(path, text, record):
    if normalized_sha(relocate_text(text, reverse=True)) != record["normalized_sha256"]:
        raise ValueError(f"Caller changed beyond configuration paths: {path}")
    if references(text) or references(text, relocated=True) != record["config_sequence"]:
        raise ValueError(f"Configuration references missing or reordered: {path}")


def verify(write_receipt=False, check_receipt=False):
    before = read(AUDIT / "before.json")
    if before["mapping"] != MOVES or set(before["callers"]) != CALLERS:
        raise ValueError("Mapping or caller scope changed after capture")
    configs = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.yaml")}
    if configs != set(MOVES.values()) or any((ROOT / old).exists() for old in MOVES):
        raise ValueError("Unexpected configuration inventory or duplicate old entry")
    package_paths = {r["after"] for r in before["files"] if r["before"].startswith("experiments/")}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*") if p.is_file()}
    if actual != package_paths:
        raise ValueError("Experiment package inventory changed")
    changes, unchanged = [], []
    for row in before["files"]:
        current = digest(ROOT / row["after"])
        if current == row["sha256"]:
            unchanged.append(row["after"])
        elif row["after"] not in CALLERS | NAVIGATION:
            raise ValueError(f"Protected configuration/evidence/helper changed: {row['after']}")
        else:
            changes.append({**row, "after_sha256": current})
    for path, record in before["callers"].items():
        check_caller(path, (ROOT / path).read_text(encoding="utf-8"), record)
    result = {
        "before_manifest_sha256": digest(AUDIT / "before.json"), "baseline_commit": before["baseline_commit"],
        "checked_existing_files": len(before["files"]), "byte_identical_files": len(unchanged),
        "config_moves": len(MOVES), "config_contents_unchanged": True,
        "callers_with_path_only_changes": len(CALLERS),
        "unique_caller_config_pairs": sum(len(set(row["config_sequence"])) for row in before["callers"].values()),
        "configuration_reference_order_unchanged": True,
        "package_file_count": len(package_paths), "changed_files": changes,
        "scientific_files_and_integrity_seals_unchanged": True,
        "training_performed": False, "old_materials_deleted": False,
    }
    if write_receipt:
        write_new(AUDIT / "receipt.json", result)
    if check_receipt and read(AUDIT / "receipt.json") != result:
        raise ValueError("Current state differs from the B2b receipt")
    print(f"B2b verified: {len(before['files'])} checked, {len(unchanged)} byte-identical, "
          f"{len(changes)} reviewed edits; 34 unchanged configs, 29 path-only callers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-receipt", action="store_true")
    group.add_argument("--check-receipt", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    else:
        verify(args.write_receipt, args.check_receipt)
