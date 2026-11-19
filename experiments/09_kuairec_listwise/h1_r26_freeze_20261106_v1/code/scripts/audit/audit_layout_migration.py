"""One-time Batch A receipt: prove that output -> experiments preserves research.

This is a historical migration audit, not an instruction to reseal later
scientific changes. It does not modify packages, evidence, or integrity seals.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reproducibility/layout_migrations/20261007_output_to_experiments"
BASELINE = "32f3b4e2039f40129f709646b535742eadd2de53"
EDITABLE = {
    ".gitattributes", "README.md", "configs/README.md", "scripts/README.md",
    "docs/README.md", "docs/EXPERIMENT_WORKFLOW.md", "reproducibility/README.md",
    "scripts/project_repro.py", "scripts/stage05_repro.py",
    "scripts/build_stage1_pdf_report.py", "scripts/build_stage1_readme_assets.py",
    "scripts/render_stage1_pdf.py", "output/README.md",
    "output/05_crosspse_joint/reproducibility/README.md",
    "output/experiment_registry.json",
}
HELPERS = {"scripts/project_repro.py", "scripts/stage05_repro.py"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relocated(path):
    return "experiments/" + path[len("output/"):] if path.startswith("output/") else path


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != BASELINE or not (ROOT / "output").is_dir() or (ROOT / "experiments").exists():
        raise ValueError("Capture requires the approved pre-migration checkout and original layout")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = sorted(set(filter(None, tracked)) | {
        p.relative_to(ROOT).as_posix() for p in (ROOT / "output").rglob("*") if p.is_file()
    })
    seals = {p: read(ROOT / p) for p in paths if p.startswith("output/") and p.endswith("/package_integrity.json")}
    write_new(AUDIT / "before.json", {
        "baseline_commit": head, "mapping": {"output/": "experiments/"},
        "files": [{"before": p, "after": relocated(p), "sha256": digest(ROOT / p)} for p in paths],
        "registry": read(ROOT / "output/experiment_registry.json"), "integrity_seals": seals,
    })
    print(f"Captured {len(paths)} files before migration")


def check_registry(before, after):
    expected = copy.deepcopy(before)
    for row in expected["runs"]:
        row["package"] = relocated(row["package"])
    if expected != after:
        raise ValueError("Registry changed beyond package location")


def check_seal(before, after, root=ROOT):
    expected = copy.deepcopy(before)
    for key in ("shared", "helpers"):
        for row in expected.get(key, []):
            if row["path"] in HELPERS:
                row["sha256"] = digest(root / row["path"])
    if expected != after:
        raise ValueError("Integrity seal changed beyond the two reviewed helper hashes")


def verify(write_receipt=False):
    before = read(AUDIT / "before.json")
    if (ROOT / "output").exists():
        raise ValueError("Old output directory still exists")
    records = before["files"]
    expected_paths = {r["after"] for r in records if r["before"].startswith("output/")}
    actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*") if p.is_file()}
    if expected_paths != actual_paths:
        raise ValueError(f"Package files added/removed: {sorted(expected_paths ^ actual_paths)}")
    allowed = EDITABLE | set(before["integrity_seals"])
    changes, preserved = [], []
    for record in records:
        path = ROOT / record["after"]
        if not path.is_file():
            raise ValueError(f"Missing file: {record['after']}")
        actual = digest(path)
        if actual == record["sha256"]:
            preserved.append(record["after"])
        elif record["before"] not in allowed:
            raise ValueError(f"Protected file changed: {record['before']}")
        else:
            changes.append({**record, "after_sha256": actual})
    check_registry(before["registry"], read(ROOT / "experiments/experiment_registry.json"))
    for old_path, value in before["integrity_seals"].items():
        check_seal(value, read(ROOT / relocated(old_path)))
    result = {
        "baseline_commit": before["baseline_commit"], "before_manifest_sha256": digest(AUDIT / "before.json"),
        "mapping": before["mapping"], "package_file_count": len(expected_paths),
        "checked_existing_files": len(records), "byte_identical_files": len(preserved),
        "byte_identical_package_files": len(set(preserved) & expected_paths),
        "changed_files": changes, "registry_non_path_fields_unchanged": True,
        "seals_only_updated_reviewed_helper_hashes": True,
        "note": "No historical code snapshots, evidence, recipes, metrics or initialization locks changed. No GPU replay performed.",
    }
    if write_receipt:
        write_new(AUDIT / "receipt.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "changed_files"}, ensure_ascii=False, indent=2))
    print(f"Reviewed path/document/seal changes: {len(changes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    else:
        verify(args.write_receipt)
