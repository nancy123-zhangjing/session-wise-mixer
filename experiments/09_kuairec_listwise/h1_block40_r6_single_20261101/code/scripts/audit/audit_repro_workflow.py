"""One-time B1 workflow audit; never changes scientific files or reseals packages."""
import argparse
import copy
import hashlib
from pathlib import Path
import subprocess

from audit_layout_migration import digest, read, write_new


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reproducibility/layout_migrations/20261007_b1_workflow"
ALLOWED = {
    ".gitattributes", ".gitignore", "README.md", "scripts/README.md", "docs/README.md",
    "docs/EXPERIMENT_WORKFLOW.md", "docs/PROJECT_RESTRUCTURE_PLAN.md", "reproducibility/README.md",
    "experiments/05_crosspse_joint/reproducibility/README.md", "archive/README.md",
    "scripts/project_repro.py", "scripts/stage05_repro.py",
}
CHANGED_HELPERS = {"scripts/project_repro.py", "scripts/stage05_repro.py"}
NEW_HELPER = "scripts/asset_locations.py"


def capture():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = sorted(set(filter(None, paths)) | {"archive/README.md"})
    staged = subprocess.check_output(["git", "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary"], cwd=ROOT)
    write_new(AUDIT / "before.json", {
        "git_head_before": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
        "basis": "Reviewed Batch A working files; A commit awaiting separate confirmation at capture time",
        "files": [{"path": p, "sha256": digest(ROOT / p)} for p in paths],
        "seals": {p: read(ROOT / p) for p in paths if p.startswith("experiments/") and p.endswith("/package_integrity.json")},
    })
    print(f"B1 baseline: {len(paths)} files recorded")


def verify(write_receipt=False):
    before = read(AUDIT / "before.json")
    package_paths = {r["path"] for r in before["files"] if r["path"].startswith("experiments/")}
    actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*") if p.is_file()}
    if package_paths != actual_paths:
        raise ValueError(f"Unexpected package file additions/removals: {sorted(package_paths ^ actual_paths)}")
    changed, unchanged = [], []
    for record in before["files"]:
        path = ROOT / record["path"]
        current = digest(path)
        if current == record["sha256"]:
            unchanged.append(record["path"])
        elif record["path"] not in ALLOWED and record["path"] not in before["seals"]:
            raise ValueError(f"Protected file changed: {record['path']}")
        else:
            changed.append({**record, "after_sha256": current})
    for path, seal in before["seals"].items():
        expected = copy.deepcopy(seal)
        key = "shared" if "shared" in seal else "helpers"
        for row in expected[key]:
            if row["path"] in CHANGED_HELPERS:
                row["sha256"] = digest(ROOT / row["path"])
        # Four existing helpers precede the shared locks in stages 01-04.
        index = 4 if key == "shared" else len(expected[key])
        expected[key].insert(index, {"path": NEW_HELPER, "sha256": digest(ROOT / NEW_HELPER)})
        if read(ROOT / path) != expected:
            raise ValueError(f"Unexpected control binding change: {path}")
    result = {
        "before_manifest_sha256": digest(AUDIT / "before.json"),
        "checked_existing_files": len(before["files"]), "byte_identical_files": len(unchanged),
        "package_file_count": len(package_paths),
        "changed_files": changed, "scientific_files_unchanged": True,
        "allowed_new_helper_binding": NEW_HELPER,
        "gpu_replay_performed": False, "archives_deleted": False,
    }
    if write_receipt:
        write_new(AUDIT / "receipt.json", result)
    print(f"B1 verified: {len(unchanged)} byte-identical files; {len(changed)} reviewed workflow/control changes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    else:
        verify(args.write_receipt)
