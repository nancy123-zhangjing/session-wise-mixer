"""One-time historical imports into an isolated, unsealed checkout.

The default is a read-only plan. This is not a seal repair or rebuild command.
Existing or partial target packages are never overwritten, even with --execute.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

# Importing a daily helper must not create caches before the destination guard.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "01-04": (
        "01_ar_baseline_3x256", "02_ar_capacity_ablation",
        "03_diffgrm_tokenizer_comparison", "04_crosspse_offline",
    ),
    "05": ("05_crosspse_joint",),
}
SHARED_CONTROLS = (
    "sources.lock.json", "migration_assets.lock.json", "code_versions.json",
    "source_comparison.json", "code", "environment",
)
ENVIRONMENT_FILES = ("grm-exp.environment.yml", "grm-exp.pip-freeze.txt", "SHA256SUMS")
SMALL_ARCHIVES = (
    "01_ar_baseline_full_evidence.tar.gz", "01_ar_baseline_summary.tar.gz",
    "01_ar_baseline_attention.tar.gz", "02_ar_capacity_evidence_seed2024.tar.gz",
    "02_ar_capacity_attention_seed2024.tar.gz", "03_diffgrm_tokenizers_evidence_seed2024.tar.gz",
)


def present(path):
    """A broken link is also an occupied destination, never a fresh location."""
    return path.exists() or path.is_symlink()


def destination_blockers(stages, root=ROOT):
    """Inspect only filesystem names; do not import builders, read locks or Git."""
    root = Path(root)
    blockers = []
    experiments = root / "experiments"
    if (experiments.is_symlink() or not experiments.resolve().is_relative_to(root.resolve())
            or (present(experiments) and not experiments.is_dir())):
        blockers.append(experiments)
    for name in PACKAGES[stages]:
        package = experiments / name
        if (package.is_symlink() or not package.resolve().is_relative_to(root.resolve())
                or (present(package) and not package.is_dir())):
            blockers.append(package)
        elif package.is_dir():
            # A README template is the only permitted pre-existing package file.
            blockers.extend(p for p in package.iterdir()
                            if p.name != "README.md" or not p.is_file() or p.is_symlink())
    shared = root / "reproducibility"
    if (shared.is_symlink() or not shared.resolve().is_relative_to(root.resolve())
            or (present(shared) and not shared.is_dir())):
        blockers.append(shared)
    else:
        blockers.extend(shared / name for name in SHARED_CONTROLS if present(shared / name))
    registry = experiments / "experiment_registry.json"
    if stages == "01-04" and present(registry):
        blockers.append(registry)
    return sorted({p.as_posix() for p in blockers})


def require_fresh_destination(stages, root=ROOT):
    blockers = destination_blockers(stages, root)
    if blockers:
        raise ValueError(
            "Historical import refused: existing or partial target package/shared controls. "
            "Use an isolated first-import checkout; never remove controls to repair a seal.\n"
            + "\n".join(blockers)
        )


def source_root(value):
    source = Path(value).resolve()
    return source if source.name == "experiment1" else source / "experiment1"


def require_sources(args, root=ROOT):
    """Check coarse prerequisites before old importers can leave partial output."""
    root = Path(root)
    archive = Path(args.archive).resolve()
    required_files = [archive, *(archive.parent / name for name in ENVIRONMENT_FILES)]
    required_directories = []
    if args.stages == "01-04":
        cache = root / "archive/evidence_bundles/repro_sources_01_04"
        if not cache.resolve().is_relative_to(root.resolve()):
            raise ValueError("Historical evidence cache resolves outside the isolated checkout")
        required_files += [root / "archive/evidence_bundles" / name for name in SMALL_ARCHIVES]
        required_files += [root / "experiments" / name / "README.md" for name in PACKAGES["01-04"]]
        # The original registry builder reads Stage 05 after writing stages 01-04.
        # Detect an unavailable prerequisite before any of those writes instead.
        required_files.append(root / "experiments/05_crosspse_joint/runs.csv")
        cross = root / "archive/deprecated_and_temp/tmp_browser_cache/crosspse_report_20260829/crosspse_report_20260829_094937"
        required_directories += [cross / "reports", cross / "tokenizers/pilots",
                                 root / "outputs/run_metadata/AmazonReviews2014/DIFF_GRM"]
    else:
        if not args.source:
            raise ValueError("Stage 05 historical import requires --source")
        source = source_root(args.source)
        required_directories += [source / "outputs/run_metadata/AmazonReviews2014/DIFF_GRM",
                                 source / "outputs/reports", source / "configs"]
    missing = [p.as_posix() for p in required_files if not p.is_file()]
    missing += [p.as_posix() for p in required_directories if not p.is_dir()]
    if missing:
        raise ValueError("Historical import prerequisites are missing; no import performed:\n" + "\n".join(missing))


def require_expected_commit(expected, root=ROOT):
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", expected):
        raise ValueError("--execute requires --expected-root-commit with the complete reviewed Git commit ID")
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], text=True
    ).strip()
    if actual.lower() != expected.lower():
        raise ValueError(f"Root commit mismatch: expected {expected}, found {actual}; no import performed")


def load_builder(stages):
    scripts = str(ROOT / "scripts" / "repro")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("project_repro" if stages == "01-04" else "stage05_repro")._legacy_build


def run(args, root=ROOT):
    if not args.execute:
        plan = {
            "mode": "plan_only", "stages": args.stages,
            "checkout": Path(root).resolve().as_posix(), "archive": str(args.archive),
            "source": args.source,
            "target_packages": ["experiments/" + name for name in PACKAGES[args.stages]],
            "execution_blockers": destination_blockers(args.stages, root),
            "requirements": [
                "Isolated first-import checkout with manually prepared historical sources.",
                "Explicit --execute and --expected-root-commit; no overwrite or force option.",
                "Import Stage 05 before stages 01-04; the latter require Stage 05 runs.csv.",
                "Existing locks and code snapshots are not repaired or rebound.",
            ],
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("Plan only. No locks read, Git HEAD queried, import performed, or files written.")
        return
    # This ordering is intentional: occupied destinations fail before reading
    # any lock/HEAD, resolving source archives, importing or invoking a builder.
    require_fresh_destination(args.stages, root)
    require_sources(args, root)
    require_expected_commit(args.expected_root_commit, root)
    load_builder(args.stages)(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", choices=PACKAGES, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source", help="Stage 05 extracted source, or its parent; required to execute Stage 05")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-root-commit", help="Full reviewed checkout commit; required with --execute")
    args = parser.parse_args(argv)
    if args.stages != "05" and args.source:
        parser.error("--source applies only to Stage 05")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
