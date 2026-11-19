"""CPU regression tests: daily reproduction must not read historical sources.

Run this file from any working directory. Audit hooks live only in isolated
``python -B`` children. Real packages are read-only; analysis writes are allowed
only inside an explicitly created temporary package copy. No training runs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
MIGRATION_ARCHIVE = (
    "C:/Users/nancy/Downloads/experiment1_migration_20260923_retry/"
    "experiment1.tar.gz"
)


class SafetyViolation(RuntimeError):
    """Raised before an audited operation can touch a forbidden location."""


def normalized(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fsdecode(path))))


def contained(path, directory):
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:  # Different Windows drives are never nested.
        return False


class FilesystemGuard:
    """Deny legacy sources and writes outside an optional scratch directory."""

    def __init__(self, write_root=None):
        self.archive = normalized(ROOT / "archive")
        self.migration = normalized(MIGRATION_ARCHIVE)
        self.local_mapping = normalized(
            ROOT / "reproducibility/asset_locations.local.json"
        )
        self.write_root = normalized(write_root) if write_root else None
        self.opens = 0
        self.enumerations = 0
        self.writes = 0

    def source_check(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        path = normalized(value)
        if (
            contained(path, self.archive)
            or path in (self.migration, self.local_mapping)
            # Also catch the same migration archive copied to a new location.
            or os.path.basename(path).lower() == "experiment1.tar.gz"
        ):
            raise SafetyViolation("Forbidden historical/local source access: " + path)
        return path

    def write_check(self, value):
        path = self.source_check(value)
        if path is None:
            raise SafetyViolation("Unresolved filesystem write target")
        if self.write_root is None or not contained(path, self.write_root):
            raise SafetyViolation("Write outside temporary package copies: " + path)
        self.writes += 1

    def __call__(self, event, args):
        if event == "open":
            self.source_check(args[0])
            mode, flags = args[1:3]
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing = writing or (
                isinstance(flags, int)
                and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            )
            if writing:
                self.write_check(args[0])
            self.opens += 1
        elif event in ("os.listdir", "os.scandir"):
            self.source_check(args[0])
            self.enumerations += 1
        elif event in (
            "os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown",
            "os.utime", "os.truncate", "os.setxattr", "os.removexattr",
        ):
            self.write_check(args[0])
        elif event in ("os.rename", "os.replace"):
            self.write_check(args[0])
            self.write_check(args[1])
        elif event in ("os.link", "os.symlink"):
            raise SafetyViolation("Links are unnecessary for daily reproduction tests")
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "pty.spawn"):
            raise SafetyViolation("Child commands/training are forbidden in audited tests")


def prove_guard(guard):
    """Verify that negative probes are rejected before even nonexistent paths open."""
    probes = [
        lambda: (ROOT / "archive/__daily_guard_probe__").open("rb"),
        lambda: os.scandir(ROOT / "archive"),
        lambda: Path(MIGRATION_ARCHIVE).open("rb"),
        lambda: (ROOT / "reproducibility/asset_locations.local.json").open("rb"),
        lambda: (ROOT / "__daily_guard_must_not_write__").open("wb"),
    ]
    for probe in probes:
        try:
            result = probe()
        except SafetyViolation:
            continue
        if hasattr(result, "close"):
            result.close()
        raise AssertionError("Audit guard did not reject a forbidden probe")
    return len(probes)


def load_daily_modules():
    sys.path.insert(0, str(ROOT / "scripts" / "repro"))
    import project_repro
    import project_replay
    import stage05_repro
    return project_repro, project_replay, stage05_repro


def readonly_checks():
    guard = FilesystemGuard()
    sys.addaudithook(guard)
    probes = prove_guard(guard)
    project, replay, stage05 = load_daily_modules()
    report = {"guard_negative_probes": probes, "training_started": False}
    with contextlib.redirect_stdout(io.StringIO()):
        for stage in project.STAGES:
            project.verify(stage)
            project.analyze(stage, write=False)
        items = project.specs()
        if len(items) != 17:
            raise AssertionError("Expected the 17 sealed stage 01-04 recipe references")
        for item in items:
            # Exercise the actual recipe-plan CLI, not only recipe parsing.
            argv = [
                str(ROOT / "scripts/repro/project_repro.py"), "plan",
                "--stage", item["stage"], "--experiment", item["id"],
            ]
            with patch.object(sys, "argv", argv):
                project.main()
            replay.run_replay(SimpleNamespace(
                stage=item["stage"], experiment=item["id"], mode="smoke",
                execute=False, project_root=str(ROOT),
            ))
        stage05.verify_package(stage05.PACKAGE)
        stage05_plans = []
        for key, *_ in stage05.SPECS:
            stage05.analyze_run(stage05.PACKAGE, key)
            try:
                stage05.run_replay(SimpleNamespace(
                    experiment=key, mode="smoke", execute=False,
                    project_root=str(ROOT),
                ))
            except ValueError as error:
                lines = str(error).splitlines()
                # A missing local input is expected on a report-only checkout.
                # Never hide hash mismatches or other unexpected failures.
                if not lines or not all(line.startswith("Missing input:") for line in lines):
                    raise
                stage05_plans.append({
                    "id": key, "status": "missing_local_inputs",
                    "missing_count": len(lines),
                })
            else:
                stage05_plans.append({
                    "id": key, "status": "plan_only_inputs_checked",
                })
    report.update(
        verified_stages=list(project.STAGES) + ["05"],
        project_readonly_analyses=len(project.STAGES),
        project_recipe_plans=len(items),
        project_replay_dry_plans=len(items),
        stage05_analyses=len(stage05.SPECS),
        stage05_dry_plans=stage05_plans,
        filesystem_opens=guard.opens,
        filesystem_enumerations=guard.enumerations,
        filesystem_writes=guard.writes,
    )
    return report


def csv_records(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline="")))


def compare_csv(expected, actual, label):
    if csv_records(actual) != csv_records(expected):
        raise AssertionError("Recomputed CSV fields/values changed: " + label)
    if actual != expected:
        raise AssertionError("Recomputed CSV bytes changed: " + label)


def snapshot_files(package, names):
    return {name: (package / name).read_bytes() for name in names}


def recompute_twice(source, target, analyze, names):
    expected = snapshot_files(source, names)
    shutil.copytree(source, target)
    for name in names:
        (target / name).unlink()
    analyze()
    first = snapshot_files(target, names)
    for name in names:
        compare_csv(expected[name], first[name], str(source / name))
    analyze()
    second = snapshot_files(target, names)
    if first != second:
        raise AssertionError("Repeated analysis changed generated CSV bytes: " + str(target))
    if snapshot_files(source, names) != expected:
        raise AssertionError("Real package changed during temporary recomputation")
    return len(names)


def temporary_analysis_checks(write_root):
    scratch = Path(write_root).resolve()
    if not scratch.is_dir() or contained(normalized(scratch), normalized(ROOT)):
        raise AssertionError("Analysis scratch must be an existing directory outside the project")
    guard = FilesystemGuard(scratch)
    sys.addaudithook(guard)
    probes = prove_guard(guard)
    project, _, stage05 = load_daily_modules()
    checked_csv = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for stage in project.STAGES:
            source = project.package(stage)
            target = scratch / source.name
            # Only package lookup changes; analysis still uses its real recipes.
            with patch.object(project, "package", return_value=target):
                checked_csv += recompute_twice(
                    source, target, lambda: project.analyze(stage),
                    ("results.csv", "summary.csv", "runs.csv"),
                )
        source = stage05.PACKAGE
        target = scratch / source.name
        original_svg = (source / "figures/validation_decoder_epochs.svg").read_bytes()
        checked_csv += recompute_twice(
            source, target, lambda: stage05.analyze(target),
            ("results.csv", "runs.csv"),
        )
        first_svg = (target / "figures/validation_decoder_epochs.svg").read_bytes()
        stage05.analyze(target)
        second_svg = (target / "figures/validation_decoder_epochs.svg").read_bytes()
        if first_svg != second_svg:
            raise AssertionError("Repeated stage 05 analysis changed SVG bytes")
        # Path.write_text uses native newlines on Python 3.10; the SVG's text
        # must match across Windows/Linux, while repeated local bytes must match.
        if first_svg.replace(b"\r\n", b"\n") != original_svg.replace(b"\r\n", b"\n"):
            raise AssertionError("Recomputed stage 05 SVG content changed")
        if (source / "figures/validation_decoder_epochs.svg").read_bytes() != original_svg:
            raise AssertionError("Real stage 05 SVG changed")
    return {
        "guard_negative_probes": probes,
        "temporary_packages_recomputed": len(project.STAGES) + 1,
        "csv_value_and_byte_comparisons": checked_csv,
        "repeat_csv_byte_stability": True,
        "stage05_svg_content_and_repeat_byte_stability": True,
        "filesystem_opens": guard.opens,
        "filesystem_enumerations": guard.enumerations,
        "temporary_filesystem_writes": guard.writes,
        "training_started": False,
    }


class DailyReproductionTests(unittest.TestCase):
    def run_isolated(self, mode):
        with tempfile.TemporaryDirectory(prefix="daily-repro-audit-") as temporary:
            command = [sys.executable, "-B", str(SCRIPT), "--child", mode]
            if mode == "temporary-analysis":
                command += ["--write-root", temporary]
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                command, cwd=temporary, env=environment, capture_output=True,
                text=True, encoding="utf-8", timeout=120,
            )
            self.assertEqual(
                result.returncode, 0,
                "Audited child failed:\n" + result.stdout + "\n" + result.stderr,
            )
            report = json.loads(result.stdout)
            print(mode + ": " + json.dumps(report, ensure_ascii=True), flush=True)
            return report

    def test_daily_readonly_operations_never_read_historical_sources(self):
        report = self.run_isolated("readonly")
        self.assertEqual(report["filesystem_writes"], 0)
        self.assertEqual(report["project_recipe_plans"], 17)
        self.assertEqual(report["project_replay_dry_plans"], 17)
        self.assertEqual(report["stage05_analyses"], 6)
        self.assertEqual(len(report["stage05_dry_plans"]), 6)
        self.assertFalse(report["training_started"])

    def test_temporary_recomputation_matches_and_is_repeatable(self):
        report = self.run_isolated("temporary-analysis")
        self.assertEqual(report["temporary_packages_recomputed"], 5)
        self.assertEqual(report["csv_value_and_byte_comparisons"], 14)
        self.assertTrue(report["repeat_csv_byte_stability"])
        self.assertTrue(report["stage05_svg_content_and_repeat_byte_stability"])
        self.assertFalse(report["training_started"])


if __name__ == "__main__":
    if "--child" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--child", choices=("readonly", "temporary-analysis"), required=True)
        parser.add_argument("--write-root")
        args = parser.parse_args()
        if args.child == "temporary-analysis":
            if not args.write_root:
                parser.error("temporary-analysis requires --write-root")
            output = temporary_analysis_checks(args.write_root)
        else:
            output = readonly_checks()
        print(json.dumps(output, ensure_ascii=True))
    else:
        unittest.main()
