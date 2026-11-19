"""CPU-only legacy-entry guards; no real import/build/seal is ever invoked."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))

import project_repro as project
import stage05_repro as stage05

SPEC = importlib.util.spec_from_file_location(
    "legacy_import_experiments", ROOT / "scripts/legacy/import_experiments.py"
)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


class LegacyImportTests(unittest.TestCase):
    def arguments(self, stages="05", execute=True):
        return argparse.Namespace(stages=stages, archive="unused.tar.gz", source="unused",
                                  execute=execute, expected_root_commit="a" * 40)

    def test_daily_builds_refuse_before_calling_old_builders(self):
        for module in (project, stage05):
            with self.subTest(module=module.__name__), patch.object(module, "_legacy_build") as builder:
                with self.assertRaisesRegex(ValueError, "build is retired"):
                    module.build(argparse.Namespace())
                builder.assert_not_called()

    def test_package_names_match_daily_entry_points(self):
        self.assertEqual(tuple(project.STAGES.values()), legacy.PACKAGES["01-04"])
        self.assertEqual(stage05.PACKAGE.name, legacy.PACKAGES["05"][0])

    def test_default_plan_never_reads_locks_head_or_invokes_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(legacy, "require_sources") as sources, \
                    patch.object(legacy, "require_expected_commit") as commit, \
                    patch.object(legacy, "load_builder") as builder, \
                    patch.object(Path, "read_text", side_effect=AssertionError("No lock reads")), \
                    patch.object(Path, "mkdir", side_effect=AssertionError("No writes")), \
                    redirect_stdout(io.StringIO()) as output:
                legacy.run(self.arguments(execute=False), root)
            sources.assert_not_called()
            commit.assert_not_called()
            builder.assert_not_called()
            self.assertIn('"mode": "plan_only"', output.getvalue())
            self.assertEqual(list(root.iterdir()), [])

    def test_readme_only_template_is_permitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "experiments" / legacy.PACKAGES["05"][0]
            package.mkdir(parents=True)
            (package / "README.md").write_text("template", encoding="utf-8")
            legacy.require_fresh_destination("05", root)

    def test_empty_or_partial_package_outputs_block_before_sources_head_and_builder(self):
        for name in ("reproducibility", "evidence", "derived", "figures", "results.csv", "runs.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                target = root / "experiments" / legacy.PACKAGES["05"][0] / name
                if name.endswith(".csv"):
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"preserved")
                else:
                    target.mkdir(parents=True)
                with patch.object(legacy, "require_sources") as sources, \
                        patch.object(legacy, "require_expected_commit") as commit, \
                        patch.object(legacy, "load_builder") as builder:
                    with self.assertRaisesRegex(ValueError, "existing or partial"):
                        legacy.run(self.arguments(), root)
                sources.assert_not_called()
                commit.assert_not_called()
                builder.assert_not_called()
                self.assertTrue(target.exists())
                if target.is_file():
                    self.assertEqual(target.read_bytes(), b"preserved")

    def test_shared_controls_block_both_import_modes_without_reading_them(self):
        for stages in legacy.PACKAGES:
            for name in legacy.SHARED_CONTROLS:
                with self.subTest(stages=stages, name=name), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    target = root / "reproducibility" / name
                    if name in ("code", "environment"):
                        target.mkdir(parents=True)
                    else:
                        target.parent.mkdir(parents=True)
                        target.write_bytes(b"not parsed")
                    with patch.object(legacy, "require_sources") as sources, \
                            patch.object(legacy, "require_expected_commit") as commit, \
                            patch.object(Path, "read_text", side_effect=AssertionError("No lock reads")):
                        with self.assertRaisesRegex(ValueError, "existing or partial"):
                            legacy.run(self.arguments(stages), root)
                    sources.assert_not_called()
                    commit.assert_not_called()

    def test_stage05_package_does_not_block_later_01_04_destination_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments/05_crosspse_joint/reproducibility").mkdir(parents=True)
            legacy.require_fresh_destination("01-04", root)

    def test_existing_registry_blocks_01_04_but_is_not_written_by_stage05(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "experiments/experiment_registry.json"
            registry.parent.mkdir()
            registry.write_bytes(b"preserved registry")
            with self.assertRaisesRegex(ValueError, "experiment_registry.json"):
                legacy.require_fresh_destination("01-04", root)
            legacy.require_fresh_destination("05", root)
            self.assertEqual(registry.read_bytes(), b"preserved registry")

    def test_file_at_experiments_parent_is_not_a_fresh_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments").write_bytes(b"preserved")
            with self.assertRaisesRegex(ValueError, "existing or partial"):
                legacy.require_fresh_destination("05", root)
            self.assertEqual((root / "experiments").read_bytes(), b"preserved")

    def test_01_04_missing_stage05_index_fails_before_head_or_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(legacy, "require_expected_commit") as commit, \
                    patch.object(legacy, "load_builder") as builder:
                with self.assertRaisesRegex(ValueError, "05_crosspse_joint/runs.csv"):
                    legacy.run(self.arguments("01-04"), root)
            commit.assert_not_called()
            builder.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_execution_requires_full_explicit_commit_before_git(self):
        for expected in (None, "", "HEAD", "a" * 12, "g" * 40):
            with self.subTest(expected=expected), patch.object(legacy.subprocess, "check_output") as git:
                with self.assertRaisesRegex(ValueError, "complete reviewed Git commit"):
                    legacy.require_expected_commit(expected)
                git.assert_not_called()

    def test_changed_head_is_rejected_without_loading_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(legacy, "require_sources"), \
                    patch.object(legacy.subprocess, "check_output", return_value="b" * 40 + "\n"), \
                    patch.object(legacy, "load_builder") as builder:
                with self.assertRaisesRegex(ValueError, "Root commit mismatch"):
                    legacy.run(self.arguments(), root)
                builder.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_reviewed_first_import_dispatches_only_to_mock_after_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments()
            calls = []
            with patch.object(legacy, "require_fresh_destination", side_effect=lambda *a: calls.append("destination")), \
                    patch.object(legacy, "require_sources", side_effect=lambda *a: calls.append("sources")), \
                    patch.object(legacy, "require_expected_commit", side_effect=lambda *a: calls.append("commit")), \
                    patch.object(legacy, "load_builder") as loader:
                loader.return_value.side_effect = lambda *a: calls.append("mock_builder")
                legacy.run(args, root)
                loader.assert_called_once_with("05")
                loader.return_value.assert_called_once_with(args)
            self.assertEqual(calls, ["destination", "sources", "commit", "mock_builder"])
            self.assertEqual(list(root.iterdir()), [])

    def test_stage05_source_normalization_matches_old_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            self.assertEqual(legacy.source_root(parent), parent / "experiment1")
            self.assertEqual(legacy.source_root(parent / "experiment1"), parent / "experiment1")

    def test_old_build_functions_remain_available_only_for_explicit_import(self):
        self.assertTrue(callable(project._legacy_build))
        self.assertTrue(callable(stage05._legacy_build))


if __name__ == "__main__":
    unittest.main(verbosity=2)
