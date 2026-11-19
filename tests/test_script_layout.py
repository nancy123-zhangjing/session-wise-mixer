"""Stdlib-only B2c script-layout checks; no model imports, training or resealing."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import audit_script_layout as audit


class ScriptLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = audit.baseline()
        cls.records = {row["before"]: row for row in cls.before["files"]}

    def test_all_75_moves_have_one_destination_and_no_flat_copies(self):
        self.assertEqual(self.before["mapping"], audit.MOVES)
        self.assertEqual(len(audit.MOVES), 75)
        self.assertEqual(len(set(audit.MOVES.values())), 75)
        self.assertEqual(len(self.records), 732)
        self.assertTrue(self.before["baseline_commit"].startswith("3d41254"))
        for old, new in audit.MOVES.items():
            with self.subTest(script=old):
                self.assertEqual(Path(old).name, Path(new).name)
                self.assertFalse((ROOT / old).exists())
                self.assertTrue((ROOT / new).is_file())
        expected = set(audit.MOVES.values()) | {
            audit.LEGACY_IMPORT, "scripts/audit/audit_script_layout.py",
        }
        actual = {path for path in audit.file_inventory("scripts")
                  if Path(path).suffix in {".py", ".sh", ".ps1"}}
        self.assertEqual(actual, expected)

    def test_all_75_script_bodies_reverse_to_captured_hashes(self):
        for old, new in audit.MOVES.items():
            with self.subTest(script=old):
                current = (ROOT / new).read_text(encoding="utf-8")
                restored = audit.check_script(old, current, self.before)
                self.assertEqual(audit.normalized_sha(restored),
                                 self.before["script_normalized_sha256"][old])

    def test_four_historical_auditors_change_only_root_depth(self):
        for name in audit.GROUPS["audit"]:
            old = "scripts/" + name
            current = (ROOT / audit.MOVES[old]).read_text(encoding="utf-8")
            restored = current.replace("Path(__file__).resolve().parents[2]",
                                       "Path(__file__).resolve().parents[1]")
            with self.subTest(auditor=name):
                self.assertEqual(audit.normalized_sha(restored),
                                 self.before["script_normalized_sha256"][old])
        old = "scripts/audit_entrypoint_layout.py"
        current = (ROOT / audit.MOVES[old]).read_text(encoding="utf-8")
        wrong = current.replace("scripts/project_repro.py", "scripts/repro/project_repro.py")
        self.assertNotEqual(wrong, current)
        with self.assertRaisesRegex(ValueError, "beyond reviewed"):
            audit.check_script(old, wrong, self.before)

    def test_six_stdlib_adapters_and_legacy_loader_keep_original_logic(self):
        self.assertEqual(len(audit.STRICT_TESTS), 6)
        for path in audit.STRICT_TESTS:
            with self.subTest(test=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                restored = audit.restore_test_text(path, text, self.before)
                self.assertIn(self.records[path]["sha256"], audit.byte_hash_candidates(restored))
                with self.assertRaises(ValueError):
                    audit.restore_test_text(path, text + "\n# unreviewed test edit\n", self.before)
                if path == "tests/test_daily_repro.py":
                    with self.assertRaisesRegex(ValueError, "Daily plan CLI"):
                        audit.restore_test_text(path, text.replace("scripts/repro/project_repro.py",
                                                                   "scripts/project_repro.py"), self.before)
        current = (ROOT / audit.LEGACY_IMPORT).read_text(encoding="utf-8")
        audit.check_script(audit.LEGACY_IMPORT, current, self.before)
        with self.assertRaises(ValueError):
            audit.check_script(audit.LEGACY_IMPORT,
                               current.replace('ROOT / "scripts" / "repro"', 'ROOT / "scripts"'), self.before)

    def test_exact_slash_windows_and_pathlib_spellings_round_trip(self):
        for old, new in audit.MOVES.items():
            for original, relocated in audit.spellings(old, new):
                with self.subTest(script=old, spelling=original):
                    self.assertEqual(audit.relocate_text(original), relocated)
                    self.assertEqual(audit.relocate_text(relocated, reverse=True), original)

    def test_script_guard_rejects_parameter_order_import_and_body_changes(self):
        old = "scripts/preflight_formal.py"
        original = (
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'command = "python scripts/prepare_formal_embeddings.py --lr 0.001 --seed 2024"\n'
        )
        before = {"script_normalized_sha256": {old: audit.normalized_sha(original)}}
        current = audit.relocate_text(original).replace("parents[1]", "parents[2]")
        for text in (current, current.replace("\n", "\r\n")):
            audit.check_script(old, text, before)
        wrong = (
            current.replace("--lr 0.001", "--lr 0.01"),
            current.replace("--lr 0.001 --seed 2024", "--seed 2024 --lr 0.001"),
            current.replace("--seed 2024", "--seed 2025"),
            current + "import torch\n",
            current + "# undocumented body change\n",
        )
        for text in wrong:
            with self.subTest(tampered=text):
                with self.assertRaisesRegex(ValueError, "beyond reviewed"):
                    audit.check_script(old, text, before)

    def test_script_guard_rejects_missing_root_and_runtime_path_relocations(self):
        paths = ("scripts/project_repro.py", "scripts/run_formal_pair.sh", "scripts/run_ar_smoke_v2.ps1")
        for old in paths:
            text = (ROOT / audit.MOVES[old]).read_text(encoding="utf-8")
            wrong = text.replace("parents[2]", "parents[1]")
            wrong = wrong.replace('$(dirname "${BASH_SOURCE[0]}")/../..',
                                  '$(dirname "${BASH_SOURCE[0]}")/..')
            wrong = wrong.replace("Split-Path -Parent (Split-Path -Parent $PSScriptRoot)",
                                  "Split-Path -Parent $PSScriptRoot")
            with self.subTest(script=old):
                self.assertNotEqual(wrong, text)
                with self.assertRaisesRegex(ValueError, "root depth"):
                    audit.check_script(old, wrong, self.before)
        old = "scripts/preflight_diff_formal.py"
        text = (ROOT / audit.MOVES[old]).read_text(encoding="utf-8")
        wrong = text.replace('root / "scripts" / "train" / runner_name', 'root / "scripts" / runner_name')
        self.assertNotEqual(wrong, text)
        with self.assertRaisesRegex(ValueError, "Dynamic training runner"):
            audit.check_script(old, wrong, self.before)
        old = "scripts/run_formal_pair.sh"
        text = (ROOT / audit.MOVES[old]).read_text(encoding="utf-8")
        wrong = text.replace("scripts/prepare/prepare_formal_embeddings.py", "scripts/prepare_formal_embeddings.py")
        self.assertNotEqual(wrong, text)
        with self.assertRaisesRegex(ValueError, "flat entry"):
            audit.check_script(old, wrong, self.before)

    def test_five_current_seals_rebind_only_the_six_reviewed_helpers(self):
        self.assertEqual(len(self.before["seals"]), 5)
        self.assertEqual(len(audit.SEALED_HELPERS), 6)
        for path, old in self.before["seals"].items():
            with self.subTest(package=path):
                audit.check_seal(old, audit.read(ROOT / path))

    def test_seal_guard_rejects_science_new_bindings_and_wrong_helper_hash(self):
        path, old = next(iter(self.before["seals"].items()))
        current = audit.read(ROOT / path)
        wrong_science = copy.deepcopy(current)
        wrong_science["files"][0]["sha256"] = "changed scientific evidence"
        bindings = "shared" if "shared" in current else "helpers"
        new_binding = copy.deepcopy(current)
        new_binding[bindings].append({"path": "unreviewed.py", "sha256": "unreviewed"})
        wrong_helper = copy.deepcopy(current)
        helper = next(row for row in wrong_helper[bindings]
                      if row["path"].startswith("scripts/repro/"))
        helper["sha256"] = "incorrect helper hash"
        for wrong in (wrong_science, new_binding, wrong_helper):
            with self.assertRaisesRegex(ValueError, "beyond six reviewed"):
                audit.check_seal(old, wrong)

    def test_model_tests_runner_configs_science_and_old_receipts_stay_byte_identical(self):
        editable = set(audit.MOVES) | audit.STRICT_TESTS | audit.LAYOUT_TESTS | audit.NAVIGATION
        editable |= {audit.LEGACY_IMPORT} | set(self.before["seals"])
        protected = [row for row in self.before["files"] if row["before"] not in editable]
        self.assertGreater(len(protected), 600)
        for row in protected:
            with self.subTest(path=row["before"]):
                self.assertEqual(audit.digest(ROOT / row["after"]), row["sha256"])

    def test_stdlib_entrypoint_help_works_from_another_cwd(self):
        entrypoints = (
            "scripts/repro/project_repro.py", "scripts/repro/stage05_repro.py",
            "scripts/legacy/import_experiments.py", "scripts/audit/audit_script_layout.py",
            "tests/run_repro_checks.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for entry in entrypoints:
                with self.subTest(entrypoint=entry):
                    result = subprocess.run([sys.executable, "-B", str(ROOT / entry), "--help"],
                                            cwd=temporary, capture_output=True, text=True, check=False, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_runner_failure_and_missing_suite_return_nonzero_without_training(self):
        spec = importlib.util.spec_from_file_location("b2c_runner_test", ROOT / "tests/run_repro_checks.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for code in (7, -9):
            with self.subTest(exit_code=code), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=code)) as run:
                    self.assertEqual(runner.main([]), code if code > 0 else 1)
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(run.call_args.args[0][:2], [sys.executable, "-B"])
                    self.assertEqual(run.call_args.kwargs["cwd"], ROOT)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch.object(runner.Path, "is_file", return_value=False), patch.object(runner.subprocess, "run") as run:
                self.assertEqual(runner.main([]), 1)
                run.assert_not_called()

    def test_script_directory_index_links_resolve(self):
        indexes = [ROOT / "scripts/README.md"] + sorted((ROOT / "scripts").glob("*/README.md"))
        self.assertEqual(len(indexes), 10)
        checked = 0
        for index in indexes:
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
                target = target.strip().strip("<>")
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                target = unquote(target.split("#", 1)[0])
                with self.subTest(index=str(index.relative_to(ROOT)), target=target):
                    self.assertTrue((index.parent / target).exists())
                checked += 1
        self.assertGreater(checked, 0)

    def test_inventory_ignores_only_the_private_local_asset_pointer(self):
        private = ROOT / "reproducibility/asset_locations.local.json"
        protected = ROOT / "reproducibility/asset_manifest.json"
        with patch.object(audit.Path, "rglob", return_value=iter((private, protected))), \
                patch.object(audit.Path, "is_file", return_value=True), \
                patch.object(audit.Path, "read_text", side_effect=AssertionError("Do not read local pointers")):
            self.assertEqual(audit.file_inventory("reproducibility"), {"reproducibility/asset_manifest.json"})

    def test_complete_current_layout_audit(self):
        with redirect_stdout(io.StringIO()):
            result = audit.verify()
        self.assertEqual(result["checked_existing_files"], 732)
        self.assertEqual(result["script_moves"], 75)
        self.assertTrue(result["scientific_files_unchanged"])
        self.assertFalse(result["training_performed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
