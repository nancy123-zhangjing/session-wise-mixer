"""Stdlib checks for the B2a tests/docs relocation; no model imports."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import audit_entrypoint_layout as audit
from audit_config_layout import relocate_text
import audit_script_layout as scripts_audit


class EntrypointLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = audit.read(audit.AUDIT / "before.json")
        cls.records = {r["before"]: r for r in cls.before["files"]}
        cls.b2b_snapshot = scripts_audit.baseline()

    def test_all_31_moves_have_one_destination_and_no_old_entry(self):
        self.assertEqual(len(audit.MOVES), 31)
        self.assertEqual(len(set(audit.MOVES.values())), 31)
        for old, new in audit.MOVES.items():
            with self.subTest(path=old):
                self.assertFalse((ROOT / old).exists())
                self.assertTrue((ROOT / new).is_file())

    def test_17_method_and_history_originals_are_byte_identical(self):
        names = audit.DOC_GROUPS["methods"] + audit.DOC_GROUPS["history"]
        self.assertEqual(len(names), 17)
        for name in names:
            old = "docs/" + name
            with self.subTest(path=old):
                self.assertEqual(audit.digest(ROOT / audit.MOVES[old]), self.records[old]["sha256"])

    def test_model_tests_and_daily_guard_were_only_moved(self):
        editable = {Path(p).name for p in audit.EDITABLE if p.startswith("tests/")}
        unchanged = [name for name in audit.TEST_NAMES if name not in editable]
        self.assertEqual(len(unchanged), 7)
        for name in unchanged:
            with self.subTest(path=name):
                if name == "test_daily_repro.py":
                    # Undo only the separately hash-verified B2c bootstrap.
                    restored = scripts_audit.restore_test_text(
                        "tests/" + name, (ROOT / "tests" / name).read_text(encoding="utf-8"))
                    self.assertIn(self.records["scripts/" + name]["sha256"],
                                  scripts_audit.byte_hash_candidates(restored))
                else:
                    self.assertEqual(audit.digest(ROOT / "tests" / name),
                                     self.records["scripts/" + name]["sha256"])

    def test_training_callers_changed_paths_only(self):
        for name in audit.CALLER_NAMES:
            path = "scripts/" + name
            with self.subTest(path=path):
                # First undo the separately hash-verified B2c script relocation,
                # then the separately verified B2b config relocation;
                # the original B2a reversal and final byte-hash check stay strict.
                reverted = relocate_text(scripts_audit.historical_text(path), reverse=True).encode("utf-8")
                reverted = reverted.replace(b"tests/test_", b"scripts/test_")
                reverted = reverted.replace(b'/ "tests" / "test_', b'/ "scripts" / "test_')
                reverted = reverted.replace(b'/ "docs" / "methods" /', b'/ "docs" /')
                # Checkout line endings are not a training algorithm change.
                lf = reverted.replace(b"\r\n", b"\n")
                candidates = {reverted, lf, lf.replace(b"\n", b"\r\n")}
                self.assertIn(self.records[path]["sha256"],
                              {hashlib.sha256(value).hexdigest() for value in candidates})

    def test_all_five_seals_allow_only_reviewed_helper_rebinding(self):
        self.assertEqual(len(self.before["seals"]), 5)
        for path, old in self.before["seals"].items():
            with self.subTest(path=path):
                # Exercise the unchanged historical B2a rule against the saved
                # B2b state. B2c current seals have their own strict tests.
                with patch.object(audit, "digest", scripts_audit.snapshot_digest):
                    audit.check_seal(old, self.b2b_snapshot["seals"][path])

    def test_seal_guard_rejects_science_changes_and_new_helpers(self):
        path, old = next(iter(self.before["seals"].items()))
        current = self.b2b_snapshot["seals"][path]
        wrong = copy.deepcopy(current)
        wrong["files"][0]["sha256"] = "not the original scientific record"
        with patch.object(audit, "digest", scripts_audit.snapshot_digest), self.assertRaises(ValueError):
            audit.check_seal(old, wrong)
        wrong = copy.deepcopy(current)
        wrong["shared"].append({"path": "unreviewed.py", "sha256": "unreviewed"})
        with patch.object(audit, "digest", scripts_audit.snapshot_digest), self.assertRaises(ValueError):
            audit.check_seal(old, wrong)

    def test_runner_lists_only_six_stdlib_suites_from_another_cwd(self):
        expected = {"test_asset_locations.py", "test_daily_repro.py", "test_legacy_import.py",
                    "test_project_layout.py", "test_project_repro.py", "test_stage05_repro.py"}
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tests/run_repro_checks.py"), "--list"],
                cwd=temporary, text=True, capture_output=True, check=True,
            )
        self.assertEqual(set(result.stdout.splitlines()), expected)

    def test_live_scripts_have_no_old_test_paths(self):
        for path in (ROOT / "scripts").rglob("*"):
            if path.suffix not in {".py", ".sh", ".ps1"} or path.name.startswith("audit_"):
                continue
            text = path.read_text(encoding="utf-8-sig")
            with self.subTest(path=path.name):
                self.assertIsNone(re.search(r'scripts/test_\w+\.py', text))
                self.assertIsNone(re.search(r'"scripts"\s*/\s*"test_\w+\.py"', text))

    def test_python_entrypoints_are_python310_syntax_compatible(self):
        for directory in ("scripts", "tests"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
