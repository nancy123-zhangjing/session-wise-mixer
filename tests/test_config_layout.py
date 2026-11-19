"""Stdlib-only B2b config relocation checks; never import or run training."""
from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import audit_config_layout as audit
import audit_script_layout as scripts_audit


class ConfigLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = audit.read(audit.AUDIT / "before.json")
        cls.records = {row["before"]: row for row in cls.before["files"]}
        cls.b2b_snapshot = scripts_audit.baseline()

    def test_all_34_configs_have_unique_destinations_and_unchanged_bytes(self):
        self.assertEqual(self.before["mapping"], audit.MOVES)
        self.assertEqual(len(audit.MOVES), 34)
        self.assertEqual(len(set(audit.MOVES.values())), 34)
        actual = {path.relative_to(ROOT).as_posix()
                  for path in (ROOT / "configs").rglob("*.yaml")}
        self.assertEqual(actual, set(audit.MOVES.values()))
        for old, new in audit.MOVES.items():
            with self.subTest(config=old):
                self.assertEqual(Path(old).name, Path(new).name)
                self.assertFalse((ROOT / old).exists())
                self.assertEqual(audit.digest(ROOT / new), self.records[old]["sha256"])

    def test_all_29_callers_reverse_to_baseline_and_keep_reference_sequence(self):
        self.assertEqual(len(self.before["callers"]), 29)
        self.assertEqual(set(self.before["callers"]), audit.CALLERS)
        for path, record in self.before["callers"].items():
            with self.subTest(caller=path):
                # Strictly undo B2c first; do not relax the historical B2b
                # parameter hash or config-layer sequence checks.
                text = scripts_audit.historical_text(path)
                audit.check_caller(path, text, record)
                self.assertEqual(audit.references(text), [])
                self.assertEqual(audit.references(text, relocated=True), record["config_sequence"])
                self.assertEqual(audit.normalized_sha(audit.relocate_text(text, reverse=True)),
                                 record["normalized_sha256"])

    def test_all_137_unique_caller_config_pairs_are_preserved(self):
        expected = {(path, config) for path, row in self.before["callers"].items()
                    for config in row["config_sequence"]}
        self.assertEqual(len(expected), 137)
        actual = {(path, config) for path in audit.CALLERS
                  for config in audit.references(scripts_audit.historical_text(path), relocated=True)}
        self.assertEqual(actual, expected)
        for _, old in actual:
            self.assertTrue((ROOT / audit.MOVES[old]).is_file())

    def test_only_reviewed_historical_annotations_keep_old_config_paths(self):
        # This report footer cites the historical research design; it is not
        # a runtime config read. Preserve its text; B2c permits only root depth.
        historical_annotations = {
            "scripts/build_stage1_pdf_report.py": ["configs/ar_formal_common.yaml"],
        }
        for path in (ROOT / "scripts").rglob("*"):
            if path.suffix not in {".py", ".sh", ".ps1"} or path.name.startswith("audit_"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            original = {new: old for old, new in scripts_audit.MOVES.items()}.get(relative, relative)
            with self.subTest(caller=relative):
                self.assertEqual(audit.references(path.read_text(encoding="utf-8-sig")),
                                 historical_annotations.get(original, []))
                if original in historical_annotations:
                    self.assertIn(self.records[original]["sha256"],
                                  scripts_audit.byte_hash_candidates(scripts_audit.historical_text(original)))

    def test_exact_slash_windows_and_pathlib_spellings_round_trip(self):
        for old, new in audit.MOVES.items():
            spellings = (
                (old, new),
                (old.replace("/", "\\"), new.replace("/", "\\")),
                ('"' + old.replace("/", '" / "') + '"',
                 '"' + new.replace("/", '" / "') + '"'),
                ("'" + old.replace("/", "' / '") + "'",
                 "'" + new.replace("/", "' / '") + "'"),
            )
            for original, relocated in spellings:
                with self.subTest(config=old, spelling=original):
                    self.assertEqual(audit.relocate_text(original), relocated)
                    self.assertEqual(audit.relocate_text(relocated, reverse=True), original)
                    self.assertEqual(audit.references(original), [old])
                    self.assertEqual(audit.references(relocated, relocated=True), [old])
                    self.assertEqual(audit.references(relocated), [])

    def caller_fixture(self):
        original = (
            "python main.py --model DIFF_GRM "
            "--config configs/diff_formal_common.yaml "
            "--config configs/diff_formal_repo_v6.yaml "
            "--config configs/diff_formal_crosspse_joint_diger.yaml "
            "--lr 0.001 --rand_seed 2024\n"
        )
        record = {"normalized_sha256": audit.normalized_sha(original),
                  "config_sequence": audit.references(original)}
        return audit.relocate_text(original), record

    def test_caller_guard_accepts_only_paths_and_platform_newlines(self):
        text, record = self.caller_fixture()
        for value in (text, text.replace("\n", "\r\n")):
            with self.subTest(crlf="\r\n" in value):
                audit.check_caller("fixture.sh", value, record)

    def test_caller_guard_rejects_parameter_and_learning_rate_changes(self):
        text, record = self.caller_fixture()
        changes = (("--lr 0.001", "--lr 0.01"),
                   ("--rand_seed 2024", "--rand_seed 2025"),
                   ("--model DIFF_GRM", "--model AR_GRM"))
        for original, replacement in changes:
            tampered = text.replace(original, replacement)
            with self.subTest(parameter=original):
                self.assertNotEqual(tampered, text)
                self.assertEqual(audit.references(tampered, relocated=True), record["config_sequence"])
                with self.assertRaisesRegex(ValueError, "beyond configuration paths"):
                    audit.check_caller("fixture.sh", tampered, record)

    def test_caller_guard_rejects_reordered_or_missing_config_layers(self):
        text, record = self.caller_fixture()
        first, second = (audit.MOVES[name] for name in record["config_sequence"][:2])
        reordered = text.replace(first, "__FIRST_CONFIG__").replace(second, first).replace("__FIRST_CONFIG__", second)
        missing = text.replace("--config " + second + " ", "")
        for tampered in (reordered, missing):
            with self.subTest(kind="reordered" if tampered == reordered else "missing"):
                self.assertNotEqual(tampered, text)
                with self.assertRaises(ValueError):
                    audit.check_caller("fixture.sh", tampered, record)
                # Independently exercise sequence validation, even if a hash
                # record were accidentally refreshed to the tampered text.
                refreshed = {**record, "normalized_sha256": audit.normalized_sha(
                    audit.relocate_text(tampered, reverse=True))}
                with self.assertRaisesRegex(ValueError, "missing or reordered"):
                    audit.check_caller("fixture.sh", tampered, refreshed)

    def test_shared_configs_remain_single_files_used_across_stages(self):
        self.assertEqual(set(audit.CONFIG_GROUPS["shared"]), {
            "ar_formal_common.yaml", "diff_formal_common.yaml", "diff_formal_repo_v6.yaml",
            "diff_smoke.yaml", "diff_profile.yaml",
        })
        uses = {
            "ar_formal_common.yaml": ("run_formal_pair.sh", "run_ar_internal_study.sh"),
            "diff_formal_common.yaml": ("run_diff_formal.sh", "run_diff_pse_refinement.sh", "run_diff_crosspse_joint.sh"),
            "diff_formal_repo_v6.yaml": ("run_diff_formal.sh", "run_diff_pse_refinement.sh", "run_diff_crosspse_joint.sh"),
            "diff_smoke.yaml": ("run_diff_formal.sh", "run_diff_pse_refinement.sh"),
            "diff_profile.yaml": ("run_diff_formal.sh", "run_diff_pse_refinement.sh"),
        }
        for name, callers in uses.items():
            old = "configs/" + name
            with self.subTest(config=name):
                self.assertEqual(audit.MOVES[old], "configs/shared/" + name)
                self.assertEqual(list((ROOT / "configs").rglob(name)), [ROOT / audit.MOVES[old]])
                for caller in callers:
                    path = "scripts/" + caller
                    self.assertIn(old, self.before["callers"][path]["config_sequence"])
                    self.assertIn(old, audit.references(scripts_audit.historical_text(path), relocated=True))

    def test_b2b_snapshot_kept_five_seals_and_six_fixed_helpers_byte_identical(self):
        # This is the B2b invariant, proved against its saved post-migration
        # snapshot. B2c separately checks current helper paths/hashes and seals.
        seals = [name for name in self.records
                 if name.startswith("experiments/") and name.endswith("/package_integrity.json")]
        self.assertEqual(len(seals), 5)
        helpers = set()
        for name in seals:
            with self.subTest(seal=name):
                self.assertEqual(scripts_audit.snapshot_digest(ROOT / name), self.records[name]["sha256"])
                seal = self.b2b_snapshot["seals"][name]
                helpers.update(row["path"] for row in seal.get("shared", seal.get("helpers", []))
                               if row["path"].startswith(("scripts/", "tests/")))
        self.assertEqual(helpers, {
            "scripts/project_repro.py", "scripts/project_replay.py", "scripts/stage05_repro.py",
            "scripts/asset_locations.py", "tests/test_project_repro.py", "tests/test_stage05_repro.py",
        })
        for name in helpers:
            with self.subTest(helper=name):
                self.assertEqual(scripts_audit.snapshot_digest(ROOT / name), self.records[name]["sha256"])

    def test_historical_evidence_sources_receipts_and_audits_are_byte_identical(self):
        navigation = {"reproducibility/README.md", "experiments/05_crosspse_joint/reproducibility/README.md"}
        old_audits = {"scripts/audit_layout_migration.py", "scripts/audit_repro_workflow.py",
                      "scripts/audit_entrypoint_layout.py"}
        protected = [name for name in self.records if name in old_audits or
                     (name.startswith(("reproducibility/", "experiments/")) and name not in navigation)]
        self.assertTrue(protected)
        for name in protected:
            with self.subTest(path=name):
                if name in old_audits:
                    self.assertIn(self.records[name]["sha256"],
                                  scripts_audit.byte_hash_candidates(scripts_audit.historical_text(name)))
                elif name in self.b2b_snapshot["seals"] or name in scripts_audit.NAVIGATION:
                    # B2c navigation edits do not erase the old B2b byte claim.
                    self.assertEqual(scripts_audit.snapshot_digest(ROOT / name), self.records[name]["sha256"])
                else:
                    self.assertEqual(audit.digest(ROOT / name), self.records[name]["sha256"])

    def test_config_index_local_links_resolve(self):
        indexes = [ROOT / "configs/README.md"]
        indexes += sorted((ROOT / "configs").glob("*/README.md"))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
