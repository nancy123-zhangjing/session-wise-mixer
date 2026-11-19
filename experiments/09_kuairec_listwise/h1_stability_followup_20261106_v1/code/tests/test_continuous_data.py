"""Standard-library CPU contract tests; no model/GPU dependencies or skips."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prep = module("continuous_prepare_tests", ROOT / "scripts/prepare/prepare_continuous_data.py")
check = module("continuous_verify_tests", ROOT / "scripts/checks/verify_continuous_data.py")
LEGACY_SOURCE = ROOT / "external/DiffGRM/genrec/dataset.py"


class ContinuousDataTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "tmp/continuous_data_tests"
        base.mkdir(parents=True, exist_ok=True)
        if not base.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("Test temporary root escaped the workspace")
        self.temp = tempfile.TemporaryDirectory(prefix="case_", dir=base)
        self.root = Path(self.temp.name).resolve()
        if not self.root.is_relative_to(base.resolve()):
            raise RuntimeError("Test cleanup target escaped its verified temporary root")
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / "source"
        self.source.mkdir()
        # Deliberately different insertion order, nonnumeric original keys,
        # repeated items, a long sequence, and a minimum-length boundary user.
        self.sequences = {
            "user_b": ["item_1", "item_1", "item_3", "item_1", "item_4", "item_1"],
            "user_a": [f"item_{i}" for i in range(1, 71)],
            "user_c": ["item_3", "item_4", "item_3"],
        }
        self.mapping = {
            "user2id": {"[PAD]": 0, "user_a": 1, "user_b": 2, "user_c": 3},
            "id2user": ["[PAD]", "user_a", "user_b", "user_c"],
            "item2id": {"[PAD]": 0, **{f"item_{i}": i for i in range(1, 71)}},
            "id2item": ["[PAD]"] + [f"item_{i}" for i in range(1, 71)],
        }
        self.raw = {"all_item_seqs.json": prep.json_bytes(self.sequences),
                    "id_mapping.json": prep.json_bytes(self.mapping)}
        self.config = deepcopy(prep.SEMANTICS)
        self.config.update({"asset_lock": "assets.lock.json", "expected": {
            "users": 3, "items": 70, "interactions": 79,
            "train_targets": {prep.PROTOCOLS[0]: 70, prep.PROTOCOLS[1]: 51},
            "validation": 3, "test": 3}})
        self.assets = {name: {"path": prep.INPUT_PREFIX + name,
                              "archive_member": "experiment1/" + prep.INPUT_PREFIX + name,
                              "sha256": prep.sha256_bytes(raw), "size_bytes": len(raw)}
                       for name, raw in self.raw.items()}
        for name, raw in self.raw.items():
            (self.source / name).write_bytes(raw)
        (self.root / "assets.lock.json").write_bytes(prep.json_bytes({
            "artifacts": {prep.INPUT_PREFIX + name: entry for name, entry in self.assets.items()}}))
        self.config_path = self.root / "config.json"
        self.config_path.write_bytes(prep.json_bytes(self.config))

    def build(self, name="run_a", **kwargs):
        output = self.root / "outputs/stage06_continuous" / name
        source = kwargs or {"input_dir": self.source}
        return prep.prepare(root=self.root, config_path=self.config_path, output=output, **source)

    def verify(self, output):
        return check.verify(output, root=self.root, config_path=self.config_path, legacy_source=LEGACY_SOURCE)

    def read(self, output, protocol, split):
        return [json.loads(line) for line in (Path(output) / protocol / f"{split}.jsonl").read_text().splitlines()]

    def resign(self, output, relative):
        """An edited file with a valid self-reported hash must still fail the oracle."""
        output = Path(output)
        manifest = json.loads((output / "manifest.json").read_bytes())
        entry = manifest["files"][relative]
        entry.update(sha256=prep.sha256_file(output / relative), size_bytes=(output / relative).stat().st_size)
        (output / "manifest.json").write_bytes(prep.json_bytes(manifest))
        marker = json.loads((output / "PREPARED.json").read_bytes())
        marker["manifest_sha256"] = prep.sha256_file(output / "manifest.json")
        (output / "PREPARED.json").write_bytes(prep.json_bytes(marker))

    def test_minimum_five_item_example_has_exact_targets(self):
        seq = [(7, (11, 12, 13, 14, 15))]
        rows = list(prep.iter_records(seq, prep.PROTOCOLS[0], "train"))
        self.assertEqual(rows, [
            {"user_id": 7, "target_position": 1, "history_start": 0, "history": [11], "target": 12},
            {"user_id": 7, "target_position": 2, "history_start": 0, "history": [11, 12], "target": 13},
        ])
        self.assertEqual(list(prep.iter_records(seq, prep.PROTOCOLS[1], "train")), rows[1:])

    def test_holdout_positions_and_test_history_are_exact(self):
        seq = [(7, (11, 12, 13, 14, 15))]
        for protocol in prep.PROTOCOLS:
            val = list(prep.iter_records(seq, protocol, "validation"))[0]
            test = list(prep.iter_records(seq, protocol, "test"))[0]
            self.assertEqual((val["history"], val["target_position"], val["target"]), ([11, 12, 13], 3, 14))
            self.assertEqual((test["history"], test["target_position"], test["target"]), ([11, 12, 13, 14], 4, 15))

    def test_three_item_boundary_has_no_train_and_nonempty_holdouts(self):
        seq = [(1, (2, 3, 4))]
        for protocol in prep.PROTOCOLS:
            self.assertEqual(list(prep.iter_records(seq, protocol, "train")), [])
            self.assertEqual(list(prep.iter_records(seq, protocol, "validation"))[0]["history"], [2])

    def test_long_sequence_retains_late_targets_with_recent_fifty_context(self):
        rows = list(prep.iter_records([(1, tuple(range(1, 71)))], prep.PROTOCOLS[0], "train"))
        self.assertEqual(len(rows), 67)
        self.assertEqual(rows[-1], {"user_id": 1, "target_position": 67, "history_start": 17,
                                   "history": list(range(18, 68)), "target": 68})

    def test_legacy_fifty_boundary_does_not_turn_into_rolling_window(self):
        for length in (53, 54, 70):
            rows = list(prep.iter_records([(1, tuple(range(1, length + 1)))], prep.PROTOCOLS[1], "train"))
            self.assertEqual(len(rows), 49)
            self.assertEqual((rows[-1]["target_position"], rows[-1]["history_start"], rows[-1]["target"]), (50, 0, 51))

    def test_legacy_matches_actual_project_method(self):
        output = self.build()["output"]
        reference = check.legacy_reference(self.sequences, LEGACY_SOURCE)
        for split in prep.SPLITS:
            expected = list(check.legacy_records(reference, self.mapping, split))
            self.assertEqual(self.read(output, prep.PROTOCOLS[1], split), expected)

    def test_repeated_heldout_item_value_is_legal(self):
        rows = list(prep.iter_records([(1, (5, 5, 5, 5, 5))], prep.PROTOCOLS[0], "train"))
        self.assertEqual([row["target_position"] for row in rows], [1, 2])
        self.assertEqual(rows[-1]["history"], [5, 5])
        self.assertEqual(self.verify(self.build()["output"])["status"], "PASS")

    def test_original_mapping_and_ids_are_preserved_byte_for_byte(self):
        output = Path(self.build()["output"])
        self.assertEqual((output / "inputs/id_mapping.json").read_bytes(), self.raw["id_mapping.json"])
        sequences, mapping = prep.normalize_inputs(self.raw)
        self.assertEqual([user for user, _ in sequences], [1, 2, 3])
        self.assertEqual(mapping, self.mapping)

    def test_pad_is_excluded_from_catalog_and_all_real_records(self):
        output = Path(self.build()["output"])
        self.assertEqual(json.loads((output / "catalog_item_ids.json").read_bytes()), list(range(1, 71)))
        for protocol in prep.PROTOCOLS:
            for split in prep.SPLITS:
                for row in self.read(output, protocol, split):
                    self.assertNotEqual(row["target"], 0)
                    self.assertNotIn(0, row["history"])
                    self.assertTrue(1 <= len(row["history"]) <= 50)

    def test_mapping_reverse_mismatch_rejected(self):
        self.mapping["id2item"][1] = "wrong"
        with self.assertRaises(ValueError):
            prep.validate_mapping(self.mapping)

    def test_boolean_mapping_id_rejected(self):
        self.mapping["item2id"]["item_1"] = True
        with self.assertRaises(ValueError):
            prep.validate_mapping(self.mapping)

    def test_nonzero_pad_rejected(self):
        self.mapping["item2id"]["[PAD]"] = 1
        with self.assertRaises(ValueError):
            prep.validate_mapping(self.mapping)

    def test_noncontiguous_and_duplicate_ids_rejected(self):
        for replacement in (-1, 71, 2):
            changed = deepcopy(self.mapping)
            changed["item2id"]["item_1"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                prep.validate_mapping(changed)

    def test_unknown_item_rejected(self):
        self.sequences["user_b"][0] = "missing"
        changed = dict(self.raw, **{"all_item_seqs.json": prep.json_bytes(self.sequences)})
        with self.assertRaises(ValueError):
            prep.normalize_inputs(changed)

    def test_pad_in_raw_interactions_rejected(self):
        self.sequences["user_b"][0] = "[PAD]"
        changed = dict(self.raw, **{"all_item_seqs.json": prep.json_bytes(self.sequences)})
        with self.assertRaises(ValueError):
            prep.normalize_inputs(changed)

    def test_user_set_change_rejected(self):
        del self.sequences["user_c"]
        changed = dict(self.raw, **{"all_item_seqs.json": prep.json_bytes(self.sequences)})
        with self.assertRaises(ValueError):
            prep.normalize_inputs(changed)

    def test_too_short_sequence_rejected(self):
        self.sequences["user_c"] = ["item_1", "item_2"]
        changed = dict(self.raw, **{"all_item_seqs.json": prep.json_bytes(self.sequences)})
        with self.assertRaises(ValueError):
            prep.normalize_inputs(changed)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prep.parse_json(b'{"user":1,"user":2}')

    def test_versioned_protocol_settings_cannot_silently_change(self):
        for key, value in (("schema_version", 2), ("history", {"max_length": 100}),
                           ("supervision", "all_positions"), ("evaluation", {"filter_history": True})):
            changed = dict(self.config, **{key: value})
            self.config_path.write_bytes(prep.json_bytes(changed))
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.load_config(self.config_path)

    def test_expected_count_type_and_unknown_config_fields_rejected(self):
        for changed in (dict(self.config, surprise=True), deepcopy(self.config)):
            if "surprise" not in changed:
                changed["expected"]["users"] = True
            self.config_path.write_bytes(prep.json_bytes(changed))
            with self.assertRaises(ValueError):
                prep.load_config(self.config_path)

    def test_count_mismatch_fails_without_creating_output(self):
        self.config["expected"]["users"] = 4
        self.config_path.write_bytes(prep.json_bytes(self.config))
        with self.assertRaisesRegex(ValueError, "statistic mismatch"):
            self.build()
        self.assertFalse((self.root / "outputs/stage06_continuous/run_a").exists())

    def test_input_hash_mismatch_fails_without_creating_output(self):
        (self.source / "id_mapping.json").write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.build()
        self.assertFalse((self.root / "outputs/stage06_continuous/run_a").exists())

    def test_source_selection_requires_exactly_one_source(self):
        for kwargs in ({}, {"input_dir": self.source, "archive": self.root / "unused"}):
            with self.assertRaises(ValueError):
                prep.read_inputs(self.assets, **kwargs)

    def test_existing_output_is_not_overwritten(self):
        output = Path(self.build()["output"])
        before = prep.sha256_file(output / "manifest.json")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(prep.sha256_file(output / "manifest.json"), before)

    def test_output_cannot_escape_into_historical_data_or_parent(self):
        for path in (self.root / "data/old", self.root / "outputs/stage06_continuous",
                     self.root / "outputs/stage06_continuous/../../old"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prep.safe_new_output(self.root, path)

    def make_archive(self, selected_symlink=False):
        path = self.root / "migration.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            extra = tarfile.TarInfo("../../must_not_be_extracted.txt")
            extra.size = 1
            bundle.addfile(extra, io.BytesIO(b'x'))
            for i, name in enumerate(prep.INPUT_NAMES):
                entry = tarfile.TarInfo(self.assets[name]["archive_member"])
                if selected_symlink and i == 0:
                    entry.type, entry.linkname = tarfile.SYMTYPE, "/outside"
                    bundle.addfile(entry)
                else:
                    entry.size = len(self.raw[name])
                    bundle.addfile(entry, io.BytesIO(self.raw[name]))
        return path

    def test_archive_reads_only_two_allowlisted_regular_files(self):
        output = Path(self.build(archive=self.make_archive())["output"])
        self.assertEqual({path.name for path in (output / "inputs").iterdir()}, set(prep.INPUT_NAMES))
        self.assertFalse((self.root / "must_not_be_extracted.txt").exists())
        self.assertEqual(self.verify(output)["status"], "PASS")

    def test_selected_archive_link_is_rejected_without_extracting(self):
        with self.assertRaisesRegex(ValueError, "Invalid selected archive"):
            prep.read_inputs(self.assets, archive=self.make_archive(selected_symlink=True))

    def test_actual_counts_and_identical_holdouts_across_protocols(self):
        result = self.verify(self.build()["output"])
        self.assertEqual(result["counts"], {
            prep.PROTOCOLS[0]: {"train": 70, "validation": 3, "test": 3},
            prep.PROTOCOLS[1]: {"train": 51, "validation": 3, "test": 3}})
        for split in ("validation", "test"):
            self.assertEqual(result["content_sha256"][f"{prep.PROTOCOLS[0]}/{split}.jsonl"],
                             result["content_sha256"][f"{prep.PROTOCOLS[1]}/{split}.jsonl"])

    def test_independent_generations_have_identical_content_hashes(self):
        left, right = self.build("a")["output"], self.build("b")["output"]
        result = check.compare(left, right, root=self.root, config_path=self.config_path, legacy_source=LEGACY_SOURCE)
        self.assertTrue(result["all_content_hashes_equal"])
        self.assertEqual(result["independent_outputs"], 2)

    def test_same_directory_is_not_independent_determinism_evidence(self):
        output = self.build()["output"]
        with self.assertRaisesRegex(ValueError, "different output"):
            check.compare(output, output)

    def test_resigned_wrong_target_fails_row_oracle(self):
        output = Path(self.build()["output"])
        relative = f"{prep.PROTOCOLS[0]}/train.jsonl"
        rows = self.read(output, prep.PROTOCOLS[0], "train")
        rows[0]["target"] = 0
        (output / relative).write_bytes(b''.join(prep.json_bytes(row) for row in rows))
        self.resign(output, relative)
        with self.assertRaisesRegex(ValueError, "original-position oracle"):
            self.verify(output)

    def test_resigned_wrong_context_and_reordered_rows_fail(self):
        output = Path(self.build()["output"])
        relative = f"{prep.PROTOCOLS[1]}/train.jsonl"
        rows = self.read(output, prep.PROTOCOLS[1], "train")
        rows[0], rows[1] = rows[1], rows[0]
        (output / relative).write_bytes(b''.join(prep.json_bytes(row) for row in rows))
        self.resign(output, relative)
        with self.assertRaisesRegex(ValueError, "original-position oracle"):
            self.verify(output)

    def test_resigned_history_with_future_interaction_fails(self):
        output = Path(self.build()["output"])
        relative = f"{prep.PROTOCOLS[0]}/train.jsonl"
        rows = self.read(output, prep.PROTOCOLS[0], "train")
        rows[0]["history"].append(rows[0]["target"])
        (output / relative).write_bytes(b''.join(prep.json_bytes(row) for row in rows))
        self.resign(output, relative)
        with self.assertRaisesRegex(ValueError, "original-position oracle"):
            self.verify(output)

    def test_resigned_omitted_legal_target_fails(self):
        output = Path(self.build()["output"])
        relative = f"{prep.PROTOCOLS[0]}/train.jsonl"
        rows = self.read(output, prep.PROTOCOLS[0], "train")[1:]
        (output / relative).write_bytes(b''.join(prep.json_bytes(row) for row in rows))
        self.resign(output, relative)
        with self.assertRaisesRegex(ValueError, "original-position oracle"):
            self.verify(output)

    def test_missing_preparation_marker_rejected(self):
        output = Path(self.build()["output"])
        (output / "PREPARED.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify(output)

    def test_unbound_extra_output_file_rejected(self):
        output = Path(self.build()["output"])
        (output / "extra.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "Unexpected or missing"):
            self.verify(output)

    def test_cli_runs_from_other_cwd_with_no_site_packages(self):
        output = self.root / "outputs/stage06_continuous/cli"
        command = [sys.executable, "-S", "-B", str(ROOT / "scripts/prepare/prepare_continuous_data.py"),
                   "--project-root", str(self.root), "--config", str(self.config_path),
                   "--input-dir", str(self.source), "--output", str(output)]
        result = subprocess.run(command, cwd=self.source, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["statistics"]["users"], 3)
        self.assertEqual(self.verify(output)["status"], "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContinuousDataTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "elapsed_seconds": time.perf_counter() - started,
              "python": platform.python_version(), "command": [sys.executable, "-B", *sys.argv],
              "source_sha256": {path.relative_to(ROOT).as_posix(): prep.sha256_file(path) for path in (
                  Path(__file__), ROOT / "scripts/prepare/prepare_continuous_data.py",
                  ROOT / "scripts/checks/verify_continuous_data.py", LEGACY_SOURCE)},
              "failure_details": [str(test) + "\n" + trace for test, trace in result.failures + result.errors]}
    if args.report:
        report_path = args.report.resolve()
        if not report_path.is_relative_to(ROOT):
            raise ValueError("Test report must stay within the project")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as handle:
            handle.write(prep.json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
