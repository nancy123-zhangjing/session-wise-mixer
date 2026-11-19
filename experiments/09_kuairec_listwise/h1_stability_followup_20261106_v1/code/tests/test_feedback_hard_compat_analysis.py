"""Synthetic, standard-library tests for fixed-recommender SID compatibility.

These records test accounting and guards, not a measured model improvement.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/analysis/feedback_hard_compat_analysis.py"
spec = importlib.util.spec_from_file_location("_feedback_hard_compat_analysis_tests", PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load hard compatibility analysis: {PATH}")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class HardCompatibilityAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.initial = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
        self.catalogs = {arm: copy.deepcopy(self.initial) for arm in helper.ARMS}
        self.catalogs["cross_recon"][2] = [1, 0, 0, 1]
        self.catalogs["cross_rec"][3] = [1, 1, 0, 1]
        self.expected_counts = {"train": 4, "val": 3}

    def records(self):
        samples = [(3, [1, 0]), (1, [3, 0]), (2, [1, 0]), (4, [3, 0])]
        records = {}
        for split, count in self.expected_counts.items():
            records[split] = {}
            for arm in helper.ARMS:
                rows = []
                for index, (target, history) in enumerate(samples[:count]):
                    label = list(self.catalogs[arm][target - 1])
                    baseline_label = self.initial[target - 1]
                    changed_history = any(item and self.catalogs[arm][item - 1] != self.initial[item - 1]
                                          for item in history)
                    anchor_nll = [base + index / 100 + (0.1 if changed_history else 0.0)
                                  for base in (2.0, 2.5, 3.0, 3.5)]
                    anchor_loss = [value + 0.8 for value in anchor_nll]
                    own_nll = [value + (0.2 if arm == "cross_recon" else -0.2)
                               if label[position] != baseline_label[position] else value
                               for position, value in enumerate(anchor_nll)]
                    own_loss = [loss + 0.9 * (own - anchor)
                                for loss, own, anchor in zip(anchor_loss, own_nll, anchor_nll)]
                    predicted = list(baseline_label)
                    if changed_history and index == 1:
                        predicted[-1] = 1
                    correct = [int(prediction == truth) for prediction, truth in zip(predicted, label)]
                    rows.append(dict(sample_index=index, target_item_id=target,
                                     history_item_ids=list(history), target_sid=label, predicted_sid=predicted,
                                     position_nll=own_nll, position_loss=own_loss, position_correct=correct,
                                     initial_label_position_nll=anchor_nll,
                                     initial_label_position_loss=anchor_loss))
                records[split][arm] = rows
        return records

    def analyze(self, records=None, counts=None):
        return helper.analyze_records(self.records() if records is None else records, self.initial,
                                      self.catalogs, self.expected_counts if counts is None else counts,
                                      label_smoothing=0.1)

    def test_each_split_preserves_all_three_arms_and_expected_samples(self):
        result = self.analyze()
        self.assertEqual(set(result["splits"]), {"train", "val"})
        for split, count in self.expected_counts.items():
            self.assertEqual(result["splits"][split]["overall"]["count"], count)
            self.assertEqual(set(result["splits"][split]["overall"]["arms"]), set(helper.ARMS))

    def test_shared_union_groups_partition_train_and_validation(self):
        result = self.analyze()
        for split, counts in (("train", (2, 1, 1)), ("val", (1, 1, 1))):
            groups = result["splits"][split]["groups"]
            for name, count in zip(("target_changed", "history_only", "unchanged"), counts):
                self.assertEqual(groups[name]["count"], count)
                self.assertEqual(set(groups[name]["arms"]), set(helper.ARMS))
            self.assertEqual(sum(value["count"] for value in groups.values()), self.expected_counts[split])

    def test_weighted_subgroup_mean_nll_reproduces_overall(self):
        result = self.analyze()
        for split, count in self.expected_counts.items():
            section = result["splits"][split]
            for arm in helper.ARMS:
                regrouped = math.fsum(group["count"] * group["arms"][arm]["mean_nll"]
                                      for group in section["groups"].values() if group["count"]) / count
                self.assertAlmostEqual(regrouped, section["overall"]["arms"][arm]["mean_nll"], places=13)

    def test_negative_delta_mean_nll_means_left_arm_lower_loss(self):
        records = self.records()
        result = self.analyze(records)
        for split, count in self.expected_counts.items():
            left = [math.fsum(row["position_nll"]) / 4 for row in records[split]["cross_rec"]]
            right = [math.fsum(row["position_nll"]) / 4 for row in records[split]["cross_recon"]]
            expected = math.fsum(a - b for a, b in zip(left, right)) / count
            pair = result["splits"][split]["overall"]["pairs"]["cross_rec_minus_cross_recon"]
            self.assertAlmostEqual(pair["delta_mean_nll"], expected, places=13)
            self.assertLess(pair["delta_mean_nll"], 0)
            self.assertEqual(pair["lower_mean_nll_count"] + pair["higher_mean_nll_count"]
                             + pair["tied_mean_nll_count"], count)

    def test_own_minus_anchor_decomposition_matches_overall_delta(self):
        result = self.analyze()
        for split in self.expected_counts:
            for pair in result["splits"][split]["overall"]["pairs"].values():
                self.assertAlmostEqual(pair["delta_mean_nll"],
                                       pair["delta_mean_initial_label_nll"]
                                       + pair["delta_mean_own_minus_initial_nll"], places=13)

    def test_unchanged_samples_have_identical_model_outputs_and_scores(self):
        result = self.analyze()
        for split in self.expected_counts:
            check = result["splits"][split]["unchanged_consistency"]
            self.assertEqual(check["count"], 1)
            self.assertEqual(check["predicted_sid_mismatches"], 0)
            for name in ("max_position_nll_discrepancy", "max_position_loss_discrepancy",
                         "max_initial_label_position_nll_discrepancy", "max_initial_label_position_loss_discrepancy"):
                self.assertEqual(check[name], 0.0)

    def test_new_collision_uses_common_union_for_all_models(self):
        self.catalogs["cross_rec"][3] = list(self.catalogs["cross_rec"][2])
        result = self.analyze()
        for split, count in self.expected_counts.items():
            section = result["splits"][split]
            self.assertEqual(section["collision_strata"]["union"]["singleton"]["count"], 0)
            self.assertEqual(section["collision_strata"]["union"]["collided"]["count"], count)
            self.assertEqual(section["collision_strata"]["baseline"]["singleton"]["count"], 2 if split == "train" else 1)
            self.assertEqual(section["collision_strata"]["union"]["collided"]["arms"], section["overall"]["arms"])

    def test_no_input_records_or_catalog_are_mutated(self):
        records = self.records()
        before = copy.deepcopy((records, self.initial, self.catalogs))
        self.analyze(records)
        self.assertEqual((records, self.initial, self.catalogs), before)

    def test_missing_sample_or_wrong_count_rejected(self):
        records = self.records()
        records["val"]["cross_rec"].pop()
        with self.assertRaises(ValueError):
            self.analyze(records)
        with self.assertRaises(ValueError):
            self.analyze(counts={"train": 5, "val": 3})

    def test_duplicate_or_reordered_sample_indices_rejected(self):
        for operation in ("duplicate", "reordered"):
            records = self.records()
            rows = records["train"]["cross_rec"]
            if operation == "duplicate":
                rows[1] = copy.deepcopy(rows[0])
            else:
                rows[0], rows[1] = rows[1], rows[0]
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.analyze(records)

    def test_collided_sid_does_not_hide_different_target_identity(self):
        records = self.records()
        records["train"]["cross_rec"][1]["target_item_id"] = 2
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_ordered_history_identity_mismatch_rejected(self):
        records = self.records()
        records["train"]["cross_rec"][1]["history_item_ids"] = [4, 0]
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_stale_collate_target_sid_rejected(self):
        records = self.records()
        records["train"]["cross_recon"][0]["target_sid"] = list(self.initial[2])
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_four_position_vector_lengths_are_required(self):
        fields = ("position_nll", "position_loss", "position_correct", "initial_label_position_nll",
                  "initial_label_position_loss", "predicted_sid")
        for name in fields:
            records = self.records()
            records["train"]["cross_rec"][0][name].pop()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.analyze(records)

    def test_position_accuracy_must_match_predicted_and_target_sid(self):
        records = self.records()
        records["train"]["cross_rec"][0]["position_correct"][0] = 0
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_nonfinite_and_negative_scores_rejected(self):
        fields = ("position_nll", "position_loss", "initial_label_position_nll", "initial_label_position_loss")
        for name in fields:
            for value in (float("nan"), float("inf"), -0.1):
                records = self.records()
                records["train"]["cross_rec"][0][name][0] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.analyze(records)

    def test_sid_code_must_be_integer_and_inside_codebook(self):
        for name in ("target_sid", "predicted_sid"):
            for value in (-1, 256, 0.5, True):
                records = self.records()
                records["train"]["cross_rec"][0][name][0] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.analyze(records)

    def test_unchanged_target_position_own_and_anchor_scores_must_agree(self):
        # Other positions may change labels; this position did not.
        records = self.records()
        records["train"]["cross_recon"][0]["position_nll"][0] += 0.1
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_changed_label_position_requires_same_logit_smoothing_identity(self):
        records = self.records()
        # Label changed at the final position, so the unchanged-label equality
        # guard cannot catch this. Same-logit CE/NLL differences still must.
        records["train"]["cross_recon"][0]["position_loss"][-1] += 0.01
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_resolved_point_two_smoothing_uses_point_eight_same_logit_difference(self):
        records = self.records()
        for by_arm in records.values():
            for rows in by_arm.values():
                for row in rows:
                    row["position_loss"] = [loss + 0.8 * (own - anchor)
                                            for loss, own, anchor in zip(
                                                row["initial_label_position_loss"], row["position_nll"],
                                                row["initial_label_position_nll"])]
        result = helper.analyze_records(records, self.initial, self.catalogs,
                                        self.expected_counts, label_smoothing=0.2)
        self.assertEqual(result["label_smoothing"], 0.2)
        with self.assertRaises(ValueError):
            helper.analyze_records(records, self.initial, self.catalogs,
                                   self.expected_counts, label_smoothing=0.1)

    def test_unchanged_sample_cross_arm_score_discrepancy_rejected(self):
        records = self.records()
        row = records["train"]["cross_rec"][2]
        for name in ("position_nll", "position_loss", "initial_label_position_nll", "initial_label_position_loss"):
            row[name][0] += 0.1
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_changed_target_without_changed_history_cannot_change_anchor_scores(self):
        records = self.records()
        row = records["train"]["cross_recon"][0]
        for name in ("position_nll", "position_loss", "initial_label_position_nll", "initial_label_position_loss"):
            row[name][0] += 0.1
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_changed_target_without_changed_history_cannot_change_prediction(self):
        records = self.records()
        row = records["train"]["cross_recon"][0]
        row["predicted_sid"][0] = 0
        row["position_correct"][0] = 0
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_extra_split_or_arm_is_rejected(self):
        records = self.records()
        records["test"] = copy.deepcopy(records["val"])
        with self.assertRaises(ValueError):
            self.analyze(records)
        records = self.records()
        records["val"]["extra"] = copy.deepcopy(records["val"]["frozen"])
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_frozen_catalog_cannot_differ_from_initial(self):
        self.catalogs["frozen"][0] = [0, 0, 0, 1]
        with self.assertRaises(ValueError):
            self.analyze()

    def test_standard_library_helper_does_not_import_model_dependencies(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import numpy", source)

    def test_markdown_report_keeps_compatibility_and_supervision_limits(self):
        result = self.analyze()
        report = helper.markdown_report(result)
        self.assertIn("train", report)
        self.assertIn("val", report)
        self.assertIn("NLL", report)
        self.assertTrue("固定" in report or "兼容" in report)

    def write_bundle(self, root):
        root.mkdir()
        with (root / "catalog_ids.json").open("x", encoding="utf-8") as handle:
            json.dump(dict(initial=self.initial, arms=self.catalogs), handle)
        for arm in helper.ARMS:
            (root / arm).mkdir()
        for split, by_arm in self.records().items():
            for arm, rows in by_arm.items():
                with gzip.open(root / arm / f"{split}.records.jsonl.gz", "xt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

    def cli_args(self, root, destination):
        return ["--input-dir", str(root), "--output-dir", str(destination),
                "--expected-train", "4", "--expected-val", "3", "--label-smoothing", "0.1"]

    def test_cli_requires_explicit_label_smoothing_instead_of_guessing_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            self.write_bundle(root)
            destination = Path(directory) / "new_analysis"
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as error:
                helper.main(self.cli_args(root, destination)[:-2])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("--label-smoothing", stderr.getvalue())
            self.assertFalse(destination.exists())

    def test_cli_requires_new_output_and_preserves_old_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            self.write_bundle(root)
            destination = Path(directory) / "preserved_analysis"
            destination.mkdir()
            preserved = destination / "compatibility_summary.json"
            with preserved.open("x", encoding="utf-8") as handle:
                handle.write("preserved result")
            with self.assertRaises(FileExistsError):
                helper.main(self.cli_args(root, destination))
            self.assertEqual(preserved.read_text(encoding="utf-8"), "preserved result")
            self.assertFalse((destination / "README.md").exists())

    def test_cli_record_path_cannot_escape_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            root.mkdir()
            manifest = dict(initial=self.initial, arms=self.catalogs,
                            records={"train": {"frozen": "../outside.records.jsonl.gz"}})
            with (root / "catalog_ids.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            destination = Path(directory) / "new_analysis"
            with self.assertRaises(ValueError):
                helper.main(self.cli_args(root, destination))
            self.assertFalse(destination.exists())

    def test_cli_separate_analysis_keeps_all_raw_evidence_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            self.write_bundle(root)
            original = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            destination = Path(directory) / "new_analysis"
            with redirect_stdout(io.StringIO()) as stdout:
                helper.main(self.cli_args(root, destination))
            self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
            self.assertEqual(original, {path.relative_to(root): path.read_bytes()
                                        for path in root.rglob("*") if path.is_file()})
            self.assertTrue((destination / "README.md").is_file())
            result = json.loads((destination / "compatibility_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["expected_counts"], self.expected_counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
