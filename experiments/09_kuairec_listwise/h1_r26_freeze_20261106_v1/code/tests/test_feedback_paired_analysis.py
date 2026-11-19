"""CPU/standard-library tests for inference-only feedback sample pairing.

Synthetic records test accounting and safety; no GPU, training, or remote data.
Run: python -B tests/test_feedback_paired_analysis.py
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/analysis/feedback_paired_analysis.py"
spec = importlib.util.spec_from_file_location("_feedback_paired_analysis_tests", HELPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load paired analysis helper: {HELPER_PATH}")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def f32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.initial = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
        self.codes = {arm: copy.deepcopy(self.initial) for arm in helper.ARMS}
        self.codes["cross_recon"][2] = [1, 0, 0, 1]
        self.codes["cross_rec"][3] = [1, 1, 0, 1]

    def record(self, arm, index, target, history, rank):
        codes = self.codes[arm]
        label = list(codes[target - 1])
        filler = next(list(sid) for sid in codes if list(sid) != label)
        predictions = [list(filler) for _ in range(10)]
        if rank:
            predictions[rank - 1] = list(label)
        metrics = {}
        for k in (5, 10):
            hit = float(bool(rank and rank <= k))
            metrics[f"hit_rate@{k}"] = hit
            metrics[f"recall@{k}"] = hit
            metrics[f"ndcg@{k}"] = f32(1 / math.log2(rank + 1)) if hit else 0.0
        metrics["weighted_score"] = f32(
            f32(0.8 * metrics["ndcg@10"]) + f32(0.2 * metrics["hit_rate@10"])
        )
        metrics["legal@10"] = 1.0
        metrics["duplicate@10"] = f32(1 - len({tuple(sid) for sid in predictions}) / 10)
        return dict(sample_index=index, target_item_id=target, history_item_ids=list(history),
                    target_sid=label, top10_sids=predictions, first_hit_rank=rank, metrics=metrics)

    def records(self):
        samples = [(3, [1, 0]), (1, [3, 0]), (2, [1, 0]), (4, [3, 0])]
        ranks = {"frozen": [1, 0, 6, 10], "cross_recon": [2, 0, 6, 10],
                 "cross_rec": [1, 3, 0, 10]}
        return {arm: [self.record(arm, index, target, history, ranks[arm][index])
                      for index, (target, history) in enumerate(samples)]
                for arm in helper.ARMS}

    def analyze(self, records=None, expected_count=4):
        return helper.analyze_records(self.records() if records is None else records,
                                      self.initial, self.codes, expected_count)

    def test_change_union_uses_one_based_catalog_ids_and_any_digit(self):
        self.assertEqual(helper.changed_items(self.initial, self.codes), {3, 4})

    def test_identical_codes_have_empty_change_set(self):
        codes = {arm: copy.deepcopy(self.initial) for arm in helper.ARMS}
        self.assertEqual(helper.changed_items(self.initial, codes), set())

    def test_catalog_collisions_count_all_group_members(self):
        self.assertEqual(helper.collision_sizes(self.initial), {1: 2, 2: 2, 3: 1, 4: 1})
        split = copy.deepcopy(self.initial)
        split[1] = [0, 0, 0, 1]
        self.assertEqual(helper.collision_sizes(split), {1: 1, 2: 1, 3: 1, 4: 1})

    def test_sample_identity_removes_padding_but_preserves_order_and_repeats(self):
        expected = dict(sample_index=7, target_item_id=3, history_item_ids=[2, 1, 2])
        self.assertEqual(helper.sample_identity(7, 3, [2, 1, 2, 0, 0]), expected)

    def test_target_changed_takes_precedence_over_changed_history(self):
        self.assertEqual(helper.classify_sample(3, [4, 0], {3, 4}), "target_changed")

    def test_only_history_changed_group(self):
        self.assertEqual(helper.classify_sample(1, [3, 0], {3, 4}), "history_only")

    def test_unchanged_group_ignores_pad_zero(self):
        self.assertEqual(helper.classify_sample(1, [2, 0], {3, 4}), "unchanged")

    def test_groups_use_the_same_union_across_all_arms(self):
        result = self.analyze()
        self.assertEqual(result["groups"]["target_changed"]["count"], 2)
        self.assertEqual(result["groups"]["history_only"]["count"], 1)
        self.assertEqual(result["groups"]["unchanged"]["count"], 1)
        self.assertEqual(sum(group["count"] for group in result["groups"].values()), 4)
        for group in result["groups"].values():
            self.assertEqual(set(group["arms"]), set(helper.ARMS))

    def test_pairwise_gains_and_losses_are_not_hidden_by_equal_overall_hr(self):
        result = self.analyze()
        pair = result["overall"]["pairs"]["cross_rec_minus_cross_recon"]
        self.assertEqual(pair["new_hits_at10"], 1)
        self.assertEqual(pair["lost_hits_at10"], 1)
        self.assertEqual(pair["net_hits_at10"], 0)
        self.assertEqual(pair["delta_hr_at10"], 0.0)
        self.assertEqual(pair["ndcg_improved_at10"] + pair["ndcg_worse_at10"]
                         + pair["ndcg_tied_at10"], 4)

    def test_collision_strata_partition_shared_samples(self):
        result = self.analyze()
        for source in ("baseline", "union"):
            strata = result["collision_strata"][source]
            self.assertEqual(strata["singleton"]["count"], 2)
            self.assertEqual(strata["collided"]["count"], 2)
            self.assertEqual(strata["singleton"]["count"] + strata["collided"]["count"], 4)

    def test_new_collision_in_one_update_arm_uses_shared_union_stratum(self):
        # Initially singleton items 3/4 collide only in cross_rec after update.
        # All three models must then use the same collided sample subset.
        self.codes["cross_rec"][3] = list(self.codes["cross_rec"][2])
        result = self.analyze()
        baseline = result["collision_strata"]["baseline"]
        union = result["collision_strata"]["union"]
        self.assertEqual(baseline["singleton"]["count"], 2)
        self.assertEqual(baseline["collided"]["count"], 2)
        self.assertEqual(union["singleton"]["count"], 0)
        self.assertEqual(union["collided"]["count"], 4)
        for arm in helper.ARMS:
            self.assertEqual(union["collided"]["arms"][arm], result["overall"]["arms"][arm])
        self.assertEqual(union["collided"]["pairs"], result["overall"]["pairs"])

    def test_paired_ndcg_delta_preserves_stored_float32_scores_exactly(self):
        rows = self.records()
        result = self.analyze(rows)
        primary = "cross_rec_minus_cross_recon"
        for k in (5, 10):
            metric = f"ndcg@{k}"
            deltas = [left["metrics"][metric] - right["metrics"][metric]
                      for left, right in zip(rows["cross_rec"], rows["cross_recon"])]
            expected = math.fsum(deltas) / 4
            self.assertEqual(result["overall"]["pairs"][primary][f"delta_ndcg_at{k}"], expected)
            for name, indices in (("target_changed", (0, 3)), ("history_only", (1,)), ("unchanged", (2,))):
                group_expected = math.fsum(deltas[index] for index in indices) / len(indices)
                self.assertEqual(result["groups"][name]["pairs"][primary][f"delta_ndcg_at{k}"], group_expected)
            for arm in ("cross_rec", "cross_recon"):
                saved_mean = math.fsum(row["metrics"][metric] for row in rows[arm]) / 4
                self.assertEqual(result["overall"]["arms"][arm][metric], saved_mean)
        # The fixture is intentionally rounded to float32, not idealized DCG.
        analytic_delta = (1 - 1 / math.log2(3) + 1 / math.log2(4) - 1 / math.log2(7)) / 4
        self.assertNotEqual(result["overall"]["pairs"][primary]["delta_ndcg_at10"], analytic_delta)

    def test_rank_ten_is_hit_at_ten_but_not_at_five(self):
        result = self.analyze()
        expected = f32(1 / math.log2(11))
        for arm in helper.ARMS:
            record = self.records()[arm][-1]
            self.assertEqual(record["metrics"]["hit_rate@5"], 0.0)
            self.assertEqual(record["metrics"]["hit_rate@10"], 1.0)
            self.assertEqual(record["metrics"]["ndcg@10"], expected)
        self.assertEqual(result["overall"]["count"], 4)

    def test_repeated_target_predictions_only_score_first_hit(self):
        rows = self.records()
        row = rows["cross_rec"][1]
        row["top10_sids"][7] = list(row["target_sid"])
        # The hit remains raw rank 3 despite another target prediction at rank 8.
        self.assertEqual(row["first_hit_rank"], 3)
        result = self.analyze(rows)
        self.assertEqual(result["overall"]["arms"]["cross_rec"]["hit_rate@10"], .75)

    def test_empty_subgroups_are_explicit_not_nonfinite(self):
        codes = {arm: copy.deepcopy(self.initial) for arm in helper.ARMS}
        self.codes = codes
        result = self.analyze()
        self.assertEqual(result["groups"]["target_changed"]["count"], 0)
        self.assertEqual(result["groups"]["history_only"]["count"], 0)
        self.assertEqual(result["groups"]["unchanged"]["count"], 4)

    def test_duplicate_records_rejected(self):
        rows = self.records()
        rows["cross_rec"][1] = copy.deepcopy(rows["cross_rec"][0])
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_missing_records_rejected(self):
        rows = self.records()
        rows["cross_rec"].pop()
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_wrong_expected_count_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze(expected_count=5)

    def test_row_order_mismatch_rejected(self):
        rows = self.records()
        rows["cross_rec"][0], rows["cross_rec"][1] = rows["cross_rec"][1], rows["cross_rec"][0]
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_same_sid_does_not_hide_target_identity_mismatch(self):
        rows = self.records()
        # Items 1 and 2 collide: equal SID must not imply equal sample identity.
        rows["cross_rec"][1]["target_item_id"] = 2
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_history_identity_mismatch_rejected(self):
        rows = self.records()
        rows["cross_rec"][1]["history_item_ids"] = [4, 0]
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_stale_target_sid_rejected(self):
        rows = self.records()
        rows["cross_recon"][0]["target_sid"] = list(self.initial[2])
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_stale_first_hit_rank_rejected(self):
        rows = self.records()
        rows["cross_rec"][1]["first_hit_rank"] = 4
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_stale_hit_metric_rejected(self):
        rows = self.records()
        rows["cross_rec"][1]["metrics"]["hit_rate@10"] = 0.0
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_stale_ndcg_metric_rejected(self):
        rows = self.records()
        rows["cross_rec"][1]["metrics"]["ndcg@10"] = 0.7
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_nonfinite_metric_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                rows = self.records()
                rows["cross_rec"][1]["metrics"]["ndcg@10"] = value
                with self.assertRaises(ValueError):
                    self.analyze(rows)

    def test_no_mutation_of_input_records_or_catalog(self):
        records = self.records()
        before = copy.deepcopy((records, self.initial, self.codes))
        self.analyze(records)
        self.assertEqual((records, self.initial, self.codes), before)

    def test_pure_helper_import_does_not_load_torch_or_numpy(self):
        source = HELPER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import numpy", source)

    def write_bundle(self, root):
        root.mkdir()
        manifest = dict(initial=self.initial, arms=self.codes)
        with (root / "catalog_ids.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        for arm, rows in self.records().items():
            current = root / arm
            current.mkdir()
            with gzip.open(current / "records.jsonl.gz", "xt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        return manifest

    def test_cli_record_path_cannot_escape_evidence_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            root.mkdir()
            manifest = dict(initial=self.initial, arms=self.codes,
                            records={"frozen": "../outside.jsonl.gz"})
            with (root / "catalog_ids.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            destination = Path(directory) / "new_analysis"
            with self.assertRaises(ValueError):
                helper.main(["--input-dir", str(root), "--output-dir", str(destination), "--expected-count", "4"])
            self.assertFalse(destination.exists())

    def test_cli_refuses_to_overwrite_previous_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            self.write_bundle(root)
            destination = Path(directory) / "old_analysis"
            destination.mkdir()
            preserved = destination / "paired_summary.json"
            with preserved.open("x", encoding="utf-8") as handle:
                handle.write("preserved result")
            with self.assertRaises(FileExistsError):
                helper.main(["--input-dir", str(root), "--output-dir", str(destination), "--expected-count", "4"])
            self.assertEqual(preserved.read_text(encoding="utf-8"), "preserved result")
            self.assertFalse((destination / "README.md").exists())

    def test_cli_keeps_evidence_bytes_and_writes_separate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            self.write_bundle(root)
            originals = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            destination = Path(directory) / "new_analysis"
            with redirect_stdout(io.StringIO()) as stdout:
                helper.main(["--input-dir", str(root), "--output-dir", str(destination), "--expected-count", "4"])
            self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
            result = json.loads((destination / "paired_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["overall"]["count"], 4)
            self.assertTrue((destination / "README.md").is_file())
            self.assertEqual(originals, {path.relative_to(root): path.read_bytes()
                                         for path in root.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
