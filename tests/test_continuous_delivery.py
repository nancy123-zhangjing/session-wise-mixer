"""Independent reporting oracle rejects corrupted rankings and sample identities."""
import copy
import math
import csv
import contextlib
import io
import sys
import tempfile
import json
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checks.verify_continuous_predictions import audit_rows
from scripts.checks.verify_continuous_campaign import audit_curves, parameter_count
from scripts.continuous.delivery import mean_and_std
from scripts.continuous.model import ContinuousRecommender, ModelConfig
from scripts.continuous.trainer import fit
sys.path.insert(0, str(ROOT / "tests"))
from test_continuous_models import tiny_config, tiny_data


class PredictionAuditTests(unittest.TestCase):
    def rows(self):
        truth = [{"user_id": user, "target_position": 4, "target": item}
                 for user, item in ((1, 1), (2, 6), (3, 21))]
        predictions = [{**row, "rank": row["target"], "top_items": list(range(1, 21)), "nll": 2.0}
                       for row in truth]
        return truth, predictions

    def test_independent_top_item_formula(self):
        truth, predictions = self.rows()
        result = audit_rows(predictions, truth, n_items=30, expected_rows=3)
        self.assertEqual(result["HR@5"], 1/3)
        self.assertEqual(result["HR@10"], 2/3)
        self.assertAlmostEqual(result["NDCG@10"], (1 + 1/math.log2(7))/3)
        self.assertEqual(result["NLL"], 2)

    def test_rejects_label_and_rank_tampering(self):
        truth, predictions = self.rows()
        for key, value in (("target", 2), ("rank", 3), ("target_position", 5), ("nll", float("nan"))):
            changed = copy.deepcopy(predictions)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                audit_rows(changed, truth, n_items=30, expected_rows=3)

    def test_rejects_pad_duplicates_and_partial_lists(self):
        truth, predictions = self.rows()
        for top in ([0] + list(range(2, 21)), [1] * 20, list(range(1, 20))):
            changed = copy.deepcopy(predictions)
            changed[0]["top_items"] = top
            with self.assertRaises(ValueError):
                audit_rows(changed, truth, n_items=30, expected_rows=3)

    def test_rejects_missing_extra_and_duplicate_samples(self):
        truth, predictions = self.rows()
        for left, right in ((predictions[:-1], truth), (predictions, truth[:-1]),
                            (predictions + predictions[:1], truth + truth[:1])):
            with self.assertRaises(ValueError):
                audit_rows(left, right, n_items=30, expected_rows=3)


class ReportAuditTests(unittest.TestCase):
    def test_cpu_gate_binds_source_and_actual_cache_before_gpu(self):
        from scripts.continuous.gpu_smoke import validate_cpu_gate
        parent = ROOT / "tmp/continuous_delivery_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp)
            (directory/"cache.json").write_bytes(b"verified cache fixture")
            proof = {"status": "PASS", "skipped": 0, "source_sha256": "fixture-source",
                     "cache_manifest_sha256": hashlib.sha256(b"verified cache fixture").hexdigest()}
            (directory/"CPU_GATE.json").write_text(json.dumps(proof))
            validate_cpu_gate(directory/"CPU_GATE.json", directory, "fixture-source")
            with self.assertRaises(ValueError):
                validate_cpu_gate(directory/"CPU_GATE.json", directory, "other-source")
            (directory/"cache.json").write_bytes(b"different cache")
            with self.assertRaises(ValueError):
                validate_cpu_gate(directory/"CPU_GATE.json", directory, "fixture-source")
    def test_analytic_parameter_count_matches_real_tied_models(self):
        for model in ("sasrec_fullce", "esasrec"):
            for dim, layers in ((16, 1), (32, 2)):
                config = {**tiny_config(model), "dim": dim, "layers": layers}
                network = ContinuousRecommender(ModelConfig(
                    model, n_items=9, max_length=4, dim=dim, layers=layers, heads=2))
                self.assertEqual(parameter_count(config), sum(p.numel() for p in network.parameters()))

    def test_five_seed_sample_standard_deviation(self):
        values = [{f"{kind}@{k}": value for kind in ("HR", "NDCG") for k in (5, 10, 20)}
                  for value in (0.1, 0.2, 0.3, 0.4, 0.5)]
        result = mean_and_std(values)["NDCG@10"]
        self.assertAlmostEqual(result["mean"], 0.3)
        self.assertAlmostEqual(result["std"], math.sqrt(0.025))
        self.assertEqual(result["n"], 5)

    def test_real_training_curves_audited_independently(self):
        parent = ROOT / "tmp/continuous_delivery_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            config, data = tiny_config(), tiny_data()
            output = Path(tmp) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                result = fit(config, 12, "synthetic", data, data, output)
            with (output / "curves.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(audit_curves(config, result, rows, 12, 12)["updates"], 9)
            changed = copy.deepcopy(rows)
            changed[0]["train_samples"] = "11"
            with self.assertRaises(ValueError):
                audit_curves(config, result, changed, 12, 12)
            changed = copy.deepcopy(rows)
            changed[0]["validation_samples"] = "11"
            with self.assertRaises(ValueError):
                audit_curves(config, result, changed, 12, 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
