"""CPU protocol/receipt tests; no tensor runtime, training, or GPU required."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multiseed_eval_tested", ROOT / "scripts/analysis/evaluate_item_local_multiseed.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture(seed=2025, variant="control", best=0):
    campaign = f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1"
    run_id = f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"
    identity = {"seed": seed, "variant": variant, "campaign_id": campaign,
                "parent_campaign_id": evaluation.PARENT_CAMPAIGN, "run_id": run_id}
    config = {"rand_seed": seed, "run_id": run_id, "epochs": 45, "patience": 0,
              "eval_start_epoch": 5, "eval_interval": 5, "val_metric": "weighted_score",
              "vectorized_beam_search": {"val": {"beam_act": 32, "beam_max": 32}},
              "beam_search_modes": ["confidence"], "topk": [5, 10], "eval_batch_size": 8,
              "max_history_len": 50, "mixed_precision": "bf16", "joint_crosspse_enabled": True,
              "crosspse_mixer_enabled": False, "crosspse_train_base_codebooks": True,
              "joint_gradient_mode": "diger_st", "eval_expand_sid_to_items": False,
              "dedup_strategy": "simple", "item_local_attention": {"enabled": variant == "item_local"}}
    fit = {"completed_epochs": 45, "training_stop_reason": "max_epochs", "cumulative_recommender_optimizer_updates": 3640,
           "best_epoch": best, "best_val_score": .1,
           "joint_progress": {"tokenizer_epochs_completed": 10, "tokenizer_optimizer_updates": 60, "sid_refresh_count": 10}}
    result = {"campaign_id": campaign, "run_id": run_id, "variant": variant, "fit": copy.deepcopy(fit),
              "status": "formal_training_completed_validation_pending", "test_data_evaluated": False,
              "final_validation_pending": [32, 256]}
    schedule = ["recommender"] * 5 + [p for _ in range(10) for p in ("tokenizer", "recommender", "recommender", "recommender")]
    phases = [{"epoch": i + 1, "phase": phase, "optimizer_updates": 104 if phase == "recommender" else 6,
               "frozen_parameters_unchanged": True, "all_active_gradients_finite": True} for i, phase in enumerate(schedule)]
    return identity, config, result, fit, phases


class MultiseedValidationTests(unittest.TestCase):
    def test_both_seeds_and_variants_accept_best_zero_and_non25(self):
        for seed in (2025, 2026):
            for variant in ("control", "item_local"):
                for best in (0, 12, 39, 45):
                    got = evaluation.audit_training(*fixture(seed, variant, best))
                    self.assertEqual(got["seed"], seed)
                    self.assertEqual(got["best_epoch"], best)
                    self.assertEqual(got["recommender_updates"], 3640)

    def test_rejects_early_stop_incomplete_budget_and_test(self):
        cases = []
        data = list(fixture()); data[3]["completed_epochs"] = 44; data[2]["fit"] = copy.deepcopy(data[3]); cases.append(data)
        data = list(fixture()); data[3]["training_stop_reason"] = "early_stopping"; data[2]["fit"] = copy.deepcopy(data[3]); cases.append(data)
        data = list(fixture()); data[3]["cumulative_recommender_optimizer_updates"] = 3639; data[2]["fit"] = copy.deepcopy(data[3]); cases.append(data)
        data = list(fixture()); data[3]["joint_progress"]["tokenizer_optimizer_updates"] = 59; data[2]["fit"] = copy.deepcopy(data[3]); cases.append(data)
        data = list(fixture()); data[2]["test_data_evaluated"] = True; cases.append(data)
        for data in cases:
            with self.assertRaises(ValueError):
                evaluation.audit_training(*data)

    def test_rejects_phase_order_freeze_seed_and_selection_changes(self):
        for mutation in (
            lambda d: d[4][5].update(phase="recommender"),
            lambda d: d[4][5].update(frozen_parameters_unchanged=False),
            lambda d: d[1].update(rand_seed=2024),
            lambda d: d[1].update(patience=5),
            lambda d: d[1]["vectorized_beam_search"].update(val={"beam_act": 256, "beam_max": 256}),
            lambda d: d[1].update(crosspse_mixer_enabled=True),
        ):
            data = list(fixture()); mutation(data)
            with self.assertRaises(ValueError):
                evaluation.audit_training(*data)

    def test_selected_csv_supports_zero_filters_split_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            rows = [{"epoch": 0, "split": "val", "metric": key, "value": .1} for key in evaluation.SELECTED_METRICS]
            rows += [{"epoch": 0, "split": "test", "metric": "weighted_score", "value": .9},
                     {"epoch": 25, "split": "val", "metric": "weighted_score", "value": .2}]
            def save():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("epoch", "split", "metric", "value"))
                    writer.writeheader(); writer.writerows(rows)
            save()
            self.assertEqual(evaluation.selected_validation(path, 0, .1)["weighted_score"], .1)
            with self.assertRaisesRegex(ValueError, "Selected score"):
                evaluation.selected_validation(path, 0, .2)
            rows.append(rows[0].copy()); save()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluation.selected_validation(path, 0, .1)

    def test_validation_rows_exclude_test_targets_and_one_item_users(self):
        self.assertEqual(evaluation.validation_rows({"u1": [1, 2, 3, 4], "u2": [8, 9], "u3": [7]}),
                         [("u1", [1, 2, 3]), ("u2", [8])])

    def test_remapping_and_path_guard_preserve_unrelated_sources(self):
        got = evaluation.remap({"a": "/old/x", "b": ["/old", "/older/x"]}, "/old", "/private")
        self.assertEqual(got, {"a": "/private/x", "b": ["/private", "/older/x"]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                evaluation.under(directory, "../escape")

    def test_top10_rank_and_repeat_fill(self):
        result = evaluation.top10_diagnostics([[1, 2], [3, 4], [3, 4]], [3, 4], lambda row: row == (3, 4))
        self.assertEqual(result, {"target_rank": 2, "top10_unique_count": 2, "top10_unique_legal_count": 1,
                                  "inferred_repeat_fill_slots": 1, "no_legal_top10_fallback": False})

    def test_completion_rejects_changed_bytes_or_different_sid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cell = root / "beam32"; cell.mkdir()
            started = cell / "CELL_STARTED.json"; evaluation.write(started, {"identity_sha256": "id"})
            metrics_path = cell / "metrics.json"
            metrics = {"count": evaluation.N_USERS, "split": "val", "beam_act": 32, "checkpoint_sha256": "ckpt",
                       "selected_SID_sha256": "sid", "model_state_unchanged": True, "selected_SID_unchanged": True,
                       "training_gradients_created": False, "test_data_evaluated": False,
                       "original_val32_reproduction": {"status": "matched"}}
            evaluation.write(metrics_path, metrics)
            receipt = {"status": "cell_completed", "identity_sha256": "id", "beam": 32,
                       "started_receipt": evaluation.record(started), "files": {"metrics": evaluation.record(metrics_path)}}
            evaluation.write(cell / "CELL_COMPLETE.json", receipt)
            plan = {"identity_sha256": "id", "checkpoint": {"sha256": "ckpt"}, "selected_SID_sha256": "sid"}
            self.assertIsNotNone(evaluation.completed_cell(root, 32, plan))
            with self.assertRaises(ValueError):
                evaluation.completed_cell(root, 32, {**plan, "selected_SID_sha256": "other"})
            metrics_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                evaluation.completed_cell(root, 32, plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
