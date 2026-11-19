"""Offline summary tests. Synthetic evidence exists only in temporary folders."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multiseed_summary", ROOT / "experiments/07_item_local_attention/multiseed_20261021/summarize_results.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def rec(path, remote=None):
    return {"path": str(remote or path), "sha256": summary.sha(path), "size_bytes": path.stat().st_size}


def rows():
    output = []
    for index, rank in enumerate((None, 1, 3)):
        target = [index + 1] * 4
        predictions = [[100] * 4] * 10
        if rank:
            predictions[rank - 1] = target
        unique = len({tuple(row) for row in predictions})
        ndcg = 1 / math.log2(rank + 1) if rank else 0.0
        metrics = {"duplicate@10": 1 - unique / 10, "weighted_score": .8 * ndcg + .2 * bool(rank)}
        for k in (5, 10):
            metrics.update({f"hit_rate@{k}": float(bool(rank)), f"recall@{k}": float(bool(rank)), f"ndcg@{k}": ndcg})
        output.append({"index": index, "user": f"u{index}", "target_item_id": index + 1,
                       "target_SID": target, "predicted_SIDs": predictions, "target_rank": rank,
                       "top10_unique_count": unique, "inferred_repeat_fill_slots": 10 - unique, "metrics": metrics})
    return output


def jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")


def make_run(root, seed=2025, variant="control"):
    """Small fake evidence; not accepted by the real CLI's fixed 22363 count."""
    name = f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"
    run = root / name
    remote = "/synthetic_remote/" + name
    checkpoint = run / "training_state/best_model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"TEST FIXTURE, NOT A MODEL")
    checkpoint_sha = summary.sha(checkpoint)
    phases = [{"epoch": i + 1, "phase": phase, "optimizer_updates": 104 if phase == "recommender" else 6,
               "frozen_parameters_unchanged": True, "all_active_gradients_finite": True}
              for i, phase in enumerate(["recommender"] * 5 + [p for _ in range(10) for p in ("tokenizer", "recommender", "recommender", "recommender")])]
    put(run / "phase_integrity.json", phases)
    inputs = {f"input{i}": f"{i:064x}" for i in range(13)}
    initial = {"initial_base_state_sha256": f"{seed:064x}", "initial_rng_sha256": f"{seed + 1:064x}", "initial_SID_sha256": "d" * 64}
    identity = {"seed": seed, "variant": variant, "run_id": name, "engine_commit": summary.ENGINE,
                "engine_source_sha256": {"engine.py": "e" * 64}}
    config = {"rand_seed": seed, "run_id": name, "patience": 0, "epochs": 45, "val_metric": "weighted_score",
              "min_delta": 0, "eval_start_epoch": 5, "eval_interval": 5, "vectorized_beam_search": {"val": {"beam_act": 32, "beam_max": 32}},
              "item_local_attention": {"enabled": variant == "item_local"}, "crosspse_mixer_enabled": False,
              "joint_gradient_mode": "diger_st", "eval_expand_sid_to_items": False, "training_state_identity": {"inputs": inputs}}
    raw_path = run / "fixture.jsonl"
    jsonl(raw_path, rows())
    calculated = summary.recompute_rows(raw_path, 3)
    raw_path.unlink()
    fit = {"best_epoch": 0, "best_val_score": calculated["metrics"]["weighted_score"], "completed_epochs": 45,
           "training_stop_reason": "max_epochs", "cumulative_recommender_optimizer_updates": 3640,
           "joint_progress": {"tokenizer_epochs_completed": 10, "tokenizer_optimizer_updates": 60, "sid_refresh_count": 10}}
    result = {"variant": variant, "run_id": name, "status": "formal_training_completed_validation_pending",
              "fit": fit, "checkpoint_sha256": checkpoint_sha, "selected_SID_sha256": "f" * 64,
              "phase_integrity_sha256": summary.sha(run / "phase_integrity.json"),
              "test_data_evaluated": False, "final_validation_pending": [32, 256], **initial}
    documents = {"identity.json": identity, "resolved_config.json": config, "results.json": result,
                 "FIT_COMPLETE.json": fit, "initial_state_identity.json": initial,
                 "input_identities.json": {"artifacts": [{"path": k, "sha256": v} for k, v in inputs.items()]},
                 "checkpoint_reload.json": {"checkpoint_sha256": checkpoint_sha, "best_epoch": 0, "strict_all_tensor_match": True,
                                            "mapping_synchronized_without_requantization": True, "matching_committed_SID_sha256": "f" * 64}}
    for n, value in documents.items():
        put(run / n, value)
    put(run / "COMPLETE.json", {"mode": "formal", "run_id": name, "status": result["status"], "result_file": "results.json",
                               "result_sha256": summary.sha(run / "results.json"), "checkpoint_sha256": checkpoint_sha})
    csv_path = run / "diagnostics/fixture_evaluation_metrics.csv"
    csv_path.parent.mkdir()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "epoch", "metric", "value")); writer.writeheader()
        for key, value in calculated["metrics"].items():
            writer.writerow({"split": "val", "epoch": 0, "metric": key, "value": value})
    original = {n: rec(run / n, remote + "/" + n) for n in (*documents, "COMPLETE.json", "phase_integrity.json")}
    original["selected_validation_csv"] = rec(csv_path, remote + "/diagnostics/fixture_evaluation_metrics.csv")
    plan = {"run": remote, "validation_count": 3, "split": "val", "checkpoint": rec(checkpoint, remote + "/training_state/best_model.bin"),
            "selected_SID_sha256": "f" * 64, "reselection_allowed": False, "original_files": original}
    plan["identity_sha256"] = summary.json_sha(plan)
    put(run / "validation/EVALUATION_PLAN.json", plan)
    complete = {"status": "selected_validation_completed", "identity_sha256": plan["identity_sha256"], "seed": seed,
                "variant": variant, "best_epoch": 0, "test_data_evaluated": False, "training_performed": False, "cells": {}}
    for beam in (32, 256):
        cell = run / f"validation/beam{beam}"
        metrics = {"seed": seed, "variant": variant, "best_epoch": 0, "beam_act": beam, "beam_max": beam,
                   "split": "val", "count": 3, "checkpoint_sha256": checkpoint_sha, "selected_SID_sha256": "f" * 64,
                   "model_state_unchanged": True, "selected_SID_unchanged": True, "test_data_evaluated": False,
                   "training_gradients_created": False, "model_state_before_sha256": "a" * 64, "model_state_after_sha256": "a" * 64,
                   "metrics": calculated["metrics"], "user_target_order_sha256": calculated["order_sha256"],
                   "validation_user_history_sha256": "b" * 64, "original_val32_reproduction": {"status": "matched"}}
        put(cell / "metrics.json", metrics); jsonl(cell / "per_user.jsonl", rows())
        put(cell / "CELL_STARTED.json", {"identity_sha256": plan["identity_sha256"], "beam": beam})
        files = {n: rec(cell / f"{n}.{ext}", f"{remote}/validation/beam{beam}/{n}.{ext}") for n, ext in (("metrics", "json"), ("per_user", "jsonl"))}
        receipt = {"status": "cell_completed", "identity_sha256": plan["identity_sha256"], "beam": beam, "files": files,
                   "started_receipt": rec(cell / "CELL_STARTED.json", f"{remote}/validation/beam{beam}/CELL_STARTED.json")}
        put(cell / "CELL_COMPLETE.json", receipt)
        complete["cells"][f"beam{beam}"] = {"receipt": rec(cell / "CELL_COMPLETE.json", f"{remote}/validation/beam{beam}/CELL_COMPLETE.json"),
                                             "files": files, "metrics": metrics}
    put(run / "validation/EVALUATION_COMPLETE.json", complete)
    return run


class MultiseedSummaryTests(unittest.TestCase):
    def test_remote_path_mapping_requires_exact_root_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = put(root / "cell/a.json", {"x": 1})
            evidence = summary.Evidence(); item = rec(path, "/remote/run/cell/a.json")
            self.assertEqual(evidence.mapped(item, root, "/remote/run"), path.resolve())
            for bad in ("/remote/runner/cell/a.json", "/remote/run/../escape.json"):
                with self.assertRaises(ValueError):
                    evidence.mapped({**item, "path": bad}, root, "/remote/run")
            put(path, {"x": 2})
            with self.assertRaises(ValueError):
                evidence.mapped(item, root, "/remote/run")

    def test_metrics_use_first_matching_sid_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"; data = rows(); jsonl(path, data)
            got = summary.recompute_rows(path, 3)
            self.assertEqual(got["hits10"], 2)
            self.assertAlmostEqual(got["metrics"]["ndcg@10"], .5)
            for mutation in (lambda d: d[2].update(target_rank=1),
                             lambda d: d[1]["metrics"].update({"hit_rate@10": 0}),
                             lambda d: d[1].update(user="u0"),
                             lambda d: d[2].update(index=5)):
                bad = copy.deepcopy(data); mutation(bad); jsonl(path, bad)
                with self.assertRaises(ValueError):
                    summary.recompute_rows(path, 3)
            jsonl(path, data)
            with self.assertRaises(ValueError):
                summary.recompute_rows(path, 4)

    def test_new_complete_supports_best_zero_and_checks_all_bindings(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(summary, "N_USERS", 3):
            run = make_run(Path(tmp)); evidence = summary.Evidence()
            training = summary.audit_training(run, 2025, "control", evidence)
            cells = summary.read_new(run, training, evidence)
            self.assertEqual([c["beam"] for c in cells], [32, 256])
            self.assertEqual(cells[0]["best_epoch"], 0)
            self.assertEqual(cells[0]["metrics"]["hit_rate@10"], 2 / 3)
            put(run / "validation/beam256/metrics.json", {"tampered": True})
            with self.assertRaises(ValueError):
                summary.read_new(run, training, summary.Evidence())

    def test_refuses_wrong_budget_and_changed_best_without_reselection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(summary, "N_USERS", 3):
            run = make_run(Path(tmp))
            fit = summary.read(run / "FIT_COMPLETE.json"); fit["cumulative_recommender_optimizer_updates"] = 3639
            result = summary.read(run / "results.json"); result["fit"] = fit
            put(run / "FIT_COMPLETE.json", fit); put(run / "results.json", result)
            complete = summary.read(run / "COMPLETE.json"); complete["result_sha256"] = summary.sha(run / "results.json")
            put(run / "COMPLETE.json", complete)
            with self.assertRaisesRegex(ValueError, "budget"):
                summary.audit_training(run, 2025, "control", summary.Evidence())

    def test_selection_preserves_earliest_tie_and_rejects_reselected_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("split", "epoch", "metric", "value")); writer.writeheader()
                for epoch in (0, 5):
                    for key in summary.METRICS:
                        writer.writerow({"split": "val", "epoch": epoch, "metric": key, "value": .2})
            self.assertEqual(summary.selection(path, {"best_epoch": 0, "best_val_score": .2})["weighted_score"], .2)
            with self.assertRaisesRegex(ValueError, "original"):
                summary.selection(path, {"best_epoch": 5, "best_val_score": .2})

    def test_statistics_use_paired_seed_deltas_and_sample_sd(self):
        cells = []
        for seed, a, b in ((2024, .1, .12), (2025, .3, .29), (2026, .5, .53)):
            for beam in summary.BEAMS:
                for variant, value in (("control", a), ("item_local", b)):
                    cells.append({"seed": seed, "variant": variant, "beam": beam,
                                  "metrics": {"hit_rate@10": value, "ndcg@10": value / 2}})
        got = summary.paired_statistics(cells, (2025, 2026))["32"]["hit_rate@10"]["paired_delta"]
        self.assertAlmostEqual(got["mean"], .01)
        self.assertAlmostEqual(got["sample_sd"], math.sqrt(.0008))
        self.assertEqual(got["positive_seeds"], 1)

    def test_missing_four_runs_does_not_create_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / "report"
            args = ["summary", "--new-root", str(root), "--old-grid-root", str(root),
                    "--old-training-root", str(root), "--output", str(output)]
            with patch("sys.argv", args), self.assertRaises(ValueError):
                summary.main()
            self.assertFalse(output.exists())

    def test_full_new_pair_checks_initial_rng_and_user_alignment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(summary, "N_USERS", 3):
            root = Path(tmp)
            for seed in (2025, 2026):
                for variant in summary.VARIANTS:
                    make_run(root, seed, variant)
            # Deliberate test-only historical stub: production read_old is never bypassed.
            old_train, old_cells = {}, []
            for variant in summary.VARIANTS:
                run = summary.unique_run(root, 2025, variant)
                training = summary.audit_training(run, 2025, variant, summary.Evidence())
                cells = summary.read_new(run, training, summary.Evidence())
                training = copy.deepcopy(training); training["seed"] = 2024
                old_train[(2024, variant)] = training
                for cell in cells:
                    cell["seed"] = 2024
                old_cells += cells
            with patch.object(summary, "read_old", return_value=(old_train, old_cells)):
                got = summary.build_summary(root, root, root)
                self.assertEqual(got["cell_count"], 12)
                self.assertEqual(len(got["paired"]), 6)
                self.assertIn("样本标准差", summary.report(got))
                old_train[(2024, "item_local")]["initial"]["initial_rng_sha256"] = "c" * 64
                with self.assertRaisesRegex(ValueError, "Paired initialization"):
                    summary.build_summary(root, root, root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
