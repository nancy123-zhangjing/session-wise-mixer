"""Standard-library checks for non-mutating per-example inference capture.

Tensor-shaped fakes keep this suite usable on the Windows development host.
These tests are plumbing checks, not evidence of GPU metric reproduction.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/analysis/evaluate_feedback_paired.py"
spec = importlib.util.spec_from_file_location("_feedback_paired_evaluator_tests", PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load inference capture script: {PATH}")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.value)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.codes = [[0, 0, 0, 0], [1, 1, 1, 1]]
        self.destination = io.StringIO()
        self.collector = entry.SingleModeCollector(self.codes, self.destination)

    def batch(self, targets=(1, 2), histories=None):
        if histories is None:
            histories = [[2, 1, 0] if target == 1 else [1, 0, 0] for target in targets]
        return dict(target_item_id=FakeTensor(list(targets)), history_item_ids=FakeTensor(histories),
                    labels=FakeTensor([[9, 9, 9, 9] for _ in targets]))

    def capture_fixture(self, targets=(1, 2)):
        labels = [list(self.codes[target - 1]) for target in targets]
        predictions = []
        for label in labels:
            other = next(list(row) for row in self.codes if list(row) != label)
            row = [list(other) for _ in range(10)]
            row[1] = list(label)
            predictions.append(row)
        metrics = {key: FakeTensor([0.5 for _ in targets]) for key in entry.PER_USER_METRICS}
        return FakeTensor(predictions), FakeTensor(labels), metrics

    def test_capture_uses_prepared_official_labels_not_stale_collate_labels(self):
        self.collector.begin_batch(self.batch())
        self.collector.capture(*self.capture_fixture())
        self.collector.end_batch()
        self.assertEqual(self.collector.rows[0]["target_sid"], self.codes[0])
        self.assertEqual(self.collector.rows[1]["target_sid"], self.codes[1])
        self.assertEqual(self.collector.rows[0]["history_item_ids"], [2, 1])
        self.assertEqual(self.collector.rows[0]["first_hit_rank"], 2)
        self.assertIsNone(self.collector.pending)

    def test_suffix_is_ignored_without_consuming_confidence_capture(self):
        self.collector.begin_batch(self.batch())
        self.collector.capture(*self.capture_fixture(), suffix="_random")
        self.assertFalse(self.collector.consumed)
        self.assertEqual(self.collector.rows, [])
        self.collector.capture(*self.capture_fixture())
        self.collector.end_batch()
        self.assertEqual(len(self.collector.rows), 2)

    def test_ignored_suffix_does_not_require_metadata(self):
        self.collector.capture(*self.capture_fixture(), suffix="_random")
        self.assertEqual(self.collector.rows, [])

    def test_confidence_capture_without_batch_fails(self):
        with self.assertRaises(RuntimeError):
            self.collector.capture(*self.capture_fixture())

    def test_second_confidence_capture_for_same_batch_fails(self):
        self.collector.begin_batch(self.batch())
        self.collector.capture(*self.capture_fixture())
        with self.assertRaises(RuntimeError):
            self.collector.capture(*self.capture_fixture())

    def test_end_without_confidence_capture_fails(self):
        self.collector.begin_batch(self.batch())
        with self.assertRaises(RuntimeError):
            self.collector.end_batch()

    def test_previous_batch_must_be_ended_before_next_batch(self):
        self.collector.begin_batch(self.batch())
        self.collector.capture(*self.capture_fixture())
        with self.assertRaises(RuntimeError):
            self.collector.begin_batch(self.batch())

    def test_metadata_lengths_must_match(self):
        with self.assertRaises(ValueError):
            self.collector.begin_batch(self.batch(histories=[[1, 0]]))

    def test_short_final_batch_keeps_contiguous_indices(self):
        self.collector.begin_batch(self.batch())
        self.collector.capture(*self.capture_fixture())
        self.collector.end_batch()
        self.collector.begin_batch(self.batch(targets=(1,)))
        self.collector.capture(*self.capture_fixture(targets=(1,)))
        self.collector.end_batch()
        self.assertEqual([row["sample_index"] for row in self.collector.rows], [0, 1, 2])
        saved = [json.loads(line) for line in self.destination.getvalue().splitlines()]
        self.assertEqual(saved, self.collector.rows)

    def test_wrong_official_label_rejected(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        labels.value[0] = list(self.codes[1])
        with self.assertRaises(ValueError):
            self.collector.capture(predictions, labels, metrics)

    def test_target_identity_outside_catalog_rejected(self):
        for target in (0, 3, -1):
            with self.subTest(target=target):
                collector = entry.SingleModeCollector(self.codes, io.StringIO())
                collector.begin_batch(self.batch(targets=(target,)))
                predictions, labels, metrics = self.capture_fixture(targets=(1,))
                with self.assertRaises(ValueError):
                    collector.capture(predictions, labels, metrics)

    def test_batch_prediction_or_label_count_mismatch_rejected(self):
        for name in ("predictions", "labels"):
            with self.subTest(name=name):
                collector = entry.SingleModeCollector(self.codes, io.StringIO())
                collector.begin_batch(self.batch())
                predictions, labels, metrics = self.capture_fixture()
                (predictions if name == "predictions" else labels).value.pop()
                with self.assertRaises(ValueError):
                    collector.capture(predictions, labels, metrics)

    def test_short_topk_or_wrong_digit_count_rejected(self):
        for mode in ("short_topk", "wrong_digit_count"):
            with self.subTest(mode=mode):
                collector = entry.SingleModeCollector(self.codes, io.StringIO())
                collector.begin_batch(self.batch())
                predictions, labels, metrics = self.capture_fixture()
                if mode == "short_topk":
                    predictions.value[0].pop()
                else:
                    predictions.value[0][0].pop()
                with self.assertRaises(ValueError):
                    collector.capture(predictions, labels, metrics)

    def test_out_of_range_code_rejected(self):
        for code in (-1, 256):
            with self.subTest(code=code):
                collector = entry.SingleModeCollector(self.codes, io.StringIO())
                collector.begin_batch(self.batch())
                predictions, labels, metrics = self.capture_fixture()
                predictions.value[0][0][0] = code
                with self.assertRaises(ValueError):
                    collector.capture(predictions, labels, metrics)

    def test_invalid_catalog_sid_is_kept_without_reordering_raw_rank(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        predictions.value[0][0] = [99, 99, 99, 99]
        self.collector.capture(predictions, labels, metrics)
        self.collector.end_batch()
        self.assertEqual(self.collector.rows[0]["top10_sids"][0], [99, 99, 99, 99])
        self.assertEqual(self.collector.rows[0]["first_hit_rank"], 2)

    def test_repeated_matching_sid_uses_first_raw_rank(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        predictions.value[0][7] = list(labels.value[0])
        self.collector.capture(predictions, labels, metrics)
        self.collector.end_batch()
        self.assertEqual(self.collector.rows[0]["first_hit_rank"], 2)

    def test_missing_hit_is_zero_not_rank_eleven(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        predictions.value[0][1] = list(self.codes[1])
        self.collector.capture(predictions, labels, metrics)
        self.collector.end_batch()
        self.assertEqual(self.collector.rows[0]["first_hit_rank"], 0)

    def test_per_batch_single_value_cannot_be_used_as_per_user_metric(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        metrics["ndcg@10"] = FakeTensor([0.5])
        with self.assertRaises(ValueError):
            self.collector.capture(predictions, labels, metrics)

    def test_zero_dimensional_metric_scalar_is_clearly_rejected(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        metrics["ndcg@10"] = FakeTensor(0.5)
        with self.assertRaises(ValueError):
            self.collector.capture(predictions, labels, metrics)

    def test_missing_required_per_user_metric_rejected(self):
        self.collector.begin_batch(self.batch())
        predictions, labels, metrics = self.capture_fixture()
        metrics.pop("ndcg@10")
        with self.assertRaises((KeyError, ValueError)):
            self.collector.capture(predictions, labels, metrics)

    def test_nonfinite_metric_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                collector = entry.SingleModeCollector(self.codes, io.StringIO())
                collector.begin_batch(self.batch())
                predictions, labels, metrics = self.capture_fixture()
                metrics["ndcg@10"].value[0] = value
                with self.assertRaises(FloatingPointError):
                    collector.capture(predictions, labels, metrics)

    def test_observer_does_not_mutate_tensor_like_inputs(self):
        batch = self.batch()
        predictions, labels, metrics = self.capture_fixture()
        before = copy.deepcopy((batch, predictions, labels, metrics))
        self.collector.begin_batch(batch)
        self.collector.capture(predictions, labels, metrics)
        self.collector.end_batch()
        self.assertEqual(batch["history_item_ids"].value, before[0]["history_item_ids"].value)
        self.assertEqual(predictions.value, before[1].value)
        self.assertEqual(labels.value, before[2].value)
        self.assertEqual({k: v.value for k, v in metrics.items()}, {k: v.value for k, v in before[3].items()})


class LoaderAndIsolationTests(unittest.TestCase):
    def test_loader_yields_exact_original_objects_and_preserves_call_order(self):
        batches = [{"target": 1}, {"target": 2}, {"target": 3}]
        events = []
        observer = SimpleNamespace(begin_batch=lambda batch: events.append(("begin", batch)),
                                   end_batch=lambda: events.append(("end", None)))
        loader = entry.ObservingLoader(batches, observer)
        self.assertEqual(len(loader), 3)
        iterator = iter(loader)
        for batch in batches:
            actual = next(iterator)
            self.assertIs(actual, batch)
            events.append(("consumer", actual))
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertEqual([event[0] for event in events], ["begin", "consumer", "end"] * 3)

    def test_empty_loader_does_not_call_observer(self):
        observer = SimpleNamespace(begin_batch=lambda batch: self.fail("Unexpected begin"),
                                   end_batch=lambda: self.fail("Unexpected end"))
        self.assertEqual(list(entry.ObservingLoader([], observer)), [])

    def test_loader_does_not_generate_random_numbers(self):
        observer = SimpleNamespace(begin_batch=lambda batch: None, end_batch=lambda: None)
        with patch.object(entry.random, "random", side_effect=AssertionError("RNG consumed")), \
             patch.object(entry.random, "seed", side_effect=AssertionError("RNG reseeded")):
            self.assertEqual(list(entry.ObservingLoader([1, 2], observer)), [1, 2])

    def test_restore_rng_uses_saved_streams_not_new_seeds(self):
        calls = []
        fake_torch = SimpleNamespace(set_rng_state=lambda state: calls.append(("cpu", state)),
                                     cuda=SimpleNamespace(set_rng_state=lambda state, device: calls.append(("cuda", state, device))))
        fake_np = SimpleNamespace(random=SimpleNamespace(set_state=lambda state: calls.append(("numpy", state))))
        state = dict(python="python_state", numpy="numpy_state", torch_cpu="cpu_state", torch_cuda="cuda_state")
        with patch.object(entry.random, "setstate", side_effect=lambda value: calls.append(("python", value))):
            entry.restore_rng(fake_torch, fake_np, state, "cuda:0")
        self.assertEqual(calls, [("python", "python_state"), ("numpy", "numpy_state"),
                                 ("cpu", "cpu_state"), ("cuda", "cuda_state", "cuda:0")])

    def test_missing_original_cuda_rng_is_rejected(self):
        fake = SimpleNamespace(set_rng_state=lambda state: None)
        fake_np = SimpleNamespace(random=SimpleNamespace(set_state=lambda state: None))
        state = dict(python=None, numpy=None, torch_cpu=None, torch_cuda=None)
        with patch.object(entry.random, "setstate"):
            with self.assertRaises(ValueError):
                entry.restore_rng(fake, fake_np, state, "cuda:0")

    def test_new_unique_output_only_and_no_implicit_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result = entry.validate_output(root, Path("outputs/analysis/stage05_feedback/new"))
            self.assertEqual(result, root / "outputs/analysis/stage05_feedback/new")
            self.assertFalse(result.exists())
            result.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                entry.validate_output(root, result)

    def test_historical_or_nested_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for candidate in ("outputs/pilots/stage05_feedback/existing", "experiments/05_crosspse_joint/evidence/new",
                              "outputs/analysis/stage05_feedback", "outputs/analysis/stage05_feedback/new/nested"):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    entry.validate_output(Path(directory).resolve(), Path(candidate))

    def test_inference_entry_never_calls_training_or_sid_refresh(self):
        tree = ast.parse(PATH.read_text(encoding="utf-8"))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"fit", "backward", "step", "train_phase", "refresh_joint_codes", "refresh_hard_codes"})
        self.assertIn("validate_snapshot", calls)
        self.assertIn("sync_joint_tokenizer_mapping", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
