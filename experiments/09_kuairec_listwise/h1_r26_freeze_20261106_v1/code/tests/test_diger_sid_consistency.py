"""CPU tests for the standalone SID-consistency diagnostic.

Run from the project root with an existing NumPy/PyTorch environment:
    python -B tests/test_diger_sid_consistency.py

The reference tests load only the two runtime source modules, not the full
DiffGRM package, and never instantiate models from historical checkpoints.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("DIFFGRM_PROJECT_ROOT", str(ROOT)))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = load_module(
    "sid_consistency_diagnostic",
    ROOT / "scripts" / "analysis" / "analyze_diger_sid_consistency.py",
)


def runtime_modules():
    """Isolate relative imports from unrelated package-level dependencies."""
    package_name = "_sid_consistency_runtime_reference"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    source = ENGINE_ROOT / "external" / "DiffGRM" / "genrec" / "models" / "DIFF_GRM"
    cross = load_module(f"{package_name}.cross_pse", source / "cross_pse.py")
    joint = load_module(
        f"{package_name}.joint_cross_pse", source / "joint_cross_pse.py"
    )
    return cross, joint


class CatalogAlignmentTests(unittest.TestCase):
    def test_reorders_without_changing_values_or_dtype(self):
        array = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.int64)
        aligned = diagnostic.align_catalog(array, ["B", "C", "A"], ["A", "B", "C"])
        np.testing.assert_array_equal(aligned, array[[2, 0, 1]])
        self.assertEqual(aligned.dtype, array.dtype)

    def test_rejects_duplicate_source_keys(self):
        with self.assertRaises(ValueError):
            diagnostic.align_catalog(np.zeros((2, 1)), ["A", "A"], ["A", "B"])

    def test_rejects_duplicate_catalog_keys(self):
        with self.assertRaises(ValueError):
            diagnostic.align_catalog(np.zeros((2, 1)), ["A", "B"], ["A", "A"])

    def test_rejects_missing_or_extra_keys(self):
        with self.assertRaises(ValueError):
            diagnostic.align_catalog(np.zeros((2, 1)), ["A", "B"], ["A", "C"])
        with self.assertRaises(ValueError):
            diagnostic.align_catalog(np.zeros((2, 1)), ["A", "B"], ["A"])

    def test_rejects_row_count_mismatch(self):
        with self.assertRaises(ValueError):
            diagnostic.align_catalog(np.zeros((3, 1)), ["A", "B"], ["A", "B"])


class AssignmentTests(unittest.TestCase):
    def test_distances_match_explicit_broadcasting(self):
        slots = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        books = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]])
        expected = (slots[:, :, None, :] - books[None, :, :, :]).square().sum(-1)
        self.assertTrue(torch.equal(diagnostic.squared_distances(slots, books), expected))

    def test_distances_bitwise_equal_runtime(self):
        runtime_cross, _ = runtime_modules()
        generator = torch.Generator().manual_seed(2024)
        slots = torch.randn(7, 4, 8, generator=generator)
        books = torch.randn(4, 16, 8, generator=generator)
        expected = runtime_cross.CrossPSECodebookMixer.squared_distances(slots, books)
        self.assertTrue(torch.equal(diagnostic.squared_distances(slots, books), expected))

    def test_zero_noise_is_deterministic_including_ties(self):
        logits = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 3.0, 1.0]]])
        actual = diagnostic.sample_hard(logits, 0.0, 0.4, torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(actual, logits.argmax(-1)))

    def test_same_seed_reproduces_sample(self):
        logits = torch.zeros(64, 4, 8)
        left = diagnostic.sample_hard(logits, 1.0, 1.0, torch.Generator().manual_seed(9))
        right = diagnostic.sample_hard(logits, 1.0, 1.0, torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(left, right))
        changed_seed = diagnostic.sample_hard(
            logits, 1.0, 1.0, torch.Generator().manual_seed(10)
        )
        self.assertFalse(torch.equal(left, changed_seed))

    def test_positive_temperature_does_not_change_hard_choice(self):
        logits = torch.tensor([[[0.0, -0.3, -0.8], [-0.8, 0.0, -0.3]]]).repeat(32, 1, 1)
        warm = diagnostic.sample_hard(logits, 0.7, 1.0, torch.Generator().manual_seed(7))
        cool = diagnostic.sample_hard(logits, 0.7, 0.2, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(warm, cool))

    def test_hard_choices_match_actual_runtime_bridge(self):
        runtime_cross, runtime_joint = runtime_modules()
        generator = torch.Generator().manual_seed(11)
        slots = torch.randn(13, 2, 4, generator=generator)
        books = torch.randn(2, 8, 4, generator=generator)

        class FixedMixer(nn.Module):
            n_digit = 2
            codebook_size = 8
            squared_distances = staticmethod(runtime_cross.CrossPSECodebookMixer.squared_distances)

            def __init__(self):
                super().__init__()
                self.register_buffer("books", books)

            def crossed_codebooks(self, return_attention=False):
                return self.books, None

        bridge = runtime_joint.JointCrossPSEAlignment.__new__(runtime_joint.JointCrossPSEAlignment)
        nn.Module.__init__(bridge)
        bridge.register_buffer("item_slots", torch.cat([torch.zeros(1, 2, 4), slots]))
        bridge.mixer = FixedMixer()
        bridge.diger_assignment_temperature = 0.2
        bridge.diger_schedule_state = lambda: {
            "gumbel_noise_scale": 0.7,
            "softmax_temperature": 0.9,
        }
        bridge.train()
        logits = -diagnostic.squared_distances(slots, books) / 0.2
        expected = diagnostic.sample_hard(
            logits, 0.7, 0.9, torch.Generator().manual_seed(2024)
        )
        torch.manual_seed(2024)
        actual_one_hot, _ = bridge.diger_assignments(torch.arange(1, 14), stochastic=True)
        self.assertTrue(torch.equal(actual_one_hot.argmax(-1), expected))
        self.assertTrue(torch.equal(actual_one_hot.sum(-1), torch.ones(13, 2)))

    def test_analytic_uniform_probabilities(self):
        result = diagnostic.analytic_agreement(torch.zeros(2, 3, 4), 1.0)
        for name in ("deterministic_digit_match", "pair_digit_match"):
            torch.testing.assert_close(result[name], torch.full((2, 3), 0.25, dtype=torch.float64))
        for name in ("deterministic_item_match", "pair_item_match"):
            torch.testing.assert_close(result[name], torch.full((2,), 0.25**3, dtype=torch.float64))

    def test_analytic_no_noise_is_exactly_one(self):
        result = diagnostic.analytic_agreement(torch.tensor([[[0.0, 0.0, -1.0]]]), 0.0)
        for value in result.values():
            self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_analytic_peaked_distribution(self):
        logits = torch.tensor([[[0.0, -2.0], [-3.0, 0.0]]], dtype=torch.float64)
        result = diagnostic.analytic_agreement(logits, 0.5)
        probabilities = torch.softmax(logits / 0.5, -1)
        digit_match = probabilities.max(-1).values
        pair_match = probabilities.square().sum(-1)
        torch.testing.assert_close(result["deterministic_digit_match"], digit_match)
        torch.testing.assert_close(result["deterministic_item_match"], digit_match.prod(-1))
        torch.testing.assert_close(result["pair_digit_match"], pair_match)
        torch.testing.assert_close(result["pair_item_match"], pair_match.prod(-1))

    def test_analytic_prediction_matches_monte_carlo(self):
        logits = torch.tensor([[[0.0, -0.7, -1.4], [-1.4, -0.7, 0.0]]]).repeat(20000, 1, 1)
        result = diagnostic.analytic_agreement(logits[:1], 0.8)
        left = diagnostic.sample_hard(logits, 0.8, 1.0, torch.Generator().manual_seed(55))
        right = diagnostic.sample_hard(logits, 0.8, 1.0, torch.Generator().manual_seed(56))
        deterministic = logits.argmax(-1)
        observed_item_match = left.eq(deterministic).all(-1).float().mean()
        observed_pair_match = left.eq(right).all(-1).float().mean()
        self.assertLess(abs(float(observed_item_match - result["deterministic_item_match"][0])), 0.015)
        self.assertLess(abs(float(observed_pair_match - result["pair_item_match"][0])), 0.015)

    def test_rejects_invalid_sampling_parameters(self):
        logits = torch.zeros(1, 2, 3)
        with self.assertRaises(ValueError):
            diagnostic.sample_hard(logits, -1.0, 1.0, torch.Generator())
        with self.assertRaises(ValueError):
            diagnostic.sample_hard(logits, 1.0, 0.0, torch.Generator())
        with self.assertRaises(ValueError):
            diagnostic.sample_hard(torch.full((1, 2, 3), float("nan")), 1.0, 1.0, torch.Generator())


class DeterministicChangeTests(unittest.TestCase):
    def test_count_denominators_are_distinct(self):
        initial = np.array([[0, 1], [1, 0], [2, 2]])
        current = np.array([[0, 1], [2, 0], [2, 1]])
        result = diagnostic.compare_codes(initial, current)
        self.assertEqual(result["changed_items"], 2)
        self.assertEqual(result["changed_digits"], 2)
        self.assertAlmostEqual(result["item_change_rate"], 2 / 3)
        self.assertAlmostEqual(result["digit_change_rate"], 2 / 6)
        np.testing.assert_allclose(result["per_position_change_rate"], [1 / 3, 1 / 3])

    def test_no_change(self):
        codes = np.array([[0, 1], [1, 0]])
        result = diagnostic.compare_codes(codes, codes.copy())
        self.assertEqual(result["changed_items"], 0)
        self.assertEqual(result["changed_digits"], 0)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            diagnostic.compare_codes(np.zeros((2, 4)), np.zeros((3, 4)))


class ValidationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.catalog = list("ABCDEF")
        self.sequences = {
            "too-short": ["A"],
            "length-two": ["A", "B"],
            "length-three": ["A", "B", "C"],
            "length-four": ["B", "C", "D", "E"],
            "length-six": ["A", "B", "C", "D", "E", "F"],
        }

    def test_validation_is_second_last_item_not_test_target(self):
        changed = np.array([False, False, False, True, False, False])
        result = diagnostic.validation_coverage(self.sequences, self.catalog, changed, max_history_len=2)
        self.assertEqual(result["examples"], 4)
        self.assertEqual(result["target_changed_examples"], 1)
        self.assertEqual(result["history_changed_examples"], 1)
        self.assertEqual(result["either_changed_examples"], 2)

    def test_history_truncation_and_empty_history_are_preserved(self):
        changed = np.array([True, False, False, False, False, False])
        result = diagnostic.validation_coverage(self.sequences, self.catalog, changed, max_history_len=2)
        self.assertEqual(result["examples"], 4)
        self.assertEqual(result["target_changed_examples"], 1)
        self.assertEqual(result["history_changed_examples"], 1)
        self.assertEqual(result["either_changed_examples"], 2)

    def test_no_changed_items_gives_zero_coverage(self):
        result = diagnostic.validation_coverage(self.sequences, self.catalog, np.zeros(6, dtype=bool))
        self.assertEqual(result["examples"], 4)
        self.assertEqual(result["either_changed_examples"], 0)

    def test_minimum_length_is_applied_to_raw_sequence(self):
        result = diagnostic.validation_coverage(
            self.sequences, self.catalog, np.zeros(6, dtype=bool), min_seq_len=3
        )
        self.assertEqual(result["examples"], 3)

    def test_rejects_unknown_item(self):
        with self.assertRaises(ValueError):
            diagnostic.validation_coverage({"user": ["A", "UNKNOWN", "C"]}, self.catalog, np.zeros(6, dtype=bool))

    def test_rejects_wrong_mask_length(self):
        with self.assertRaises(ValueError):
            diagnostic.validation_coverage(self.sequences, self.catalog, np.zeros(5, dtype=bool))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
