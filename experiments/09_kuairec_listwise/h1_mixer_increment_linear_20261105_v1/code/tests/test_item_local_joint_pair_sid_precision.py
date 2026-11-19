"""Focused CPU regression for exact shared-mode SID artifact acceptance.

No model construction, saved production data, CUDA calls or training.
python -X utf8 -B tests/test_item_local_joint_pair_sid_precision.py
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_joint_pair_precision", ROOT / "scripts/analysis/audit_item_local_joint_pair.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
torch.set_num_threads(1)


def near_boundary(count=1):
    # Exact distances: approximately .2501^2 versus .2502^2, gap 5e-5.
    # Dot-only ten-bit rounding reverses this pair by about 3.4e-5.
    slots, books = torch.zeros(count, 1, 64), torch.zeros(1, 3, 64)
    slots[:, 0, 0] = .75
    books[0, :, 0] = torch.tensor([1.0001, .4998, 3.0])
    return slots, books


class SIDPrecisionAcceptance(unittest.TestCase):
    def test_rounding_hand_specified_bits(self):
        cases = [
            (0x00000000, 0x00000000, 0x00000000),
            (0x80000000, 0x80000000, 0x80000000),
            (0x3F800000, 0x3F800000, 0x3F800000),
            (0xBF800000, 0xBF800000, 0xBF800000),
            (0x3F800FFF, 0x3F800000, 0x3F800000),
            (0xBF800FFF, 0xBF800000, 0xBF800000),
            (0x3F801001, 0x3F802000, 0x3F802000),
            (0xBF801001, 0xBF802000, 0xBF802000),
            (0x3F801000, 0x3F800000, 0x3F802000),
            (0xBF801000, 0xBF800000, 0xBF802000),
            (0x3F803000, 0x3F804000, 0x3F804000),
            (0xBF803000, 0xBF804000, 0xBF804000),
            (0x3FA66666, 0x3FA66000, 0x3FA66000),
            (0x00001000, 0x00000000, 0x00002000),
            (0x80001000, 0x80000000, 0x80002000),
        ]
        bits = np.array([row[0] for row in cases], dtype=np.uint32)
        original = bits.tobytes()
        for mode, index in (("nearest_even", 1), ("nearest_ties_away", 2)):
            with self.subTest(mode=mode):
                values = torch.from_numpy(bits.view(np.float32))
                rounded = audit.round_tf32_operand(values, mode)
                np.testing.assert_array_equal(rounded.numpy().view(np.uint32), np.array([row[index] for row in cases], dtype=np.uint32))
                self.assertTrue(torch.equal(audit.round_tf32_operand(rounded, mode).view(torch.int32), rounded.view(torch.int32)))
                self.assertEqual(bits.tobytes(), original)

    def test_strict_fp32_full_catalogue_accepted(self):
        slots, books = near_boundary(3)
        report = audit.audit_sid_precision(slots, books, torch.zeros(3, 1, dtype=torch.int64), chunk_size=2)
        self.assertEqual(report["matching_modes"], ["strict_fp32_cpu"])
        self.assertEqual(report["code_positions"], 3)
        self.assertEqual(report["strict_fp64_reference"]["old_tolerance_failures"], 0)
        self.assertEqual(report["modes"]["tf32_nearest_even_cpu"]["argmin_mismatches"], 3)

    def test_quantified_tf32_boundary_and_shared_mode_accepted(self):
        slots, books = near_boundary()
        report = audit.audit_sid_precision(slots, books, torch.ones(1, 1, dtype=torch.int64))
        expected = ["tf32_nearest_even_cpu", "tf32_nearest_ties_away_cpu"]
        self.assertEqual(report["matching_modes"], expected)
        self.assertEqual(report["modes"]["strict_fp32_cpu"]["argmin_mismatches"], 1)
        high = report["strict_fp64_reference"]
        self.assertEqual(high["old_tolerance_failures"], 1)
        self.assertGreater(high["maximum_saved_distance_gap"], 4.99e-5)
        self.assertLess(high["maximum_saved_distance_gap"], 5.01e-5)
        margin = report["modes"][expected[0]]["minimum_top2_margin"]
        self.assertGreater(margin, 3e-5)
        self.assertLess(margin, 4e-5)
        reports = {arm + "/" + kind: report for arm in audit.ARMS for kind in ("best", "last")}
        self.assertEqual(audit.common_sid_precision_modes(reports), expected)

    def test_corrupted_sid_rejected(self):
        slots, books = near_boundary()
        with self.assertRaisesRegex(ValueError, "No single exact full-catalogue"):
            audit.audit_sid_precision(slots, books, torch.full((1, 1), 2, dtype=torch.int64))

    def test_per_position_precision_mixture_rejected(self):
        slots, books = near_boundary(2)
        # Each row is legal under some arithmetic, but no one mode explains both.
        with self.assertRaisesRegex(ValueError, "No single exact full-catalogue"):
            audit.audit_sid_precision(slots, books, torch.tensor([[0], [1]], dtype=torch.int64), chunk_size=1)

    def test_four_states_without_common_precision_rejected(self):
        slots, books = near_boundary()
        strict = audit.audit_sid_precision(slots, books, torch.zeros(1, 1, dtype=torch.int64))
        reduced = audit.audit_sid_precision(slots, books, torch.ones(1, 1, dtype=torch.int64))
        reports = {"control/best": strict, "control/last": strict, "item_local/best": reduced, "item_local/last": reduced}
        with self.assertRaisesRegex(ValueError, "no common exact SID arithmetic mode"):
            audit.common_sid_precision_modes(reports)


if __name__ == "__main__":
    unittest.main(verbosity=2)
