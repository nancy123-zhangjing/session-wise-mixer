"""Standard-library tests for the shared-start DIGER noise pilot protocol.

These checks need neither PyTorch nor a GPU and never load checkpoint tensors.
Run from the project root with ``python -B tests/test_diger_noise_pilot.py``.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json"


def load_helper():
    path = ROOT / "scripts/checks/diger_noise_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("_diger_noise_pilot_protocol_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pilot helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol_helper = load_helper()


def load_runner():
    path = ROOT / "scripts/train/run_diger_noise_pilot.py"
    spec = importlib.util.spec_from_file_location("_diger_noise_pilot_runner_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot_runner = load_runner()


class FakeModule:
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        if hasattr(self, "joint_alignment"):
            self.joint_alignment.train(mode)
        return self

    def eval(self):
        return self.train(False)


class FakeAlignment(FakeModule):
    def __init__(self, noise):
        super().__init__()
        self.noise = noise

    def diger_schedule_state(self):
        return {"gumbel_noise_scale": self.noise}

    def diger_assignments(self, item_ids, stochastic=True):
        return (item_ids, {"stochastic": stochastic})


class PhaseModeGuardTests(unittest.TestCase):
    def test_tokenizer_mode_is_restored_after_recursive_eval(self):
        model = FakeModule()
        model.joint_alignment = FakeAlignment(1.0)
        for _ in range(3):
            pilot_runner.restore_phase_mode(model, "tokenizer")
            self.assertFalse(model.training)
            self.assertTrue(model.joint_alignment.training)
            model.eval()  # emulate upstream guided scoring

    def test_recommender_mode_keeps_frozen_tokenizer_in_eval(self):
        model = FakeModule()
        model.joint_alignment = FakeAlignment(1.0)
        pilot_runner.restore_phase_mode(model, "recommender")
        self.assertTrue(model.training)
        self.assertFalse(model.joint_alignment.training)

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot_runner.restore_phase_mode(FakeModule(), "bad")

    def test_assignment_counts_distinguish_noise_from_schedule_zero(self):
        original_torch = getattr(pilot_runner, "torch", None)
        pilot_runner.torch = SimpleNamespace(random=SimpleNamespace(
            fork_rng=lambda **kwargs: nullcontext(),
            default_generator=SimpleNamespace(manual_seed=lambda seed: None),
        ))
        try:
            for noise in (0.0, 1.0):
                alignment = FakeAlignment(noise)
                guard = pilot_runner.IsolatedGumbel(alignment, SimpleNamespace(type="cpu"), 123)
                with guard:
                    alignment.diger_assignments(1)
                    alignment.diger_assignments(2)
                self.assertEqual(guard.calls, 2)
                self.assertEqual(guard.training_enabled_calls, 2)
                self.assertEqual(guard.noise_applied_calls, 2 if noise else 0)
                self.assertNotIn("diger_assignments", alignment.__dict__)
        finally:
            pilot_runner.torch = original_torch

    def test_eval_assignment_fails_before_silently_skipping_noise(self):
        alignment = FakeAlignment(1.0).eval()
        guard = pilot_runner.IsolatedGumbel(alignment, SimpleNamespace(type="cpu"), 123)
        with guard, self.assertRaisesRegex(RuntimeError, "silently skipped"):
            alignment.diger_assignments(1)
        self.assertEqual(guard.calls, 0)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol_helper.load_protocol(CONFIG)

    def test_repository_protocol_is_valid_and_json_serializable(self):
        protocol_helper.validate_protocol(self.protocol)
        json.dumps(self.protocol)

    def test_normal_arms_have_matching_decoder_budget(self):
        settings = {
            arm: protocol_helper.arm_settings(self.protocol, arm)
            for arm in ("A", "B", "C")
        }
        for arm, row in settings.items():
            with self.subTest(arm=arm):
                self.assertEqual(row["decoder_epochs"], 3)
                self.assertEqual(row["gradient_accumulation_steps"], 16)
                self.assertGreater(row["decoder_lr"], 0)
                for key in (
                    "decoder_epochs", "decoder_batch_limit", "eval_batch_limit",
                    "decoder_lr", "gradient_accumulation_steps",
                ):
                    self.assertEqual(row[key], settings["A"][key])
        self.assertEqual(settings["A"]["tokenizer_batches"], 0)
        self.assertEqual(settings["B"]["tokenizer_batches"], 96)
        self.assertEqual(settings["C"]["tokenizer_batches"], 96)

    def test_b_has_zero_noise_while_c_preserves_historical_horizon(self):
        b = protocol_helper.arm_settings(self.protocol, "B")
        c = protocol_helper.arm_settings(self.protocol, "C")
        self.assertEqual(b["noise_start"], 0.0)
        self.assertEqual(b["noise_end"], 0.0)
        self.assertEqual(c["noise_start"], 1.0)
        self.assertEqual(c["noise_end"], 0.2)
        self.assertEqual(c["noise_total_steps"], 60)
        self.assertNotEqual(c["noise_total_steps"], 96 // 16)

    def test_smoke_bounds_the_work_without_changing_decoder_controls(self):
        settings = {
            arm: protocol_helper.arm_settings(self.protocol, arm, smoke=True)
            for arm in ("A", "B", "C")
        }
        for arm, row in settings.items():
            with self.subTest(arm=arm):
                self.assertEqual(row["decoder_epochs"], 1)
                self.assertEqual(row["decoder_batch_limit"], 16)
                self.assertEqual(row["eval_batch_limit"], 2)
                self.assertEqual(row["gradient_accumulation_steps"], 16)
                self.assertEqual(row["decoder_lr"], settings["A"]["decoder_lr"])
        self.assertEqual(settings["A"]["tokenizer_batches"], 0)
        self.assertEqual(settings["B"]["tokenizer_batches"], 16)
        self.assertEqual(settings["C"]["tokenizer_batches"], 16)
        self.assertEqual(settings["C"]["noise_total_steps"], 60)

    def test_arm_settings_does_not_mutate_protocol(self):
        original = copy.deepcopy(self.protocol)
        for arm in ("A", "B", "C"):
            protocol_helper.arm_settings(self.protocol, arm)
            protocol_helper.arm_settings(self.protocol, arm, smoke=True)
        self.assertEqual(self.protocol, original)

    def test_unknown_arm_is_rejected(self):
        for arm in ("D", "", "all", "A/B/C"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                protocol_helper.arm_settings(self.protocol, arm)

    def test_malformed_json_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory(prefix="diger-pilot-json-") as temp:
            path = Path(temp) / "broken.json"
            path.write_text("{not valid JSON", encoding="utf-8")
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                protocol_helper.load_protocol(path)

    def test_missing_protocol_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="diger-pilot-json-") as temp:
            with self.assertRaises(FileNotFoundError):
                protocol_helper.load_protocol(Path(temp) / "missing.json")

    def test_shared_start_cannot_silently_move_past_tokenizer_update(self):
        for key, value in (("expected_best_epoch", 12), ("expected_tokenizer_updates", 1)):
            candidate = copy.deepcopy(self.protocol)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol_helper.validate_protocol(candidate)

    def test_noise_horizon_cannot_finish_before_one_tokenizer_phase(self):
        candidate = copy.deepcopy(self.protocol)
        candidate["tokenizer_schedule_total_updates"] = 5
        with self.assertRaises(ValueError):
            protocol_helper.validate_protocol(candidate)

    def test_absolute_or_traversing_input_paths_are_rejected(self):
        for value in ("/tmp/weights.bin", "C:/weights.bin", "../weights.bin", r"data\weights.bin"):
            candidate = copy.deepcopy(self.protocol)
            candidate["shared_checkpoint"] = value
            with self.subTest(path=value), self.assertRaises(ValueError):
                protocol_helper.validate_protocol(candidate)

    def test_duplicate_asset_paths_are_rejected(self):
        candidate = copy.deepcopy(self.protocol)
        candidate["required_assets"].append(candidate["required_assets"][0])
        with self.assertRaises(ValueError):
            protocol_helper.validate_protocol(candidate)

    def test_checkpoint_hash_must_be_an_explicit_sha256(self):
        for value in ("", "abc", "g" * 64, "0" * 63):
            candidate = copy.deepcopy(self.protocol)
            candidate["shared_checkpoint_sha256"] = value
            with self.subTest(hash=value), self.assertRaises(ValueError):
                protocol_helper.validate_protocol(candidate)

    def test_nonfinite_loss_temperatures_are_rejected(self):
        for key in ("assignment_temperature", "softmax_temperature", "decoder_lr"):
            for value in (float("nan"), float("inf"), 0.0):
                candidate = copy.deepcopy(self.protocol)
                candidate[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol_helper.validate_protocol(candidate)


class PhaseSeedTests(unittest.TestCase):
    def test_same_phase_and_index_reproduce_seed(self):
        for kind in ("tokenizer", "decoder"):
            for index in (0, 1, 2):
                with self.subTest(kind=kind, index=index):
                    left = protocol_helper.phase_seed(2024, kind, index)
                    right = protocol_helper.phase_seed(2024, kind, index)
                    self.assertIsInstance(left, int)
                    self.assertEqual(left, right)

    def test_arms_share_phase_seeds_by_construction(self):
        seeds = {
            arm: [protocol_helper.phase_seed(2024, "decoder", i) for i in range(3)]
            for arm in ("A", "B", "C")
        }
        self.assertEqual(seeds["A"], seeds["B"])
        self.assertEqual(seeds["B"], seeds["C"])

    def test_seed_phase_and_epoch_each_change_the_stream(self):
        seeds = {
            protocol_helper.phase_seed(seed, kind, index)
            for seed in (2024, 2025)
            for kind in ("tokenizer", "decoder")
            for index in (0, 1, 2)
        }
        self.assertEqual(len(seeds), 12)

    def test_unknown_stream_and_invalid_indices_are_rejected(self):
        for arguments in ((2024, "unknown", 0), (2024, "decoder", -1), (-1, "decoder", 0), (2024, "decoder", 0.5)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                protocol_helper.phase_seed(*arguments)


class AccumulationTests(unittest.TestCase):
    def test_full_windows_use_normal_divisor(self):
        for index in range(32):
            with self.subTest(index=index):
                self.assertEqual(protocol_helper.loss_divisor(index, 32, 16), 16)

    def test_partial_final_window_uses_actual_batch_count(self):
        expected = [16] * 16 + [2] * 2
        actual = [protocol_helper.loss_divisor(i, 18, 16) for i in range(18)]
        self.assertEqual(actual, expected)

    def test_epoch_shorter_than_accumulation_uses_all_actual_batches(self):
        for index in range(5):
            self.assertEqual(protocol_helper.loss_divisor(index, 5, 16), 5)

    def test_single_batch_is_not_scaled_by_full_accumulation(self):
        self.assertEqual(protocol_helper.loss_divisor(0, 1, 16), 1)

    def test_every_optimizer_window_receives_unit_total_weight(self):
        for total in (1, 5, 16, 18, 31, 32, 96, 1652):
            with self.subTest(total=total):
                for start in range(0, total, 16):
                    stop = min(start + 16, total)
                    weight = sum(
                        1.0 / protocol_helper.loss_divisor(i, total, 16)
                        for i in range(start, stop)
                    )
                    self.assertAlmostEqual(weight, 1.0)

    def test_invalid_window_arguments_are_rejected(self):
        for arguments in ((-1, 16, 16), (16, 16, 16), (0, 0, 16), (0, -1, 16), (0, 16, 0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                protocol_helper.loss_divisor(*arguments)


class AssetBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="diger-pilot-assets-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.protocol = protocol_helper.load_protocol(CONFIG)
        self.protocol.update({
            "shared_checkpoint": "weights/model.bin",
            "shared_metadata": "metadata.json",
            "asset_lock": "assets.lock.json",
            "required_assets": ["data/item-input.bin"],
            "config_layers": ["configs/source.yaml"],
        })
        self.write("weights/model.bin", b"bound checkpoint fixture")
        self.write("data/item-input.bin", b"bound tokenizer input fixture")
        self.write("configs/source.yaml", b"seed: 2024\n")
        self.metadata = {
            "run_id": self.protocol["expected_source_run_id"],
            "best_epoch": 5,
            "joint_alignment": {"diger_final_schedule_state": {
                "optimizer_step": 0,
                "total_optimizer_steps": 60,
            }},
        }
        self.write_json("metadata.json", self.metadata)
        self.lock = {"artifacts": {}}
        for value in [self.protocol["shared_checkpoint"], *self.protocol["required_assets"]]:
            self.lock["artifacts"][value] = {"sha256": self.digest(self.root / value)}
        self.protocol["shared_checkpoint_sha256"] = self.lock["artifacts"]["weights/model.bin"]["sha256"]
        self.write_json("assets.lock.json", self.lock)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_json(self, relative, value):
        self.write(relative, json.dumps(value).encode("utf-8"))

    def test_valid_bindings_return_input_and_source_receipts(self):
        before = copy.deepcopy(self.protocol)
        receipts = protocol_helper.verify_assets(self.root, self.protocol)
        by_path = {row["path"]: row for row in receipts}
        expected = {"weights/model.bin", "data/item-input.bin", "metadata.json", "assets.lock.json", "configs/source.yaml"}
        self.assertEqual(set(by_path), expected)
        for path, row in by_path.items():
            with self.subTest(path=path):
                self.assertEqual(row["sha256"], self.digest(self.root / path))
                self.assertEqual(row["size_bytes"], (self.root / path).stat().st_size)
        self.assertEqual(by_path["weights/model.bin"]["binding"], "stage05_archived_asset_lock")
        self.assertEqual(by_path["configs/source.yaml"]["binding"], "current_protocol_source")
        self.assertEqual(self.protocol, before)

    def test_changed_checkpoint_content_is_rejected(self):
        self.write("weights/model.bin", b"other checkpoint")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_changed_tokenizer_input_content_is_rejected(self):
        self.write("data/item-input.bin", b"other tokenizer")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_checkpoint_lock_must_agree_with_explicit_protocol_hash(self):
        self.protocol["shared_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding disagrees"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_missing_input_is_rejected(self):
        (self.root / "data/item-input.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_missing_archive_hash_is_rejected(self):
        self.lock["artifacts"].pop("data/item-input.bin")
        self.write_json("assets.lock.json", self.lock)
        with self.assertRaises(ValueError):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_wrong_epoch_metadata_is_rejected(self):
        self.metadata["best_epoch"] = 12
        self.write_json("metadata.json", self.metadata)
        with self.assertRaisesRegex(RuntimeError, "pinned pre-tokenizer"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_post_update_metadata_is_rejected(self):
        self.metadata["joint_alignment"]["diger_final_schedule_state"]["optimizer_step"] = 1
        self.write_json("metadata.json", self.metadata)
        with self.assertRaisesRegex(RuntimeError, "pinned pre-tokenizer"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_wrong_schedule_metadata_is_rejected(self):
        self.metadata["joint_alignment"]["diger_final_schedule_state"]["total_optimizer_steps"] = 6
        self.write_json("metadata.json", self.metadata)
        with self.assertRaisesRegex(RuntimeError, "pinned pre-tokenizer"):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_missing_config_layer_is_rejected(self):
        (self.root / "configs/source.yaml").unlink()
        with self.assertRaises(FileNotFoundError):
            protocol_helper.verify_assets(self.root, self.protocol)

    def test_receipt_records_engine_source_without_hashing_weight_files(self):
        self.write("external/DiffGRM/genrec/runtime.py", b"# test engine source\n")
        self.write("external/DiffGRM/genrec/model.bin", b"not source")
        receipt = protocol_helper.source_receipt(self.root)
        self.assertEqual(set(receipt["git"]), {"project", "diffgrm"})
        self.assertEqual(receipt["source_sha256"], {
            "external/DiffGRM/genrec/runtime.py": self.digest(self.root / "external/DiffGRM/genrec/runtime.py"),
        })
        json.dumps(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
