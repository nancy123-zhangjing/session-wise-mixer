"""CPU-only standard-library checks for the paired feedback pilot controls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json"
spec = importlib.util.spec_from_file_location(
    "_diger_feedback_protocol_tests", ROOT / "scripts/checks/diger_feedback_pilot_protocol.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load feedback pilot helper")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class FeedbackProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = helper.load_protocol(CONFIG)

    def test_arm_order_and_fixed_full_budgets(self):
        self.assertEqual(helper.arm_names(self.protocol), ("frozen", "cross_recon", "cross_rec"))
        for arm in helper.arm_names(self.protocol):
            settings = helper.arm_settings(self.protocol, arm)
            self.assertEqual(settings["tokenizer_batches"], 0 if arm == "frozen" else 384)
            self.assertEqual(settings["gradient_accumulation_steps"], 16)
            self.assertEqual(settings["tokenizer_batches"] // 16, 0 if arm == "frozen" else 24)
            self.assertEqual(settings["decoder_epochs"], 3)
            self.assertEqual(settings["decoder_batch_limit"], 0)
            self.assertEqual(settings["eval_batch_limit"], 0)
            self.assertEqual(settings["noise_total_steps"], 60)

    def test_weights_and_actual_noise_controls(self):
        for arm, weight in {"frozen": 1.0, "cross_recon": 0.0, "cross_rec": 1.0}.items():
            settings = helper.arm_settings(self.protocol, arm)
            self.assertEqual(settings["recommendation_weight"], weight)
            self.assertEqual((settings["noise_start"], settings["noise_end"]), (0.0, 0.0))
            self.assertTrue(settings["audit_recommendation_feedback"])

    def test_probe_budgets_are_diagnostic_and_not_selected(self):
        self.assertEqual(helper.arm_settings(self.protocol, "frozen")["tokenizer_probe_updates"], [])
        self.assertEqual(helper.arm_settings(self.protocol, "cross_rec")["tokenizer_probe_updates"], [6, 24])
        for arm in helper.arm_names(self.protocol):
            settings = helper.arm_settings(self.protocol, arm, smoke=True)
            self.assertEqual(settings["tokenizer_batches"], 0 if arm == "frozen" else 16)
            self.assertEqual(settings["tokenizer_probe_updates"], [] if arm == "frozen" else [1])
            self.assertEqual((settings["decoder_batch_limit"], settings["eval_batch_limit"], settings["decoder_epochs"]), (16, 2, 1))

    def test_settings_and_resource_projection_do_not_mutate_protocol(self):
        before = copy.deepcopy(self.protocol)
        settings = helper.arm_settings(self.protocol, "cross_rec")
        settings["tokenizer_probe_updates"].append(99)
        helper.validate_protocol(self.protocol)
        self.assertEqual(self.protocol, before)

    def test_invalid_fixed_controls_are_rejected(self):
        changes = {
            "protocol_id": "stage05_diger_noise_once_freeze_recover_v1",
            "arms": ["cross_rec", "cross_recon", "frozen"],
            "gradient_accumulation_steps": 8,
            "tokenizer_batches": 96,
            "decoder_epochs": 4,
            "tokenizer_schedule_total_updates": 24,
            "decoder_lr": 0.001,
            "decoder_weight_decay": 0.01,
            "assignment_temperature": 0.2,
            "softmax_temperature": 0.5,
            "noise_start": 0.1,
            "noise_end": 0.1,
            "tokenizer_probe_updates": [24],
            "recommendation_weights": {"frozen": 1.0, "cross_recon": 1.0, "cross_rec": 1.0},
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                protocol = copy.deepcopy(self.protocol)
                protocol[key] = value
                with self.assertRaises(ValueError):
                    helper.validate_protocol(protocol)

    def test_invalid_types_and_nonfinite_values_are_rejected(self):
        for key, value in [("noise_start", False), ("softmax_temperature", True),
                           ("decoder_lr", float("nan")), ("tokenizer_batches", 384.0),
                           ("seed", True), ("tokenizer_probe_updates", [6.0, 24])]:
            with self.subTest(key=key):
                protocol = copy.deepcopy(self.protocol)
                protocol[key] = value
                with self.assertRaises(ValueError):
                    helper.validate_protocol(protocol)
        for key in helper.CONTROLLED_TOKENIZER_SETTINGS:
            with self.subTest(controlled=key):
                protocol = copy.deepcopy(self.protocol)
                protocol["controlled_tokenizer_settings"][key] += 1
                with self.assertRaises(ValueError):
                    helper.validate_protocol(protocol)

    def test_smoke_controls_and_unknown_arms_are_rejected(self):
        for key, value in [("tokenizer_batches", 32), ("decoder_epochs", 3),
                           ("tokenizer_probe_updates", [1.0]), ("eval_batches", 0)]:
            with self.subTest(key=key):
                protocol = copy.deepcopy(self.protocol)
                protocol["smoke"][key] = value
                with self.assertRaises(ValueError):
                    helper.validate_protocol(protocol)
        with self.assertRaises(ValueError):
            helper.arm_settings(self.protocol, "C")

    def test_optional_smoke_batch_size_overrides_cannot_change_defaults(self):
        for key in ("train_batch_size", "eval_batch_size"):
            for value in (1, 4, 64, 2.0, True):
                with self.subTest(key=key, value=value):
                    protocol = copy.deepcopy(self.protocol)
                    protocol["smoke"][key] = value
                    with self.assertRaises(ValueError):
                        helper.validate_protocol(protocol)
            protocol = copy.deepcopy(self.protocol)
            protocol["smoke"][key] = 2
            helper.validate_protocol(protocol)

    def test_path_escape_and_hash_formats_are_rejected(self):
        for value in ["../outside.bin", "C:/outside.bin", "C:outside.bin", "/outside.bin", "assets\\model.bin"]:
            with self.subTest(value=value):
                protocol = copy.deepcopy(self.protocol)
                protocol["shared_checkpoint"] = value
                with self.assertRaises(ValueError):
                    helper.validate_protocol(protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol["shared_checkpoint_sha256"] = "0" * 63
        with self.assertRaises(ValueError):
            helper.validate_protocol(protocol)

    def test_seed_mapping_and_partial_accumulation_reuse_common_logic(self):
        seeds = [helper.phase_seed(self.protocol["decoder_seed_base"], "decoder", 0)
                 for _ in helper.arm_names(self.protocol)]
        self.assertEqual(len(set(seeds)), 1)
        self.assertNotEqual(seeds[0], helper.phase_seed(self.protocol["decoder_seed_base"], "decoder", 1))
        self.assertEqual([helper.loss_divisor(i, 18, 16) for i in range(18)], [16] * 16 + [2] * 2)
        with self.assertRaises(ValueError):
            helper.phase_seed(2024, "arm", 0)

    def test_plan_only_is_readonly_and_keeps_feedback_protocol_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([
                sys.executable, "-B", str(ROOT / "scripts/checks/diger_feedback_pilot_protocol.py"),
                "--project-root", str(root), "--config", str(CONFIG), "--plan-only",
            ], check=True, capture_output=True, text=True, encoding="utf-8")
            plan = json.loads(result.stdout)
            self.assertEqual(plan["protocol_id"], helper.PROTOCOL_ID)
            self.assertEqual([arm["arm"] for arm in plan["arms"]], list(helper.ARMS))
            self.assertEqual(plan["asset_verification"], "not_requested")
            self.assertEqual(list(root.iterdir()), [])
        self.assertNotIn("torch", sys.modules)


class FeedbackAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = helper.load_protocol(CONFIG)
        self.protocol.update({
            "shared_checkpoint": "assets/shared.bin", "shared_metadata": "metadata.json",
            "asset_lock": "assets.lock.json", "required_assets": ["assets/codebooks.npy"],
            "config_layers": ["configs/source.yaml"],
        })
        self.write("assets/shared.bin", b"pinned checkpoint")
        self.write("assets/codebooks.npy", b"pinned codebooks")
        self.write("configs/source.yaml", b"joint_gradient_mode: diger_st\n")
        self.protocol["shared_checkpoint_sha256"] = self.digest("assets/shared.bin")
        self.write_json("assets.lock.json", {"artifacts": {
            value: {"sha256": self.digest(value)}
            for value in [self.protocol["shared_checkpoint"], *self.protocol["required_assets"]]
        }})
        self.metadata = {
            "run_id": self.protocol["expected_source_run_id"], "best_epoch": 5,
            "joint_alignment": {"diger_final_schedule_state": {"optimizer_step": 0, "total_optimizer_steps": 60}},
        }
        self.write_json("metadata.json", self.metadata)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def write_json(self, name, value):
        self.write(name, json.dumps(value).encode("utf-8"))

    def digest(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def test_verified_inputs_are_bound_and_unchanged(self):
        before = {path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.root.rglob("*") if path.is_file()}
        protocol_before = copy.deepcopy(self.protocol)
        receipt = helper.verify_assets(self.root, self.protocol)
        self.assertEqual(len(receipt), 5)
        self.assertEqual(self.protocol, protocol_before)
        self.assertEqual(before, {path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.root.rglob("*") if path.is_file()})
        self.assertEqual(receipt[0]["binding"], "stage05_archived_asset_lock")

    def test_wrong_asset_hash_or_checkpoint_binding_is_rejected(self):
        self.write("assets/codebooks.npy", b"modified")
        with self.assertRaisesRegex(RuntimeError, "Input hash mismatch"):
            helper.verify_assets(self.root, self.protocol)
        self.write("assets/codebooks.npy", b"pinned codebooks")
        self.protocol["shared_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding disagrees"):
            helper.verify_assets(self.root, self.protocol)

    def test_updated_or_wrong_source_state_is_rejected(self):
        for field, value in [("optimizer_step", 1), ("total_optimizer_steps", 24)]:
            with self.subTest(field=field):
                metadata = copy.deepcopy(self.metadata)
                metadata["joint_alignment"]["diger_final_schedule_state"][field] = value
                self.write_json("metadata.json", metadata)
                with self.assertRaisesRegex(RuntimeError, "pinned pre-tokenizer"):
                    helper.verify_assets(self.root, self.protocol)
        self.write_json("metadata.json", {**self.metadata, "best_epoch": 12})
        with self.assertRaises(RuntimeError):
            helper.verify_assets(self.root, self.protocol)

    def test_missing_assets_and_config_layers_are_rejected(self):
        (self.root / "configs/source.yaml").unlink()
        with self.assertRaises(FileNotFoundError):
            helper.verify_assets(self.root, self.protocol)

    def test_source_receipt_binds_new_and_reused_execution_sources(self):
        paths = ["scripts/train/run_diger_noise_pilot.py", "scripts/checks/diger_noise_pilot_protocol.py",
                 "scripts/train/run_diger_feedback_pilot.py", "scripts/checks/diger_feedback_pilot_protocol.py",
                 "tests/test_diger_feedback_protocol.py", "tests/test_diger_feedback_runner.py", helper.DEFAULT_CONFIG,
                 "external/DiffGRM/genrec/runtime.py"]
        for path in paths:
            self.write(path, ("# source " + path).encode("utf-8"))
        self.write("external/DiffGRM/genrec/weights.bin", b"not source")
        receipt = helper.source_receipt(self.root)
        self.assertEqual(set(receipt["source_sha256"]), set(paths))
        self.assertEqual(set(receipt["git"]), {"project", "diffgrm"})
        self.assertEqual(receipt["source_sha256"], {path: self.digest(path) for path in paths})
        json.dumps(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
