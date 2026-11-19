"""Protocol identity and actual-gradient/recovery failure regressions for stage05."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/checks"))
import diger_mixer_long_protocol as protocol

spec = importlib.util.spec_from_file_location("long_runner", ROOT / "scripts/train/run_diger_mixer_long.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
wrapper_spec = importlib.util.spec_from_file_location("long_wrapper", ROOT / "scripts/train/mixer_long_gpu_job.py")
wrapper = importlib.util.module_from_spec(wrapper_spec)
wrapper_spec.loader.exec_module(wrapper)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.value = protocol.read(protocol.DEFAULT_CONFIG)

    def test_pre05_not_epoch5_or_short_twenty_four_update_budget(self):
        for key, value in (("initial_decoder", "external/DiffGRM/saved/AmazonReviews2014_Sep-01-2026_14-24-52/pytorch_model.bin"),
                           ("tokenizer_batches_per_phase", 384), ("decoder_epochs", 30),
                           ("noise_start", 0), ("mixer_scope", "within_only"), ("seed", 2025)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.value)
                changed[key] = value
                with self.assertRaises(ValueError):
                    protocol.validate(changed)

    def test_actual_optimizer_clocks_and_no_extra_frozen_epochs(self):
        for arm in protocol.ARMS:
            plan = protocol.phases(self.value, arm)
            self.assertEqual([row["phase"] for row in plan[:5]], ["recommender"] * 5)
            self.assertEqual(plan[-1]["decoder_epochs"], 35)
            self.assertEqual(sum(row["optimizer_updates"] for row in plan if row["phase"] == "recommender"), 3640)
            self.assertEqual(plan[-1]["tokenizer_updates"], 60 if arm.endswith("_joint") else 0)
            self.assertEqual(len(plan), 45 if arm.endswith("_joint") else 35)
        joint = protocol.phases(self.value, "mixer_joint")
        self.assertEqual([(row["phase"], row["logical_index"]) for row in joint[5:9]],
                         [("tokenizer", 0), ("recommender", 6), ("recommender", 7), ("recommender", 8)])

    def test_noise_does_not_reset_each_cycle_or_decay_in_first_six_updates(self):
        self.assertEqual(protocol.noise_scale(0, self.value), 1)
        self.assertGreater(protocol.noise_scale(6, self.value), .98)
        self.assertAlmostEqual(protocol.noise_scale(30, self.value), .6)
        self.assertAlmostEqual(protocol.noise_scale(60, self.value), .2)
        self.assertEqual(sum(2 * row["batches"] for row in protocol.phases(self.value, "mixer_joint")
                             if row["phase"] == "tokenizer"), 1920)

    def test_recommender_clock_ignores_tokenizer_updates(self):
        self.assertAlmostEqual(runner.scheduler_factor(0, self.value), 1 / 500)
        self.assertAlmostEqual(runner.scheduler_factor(500, self.value), 1)
        self.assertAlmostEqual(runner.scheduler_factor(3640, self.value), 0)

    def test_input_copy_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source.bin", root / "new/input.bin"
            source.write_bytes(b"canonical offline state")
            digest = protocol.sha(source)
            protocol.copy_bound(source, destination, digest)
            self.assertEqual(protocol.sha(destination), digest)
            destination.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                protocol.copy_bound(source, destination, digest)
            with self.assertRaises(ValueError):
                protocol.relative(root, "../other/input.bin")


class TensorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.torch, cls.np = torch, np
        cls.runtime = runner.module("_long_test_runtime", ROOT / "scripts/train/run_diger_noise_pilot.py")
        cls.runtime.torch, cls.runtime.np = torch, np

    def model(self):
        torch = self.torch
        class Alignment(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = torch.nn.Linear(3, 3)
                self.mixer.mixer_enabled = True
                self.register_buffer("hard_codes", torch.zeros(3, 4, dtype=torch.long))
            def set_recommender_cache(self, enabled):
                self.cache_enabled = enabled
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = torch.nn.Linear(3, 2)
                self.other = torch.nn.Linear(2, 1)
                self.joint_alignment = Alignment()
                self.config = {}
            def sync_joint_tokenizer_mapping(self):
                self.mapping_synced = True
        return Model()

    def test_partial_missing_gradient_is_rejected_even_with_nonzero_other_gradient(self):
        model = self.model()
        optimizer = self.torch.optim.AdamW(list(model.core.parameters()) + list(model.other.parameters()), lr=.01)
        context = dict(pipeline=types.SimpleNamespace(model=model), deadline=10**12)
        model.core(self.torch.randn(4, 3)).sum().backward()
        self.assertGreater(float(model.core.weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "no gradient"):
            with runner.PhaseAudit(context, optimizer, "tokenizer", None):
                optimizer.step()

    def test_actual_update_and_complete_adam_states_are_checked(self):
        model = self.model()
        optimizer = self.torch.optim.AdamW(list(model.core.parameters()) + list(model.other.parameters()), lr=.01)
        context = dict(pipeline=types.SimpleNamespace(model=model), deadline=10**12)
        model.other(model.core(self.torch.randn(4, 3))).square().sum().backward()
        with runner.PhaseAudit(context, optimizer, "tokenizer", None) as audit:
            optimizer.step()
        self.assertEqual(audit.steps, 1)
        self.assertTrue(audit.evidence["all_expected_Adam_states_present"])
        self.assertEqual(audit.evidence["expected_parameter_count"], 4)

    def test_complete_restore_includes_optimizer_schedule_rng_and_anchor_identity(self):
        torch = self.torch
        model = self.model()
        optimizer = torch.optim.AdamW(model.core.parameters(), lr=.01)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        model.core(torch.randn(4, 3)).square().sum().backward()
        optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
        value = protocol.read(protocol.DEFAULT_CONFIG)
        context = dict(pipeline=types.SimpleNamespace(model=model), identity={"source": "locked"}, mode="smoke",
                       protocol=value, runtime=self.runtime, device=torch.device("cpu"))
        optim = {"recommender": optimizer, "tokenizer": None}
        saved = copy.deepcopy(runner.payload(context, "mixer_frozen", optim, schedule, {"next_stage": 3}))
        before_rng = runner.digest(saved["rng"])
        model.core.weight.data.zero_()
        random.random(); self.np.random.rand(); torch.rand(7)
        progress = runner.restore(context, "mixer_frozen", saved, optim, schedule)
        self.assertEqual(progress, {"next_stage": 3})
        self.assertEqual(runner.digest(self.runtime.phase_rng_state(torch.device("cpu"))), before_rng)
        self.assertTrue(model.mapping_synced)
        wrong = copy.deepcopy(saved)
        wrong["E0_array_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "anchor"):
            runner.restore(context, "mixer_frozen", wrong, optim, schedule)

    def test_replayed_update_does_not_mutate_the_saved_cpu_optimizer_payload(self):
        torch = self.torch
        model = self.model()
        optimizer = torch.optim.AdamW(model.core.parameters(), lr=.01)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        model.core(torch.randn(4, 3)).square().sum().backward()
        optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
        context = dict(pipeline=types.SimpleNamespace(model=model), identity={"source": "locked"}, mode="smoke",
                       protocol=protocol.read(protocol.DEFAULT_CONFIG), runtime=self.runtime, device=torch.device("cpu"))
        optim = {"recommender": optimizer, "tokenizer": None}
        saved = copy.deepcopy(runner.payload(context, "mixer_frozen", optim, schedule, {"next_stage": 1}))
        protected = runner.digest(saved)
        repeated = []
        for _ in range(2):
            runner.restore(context, "mixer_frozen", saved, optim, schedule)
            model.core(torch.randn(4, 3)).square().sum().backward()
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
            repeated.append(runner.digest(runner.payload(context, "mixer_frozen", optim, schedule, {"next_stage": 2})))
            self.assertEqual(runner.digest(saved), protected, "Restored Adam tensors alias the immutable CPU payload")
        self.assertEqual(repeated[0], repeated[1])


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.ledger = dict(campaign_id=protocol.CAMPAIGN, platform_prior_allocation_verified=True,
                           allocated_gpu_seconds=0, acceptance_allocated_gpu_seconds=0, prior_jobs=[])

    def test_failed_acceptance_counts_toward_both_caps(self):
        wrapper.budget(self.ledger, 1500, 1800, "smoke")
        self.ledger.update(allocated_gpu_seconds=200, acceptance_allocated_gpu_seconds=200)
        with self.assertRaisesRegex(ValueError, "acceptance"):
            wrapper.budget(self.ledger, 1500, 1800, "smoke")
        wrapper.budget(self.ledger, 1400, 1600, "smoke")
        self.ledger["allocated_gpu_seconds"] = 42000
        with self.assertRaisesRegex(ValueError, "12 GPU"):
            wrapper.budget(self.ledger, 1400, 1600, "formal")

    def test_cannot_overlap_unreleased_previous_job_or_omit_overhead(self):
        with self.assertRaisesRegex(ValueError, "150 seconds"):
            wrapper.budget(self.ledger, 1500, 1500, "smoke")
        self.ledger["prior_jobs"] = [dict(released=False)]
        with self.assertRaisesRegex(ValueError, "released"):
            wrapper.budget(self.ledger, 1500, 1800, "smoke")

    def test_parallel_reservations_share_the_same_total_budget(self):
        self.ledger["reservations"] = [dict(attempt="arm_" + str(index), mode="train_arm", arm=arm,
                                               platform_cap_seconds=9000) for index, arm in enumerate(protocol.ARMS)]
        wrapper.budget(self.ledger, 8800, 9000, "train_arm", attempt="arm_0", arm=protocol.ARMS[0])
        self.ledger["allocated_gpu_seconds"] = 8000
        with self.assertRaisesRegex(ValueError, "12 GPU"):
            wrapper.budget(self.ledger, 8800, 9000, "train_arm", attempt="arm_0", arm=protocol.ARMS[0])
        self.ledger["allocated_gpu_seconds"] = 0
        self.ledger["reservations"][1]["arm"] = protocol.ARMS[0]
        with self.assertRaisesRegex(ValueError, "distinct training"):
            wrapper.budget(self.ledger, 8800, 9000, "train_arm", attempt="arm_0", arm=protocol.ARMS[0])

    def test_parallel_job_cannot_omit_or_change_its_reservation(self):
        with self.assertRaisesRegex(ValueError, "reservations"):
            wrapper.budget(self.ledger, 8800, 9000, "train_arm", attempt="arm_0", arm=protocol.ARMS[0])
        self.ledger["reservations"] = [dict(attempt="arm_0", mode="train_arm", arm=protocol.ARMS[0], platform_cap_seconds=9000)]
        with self.assertRaisesRegex(ValueError, "reservation differs"):
            wrapper.budget(self.ledger, 8800, 9100, "train_arm", attempt="arm_0", arm=protocol.ARMS[0])


@unittest.skipIf(os.name == "nt", "Real shared-filesystem lock gate runs on the Linux CPU bridge")
class ProcessLockTests(unittest.TestCase):
    def test_distinct_arms_overlap_but_duplicate_and_evaluation_wait_for_release(self):
        code = ("import importlib.util,sys; from pathlib import Path; "
                "s=importlib.util.spec_from_file_location('r',sys.argv[1]); "
                "r=importlib.util.module_from_spec(s); s.loader.exec_module(r); "
                "lock=r.execution_lock(Path(sys.argv[2]),sys.argv[3]); lock.__enter__(); "
                "print('LOCKED',flush=True); sys.stdin.readline(); lock.__exit__(None,None,None)")
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-B", "-c", code, str(ROOT / "scripts/train/run_diger_mixer_long.py"), temporary]
            holder = subprocess.Popen(command + [protocol.ARMS[0]], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "LOCKED")
                other = subprocess.run(command + [protocol.ARMS[1]], input="", text=True, capture_output=True, timeout=15)
                self.assertEqual(other.returncode, 0, other.stderr)
                for scope in (protocol.ARMS[0], "all"):
                    blocked = subprocess.run(command + [scope], input="", text=True, capture_output=True, timeout=15)
                    self.assertNotEqual(blocked.returncode, 0)
                    self.assertIn("already held", blocked.stderr)
            finally:
                holder.kill()
                holder.communicate(timeout=15)
            recovered = subprocess.run(command + ["all"], input="", text=True, capture_output=True, timeout=15)
            self.assertEqual(recovered.returncode, 0, recovered.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
