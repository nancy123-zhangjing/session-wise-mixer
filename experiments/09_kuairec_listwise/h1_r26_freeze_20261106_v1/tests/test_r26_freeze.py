"""Tiny CPU models only: forks, schedules, frozen state and exact resume."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/tests")]
import torch
from test_listwise_increment import tiny
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.recovery_increment import IncrementController
from scripts.listwise_diffgrm.recovery_stability import FollowupController, effective_parent
from scripts.listwise_diffgrm.recovery_stability_protocol import (
    ARMS, R26_ALPHA, coefficient, load_spec, normalize_boundary, specification,
)
from scripts.listwise_diffgrm.recovery_stability_run import evaluation_event, select_best
from scripts.listwise_diffgrm.training import cpu_copy, seed_all


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        with redirect_stdout(io.StringIO()):
            trainer, batches = tiny(cls.directory)
        binding = dict(config=copy.deepcopy(trainer.config), asset_manifest="fixture", data={}, engine={}, code={})
        for phase, row in enumerate(trainer.plan[:25]):
            seed_all(2024 + 100003 * (phase + 1))
            for _ in range(row["updates"]):
                trainer.update(batches, row["kind"])
            if row["kind"] == "T":
                trainer.refresh_catalog()
        original = trainer.checkpoint(binding, dict(phase=25, update=0), "H1")
        linear = IncrementController(trainer, binding, binding, "tiny-original")
        linear.fork(original)
        while linear.progress["phase"] < 33:
            row = linear.row
            for _ in range(row["updates"]):
                linear.step(batches, row["kind"])
            linear.prepare_boundary()
            if linear.progress["phase"] == 32:
                cls.increment_parent = linear.checkpoint()  # Pending completed R26 boundary.
            linear.complete_phase()
        with redirect_stdout(io.StringIO()):
            release_trainer, _ = tiny(cls.directory)
        release_trainer.restore(original, binding, arm="H1")
        mixer = release_trainer.alignment.mixer
        release_trainer.alignment.eval()
        with torch.no_grad():
            raw, _ = mixer.crossed_codebooks(False)
            correction = raw - mixer.base_codebooks
        cls.release_parent = dict(inner=copy.deepcopy(original), protocol=dict(policy="release1", gamma=1.0),
            release1=cpu_copy(dict(raw_M=correction, effective_M=0.3 * correction,
                                 C=mixer.base_codebooks + 0.3 * correction,
                                 parent_sha256="aa565c3bf6daa2809ff1d30230d07aa57468a1286da1fe3eeb177a83380228ea")))
        cls.release_parent["inner"]["progress"] = dict(phase=25, update=0, boundary_prepared=False)
        cls.release_parent["inner"]["arm"] = "H1-release1"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def controller(self, arm):
        spec = specification(arm)
        parent = self.increment_parent if spec["parent"] == "increment_r26" else self.release_parent
        with redirect_stdout(io.StringIO()):
            trainer, batches = tiny(self.directory)
        binding = parent["inner"]["binding"]
        ctrl = FollowupController(trainer, binding, binding, "synthetic-parent", spec)
        ctrl.fork_parent(parent)
        return ctrl, batches, parent

    def test_fork_preserves_parent_models_optimizer_rng_and_effective_codes(self):
        for arm in ARMS:
            ctrl, _, parent = self.controller(arm)
            result = ctrl.checkpoint()
            for key in ("model", "alignment", "r_optimizer", "t_optimizer", "scheduler", "rng"):
                self.assertEqual(fingerprint(parent["inner"][key]), fingerprint(result["inner"][key]), (arm, key))
            with torch.no_grad():
                self.assertTrue(torch.equal(ctrl.gate.c0, ctrl.trainer.alignment.mixer.crossed_codebooks(False)[0]))
            if arm.startswith("release"):
                raw, _ = ctrl.gate.original(False)
                self.assertFalse(torch.allclose(raw, ctrl.gate.c0))  # No accidental ungating.
            old = fingerprint(ctrl.gate.state())
            ctrl.trainer.alignment.train()
            ctrl.trainer.alignment.mixer.crossed_codebooks(False)
            ctrl.trainer.alignment.eval()
            ctrl.trainer.alignment.mixer.crossed_codebooks(False)
            self.assertEqual(old, fingerprint(ctrl.gate.state()))

    def test_complete_four_arm_budgets_and_freeze_invariance(self):
        for arm in ARMS:
            ctrl, batches, _ = self.controller(arm)
            start = ctrl.checkpoint()
            before_gate = fingerprint(ctrl.gate.state())
            while not ctrl.done:
                kind = ctrl.row["kind"]
                skip = ctrl.policy == "freeze" and kind == "T"
                if not skip:
                    for _ in range(ctrl.row["updates"]):
                        ctrl.step(batches, kind)
                ctrl.prepare_boundary()
                ctrl.complete_phase()
            final = ctrl.checkpoint()
            self.assertEqual(ctrl.trainer.r_updates, 35)  # Tiny fixture has one update per R.
            self.assertEqual(ctrl.trainer.t_updates, ctrl.spec["stop_t"])
            self.assertEqual(ctrl.gate.completed, ctrl.spec["updates"])
            if ctrl.spec["mode"] == "freeze":
                self.assertEqual(before_gate, fingerprint(ctrl.gate.state()))
                for key in ("alignment", "t_optimizer"):
                    self.assertEqual(fingerprint(start["inner"][key]), fingerprint(final["inner"][key]))
                restored, _, _ = self.controller(arm)
                restored.restore(final)
                self.assertEqual(fingerprint(final), fingerprint(restored.checkpoint()))
                bad = copy.deepcopy(final)
                bad["inner"]["alignment"]["hard_codes"][0, 0] += 1
                with self.assertRaises(ValueError): restored.restore(bad)

if __name__ == "__main__":
    unittest.main()
