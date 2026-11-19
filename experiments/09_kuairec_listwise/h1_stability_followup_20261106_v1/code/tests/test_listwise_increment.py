"""Targeted CPU checks with real tiny models; no KuaiRec training or submission."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_listwise_training import fixture
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.recovery_increment import (
    IncrementController, IncrementGate, SCHEDULE, acceptance, gate_metrics,
)
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all
from scripts.listwise_diffgrm import recovery_increment_run as runner


def tiny(directory):
    with redirect_stdout(io.StringIO()):
        original, batch = fixture(directory, "H", True)
    config = copy.deepcopy(original.config)
    config["training"].update(warmup_epochs=5, cycles=10, r_epochs_per_cycle=3,
                              t_updates_per_cycle=40)
    trainer = AlternatingTrainer(original.model, original.alignment, config, 4,
                                 precision="bf16")
    return trainer, [batch, {k: v.clone() for k, v in batch.items()}]


class IncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        trainer, batches = tiny(cls.directory)
        binding = dict(config=copy.deepcopy(trainer.config), asset_manifest="fixture",
                       data={}, engine={}, code={})
        for phase, row in enumerate(trainer.plan[:25]):
            seed_all(trainer.config["seed"] + 100003 * (phase + 1))
            for _ in range(row["updates"]):
                trainer.update(batches, row["kind"])
            if row["kind"] == "T":
                trainer.refresh_catalog()
        cls.parent = trainer.checkpoint(binding, dict(phase=25, update=0), "H1")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def controller(self):
        trainer, batches = tiny(self.directory)
        binding = self.parent["binding"]
        controller = IncrementController(trainer, binding, binding, "synthetic-parent")
        controller.fork(self.parent)
        return controller, batches

    def test_schedule_and_complete_path_integral_without_pullback(self):
        values = [acceptance(k) for k in range(200)]
        self.assertEqual(values[0], 0.1)
        self.assertAlmostEqual(values[-1], 0.01)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual(acceptance(200), values[-1])
        for k in (-1, 201, 1.5, True):
            with self.assertRaises(ValueError):
                acceptance(k)
        controller, _ = self.controller()
        gate = controller.increment_gate
        expected = torch.zeros_like(gate.accepted_delta)
        raw = gate.previous_raw.clone()
        # A known alternating raw path, independent of optimization/model fit.
        for k, alpha in enumerate(values):
            new_raw = raw + (0.125 if k % 2 == 0 else -0.0625)
            expected += alpha * (new_raw - raw)
            with patch.object(gate, "raw_correction", return_value=new_raw):
                gate.commit()
            raw = new_raw
            self.assertTrue(torch.equal(gate.accepted_delta, expected))
        self.assertEqual(gate.completed, 200)
        with self.assertRaises(ValueError):
            gate.commit()
        # Merely lowering the next coefficient cannot retract accepted state.
        before = gate.mixer.crossed_codebooks(False)[0].detach()
        gate.completed = 100
        after = gate.mixer.crossed_codebooks(False)[0].detach()
        self.assertTrue(torch.equal(before, after))

    def test_start_identity_optimizers_rng_and_sid(self):
        controller, _ = self.controller()
        trainer, gate = controller.trainer, controller.increment_gate
        saved = controller.checkpoint()
        for key in ("model", "alignment", "r_optimizer", "t_optimizer", "scheduler", "rng"):
            self.assertEqual(fingerprint(saved["inner"][key]), fingerprint(self.parent[key]))
        self.assertTrue(torch.equal(gate.mixer.crossed_codebooks(False)[0], gate.c0))
        old = trainer.alignment.hard_codes.clone()
        trainer.refresh_catalog()
        self.assertTrue(torch.equal(old, trainer.alignment.hard_codes))
        gate_state = fingerprint(gate.state())
        for attention in (False, True, False):
            gate.mixer.crossed_codebooks(attention)
        gate_metrics(trainer)
        self.assertEqual(gate_state, fingerprint(gate.state()))

    def test_real_T_update_formula_gradients_and_R_isolation(self):
        controller, batches = self.controller()
        trainer, gate = controller.trainer, controller.increment_gate
        lrs = [g["lr"] for g in trainer.t_optimizer.param_groups]
        old_model = fingerprint(trainer.model.state_dict())
        raw0 = gate.previous_raw.clone()
        result = controller.step(batches, "T")
        self.assertEqual(result["increment_alpha_used"], 0.1)
        self.assertTrue(torch.equal(gate.accepted_delta, 0.1 * (gate.previous_raw - raw0)))
        self.assertEqual(old_model, fingerprint(trainer.model.state_dict()))
        self.assertGreater(float(gate.mixer.base_codebooks.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in gate.mixer.named_parameters() if n != "base_codebooks"))
        for _ in range(39):
            controller.step(batches, "T")
        self.assertEqual(lrs, [g["lr"] for g in trainer.t_optimizer.param_groups])
        controller.prepare_boundary()
        controller.complete_phase()
        before_gate = fingerprint(gate.state())
        before_alignment = fingerprint(trainer.alignment.state_dict())
        before_model = fingerprint(trainer.model.state_dict())
        controller.step(batches, "R")
        self.assertEqual(before_gate, fingerprint(gate.state()))
        self.assertEqual(before_alignment, fingerprint(trainer.alignment.state_dict()))
        self.assertNotEqual(before_model, fingerprint(trainer.model.state_dict()))
        controller.checkpoint()  # validates optimizer, noise and schedule clocks

    def test_disk_resume_exact_next_step_and_tamper_rejection(self):
        controller, batches = self.controller()
        controller.step(batches, "T")
        controller.step(batches, "T")
        path = self.directory / "increment_resume.pt"
        torch.save(controller.checkpoint(), path)
        expected_result = controller.step(batches, "T")
        expected = controller.checkpoint()
        restored, batches2 = self.controller()
        state = torch.load(path, map_location="cpu", weights_only=False)
        restored.restore(state)
        actual_result = restored.step(batches2, "T")
        self.assertEqual(expected_result, actual_result)
        self.assertEqual(fingerprint(expected), fingerprint(restored.checkpoint()))
        for key in ("accepted_delta", "previous_raw"):
            bad = copy.deepcopy(state)
            bad["increment_gate"]["payload"][key] += 1
            with self.assertRaises(ValueError):
                restored.restore(bad)
        for edit in ("clock", "schedule", "anchor", "protocol", "missing", "lr"):
            bad = copy.deepcopy(state)
            payload = bad["increment_gate"]["payload"]
            if edit == "clock":
                payload["completed"] += 1
            elif edit == "schedule":
                payload["schedule"]["end"] = 0.02
            elif edit == "anchor":
                payload["anchor"]["C0"] += 1
            elif edit == "protocol":
                bad["protocol"]["name"] = "other"
            elif edit == "lr":
                bad["inner"]["t_optimizer"]["param_groups"][0]["lr"] *= 0.5
            bad["increment_gate"]["sha256"] = fingerprint(payload)
            if edit == "missing":
                del bad["increment_gate"]
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                restored.restore(bad)

    def test_evaluation_cache_binds_gate_and_best_tie_is_earlier(self):
        controller, _ = self.controller()
        output = self.directory / "evaluation_check"
        with patch.object(runner, "_evaluation_event") as evaluate:
            runner.evaluation_event(controller, None, None, None, output, "R23", full=True)
            self.assertEqual(evaluate.call_count, 1)
            controller.increment_gate.accepted_delta += 0.1
            with self.assertRaises(ValueError):
                runner.evaluation_event(controller, None, None, None, output, "R23", full=True)
        records = output / "validation_records"
        records.mkdir()
        for epoch in runner.BEST_EPOCHS:
            runner.write_json(records / f"R{epoch:02d}.json", dict(ndcg5=0.1))
        self.assertEqual(runner._selection(output)["selected"]["epoch"], 20)


if __name__ == "__main__":
    unittest.main()
