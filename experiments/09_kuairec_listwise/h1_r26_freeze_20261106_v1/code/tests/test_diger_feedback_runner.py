"""Runtime guard tests; optional tensor tests require Torch and NumPy, not a GPU.

python -B tests/test_diger_feedback_runner.py
Use --require-torch on the research server: a skipped tensor test then fails.
These are synthetic checks, not a substitute for the three-arm GPU smoke.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("_feedback_runtime_tests", ROOT / "scripts/train/run_diger_noise_pilot.py")
entry = load_module("_feedback_entry_tests", ROOT / "scripts/train/run_diger_feedback_pilot.py")
helper = load_module("_feedback_protocol_tests", ROOT / "scripts/checks/diger_feedback_pilot_protocol.py")
try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


class Scalar(float):
    def detach(self):
        return self

    def __add__(self, other):
        return Scalar(float(self) + float(other))

    __radd__ = __add__

    def __mul__(self, other):
        return Scalar(float(self) * float(other))

    __rmul__ = __mul__


class RuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.protocol = helper.load_protocol(ROOT / helper.DEFAULT_CONFIG)

    def test_new_arm_parse(self):
        args = entry.parse_args(["--arm", "cross_recon", "--smoke", "--output", "outputs/pilots/stage05_feedback/check"])
        self.assertEqual(args.arm, "cross_recon")
        self.assertTrue(args.smoke)

    def test_legacy_arm_not_accepted_by_new_entry(self):
        with self.assertRaises(SystemExit):
            entry.parse_args(["--arm", "A", "--output", "unused"])

    def test_new_empty_output_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = root / "outputs/pilots/stage05_feedback/new"
            self.assertEqual(entry.validate_output(root, Path("outputs/pilots/stage05_feedback/new")), expected)
            self.assertFalse(expected.exists())

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "outputs/pilots/stage05_feedback/existing"
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                entry.validate_output(root, target)

    def test_historical_or_nested_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("experiments/05_crosspse_joint/evidence/new", "outputs/pilots/stage05_noise/new",
                          "outputs/pilots/stage05_feedback", "outputs/pilots/stage05_feedback/new/nested",
                          "outputs/pilots/stage05_feedback/../outside"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    entry.validate_output(Path(directory), Path(value))

    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            external = root / "other"
            external.mkdir()
            try:
                (root / "outputs").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("Host does not permit creating directory symlinks")
            with self.assertRaises(ValueError):
                entry.validate_output(root, Path("outputs/pilots/stage05_feedback/new"))

    def test_feedback_weight_and_update_budget_are_applied(self):
        for arm, updating, weight in (("frozen", False, 1.0), ("cross_recon", True, 0.0), ("cross_rec", True, 1.0)):
            pipeline = SimpleNamespace(config={}, model=SimpleNamespace(config={}, joint_alignment=SimpleNamespace()))
            self.assertEqual(runner.configure_arm(pipeline, helper.arm_settings(self.protocol, arm)), updating)
            self.assertEqual(pipeline.config["joint_diger_recommendation_weight"], weight)
            self.assertEqual(pipeline.model.config["joint_diger_recommendation_weight"], weight)
            self.assertEqual(pipeline.model.joint_alignment.diger_noise_start, 0.0)

    def test_old_noise_arms_keep_their_budgets_and_weight(self):
        common = helper._common
        protocol = common.load_protocol(ROOT / "configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json")
        for arm in ("A", "B", "C"):
            pipeline = SimpleNamespace(config={}, model=SimpleNamespace(config={}, joint_alignment=SimpleNamespace()))
            self.assertEqual(runner.configure_arm(pipeline, common.arm_settings(protocol, arm)), arm != "A")
            self.assertEqual(pipeline.model.config["joint_diger_recommendation_weight"], 1.0)

    def test_resolved_controls_must_match(self):
        pinned = dict(helper.CONTROLLED_TOKENIZER_SETTINGS, train_batch_size=64, eval_batch_size=8,
                      gradient_accumulation_steps=16)
        runner.verify_controlled_settings(pinned, self.protocol)
        changed = dict(pinned, joint_mixer_lr=0.001)
        with self.assertRaises(ValueError):
            runner.verify_controlled_settings(changed, self.protocol)
        with self.assertRaises(ValueError):
            runner.verify_controlled_settings({}, self.protocol)
        with self.assertRaises(ValueError):
            runner.verify_controlled_settings(dict(pinned, train_batch_size=32), self.protocol)

    def loss_fixture(self, weight):
        config = dict(joint_diger_recommendation_weight=weight, joint_tokenizer_sia_weight=0,
                      joint_balance_weight=0.01, joint_anchor_weight=0.01)
        output = SimpleNamespace(reconstruction_loss=Scalar(2), balance_loss=Scalar(3),
                                 anchor_loss=Scalar(4), recommendation_loss=Scalar(5),
                                 loss=Scalar(2.07 + weight * 5))
        return config, output

    def test_actual_zero_feedback_loss_excludes_recommendation(self):
        config, output = self.loss_fixture(0)
        fake_torch = SimpleNamespace(allclose=lambda a, b, **kwargs: math.isclose(a, b, rel_tol=kwargs["rtol"], abs_tol=kwargs["atol"]))
        with patch.object(runner, "torch", fake_torch, create=True):
            aux, weighted = runner.feedback_loss_terms(output, config, 0)
        self.assertAlmostEqual(aux, 2.07)
        self.assertEqual(weighted, 0)

    def test_stale_weight_or_wrong_total_fails(self):
        config, output = self.loss_fixture(0)
        fake_torch = SimpleNamespace(allclose=lambda a, b, **kwargs: math.isclose(a, b, rel_tol=kwargs["rtol"], abs_tol=kwargs["atol"]))
        with patch.object(runner, "torch", fake_torch, create=True):
            with self.assertRaises(RuntimeError):
                runner.feedback_loss_terms(output, config, 1)
            output.loss = Scalar(7.07)
            with self.assertRaises(RuntimeError):
                runner.feedback_loss_terms(output, config, 0)

    def test_sia_cannot_enter_feedback_ablation(self):
        config, output = self.loss_fixture(1)
        config["joint_tokenizer_sia_weight"] = 0.1
        with self.assertRaises(RuntimeError):
            runner.feedback_loss_terms(output, config, 1)

    def test_probe_code_never_calls_mapping_or_mode_changes(self):
        tree = ast.parse((ROOT / "scripts/train/run_diger_noise_pilot.py").read_text(encoding="utf-8"))
        probe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tokenizer_probe")
        calls = {node.func.attr for node in ast.walk(probe) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"refresh_joint_codes", "refresh_hard_codes", "sync_joint_tokenizer_mapping", "train", "eval", "set_joint_phase"})
        self.assertIn("named_buffers", calls)

    def test_wrapper_injects_executed_helper_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fake_helper = SimpleNamespace(load_protocol=lambda path: {})
            invocations = []
            fake_runtime = SimpleNamespace(run=lambda args, protocol_helper: invocations.append((args, protocol_helper)))
            with patch.object(entry, "load_adjacent_module", side_effect=[fake_helper, fake_runtime]):
                entry.main(["--project-root", str(root), "--output", "outputs/pilots/stage05_feedback/check"])
            args, passed = invocations[0]
            self.assertIs(passed, fake_helper)
            self.assertEqual(args.entrypoint, ROOT / "scripts/train/run_diger_feedback_pilot.py")
            self.assertFalse(args.output.exists())

    def test_wrapper_does_not_write_failure_to_unowned_path(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(args, protocol_helper):
                raise RuntimeError("No GPU")
            fake_runtime = SimpleNamespace(run=fail)
            with patch.object(entry, "load_adjacent_module", side_effect=[SimpleNamespace(load_protocol=lambda path: {}), fake_runtime]):
                with self.assertRaises(RuntimeError):
                    entry.main(["--project-root", directory, "--output", "outputs/pilots/stage05_feedback/check"])
            self.assertFalse((Path(directory) / "outputs").exists())


@unittest.skipIf(torch is None or np is None, "Torch/NumPy unavailable locally; run --require-torch on research server")
class TensorRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "torch", torch, create=True))
        self.stack.enter_context(patch.object(runner, "np", np, create=True))
        self.stack.enter_context(patch.object(runner, "phase_seed", helper.phase_seed, create=True))
        self.stack.enter_context(patch.object(runner, "loss_divisor", helper.loss_divisor, create=True))

    def toy_loss(self, weight):
        alignment = torch.nn.Linear(1, 1, bias=False)
        p = alignment.weight
        output = SimpleNamespace(reconstruction_loss=(p - 1).square().sum(), balance_loss=p.square().sum(),
                                 anchor_loss=(p + 1).square().sum(), recommendation_loss=(2 * p - 3).square().sum())
        config = dict(joint_diger_recommendation_weight=weight, joint_tokenizer_sia_weight=0,
                      joint_balance_weight=0.01, joint_anchor_weight=0.01)
        aux = output.reconstruction_loss + .01 * output.balance_loss + .01 * output.anchor_loss
        output.loss = aux + weight * output.recommendation_loss
        output.diger_direct_gradient_norm = torch.autograd.grad(output.recommendation_loss, [p], retain_graph=True)[0].norm()
        return SimpleNamespace(joint_alignment=alignment, config=config), output

    def test_actual_feedback_contribution_zero_or_positive(self):
        for weight in (0, 1):
            model, output = self.toy_loss(weight)
            aux, _ = runner.feedback_loss_terms(output, model.config, weight)
            audit = runner.audit_feedback_gradient(model, output, aux, weight)
            self.assertGreater(audit["raw_direct_gradient_norm"], 0)
            if weight == 0:
                self.assertEqual(audit["weighted_direct_gradient_norm"], 0)
            else:
                self.assertGreater(audit["weighted_direct_gradient_norm"], 0)
            output.loss.backward()
            self.assertGreater(float(model.joint_alignment.weight.grad.norm()), 0)

    def test_rng_receipt_is_stable_and_detects_consumption(self):
        device = torch.device("cpu")
        before = runner.rng_state_hash(device)
        self.assertEqual(before, runner.rng_state_hash(device))
        torch.rand(1)
        self.assertNotEqual(before, runner.rng_state_hash(device))

    def test_training_loop_audits_feedback_freezes_decoder_and_calls_probes_after_update(self):
        class Alignment(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.25))
                self.diger_noise_start = self.diger_noise_end = 0.0
                self.steps = 0

            def diger_assignments(self, ids, stochastic=True):
                return ids

            def diger_schedule_state(self):
                return {"gumbel_noise_scale": 0.0}

            def advance_diger_step(self):
                self.steps += 1

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.joint_alignment = Alignment()
                self.decoder = torch.nn.Parameter(torch.tensor(2.0))
                self.config = dict(joint_diger_recommendation_weight=weight, joint_tokenizer_sia_weight=0,
                                   joint_balance_weight=.01, joint_anchor_weight=.01)

            def set_joint_phase(self, phase):
                self.decoder.requires_grad_(phase != "tokenizer")
                self.joint_alignment.value.requires_grad_(phase == "tokenizer")

            def forward_joint_tokenizer(self, batch, measure_direct_gradient=False):
                self.joint_alignment.diger_assignments(batch["history_item_ids"])
                self.joint_alignment.diger_assignments(batch["target_item_id"])
                value = self.joint_alignment.value
                output = SimpleNamespace(reconstruction_loss=(value - 1).square(), balance_loss=value.square(),
                                         anchor_loss=(value + 1).square(), recommendation_loss=(value * self.decoder - 3).square())
                output.loss = (output.reconstruction_loss + .01 * output.balance_loss + .01 * output.anchor_loss
                               + self.config["joint_diger_recommendation_weight"] * output.recommendation_loss)
                output.diger_direct_gradient_norm = (torch.autograd.grad(output.recommendation_loss, [value], retain_graph=True)[0].abs()
                                                     if measure_direct_gradient else None)
                return output

        protocol = helper.load_protocol(ROOT / helper.DEFAULT_CONFIG)
        loader = [{"history_item_ids": torch.tensor([[1, 2]]), "target_item_id": torch.tensor([2])} for _ in range(2)]
        for weight in (0, 1):
            model = Model(weight)
            pipeline = SimpleNamespace(model=model, config={"max_grad_norm": 1.0})
            optimizer = torch.optim.AdamW(model.joint_alignment.parameters(), lr=.01)
            observations = []

            def observe(updates, batches):
                self.assertIsNone(model.joint_alignment.value.grad)
                observations.append((updates, batches, model.joint_alignment.steps))

            with patch.object(runner, "make_train_loader", return_value=loader):
                result = runner.train_phase(pipeline, optimizer, "tokenizer", 2, 0, protocol, torch.device("cpu"),
                                            1, 1, gumbel_seed=10, feedback_weight=weight, update_callback=observe)
            self.assertEqual(result["optimizer_updates"], 2)
            self.assertEqual(result["gumbel_calls"], 4)
            self.assertEqual(result["noise_applied_calls"], 0)
            self.assertEqual(result["last_gradient_sides"]["recommender"], 0)
            self.assertIsNone(model.decoder.grad)
            self.assertEqual(model.decoder.item(), 2.0)
            self.assertEqual(observations, [(1, 1, 1), (2, 2, 2)])
            norm = result["feedback_gradient_audit"]["weighted_direct_gradient_norm"]
            self.assertEqual(norm == 0, weight == 0)

    def test_fresh_collision_definition_counts_all_collided_items(self):
        codes = np.array([[0, 0], [0, 0], [1, 0], [1, 1]])
        stats = runner.sid_catalog_stats(codes, [2, 2])
        self.assertEqual(stats["full_sid"]["collision_rate"], .5)
        self.assertEqual(stats["full_sid"]["unique_semantic_ids"], 3)
        self.assertEqual(stats["positions"][0]["utilization_rate"], 1)

    def test_probe_uses_fresh_codes_and_preserves_cache_modes_rng(self):
        class Mixer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.books = torch.nn.Parameter(torch.tensor([[[0.0], [10.0]], [[0.0], [10.0]]]))

            def crossed_codebooks(self, return_attention=False):
                return self.books, None

            def squared_distances(self, slots, books):
                return (slots[:, :, None, :] - books[None, :, :, :]).square().sum(-1)

            def gather_codes(self, books, codes):
                return books[torch.arange(books.shape[0])[None, :], codes]

        alignment = torch.nn.Module()
        alignment.mixer = Mixer()
        alignment.register_buffer("item_slots", torch.tensor([[[0.0], [0.0]], [[1.0], [9.0]], [[9.0], [1.0]]]))
        alignment.register_buffer("hard_codes", torch.zeros(3, 2, dtype=torch.long))
        model = torch.nn.Module()
        model.joint_alignment = alignment
        model.register_buffer("cache", torch.tensor([7.0]), persistent=False)
        pipeline = SimpleNamespace(model=model, config={"max_hist_len": 50},
                                   tokenizer=SimpleNamespace(id2item=["PAD", "a", "b"]))
        with tempfile.TemporaryDirectory() as directory:
            stats = runner.tokenizer_probe(pipeline, Path(directory), 6, 96, np.zeros((2, 2), dtype=np.int64),
                                            {"user": ["a", "b", "a"]}, torch.device("cpu"))
            fresh = np.load(Path(directory) / "tokenizer_probes/update0006/fresh_probe_codes.npy")
        self.assertEqual(fresh.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(stats["deterministic_sid_change"]["changed_items"], 2)
        self.assertEqual(alignment.hard_codes.sum().item(), 0)
        self.assertEqual(model.cache.item(), 7)
        self.assertTrue(model.training and alignment.training)


if __name__ == "__main__":
    require = "--require-torch" in sys.argv
    if require:
        sys.argv.remove("--require-torch")
        if torch is None or np is None:
            raise SystemExit("Torch/NumPy missing: tensor preflight cannot be skipped")
    unittest.main(verbosity=2)
