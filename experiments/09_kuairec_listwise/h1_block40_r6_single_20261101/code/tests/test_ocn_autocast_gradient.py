"""CPU regressions for guided OCN teacher casts and real BF16 recovery.

Run with the existing CPU PyTorch environment:
    .runtime/continuous_env/Scripts/python.exe -X utf8 -B tests/test_ocn_autocast_gradient.py

The actual four-layer Decoder, loss, AdamW and epoch-boundary state helpers run
without a Trainer, dataset download, CUDA work, or historical artifact writes.
The frozen 1dc2a0c model is an explicit negative control, never patched in place.
"""
from __future__ import annotations

import copy
import importlib.util
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FROZEN_REVISION = "1dc2a0c5ad01e2a2878b9de400b91f3a7e809f25"
fixture_spec = importlib.util.spec_from_file_location(
    "_ocn_gradient_fixture", ROOT / "tests/test_item_local_attention.py"
)
assert fixture_spec is not None and fixture_spec.loader is not None
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)
state_spec = importlib.util.spec_from_file_location(
    f"{fixture.PACKAGE}.ocn_training_state", fixture.SOURCE / "training_state.py"
)
assert state_spec is not None and state_spec.loader is not None
state = importlib.util.module_from_spec(state_spec)
sys.modules[state_spec.name] = state
state_spec.loader.exec_module(state)


def frozen_model_class():
    source = subprocess.run(
        ["git", "-C", str(fixture.ENGINE), "show",
         f"{FROZEN_REVISION}:genrec/models/DIFF_GRM/model.py"],
        check=True, capture_output=True, encoding="utf-8",
    ).stdout
    module = types.ModuleType(f"{fixture.PACKAGE}.frozen_ocn_model")
    module.__package__ = fixture.PACKAGE
    with mock.patch.dict(sys.modules, {
        "genrec.model": fixture.abstract_module,
        "genrec.dataset": fixture.dataset_module,
        "genrec.tokenizer": fixture.tokenizer_module,
    }):
        exec(compile(source, f"{FROZEN_REVISION}:model.py", "exec"), module.__dict__)
    return module.DIFF_GRM


FrozenModel = frozen_model_class()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(enabled, *, cls=fixture.DIFF_GRM, seed=2024, dropout=0.0):
    seed_all(seed)
    config = fixture.tiny_config(enabled)
    config.update(decoder_n_layer=4, dropout=dropout, attn_pdrop=dropout, resid_pdrop=dropout)
    with redirect_stdout(io.StringIO()):
        model = cls(config, object(), fixture.TinyTokenizer()).train()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    return model


def make_components(enabled, *, seed=2024):
    model = make_model(enabled, seed=seed, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    return model, optimizer, scheduler


def randomized_batch():
    """Make the continuation depend on Python, NumPy and torch RNGs."""
    draws = (random.randrange(8), int(np.random.randint(0, 8)), int(torch.randint(0, 8, ()).item()))
    batch = fixture.tiny_batch()
    shift = sum(draws) % 8
    valid = batch["history_sid"].ne(-1)
    batch["history_sid"][valid] = (batch["history_sid"][valid] + shift) % 8
    batch["decoder_labels"] = (batch["decoder_labels"] + shift) % 8
    batch["decoder_input_ids"] = batch["decoder_labels"].clone()
    return batch, draws


def forward_loss(model, batch, *, bf16=True):
    with redirect_stdout(io.StringIO()), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=True,
    ):
        return model(batch).loss


def train_step(model, optimizer, scheduler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch, draws = randomized_batch()
    loss = forward_loss(model, batch)
    loss.backward()
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
    optimizer.step()
    scheduler.step()
    return {"loss": loss.detach().clone(), "draws": draws, "gradients": gradients}


def snapshot(model, optimizer, scheduler):
    # The helper's CPU branch is used explicitly; no CUDA API is invoked.
    with mock.patch.object(state.torch.cuda, "is_available", return_value=False):
        rng = state.capture_rng(None)
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": rng})


def expected_missing_decoder_parameters():
    suffixes = ("self_attn.qkv.weight", "self_attn.proj.weight", "self_attn.proj.bias",
                "cross_attn.qkv.weight", "cross_attn.proj.weight", "cross_attn.proj.bias",
                "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")
    return {f"decoder_blocks.{layer}.{suffix}" for layer in range(4) for suffix in suffixes}


class OcnAutocastGradientTests(unittest.TestCase):
    def assert_nested_equal(self, left, right, path="state"):
        if torch.is_tensor(left):
            self.assertTrue(torch.is_tensor(right), path)
            self.assertEqual((left.dtype, left.shape, left.device), (right.dtype, right.shape, right.device), path)
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assert_nested_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right), path)
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_nested_equal(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def assert_finite_gradients(self, model):
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"Missing gradient: {name}")
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), f"Nonfinite gradient: {name}")

    def test_frozen_cache_on_bf16_reproduces_forty_missing_decoder_gradients(self):
        expected = expected_missing_decoder_parameters()
        self.assertEqual(len(expected), 40)
        for enabled in (False, True):
            with self.subTest(local_attention=enabled):
                model = make_model(enabled, cls=FrozenModel)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
                loss = forward_loss(model, fixture.tiny_batch())
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                missing = {name for name, parameter in model.named_parameters() if parameter.grad is None}
                self.assertEqual(missing, expected)
                optimizer.step()
                self.assertEqual(len(optimizer.state), len(list(model.parameters())) - 40)
                self.assertEqual(len(optimizer.state), 50 if enabled else 44)

    def test_fixed_bf16_two_steps_have_finite_gradients_complete_adamw_and_updates(self):
        for enabled in (False, True):
            with self.subTest(local_attention=enabled):
                model, optimizer, scheduler = make_components(enabled)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                for step in range(1, 3):
                    result = train_step(model, optimizer, scheduler)
                    self.assertTrue(bool(torch.isfinite(result["loss"])))
                    self.assert_finite_gradients(model)
                    self.assertEqual(set(optimizer.state), set(model.parameters()))
                    for parameter in model.parameters():
                        entry = optimizer.state[parameter]
                        self.assertEqual(int(entry["step"].item()), step)
                        for key in ("exp_avg", "exp_avg_sq"):
                            self.assertEqual(entry[key].shape, parameter.shape)
                            self.assertTrue(bool(torch.isfinite(entry[key]).all()))
                changed = {name for name, parameter in model.named_parameters()
                           if not torch.equal(before[name], parameter.detach())}
                self.assertEqual(changed, set(before), "Each trainable tensor must actually update; weight_decay is zero")
                self.assertTrue(expected_missing_decoder_parameters() <= changed)

    def test_teacher_scores_loss_and_float32_gradients_match_frozen_behavior(self):
        for enabled in (False, True):
            for bf16 in (False, True):
                with self.subTest(local_attention=enabled, bf16=bf16):
                    old = make_model(enabled, cls=FrozenModel)
                    fixed = make_model(enabled)
                    fixed.load_state_dict(old.state_dict(), strict=True)
                    observations = []
                    losses = []
                    for model in (old, fixed):
                        captured = []
                        original = model.forward_decoder_only

                        def score(*args, **kwargs):
                            answer = original(*args, **kwargs)
                            captured.append({"logits": answer.logits.detach().clone(),
                                             "grad_enabled": torch.is_grad_enabled(),
                                             "autocast_enabled": torch.is_autocast_enabled("cpu")})
                            return answer

                        with mock.patch.object(model, "forward_decoder_only", side_effect=score):
                            losses.append(forward_loss(model, fixture.tiny_batch(), bf16=bf16))
                        self.assertTrue(model.training, "Teacher must restore training mode")
                        self.assertEqual(len(captured), 1)
                        self.assertFalse(captured[0]["grad_enabled"])
                        self.assertEqual(captured[0]["autocast_enabled"], bf16, "FP32 must not accidentally enable autocast")
                        observations.append(captured)
                    self.assert_nested_equal(observations[0], observations[1], "teacher")
                    self.assertTrue(torch.equal(losses[0], losses[1]), "Fix must preserve the initial forward objective")
                    if not bf16:
                        losses[0].backward()
                        losses[1].backward()
                        self.assert_finite_gradients(old)
                        self.assert_finite_gradients(fixed)
                        self.assert_nested_equal({name: p.grad for name, p in old.named_parameters()},
                                                 {name: p.grad for name, p in fixed.named_parameters()}, "float32_gradients")

    def test_bf16_save_load_continuation_matches_uninterrupted_all_rng_and_optimizer_state(self):
        for enabled in (False, True):
            with self.subTest(local_attention=enabled), tempfile.TemporaryDirectory(prefix="ocn_bf16_recovery_") as directory:
                direct = make_components(enabled)
                expected_steps = [train_step(*direct) for _ in range(3)]
                expected_state = snapshot(*direct)

                interrupted = make_components(enabled)
                for index in range(2):
                    self.assert_nested_equal(train_step(*interrupted), expected_steps[index], f"before_save_step_{index + 1}")
                path = Path(directory) / "boundary.pt"
                fingerprint = state.configuration_fingerprint(interrupted[0].config)
                progress = {"completed_optimizer_updates": 2}
                accelerator = types.SimpleNamespace(unwrap_model=lambda model: model, scaler=None, step=999)
                with mock.patch.object(state.torch.cuda, "is_available", return_value=False):
                    state.save_boundary(
                        path, model=interrupted[0], optimizer=interrupted[1], scheduler=interrupted[2],
                        accelerator=accelerator, loader=None, progress=progress,
                        config_fingerprint=fingerprint, diagnostic_paths=[],
                        best_checkpoint=Path(directory) / "no_best_selected.bin",
                    )
                # Fresh construction and random draws model a different process.
                resumed = make_components(enabled, seed=99)
                random.random()
                np.random.random(7)
                torch.rand(11)
                payload = state.load_boundary(path)
                self.assertEqual(len(payload["optimizer"]["state"]), len(list(resumed[0].parameters())))
                with mock.patch.object(state.torch.cuda, "is_available", return_value=False):
                    restored_progress = state.restore_boundary(
                        payload, model=resumed[0], optimizer=resumed[1], scheduler=resumed[2],
                        accelerator=accelerator, loader=None, config_fingerprint=fingerprint,
                    )
                self.assertEqual(restored_progress, progress)
                self.assertEqual(accelerator.step, 0)
                self.assert_nested_equal(train_step(*resumed), expected_steps[2], "resumed_step_3")
                self.assert_finite_gradients(resumed[0])
                self.assert_nested_equal(snapshot(*resumed), expected_state, "complete_resumed_state")


if __name__ == "__main__":
    unittest.main(verbosity=2)
