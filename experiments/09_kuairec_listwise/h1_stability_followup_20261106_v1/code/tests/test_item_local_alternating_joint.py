"""Real CPU BF16 gates for item-local attention with alternating DIGER-ST.

Run with the existing CPU environment; no optional skips or CUDA work:
    .runtime/continuous_env/Scripts/python.exe -X utf8 -B tests/test_item_local_alternating_joint.py

Only catalog/data abstractions are synthetic. The four-layer Decoder, local
attention, JointCrossPSEAlignment, ST assignments and recommendation loss are
the actual engine implementations. Tokenizer tests backpropagate only
recommendation_loss, excluding the separately computed reconstruction loss.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, redirect_stdout
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_item_local_alternating_fixture", ROOT / "tests/test_item_local_attention.py"
)
assert SPEC is not None and SPEC.loader is not None
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
Mixer = importlib.import_module(f"{fixture.PACKAGE}.cross_pse").CrossPSECodebookMixer


class CatalogTokenizer(fixture.TinyTokenizer):
    def __init__(self, items):
        self.id2item = ["<pad>", *items]
        self.item2id = {item: index for index, item in enumerate(self.id2item)}
        self._cb2items = None

    def _create_reverse_mapping(self):
        return {tokens: self.item2id[item] for item, tokens in self.item2tokens.items()}

    def _compute_collision_stats(self, mapping):
        counts = Counter(mapping.values())
        return {"items": len(mapping), "unique": len(counts),
                "collision_rate": sum(count for count in counts.values() if count > 1) / len(mapping)}

    def _finalize_position_stats(self, values):
        return values


@contextmanager
def joint_model(*, mixer_enabled=False):
    torch.manual_seed(2024)
    generator = np.random.default_rng(2024)
    items = [f"item-{index}" for index in range(7)]
    with tempfile.TemporaryDirectory(prefix="item_local_joint_") as temporary:
        directory = Path(temporary)
        base = generator.normal(0, 0.2, size=(4, 8, 8)).astype("float32")
        slots = generator.normal(0, 0.2, size=(len(items), 4, 8)).astype("float32")
        codes = ((slots[:, :, None, :] - base[None]) ** 2).sum(-1).argmin(-1).astype("int64")
        np.save(directory / "base_pse_codebooks.npy", base)
        np.save(directory / "opq_residual_embeddings.npy", slots.reshape(len(items), -1))
        np.save(directory / "semantic_ids.npy", codes)
        (directory / "items.json").write_text(json.dumps(items), encoding="utf-8")
        (directory / "summary.json").write_text(
            json.dumps({"mixer_enabled": mixer_enabled}), encoding="utf-8"
        )
        initialized = Mixer(
            torch.from_numpy(base), n_heads=2, dropout=0.0,
            residual_scale_init=1.0, learn_residual_scale=False,
            zero_init_attention_output=True, train_base_codebooks=True,
            mixer_enabled=mixer_enabled,
        )
        torch.save({"model": initialized.state_dict()}, directory / "best.pt")
        config = fixture.tiny_config(True)
        config.update(
            decoder_n_layer=4, joint_crosspse_enabled=True,
            joint_crosspse_artifact_dir=str(directory),
            joint_crosspse_items_path=str(directory / "items.json"),
            crosspse_heads=2, crosspse_dropout=0.0,
            crosspse_mixer_enabled=mixer_enabled,
            crosspse_train_base_codebooks=True,
            crosspse_residual_scale_init=1.0,
            crosspse_learn_residual_scale=False,
            crosspse_zero_init_attention_output=True,
            joint_gradient_mode="diger_st",
            joint_diger_assignment_temperature=0.2,
            joint_diger_softmax_temperature_start=1.0,
            joint_diger_softmax_temperature_end=1.0,
            joint_diger_gumbel_noise_start=0.0,
            joint_diger_gumbel_noise_end=0.0,
            joint_recommender_sia_weight=0.0,
            joint_tokenizer_sia_weight=0.0,
            joint_balance_weight=0.0, joint_anchor_weight=0.0,
            joint_diger_recommendation_weight=1.0,
        )
        with redirect_stdout(io.StringIO()):
            model = fixture.DIFF_GRM(config, object(), CatalogTokenizer(items))
        model.joint_alignment.configure_diger_schedule(8)
        assert len(model.decoder_blocks) == 4
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        yield model


def joint_batch():
    history = torch.tensor([[0, 1, 2], [0, 0, 3]])
    return {"history_item_ids": history, "history_mask": history.ne(0),
            "target_item_id": torch.tensor([4, 5])}


def snapshot_parameters(model, *, alignment):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if name.startswith("joint_alignment.") == alignment}


def module_modes(model):
    return {name: module.training for name, module in model.named_modules()}


@contextmanager
def cpu_bf16():
    with redirect_stdout(io.StringIO()), torch.autocast(
        "cpu", dtype=torch.bfloat16, cache_enabled=True,
    ):
        yield


class AlternatingItemLocalJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def assert_unchanged_and_no_gradient(self, model, before):
        parameters = dict(model.named_parameters())
        for name, value in before.items():
            self.assertFalse(parameters[name].requires_grad, name)
            self.assertIsNone(parameters[name].grad, name)
            self.assertTrue(torch.equal(parameters[name].detach(), value), name)

    def assert_finite_gradient(self, parameter, name, *, nonzero=False):
        self.assertIsNotNone(parameter.grad, name)
        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        if nonzero:
            self.assertGreater(float(parameter.grad.float().abs().sum()), 0.0, name)

    def recommender_updates(self, model, *, steps=3):
        model.zero_grad(set_to_none=True)
        model.set_joint_phase("recommender")
        self.assertTrue(model.training)
        self.assertTrue(model.item_local_attention.training)
        self.assertFalse(model.joint_alignment.training)
        before = snapshot_parameters(model, alignment=False)
        frozen = snapshot_parameters(model, alignment=True)
        codes = model.joint_alignment.hard_codes.detach().clone()
        optimizer = torch.optim.AdamW(list(model.joint_recommender_parameters()), lr=0.01, weight_decay=0.0)
        decoder_linears = {name: parameter for name, parameter in model.named_parameters()
                           if name.startswith("decoder_blocks.") and ".ln_" not in name}
        self.assertEqual(len(decoder_linears), 40)
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            modes_before = module_modes(model)
            with cpu_bf16():
                output = model(joint_batch(), return_loss=True)
                loss = output.recommendation_loss
            self.assertEqual(module_modes(model), modes_before, "OCN teacher changed recommender mixed modes")
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(torch.equal(output.loss, loss))
            loss.backward()
            for name, parameter in decoder_linears.items():
                self.assert_finite_gradient(parameter, name)
            for name, parameter in model.item_local_attention.named_parameters():
                self.assert_finite_gradient(parameter, "item_local_attention." + name)
            # This adapter belongs to the optional SIA objective, disabled here.
            for parameter in model.joint_sequence_adapter.parameters():
                self.assertIsNone(parameter.grad)
            self.assert_unchanged_and_no_gradient(model, frozen)
            optimizer.step()
        current = dict(model.named_parameters())
        expected_changed = set(decoder_linears) | {
            "item_local_attention." + name for name, _ in model.item_local_attention.named_parameters()
        }
        for name in expected_changed:
            self.assertFalse(torch.equal(before[name], current[name].detach()), name)
            self.assertEqual(int(optimizer.state[current[name]]["step"].item()), steps, name)
        self.assertTrue(torch.equal(model.joint_alignment.hard_codes, codes))
        self.assert_unchanged_and_no_gradient(model, frozen)

    def test_recommender_bf16_updates_four_layer_decoder_and_local_with_frozen_codebooks(self):
        for mixer_enabled in (False, True):
            with self.subTest(global_mixer=mixer_enabled), joint_model(mixer_enabled=mixer_enabled) as model:
                self.recommender_updates(model)

    def test_tokenizer_recommendation_only_updates_codebooks_while_recommender_and_local_stay_frozen(self):
        for mixer_enabled in (False, True):
            with self.subTest(global_mixer=mixer_enabled), joint_model(mixer_enabled=mixer_enabled) as model:
                self.recommender_updates(model, steps=2)
                model.zero_grad(set_to_none=True)
                model.set_joint_phase("tokenizer")
                self.assertFalse(model.training)
                self.assertFalse(model.item_local_attention.training)
                self.assertTrue(model.joint_alignment.training)
                frozen = snapshot_parameters(model, alignment=False)
                before = model.joint_alignment.mixer.base_codebooks.detach().clone()
                groups = model.joint_tokenizer_parameter_groups()
                optimizer = torch.optim.AdamW([parameter for _, values in groups for parameter in values],
                                              lr=0.001, weight_decay=0.0)
                for _ in range(2):
                    model.zero_grad(set_to_none=True)
                    modes_before = module_modes(model)
                    with cpu_bf16():
                        output = model.forward_joint_tokenizer(joint_batch(), measure_direct_gradient=True)
                    self.assertEqual(module_modes(model), modes_before, "OCN teacher changed tokenizer mixed modes")
                    self.assertTrue(bool(torch.isfinite(output.loss)))
                    self.assertGreater(float(output.diger_direct_gradient_norm), 0.0)
                    # Reconstruction is computed by the real API but excluded
                    # from this backward, so it cannot supply the tested gradient.
                    output.recommendation_loss.backward()
                    self.assert_finite_gradient(model.joint_alignment.mixer.base_codebooks,
                                                "base_codebooks", nonzero=True)
                    self.assert_unchanged_and_no_gradient(model, frozen)
                    optimizer.step()
                    model.joint_alignment.advance_diger_step()
                self.assertFalse(torch.equal(before, model.joint_alignment.mixer.base_codebooks.detach()))
                self.assert_unchanged_and_no_gradient(model, frozen)
                if not mixer_enabled:
                    for name, parameter in model.joint_alignment.mixer.named_parameters():
                        if name != "base_codebooks":
                            self.assertFalse(parameter.requires_grad, name)
                            self.assertIsNone(parameter.grad, name)

    def test_ocn_teacher_exception_preserves_every_joint_module_mode(self):
        class TeacherFailure(RuntimeError):
            pass

        with joint_model() as model:
            for phase in ("recommender", "tokenizer"):
                with self.subTest(phase=phase):
                    model.set_joint_phase(phase)
                    before = module_modes(model)
                    with mock.patch.object(model, "forward_decoder_only", side_effect=TeacherFailure("teacher probe")):
                        with self.assertRaisesRegex(TeacherFailure, "teacher probe"), cpu_bf16():
                            if phase == "recommender":
                                model(joint_batch(), return_loss=True)
                            else:
                                model.forward_joint_tokenizer(joint_batch())
                    self.assertEqual(module_modes(model), before, "Teacher exception changed mixed train/eval modes")

    def test_history_only_st_path_crosses_local_and_item_mlp_to_real_recommendation_loss(self):
        with joint_model() as model:
            self.recommender_updates(model, steps=2)
            model.zero_grad(set_to_none=True)
            model.set_joint_phase("tokenizer")
            frozen = snapshot_parameters(model, alignment=False)
            observed = {}

            def local_hook(_module, args, output):
                observed["input"], observed["output"] = args[0], output
                args[0].retain_grad()
                output.retain_grad()

            def mlp_hook(_module, args):
                observed["compressed"] = args[0]

            handles = [model.item_local_attention.register_forward_hook(local_hook),
                       model.item_mlp.register_forward_pre_hook(mlp_hook)]
            try:
                with cpu_bf16():
                    prepared = model.prepare_joint_batch(joint_batch(), training=True, differentiable=True)
                    self.assertTrue(prepared["history_sid_st"].requires_grad)
                    # Remove the target-ST route: codebook gradients below must
                    # traverse the actual history/local-attention/Encoder path.
                    prepared["target_sid_st"] = prepared["target_sid_st"].detach()
                    loss = model(prepared, return_loss=True).recommendation_loss
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assert_finite_gradient(observed["input"], "local input", nonzero=True)
                self.assert_finite_gradient(observed["output"], "local output", nonzero=True)
                torch.testing.assert_close(observed["compressed"], observed["output"].reshape(2, 3, 64), rtol=0, atol=0)
                self.assert_finite_gradient(model.joint_alignment.mixer.base_codebooks, "history-only codebooks", nonzero=True)
                self.assert_unchanged_and_no_gradient(model, frozen)
            finally:
                for handle in handles:
                    handle.remove()

    def test_noise_free_st_hard_codes_match_refresh_and_pad_is_zero_and_finite(self):
        for mixer_enabled in (False, True):
            with self.subTest(global_mixer=mixer_enabled), joint_model(mixer_enabled=mixer_enabled) as model:
                model.set_joint_phase("tokenizer")
                model.refresh_joint_codes()
                item_ids = torch.tensor([[0, 1, 2], [0, 0, 3]])
                with cpu_bf16():
                    assignments, diagnostics = model.joint_alignment.diger_assignments(item_ids, stochastic=True)
                    prepared = model.prepare_joint_batch(joint_batch(), training=True, differentiable=True)
                    output = model(prepared, return_loss=True)
                self.assertTrue(bool(torch.isfinite(assignments).all()))
                self.assertEqual(float(diagnostics["gumbel_noise_scale"]), 0.0)
                self.assertTrue(torch.equal(assignments[item_ids.eq(0)], torch.zeros_like(assignments[item_ids.eq(0)])))
                valid = item_ids.ne(0)
                expected = model.joint_alignment.hard_codes[item_ids][valid]
                actual = assignments.detach().argmax(-1)[valid]
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(assignments.detach()[valid], torch.nn.functional.one_hot(expected, 8).float()))
                self.assertTrue(torch.equal(prepared["history_sid"][item_ids.eq(0)],
                                            torch.full_like(prepared["history_sid"][item_ids.eq(0)], -1)))
                self.assertTrue(bool(torch.isfinite(output.recommendation_loss)))
                self.assertTrue(bool(torch.isfinite(output.hidden_states).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
