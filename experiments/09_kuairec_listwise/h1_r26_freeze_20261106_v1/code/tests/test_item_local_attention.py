"""Real CPU tensor gates for the item-local DiffGRM pilot; no optional skips.

Run with the existing torch environment:
    .runtime/continuous_env/Scripts/python.exe -X utf8 -B tests/test_item_local_attention.py

Only the data abstraction annotations are stubbed. Neural model, attention,
loss, and beam code are executed directly from the engine sources. The package
entry point is bypassed because it also imports unrelated training dependencies.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "external" / "DiffGRM"
SOURCE = ENGINE / "genrec" / "models" / "DIFF_GRM"
PACKAGE = "_item_local_attention_runtime"
BASE_ENGINE_REVISION = "39e94e6"
torch.set_num_threads(2)


def load_runtime():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(SOURCE)]
    sys.modules[PACKAGE] = package
    dataset = types.ModuleType("genrec.dataset")
    dataset.AbstractDataset = type("AbstractDataset", (), {})
    tokenizer = types.ModuleType("genrec.tokenizer")
    tokenizer.AbstractTokenizer = type("AbstractTokenizer", (), {})
    abstract_spec = importlib.util.spec_from_file_location(
        "genrec.model", ENGINE / "genrec" / "model.py"
    )
    assert abstract_spec is not None and abstract_spec.loader is not None
    abstract_module = importlib.util.module_from_spec(abstract_spec)
    with mock.patch.dict(sys.modules, {
        "genrec.dataset": dataset, "genrec.tokenizer": tokenizer,
    }):
        abstract_spec.loader.exec_module(abstract_module)
        with mock.patch.dict(sys.modules, {"genrec.model": abstract_module}):
            spec = importlib.util.spec_from_file_location(
                f"{PACKAGE}.model", SOURCE / "model.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return module, abstract_module, dataset, tokenizer


runtime, abstract_module, dataset_module, tokenizer_module = load_runtime()
DIFF_GRM = runtime.DIFF_GRM
ItemLocalAttention = runtime.ItemLocalAttention


def load_previous_model():
    """Use the actual pre-change model as the disabled-path compatibility oracle."""
    result = subprocess.run(
        ["git", "-C", str(ENGINE), "show",
         f"{BASE_ENGINE_REVISION}:genrec/models/DIFF_GRM/model.py"],
        check=False, capture_output=True,
    )
    # Portable execution snapshots need not carry disconnected shallow Git
    # history. The fallback is the byte-exact model from the same pinned
    # revision, verified before it can act as the compatibility oracle.
    if result.returncode == 0:
        source = result.stdout
    else:
        source = (Path(__file__).resolve().parent / 'fixtures/diffgrm_pre_item_local_model.py').read_bytes()
    import hashlib
    if hashlib.sha256(source).hexdigest() != '624bd29e63f265949d96494b62ad126283552eac80b3fb79995a84bf406031d0':
        raise ValueError('Pinned pre-item-local model source differs')
    module = types.ModuleType(f"{PACKAGE}.previous_model")
    module.__package__ = PACKAGE
    with mock.patch.dict(sys.modules, {
        "genrec.model": abstract_module,
        "genrec.dataset": dataset_module,
        "genrec.tokenizer": tokenizer_module,
    }):
        exec(compile(source, f"{BASE_ENGINE_REVISION}:model.py", "exec"), module.__dict__)
    return module.DIFF_GRM


class TinyTokenizer:
    sid_offset = 3
    vocab_size = 3 + 4 * 8
    mask_token = -1

    def codebooks_to_item_id(self, codes):
        # A synthetic full Cartesian catalog avoids unrelated illegal-ID fill.
        if len(codes) != 4 or any(code < 0 or code >= 8 for code in codes):
            return None
        return sum(int(code) * 8 ** digit for digit, code in enumerate(codes)) + 1


def tiny_config(enabled=None):
    config = {
        "n_digit": 4, "codebook_size": 8,
        "n_embd": 16, "n_head": 4, "n_inner": 32,
        "encoder_n_layer": 1, "decoder_n_layer": 1,
        "dropout": 0.0, "attn_pdrop": 0.0, "resid_pdrop": 0.0,
        "max_history_len": 3, "label_smoothing": 0.2,
        "masking_strategy": "guided", "guided_steps": 4,
        "guided_select": "least", "guided_conf_metric": "msp",
        "guided_refresh_each_step": False,
        "diffusion_loss_reduction": "per_view",
        "dedup_strategy": "simple", "current_split": "val",
        "vectorized_beam_search": {
            "val": {"beam_act": 4, "beam_max": 4},
            "test": {"beam_act": 4, "beam_max": 4},
            "top_k_final": 2, "neg_inf_fp32": -1e9, "neg_inf_fp16": -10000,
        },
    }
    if enabled is not None:
        config["item_local_attention"] = {
            "enabled": enabled, "num_heads": 4, "dropout": 0.0,
        }
    return config


def tiny_batch():
    codes = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    return {
        "history_sid": torch.tensor([
            [[-1, -1, -1, -1], [0, 1, 2, 3], [4, 5, 6, 7]],
            [[-1, -1, -1, -1], [-1, -1, -1, -1], [7, 6, 5, 4]],
        ]),
        "history_mask": torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.bool),
        "decoder_input_ids": codes,
        "decoder_labels": codes.clone(),
    }


def make_model(enabled=True, cls=DIFF_GRM):
    with redirect_stdout(io.StringIO()):
        return cls(tiny_config(enabled), object(), TinyTokenizer())


def activate_attention(module):
    with torch.no_grad():
        module.attention.out_proj.weight.copy_(torch.eye(module.embedding_dim))
        module.attention.out_proj.bias.zero_()


class ItemLocalBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_zero_initialization_is_exact_identity_and_correct_parameter_count(self):
        block = ItemLocalAttention(16, num_heads=4, dropout=0.0)
        x = torch.randn(2, 3, 4, 16)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        valid[0, 0] = False
        valid[1, 1, 2:] = False
        self.assertTrue(torch.equal(block(x, valid), x))
        full = ItemLocalAttention(256, num_heads=4, dropout=0.0)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 263680)

    def test_padding_is_finite_after_activation_and_does_not_leak(self):
        block = ItemLocalAttention(16, num_heads=4, dropout=0.0)
        activate_attention(block)
        x = torch.randn(2, 3, 4, 16, requires_grad=True)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        valid[0, 0] = False
        valid[1, 1, 2:] = False
        y = block(x, valid)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(y[~valid], x[~valid]))
        changed = x.detach().clone()
        changed[~valid] = 1e6
        torch.testing.assert_close(block(changed, valid)[valid], y[valid], rtol=0, atol=0)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        all_pad = torch.zeros_like(valid)
        self.assertTrue(torch.equal(block(x.detach(), all_pad), x.detach()))

    def test_attention_stays_inside_each_item_but_mixes_its_slots(self):
        block = ItemLocalAttention(16, num_heads=4, dropout=0.0).eval()
        activate_attention(block)
        x = torch.randn(2, 3, 4, 16)
        changed = x.clone()
        changed[0, 1, 0] += torch.linspace(-5, 5, 16)
        left, right = block(x), block(changed)
        unaffected = torch.ones(2, 3, dtype=torch.bool)
        unaffected[0, 1] = False
        self.assertTrue(torch.equal(left[unaffected], right[unaffected]))
        self.assertGreater(float((left[0, 1, 1:] - right[0, 1, 1:]).abs().sum()), 1e-6)

    def test_zero_output_projection_can_learn_then_routes_qkv_gradients(self):
        block = ItemLocalAttention(16, num_heads=4, dropout=0.0)
        x, target = torch.randn(2, 3, 4, 16), torch.randn(2, 3, 4, 16)
        optimizer = torch.optim.Adam(block.parameters(), lr=0.03)
        before = block.attention.in_proj_weight.detach().clone()
        first = block(x).detach().clone()
        qkv_gradients = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (block(x) - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            qkv_gradients.append(float(block.attention.in_proj_weight.grad.abs().sum()))
            optimizer.step()
        self.assertEqual(qkv_gradients[0], 0.0)
        self.assertGreater(qkv_gradients[1], 0.0)
        self.assertFalse(torch.equal(before, block.attention.in_proj_weight))
        self.assertFalse(torch.equal(first, block(x)))

    def test_bfloat16_autocast_padding_forward_and_backward_are_finite(self):
        block = ItemLocalAttention(16, num_heads=4, dropout=0.0)
        activate_attention(block)
        x = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        valid[0, 0] = False
        valid[1, 1, 2:] = False
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y = block(x, valid)
            loss = y.float().square().mean()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(y[~valid], x[~valid]))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class DiffGRMIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_disabled_path_matches_prechange_model_and_enabled_preserves_base_rng(self):
        previous_class = load_previous_model()
        torch.manual_seed(2024)
        previous = make_model(None, cls=previous_class)
        previous_rng = torch.get_rng_state().clone()
        for enabled in (None, False, True):
            torch.manual_seed(2024)
            current = make_model(enabled)
            self.assertTrue(torch.equal(previous_rng, torch.get_rng_state()))
            base_state = previous.state_dict()
            current_state = current.state_dict()
            for name, value in base_state.items():
                self.assertTrue(torch.equal(value, current_state[name]), name)
            extra = set(current_state) - set(base_state)
            if enabled:
                self.assertTrue(extra)
                self.assertTrue(all(name.startswith("item_local_attention.") for name in extra))
            else:
                self.assertEqual(extra, set())
                self.assertIsNone(current.item_local_attention)
            previous.eval()
            current.eval()
            with redirect_stdout(io.StringIO()):
                old_output = previous(tiny_batch())
                new_output = current(tiny_batch())
            self.assertTrue(torch.equal(old_output.loss, new_output.loss))
            torch.testing.assert_close(
                previous(tiny_batch(), return_loss=False).hidden_states,
                current(tiny_batch(), return_loss=False).hidden_states,
                rtol=0, atol=0,
            )

    def test_training_and_real_generation_use_module_before_item_mlp(self):
        model = make_model().eval()
        activate_attention(model.item_local_attention)
        inputs, mixed, mlp_inputs = [], [], []
        attention_hook = model.item_local_attention.register_forward_hook(
            lambda _module, args, output: (inputs.append(args[0].detach().clone()),
                                          mixed.append(output.detach().clone())) and None
        )
        mlp_hook = model.item_mlp.register_forward_pre_hook(
            lambda _module, args: mlp_inputs.append(args[0].detach().clone())
        )
        try:
            with redirect_stdout(io.StringIO()):
                train_output = model(tiny_batch())
                encoded = model(tiny_batch(), return_loss=False).hidden_states
                generated = model.generate(tiny_batch(), n_return_sequences=2)
            self.assertTrue(torch.isfinite(train_output.loss))
            self.assertEqual(len(inputs), 3)
            self.assertEqual(len(mlp_inputs), 3)
            for original, updated, compressed in zip(inputs, mixed, mlp_inputs):
                self.assertEqual(tuple(updated.shape), (2, 3, 4, 16))
                self.assertFalse(torch.equal(original[:, -1], updated[:, -1]))
                torch.testing.assert_close(compressed, updated.reshape(2, 3, 64), rtol=0, atol=0)
            codes = generated[0] if isinstance(generated, tuple) else generated
            self.assertEqual(tuple(codes.shape), (2, 2, 4))
            self.assertTrue(((codes >= 0) & (codes < 8)).all())
            # Already encoded caches are intentionally not encoded a second time.
            with redirect_stdout(io.StringIO()):
                cached = model.generate({"encoder_hidden": encoded}, n_return_sequences=2)
            self.assertEqual(len(inputs), 3)
            cached_codes = cached[0] if isinstance(cached, tuple) else cached
            self.assertTrue(torch.equal(codes, cached_codes))
        finally:
            attention_hook.remove()
            mlp_hook.remove()

    def test_recommendation_loss_trains_new_module_and_checkpoint_reloads(self):
        model = make_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = copy.deepcopy(model.item_local_attention.state_dict())
        with redirect_stdout(io.StringIO()):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = model(tiny_batch()).loss
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.embedding.weight.grad)
                self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.0)
                optimizer.step()
        self.assertFalse(torch.equal(
            initial["attention.out_proj.weight"],
            model.item_local_attention.attention.out_proj.weight,
        ))
        self.assertFalse(torch.equal(
            initial["attention.in_proj_weight"],
            model.item_local_attention.attention.in_proj_weight,
        ))
        model.eval()
        expected = model(tiny_batch(), return_loss=False).hidden_states.detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            restored = make_model().eval()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(payload["model"], strict=True)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.02)
            restored_optimizer.load_state_dict(payload["optimizer"])
            actual = restored(tiny_batch(), return_loss=False).hidden_states
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
            with redirect_stdout(io.StringIO()):
                before = model.generate(tiny_batch(), n_return_sequences=2)
                after = restored.generate(tiny_batch(), n_return_sequences=2)
            before = before[0] if isinstance(before, tuple) else before
            after = after[0] if isinstance(after, tuple) else after
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
