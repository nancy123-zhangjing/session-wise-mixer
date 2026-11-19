"""CPU checks against the real four-layer DiffGRM decoder, never a fake backbone."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm.model import ListwiseDiffGRM
from scripts.listwise_diffgrm.runtime import engine_module

torch.set_num_threads(2)
DIFF_GRM = engine_module('model').DIFF_GRM


class TinyTokenizer:
    sid_offset = 3
    vocab_size = 3 + 4 * 8
    mask_token = -1


def configuration(route='J', k=5):
    return {
        'architecture': route, 'list_length': k, 'observed_history_max': 3,
        'n_digit': 4, 'codebook_size': 8, 'n_embd': 16, 'n_head': 4,
        'n_inner': 32, 'encoder_n_layer': 1, 'decoder_n_layer': 4,
        'dropout': 0.0, 'attn_pdrop': 0.0, 'resid_pdrop': 0.0,
        'max_history_len': 3 + k - 1, 'label_smoothing': 0.2,
        'masking_strategy': 'guided', 'guided_steps': 4, 'guided_select': 'least',
        'guided_conf_metric': 'msp', 'guided_refresh_each_step': False,
        'diffusion_loss_reduction': 'per_view', 'joint_crosspse_enabled': False,
        'item_local_attention': {'enabled': False}, 'current_split': 'val',
        'dedup_strategy': 'simple',
        'vectorized_beam_search': {
            'val': {'beam_act': 4, 'beam_max': 4},
            'test': {'beam_act': 4, 'beam_max': 4}, 'top_k_final': 2,
            'neg_inf_fp32': -1e9, 'neg_inf_fp16': -10000,
        },
    }


def make_model(route='J', k=5):
    cfg = configuration(route, k)
    with redirect_stdout(io.StringIO()):
        core = DIFF_GRM(cfg, None, TinyTokenizer())
    core._debug_printed = True
    return ListwiseDiffGRM(cfg, core)


def example(k=5):
    history = torch.tensor([
        [[-1, -1, -1, -1], [0, 1, 2, 3], [4, 5, 6, 7]],
        [[-1, -1, -1, -1], [-1, -1, -1, -1], [7, 6, 5, 4]],
    ])
    mask = history[:, :, 0].ne(-1)
    target = (torch.arange(2 * k * 4).reshape(2, k, 4) + 1) % 8
    return history, mask, target


class ListwiseModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_k1_loss_and_parameter_gradients_match_original_for_both_routes(self):
        history, mask, target = example(1)
        for route in ('J', 'H'):
            model = make_model(route, 1).train()
            original = copy.deepcopy(model.backbone)
            result = model.forward_loss(history, mask, target)
            reference = original({'history_sid': history, 'history_mask': mask,
                                  'decoder_input_ids': target[:, 0],
                                  'decoder_labels': target[:, 0]})
            torch.testing.assert_close(result['loss'], reference.loss, atol=3e-7, rtol=1e-6)
            result['loss'].backward()
            reference.loss.backward()
            for (name, parameter), (ref_name, ref) in zip(
                    model.backbone.named_parameters(), original.named_parameters()):
                self.assertEqual(name, ref_name)
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-7, rtol=3e-5,
                                           msg=lambda msg: f'{route}/{name}: {msg}')

    def test_k1_projected_inference_matches_original_cached_path(self):
        history, mask, _ = example(1)
        for route in ('J', 'H'):
            model = make_model(route, 1).eval()
            partial = torch.tensor([[-1, 2, -1, 4], [3, -1, 1, -1]])
            encoded = model.backbone({'history_sid': history, 'history_mask': mask},
                                     return_loss=False).hidden_states
            reference = model.backbone.forward_decoder_only({
                'encoder_hidden': encoded, 'decoder_input_ids': partial.clamp_min(0),
                'mask_positions': partial.eq(-1)}, use_cache=True).logits
            actual = model.decode_logits(history, mask, partial,
                                         item_index=0 if route == 'H' else None)
            torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)

    def test_k1_st_loss_and_assignment_gradients_match_original(self):
        history, mask, target = example(1)
        model = make_model('J', 1).train()
        original = copy.deepcopy(model.backbone)
        hs = F.one_hot(history.clamp_min(0), 8).float().requires_grad_()
        ts = F.one_hot(target, 8).float().requires_grad_()
        rh, rt = hs.detach().clone().requires_grad_(), ts.detach().clone().requires_grad_()
        actual = model.forward_loss(history, mask, target, hs, ts)['loss']
        reference = original({'history_sid': history, 'history_mask': mask,
                              'history_item_ids': mask.long(), 'history_sid_st': rh,
                              'target_sid_st': rt[:, 0], 'decoder_input_ids': target[:, 0],
                              'decoder_labels': target[:, 0]}).loss
        actual.backward()
        reference.backward()
        torch.testing.assert_close(actual, reference, atol=3e-7, rtol=1e-6)
        torch.testing.assert_close(hs.grad, rh.grad, atol=2e-7, rtol=3e-5)
        torch.testing.assert_close(ts.grad, rt.grad, atol=2e-7, rtol=3e-5)

    def test_j_uses_four_shared_heads_and_distinguishable_item_positions(self):
        history, mask, _ = example()
        model = make_model('J').eval()
        partial = torch.full((2, 20), -1)
        actual = model.decode_logits(history, mask, partial)
        self.assertEqual(actual.shape, (2, 20, 8))
        self.assertEqual(model.backbone.n_digit, 4)
        self.assertEqual(model.backbone.mask_emb_table.weight.shape, (4, 16))
        self.assertEqual(model.backbone.embedding.weight.shape, (35, 16))
        self.assertGreater(float((actual[:, :4] - actual[:, 4:8]).abs().sum()), 1e-5)
        with torch.no_grad():
            model.item_position_embedding.weight.zero_()
        symmetric = model.decode_logits(history, mask, partial)
        torch.testing.assert_close(symmetric[:, :4], symmetric[:, 4:8], atol=1e-7, rtol=1e-6)

    def test_j_masks_have_full_view_nested_depths_and_expected_budget(self):
        history, mask, target = example()
        model = make_model('J')
        result = model.forward_loss(history, mask, target, mask_depths=[2, 7, 14, 20])
        diag = result['diagnostics']
        self.assertEqual(diag['mask_depths'].shape, (2, 4))
        self.assertEqual(diag['ocn_order'].shape, (2, 20))
        self.assertEqual(diag['supervised_positions'], 2 * (2 + 7 + 14 + 20))
        self.assertTrue(bool((diag['masked_positions'][:, :-1] <= diag['masked_positions'][:, 1:]).all()))
        self.assertTrue(bool(diag['masked_positions'][:, -1].all()))
        sampled = model._depths(20000, 20, torch.device('cpu'))
        self.assertLess(abs(float(sampled.sum(-1).float().mean()) - 50), 0.3)
        self.assertEqual(set(sampled[:, :3].flatten().tolist()), set(range(1, 20)))

    def test_h_only_sees_true_prefix_and_keeps_original_history(self):
        history, mask, target = example()
        model = make_model('H').eval()
        original_encode = model._encode
        inputs = []
        def capture(sid, history_mask, history_st=None):
            inputs.append((sid.detach().clone(), history_mask.detach().clone()))
            return original_encode(sid, history_mask, history_st)
        with patch.object(model, '_encode', side_effect=capture):
            first = model.forward_loss(history, mask, target)
        for j, (sid, observed_mask) in enumerate(inputs):
            self.assertEqual(sid.shape[1], history.shape[1] + j)
            self.assertTrue(torch.equal(sid[:, :3], history))
            self.assertTrue(torch.equal(sid[:, 3:], target[:, :j]))
            self.assertTrue(bool(observed_mask[:, 3:].all()))
        changed = target.clone()
        changed[:, -1] = (changed[:, -1] + 3) % 8
        later = model.forward_loss(history, mask, changed)
        torch.testing.assert_close(first['diagnostics']['per_item_loss'][:, :4],
                                   later['diagnostics']['per_item_loss'][:, :4], rtol=0, atol=0)
        self.assertEqual(first['diagnostics']['supervised_positions'], 2 * 50)
        torch.testing.assert_close(first['per_list_loss'],
                                   first['diagnostics']['per_item_loss'].mean(1), rtol=0, atol=0)

    def test_h_prefix_retains_st_gradient_with_only_final_item_loss(self):
        history, mask, target = example()
        model = make_model('H').eval()
        model.requires_grad_(False)
        target_st = F.one_hot(target, 8).float().requires_grad_()
        original_loss = model._condition_loss
        def final_only(*args, **kwargs):
            losses, report = original_loss(*args, **kwargs)
            return (losses if kwargs['item_index'] == 4 else losses * 0), report
        with patch.object(model, '_condition_loss', side_effect=final_only):
            loss = model.forward_loss(history, mask, target, target_st=target_st)['loss']
        loss.backward()
        self.assertGreater(float(target_st.grad[:, :4].abs().sum()), 1e-7)
        self.assertGreater(float(target_st.grad[:, 4].abs().sum()), 1e-7)

    def test_frozen_recommender_st_inputs_still_receive_gradients(self):
        history, mask, target = example()
        for route in ('J', 'H'):
            model = make_model(route).eval().requires_grad_(False)
            hs = F.one_hot(history.clamp_min(0), 8).float().requires_grad_()
            ts = F.one_hot(target, 8).float().requires_grad_()
            model.forward_loss(history, mask, target, hs, ts)['loss'].backward()
            self.assertGreater(float(hs.grad[mask].abs().sum()), 1e-7)
            self.assertGreater(float(ts.grad.abs().sum()), 1e-7)
            self.assertEqual(float(hs.grad[~mask].abs().sum()), 0.0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_bf16_ocn_preserves_gradients_in_all_real_decoder_layers(self):
        history, mask, target = example()
        for route in ('J', 'H'):
            model = make_model(route).train()
            self.assertEqual(len(model.backbone.decoder_blocks), 4)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = model.forward_loss(history, mask, target)['loss']
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, f'{route}/{name}')
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0, f'{route}/{name}')
                self.assertEqual(parameter.device.type, 'cpu')

    def test_ocn_restores_each_module_mode(self):
        history, mask, target = example()
        model = make_model('J').train()
        model.backbone.decoder_blocks[2].eval()
        before = [module.training for module in model.modules()]
        model.forward_loss(history, mask, target)
        self.assertEqual(before, [module.training for module in model.modules()])

    def test_explicit_context_reuse_and_parent_reordering_match_uncached(self):
        history, mask, _ = example()
        for route in ('J', 'H'):
            model = make_model(route).eval()
            positions = 20 if route == 'J' else 4
            index = None if route == 'J' else 0
            partial = torch.full((2, positions), -1)
            partial[0, 1] = 3
            partial[1, 2] = 4
            context = model.encode_context(history, mask)
            expected = model.decode_logits(history, mask, partial, item_index=index)
            # One parent may survive as multiple children, another may disappear.
            parents = torch.tensor([1, 0, 1])
            reused = model.decode_with_context(context[parents], partial[parents], item_index=index)
            torch.testing.assert_close(reused, expected[parents], rtol=1e-6, atol=1e-7)
            with patch.object(model, '_encode', side_effect=AssertionError('must reuse context')):
                model.decode_with_context(context, partial, item_index=index)

    def test_expansion_preserves_old_position_weights(self):
        cfg = configuration('H')
        cfg['max_history_len'] = 3
        with redirect_stdout(io.StringIO()):
            core = DIFF_GRM(cfg, None, TinyTokenizer())
        before = core.pos_emb_enc.weight.detach().clone()
        model = ListwiseDiffGRM(cfg, core)
        self.assertEqual(model.backbone.pos_emb_enc.num_embeddings, 7)
        self.assertTrue(torch.equal(before, model.backbone.pos_emb_enc.weight[:3]))

    def test_invalid_shapes_and_configs_fail_loudly(self):
        history, mask, target = example()
        model = make_model('J')
        with self.assertRaisesRegex(ValueError, 'assignments'):
            model.forward_loss(history, mask, target, target_st=torch.zeros(2, 5, 4, 16))
        with self.assertRaisesRegex(ValueError, 'depths'):
            model.forward_loss(history, mask, target, mask_depths=[1, 1, 2, 20])
        with self.assertRaisesRegex(ValueError, 'visible historical'):
            model.decode_logits(torch.full_like(history, -1), torch.zeros_like(mask),
                                torch.full((2, 20), -1))
        with self.assertRaisesRegex(ValueError, 'item_index'):
            make_model('H').decode_logits(history, mask, torch.full((2, 4), -1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
