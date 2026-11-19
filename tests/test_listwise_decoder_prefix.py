"""Real-backbone CPU checks for both decoder-prefix routes, no dataset runs."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from tests.test_listwise_model import configuration, DIFF_GRM, TinyTokenizer, example
from tests.test_listwise_training import fixture, state
from scripts.listwise_diffgrm.model import ListwiseDiffGRM
from scripts.listwise_diffgrm.prefix_decoder import asymmetric_mask, decoder_layer
from scripts.listwise_diffgrm.runtime import build_model, validate_config
from scripts.listwise_diffgrm.training import AlternatingTrainer
from scripts.listwise_diffgrm.decoding import SIDCatalog, generate_lists

ROOT = Path(__file__).resolve().parents[1]
MODES = ('asymmetric', 'cross_attention')


def model_for(mode):
    cfg = configuration('H')
    cfg['h_prefix_mode'] = mode
    with redirect_stdout(io.StringIO()):
        core = DIFF_GRM(cfg, None, TinyTokenizer())
    core._debug_printed = True
    return ListwiseDiffGRM(cfg, core)


class PrefixTests(unittest.TestCase):
    def test_optimized_checks_scope_cache_and_invalid_inputs(self):
        history, mask, target = example()
        for mode in MODES:
            model = model_for(mode)
            model.config['optimized_model_checks'] = True
            result = model.forward_loss(history, mask, target)
            self.assertEqual(result['diagnostics']['supervised_positions'], 100)
            self.assertFalse(model._sid_inputs_validated)
            constants = dict(model._execution_constants)
            model.forward_loss(history, mask, target)
            self.assertTrue(all(model._execution_constants[k] is v for k,v in constants.items()))
            self.assertTrue(all(bool(v.eq(-1).all()) for k,v in constants.items() if k[0]=='ocn_full_mask'))
            bad = target.clone(); bad[0,0,0] = 8
            with self.assertRaises(ValueError): model.forward_loss(history, mask, bad)
            bad_history = history.clone(); bad_history[0,0,0] = 0
            with self.assertRaises(ValueError): model.forward_loss(bad_history, mask, target)
            with self.assertRaises(ValueError): model._embed_prefix(bad[:,:1], 1)
            with patch.object(model, '_condition_loss', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError): model.forward_loss(history, mask, target)
            self.assertFalse(model._sid_inputs_validated)

    def setUp(self):
        torch.manual_seed(2024)

    def test_empty_prefix_matches_original_and_checkpoint_roundtrip(self):
        history, mask, _ = example()
        partial = torch.full((2, 4), -1)
        original = model_for('history').eval()
        for mode in MODES:
            model = model_for(mode).eval()
            missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all(k.startswith('prefix_cross_layers.') for k in missing))
            expected = original.decode_logits(history, mask, partial, item_index=0)
            actual = model.decode_logits(history, mask, partial, item_index=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            clone = model_for(mode).eval()
            clone.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(clone.decode_logits(history, mask, partial, item_index=0), actual)

    def test_history_once_teacher_prefix_and_only_current_supervision(self):
        history, mask, target = example()
        for mode in MODES:
            model = model_for(mode)
            with patch.object(model, '_encode', wraps=model._encode) as enc, \
                    patch.object(model, '_decode', wraps=model._decode) as dec:
                result = model.forward_loss(history, mask, target)
            self.assertEqual(enc.call_count, 1)
            self.assertEqual(enc.call_args.args[0].shape[1], 3)
            self.assertEqual(dec.call_count, 10)  # OCN + student per item
            for j in range(5):
                teacher, student = dec.call_args_list[2*j:2*j+2]
                self.assertTrue(teacher.kwargs['teacher'])
                self.assertTrue(teacher.args[1].eq(-1).all())
                torch.testing.assert_close(teacher.kwargs['prefix_sid'], target[:, :j])
                torch.testing.assert_close(student.kwargs['prefix_sid'], target[:, :j].repeat(4, 1, 1))
            self.assertEqual(result['diagnostics']['supervised_positions'], 2 * 5 * 10)
            self.assertEqual(result['diagnostics']['per_item_loss'].shape, (2, 5))
            result['loss'].backward()
            self.assertTrue(torch.isfinite(result['loss']))

    def test_asymmetric_prefix_independent_of_current_at_every_layer(self):
        model = model_for('asymmetric').eval()
        core = model.backbone
        encoded = torch.randn(2, 3, 16)
        x = torch.randn(2, 12, 16)
        y = x.clone()
        y[:, 8:] = torch.randn_like(y[:, 8:]) * 4
        mask = asymmetric_mask(8, 4, x.device)
        self.assertTrue(mask[0, :8, :8].all())  # earlier prefix may read later prefix
        self.assertFalse(mask[0, :8, 8:].any())
        for block in core.decoder_blocks:
            kv = block.cross_attn.qkv(encoded)[..., 16:]
            x = decoder_layer(block, x, encoded, kv, mask)
            y = decoder_layer(block, y, encoded, kv, mask)
            torch.testing.assert_close(x[:, :8], y[:, :8], atol=0, rtol=0)

    def test_prefix_has_effect_and_st_gradient_without_changing_sid(self):
        history, mask, target = example()
        for mode in MODES:
            model = model_for(mode).eval()
            encoded = model._encode(history, mask)
            prefix = target[:, :2].clone()
            snapshot = prefix.clone()
            assignment = F.one_hot(prefix, 8).float().requires_grad_()
            partial = torch.full((2, 4), -1)
            logits = model._decode(encoded, partial, item_index=2,
                                   prefix_sid=prefix, prefix_st=assignment)
            hard = model._decode(encoded, partial, item_index=2, prefix_sid=prefix)
            torch.testing.assert_close(logits, hard)
            changed = model._decode(encoded, partial, item_index=2, prefix_sid=(prefix + 1) % 8)
            self.assertGreater((hard - changed).abs().max().item(), 1e-7)
            logits.square().sum().backward()
            self.assertTrue(torch.isfinite(assignment.grad).all())
            self.assertGreater(assignment.grad.abs().sum().item(), 0)
            torch.testing.assert_close(prefix, snapshot)
            if mode == 'cross_attention':
                self.assertTrue(all(layer.attention.qkv.weight.grad.abs().sum() > 0
                                    for layer in model.prefix_cross_layers))
            with self.assertRaises(ValueError):
                model.decode_with_context(encoded, partial, item_index=2)

    def test_cached_beam_matches_uncached_reference_with_branch_prefixes(self):
        history, mask, _ = example()
        codes = np.array([[i, (i+1) % 8, (i+2) % 8, (i+3) % 8] for i in range(8)])
        catalog = SIDCatalog(np.arange(1, 9), codes, np.zeros(8), codebook_size=8)
        for mode in MODES:
            model = model_for(mode).eval()

            class Uncached(torch.nn.Module):
                def decode_logits(self, hist, valid, partial, item_index=None):
                    return model.decode_logits(hist[:, :3], valid[:, :3], partial,
                                               item_index=item_index, prefix_sid=hist[:, 3:])

            with patch.object(model, '_encode', wraps=model._encode) as enc:
                cached = generate_lists(model, history, mask, catalog, route='H', list_length=5, beam_size=3)
            self.assertEqual(enc.call_count, 1)
            reference = generate_lists(Uncached(), history, mask, catalog, route='H', list_length=5, beam_size=3)
            torch.testing.assert_close(cached.sid, reference.sid)
            torch.testing.assert_close(cached.scores, reference.scores)
            self.assertTrue(cached.complete_mask[cached.beam_mask].all())
            self.assertTrue(cached.complete_mask.any(dim=1).all())

    def test_h0_h1_real_r_t_updates_and_resume(self):
        for mode in MODES:
            for mixer in (False, True):
                with self.subTest(mode=mode, mixer=mixer), tempfile.TemporaryDirectory() as tmp:
                    with redirect_stdout(io.StringIO()):
                        old, batch = fixture(tmp, 'H', mixer)
                    cfg = copy.deepcopy(old.config)
                    cfg['model']['h_prefix_mode'] = mode
                    with redirect_stdout(io.StringIO()):
                        model = build_model(cfg, old.model.backbone.tokenizer)
                    model.backbone._debug_printed = True
                    trainer = AlternatingTrainer(model, old.alignment, cfg, 4, precision='bf16')
                    before_t = state(trainer.alignment)
                    trainer.update([batch], 'R')
                    self.assertTrue(all(torch.equal(v, before_t[k]) for k, v in state(trainer.alignment).items()))
                    before_r = state(model)
                    # Isolate recommendation gradient through the new architecture.
                    cfg['training'].update(reconstruction_weight=0, balance_weight=0, anchor_weight=0)
                    trainer.update([batch], 'T')
                    self.assertTrue(all(torch.equal(v, before_r[k]) for k, v in state(model).items()))
                    self.assertFalse(torch.equal(trainer.alignment.mixer.base_codebooks, before_t['mixer.base_codebooks']))
                    changed = [k for k, v in state(trainer.alignment).items()
                               if k != 'mixer.base_codebooks' and not torch.equal(v, before_t[k])]
                    self.assertEqual(bool(changed), mixer)
                    arm = 'H1' if mixer else 'H0'
                    binding = {'test': mode, 'config': cfg}
                    saved = trainer.checkpoint(binding, {'phase': 1, 'update': 1}, arm)
                    clone = copy.deepcopy(trainer)
                    clone.restore(saved, binding, arm=arm)
                    trainer.update([batch], 'R')
                    clone.update([batch], 'R')
                    for key, value in state(model).items():
                        torch.testing.assert_close(value, state(clone.model)[key], rtol=0, atol=0)

    def test_four_configs_and_invalid_route(self):
        for arm in ('h0', 'h1'):
            for variant in ('asymmetric', 'cross_attention'):
                cfg = json.loads((ROOT / f'configs/09_kuairec_listwise/{arm}_prefix_{variant}.json').read_text())
                validate_config(cfg)
                cfg['architecture'] = 'J'
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_future_targets_cannot_change_earlier_item_losses(self):
        history, mask, target = example()
        changed = target.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 8
        for mode in MODES:
            model = model_for(mode).eval()
            original = model.forward_loss(history, mask, target)['diagnostics']['per_item_loss']
            altered = model.forward_loss(history, mask, changed)['diagnostics']['per_item_loss']
            torch.testing.assert_close(original[:, :3], altered[:, :3], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                model.encode_context(torch.cat((history, target[:, :1]), 1),
                                     torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), 1))

    def test_extra_initialization_preserves_shared_parameters_and_rng(self):
        torch.manual_seed(17)
        a = model_for('asymmetric')
        rng_a = torch.get_rng_state()
        torch.manual_seed(17)
        b = model_for('cross_attention')
        torch.testing.assert_close(torch.get_rng_state(), rng_a, rtol=0, atol=0)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
