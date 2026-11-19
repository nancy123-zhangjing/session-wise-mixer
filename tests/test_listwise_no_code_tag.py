"""Synthetic CPU checks only; no dataset training, checkpoints or GPU jobs."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm.runtime import engine_module, read_config, validate_config
from test_listwise_training import fixture

Mixer = engine_module('cross_pse').CrossPSECodebookMixer


class NoCodeTagTests(unittest.TestCase):
    def test_disabled_ignores_loaded_nonzero_tags_and_has_no_tag_gradient(self):
        torch.manual_seed(7)
        model = Mixer(torch.randn(4, 8, 8), n_heads=2,
                      train_base_codebooks=True, residual_scale_init=1.,
                      code_embedding_enabled=False)
        with torch.no_grad():
            model.book_embedding.normal_()
        before = model.crossed_codebooks()[0]
        with torch.no_grad():
            model.code_embedding.normal_(0, 10)
        after = model.crossed_codebooks()[0]
        self.assertTrue(torch.equal(before, after))
        after.square().mean().backward()
        self.assertIsNone(model.code_embedding.grad)
        for parameter in (model.book_embedding, model.base_codebooks,
                          model.attention.in_proj_weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Legacy keys can be strictly loaded; disabled behavior survives reload.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = Mixer(torch.zeros(4, 8, 8), n_heads=2,
                         train_base_codebooks=True, residual_scale_init=1.,
                         code_embedding_enabled=False)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertTrue(torch.equal(after, restored.crossed_codebooks()[0]))

    def test_legacy_default_and_initial_rng_are_preserved(self):
        base = torch.arange(256, dtype=torch.float32).reshape(4, 8, 8) / 100
        torch.manual_seed(12)
        old = Mixer(base, n_heads=2, residual_scale_init=1.)
        rng_old = torch.get_rng_state()
        torch.manual_seed(12)
        new = Mixer(base, n_heads=2, residual_scale_init=1., code_embedding_enabled=False)
        self.assertTrue(torch.equal(rng_old, torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        self.assertTrue(torch.equal(old.crossed_codebooks()[0], new.crossed_codebooks()[0]))
        with torch.no_grad():
            old.code_embedding.normal_()
        self.assertFalse(torch.equal(old.crossed_codebooks()[0], new.crossed_codebooks()[0]))

    def test_alignment_phase_switch_and_optimizer_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            old, _ = fixture(directory, route='H')
            new, _ = fixture(directory, route='H', code_embedding_enabled=False)
            self.assertFalse(new.alignment.mixer.code_embedding_enabled)
            self.assertEqual([len(g['params']) for g in old.t_optimizer.param_groups],
                             [len(g['params']) for g in new.t_optimizer.param_groups])
            for phase in ('T', 'R', 'T'):
                new.phase_mode(phase)
                self.assertFalse(new.alignment.mixer.code_embedding.requires_grad)
                self.assertEqual(new.alignment.mixer.book_embedding.requires_grad, phase == 'T')
            # Current shared-start state has zero tags and exactly equal output.
            self.assertTrue(torch.equal(old.alignment.mixer.crossed_codebooks()[0],
                                        new.alignment.mixer.crossed_codebooks()[0]))

    def test_config_is_single_switch_and_rejects_string_false(self):
        base = read_config(ROOT / 'configs/09_kuairec_listwise/base.json')
        ablation = read_config(ROOT / 'configs/09_kuairec_listwise/h1_no_code_tag.json')
        expected = copy.deepcopy(base)
        expected['alignment']['crosspse_code_embedding_enabled'] = False
        self.assertEqual(ablation, expected)
        ablation['alignment']['crosspse_code_embedding_enabled'] = 'false'
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            validate_config(ablation)


if __name__ == '__main__':
    unittest.main()
