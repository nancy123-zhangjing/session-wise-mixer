"""Checks for the CF round-1 screening switches (scripts/rqcd_mixer/screen.py, scripts/cf_screen)."""
import copy
import json
import unittest
from pathlib import Path

import torch

from scripts.cf_screen.variants import JOBS, VARIANTS, build, own_parent
from scripts.rqcd_mixer.losses import recommendation_losses
from scripts.rqcd_mixer.model import Recommender
from scripts.rqcd_mixer.screen import asset_view, generation_dim, validate

ROOT = Path(__file__).resolve().parents[1]


def base_config():
    cfg = json.loads((ROOT/'configs/rqcd_mixer/v2_cf.json').read_text(encoding='utf-8'))
    cfg['calibrated_teacher'] = {'stub': True}  # stands in for the first-wave calibration block
    return cfg


def model_for(cfg, seed=0):
    torch.manual_seed(seed)
    return Recommender(cfg).eval()


class VariantTest(unittest.TestCase):
    def test_eight_groups_one_change_each(self):
        base = base_config()
        self.assertEqual(sorted(sum(JOBS.values(), [])), sorted(VARIANTS))
        self.assertEqual(len(VARIANTS), 8)
        for name in VARIANTS:
            cfg = build(name, base)
            self.assertEqual(cfg['screen']['name'], name)
            if own_parent(name):
                self.assertNotEqual(asset_view(cfg), base)
                self.assertEqual({k for k in base if base[k] != cfg[k]}, {'rq'})
            else:
                # Training-only switches: the prepared parent and RQ asset are reusable.
                self.assertEqual(asset_view(cfg), base)
        self.assertEqual(build('B0', base)['screen'], {'name': 'B0'})
        self.assertEqual(build('E128', base)['rq']['latent_dim'], 128)
        self.assertEqual(build('E64', base)['rq']['hidden_dims'], [512, 256, 128])
        self.assertEqual(generation_dim(build('D1', base)), 256)
        self.assertEqual(generation_dim(build('H1', base)), 32)

    def test_invalid_switches_are_rejected(self):
        base = base_config()
        for bad in ({'landing_weight': 0}, {'unknown': 1}, {'guidance': {'p_uncond': 1.0}},
                    {'history_head': {'weight': 0.5}}, {'generation_space': 'content'}):
            with self.assertRaises(ValueError):
                validate({**base, 'screen': bad})
        with self.assertRaises(ValueError):
            build('L1', {**base, 'screen': {}})


class NoiseTest(unittest.TestCase):
    def test_extra_noise_follows_the_generation_width(self):
        """Smoke-test finding (2026-11-18): the extra evaluation noise must be 256-d for D1."""
        import tempfile
        from scripts.cf_decision.run import extra_noise
        base = base_config()
        for name, dim in (('B0', 32), ('D1', 256), ('E128', 128)):
            with tempfile.TemporaryDirectory() as tmp:
                table, _ = extra_noise(build(name, base), ['u1', 'u2'], 'validation', 3, Path(tmp))
                self.assertEqual(table.shape, (2, 2, base['list_length'], dim))


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.base = base_config()

    def test_b0_model_is_the_original_model(self):
        original, screened = model_for(self.base), model_for(build('B0', self.base))
        a, b = original.state_dict(), screened.state_dict()
        self.assertEqual(list(a), list(b))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertEqual(screened.trainable_parameters(), list(screened.parameters()))

    def test_new_modules_come_after_the_original_initialization(self):
        original = model_for(self.base).state_dict()
        for name in ('H1', 'H2'):
            screened = model_for(build(name, self.base)).state_dict()
            self.assertTrue(set(original) < set(screened))
            self.assertTrue(all(torch.equal(original[k], screened[k]) for k in original))
        d1 = model_for(build('D1', self.base))
        self.assertEqual(d1.dim, 256)
        self.assertEqual(d1.output.weight.shape, (256, d1.width))
        self.assertEqual(d1.fixed_parameter_names(), {n for n, _ in d1.named_parameters() if n.startswith('to_latent.')})
        self.assertTrue(all(not p.requires_grad for p in d1.to_latent.parameters()))

    def _batch(self, model, batch=3, seed=1):
        g = torch.Generator().manual_seed(seed)
        hist = torch.randn(batch, self.base['history_max'], model.levels, model.width, generator=g)
        valid = torch.ones(batch, self.base['history_max'], dtype=torch.bool)
        valid[:, :7] = False
        tgt = torch.randn(batch, model.length, model.levels, model.width, generator=g)
        noisy = torch.randn(batch, model.length, model.dim, generator=g)
        times = torch.randint(1, 1000, (batch, model.length), generator=g)
        return hist, valid, tgt, noisy, times

    def test_dropped_history_equals_the_null_context(self):
        model = model_for(build('H2', self.base))
        hist, valid, tgt, noisy, times = self._batch(model)
        drop = torch.ones(len(hist), dtype=torch.bool)
        with torch.no_grad():
            trained = model.training_prediction(hist, valid, tgt, noisy, times, uncond=drop)[:, 0]
            null = model.predict_uncached(noisy[:, 0], times[:, 0], tgt[:, :0], *model.null_context(len(hist), 'cpu'))
        self.assertTrue(torch.allclose(trained, null, atol=1e-5))
        keep = torch.zeros(len(hist), dtype=torch.bool)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.training_prediction(hist, valid, tgt, noisy, times, uncond=keep),
                                        model.training_prediction(hist, valid, tgt, noisy, times)))

    def test_history_head_pools_valid_tokens_and_trains(self):
        model = model_for(build('H1', self.base)).train()
        hist, valid, tgt, noisy, times = self._batch(model)
        items = torch.randn(40, self.base['rq']['input_dim'])
        _, memory, mask = model.training_prediction(hist, valid, tgt, noisy, times, return_memory=True)
        logits = model.history_logits(memory, mask, items, 0.1)
        self.assertEqual(logits.shape, (3, 40))
        self.assertLessEqual(logits.abs().max().item(), 10.0 + 1e-4)  # cosine / temperature
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5, 9])).backward()
        self.assertIsNotNone(model.history_query.grad)
        self.assertIsNotNone(model.encoder[0].self_attn.q.weight.grad)
        changed = memory.detach().clone()
        changed[:, :7] = 1e3  # invalid positions must not matter
        self.assertTrue(torch.allclose(model.history_logits(changed, mask, items, 0.1), logits, atol=1e-4))


class LossTest(unittest.TestCase):
    def setUp(self):
        self.base = base_config()
        g = torch.Generator().manual_seed(3)
        self.prediction = torch.randn(4, 5, 32, generator=g)
        self.clean = torch.randn(4, 5, 32, generator=g)
        self.stats = {'scale': torch.tensor(0.7), 'mean': torch.randn(32, generator=g)}
        self.books = torch.randn(4, 256, 32, generator=g)
        self.sid = torch.randint(0, 256, (4, 5, 4), generator=g)
        self.target_z = torch.randn(4, 5, 32, generator=g)

    def _losses(self, cfg, **kw):
        cfg = copy.deepcopy(cfg)
        cfg['loss']['mode'] = 'align'  # the landing tree needs a catalog; the weighting logic is shared
        cfg['loss']['align'] = 0.1
        return recommendation_losses(self.prediction, self.clean, self.target_z, self.sid, self.books,
                                     {**self.stats, 'distance_scale': torch.ones(4)}, cfg, **kw)

    def test_b0_total_is_the_original_expression(self):
        result = self._losses(build('B0', self.base))
        self.assertTrue(torch.equal(result['total'], result['denoise'] + 0.1*result['align']))
        original = self._losses(self.base)
        self.assertTrue(torch.equal(result['total'], original['total']))

    def test_screen_weights_and_extra_terms(self):
        cfg = build('L2', self.base)
        cfg['screen']['align_weight'] = cfg['screen'].pop('landing_weight')
        result = self._losses({**cfg, 'screen': cfg['screen']})
        self.assertTrue(torch.allclose(result['total'], 0.1*result['denoise'] + 1.0*result['align']))
        extra = self._losses(self.base, extra={'history': (0.5, torch.tensor(2.0))})
        self.assertTrue(torch.allclose(extra['total'], extra['denoise'] + 0.1*extra['align'] + 1.0))
        self.assertEqual(float(extra['history']), 2.0)

    def test_to_latent_replaces_the_affine_map(self):
        mapped = self._losses(self.base, to_latent=lambda x: x*self.stats['scale'] + self.stats['mean'])
        self.assertTrue(torch.equal(mapped['total'], self._losses(self.base)['total']))


if __name__ == '__main__':
    unittest.main()
