"""Checks for the inference-only CF decision rules (scripts/cf_decision)."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.cf_decision.rules import (DISCOUNT, LandingScorer, first_slot_vectors, generate_lists, generate_mode,
                                       grouped_logsumexp, mbr_select)
from scripts.rqcd_mixer.catalog import Catalog, distinct_history_ids, generate
from scripts.rqcd_mixer.losses import landing_loss
from scripts.rqcd_mixer.model import Diffusion, Recommender
from scripts.rqcd_mixer.tokenizer import hard_quantize

ROOT = Path(__file__).resolve().parents[1]
CFG = {'loss': {'mode': 'rq_landing', 'bucket_temperature': 0.5}, 'quantization': {'temperature': 0.7}}


def toy_catalog(seed=0, n=60, levels=3, codes=4, dim=3):
    g = torch.Generator().manual_seed(seed)
    spread = 0.5 ** torch.arange(levels, dtype=torch.float32)
    books = torch.randn(levels, codes, dim, generator=g) * spread[:, None, None]
    z = torch.randn(n, dim, generator=g)
    asset = dict(z=z, books=books, sid=hard_quantize(z, books)['sid'], original_ids=torch.arange(n) + 100,
                 mean=torch.zeros(dim), scale=torch.tensor(1.0), distance_scale=0.3 ** torch.arange(levels).float())
    return Catalog(asset, 'cpu')


class ScorerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.catalog = toy_catalog()
        self.scorer = LandingScorer(self.catalog, CFG)

    def test_distribution_is_normalised_and_respects_blocking(self):
        z = torch.randn(6, self.catalog.dim)
        blocked = torch.tensor([[0, 0], [3, 0], [3, 3], [5, 17], [1, 2], [60, 59]])
        log_p = self.scorer.log_prob(z, blocked)
        self.assertTrue(torch.allclose(log_p.exp().sum(-1), torch.ones(6, dtype=torch.float64), atol=1e-9))
        for row, items in enumerate(blocked.tolist()):
            for item in items:
                if item:
                    self.assertEqual(log_p[row, item - 1].item(), -math.inf)
        self.assertTrue(torch.isfinite(log_p[0]).all())

    def test_matches_landing_loss_without_exclusions(self):
        z = torch.randn(8, self.catalog.dim)
        target = torch.randint(1, self.catalog.count + 1, (8,))
        ours = -self.scorer.log_prob(z, target.new_zeros(8, 0)).gather(1, (target - 1)[:, None]).squeeze(1)
        for row in range(8):
            ref = landing_loss(z[row:row + 1, None], target[row:row + 1, None], self.catalog.books, self.catalog, CFG)
            self.assertAlmostEqual(ours[row].item(), ref['landing'].item(), places=4)

    def test_matches_landing_loss_with_an_excluded_earlier_item(self):
        z = torch.randn(5, 2, self.catalog.dim)
        target = torch.stack((torch.arange(1, 6), torch.arange(30, 35)), 1)
        ref = landing_loss(z, target, self.catalog.books, self.catalog, CFG)['landing'].item()
        first = -self.scorer.log_prob(z[:, 0], target.new_zeros(5, 0)).gather(1, (target[:, 0] - 1)[:, None])
        second = -self.scorer.log_prob(z[:, 1], target[:, :1]).gather(1, (target[:, 1] - 1)[:, None])
        # landing_loss sums the path terms per row, then adds the mean bucket term: both are row means.
        self.assertAlmostEqual(torch.cat((first, second)).mean().item(), ref, places=4)

    def test_grouped_logsumexp_handles_empty_groups(self):
        values = torch.tensor([[0.0, -math.inf, 1.0, -math.inf]], dtype=torch.float64)
        out = grouped_logsumexp(values, torch.tensor([0, 0, 1, 2]), 3)
        self.assertEqual(out[0, 0].item(), 0.0)
        self.assertEqual(out[0, 1].item(), 1.0)
        self.assertEqual(out[0, 2].item(), -math.inf)


class ListChoiceTest(unittest.TestCase):
    def test_consensus_list_wins_and_ties_keep_the_first(self):
        lists = np.array([[[1, 2, 3, 4, 5], [1, 2, 6, 7, 8], [1, 2, 6, 7, 9]],
                          [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [7, 8, 9, 10, 11]]])
        self.assertEqual(mbr_select(lists).tolist(), [1, 0])
        self.assertAlmostEqual(DISCOUNT[0], 1.0)


class GenerationTest(unittest.TestCase):
    """Real model classes with random weights on a random 32-d, 4 x 256 catalog."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = json.loads((ROOT/'configs/rqcd_mixer/v2_cf.json').read_text(encoding='utf-8'))
        cfg['diffusion'] = {**cfg['diffusion'], 'sample_steps': 4}
        cls.cfg = cfg
        r = cfg['rq']
        books = torch.randn(r['levels'], r['codes'], r['latent_dim']) * (0.5 ** torch.arange(r['levels']))[:, None, None]
        z = torch.randn(300, r['latent_dim'])
        cls.catalog = Catalog(dict(z=z, books=books, sid=hard_quantize(z, books)['sid'], original_ids=torch.arange(300),
                                   mean=torch.zeros(r['latent_dim']), scale=torch.tensor(1.0),
                                   distance_scale=torch.ones(r['levels'])), 'cpu')
        cls.model = Recommender(cfg).eval()
        cls.diffusion = Diffusion(cfg)
        cls.scorer = LandingScorer(cls.catalog, cfg)
        cls.runtime = {'precision': 'fp32'}
        cls.history = torch.randint(1, 301, (3, cfg['history_max']))
        cls.history[:, :20] = 0
        cls.noise = torch.randn(3, 4, cfg['list_length'], r['latent_dim'])

    def test_exact_landing_slot_one_is_the_scorer_argmax(self):
        items, _ = generate_mode(self.model, self.diffusion, self.catalog, self.scorer, self.history,
                                 self.noise[:, :1], self.runtime)
        z = first_slot_vectors(self.model, self.diffusion, self.catalog, self.history, self.noise[:, 0, 0], self.runtime)
        best = self.scorer.log_prob(z, distinct_history_ids(self.history)).argmax(-1) + 1
        self.assertEqual(items[:, 0].tolist(), best.tolist())

    def test_rules_return_legal_session_lists(self):
        for k in (1, 4):
            items, top = generate_mode(self.model, self.diffusion, self.catalog, self.scorer, self.history,
                                       self.noise[:, :k], self.runtime, reference=True)
            for row, hist in zip(items.tolist(), self.history.tolist()):
                self.assertEqual(len(set(row)), 5)
                self.assertFalse(set(row) & set(hist))
            self.assertEqual(top.shape, (3, 5))

    def test_lists_equal_rule_a_per_noise_set(self):
        lists = generate_lists(self.model, self.diffusion, self.catalog, self.history, self.noise[:, :2], self.runtime)
        for s in range(2):
            items, _ = generate(self.model, self.diffusion, self.catalog, self.history, self.noise[:, s], self.runtime,
                                filter_history=True)
            self.assertEqual(lists[:, s].tolist(), items.tolist())


if __name__ == '__main__':
    unittest.main()
