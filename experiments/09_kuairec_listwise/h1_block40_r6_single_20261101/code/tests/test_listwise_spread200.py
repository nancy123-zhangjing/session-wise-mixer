"""Meaningful tiny-H schedule/RNG/recovery checks; no formal dataset or GPU."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_listwise_training import fixture
from scripts.listwise_diffgrm.spread200 import SpreadController, paired_indices, verify_binding
from scripts.listwise_diffgrm.run import phase_indices
from scripts.listwise_diffgrm.training import AlternatingTrainer, cpu_copy, seed_all, rng_state, sha256
from scripts.listwise_diffgrm.recovery import fingerprint


def make_case(directory):
    with redirect_stdout(io.StringIO()):
        base, batch = fixture(directory, 'H', True)
    cfg = copy.deepcopy(base.config)
    cfg['training'].update(warmup_epochs=5, cycles=10, r_epochs_per_cycle=3, t_updates_per_cycle=2)
    old = AlternatingTrainer(base.model, base.alignment, cfg, 4, precision='bf16')
    batches = [batch, {k: v.clone() for k, v in batch.items()}]
    for phase, row in enumerate(old.plan[:5]):
        seed_all(cfg['seed'] + 100003 * (phase + 1))
        old.update(batches, 'R')
    old_binding = dict(config=copy.deepcopy(cfg), asset_manifest='a' * 64, data={'toy': True},
                       engine={}, code={'scripts/listwise_diffgrm/training.py': sha256(ROOT / 'scripts/listwise_diffgrm/training.py')})
    parent = old.checkpoint(old_binding, dict(phase=5, update=0), 'H1')
    new_cfg = copy.deepcopy(cfg)
    new_cfg['training']['t_updates_per_cycle'] = 1
    new = AlternatingTrainer(old.model, old.alignment, new_cfg, 4, precision='bf16')
    new_binding = copy.deepcopy(old_binding)
    new_binding['config'] = new_cfg
    new_binding['code']['scripts/listwise_diffgrm/spread200.py'] = sha256(ROOT / 'scripts/listwise_diffgrm/spread200.py')
    c = SpreadController(new, old_binding, new_binding, 'b' * 64)
    c.fork(parent)
    return c, batches, parent


def one_phase(c, batches):
    while c.progress['update'] < c.row['updates']:
        c.step(batches, c.row['kind'])
    c.prepare_boundary()
    c.complete_phase()


class SpreadTests(unittest.TestCase):
    def test_raw_T_stream_exactly_matches_original_first_200_updates(self):
        cfg = {'seed': 2024, 'training': {'warmup_epochs': 5, 't_updates_per_cycle': 20,
                                         'effective_batch_size': 4}}
        original = np.concatenate([phase_indices(cfg, 1007, 5 + 4*b, dict(kind='T', updates=40))
                                   for b in range(5)])
        spread = np.concatenate([paired_indices(cfg, 1007, 5 + 4*b, dict(kind='T', updates=20))
                                 for b in range(10)])
        np.testing.assert_array_equal(original, spread)
        for phase in (6, 7, 8, 10, 40, 44):
            np.testing.assert_array_equal(paired_indices(cfg, 1007, phase, {'kind': 'R'}),
                                          phase_indices(cfg, 1007, phase, {'kind': 'R'}))

    def test_parent_is_preserved_and_noise_total_is_not_halved(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, batches, parent = make_case(tmp)
            state = c.checkpoint()['inner']
            for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler'):
                self.assertEqual(fingerprint(state[key]), fingerprint(parent[key]))
            self.assertEqual(c.trainer.alignment.diger_schedule_state()['total_optimizer_steps'], 20)
            self.assertEqual(sum(r['updates'] for r in c.trainer.plan if r['kind'] == 'T'), 10)
            before = fingerprint(c.trainer.model.state_dict())
            used = c.step(batches, 'T')
            self.assertEqual(used['noise_used']['optimizer_step'], 0)
            self.assertEqual(fingerprint(c.trainer.model.state_dict()), before)

    def test_paired_T_rng_survives_three_R_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, batches, _ = make_case(tmp)
            original = cpu_copy(c.checkpoint())
            # Original uninterrupted two-update T stream.
            c.prepare_rng()
            c.trainer.update(batches, 'T')
            c.trainer.update(batches, 'T')
            expected_rng = fingerprint(rng_state())
            # Restore exact fork and interleave the two halves with three R phases.
            c.restore(original)
            for _ in range(4):
                one_phase(c, batches)
            self.assertEqual(c.row['kind'], 'T')
            c.step(batches, 'T')
            self.assertEqual(fingerprint(c.t_rng), expected_rng)

    def test_boundary_resume_reproduces_next_real_R_update_and_does_not_alias_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, batches, _ = make_case(tmp)
            c.step(batches, 'T')
            c.prepare_boundary()
            snapshot = c.checkpoint()
            digest = fingerprint(snapshot)
            c.complete_phase()
            c.step(batches, 'R')
            expected = c.checkpoint()
            self.assertEqual(fingerprint(snapshot), digest)
            c.restore(snapshot)
            c.prepare_boundary()  # Must not refresh twice.
            c.complete_phase()
            c.step(batches, 'R')
            self.assertEqual(fingerprint(c.checkpoint()), fingerprint(expected))

    def test_full_schedule_has_original_35_R_and_half_T_with_finite_optimizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, batches, _ = make_case(tmp)
            while not c.done:
                one_phase(c, batches)
            state = c.checkpoint()['inner']
            self.assertEqual((state['r_updates'], state['t_updates']), (35, 10))
            self.assertEqual(state['scheduler']['last_epoch'], 35)
            self.assertEqual(state['r_optimizer']['param_groups'][0]['lr'], 0)
            self.assertEqual(int(state['alignment']['diger_total_optimizer_steps']), 20)

    def test_reject_wrong_clocks_taint_source_changes_and_inspection_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, batches, parent = make_case(tmp)
            initial = c.checkpoint()
            for field in ('r_updates', 't_updates'):
                bad = copy.deepcopy(initial)
                bad['inner'][field] += 1
                with self.assertRaises(ValueError):
                    c.restore(bad)
            bad = copy.deepcopy(initial)
            bad['inner']['alignment']['diger_total_optimizer_steps'].fill_(10)
            with self.assertRaises(ValueError):
                c.restore(bad)
            bad = copy.deepcopy(initial)
            bad['preflight_only'] = True
            with self.assertRaises(ValueError):
                c.restore(bad)
            changed = copy.deepcopy(c.execution_binding)
            changed['config']['training']['mixer_lr'] *= .5
            with self.assertRaises(ValueError):
                verify_binding(c.parent_binding, changed)
            c.restore(initial, exact_rng=False)
            with self.assertRaises(RuntimeError):
                c.step(batches, 'T')


if __name__ == '__main__':
    unittest.main()
