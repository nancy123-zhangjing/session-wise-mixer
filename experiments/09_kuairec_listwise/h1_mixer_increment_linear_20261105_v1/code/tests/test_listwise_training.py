"""Real CPU alternating updates: no training of the KuaiRec dataset or CUDA."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm.assets import write_assets
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.runtime import build_model, validate_config
from scripts.listwise_diffgrm.training import AlternatingTrainer, phase_plan, seed_all, expected_update_counts
from scripts.listwise_diffgrm.run import phase_indices

torch.set_num_threads(2)
TEMP_ROOT = ROOT / 'tmp/listwise_training'
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)


def fixture(directory, route='J', enabled=True):
    cfg = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
    cfg.update(architecture=route, history_max=3)
    cfg['model'].update(codebook_size=8, n_embd=16, n_inner=32, dropout=0.0,
                        attn_pdrop=0.0, resid_pdrop=0.0, observed_history_max=3)
    cfg['training'].update(effective_batch_size=4, microbatch_size=2, warmup_epochs=1,
                           cycles=1, r_epochs_per_cycle=1, t_updates_per_cycle=2,
                           decoder_warmup_updates=1)
    cfg['alignment'].update(joint_diger_assignment_temperature=0.5)
    path = Path(directory) / 'assets'
    if not path.exists():
        rng = np.random.default_rng(2024)
        features = rng.normal(0, .2, (9, 16)).astype('float32')
        write_assets(path, binding={}, original_item_ids=np.arange(9), content_features=features,
                     residual_features=features, centroids=rng.normal(0, .2, (4, 8, 4)).astype('float32'),
                     coarse_centroid=np.zeros(16, dtype='float32'), encoder_metadata={}, toy=True)
    from scripts.listwise_diffgrm.assets import load_assets
    assets = load_assets(path)
    seed_all(2024)
    model = build_model(cfg, assets.make_tokenizer())
    alignment = build_alignment(assets, enabled, config_overrides=cfg['alignment'])
    trainer = AlternatingTrainer(model, alignment, cfg, 4, precision='bf16')
    batch = {'history_item_ids': torch.tensor([[0, 1, 2], [3, 4, 5]]),
             'history_mask': torch.tensor([[False, True, True], [True, True, True]]),
             'target_item_ids': torch.tensor([[3, 4, 5, 6, 7], [1, 2, 6, 8, 9]])}
    return trainer, batch


def state(module):
    return {k: v.detach().clone() for k, v in module.named_parameters()}


class AlternatingTests(unittest.TestCase):
    def test_unsupported_policies_fail_instead_of_silently_using_defaults(self):
        cfg = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
        validate_config(cfg)
        for key, value in (('selection', 'ndcg@50'), ('filter_history', True), ('val_beam', 256)):
            bad = copy.deepcopy(cfg)
            bad['evaluation'][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)
        bad = copy.deepcopy(cfg)
        bad['training']['sia_weight'] = .1
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_auxiliary_targets_are_aggregated_once_per_effective_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            together, batch = fixture(tmp)
            split, _ = fixture(tmp)
            left = together.update([batch], 'T')
            right = split.update([{k: v[:1] for k, v in batch.items()},
                                  {k: v[1:] for k, v in batch.items()}], 'T')
            for key in ('reconstruction', 'balance', 'anchor'):
                self.assertEqual(left[key], right[key])

    def test_actual_frozen_and_active_parameter_changes_both_routes(self):
        for route in ('J', 'H'):
            for enabled in (False, True):
                with self.subTest(route=route, mixer=enabled), tempfile.TemporaryDirectory() as tmp:
                    trainer, batch = fixture(tmp, route, enabled)
                    before_t = state(trainer.alignment)
                    before_r = state(trainer.model)
                    result = trainer.update([batch], 'R')
                    self.assertEqual(result['lists'], 2)
                    self.assertTrue(all(torch.equal(v, before_t[k]) for k, v in state(trainer.alignment).items()))
                    self.assertTrue(any(not torch.equal(v, before_r[k]) for k, v in state(trainer.model).items()))
                    before_r = state(trainer.model)
                    trainer.update([batch], 'T')
                    self.assertTrue(all(torch.equal(v, before_r[k]) for k, v in state(trainer.model).items()))
                    self.assertFalse(torch.equal(trainer.alignment.mixer.base_codebooks, before_t['mixer.base_codebooks']))
                    other = [k for k in before_t if k != 'mixer.base_codebooks']
                    changed = [k for k in other if not torch.equal(state(trainer.alignment)[k], before_t[k])]
                    self.assertEqual(bool(changed), enabled)
                    self.assertEqual((trainer.r_updates, trainer.t_updates), (1, 1))

    def test_recommendation_only_reaches_codebook_and_mixer(self):
        for route in ('J', 'H'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                trainer, batch = fixture(tmp, route)
                trainer.config['training'].update(reconstruction_weight=0, balance_weight=0, anchor_weight=0)
                trainer.phase_mode('T')
                loss, _ = trainer.objective(batch, 'T')
                loss.backward()
                gradient = trainer.alignment.mixer.base_codebooks.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in trainer.alignment.named_parameters() if n != 'mixer.base_codebooks'))
                self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))

    def test_complete_checkpoint_restores_next_real_noisy_update_exactly(self):
        for route in ('J', 'H'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                trainer, batch = fixture(tmp, route)
                trainer.update([batch], 'R')
                trainer.update([batch], 'T')
                binding = {'fixture': route}
                saved = trainer.checkpoint(binding, {'phase': 1, 'update': 1}, route + '1')
                expected = trainer.update([batch], 'T')
                expected_state = state(trainer.alignment)
                trainer.restore(saved, binding, arm=route + '1')
                actual = trainer.update([batch], 'T')
                self.assertEqual(expected, actual)
                self.assertTrue(all(torch.equal(v, expected_state[k]) for k, v in state(trainer.alignment).items()))
                self.assertEqual(trainer.scheduler.last_epoch, 1)
                self.assertEqual(int(trainer.alignment.diger_optimizer_step), 2)

    def test_shared_warmup_forks_identical_effective_books_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            off, batch = fixture(tmp, 'H', False)
            off.update([batch], 'R')
            saved = off.checkpoint({'fixture': 1}, {'phase': 1, 'update': 0}, 'warmup')
            on, _ = fixture(tmp, 'H', True)
            on.restore(saved, {'fixture': 1}, arm='H1', fork=True)
            self.assertTrue(all(torch.equal(v, state(off.model)[k]) for k, v in state(on.model).items()))
            self.assertTrue(torch.equal(on.alignment.mixer.crossed_codebooks(False)[0],
                                        off.alignment.mixer.crossed_codebooks(False)[0]))
            self.assertEqual(on.r_optimizer.state_dict()['param_groups'], off.r_optimizer.state_dict()['param_groups'])
            self.assertEqual(on.t_optimizer.state_dict()['state'], {})
            with self.assertRaises(ValueError):
                on.restore(saved, {'fixture': 2}, arm='H1', fork=True)

    def test_protocol_counts_tail_and_paired_data_order(self):
        cfg = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
        plan = phase_plan(cfg, 709443)
        self.assertEqual(sum(r['updates'] for r in plan if r['kind'] == 'R'), 24255)
        self.assertEqual(sum(r['updates'] for r in plan if r['kind'] == 'T'), 400)
        self.assertEqual(709443 % 1024, 835)
        order = phase_indices(cfg, 709443, 0, plan[0])
        self.assertEqual(len(np.unique(order)), 709443)
        self.assertTrue(np.array_equal(order, phase_indices(cfg, 709443, 0, plan[0])))
        t_order = phase_indices(cfg, 709443, 5, plan[5])
        self.assertEqual(len(t_order), 40960)

    def test_cursor_counts_include_pending_end_of_phase_work(self):
        plan = [{'kind': 'R', 'updates': 3}, {'kind': 'T', 'updates': 2},
                {'kind': 'R', 'updates': 4}]
        self.assertEqual(expected_update_counts(plan, {'phase': 0, 'update': 3}), {'R': 3, 'T': 0})
        self.assertEqual(expected_update_counts(plan, {'phase': 1, 'update': 1}), {'R': 3, 'T': 1})
        self.assertEqual(expected_update_counts(plan, {'phase': 3, 'update': 0}), {'R': 7, 'T': 2})
        for progress in ({'phase': -1, 'update': 0}, {'phase': 4, 'update': 0},
                         {'phase': 1, 'update': 3}, {'phase': 3, 'update': 1},
                         {'phase': True, 'update': 0}, {'phase': 1.0, 'update': 0}):
            with self.assertRaises(ValueError):
                expected_update_counts(plan, progress)

    def test_restore_rejects_cursor_clock_and_optimizer_corruption_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, batch = fixture(tmp, 'H')
            trainer.update([batch], 'R')
            trainer.update([batch], 'T')
            binding = {'fixture': 'cursor'}
            saved = trainer.checkpoint(binding, {'phase': 1, 'update': 1}, 'H1')
            variants = []
            bad = copy.deepcopy(saved); bad['progress']['update'] = 0; variants.append(bad)
            bad = copy.deepcopy(saved); bad['progress']['phase'] = 10; variants.append(bad)
            bad = copy.deepcopy(saved); bad['r_updates'] += 1; variants.append(bad)
            bad = copy.deepcopy(saved); next(iter(bad['r_optimizer']['state'].values()))['step'] += 1; variants.append(bad)
            bad = copy.deepcopy(saved); bad['alignment']['diger_optimizer_step'] += 1; variants.append(bad)
            bad = copy.deepcopy(saved); bad['alignment']['diger_total_optimizer_steps'] += 1; variants.append(bad)
            before = state(trainer.model)
            for bad in variants:
                with self.assertRaises(ValueError):
                    trainer.restore(bad, binding, arm='H1')
                self.assertTrue(all(torch.equal(value, state(trainer.model)[name]) for name, value in before.items()))

    def test_preflight_resume_requires_explicit_opt_in_and_never_forks(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, batch = fixture(tmp, 'J')
            trainer.update([batch], 'R')
            trainer.update([batch], 'T')
            binding = {'fixture': 'preflight'}
            progress = {'phase': 0, 'update': 0, 'preflight_only': True}
            saved = trainer.checkpoint(binding, progress, 'J1')
            with self.assertRaisesRegex(ValueError, 'Preflight-only'):
                trainer.restore(saved, binding, arm='J1')
            restored = trainer.restore(saved, binding, arm='J1', allow_preflight=True)
            self.assertEqual(restored, progress)
            with self.assertRaisesRegex(ValueError, 'Preflight-only'):
                trainer.restore(saved, binding, arm='J1', fork=True, allow_preflight=True)


if __name__ == '__main__':
    unittest.main()
