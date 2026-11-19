"""Independent real-tiny-H checks for the bounded H1 recovery experiment.

Only synthetic assets and CPU optimizer updates are used. No production
checkpoint, data, network connection, or GPU resource is changed.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_listwise_training import fixture
from scripts.listwise_diffgrm.recovery import RecoveryController, fingerprint, verify_parent_binding
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all, sha256


START, STOP = 5, 9
PARENT_SHA = 'c' * 64


def make_trainer(directory):
    with redirect_stdout(io.StringIO()):
        original, batch = fixture(directory, 'H', True)
    config = copy.deepcopy(original.config)
    # R, T/T/R, T/T/R | T/T/R, T/T/R. The parent has used half
    # of the original T clock; every original phase ID remains meaningful.
    config['training']['cycles'] = 4
    trainer = AlternatingTrainer(original.model, original.alignment, config, 4,
                                 precision='bf16')
    return trainer, [batch, {key: value.clone() for key, value in batch.items()}]


def make_bindings(trainer, directory):
    parent = dict(
        config=copy.deepcopy(trainer.config),
        asset_manifest=sha256(Path(directory) / 'assets/manifest.json'),
        data={'fixture': 'synthetic-nine-item-H', 'train_count': 4},
        code={'scripts/listwise_diffgrm/training.py': sha256(ROOT / 'scripts/listwise_diffgrm/training.py')},
        engine={'genrec/models/DIFF_GRM/cross_pse.py': sha256(ROOT / 'external/DiffGRM/genrec/models/DIFF_GRM/cross_pse.py')},
    )
    execution = copy.deepcopy(parent)
    execution['code']['scripts/listwise_diffgrm/recovery.py'] = sha256(ROOT / 'scripts/listwise_diffgrm/recovery.py')
    verify_parent_binding(parent, execution)
    return parent, execution


def make_parent(directory):
    trainer, batches = make_trainer(directory)
    parent_binding, execution_binding = make_bindings(trainer, directory)
    for phase, row in enumerate(trainer.plan[:START]):
        seed_all(trainer.config['seed'] + 100003 * (phase + 1))
        for _ in range(row['updates']):
            trainer.update(batches, row['kind'])
        if row['kind'] == 'T':
            trainer.refresh_catalog()
    state = trainer.checkpoint(parent_binding, {'phase': START, 'update': 0}, 'H1')
    return state, parent_binding, execution_binding


def active_state(controller):
    trainer = controller.trainer
    return fingerprint(dict(model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict(),
                            r_optimizer=trainer.r_optimizer.state_dict(),
                            t_optimizer=trainer.t_optimizer.state_dict(),
                            scheduler=trainer.scheduler.state_dict(),
                            r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                            progress=controller.progress))


class H1RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.parent, self.parent_binding, self.execution_binding = make_parent(self.directory)

    def controller(self, policy):
        trainer, batches = make_trainer(self.directory)
        controller = RecoveryController(trainer, policy, self.parent_binding,
                                        self.execution_binding, PARENT_SHA,
                                        start_phase=START, stop_phase=STOP)
        controller.fork(self.parent)
        return controller, batches

    def assert_rejected_without_mutation(self, controller, bad):
        before = active_state(controller)
        with self.assertRaises((ValueError, RuntimeError)):
            controller.restore(bad)
        self.assertEqual(active_state(controller), before)

    def test_two_arms_share_the_authenticated_parent_and_original_r_schedule(self):
        original_parent = fingerprint(self.parent)
        decay, _ = self.controller('decay')
        freeze, _ = self.controller('freeze')
        for key in ('model', 'alignment', 'r_optimizer', 'scheduler'):
            expected = fingerprint(self.parent[key])
            self.assertEqual(fingerprint(decay.checkpoint()['inner'][key]), expected)
            self.assertEqual(fingerprint(freeze.checkpoint()['inner'][key]), expected)
        self.assertEqual(decay.trainer.r_updates, 3)
        self.assertEqual(decay.trainer.t_updates, 4)
        self.assertEqual(decay.trainer.scheduler.last_epoch, 3)
        self.assertEqual(int(decay.trainer.alignment.diger_total_optimizer_steps), 8)
        self.assertEqual(fingerprint(self.parent), original_parent)
        # Decay is intentionally applied after loading the optimizer, not
        # erased by load_state_dict; Freeze does not modify the optimizer.
        self.assertEqual(fingerprint(freeze.trainer.t_optimizer.state_dict()),
                         fingerprint(self.parent['t_optimizer']))
        for group in decay.trainer.t_optimizer.param_groups:
            self.assertAlmostEqual(group['lr'], decay.peak_lrs[group['group_name']] * .5, places=15)

    def test_decay_records_used_lr_and_stores_next_lr_from_completed_t_clock(self):
        controller, batches = self.controller('decay')
        r_lr = controller.trainer.r_optimizer.param_groups[0]['lr']
        for completed in (4, 5):
            result = controller.step(batches, 'T')
            for group in controller.trainer.t_optimizer.param_groups:
                name, peak = group['group_name'], controller.peak_lrs[group['group_name']]
                used = peak * .5 * (1 + math.cos(math.pi * completed / 8))
                next_lr = peak * .5 * (1 + math.cos(math.pi * (completed + 1) / 8))
                self.assertAlmostEqual(result['learning_rates_used'][name], used, places=15)
                self.assertAlmostEqual(group['lr'], next_lr, places=15)
            self.assertEqual(result['phase_index'], START)
            self.assertEqual(result['learning_rates_used']['R'], r_lr)
            self.assertEqual(controller.trainer.scheduler.last_epoch, 3)
        self.assertEqual(controller.checkpoint()['inner']['t_updates'], 6)
        with self.assertRaises(ValueError):
            controller.step(batches, 'T')

    def test_freeze_skips_original_t_nodes_and_updates_only_recommender(self):
        controller, batches = self.controller('freeze')
        initial_model = fingerprint(controller.trainer.model.state_dict())
        initial_alignment = fingerprint(controller.trainer.alignment.state_dict())
        initial_t_optimizer = fingerprint(controller.trainer.t_optimizer.state_dict())
        r_phases, skipped_t = [], []
        while not controller.done:
            phase, row = controller.progress['phase'], controller.row
            seed_all(controller.trainer.config['seed'] + 100003 * (phase + 1))
            if row['kind'] == 'T':
                skipped_t.append(phase)
                with self.assertRaises(ValueError):
                    controller.step(batches, 'T')
                with patch.object(controller.trainer, 'refresh_catalog',
                                  side_effect=AssertionError('Freeze must not refresh SID')):
                    controller.prepare_boundary()
            else:
                r_phases.append(phase)
                controller.step(batches, 'R')
                controller.prepare_boundary()
            controller.checkpoint()
            controller.complete_phase()
            self.assertEqual(fingerprint(controller.trainer.alignment.state_dict()), initial_alignment)
            self.assertEqual(fingerprint(controller.trainer.t_optimizer.state_dict()), initial_t_optimizer)
        self.assertEqual(skipped_t, [5, 7])
        self.assertEqual(r_phases, [6, 8])
        self.assertEqual(controller.progress, dict(phase=STOP, update=0, boundary_prepared=False))
        self.assertEqual(controller.trainer.r_updates, 5)
        self.assertEqual(controller.trainer.t_updates, 4)
        self.assertEqual(controller.trainer.scheduler.last_epoch, 5)
        self.assertNotEqual(fingerprint(controller.trainer.model.state_dict()), initial_model)
        controller.checkpoint()
        with self.assertRaises(RuntimeError):
            controller.step(batches, 'R')

    def test_restoring_a_snapshot_does_not_alias_its_optimizer_tensors(self):
        controller, batches = self.controller('decay')
        saved = controller.checkpoint()
        expected = fingerprint(saved)
        controller.restore(saved)
        controller.step(batches, 'T')
        self.assertEqual(fingerprint(saved), expected)
        controller.restore(saved)
        self.assertEqual(fingerprint(controller.checkpoint()), expected)
        frozen, batches = self.controller('freeze')
        frozen.prepare_boundary()
        frozen.complete_phase()
        saved = frozen.checkpoint()
        expected = fingerprint(saved)
        frozen.restore(saved)
        frozen.step(batches, 'R')
        self.assertEqual(fingerprint(saved), expected)
        frozen.restore(saved)
        self.assertEqual(fingerprint(frozen.checkpoint()), expected)

    def test_mid_t_resume_replays_the_next_real_stochastic_update_exactly(self):
        controller, batches = self.controller('decay')
        seed_all(19871)
        controller.step(batches, 'T')
        saved = controller.checkpoint()
        expected_result = controller.step(batches, 'T')
        expected_state = fingerprint(controller.checkpoint())
        resumed, resumed_batches = self.controller('decay')
        resumed.restore(saved)
        actual_result = resumed.step(resumed_batches, 'T')
        self.assertEqual(actual_result, expected_result)
        self.assertEqual(fingerprint(resumed.checkpoint()), expected_state)

    def test_prepared_t_boundary_resume_does_not_refresh_or_update_twice(self):
        controller, batches = self.controller('decay')
        for _ in range(controller.row['updates']):
            controller.step(batches, 'T')
        controller.prepare_boundary()
        saved = controller.checkpoint()
        resumed, _ = self.controller('decay')
        resumed.restore(saved)
        with patch.object(resumed.trainer, 'refresh_catalog',
                          side_effect=AssertionError('Already prepared boundary was refreshed twice')):
            with patch.object(resumed.trainer, 'update',
                              side_effect=AssertionError('Completed T updates were repeated')):
                resumed.prepare_boundary()
                resumed.complete_phase()
        self.assertEqual(resumed.progress['phase'], START + 1)
        self.assertEqual((resumed.trainer.r_updates, resumed.trainer.t_updates), (3, 6))
        self.assertEqual(fingerprint(resumed.trainer.alignment.state_dict()),
                         fingerprint(saved['inner']['alignment']))

    def test_completed_r_resume_finishes_boundary_without_replaying_update(self):
        controller, batches = self.controller('freeze')
        controller.prepare_boundary()
        controller.complete_phase()
        controller.step(batches, 'R')
        saved = controller.checkpoint()
        self.assertFalse(saved['inner']['progress']['boundary_prepared'])
        resumed, _ = self.controller('freeze')
        resumed.restore(saved)
        with patch.object(resumed.trainer, 'update',
                          side_effect=AssertionError('Completed R tail was replayed')):
            resumed.prepare_boundary()
            resumed.complete_phase()
        self.assertEqual(resumed.progress['phase'], START + 2)
        self.assertEqual(resumed.trainer.r_updates, 4)

    def test_identity_cursor_optimizer_and_clock_tampering_is_rejected_before_load(self):
        controller, batches = self.controller('decay')
        controller.step(batches, 'T')
        saved = controller.checkpoint()
        mutations = {
            'outer_parent_hash': lambda s: s['parent_hashes'].update(model='f' * 64),
            'outer_binding': lambda s: s['execution_binding']['data'].update(train_count=5),
            'inner_binding': lambda s: s['inner']['binding']['data'].update(train_count=5),
            'inner_arm': lambda s: s['inner'].update(arm='H1-freeze'),
            'cursor_before_fork': lambda s: s['inner']['progress'].update(phase=START - 1),
            'cursor_past_stop': lambda s: s['inner']['progress'].update(phase=STOP + 1),
            'cursor_boolean': lambda s: s['inner']['progress'].update(update=True),
            'count': lambda s: s['inner'].update(t_updates=s['inner']['t_updates'] + 1),
            'adam_step': lambda s: next(iter(s['inner']['t_optimizer']['state'].values()))['step'].add_(1),
            'r_scheduler': lambda s: s['inner']['scheduler'].update(last_epoch=0),
            'r_current_lr': lambda s: s['inner']['r_optimizer']['param_groups'][0].update(lr=0.123),
            'r_initial_lr': lambda s: s['inner']['r_optimizer']['param_groups'][0].update(initial_lr=0.123),
            'r_scheduler_base_lr': lambda s: s['inner']['scheduler'].update(base_lrs=[0.123]),
            'r_scheduler_last_lr': lambda s: s['inner']['scheduler'].update(_last_lr=[0.123]),
            'r_betas': lambda s: s['inner']['r_optimizer']['param_groups'][0].update(betas=(0.8, 0.99)),
            'r_epsilon': lambda s: s['inner']['r_optimizer']['param_groups'][0].update(eps=1e-4),
            'r_weight_decay': lambda s: s['inner']['r_optimizer']['param_groups'][0].update(weight_decay=0.2),
            't_betas': lambda s: s['inner']['t_optimizer']['param_groups'][1].update(betas=(0.8, 0.99)),
            't_epsilon': lambda s: s['inner']['t_optimizer']['param_groups'][0].update(eps=1e-4),
            't_weight_decay': lambda s: s['inner']['t_optimizer']['param_groups'][1].update(weight_decay=0.2),
            'diger_step': lambda s: s['inner']['alignment']['diger_optimizer_step'].add_(1),
            'noise_total': lambda s: s['inner']['alignment']['diger_total_optimizer_steps'].add_(1),
            't_lr': lambda s: s['inner']['t_optimizer']['param_groups'][1].update(lr=1e-4),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                bad = copy.deepcopy(saved)
                mutate(bad)
                self.assert_rejected_without_mutation(controller, bad)

    def test_frozen_alignment_sid_and_optimizer_cannot_change_or_replace_parent_hash(self):
        controller, _ = self.controller('freeze')
        saved = controller.checkpoint()
        mutations = {
            'base': lambda s: s['inner']['alignment']['mixer.base_codebooks'].add_(1),
            'sid': lambda s: s['inner']['alignment']['hard_codes'].__setitem__(
                (1, 0), (s['inner']['alignment']['hard_codes'][1, 0] + 1) % 8),
            't_optimizer': lambda s: s['inner']['t_optimizer']['param_groups'][0].update(weight_decay=0),
            'frozen_t_update': lambda s: s['inner']['progress'].update(update=1),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                bad = copy.deepcopy(saved)
                mutate(bad)
                self.assert_rejected_without_mutation(controller, bad)
        bad = copy.deepcopy(saved)
        bad['inner']['alignment']['mixer.base_codebooks'].add_(1)
        bad['parent_hashes']['alignment'] = fingerprint(bad['inner']['alignment'])
        self.assert_rejected_without_mutation(controller, bad)

    def test_only_finished_phases_can_be_marked_boundary_prepared(self):
        controller, batches = self.controller('decay')
        controller.step(batches, 'T')
        saved = controller.checkpoint()
        self.assertEqual(saved['inner']['progress']['update'], 1)
        for marker in (True, 'yes', 1):
            with self.subTest(boundary_prepared=marker):
                bad = copy.deepcopy(saved)
                bad['inner']['progress']['boundary_prepared'] = marker
                self.assert_rejected_without_mutation(controller, bad)

    def test_nested_preflight_and_invalid_inner_format_are_not_formal_recovery(self):
        controller, _ = self.controller('decay')
        saved = controller.checkpoint()
        for field in ('nested_preflight', 'inner_format'):
            with self.subTest(corruption=field):
                bad = copy.deepcopy(saved)
                if field == 'nested_preflight':
                    bad['inner']['progress']['preflight_only'] = True
                else:
                    bad['inner']['format'] = 'unrecognized_checkpoint'
                self.assert_rejected_without_mutation(controller, bad)

    def test_resume_requires_an_independently_authenticated_parent(self):
        controller, _ = self.controller('freeze')
        saved = controller.checkpoint()
        trainer, _ = make_trainer(self.directory)
        uninitialized = RecoveryController(trainer, 'freeze', self.parent_binding,
                                           self.execution_binding, PARENT_SHA,
                                           start_phase=START, stop_phase=STOP)
        with self.assertRaisesRegex(RuntimeError, 'parent'):
            uninitialized.restore(saved)
        bad_parent = copy.deepcopy(self.parent)
        bad_parent['binding']['data']['train_count'] += 1
        with self.assertRaises(ValueError):
            uninitialized.fork(bad_parent)

    def test_binding_proof_allows_only_paths_and_recovery_extensions(self):
        relocated = copy.deepcopy(self.execution_binding)
        relocated['config']['dataset'] = '/new/verified/dataset'
        relocated['config']['assets'] = '/new/verified/assets'
        proof = verify_parent_binding(self.parent_binding, relocated)
        self.assertEqual(set(proof['relocations']), {'dataset', 'assets'})
        self.assertEqual(set(proof['extensions']), {'scripts/listwise_diffgrm/recovery.py'})
        mutations = {
            'scientific_config': lambda b: b['config']['training'].update(mixer_lr=3e-5),
            'asset': lambda b: b.update(asset_manifest='wrong'),
            'data': lambda b: b['data'].update(train_count=5),
            'engine': lambda b: b['engine'].update({'genrec/models/DIFF_GRM/cross_pse.py': 'wrong'}),
            'original_source': lambda b: b['code'].update({'scripts/listwise_diffgrm/training.py': 'wrong'}),
            'extra_source': lambda b: b['code'].update({'scripts/listwise_diffgrm/unrelated.py': 'extra'}),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                bad = copy.deepcopy(self.execution_binding)
                mutate(bad)
                with self.assertRaises(ValueError):
                    verify_parent_binding(self.parent_binding, bad)


if __name__ == '__main__':
    unittest.main()
