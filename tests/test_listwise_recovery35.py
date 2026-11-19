"""Real tiny-H CPU updates for ABC; no real dataset, checkpoint or GPU use."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
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
from scripts.listwise_diffgrm import recovery35 as m
from scripts.listwise_diffgrm.recovery import RecoveryController as Legacy
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all, sha256


def trainer_at(directory, enabled):
    with redirect_stdout(io.StringIO()):
        original, batch = fixture(directory, 'H', enabled)
    cfg = copy.deepcopy(original.config)
    cfg['training'].update(warmup_epochs=5, cycles=10, r_epochs_per_cycle=3, t_updates_per_cycle=1)
    trainer = AlternatingTrainer(original.model, original.alignment, cfg, 4, precision='bf16')
    return trainer, [batch, {k: v.clone() for k, v in batch.items()}]


def bindings(trainer):
    parent = dict(config=copy.deepcopy(trainer.config), asset_manifest='a' * 64,
                  data={'synthetic_fixture': True}, engine={'synthetic_engine': 'e' * 64},
                  code={'scripts/listwise_diffgrm/training.py': sha256(ROOT / 'scripts/listwise_diffgrm/training.py')})
    legacy = copy.deepcopy(parent)
    for name, digest in m.LEGACY_EXTENSIONS.items():
        legacy['code']['scripts/listwise_diffgrm/' + name] = digest
    execution = copy.deepcopy(legacy)
    execution['code']['scripts/listwise_diffgrm/recovery35.py'] = sha256(ROOT / 'scripts/listwise_diffgrm/recovery35.py')
    return parent, legacy, execution


def advance(controller, batches, count=None):
    steps = 0
    while not controller.done and (count is None or steps < count):
        phase, row = controller.progress['phase'], controller.row
        seed_all(controller.trainer.config['seed'] + 100003 * (phase + 1))
        if not controller.skip_t:
            while controller.progress['update'] < row['updates']:
                controller.step(batches, row['kind'])
        controller.prepare_boundary()
        controller.checkpoint()
        controller.complete_phase()
        steps += 1


def gate_for(controller):
    return dict(status='passed', branch='C',
                checkpoint_state_sha256=m.fingerprint(controller.checkpoint()['inner']),
                protocol_sha256=controller.protocol_sha256,
                validation_sha256='a' * 64,
                reference_prediction_sha256=m.REFERENCE_R20_PREDICTION_SHA256,
                candidate_prediction_sha256='b' * 64)


class Recovery35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.sources = {}
        # One original H0 trace supplies true R5 and R32 synthetic states.
        trainer, batches = trainer_at(cls.directory, False)
        parent, legacy, execution = bindings(trainer)
        for phase, row in enumerate(trainer.plan[:41]):
            seed_all(trainer.config['seed'] + 100003 * (phase + 1))
            for _ in range(row['updates']):
                trainer.update(batches, row['kind'])
            if row['kind'] == 'T':
                trainer.refresh_catalog()
            if phase + 1 in (5, 41):
                cls.sources['C' if phase + 1 == 5 else 'B'] = trainer.checkpoint(
                    parent, dict(phase=phase + 1, update=0), 'H0')
        # Authenticate a real tiny old-H1 freeze continuation for A's envelope.
        trainer, batches = trainer_at(cls.directory, True)
        parent, legacy, execution = bindings(trainer)
        for phase, row in enumerate(trainer.plan[:25]):
            seed_all(trainer.config['seed'] + 100003 * (phase + 1))
            for _ in range(row['updates']):
                trainer.update(batches, row['kind'])
            if row['kind'] == 'T':
                trainer.refresh_catalog()
        ancestor = trainer.checkpoint(parent, dict(phase=25, update=0), 'H1')
        old = Legacy(trainer, 'freeze', parent, legacy, m.LEGACY_PARENT_SHA256)
        old.fork(ancestor)
        while not old.done:
            row = old.row
            if row['kind'] == 'R':
                old.step(batches, 'R')
            old.prepare_boundary()
            old.complete_phase()
        cls.sources['A'] = old.checkpoint()['inner']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def controller(self, branch):
        trainer, batches = trainer_at(self.directory, branch == 'A')
        _, _, execution = bindings(trainer)
        source = self.sources[branch]
        controller = m.Recovery35Controller(trainer, branch, source['binding'], execution,
                                            m.BRANCHES[branch]['parent_sha256'])
        controller.fork(source)
        return controller, batches

    def test_abc_authenticated_sources_preserve_both_optimizers_and_full_schedule(self):
        for branch in 'ABC':
            with self.subTest(branch=branch):
                controller, _ = self.controller(branch)
                saved = controller.checkpoint()['inner']
                for key in m.STATE_KEYS:
                    self.assertEqual(m.fingerprint(saved[key]), m.fingerprint(self.sources[branch][key]))
                self.assertEqual(saved['scheduler']['last_epoch'], saved['r_updates'])
                self.assertEqual(controller.progress['phase'], m.BRANCHES[branch]['start_phase'])

    def test_ab_skip_t_without_refresh_and_end_at_r35_with_original_scheduler_zero(self):
        for branch in 'AB':
            with self.subTest(branch=branch):
                c, batches = self.controller(branch)
                initial = c.checkpoint()
                with patch.object(c.trainer, 'refresh_catalog', side_effect=AssertionError('Frozen SID refreshed')):
                    advance(c, batches)
                end = c.checkpoint()['inner']
                self.assertEqual((end['r_updates'], end['t_updates']),
                                 (35, self.sources[branch]['t_updates']))
                self.assertEqual(end['progress'], dict(phase=45, update=0, boundary_prepared=False))
                self.assertEqual(end['scheduler']['last_epoch'], 35)
                self.assertEqual(end['r_optimizer']['param_groups'][0]['lr'], 0)
                self.assertEqual(m.fingerprint(end['alignment']), initial['parent_hashes']['alignment'])
                self.assertEqual(m.fingerprint(end['t_optimizer']), initial['parent_hashes']['t_optimizer'])
                with self.assertRaises(RuntimeError):
                    c.step(batches, 'R')

    def test_c_real_replay_gate_then_frozen_continuation(self):
        c, batches = self.controller('C')
        advance(c, batches, 20)
        self.assertEqual(c.progress, dict(phase=25, update=0, boundary_prepared=False))
        self.assertEqual((c.trainer.r_updates, c.trainer.t_updates), (20, 5))
        self.assertTrue(c.needs_replay_gate)
        for op in (lambda: c.step(batches, 'T'), c.prepare_boundary, c.complete_phase):
            with self.assertRaisesRegex(RuntimeError, 'gate'):
                op()
        pre_gate = m.fingerprint(c.checkpoint()['inner'])
        receipt = gate_for(c)
        c.accept_replay_gate(receipt)
        self.assertEqual(m.fingerprint(c.checkpoint()['inner']), pre_gate)
        anchor = copy.deepcopy(c.freeze_hashes)
        with patch.object(c.trainer, 'refresh_catalog', side_effect=AssertionError('Frozen SID refreshed')):
            advance(c, batches)
        end = c.checkpoint()
        self.assertEqual((end['inner']['r_updates'], end['inner']['t_updates']), (35, 5))
        self.assertEqual(end['freeze_hashes'], anchor)
        self.assertEqual(end['replay_gate'], receipt)

    def test_c_gate_rejects_wrong_reference_state_protocol_and_failed_audit(self):
        c, batches = self.controller('C')
        advance(c, batches, 20)
        valid = gate_for(c)
        for key, value in [('status', 'failed'), ('branch', 'A'), ('checkpoint_state_sha256', 'e' * 64),
                           ('reference_prediction_sha256', 'f' * 64), ('protocol_sha256', 'c' * 64),
                           ('candidate_prediction_sha256', None)]:
            bad = dict(valid, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.accept_replay_gate(bad)
        c.accept_replay_gate(valid)
        with self.assertRaises(ValueError):
            c.accept_replay_gate(valid)

    def test_mid_replay_t_resume_reproduces_next_stochastic_update_and_no_alias(self):
        c, batches = self.controller('C')
        # T block has one real update: resume before boundary/refresh.
        c.step(batches, 'T')
        snapshot = c.checkpoint()
        frozen_snapshot = m.fingerprint(snapshot)
        c.prepare_boundary(); c.complete_phase()
        expected = c.step(batches, 'R')
        expected_state = m.fingerprint(c.checkpoint())
        resumed, rb = self.controller('C')
        resumed.restore(snapshot)
        resumed.prepare_boundary(); resumed.complete_phase()
        self.assertEqual(resumed.step(rb, 'R'), expected)
        self.assertEqual(m.fingerprint(resumed.checkpoint()), expected_state)
        self.assertEqual(m.fingerprint(snapshot), frozen_snapshot)

    def test_prepared_replay_t_boundary_does_not_refresh_twice(self):
        c, batches = self.controller('C')
        c.step(batches, 'T'); c.prepare_boundary()
        saved = c.checkpoint()
        resumed, _ = self.controller('C'); resumed.restore(saved)
        with patch.object(resumed.trainer, 'refresh_catalog', side_effect=AssertionError('Duplicate refresh')):
            resumed.prepare_boundary(); resumed.complete_phase()
        self.assertEqual(resumed.progress['phase'], 6)

    def test_freeze_r_resume_and_reusable_optimizer_snapshot(self):
        c, batches = self.controller('A')
        advance(c, batches, 1)  # skip original T node
        saved = c.checkpoint(); digest = m.fingerprint(saved)
        expected = c.step(batches, 'R'); expected_state = m.fingerprint(c.checkpoint())
        c.restore(saved)
        self.assertEqual(c.step(batches, 'R'), expected)
        self.assertEqual(m.fingerprint(c.checkpoint()), expected_state)
        self.assertEqual(m.fingerprint(saved), digest)

    def test_restore_c_pending_gate_and_then_post_gate(self):
        c, batches = self.controller('C'); early = c.checkpoint()
        advance(c, batches, 20); pending = c.checkpoint()
        resumed, rb = self.controller('C'); resumed.restore(pending)
        self.assertTrue(resumed.needs_replay_gate)
        resumed.accept_replay_gate(gate_for(resumed)); advance(resumed, rb, 2)
        later = resumed.checkpoint()
        c.restore(later)
        self.assertFalse(c.needs_replay_gate)
        c.restore(early)
        self.assertFalse(c.frozen)
        self.assertIsNone(c.freeze_hashes)

    def test_final_controller_can_audit_all_earlier_replay_clock_boundaries(self):
        c, batches = self.controller('C'); early = c.checkpoint()
        advance(c, batches, 20); r20 = c.checkpoint()
        c.accept_replay_gate(gate_for(c)); advance(c, batches)
        # Artifact coverage audits use the final controller on historical nodes.
        c.validate_clocks(early['inner'])
        c.validate_clocks(r20['inner'])
        c.validate_clocks(c.checkpoint()['inner'])

    def test_reject_tampering_before_model_or_optimizer_mutation(self):
        c, batches = self.controller('A'); saved = c.checkpoint()
        cases = []
        for key, value in [('preflight_only', True), ('inspection_only', True), ('format', 'wrong')]:
            bad = copy.deepcopy(saved); bad[key] = value; cases.append(bad)
        for key in ('r_updates', 't_updates'):
            bad = copy.deepcopy(saved); bad['inner'][key] += 1; cases.append(bad)
        bad = copy.deepcopy(saved); bad['inner']['scheduler']['last_epoch'] = 0; cases.append(bad)
        bad = copy.deepcopy(saved); bad['inner']['t_optimizer']['param_groups'][0]['lr'] *= 2; cases.append(bad)
        bad = copy.deepcopy(saved); bad['inner']['alignment']['hard_codes'][1, 0] += 1; cases.append(bad)
        bad = copy.deepcopy(saved); bad['inner']['progress']['preflight_only'] = True; cases.append(bad)
        bad = copy.deepcopy(saved)
        state = next(iter(bad['inner']['r_optimizer']['state'].values()))
        state['exp_avg'] = torch.zeros(17); cases.append(bad)
        for index, bad in enumerate(cases):
            before = m.fingerprint(c.checkpoint())
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                c.restore(bad)
            self.assertEqual(m.fingerprint(c.checkpoint()), before)

    def test_resume_c_cannot_skip_gate_or_supply_gate_before_r20(self):
        c, batches = self.controller('C'); early = c.checkpoint()
        advance(c, batches, 20); pending = c.checkpoint()
        bad = copy.deepcopy(pending); bad['inner']['progress']['phase'] += 1
        with self.assertRaisesRegex(ValueError, 'gate'):
            c.restore(bad)
        bad = copy.deepcopy(early); bad['replay_gate'] = gate_for(c); bad['replay_gate_state_sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'predates'):
            c.restore(bad)

    def test_accepted_r20_gate_cannot_be_transplanted_to_another_r20_model(self):
        c, batches = self.controller('C'); advance(c, batches, 20)
        c.accept_replay_gate(gate_for(c)); gated = c.checkpoint()
        c.restore(gated)
        before = m.fingerprint(c.checkpoint())
        bad = copy.deepcopy(gated)
        next(v for v in bad['inner']['model'].values() if v.is_floating_point()).add_(.1)
        with self.assertRaisesRegex(ValueError, 'different current state'):
            c.restore(bad)
        self.assertEqual(m.fingerprint(c.checkpoint()), before)

    def test_binding_accepts_only_locked_legacy_and_recovery35_additions(self):
        c, _ = self.controller('C'); parent, current = c.parent_binding, c.execution_binding
        m.verify_binding(parent, current)
        for mutate in ('legacy', 'old', 'unrelated', 'engine', 'config'):
            bad = copy.deepcopy(current)
            if mutate == 'legacy': bad['code']['scripts/listwise_diffgrm/recovery.py'] = '1' * 64
            elif mutate == 'old': bad['code']['scripts/listwise_diffgrm/training.py'] = '2' * 64
            elif mutate == 'unrelated': bad['code']['scripts/listwise_diffgrm/other.py'] = '3' * 64
            elif mutate == 'engine': bad['engine']['synthetic_engine'] = '4' * 64
            else: bad['config']['training']['decoder_lr'] *= 2
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                m.verify_binding(parent, bad)

    def test_cpu_inspection_cannot_step_or_be_formal_resume_source(self):
        c, batches = self.controller('B')
        c.restore(c.checkpoint(), exact_rng=False)
        with self.assertRaisesRegex(RuntimeError, 'inspection'):
            c.prepare_boundary()
        with self.assertRaisesRegex(ValueError, 'inspection'):
            c.restore(c.checkpoint())

    def test_builder_rejects_file_sha_before_loading_or_building_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'wrong.pt'; source.write_bytes(b'not a checkpoint')
            with patch.object(m.torch, 'load', side_effect=AssertionError('Untrusted pickle loaded')):
                with self.assertRaisesRegex(ValueError, 'SHA'):
                    m.build_recovery35(source, 'B', 'cpu')

    def test_wrong_source_arm_or_boundary_is_rejected(self):
        trainer, _ = trainer_at(self.directory, False)
        source = self.sources['B']; _, _, execution = bindings(trainer)
        c = m.Recovery35Controller(trainer, 'B', source['binding'], execution, m.BRANCHES['B']['parent_sha256'])
        for key, value in [('arm', 'H1'), ('preflight_only', True)]:
            bad = copy.deepcopy(source); bad[key] = value
            with self.assertRaises(ValueError): c.fork(bad)
        bad = copy.deepcopy(source); bad['progress']['phase'] = 40
        with self.assertRaises(ValueError): c.fork(bad)


if __name__ == '__main__':
    unittest.main()
