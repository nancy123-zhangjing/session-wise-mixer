"""Synthetic CPU trajectory gates and the sealed small phase-order metadata."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm import recovery35_trajectory as m
from scripts.listwise_diffgrm.training import sha256


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'tmp/recovery35_trajectory'; base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.reference = self.root / 'original.jsonl'
        self.output = self.root / 'run'; self.output.mkdir()
        self.rows = [dict(attempt='old', phase_index=5, epoch=5, update=i, recommendation=6., loss=7.,
                          reconstruction=1., balance=.1, anchor=0., lists=2, phase='T',
                          gradient_norm=.1, r_updates=3465, t_updates=i) for i in (1, 2)]
        write_rows(self.reference, self.rows)
        self.new = [dict(row, branch='C', seconds=.01, learning_rates_used={'R': .001}) for row in self.rows]
        write_rows(self.output / 'training.jsonl', self.new)
        self.row = dict(kind='T', epoch=5, updates=2)
        cfg = dict(seed=2024, training={'effective_batch_size': 2})
        self.controller = SimpleNamespace(branch='C', trainer=SimpleNamespace(
            config=cfg, plan=[self.row] * 6, r_updates=3465, t_updates=2),
            progress=dict(phase=5, update=2, boundary_prepared=True))
        self.indices = m.phase_indices(cfg, 4, 5, self.row)
        digest = hashlib.sha256(self.indices.astype('<i8').tobytes()).hexdigest()
        for patcher in (patch.object(m, 'TRAIN_COUNT', 4),
                        patch.object(m, 'REFERENCE_LOG_SHA256', sha256(self.reference)),
                        patch.object(m, 'PHASES', {5: ('T', 5, 2, 602042, 4, digest)})):
            patcher.start(); self.addCleanup(patcher.stop)
        m.record_phase_order(self.controller, self.output, 5, self.indices)

    def audit(self):
        return m.audit_phase(self.controller, self.output, 5, reference_log=self.reference)

    def failure(self):
        with self.assertRaises(m.ReplayTrajectoryError) as caught:
            self.audit()
        result = caught.exception.result
        self.assertEqual(result['status'], 'requires_review')
        path = Path(result['report_path'])
        self.assertTrue(path.is_file())
        self.assertEqual(sha256(path), result['report_sha256'])
        return result

    def test_exact_replay_reports_actual_zero_deltas_and_is_idempotent(self):
        result = self.audit()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['candidate_counts']['unique_updates'], 2)
        self.assertEqual(result['failed_updates'], 0)
        for values in result['metrics'].values():
            self.assertEqual(values, dict(compared=2, max_absolute_delta=0., mean_absolute_delta=0.))
        self.assertEqual(self.audit(), result)

    def test_fixed_loss_and_gradient_tolerances_allow_small_nonzero_deviation(self):
        self.new[0]['loss'] += .5 * (1e-6 + 1e-4 * 7.)
        self.new[1]['gradient_norm'] += .5 * (1e-6 + 1e-3 * .1)
        self.new[0]['anchor'] += .5e-6
        write_rows(self.output / 'training.jsonl', self.new)
        result = self.audit()
        self.assertEqual(result['status'], 'passed')
        self.assertGreater(result['metrics']['loss']['max_absolute_delta'], 0)
        self.assertAlmostEqual(result['metrics']['loss']['mean_absolute_delta'],
                               result['metrics']['loss']['max_absolute_delta'] / 2)

    def test_excessive_loss_and_gradient_are_preserved_then_stop(self):
        self.new[0]['loss'] += .01; self.new[1]['gradient_norm'] += .01
        write_rows(self.output / 'training.jsonl', self.new)
        result = self.failure()
        self.assertEqual(result['failed_updates'], 2)
        self.assertAlmostEqual(result['metrics']['loss']['max_absolute_delta'], .01)
        self.assertEqual({x['metric'] for x in result['failure_examples']}, {'loss', 'gradient_norm'})

    def test_only_exact_scientific_duplicates_are_deduplicated(self):
        duplicate = dict(self.new[0], attempt='resumed', seconds=9.)
        write_rows(self.output / 'training.jsonl', self.new + [duplicate])
        result = self.audit()
        self.assertEqual(result['candidate_counts'], dict(physical_rows=3, unique_updates=2, identical_duplicates=1))
        duplicate['loss'] += 1e-8  # Within loss tolerance, but NOT an exact repeated update.
        write_rows(self.output / 'training.jsonl', self.new + [duplicate])
        self.assertIn('Conflicting duplicate', self.failure()['error'])

    def test_integer_counts_and_branch_are_strict(self):
        for key, value in [('lists', 3), ('r_updates', 3466), ('update', 2), ('branch', 'A'), ('epoch', 6)]:
            with self.subTest(key=key):
                rows = copy.deepcopy(self.new); rows[0][key] = value
                write_rows(self.output / 'training.jsonl', rows)
                self.failure()

    def test_missing_updates_wrong_order_and_nonintegral_counter_fail(self):
        for rows in (self.new[:1], self.new[::-1], [dict(self.new[0], lists=2.), self.new[1]]):
            write_rows(self.output / 'training.jsonl', rows)
            self.failure()

    def test_wrong_reference_bytes_are_not_trusted_at_another_path(self):
        self.reference.write_text('arbitrary changed reference\n')
        self.assertIn('Reference training log bytes', self.failure()['error'])

    def test_live_counters_and_prepared_boundary_must_match_completed_log(self):
        self.controller.trainer.t_updates = 3
        self.assertIn('Live controller counters', self.failure()['error'])
        self.controller.trainer.t_updates = 2
        self.controller.progress['boundary_prepared'] = False
        self.assertIn('prepared replay phase', self.failure()['error'])

    def test_observed_order_mismatch_is_saved_and_immutable_success_not_overwritten(self):
        original = self.output / 'phase_order/phase05.json'; original_sha = sha256(original)
        with self.assertRaises(m.ReplayTrajectoryError) as caught:
            m.record_phase_order(self.controller, self.output, 5, self.indices[::-1])
        self.assertEqual(caught.exception.result['status'], 'requires_review')
        self.assertEqual(sha256(original), original_sha)
        self.assertTrue(Path(caught.exception.result['report_path']).is_file())
        self.assertEqual(m.record_phase_order(self.controller, self.output, 5, self.indices)['status'], 'passed')

    def test_missing_or_tampered_observed_order_is_not_replaced_by_rederived_order(self):
        order = self.output / 'phase_order/phase05.json'
        saved = json.loads(order.read_text()); saved['observed_count'] += 1
        order.write_text(json.dumps(saved))
        self.assertIn('phase order', self.failure()['error'])

    def test_nan_or_partial_json_is_not_silently_skipped(self):
        for content in ('{"phase_index": 5, "loss": NaN}\n', '{"phase_index": 5'):
            (self.output / 'training.jsonl').write_text(content)
            self.failure()


class SealedMetadataTests(unittest.TestCase):
    def test_embedded_20_phase_table_matches_authenticated_input_audit(self):
        path = ROOT / 'experiments/09_kuairec_listwise/h_freeze35_20261029/INPUT_AUDIT_COVERAGE.json'
        self.assertEqual(sha256(path), m.INPUT_AUDIT_SHA256)
        audit = json.loads(path.read_text())['source_replay_training_log']
        self.assertEqual(audit['sha256'], m.REFERENCE_LOG_SHA256)
        self.assertEqual(set(m.PHASES), set(range(5, 25)))
        for row in audit['phases']:
            self.assertEqual(m.PHASES[row['phase']], (row['kind'], row['epoch'], row['logged_updates'],
                row['phase_seed'], row['data_order_count'], row['data_order_sha256_int64_le']))


if __name__ == '__main__':
    unittest.main()
