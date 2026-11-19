"""Audit failures tested against hash-bound REAL incomplete training prefixes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mixer_long_audit', ROOT / 'scripts/analysis/audit_mixer_long_campaign.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / 'tests/fixtures/stage05_long_prefix'
        provenance = audit.read(directory / 'PROVENANCE.json')
        cls.real = {}
        for arm in audit.ARMS:
            path = directory / (arm + '.json')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['files'][path.name]
            cls.real[arm] = audit.read(path)

    def test_real_prefix_is_partial_and_cannot_be_a_completed_long_run(self):
        for arm, value in self.real.items():
            self.assertFalse(audit.audit_progress(value, arm, complete=False)['complete'])
            with self.assertRaisesRegex(ValueError, 'complete phase budget'):
                audit.audit_progress(value, arm, complete=True)
        controls = audit.paired_controls(self.real, complete=False)
        self.assertEqual(controls['common_recommender_epochs'], [1,2,3,4])
        self.assertEqual(controls['common_tokenizer_cycles'], [0,1])
        with self.assertRaisesRegex(ValueError, 'Incomplete four-arm controls'):
            audit.paired_controls(self.real, complete=True)

    def test_claimed_terminal_counters_do_not_hide_missing_real_phases(self):
        arm = 'mixer_joint'
        value = copy.deepcopy(self.real[arm])
        value.update(decoder_epochs=35, decoder_updates=3640, tokenizer_updates=60, noisy_calls=1920)
        with self.assertRaises(ValueError):
            audit.audit_progress(value, arm, complete=True)

    def test_real_noise_calls_and_forward_values_are_both_required(self):
        for field, changed in [('noise_applied_calls', 0), ('batches', 384), ('optimizer_updates', 24)]:
            value = copy.deepcopy(self.real['mixer_joint'])
            row = next(r for r in value['training'] if r['phase'] == 'tokenizer')
            row[field] = changed
            with self.assertRaises(ValueError):
                audit.audit_progress(value, 'mixer_joint', complete=False)
        value = copy.deepcopy(self.real['mixer_joint'])
        next(r for r in value['training'] if r['phase'] == 'tokenizer')['loss_means']['noise_scale'] = 0
        with self.assertRaisesRegex(ValueError, 'Actual forward noise'):
            audit.audit_progress(value, 'mixer_joint', complete=False)

    def test_cumulative_annealing_cannot_reset_on_second_tokenizer_cycle(self):
        value = copy.deepcopy(self.real['mixer_joint'])
        row = [r for r in value['training'] if r['phase'] == 'tokenizer'][1]
        row['tokenizer_schedule_start']['optimizer_step'] = 0
        with self.assertRaisesRegex(ValueError, 'cumulative Tokenizer step'):
            audit.audit_progress(value, 'mixer_joint', complete=False)

    def test_different_data_order_or_gumbel_stream_is_not_a_paired_control(self):
        for phase, field, changed in [('recommender','data_order_sha256','a'*64), ('tokenizer','gumbel_stream_seed',123)]:
            values = copy.deepcopy(self.real)
            next(r for r in values['mixer_joint']['training'] if r['phase'] == phase)[field] = changed
            with self.assertRaisesRegex(ValueError, 'Unpaired'):
                audit.paired_controls(values, complete=False)

    def test_optimizer_reset_and_inactive_side_gradient_are_rejected(self):
        for field, changed in [('lr_start', .001), ('optimizer_updates', 103)]:
            value = copy.deepcopy(self.real['no_mixer_frozen'])
            value['training'][1][field] = changed
            with self.assertRaises(ValueError):
                audit.audit_progress(value, 'no_mixer_frozen', complete=False)
        value = copy.deepcopy(self.real['no_mixer_frozen'])
        value['training'][0]['last_gradient_sides']['tokenizer'] = .01
        with self.assertRaisesRegex(ValueError, 'gradient routing'):
            audit.audit_progress(value, 'no_mixer_frozen', complete=False)

    def test_initial_best_is_valid_but_forced_later_epoch_or_wrong_metric_is_not(self):
        value = copy.deepcopy(self.real['mixer_joint'])
        self.assertEqual(audit.audit_progress(value,'mixer_joint',complete=False)['best_epoch'], 0)
        value['best_decoder_epoch'] = 5
        with self.assertRaisesRegex(ValueError, 'selected best'):
            audit.audit_progress(value, 'mixer_joint', complete=False)
        value = copy.deepcopy(self.real['mixer_joint'])
        value['validation'][0]['metrics']['weighted_score'] += .01
        with self.assertRaisesRegex(ValueError, 'Weighted score'):
            audit.audit_progress(value, 'mixer_joint', complete=False)

    def test_frozen_sid_motion_or_loss_drift_is_a_control_failure(self):
        value = copy.deepcopy(self.real['mixer_frozen'])
        value['training'][0]['formal_SID_change_since_initial']['changed_items'] = 1
        with self.assertRaisesRegex(ValueError, 'Frozen catalogue'):
            audit.audit_progress(value, 'mixer_frozen', complete=False)
        values = copy.deepcopy(self.real)
        values['mixer_frozen']['training'][3]['loss_means']['total_loss'] += .01
        with self.assertRaises(ValueError):
            audit.paired_controls(values, complete=False)

    def test_impossible_sid_counts_utilization_and_rates_are_rejected(self):
        for field, changed in [('collided_items',121010),('collision_rate',10),
                               ('unique_semantic_ids',0),('collision_groups',12101),('largest_group',0)]:
            value = copy.deepcopy(self.real['mixer_joint'])
            # Keep full and four-digit prefix aliases paired so range/count checks must catch it.
            for record in (value['training'][-1]['formal_SID_statistics']['full_sid'],
                           value['training'][-1]['formal_SID_statistics']['prefixes']['4']):
                record[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.audit_progress(value,'mixer_joint',complete=False)
        value = copy.deepcopy(self.real['mixer_joint'])
        value['training'][-1]['formal_SID_statistics']['positions'][0].update(used_codes=1000,utilization_rate=1000/256)
        with self.assertRaisesRegex(ValueError,'used codeword'):
            audit.audit_progress(value,'mixer_joint',complete=False)
        value = copy.deepcopy(self.real['mixer_joint'])
        value['training'][-1]['formal_SID_change_since_initial']['item_change_rate'] = 0
        with self.assertRaisesRegex(ValueError,'SID change'):
            audit.audit_progress(value,'mixer_joint',complete=False)

    def test_recommender_phase_cannot_change_sid_hash_or_recorded_observations(self):
        value = copy.deepcopy(self.real['mixer_joint'])
        next(r for r in value['training'] if r['phase']=='recommender' and r['logical_index']==6)['formal_SID_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'SID digest'):
            audit.audit_progress(value,'mixer_joint',complete=False)
        value = copy.deepcopy(self.real['mixer_joint'])
        next(r for r in value['training'] if r['phase']=='recommender' and r['logical_index']==6)['formal_SID_statistics'] = copy.deepcopy(value['training'][0]['formal_SID_statistics'])
        with self.assertRaisesRegex(ValueError,'SID observations'):
            audit.audit_progress(value,'mixer_joint',complete=False)

    def test_weighted_feedback_gradient_must_be_positive_finite_and_match_raw(self):
        for changed in (0,float('nan'),123456789):
            value = copy.deepcopy(self.real['mixer_joint'])
            next(r for r in value['training'] if r['phase']=='tokenizer')['feedback_gradient_audit']['weighted_direct_gradient_norm'] = changed
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.audit_progress(value,'mixer_joint',complete=False)

    def test_best_prefix_retains_exact_actual_selection_and_update_counters(self):
        for arm,value in self.real.items():
            best = audit.selected_progress(value)
            self.assertEqual(best['decoder_epochs'],0)
            self.assertEqual(best['decoder_updates'],0)
            self.assertEqual(best['tokenizer_updates'],0)
            self.assertEqual(best['noisy_calls'],0)
            self.assertEqual(best['training'],[])
            self.assertEqual(best['validation'],value['validation'][:1])
            self.assertEqual(best['best_score'],value['validation'][0]['metrics']['weighted_score'])

    def test_actual_sid_statistics_recomputation_is_independent_of_recorded_rates(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy is required only for actual checkpoint SID recomputation')
        # Small arithmetic pattern expanded to the required catalogue shape; this is
        # a function-level SID test, never a completed model or training observation.
        initial = (np.arange(audit.ITEMS*4,dtype=np.int64).reshape(audit.ITEMS,4) % 256)
        codes = initial.copy(); codes[0,0] = (codes[0,0]+1) % 256
        observed = audit.sid_from_codes(codes,initial)
        self.assertEqual(observed['changes']['changed_items'],1)
        self.assertEqual(observed['changes']['changed_digits'],1)
        self.assertEqual(observed['changes']['per_position_change_rate'],[1/audit.ITEMS,0,0,0])
        self.assertEqual(observed['statistics']['full_sid']['unique_semantic_ids'],65)
        self.assertEqual(observed['SID_sha256'],hashlib.sha256(codes.tobytes()).hexdigest())
        self.assertEqual(audit.sid_from_codes(initial,initial)['changes']['changed_items'],0)
        with self.assertRaisesRegex(ValueError,'actual checkpoint SID'):
            audit.sid_from_codes(codes.astype(float),initial)

    def test_checkpoint_progress_must_match_the_exact_committed_record(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is required only for checkpoint payload checks')
        arm, identity = 'mixer_joint', {'unit_test_identity':True}
        protocol = audit.read(ROOT / 'configs/05_crosspse_joint/long/mixer_long_seed2024.json')
        original = copy.deepcopy(self.real[arm])
        changed = copy.deepcopy(original); changed['training'][0]['elapsed_seconds'] += 1
        # Deliberately incomplete payload: the exact-record mismatch must be rejected
        # before any model claim. This is never a synthetic completed checkpoint.
        state = dict(format='stage05_long_phase_boundary_v1',protocol_id=protocol['protocol_id'],arm=arm,
                     mode='formal',identity=identity,E0_array_sha256=protocol['canonical_E0_array_sha256'],progress=original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial_metadata_only.pt'
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'Checkpoint progress differs'):
                audit.checkpoint(path,arm,identity,protocol,terminal=False,initial_codes=None,expected_progress=changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
