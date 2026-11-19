"""Synthetic arithmetic tests only; these values are not experiment measurements."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('freeze35_estimate', Path(__file__).with_name('estimate_budget.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    shared = dict(config=dict(architecture='H', seed=2024, list_length=5,
        training=dict(effective_batch_size=1024, microbatch_size=256, checkpoint_every_updates=100),
        evaluation=dict(batch_size=32, val_beam=32, test_beam=32)),
        data={'synthetic': True}, asset_manifest='a' * 64, code={'synthetic.py': 'b' * 64}, engine={'synthetic.py': 'c' * 64})
    update = lambda seconds: dict(seconds=seconds, data_prepare_seconds=1, effective_lists=1024, microbatch=256, accumulation=4)
    saved = dict(seconds=1, state_materialization_seconds=2, payload_fingerprint_seconds=3,
                 file_sha256_seconds=4, bytes=80_000_000, fsync_completed=True)
    replay = dict(save=saved, load=dict(seconds=.5), restore=dict(seconds=.25), exact_loaded_state=True,
                  exact_next_update=True, reusable_snapshot_unchanged=True, formal_restore_rejected=True)
    report = dict(format='stage09_recovery35_preflight_v1', status='passed', synthetic_fixture=True,
        preflight_only=True, formal_training_started=False, max_seconds=1620, worker_seconds=55,
        gpu=dict(visible_gpu_count=1, name='synthetic H100', mig_mode='disabled', total_memory_bytes=80_000_000_000), branches={})
    for branch in 'ABC':
        phase, r, t = MODULE.STARTS[branch]
        training = dict(R={'records': [update(10), update(2), update(4)]}, R_disk_replay=copy.deepcopy(replay))
        if branch == 'C':
            training.update(T={'records': [update(2) for _ in range(40)]}, T_disk_replay=copy.deepcopy(replay),
                            T40_boundary_checkpoint=copy.deepcopy(saved), T40_refresh={'seconds': 2})
        report['branches'][branch] = dict(format='stage09_recovery35_branch_preflight_v1', branch=branch,
            status='passed', parent_sha256=MODULE.PARENTS[branch], preflight_only=True, formal_training_started=False,
            finished_at_exact_parent_state=True, formal_resume_from_preflight_forbidden=True,
            initial_counts=dict(phase=phase, R=r, T=t), source_binding=copy.deepcopy(shared),
            training_profile=training, initial_checkpoint=copy.deepcopy(saved), build_timing={'seconds': 5},
            decode_profile=dict(exact_output_repeat=True, full_training_state_rng_modes_gradients_unchanged=True,
                quality_metric=False, catalog_timing={'seconds': 2}, records=[dict(seconds=seconds,
                data_prepare_seconds=1, end_to_end_seconds=seconds + 1, batch_size=32, beam=32,
                inference_precision='fp32') for seconds in (12, 2, 4)]))
    zero = dict(low=0, high=0, basis='operator_allowance', evidence='Synthetic test only: zero extra cost')
    extras = dict(format='stage09_freeze35_budget_overheads_v1', **{key: copy.deepcopy(zero) for key in MODULE.EXTRA_KEYS},
                  extra_gpu_allocated_cpu_audit={branch: copy.deepcopy(zero) for branch in 'ABC'})
    supervisor = dict(status='passed', preflight_only=True, returncode=0, reason=None,
                      allocated_gpu_seconds_during_script=60, wall_seconds=60, max_seconds=1620)
    return report, supervisor, extras


class EstimateTests(unittest.TestCase):
    def estimate(self, report=None, supervisor=None, extras=None, **kwargs):
        fixture_report, fixture_supervisor, fixture_extras = fixture()
        return MODULE.estimate(report or fixture_report, supervisor or fixture_supervisor, extras or fixture_extras,
                               spent_gpu_hours=kwargs.pop('spent_gpu_hours', .5),
                               spent_evidence='Synthetic allocation receipt', allow_synthetic=True, **kwargs)

    def test_exact_loop_counts_and_capacity_vs_traffic(self):
        result = self.estimate()
        expected = {'A': (6237, 0, 91, 14, 13), 'B': (2079, 0, 31, 6, 5), 'C': (20790, 200, 302, 43, 42)}
        for branch, numbers in expected.items():
            counts = result['arms'][branch]['workload']
            self.assertEqual(tuple(counts[key] for key in ('r_updates', 't_updates', 'rolling_writes',
                             'immutable_writes', 'covered_checkpoint_reads')), numbers)
        self.assertEqual(sum(row['workload']['rolling_writes'] for row in result['arms'].values()), 424)
        self.assertEqual(sum(row['workload']['immutable_writes'] for row in result['arms'].values()), 63)
        self.assertEqual(result['arms']['C']['checkpoint_write_traffic_bytes'], 345 * 80_000_000)
        self.assertEqual(result['arms']['C']['checkpoint_retained_capacity_bytes'], 44 * 80_000_000)

    def test_data_prep_warm_median_max_cold_extra_and_io_included(self):
        result = self.estimate()
        parts = result['arms']['A']['components_seconds']
        self.assertEqual(parts['R_compute_and_data']['low'], 6237 * 4 + 7)
        self.assertEqual(parts['R_compute_and_data']['high'], 6237 * 5 + 6)
        self.assertEqual(parts['checkpoint_write_and_fsync']['high'], 105)
        self.assertEqual(parts['checkpoint_state_materialization']['high'], 210)
        self.assertEqual(parts['checkpoint_payload_fingerprint']['high'], 81)
        self.assertEqual(parts['checkpoint_file_sha256']['high'], 120)
        self.assertEqual(parts['full_validation_decode_and_data']['units'], 3 * 2240)
        self.assertEqual(parts['maximum_test_decode_and_data']['units'], 2 * 1663)

    def test_margin_applies_to_remaining_only_and_rounds_up(self):
        result = self.estimate()
        raw = sum(row['unbuffered_wall_seconds']['high'] for row in result['arms'].values())
        self.assertAlmostEqual(result['cumulative_gpu_hours']['high'], .5 + raw * 1.25 / 3600)
        self.assertEqual(result['suggested_cumulative_budget_integer_gpu_hours'], math.ceil(.5 + raw * 1.25 / 3600))
        self.assertAlmostEqual(result['three_card_parallel_wall_hours']['high'],
            max(row['unbuffered_wall_seconds']['high'] for row in result['arms'].values()) * 1.25 / 3600)

    def test_missing_allowance_does_not_silently_become_zero(self):
        _, _, extras = fixture()
        extras[MODULE.EXTRA_KEYS[0]]['high'] = None
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            self.estimate(extras=extras)

    def test_cpu_work_while_gpu_allocated_is_in_budget_offline_is_not(self):
        _, _, extras = fixture()
        extras['extra_gpu_allocated_cpu_audit']['C'].update(low=100, high=200)
        extras['offline_cpu_audit_total'] = dict(low=300, high=600, basis='operator_allowance', evidence='Synthetic offline work')
        base, changed = self.estimate(), self.estimate(extras=extras)
        self.assertAlmostEqual(changed['remaining_gpu_hours']['high'] - base['remaining_gpu_hours']['high'], 200 * 1.25 / 3600)
        self.assertEqual(changed['offline_cpu_audit_seconds_excluded_from_gpu']['high'], 600)

    def test_reject_failed_wrong_parent_or_changed_binding(self):
        for change in ('failed', 'parent', 'binding'):
            report, _, _ = fixture()
            if change == 'failed':
                report['branches']['C']['status'] = 'failed'
            elif change == 'parent':
                report['branches']['B']['parent_sha256'] = '0' * 64
            else:
                report['branches']['C']['source_binding']['code']['synthetic.py'] = '0' * 64
            with self.assertRaises(ValueError):
                self.estimate(report=report)

    def test_refuse_worker_time_as_complete_platform_usage(self):
        with self.assertRaisesRegex(ValueError, 'below supervised'):
            self.estimate(spent_gpu_hours=.001)

    def test_production_entry_rejects_synthetic(self):
        report, supervisor, extras = fixture()
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            MODULE.estimate(report, supervisor, extras, spent_gpu_hours=.5, spent_evidence='Synthetic fixture')


if __name__ == '__main__':
    unittest.main()
