"""Budget arithmetic uses explicitly synthetic timings; no GPU or production run."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analysis.h1_recovery_budget import (
    PARENT_SHA256, PREFLIGHT_FORMAT, estimate, main, representative,
    startup_from_events, validate_pair,
)


def digest(label):
    return hashlib.sha256(('synthetic budget fixture/' + label).encode()).hexdigest()


def timing(total):
    return dict(seconds=total - .25, data_prepare_seconds=.25, end_to_end_seconds=total)


def synthetic_report(policy):
    config = dict(architecture='H', seed=2024, list_length=5, history_max=50,
        model=dict(n_digit=4, codebook_size=256, n_embd=256, n_head=4,
                   encoder_n_layer=1, decoder_n_layer=4, n_inner=1024),
        training=dict(effective_batch_size=1024, microbatch_size=256, warmup_epochs=5,
                      cycles=10, r_epochs_per_cycle=3, t_updates_per_cycle=40,
                      deterministic_algorithms=True),
        evaluation=dict(batch_size=32, val_beam=32, test_beam=32, candidate_lists=32, selection='ndcg@5'))
    binding = dict(config=config, asset_manifest=digest('assets'),
        data=dict(train_rows=709443, manifest_sha256=digest('data')),
        code={f'code-{i}.py': digest(f'code-{i}') for i in range(25)},
        engine={f'engine-{i}.py': digest(f'engine-{i}') for i in range(20)})
    parent_hashes = {key: digest(key) for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')}
    fork_hashes = dict(parent_hashes)
    if policy == 'decay':
        fork_hashes['t_optimizer'] = digest('halved-T-lr-same-moments')
    profiles = {}
    times = {'R': [9., 2., 3.] if policy == 'decay' else [8., 4., 5.]}
    if policy == 'decay':
        times['T'] = [10., 5., 6.]
    for kind, samples in times.items():
        rows = []
        for index, total in enumerate(samples):
            rows.append(dict(timing(total), iteration=index + 1, effective_lists=1024,
                microbatch=256, accumulation=4, original_phase_index=25 if kind == 'T' else 26,
                frozen_module_and_optimizer_unchanged=True, hard_sids_unchanged=True,
                values=dict(lists=1024, phase=kind,
                    r_updates=13860 + (index + 1 if kind == 'R' else 0),
                    t_updates=200 + (index + 1 if kind == 'T' else 0))))
        profiles[kind] = dict(records=rows, median_update_seconds=999999.,
            replay=dict(phase=kind, preflight_only=True, exact_loaded_state=True,
                        exact_next_update=True, formal_restore_rejection='Synthetic formal preflight rejection'))
    decode_times = [7., 1., 2., 3.] if policy == 'decay' else [8., 2., 3., 4.]
    decode = dict(beam=32, precision='fp32', reset_R20_output_exact=True, examples=128,
        first_batch_output_sha256=digest('identical-R20-output'),
        catalog_timing=dict(seconds=2. if policy == 'decay' else 3.),
        records=[dict(timing(total), batch_size=32, beam=32, precision='fp32',
                      sample_ids=[f'fixed-val-{32 * index + j}' for j in range(32)])
                 for index, total in enumerate(decode_times)])
    return dict(format=PREFLIGHT_FORMAT, status='passed', policy=policy, synthetic_fixture=True,
        preflight_only=True, formal_training_started=False, finished_at_parent_state=True,
        formal_resume_from_preflight_forbidden=True, parent_sha256=PARENT_SHA256,
        gpu='Synthetic H100 fixture', visible_gpu_count=1, torch_version='synthetic', cuda_version='synthetic',
        profile_updates=3, validation_samples=128, profile=profiles, source_binding=binding,
        initial=dict(r_updates=13860, t_updates=200, shared_model_and_alignment_exact=True,
            optimizer_moments_exact=True, parent_hashes=parent_hashes, fork_hashes=fork_hashes,
            hard_sid_sha256=digest('initial-sid')),
        backend=dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
            cudnn_deterministic=True, cudnn_benchmark=False, cuda_matmul_tf32=False, cudnn_tf32=False),
        validation_subset=dict(population_count=71657, count=128, split='validation',
            selection='sha256_utf8_sample_id_then_sample_id', rows_sha256=digest('same-fixed-subset')),
        decode_profile=decode, catalog_diagnostic=dict(seconds=.5 if policy == 'decay' else .75,
            full_state_unchanged=True, hard_sids_match_parent=True, metrics=dict(num_items=10728)),
        worker_seconds=999999999.)


def arguments():
    return dict(spent_gpu_seconds=1800., price_per_gpu_hour=2.,
        spent_evidence='synthetic platform allocation fixture, not a real invoice',
        price_evidence='synthetic 2 points/GPU hour fixture',
        startups={policy: dict(seconds=seconds, basis='synthetic explicit startup', evidence='test only')
                  for policy, seconds in [('decay', 100.), ('freeze', 80.)]}, allow_synthetic=True)


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.decay = synthetic_report('decay')
        self.freeze = synthetic_report('freeze')
        self.temp_root = ROOT / 'tmp/h1_recovery_budget_tests'
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def test_workload_and_hand_calculated_gpu_hours_wall_and_price(self):
        result = estimate(self.decay, self.freeze, **arguments())
        decay, freeze = result['arms']['decay'], result['arms']['freeze']
        self.assertEqual(decay['workload']['r_updates'], 4158)
        self.assertEqual(decay['workload']['t_updates'], 80)
        self.assertEqual(freeze['workload']['t_updates'], 0)
        self.assertEqual(decay['workload']['full_validation_batches_each'], 2240)
        self.assertEqual(decay['workload']['subset_batches_each'], 256)
        self.assertEqual(decay['workload']['extra_subset_evaluations'], 7)
        self.assertEqual(freeze['workload']['extra_subset_evaluations'], 5)
        self.assertEqual(decay['workload']['catalog_diagnostics'], 9)
        self.assertEqual(freeze['workload']['catalog_diagnostics'], 1)
        # Independent arithmetic: full D=2240*3+4+2; subset D=256*3+4+2.
        self.assertEqual(decay['seconds_by_component']['full_validation'], 13452.)
        self.assertEqual(decay['seconds_by_component']['extra_subset_evaluation'], 5418.)
        self.assertEqual(decay['remaining_gpu_seconds_raw'], 31938.5)
        self.assertEqual(freeze['remaining_gpu_seconds_raw'], 43962.75)
        self.assertEqual(result['totals']['remaining_gpu_seconds_raw'], 75901.25)
        self.assertEqual(result['totals']['remaining_gpu_seconds_with_margin'], 94876.5625)
        self.assertAlmostEqual(result['totals']['total_gpu_hours_including_spent'], 96676.5625 / 3600)
        self.assertEqual(result['totals']['two_gpu_parallel_remaining_wall_seconds'], 54953.4375)
        self.assertEqual(result['totals']['shared_two_gpu_job_remaining_allocated_gpu_seconds'], 109906.875)
        self.assertEqual(result['totals']['shared_two_gpu_job_idle_tail_gpu_seconds'], 15030.3125)
        self.assertAlmostEqual(result['price']['shared_two_gpu_job_total_points_estimate'], 111706.875 / 1800)
        self.assertAlmostEqual(result['price']['total_points_estimate'], 96676.5625 / 1800)
        self.assertFalse(result['price']['settlement_verified'])
        self.assertFalse(result['submission_performed'])
        self.assertFalse(result['budget_limit_changed'])
        self.assertEqual(result['status'], 'synthetic_estimate_not_for_authorization')
        json.dumps(result, allow_nan=False)

    def test_worker_duration_is_never_used_as_platform_allocated_seconds(self):
        before = estimate(self.decay, self.freeze, **arguments())
        self.decay['worker_seconds'] = .00001
        self.freeze['worker_seconds'] = 1e100
        after = estimate(self.decay, self.freeze, **arguments())
        self.assertEqual(before['totals'], after['totals'])
        self.assertEqual(after['platform_usage']['already_spent_gpu_seconds'], 1800.)
        self.assertFalse(after['platform_usage']['preflight_worker_seconds_used'])

    def test_first_step_is_excluded_but_positive_cold_excess_is_counted_once(self):
        result = estimate(self.decay, self.freeze, **arguments())
        rates = result['arms']['decay']['measured_rates']
        self.assertEqual(rates['R']['seconds'], 3.)
        self.assertEqual(rates['R']['retained_count'], 2)
        self.assertEqual(rates['R']['excluded_first_seconds'], 9.)
        self.assertEqual(result['arms']['decay']['seconds_by_component']['training_first_step_excess'], 10.)
        median = estimate(self.decay, self.freeze, statistic='median', **arguments())
        self.assertEqual(median['arms']['decay']['measured_rates']['R']['seconds'], 2.5)
        self.assertEqual(median['assumptions']['statistic'], 'median_after_first')
        self.assertLess(median['totals']['remaining_gpu_seconds_raw'], result['totals']['remaining_gpu_seconds_raw'])

    def test_unaccepted_or_mismatched_source_reports_are_rejected(self):
        cases = [
            lambda report: report.update(status='running'),
            lambda report: report.update(parent_sha256='0' * 64),
            lambda report: report['source_binding']['code'].update({'code-0.py': '0' * 64}),
            lambda report: report['initial']['parent_hashes'].update(model='0' * 64),
            lambda report: report['source_binding']['config']['training'].update(microbatch_size=128),
            lambda report: report['decode_profile'].update(beam=256),
            lambda report: report['catalog_diagnostic'].update(full_state_unchanged=False),
            lambda report: report['validation_subset'].update(rows_sha256='0' * 64),
            lambda report: report['decode_profile'].update(first_batch_output_sha256='0' * 64),
        ]
        for change in cases:
            with self.subTest(change=change):
                broken = copy.deepcopy(self.decay)
                change(broken)
                with self.assertRaises(ValueError):
                    estimate(broken, self.freeze, **arguments())

    def test_explicit_protocols_may_differ_only_in_policy(self):
        for policy, report in [('decay', self.decay), ('freeze', self.freeze)]:
            report['protocol'] = dict(format='stage09_h1_recovery_v1', policy=policy,
                parent_sha256=PARENT_SHA256, start_phase=25, stop_phase=33, t_lr='same-clock-formula')
        accepted = validate_pair(self.decay, self.freeze, allow_synthetic=True)
        self.assertTrue(accepted['protocol_validation'].startswith('explicit'))
        self.freeze['protocol']['t_lr'] = 'different'
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            validate_pair(self.decay, self.freeze, allow_synthetic=True)

    def test_missing_short_or_inconsistent_timings_fail_closed(self):
        bad = copy.deepcopy(self.decay)
        bad['profile']['R']['records'].pop()
        with self.assertRaises(ValueError):
            estimate(bad, self.freeze, **arguments())
        for value in (float('nan'), float('inf'), 0., -1., True):
            with self.subTest(seconds=value):
                bad = copy.deepcopy(self.decay)
                bad['profile']['R']['records'][1]['end_to_end_seconds'] = value
                with self.assertRaises(ValueError):
                    estimate(bad, self.freeze, **arguments())
        bad = copy.deepcopy(self.freeze)
        bad['decode_profile']['records'][1]['sample_ids'] = bad['decode_profile']['records'][0]['sample_ids']
        with self.assertRaisesRegex(ValueError, 'repeated validation'):
            estimate(self.decay, bad, **arguments())

    def test_startup_and_platform_price_inputs_must_be_explicit_and_finite(self):
        for key in ('spent_gpu_seconds', 'price_per_gpu_hour'):
            for value in (-1., float('nan'), float('inf'), True):
                with self.subTest(key=key, value=value):
                    args = arguments()
                    args[key] = value
                    with self.assertRaises(ValueError):
                        estimate(self.decay, self.freeze, **args)
        for change in (lambda a: a.update(startups={}), lambda a: a.update(spent_evidence=''),
                       lambda a: a.update(price_evidence=''),
                       lambda a: a['startups']['decay'].update(evidence='')):
            args = arguments()
            change(args)
            with self.assertRaises(ValueError):
                estimate(self.decay, self.freeze, **args)
        args = arguments()
        args.update(spent_gpu_seconds=0., price_per_gpu_hour=0.)
        self.assertEqual(estimate(self.decay, self.freeze, **args)['price']['total_points_estimate'], 0.)

    def test_startup_log_uses_explicit_stages_not_total_worker_time(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(self.temp_root.resolve()))
            path = Path(directory) / 'events.jsonl'
            rows = [dict(event='stage_pass', stage='authenticated_R20_fork', seconds=7.),
                    dict(event='stage_pass', stage='load_fixed_validation_subset', seconds=3.),
                    dict(event='stage_pass', stage='profile_R_1', seconds=999999.),
                    dict(event='preflight_complete', policy='decay', elapsed=888888.)]
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            result = startup_from_events(path, 'decay')
            self.assertEqual(result['seconds'], 10.)
            self.assertEqual(len(result['components']), 2)
            self.assertEqual(len(result['evidence_sha256']), 64)
            rows.append(rows[0])
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                startup_from_events(path, 'decay')

    def test_estimation_does_not_mutate_reports_and_rejects_synthetic_production_use(self):
        original = copy.deepcopy((self.decay, self.freeze))
        estimate(self.decay, self.freeze, **arguments())
        self.assertEqual((self.decay, self.freeze), original)
        args = arguments()
        args.pop('allow_synthetic')
        with self.assertRaisesRegex(ValueError, 'synthetic fixture'):
            estimate(self.decay, self.freeze, **args)

    def test_cli_cannot_emit_a_production_budget_from_synthetic_reports(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(self.temp_root.resolve()))
            root = Path(directory)
            for policy, report in [('decay', self.decay), ('freeze', self.freeze)]:
                (root / f'{policy}.json').write_text(json.dumps(report), encoding='utf-8')
            output = root / 'should_not_exist.json'
            args = ['--decay-report', str(root / 'decay.json'), '--freeze-report', str(root / 'freeze.json'),
                    '--decay-startup-seconds', '100', '--freeze-startup-seconds', '80',
                    '--decay-startup-evidence', 'fixture', '--freeze-startup-evidence', 'fixture',
                    '--spent-gpu-seconds', '1800', '--spent-evidence', 'fixture',
                    '--price-per-gpu-hour', '2', '--price-evidence', 'fixture', '--output', str(output)]
            with self.assertRaisesRegex(ValueError, 'synthetic fixture'):
                main(args)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
