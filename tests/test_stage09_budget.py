"""Offline budget arithmetic and scheduling using explicitly synthetic timings."""
import copy
from contextlib import redirect_stderr
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.stage09_budget import (
    ARMS, CODE_FILES, DATA_FILES, ENGINE_FILES, estimate, evaluation_seconds, main,
    schedule, validate_preflight, validate_supervisor, workload,
)


def synthetic_preflight():
    """Only a test fixture; never save or advertise this as an H100 result."""
    ids, indices = [f'validation-{i}' for i in range(16)], list(range(16))
    sha = hashlib.sha256(b'explicit synthetic test fixture, not a measured source').hexdigest()
    config = dict(architecture='H', seed=2024, history_max=50, list_length=5,
        model=dict(n_digit=4, codebook_size=256, encoder_n_layer=1, decoder_n_layer=4, n_embd=256, n_head=4, n_inner=1024),
        evaluation=dict(batch_size=4, val_beam=32, test_beam=32, candidate_lists=32, sensitivity_beam=256,
                        aggregate_ks=[10, 20, 50], selection='ndcg@5', tie_break='earlier', filter_history=False,
                        collision_rule='reconstruction_error_then_original_id',
                        aggregation='list_score_order_then_item_position_first_occurrence'),
        training=dict(effective_batch_size=1024, microbatch_size=16,
        warmup_epochs=5, cycles=10, r_epochs_per_cycle=3, t_updates_per_cycle=40,
        deterministic_algorithms=True))
    binding = dict(config=config, asset_manifest=sha, data=dict(manifest_sha256=sha, history_max=50, list_length=5, pad_id=0,
        files={name: {'sha256': sha, 'bytes': 123} for name in DATA_FILES}),
        code={name: sha for name in CODE_FILES}, engine={name: sha for name in ENGINE_FILES})
    hashes = {name: sha for name in ('model', 'effective_codebooks', 'hard_sids', 'r_optimizer', 't_optimizer', 'scheduler')}
    report = dict(status='passed', purpose='disposable_real_data_preflight_not_formal_training',
        synthetic_fixture=True, formal_training_started=False, torch_version='synthetic', cuda_version='synthetic', gate_batch=2,
        gpu='TEST FIXTURE H100', effective_batch=1024, microbatch=16, eval_batch=4, decode_batches=4,
        profile_updates=2, allocated_gpus=1, common_start={a: {'equal': True, 'hashes': dict(hashes)} for a in ('J', 'H')},
        data=dict(train_rows=709443, validation_rows=71657, catalog_items=10728, binding=binding,
                  fixed_training_sample_ids=['train-a', 'train-b'],
                  fixed_validation_sample_ids=ids, fixed_validation_indices=indices), arms={})
    for arm, factor in zip(ARMS, (1.0, 1.1, 2.0, 2.2)):
        arm_binding = copy.deepcopy(binding)
        arm_binding['config']['architecture'] = arm[0]
        backend = dict(cublas_workspace_config=':4096:8', cudnn_deterministic=True,
                       cudnn_benchmark=False, cuda_matmul_tf32=False)
        t_clock = 2 if arm.endswith('1') else 1
        item = dict(arm=arm, status='passed', precision='bf16', binding=arm_binding,
                    formal_training=False, small_batch_lists=2,
                    R=dict(tokenizer_unchanged=True, optimizer_steps=[1, 1], values=dict(loss=1.0, r_updates=1, t_updates=0),
                           changed_parameters=[f'backbone.decoder_blocks.{i}.{suffix}' for i in range(4)
                              for suffix in ('self_attn.qkv.weight', 'cross_attn.qkv.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')]),
                    T=dict(recommender_unchanged=True, optimizer_steps=[t_clock, t_clock],
                           values=dict(loss=1.0, r_updates=1, t_updates=1),
                           changed_parameters_first_step=['mixer.base_codebooks'] + (['mixer.attention.out_proj.weight'] if arm.endswith('1') else [])),
                    recommendation_gradient=dict(loss=1.0, base_gradient_l1=.1, mixer_output_gradient_l1=.1),
                    replay_R=dict(phase='R', exact_loaded_state=True, exact_next_update=True,
                                  checkpoint='/synthetic/R.pt', r_updates=2, t_updates=t_clock),
                    replay_T=dict(phase='T', exact_loaded_state=True, exact_next_update=True,
                                  checkpoint='/synthetic/T.pt', r_updates=2, t_updates=t_clock+1),
                    catalog_refresh=dict(verified_against_current_codebooks=True, changed_item_count=0),
                    generation_gate=dict(examples=4, complete_lists=8, best_lists=[[1,2,3,4,5] for _ in range(4)]),
                    replay_deterministic_algorithms=True,
                    replay_backend_settings=dict(backend, deterministic_warn_only=False, cudnn_tf32=False),
                    profile_backend_settings=dict(backend, deterministic_algorithms=True), profile={}, decode_profile={})
        for phase, core_seconds in (('R', 1.0), ('T', 2.0)):
            records = [dict(seconds=(core_seconds + .2 * i) * factor,
                            data_prepare_seconds=.1 * factor,
                            end_to_end_seconds=(core_seconds + .2 * i + .1) * factor,
                            effective_lists=1024, microbatch=16, accumulation=64) for i in range(2)]
            item['profile'][phase] = dict(records=records, median_end_to_end_seconds=(core_seconds + .2) * factor)
        for beam, seconds in ((32, .4), (256, 1.6)):
            records = [dict(precision='fp32', beam=beam, batch_size=4, sample_ids=ids[i*4:(i+1)*4],
                            dataset_indices=indices[i*4:(i+1)*4],
                            cache_state='cold' if i == 0 else 'naturally_warmed_by_distinct_requests',
                            seconds=seconds*factor, data_prepare_seconds=.1*factor,
                            end_to_end_seconds=(seconds+.1)*factor) for i in range(4)]
            item['decode_profile'][str(beam)] = dict(status='complete', planned_examples=16, measured_examples=16,
                records=records, fresh_catalog_timing={'seconds': .2*factor},
                overall_end_to_end_seconds=(.2+4*(seconds+.1)+.4)*factor,
                cache_policy='new SIDCatalog per beam; cold first batch and distinct later requests')
        report['arms'][arm] = item
    return report


def synthetic_supervisor():
    return dict(status='passed', reason=None, returncode=0, wall_seconds=200.0,
                allocated_gpu_seconds_during_script=200.0, output='/synthetic/remote/preflight')


class BudgetTests(unittest.TestCase):
    def test_shared_warmups_are_not_counted_once_per_final_model(self):
        work = workload()
        self.assertEqual(work['r_updates_per_epoch'], 693)
        self.assertEqual(work['last_train_batch_rows'], 835)
        self.assertEqual(work['shared_warmup_r_epochs'], 10)
        self.assertEqual(work['branch_r_epochs'], 120)
        self.assertEqual(work['total_executed_r_epochs'], 130)
        self.assertEqual(work['total_executed_r_updates'], 90090)
        self.assertEqual(work['total_executed_t_updates'], 1600)
        self.assertEqual(work['total_full_validations'], 42)
        self.assertEqual(work['test_rows'], 53194)
        self.assertEqual(work['final_tests'], 4)
        self.assertEqual(work['sensitivity_evaluations'], 8)

    def test_cost_adds_prior_allocation_once_and_reserves_only_future_work(self):
        result = estimate(synthetic_preflight(), used_gpu_seconds=1000,
                          price_per_gpu_hour=2.5, price_unit='fixture_units', margin=.25,
                          supervisor=synthetic_supervisor(), allow_synthetic=True)
        values = result['accounting']
        future = result['schedule_without_reserve']['gpu_seconds']
        self.assertEqual(values['already_used_gpu_seconds'], 1000)
        self.assertAlmostEqual(values['future_reserve_gpu_seconds'], .25 * future)
        self.assertAlmostEqual(values['cumulative_with_reserve_gpu_seconds'], 1000 + 1.25 * future)
        self.assertAlmostEqual(result['cost']['cumulative_with_reserve_estimate'], (1000 + 1.25*future) / 3600 * 2.5)
        self.assertFalse(result['cost']['settlement_verified'])
        self.assertEqual(result['status'], 'synthetic_test_estimate_not_measurement')
        self.assertIsNone(result['wall_time']['cumulative_elapsed_hours'])
        self.assertAlmostEqual(result['wall_time']['future_with_reserve_hours'], result['wall_time']['future_base_hours'] * 1.25)
        self.assertEqual(sum(c['r_updates'] for c in result['components'].values()), 90090)
        self.assertEqual(sum(c['t_updates'] for c in result['components'].values()), 1600)
        self.assertEqual(sum(c['full_validation_count'] for c in result['components'].values()), 42)

    def test_fp32_end_to_end_and_fixed_catalog_cost_are_used(self):
        normalized, measured = validate_preflight(synthetic_preflight(), allow_synthetic=True)
        self.assertGreater(measured, 0)
        self.assertAlmostEqual(normalized['J0']['training']['R']['median_end_to_end_seconds'], 1.2)
        decode = normalized['J0']['decoding']['32']
        self.assertAlmostEqual(decode['catalog_build_seconds'], .2)
        self.assertAlmostEqual(decode['variable_end_to_end_seconds_per_batch'], .6)
        self.assertAlmostEqual(evaluation_seconds(decode, 5), .2 + .6*2)
        self.assertAlmostEqual(evaluation_seconds(decode, 71657), .2 + .6*17915)

    def test_incomplete_missing_non_fp32_and_reused_request_profiles_are_rejected(self):
        variants = []
        report = synthetic_preflight(); report['status'] = 'partial'; variants.append(report)
        report = synthetic_preflight(); del report['arms']['H1']; variants.append(report)
        report = synthetic_preflight(); report['arms']['J0']['decode_profile']['256']['status'] = 'partial'; variants.append(report)
        report = synthetic_preflight(); report['arms']['J0']['profile']['T']['records'].pop(); variants.append(report)
        report = synthetic_preflight(); report['arms']['H0']['decode_profile']['32']['records'][1]['sample_ids'][0] = 'validation-0'; variants.append(report)
        report = synthetic_preflight(); report['arms']['J1']['decode_profile']['32']['records'][0]['precision'] = 'bf16'; variants.append(report)
        report = synthetic_preflight(); report['arms']['H1']['decode_profile']['32']['overall_end_to_end_seconds'] = .01; variants.append(report)
        for report in variants:
            with self.subTest(report=report.get('status')), self.assertRaises(ValueError):
                validate_preflight(report, allow_synthetic=True)

    def test_understated_ledger_invalid_price_and_implicit_price_are_rejected(self):
        for used, price in ((0, 1), (1000, -1), (1000, float('nan'))):
            with self.assertRaises(ValueError):
                estimate(synthetic_preflight(), used_gpu_seconds=used, price_per_gpu_hour=price, price_unit='units',
                         supervisor=synthetic_supervisor(), allow_synthetic=True)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--preflight', 'not-read.json', '--used-gpu-seconds', '1000', '--price-unit', 'units', '--output', 'not-written.json'])
        self.assertEqual(error.exception.code, 2)


class EvidenceValidationTests(unittest.TestCase):
    @staticmethod
    def all_configs(report):
        return [report['data']['binding']['config']] + [item['binding']['config'] for item in report['arms'].values()]

    def test_bound_evaluation_batch_must_equal_the_measured_batch(self):
        report = synthetic_preflight()
        for cfg in self.all_configs(report):
            cfg['evaluation']['batch_size'] = 1
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            validate_preflight(report, allow_synthetic=True)

    def test_bound_primary_beams_candidate_limit_and_sensitivity_are_locked(self):
        for key, value in (('val_beam', 256), ('test_beam', 256), ('candidate_lists', 64),
                           ('sensitivity_beam', 32), ('aggregate_ks', [5])):
            report = synthetic_preflight()
            for cfg in self.all_configs(report):
                cfg['evaluation'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'evaluation protocol'):
                validate_preflight(report, allow_synthetic=True)

    def test_four_bindings_may_differ_only_in_the_correct_architecture(self):
        for kind in ('config', 'source', 'asset', 'architecture'):
            report = synthetic_preflight()
            bound = report['arms']['J1']['binding']
            if kind == 'config': bound['config']['seed'] = 2025
            if kind == 'source': bound['code'][CODE_FILES[0]] = 'f' * 64
            if kind == 'asset': bound['asset_manifest'] = 'e' * 64
            if kind == 'architecture': bound['config']['architecture'] = 'H'
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'binding differs|architecture differs'):
                validate_preflight(report, allow_synthetic=True)

    def test_missing_gate_and_replay_evidence_is_rejected_despite_passed_status(self):
        mutations = [
            lambda arm: arm.pop('R'),
            lambda arm: arm['R']['changed_parameters'].pop(),
            lambda arm: arm['R'].update(tokenizer_unchanged=False),
            lambda arm: arm['T'].update(recommender_unchanged=False),
            lambda arm: arm['T']['changed_parameters_first_step'].remove('mixer.attention.out_proj.weight'),
            lambda arm: arm['recommendation_gradient'].update(base_gradient_l1=0),
            lambda arm: arm['recommendation_gradient'].pop('mixer_output_gradient_l1'),
            lambda arm: arm.pop('replay_R'),
            lambda arm: arm['replay_T'].update(exact_next_update=False),
            lambda arm: arm['replay_R'].update(exact_loaded_state=False),
            lambda arm: arm['catalog_refresh'].update(verified_against_current_codebooks=False),
            lambda arm: arm['generation_gate']['best_lists'].__setitem__(0, [1, 1, 2, 3, 4]),
            lambda arm: arm['replay_backend_settings'].update(deterministic_warn_only=True),
            lambda arm: arm['profile_backend_settings'].update(cuda_matmul_tf32=True),
        ]
        for i, mutate in enumerate(mutations):
            report = synthetic_preflight()
            mutate(report['arms']['J1'])
            with self.subTest(case=i), self.assertRaises(ValueError):
                validate_preflight(report, allow_synthetic=True)

    def test_missing_source_dataset_or_common_state_hashes_are_rejected(self):
        for kind in ('source', 'engine', 'asset', 'dataset', 'common'):
            report = synthetic_preflight()
            bound = report['data']['binding']
            if kind == 'source': bound['code'].pop(CODE_FILES[0])
            if kind == 'engine': bound['engine'].pop(ENGINE_FILES[0])
            if kind == 'asset': bound['asset_manifest'] = 'not-a-hash'
            if kind == 'dataset': bound['data']['files']['validation.parquet'].pop('sha256')
            if kind == 'common': report['common_start']['J']['hashes'].pop('model')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_preflight(report, allow_synthetic=True)

    def test_receipt_requires_success_and_complete_allocation_accounting(self):
        report = synthetic_preflight()
        _, lower = validate_preflight(report, allow_synthetic=True)
        self.assertEqual(validate_supervisor(synthetic_supervisor(), report, lower), 200)
        for receipt in (None, dict(synthetic_supervisor(), status='timeout'),
                        dict(synthetic_supervisor(), returncode=2),
                        dict(synthetic_supervisor(), allocated_gpu_seconds_during_script=201),
                        dict(synthetic_supervisor(), wall_seconds=1, allocated_gpu_seconds_during_script=1)):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                validate_supervisor(receipt, report, lower)
        # The aggregate timing lower bound is below 150, but the successful
        # supervisor allocated 200 seconds, so a ledger of 150 still understates.
        self.assertLess(lower, 150)
        with self.assertRaisesRegex(ValueError, 'supervised preflight allocation'):
            estimate(report, used_gpu_seconds=150, price_per_gpu_hour=1, price_unit='fixture_units',
                     supervisor=synthetic_supervisor(), allow_synthetic=True)

    def test_cli_rejects_missing_receipt_and_synthetic_measurements(self):
        with tempfile.TemporaryDirectory(prefix='budget_evidence_', dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            report = root / 'preflight.json'
            output = root / 'must_not_exist.json'
            report.write_text(json.dumps(synthetic_preflight()), encoding='utf-8')
            arguments = ['--preflight', str(report), '--used-gpu-seconds', '1000', '--price-per-gpu-hour', '1',
                         '--price-unit', 'fixture_units', '--output', str(output)]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)
            (root / 'supervisor.json').write_text(json.dumps(synthetic_supervisor()), encoding='utf-8')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            validate_preflight(synthetic_preflight())


class SchedulingTests(unittest.TestCase):
    @staticmethod
    def jobs():
        jobs = [dict(id='J_shared', depends_on=[], seconds=10, priority=0),
                dict(id='H_shared', depends_on=[], seconds=20, priority=0)]
        jobs += [dict(id=arm+'_train', depends_on=[arm[0]+'_shared'], seconds=seconds, priority=1)
                 for arm, seconds in zip(ARMS, (100, 30, 40, 80))]
        jobs += [dict(id=arm+'_final', depends_on=[other+'_train' for other in ARMS], seconds=seconds, priority=2)
                 for arm, seconds in zip(ARMS, (10, 20, 30, 40))]
        return jobs

    def test_branches_start_after_own_warmup_not_a_global_warmup_barrier(self):
        got = schedule(self.jobs(), max_gpus=4)
        jobs = {row['id']: row for row in got['jobs']}
        self.assertEqual(jobs['J_shared']['start_seconds'], 0)
        self.assertEqual(jobs['H_shared']['start_seconds'], 0)
        self.assertEqual(jobs['J0_train']['start_seconds'], 10)
        self.assertEqual(jobs['J1_train']['start_seconds'], 10)
        self.assertEqual(jobs['H0_train']['start_seconds'], 20)
        self.assertEqual(jobs['H1_train']['start_seconds'], 20)
        self.assertTrue(all(jobs[arm+'_final']['start_seconds'] == 110 for arm in ARMS))
        self.assertEqual(got['wall_seconds'], 150)
        self.assertEqual(got['gpu_seconds'], 380)
        self.assertEqual(got['peak_gpus'], 4)

    def test_resource_limit_and_dependencies_hold_for_one_to_four_cards(self):
        for limit in (1, 2, 3, 4):
            result = schedule(self.jobs(), max_gpus=limit)
            rows = {row['id']: row for row in result['jobs']}
            for row in rows.values():
                self.assertTrue(all(row['start_seconds'] >= rows[parent]['end_seconds'] for parent in row['depends_on']))
                running = sum(other['start_seconds'] <= row['start_seconds'] < other['end_seconds'] for other in rows.values())
                self.assertLessEqual(running, limit)
            self.assertEqual(result['gpu_seconds'], 380)
        self.assertEqual(schedule(self.jobs(), max_gpus=1)['wall_seconds'], 380)
        with self.assertRaises(ValueError):
            schedule(self.jobs(), max_gpus=8)
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            schedule([dict(id='x', depends_on=['x'], seconds=1, priority=0)])


if __name__ == '__main__':
    unittest.main()
