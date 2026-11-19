"""Offline Stage09 GPU-hour/fee estimate from a complete four-arm preflight.

No platform, model, torch, or training imports. A price and cumulative already
allocated GPU seconds must be supplied explicitly. Synthetic tests are not
measurements and this program never fills in missing performance profiles.
A passed supervisor.json from the same delivered preflight bundle must sit
beside the report. Both file hashes are retained in the resulting estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ARMS = ('J0', 'J1', 'H0', 'H1')
TRAIN_ROWS, VALIDATION_ROWS, TEST_ROWS = 709443, 71657, 53194
EFFECTIVE_BATCH, SUBSET_ROWS = 1024, 1024
WARMUP_EPOCHS, CYCLES, R_EPOCHS_PER_CYCLE, T_UPDATES_PER_CYCLE = 5, 10, 3, 40
DATA_FILES = ('config.json', 'item_map.json', 'item_catalog.parquet', 'item_text.jsonl',
              'train_seen.json', 'train.parquet', 'validation.parquet', 'test.parquet')
CODE_FILES = tuple('scripts/listwise_diffgrm/' + name + '.py' for name in
                   ('run', 'runtime', 'model', 'training', 'alignment', 'assets', 'data', 'decoding', 'metrics', 'preflight'))
ENGINE_FILES = ('genrec/model.py', 'genrec/dataset.py', 'genrec/tokenizer.py',
                'genrec/models/DIFF_GRM/model.py', 'genrec/models/DIFF_GRM/joint_cross_pse.py',
                'genrec/models/DIFF_GRM/cross_pse.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name, *, allow_zero=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value >= 0 if allow_zero else value > 0),
            f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} number')
    return float(value)


def same_number(actual, expected, name):
    require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6),
            f'{name} disagrees with its detailed records')


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            f'{name} needs a SHA256 identity')


def binding_identity(binding):
    """Validate source/data evidence and remove only architecture for pairing."""
    require(isinstance(binding, dict), 'Missing execution binding')
    config = binding.get('config', {})
    require(config.get('architecture') in ('J', 'H'), 'Missing bound architecture')
    digest(binding.get('asset_manifest'), 'asset manifest')
    for section, required in (('code', CODE_FILES), ('engine', ENGINE_FILES)):
        values = binding.get(section, {})
        require(isinstance(values, dict) and set(required) <= values.keys(), f'Missing {section} source hashes')
        for path, sha in values.items():
            require(isinstance(path, str) and '\\' not in path and not path.startswith('/')
                    and all(part not in ('', '.', '..') for part in path.split('/')), f'Invalid relative source identity: {path}')
            digest(sha, path)
    data = binding.get('data', {})
    digest(data.get('manifest_sha256'), 'dataset manifest')
    require((data.get('history_max'), data.get('list_length'), data.get('pad_id')) == (50, 5, 0), 'Bound dataset protocol differs')
    files = data.get('files', {})
    require(set(DATA_FILES) <= files.keys(), 'Missing dataset file hashes')
    for name in DATA_FILES:
        digest(files[name].get('sha256'), 'dataset/' + name)
        require(type(files[name].get('bytes')) is int and files[name]['bytes'] > 0, 'Invalid bound dataset file size')
    return dict(binding, config={name: value for name, value in config.items() if name != 'architecture'})


def validate_gates(current, arm, *, batch, gate_batch, catalog_items):
    require(current.get('formal_training') is False and current.get('small_batch_lists') == gate_batch,
            f'{arm}: missing disposable-gate batch identity')
    r, t = current.get('R', {}), current.get('T', {})
    required_updates = {f'backbone.decoder_blocks.{layer}.{suffix}' for layer in range(4)
                        for suffix in ('self_attn.qkv.weight', 'cross_attn.qkv.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')}
    require(r.get('tokenizer_unchanged') is True and required_updates <= set(r.get('changed_parameters', [])),
            f'{arm}: R partition or four-layer actual-update gate is missing')
    require(r.get('optimizer_steps') and all(step == 1 for step in r['optimizer_steps']), f'{arm}: R gate optimizer clocks differ')
    changed = set(t.get('changed_parameters_first_step', []))
    require(t.get('recommender_unchanged') is True and 'mixer.base_codebooks' in changed, f'{arm}: T partition gate is missing')
    t_clock = 2 if arm.endswith('1') else 1
    if arm.endswith('1'):
        require('mixer.attention.out_proj.weight' in changed, f'{arm}: Mixer actual-update gate is missing')
    else:
        require(changed == {'mixer.base_codebooks'}, f'{arm}: disabled Mixer unexpectedly changed')
    require(t.get('optimizer_steps') and all(step == t_clock for step in t['optimizer_steps']), f'{arm}: T gate optimizer clocks differ')
    for phase, section, expected_t in (('R', r, 0), ('T', t, 1)):
        values = section.get('values', {})
        positive(values.get('loss'), arm + '/' + phase + '/gate_loss')
        require(values.get('r_updates') == 1 and values.get('t_updates') == expected_t,
                f'{arm}: {phase} update evidence differs from its gate boundary')
    gradient = current.get('recommendation_gradient', {})
    positive(gradient.get('loss'), arm + '/recommendation-only loss')
    positive(gradient.get('base_gradient_l1'), arm + '/recommendation-only PSE gradient')
    if arm.endswith('1'):
        positive(gradient.get('mixer_output_gradient_l1'), arm + '/recommendation-only Mixer gradient')
    for phase, expected_t in (('R', t_clock), ('T', t_clock + 1)):
        replay = current.get('replay_' + phase, {})
        require(replay.get('phase') == phase and replay.get('exact_loaded_state') is True
                and replay.get('exact_next_update') is True, f'{arm}: missing exact {phase} checkpoint replay')
        require(replay.get('r_updates') == 2 and replay.get('t_updates') == expected_t,
                f'{arm}: replay update counts do not match the gate sequence')
        require(isinstance(replay.get('checkpoint'), str) and replay['checkpoint'], f'{arm}: replay checkpoint source is missing')
    refresh = current.get('catalog_refresh', {})
    require(refresh.get('verified_against_current_codebooks') is True, f'{arm}: SID refresh was not verified')
    require(type(refresh.get('changed_item_count')) is int and 0 <= refresh['changed_item_count'] <= catalog_items,
            f'{arm}: invalid SID refresh count')
    generation = current.get('generation_gate', {})
    lists = generation.get('best_lists', [])
    require(generation.get('examples') == batch and len(lists) == batch,
            f'{arm}: missing target-free list-generation gate')
    require(type(generation.get('complete_lists')) is int and batch <= generation['complete_lists'] <= 2 * batch,
            f'{arm}: incomplete generation gate')
    require(all(len(items) == 5 and len(set(items)) == 5
                and all(type(item) is int and 1 <= item <= catalog_items for item in items) for items in lists),
            f'{arm}: generation gate contains invalid or duplicate items')
    require(current.get('replay_deterministic_algorithms') is True, f'{arm}: replay determinism gate missing')
    replay = current.get('replay_backend_settings', {})
    profile = current.get('profile_backend_settings', {})
    require(replay.get('deterministic_warn_only') is False and profile.get('deterministic_algorithms') is True,
            f'{arm}: strict deterministic execution is missing')
    for key, value in (('cublas_workspace_config', ':4096:8'), ('cudnn_deterministic', True),
                       ('cudnn_benchmark', False), ('cuda_matmul_tf32', False)):
        require(replay.get(key) == value and profile.get(key) == value, f'{arm}: replay/profile backend differs: {key}')
    require(replay.get('cudnn_tf32') is False, f'{arm}: replay TF32 policy differs')


def workload():
    updates_per_epoch = math.ceil(TRAIN_ROWS / EFFECTIVE_BATCH)
    shared_epochs = 2 * WARMUP_EPOCHS
    branch_epochs = len(ARMS) * CYCLES * R_EPOCHS_PER_CYCLE
    return dict(train_rows=TRAIN_ROWS, validation_rows=VALIDATION_ROWS, test_rows=TEST_ROWS,
                effective_batch=EFFECTIVE_BATCH, last_train_batch_rows=TRAIN_ROWS % EFFECTIVE_BATCH,
                r_updates_per_epoch=updates_per_epoch,
                shared_warmups=2, shared_warmup_r_epochs=shared_epochs,
                shared_warmup_r_updates=shared_epochs * updates_per_epoch,
                branches=4, branch_r_epochs=branch_epochs,
                branch_r_updates=branch_epochs * updates_per_epoch,
                total_executed_r_epochs=shared_epochs + branch_epochs,
                total_executed_r_updates=(shared_epochs + branch_epochs) * updates_per_epoch,
                total_executed_t_updates=len(ARMS) * CYCLES * T_UPDATES_PER_CYCLE,
                total_t_refresh_boundaries=len(ARMS) * CYCLES,
                warmup_full_validations=2, branch_full_validations=len(ARMS) * CYCLES,
                total_full_validations=2 + len(ARMS) * CYCLES,
                final_tests=4, sensitivity_evaluations=8, sensitivity_rows_per_evaluation=SUBSET_ROWS,
                logical_final_model_r_epochs=35,
                note='Two shared 5R warmups are executed once each; four branches add 30R each, so actual total is 130R, not 140R.')


def validate_preflight(report, *, allow_synthetic=False):
    """Reject partial/failed/missing-arm profiles and repeated warm requests."""
    require(report.get('status') == 'passed', 'Preflight must be completely passed, not running/partial/failed')
    require(report.get('purpose') == 'disposable_real_data_preflight_not_formal_training', 'Unexpected preflight purpose')
    require(report.get('formal_training_started') is False, 'Preflight must explicitly exclude formal training')
    fixture = report.get('synthetic_fixture', False)
    require(fixture is False or (fixture is True and allow_synthetic), 'Synthetic fixtures cannot establish a measured budget')
    require(not allow_synthetic or fixture is True, 'Test-only synthetic mode requires an explicit fixture marker')
    require('h100' in str(report.get('gpu', '')).lower(), 'Expected real H100 profile')
    require(allow_synthetic or not any(word in str(report['gpu']).lower() for word in ('fixture', 'synthetic', 'test')),
            'Synthetic hardware labels are not measurement evidence')
    require(all(isinstance(report.get(name), str) and report[name] for name in ('torch_version', 'cuda_version')),
            'Missing runtime version evidence')
    require(report.get('effective_batch') == EFFECTIVE_BATCH, 'Expected 1024-list effective batches')
    arms = report.get('arms', {})
    require(set(arms) == set(ARMS), 'Complete J0/J1/H0/H1 profiles are all required')
    require(all(report.get('common_start', {}).get(architecture, {}).get('equal') is True for architecture in ('J', 'H')),
            'Both architecture-paired initial-state gates must pass')
    for architecture in ('J', 'H'):
        hashes = report['common_start'][architecture].get('hashes', {})
        for name in ('model', 'effective_codebooks', 'hard_sids', 'r_optimizer', 't_optimizer', 'scheduler'):
            digest(hashes.get(name), architecture + '/common_start/' + name)
    data = report['data']
    require((data['train_rows'], data['validation_rows']) == (TRAIN_ROWS, VALIDATION_ROWS), 'Profile uses another dataset size')
    require(data.get('catalog_items') == 10728, 'Profile must use the locked complete real item catalog')
    identity = binding_identity(data.get('binding'))
    config = data['binding']['config']
    training = config['training']
    model = config.get('model', {})
    require((config.get('history_max'), config.get('list_length'), model.get('n_digit'), model.get('codebook_size'),
             model.get('encoder_n_layer'), model.get('decoder_n_layer'), model.get('n_embd'), model.get('n_head'), model.get('n_inner'))
            == (50, 5, 4, 256, 1, 4, 256, 4, 1024), 'Profile is not the complete formal model')
    expected = dict(effective_batch_size=EFFECTIVE_BATCH, warmup_epochs=WARMUP_EPOCHS, cycles=CYCLES,
                    r_epochs_per_cycle=R_EPOCHS_PER_CYCLE, t_updates_per_cycle=T_UPDATES_PER_CYCLE,
                    deterministic_algorithms=True)
    require(all(training.get(name) == value for name, value in expected.items()), 'Profile configuration differs from locked formal workload')
    microbatch = report['microbatch']
    require(type(microbatch) is int and microbatch > 0 and EFFECTIVE_BATCH % microbatch == 0, 'Invalid training microbatch')
    require(training['microbatch_size'] == microbatch, 'Profile and bound training microbatches differ')
    batch, batches, repeats = report['eval_batch'], report['decode_batches'], report['profile_updates']
    require(all(type(value) is int and value > 0 for value in (batch, batches, repeats)), 'Invalid measurement counts')
    require(batches >= 4 and batch * batches >= 16, 'Need at least four distinct decode batches / sixteen validation requests')
    evaluation = config.get('evaluation', {})
    require(evaluation.get('batch_size') == batch, 'Bound evaluation batch_size differs from measured eval_batch')
    expected_eval = dict(val_beam=32, test_beam=32, candidate_lists=32, sensitivity_beam=256,
                         aggregate_ks=[10, 20, 50], selection='ndcg@5', tie_break='earlier', filter_history=False,
                         collision_rule='reconstruction_error_then_original_id',
                         aggregation='list_score_order_then_item_position_first_occurrence')
    require(all(evaluation.get(name) == value for name, value in expected_eval.items()), 'Bound evaluation protocol differs from budget workload')
    gate_batch = report.get('gate_batch')
    require(type(gate_batch) is int and gate_batch > 0, 'Missing gate batch size')
    gate_ids = data.get('fixed_training_sample_ids', [])
    require(len(gate_ids) == len(set(gate_ids)) == gate_batch and all(isinstance(value, str) and value for value in gate_ids),
            'Missing distinct training gate sample identities')
    require(type(report['allocated_gpus']) is int and report['allocated_gpus'] >= 1, 'Invalid preflight allocation count')
    ids, indices = data['fixed_validation_sample_ids'], data['fixed_validation_indices']
    require(len(ids) == batch * batches == len(indices) and len(set(ids)) == len(ids) and len(set(indices)) == len(indices),
            'Fixed validation requests must be distinct and complete')
    require(all(isinstance(value, str) and value for value in ids), 'Validation sample IDs must be exact text')
    require(all(type(index) is int and 0 <= index < VALIDATION_ROWS for index in indices), 'Invalid validation indices')
    normalized, measured_wall_lower_bound = {}, 0.0
    for arm in ARMS:
        current = arms[arm]
        require(current.get('status') == 'passed' and current.get('arm') == arm, f'{arm}: profile/gates are incomplete')
        require(current.get('precision') == 'bf16', f'{arm}: training profile is not BF16')
        require(binding_identity(current.get('binding')) == identity, f'{arm}: source/assets/data/config binding differs')
        require(current['binding']['config']['architecture'] == arm[0], f'{arm}: bound architecture differs')
        validate_gates(current, arm, batch=batch, gate_batch=gate_batch, catalog_items=data['catalog_items'])
        result = {'training': {}, 'decoding': {}}
        for phase in ('R', 'T'):
            section = current.get('profile', {}).get(phase, {})
            records = section.get('records', [])
            require(len(records) == repeats, f'{arm}/{phase}: missing update measurements')
            times = []
            for record in records:
                require(record['effective_lists'] == EFFECTIVE_BATCH and record['microbatch'] == microbatch
                        and record['accumulation'] == EFFECTIVE_BATCH // microbatch, f'{arm}/{phase}: incomplete or altered effective update')
                core = positive(record['seconds'], f'{arm}/{phase}/seconds')
                prepare = positive(record['data_prepare_seconds'], f'{arm}/{phase}/data_prepare_seconds', allow_zero=True)
                total = positive(record['end_to_end_seconds'], f'{arm}/{phase}/end_to_end_seconds')
                same_number(total, core + prepare, f'{arm}/{phase}/end_to_end_seconds')
                times.append(total)
                measured_wall_lower_bound += total
            median = statistics.median(times)
            same_number(section['median_end_to_end_seconds'], median, f'{arm}/{phase}/median')
            result['training'][phase] = dict(median_end_to_end_seconds=median, minimum=min(times), maximum=max(times),
                                             measured_updates=len(times), lists_per_update=EFFECTIVE_BATCH)
        for beam in (32, 256):
            section = current.get('decode_profile', {}).get(str(beam), {})
            require(section.get('status') == 'complete', f'{arm}/beam{beam}: partial or missing decode profile')
            require(section.get('planned_examples') == section.get('measured_examples') == len(ids), f'{arm}/beam{beam}: measured request count differs')
            records = section.get('records', [])
            require(len(records) == batches, f'{arm}/beam{beam}: incomplete measured batches')
            all_ids, all_indices, record_total = [], [], 0.0
            for iteration, record in enumerate(records):
                require(record['precision'] == 'fp32' and record['beam'] == beam and record['batch_size'] == batch,
                        f'{arm}/beam{beam}: precision, beam or batch changed')
                require(len(record['sample_ids']) == batch == len(record['dataset_indices']), f'{arm}/beam{beam}: incomplete batch identities')
                require(record['cache_state'] == ('cold' if iteration == 0 else 'naturally_warmed_by_distinct_requests'),
                        f'{arm}/beam{beam}: repeated artificially warmed request measurements are not supported')
                all_ids.extend(record['sample_ids'])
                all_indices.extend(record['dataset_indices'])
                core = positive(record['seconds'], f'{arm}/beam{beam}/seconds')
                prepare = positive(record['data_prepare_seconds'], f'{arm}/beam{beam}/prepare', allow_zero=True)
                total = positive(record['end_to_end_seconds'], f'{arm}/beam{beam}/end_to_end')
                same_number(total, core + prepare, f'{arm}/beam{beam}/end_to_end')
                record_total += total
            require(all_ids == ids and all_indices == indices, f'{arm}/beam{beam}: requests differ, repeat, or are reordered')
            catalog = positive(section['fresh_catalog_timing']['seconds'], f'{arm}/beam{beam}/catalog', allow_zero=True)
            overall = positive(section['overall_end_to_end_seconds'], f'{arm}/beam{beam}/overall')
            require(overall + 1e-6 >= catalog + record_total, f'{arm}/beam{beam}: overall duration omits measured work')
            variable_per_batch = (overall - catalog) / batches
            result['decoding'][str(beam)] = dict(catalog_build_seconds=catalog,
                measured_overall_end_to_end_seconds=overall, measured_requests=len(ids), measured_batches=batches,
                eval_batch_size=batch, variable_end_to_end_seconds_per_batch=variable_per_batch,
                sample_ids_sha256=hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
                precision='fp32', cache_policy=section['cache_policy'],
                scope='catalog + Arrow/collation + target-free generation + correctness checks + logging; excludes full metrics and prediction serialization')
            measured_wall_lower_bound += overall
        normalized[arm] = result
    return normalized, measured_wall_lower_bound * report['allocated_gpus']


def validate_supervisor(receipt, report, measured_lower_bound):
    """Check a completed bundle receipt, not a hardware authenticity proof."""
    require(isinstance(receipt, dict), 'A completed supervisor.json receipt is required')
    require(receipt.get('status') == 'passed' and type(receipt.get('returncode')) is int
            and receipt['returncode'] == 0 and receipt.get('reason') is None,
            'Supervisor did not confirm a successful completed worker')
    wall = positive(receipt.get('wall_seconds'), 'supervisor wall seconds')
    gpu = positive(receipt.get('allocated_gpu_seconds_during_script'), 'supervisor allocation GPU seconds')
    same_number(gpu, wall * report['allocated_gpus'], 'supervisor allocation accounting')
    require(gpu + 1e-6 >= measured_lower_bound, 'Supervisor allocation duration omits measured preflight work')
    require(isinstance(receipt.get('output'), str) and receipt['output'], 'Supervisor output provenance is missing')
    # The receipt can be downloaded from a remote path, so do not compare its
    # original output path with the local copy's directory.
    return gpu


def evaluation_seconds(profile, rows):
    """One fresh catalog plus measured end-to-end variable time per batch.

    Round up to a full last batch. Cold request cost is retained in the observed
    average; do not multiply a one-time catalog build by every 16 requests.
    """
    return profile['catalog_build_seconds'] + math.ceil(rows / profile['eval_batch_size']) * profile['variable_end_to_end_seconds_per_batch']


def schedule(jobs, max_gpus=4):
    """Deterministic single-GPU DAG list scheduling with prompt GPU release."""
    require(type(max_gpus) is int and 1 <= max_gpus <= 4, 'Concurrency must be 1..4 one-GPU jobs')
    remaining = {job['id']: dict(job) for job in jobs}
    require(len(remaining) == len(jobs), 'Duplicate scheduled job ID')
    require(all(set(job['depends_on']) <= remaining.keys() for job in jobs), 'Unknown job dependency')
    for job in jobs:
        positive(job['seconds'], job['id'] + ' duration')
    active, completed, timeline = [], {}, []
    now, peak = 0.0, 0
    while remaining or active:
        ready = sorted((job for job in remaining.values() if set(job['depends_on']) <= completed.keys()),
                       key=lambda job: (job['priority'], job['id']))
        for job in ready[:max_gpus - len(active)]:
            occupied = {row['gpu_slot'] for row in active}
            slot = next(slot for slot in range(max_gpus) if slot not in occupied)
            row = dict(job, gpu_slot=slot, start_seconds=now, end_seconds=now + job['seconds'])
            active.append(row)
            timeline.append(row)
            del remaining[job['id']]
        peak = max(peak, len(active))
        if not active:
            require(not remaining, 'Cyclic or unsatisfied scheduling dependency')
            break
        now = min(row['end_seconds'] for row in active)
        done = [row for row in active if row['end_seconds'] <= now]
        for row in done:
            completed[row['id']] = row['end_seconds']
            active.remove(row)
    return dict(wall_seconds=now, peak_gpus=peak, gpu_seconds=sum(job['seconds'] for job in jobs), jobs=timeline)


def estimate(report, *, used_gpu_seconds, price_per_gpu_hour, price_unit, margin=.25, max_gpus=4,
             supervisor=None, allow_synthetic=False):
    profiles, measured_lower_bound = validate_preflight(report, allow_synthetic=allow_synthetic)
    supervised_gpu_seconds = validate_supervisor(supervisor, report, measured_lower_bound)
    used = positive(used_gpu_seconds, 'already used GPU seconds', allow_zero=True)
    price = positive(price_per_gpu_hour, 'explicit price per GPU hour', allow_zero=True)
    require(isinstance(price_unit, str) and price_unit.strip(), 'Explicit price unit is required')
    require(isinstance(margin, (int, float)) and math.isfinite(margin) and 0 < margin <= 1,
            'Future-work reserve fraction must be in (0,1]; it is a planning allowance, not a measured CI')
    require(used + 1e-6 >= supervised_gpu_seconds,
            'Already-used allocation ledger understates the completed supervised preflight allocation')
    work = workload()
    per_epoch = work['r_updates_per_epoch']
    jobs, components = [], {}
    for architecture in ('J', 'H'):
        profile = profiles[architecture + '0']
        training = WARMUP_EPOCHS * per_epoch * profile['training']['R']['median_end_to_end_seconds']
        validation = evaluation_seconds(profile['decoding']['32'], VALIDATION_ROWS)
        name = architecture + '_shared_warmup'
        components[name] = dict(r_updates=WARMUP_EPOCHS * per_epoch, t_updates=0,
                                training_seconds=training, full_validation_count=1, validation_seconds=validation)
        jobs.append(dict(id=name, kind='shared_warmup_and_R5_validation', depends_on=[], priority=0,
                         seconds=training + validation))
    for arm in ARMS:
        profile = profiles[arm]
        r = CYCLES * R_EPOCHS_PER_CYCLE * per_epoch * profile['training']['R']['median_end_to_end_seconds']
        t = CYCLES * T_UPDATES_PER_CYCLE * profile['training']['T']['median_end_to_end_seconds']
        validation = CYCLES * evaluation_seconds(profile['decoding']['32'], VALIDATION_ROWS)
        components[arm] = dict(r_updates=CYCLES * R_EPOCHS_PER_CYCLE * per_epoch, t_updates=CYCLES * T_UPDATES_PER_CYCLE,
                               r_seconds=r, t_seconds=t, full_validation_count=CYCLES, validation_seconds=validation,
                               test_seconds=evaluation_seconds(profile['decoding']['32'], TEST_ROWS),
                               sensitivity_beam32_seconds=evaluation_seconds(profile['decoding']['32'], SUBSET_ROWS),
                               sensitivity_beam256_seconds=evaluation_seconds(profile['decoding']['256'], SUBSET_ROWS))
        jobs.append(dict(id=arm + '_train', kind='branch_30R_400T_and_10_validations',
                         depends_on=[arm[0] + '_shared_warmup'], priority=1, seconds=r + t + validation))
    for arm in ARMS:
        part = components[arm]
        jobs.append(dict(id=arm + '_final_evaluation', kind='test_and_fixed_best_beam32_256_sensitivity',
            depends_on=[other + '_train' for other in ARMS], priority=2,
            seconds=part['test_seconds'] + part['sensitivity_beam32_seconds'] + part['sensitivity_beam256_seconds']))
    base = schedule(jobs, max_gpus=max_gpus)
    reserved = schedule([dict(job, seconds=job['seconds'] * (1 + margin)) for job in jobs], max_gpus=max_gpus)
    future = base['gpu_seconds']
    allowance = future * margin
    cumulative = used + future + allowance
    return dict(format='stage09_offline_budget_v2',
                status='synthetic_test_estimate_not_measurement' if allow_synthetic else 'measurement_based_estimate_requiring_budget_approval',
                synthetic_fixture=bool(allow_synthetic),
                workload=work, measured_profiles=profiles, components=components,
                accounting=dict(already_used_gpu_seconds=used, measured_preflight_lower_bound_gpu_seconds=measured_lower_bound,
                    supervised_preflight_allocation_gpu_seconds=supervised_gpu_seconds,
                    future_base_gpu_seconds=future, future_reserve_fraction=margin, future_reserve_gpu_seconds=allowance,
                    future_with_reserve_gpu_seconds=future + allowance, cumulative_with_reserve_gpu_seconds=cumulative,
                    already_used_gpu_hours=used / 3600, future_base_gpu_hours=future / 3600,
                    future_reserve_gpu_hours=allowance / 3600, cumulative_with_reserve_gpu_hours=cumulative / 3600),
                wall_time=dict(scope='remaining formal work from simultaneous shared-warmup launch; excludes elapsed preparation and platform queue',
                               future_base_hours=base['wall_seconds'] / 3600,
                               future_with_reserve_hours=reserved['wall_seconds'] / 3600,
                               cumulative_elapsed_hours=None, max_parallel_single_gpu_jobs=max_gpus),
                cost=dict(price_per_gpu_hour=price, unit=price_unit,
                          already_used_estimate=used / 3600 * price,
                          future_with_reserve_estimate=(future + allowance) / 3600 * price,
                          cumulative_with_reserve_estimate=cumulative / 3600 * price,
                          settlement_verified=False),
                schedule_without_reserve=base, schedule_with_reserve=reserved,
                assumptions=[
                    'Each active job occupies one H100; idle allocations are released immediately. Reserved-node idle GPUs and external queue time are not inferred.',
                    'J/H warmups run once each using the corresponding no-Mixer arm timing; branch jobs become ready after their own architecture warmup and R5 validation.',
                    'All four best checkpoints must be locked before final test evaluation; test and two subset passes are sequential within each arm and parallel across at most four arms.',
                    'R/T use median measured BF16 end-to-end effective-1024 update time. The short epoch tail is charged as a full update, conservatively.',
                    'FP32 decoding uses distinct validation requests, including a cold first batch and natural cache reuse; catalog construction is charged once per evaluation pass.',
                    'All 42 full validations and four 53194-example tests are counted; all four arms evaluate 1024 requests at each of beam32 and beam256.',
                    'The percentage reserve applies to future work only. Already-used GPU seconds must include encoding, preflight, retries, setup and idle allocations from the actual ledger.'
                ],
                limitations=[
                    'Gate fields, matching source/data hashes and the supervisor receipt establish delivered-bundle consistency; they are not cryptographic hardware attestation.',
                    'Short preflight timings are extrapolated, not full validation/test measurements. A trained model can change beam paths and legality-cache behavior.',
                    'The reported FP32 end-to-end profile excludes full metric computation and prediction serialization; checkpoint I/O, 40 SID refreshes, dataset startup, resource contention and recovery overhead are not separately measured.',
                    'The explicit reserve covers these unmeasured overheads as a planning allowance, not a demonstrated upper bound or statistical confidence interval.',
                    'Test throughput is extrapolated from validation requests without reading test labels. Different history lengths and distribution can change runtime.',
                    'No numerical expense is an actual settlement; the explicit unit price may differ from platform final billing.'
                ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, help='Complete report.json/preflight.json with all four passed arm profiles')
    parser.add_argument('--used-gpu-seconds', type=float, required=True, help='Actual cumulative allocation ledger, including all preparation and preflight')
    parser.add_argument('--price-per-gpu-hour', type=float, required=True)
    parser.add_argument('--price-unit', required=True, help='Explicit unit, for example platform_points or CNY')
    parser.add_argument('--margin', type=float, default=.25, help='Explicit future-work allowance; default 0.25, not a measured confidence bound')
    parser.add_argument('--max-gpus', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    path = Path(args.preflight)
    try:
        raw = path.read_bytes()
        receipt_path = path.parent / 'supervisor.json'
        receipt_raw = receipt_path.read_bytes()
        result = estimate(json.loads(raw), used_gpu_seconds=args.used_gpu_seconds,
                          price_per_gpu_hour=args.price_per_gpu_hour, price_unit=args.price_unit,
                          margin=args.margin, max_gpus=args.max_gpus, supervisor=json.loads(receipt_raw))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(f'No budget generated: invalid or incomplete measurement evidence: {exc}')
    result['input'] = dict(preflight_path=str(path), preflight_sha256=hashlib.sha256(raw).hexdigest(),
                          supervisor_path=str(receipt_path), supervisor_sha256=hashlib.sha256(receipt_raw).hexdigest())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), **result['accounting'], wall_time=result['wall_time'], cost=result['cost']), ensure_ascii=False))


if __name__ == '__main__':
    main()
