"""Offline H1 recovery budget arithmetic from two accepted, real preflight reports.

No torch, platform API, submission, or budget-change action is imported here.
Already spent GPU seconds must come from platform allocation evidence supplied
by the caller; preflight worker_seconds is never used as allocated GPU time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
PREFLIGHT_FORMAT = 'stage09_h1_recovery_preflight_v1'
SAFETY_FACTOR = 1.25
R_UPDATES = 4158
FULL_VALIDATION_ROWS = 71657
SUBSET_ROWS = 8192
EVAL_BATCH = 32
WORKLOAD = {
    'decay': dict(r_updates=R_UPDATES, t_updates=80, full_validations=2,
                  extra_subset_evaluations=7, catalog_diagnostics=9),
    'freeze': dict(r_updates=R_UPDATES, t_updates=0, full_validations=2,
                   extra_subset_evaluations=5, catalog_diagnostics=1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, zero=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero else value > 0),
            f'{name} must be a finite {"nonnegative" if zero else "positive"} number')
    return float(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _digest(value):
    return isinstance(value, str) and bool(re.fullmatch('[0-9a-f]{64}', value))


def _timing(row, label):
    seconds = number(row.get('seconds'), label + '/seconds')
    prepare = number(row.get('data_prepare_seconds'), label + '/data_prepare_seconds', zero=True)
    total = number(row.get('end_to_end_seconds'), label + '/end_to_end_seconds')
    require(math.isclose(total, seconds + prepare, rel_tol=1e-8, abs_tol=1e-8),
            label + ': end-to-end time must equal measured update/generation plus preparation')
    return total


def validate_report(report, policy, *, allow_synthetic=False):
    require(report.get('format') == PREFLIGHT_FORMAT and report.get('policy') == policy,
            f'{policy}: unsupported report format/policy')
    require(report.get('status') == 'passed', f'{policy}: preflight status is not passed')
    require(not report.get('synthetic_fixture') or allow_synthetic,
            f'{policy}: synthetic fixture cannot produce a production budget')
    require(report.get('preflight_only') is True and report.get('formal_training_started') is False,
            f'{policy}: acceptance was not a disposable preflight')
    require(report.get('finished_at_parent_state') is True
            and report.get('formal_resume_from_preflight_forbidden') is True,
            f'{policy}: updates were not verified discarded')
    require(report.get('parent_sha256') == PARENT_SHA256, f'{policy}: wrong R20 parent SHA')
    require(report.get('visible_gpu_count') == 1 and 'h100' in str(report.get('gpu', '')).lower(),
            f'{policy}: timing must come from a single visible H100')
    initial = report.get('initial', {})
    require(initial.get('r_updates') == 13860 and initial.get('t_updates') == 200,
            f'{policy}: starting optimizer clocks are not R20/T200')
    require(initial.get('shared_model_and_alignment_exact') is True
            and initial.get('optimizer_moments_exact') is True,
            f'{policy}: parent restoration acceptance is missing')
    parent_hashes, fork_hashes = initial.get('parent_hashes', {}), initial.get('fork_hashes', {})
    keys = {'model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler'}
    require(set(parent_hashes) == keys and set(fork_hashes) == keys
            and all(_digest(value) for value in (*parent_hashes.values(), *fork_hashes.values())),
            f'{policy}: parent/fork component hashes are incomplete')
    for key in keys - {'t_optimizer'}:
        require(parent_hashes[key] == fork_hashes[key], f'{policy}: fork changed {key}')
    if policy == 'freeze':
        require(parent_hashes['t_optimizer'] == fork_hashes['t_optimizer'], 'Freeze changed the T optimizer')
    require(_digest(initial.get('hard_sid_sha256')), f'{policy}: hard-SID identity is missing')
    binding = report.get('source_binding', {})
    require(set(binding) == {'config', 'asset_manifest', 'data', 'code', 'engine'},
            f'{policy}: full source/data/asset binding is missing')
    require(_digest(binding['asset_manifest']) and isinstance(binding['data'], dict) and binding['data'],
            f'{policy}: asset/data identity is incomplete')
    for category in ('code', 'engine'):
        require(isinstance(binding[category], dict) and len(binding[category]) >= 20
                and all(_digest(value) for value in binding[category].values()),
                f'{policy}: bound {category} source hashes are incomplete')
    cfg = binding['config']
    expected = {'architecture': 'H', 'seed': 2024, 'list_length': 5, 'history_max': 50}
    require(all(cfg.get(key) == value for key, value in expected.items()), f'{policy}: H protocol changed')
    expected_model = dict(n_digit=4, codebook_size=256, n_embd=256, n_head=4,
                          encoder_n_layer=1, decoder_n_layer=4, n_inner=1024)
    require(all(cfg.get('model', {}).get(key) == value for key, value in expected_model.items()),
            f'{policy}: model shape differs from the profiled protocol')
    expected_training = dict(effective_batch_size=1024, microbatch_size=256, warmup_epochs=5,
                             cycles=10, r_epochs_per_cycle=3, t_updates_per_cycle=40,
                             deterministic_algorithms=True)
    require(all(cfg.get('training', {}).get(key) == value for key, value in expected_training.items()),
            f'{policy}: training protocol differs')
    expected_eval = dict(batch_size=32, val_beam=32, test_beam=32, candidate_lists=32, selection='ndcg@5')
    require(all(cfg.get('evaluation', {}).get(key) == value for key, value in expected_eval.items()),
            f'{policy}: beam/evaluation protocol differs')
    backend = report.get('backend', {})
    for key, value in dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
                           cudnn_deterministic=True, cudnn_benchmark=False,
                           cuda_matmul_tf32=False, cudnn_tf32=False).items():
        require(backend.get(key) == value, f'{policy}: backend setting {key} differs')
    profile = report.get('profile', {})
    require(set(profile) == ({'R', 'T'} if policy == 'decay' else {'R'}),
            f'{policy}: required phase timings are missing or frozen T was executed')
    for kind, section in profile.items():
        records = section.get('records', [])
        require(isinstance(records, list) and len(records) >= 3
                and len(records) == report.get('profile_updates'), f'{policy}/{kind}: at least three timing records required')
        for index, row in enumerate(records):
            label = f'{policy}/{kind}/{index + 1}'
            require(row.get('iteration') == index + 1 and row.get('effective_lists') == 1024
                    and row.get('microbatch') == 256 and row.get('accumulation') == 4,
                    label + ': update batch size or iteration differs')
            require(row.get('original_phase_index') == (25 if kind == 'T' else 26),
                    label + ': original phase numbering was lost')
            require(row.get('frozen_module_and_optimizer_unchanged') is True
                    and row.get('hard_sids_unchanged') is True, label + ': parameter/SID isolation gate missing')
            values = row.get('values', {})
            require(values.get('lists') == 1024 and values.get('phase') == kind,
                    label + ': measured update does not match its phase/batch')
            require(values.get('r_updates') == 13860 + (index + 1 if kind == 'R' else 0)
                    and values.get('t_updates') == 200 + (index + 1 if kind == 'T' else 0),
                    label + ': timing is not from the declared disposable R20 fork')
            _timing(row, label)
        replay = section.get('replay', {})
        require(replay.get('phase') == kind and replay.get('preflight_only') is True
                and replay.get('exact_loaded_state') is True and replay.get('exact_next_update') is True
                and replay.get('formal_restore_rejection'), f'{policy}/{kind}: exact restoration gate missing')
    decode = report.get('decode_profile', {})
    require(decode.get('beam') == 32 and decode.get('precision') == 'fp32'
            and decode.get('reset_R20_output_exact') is True,
            f'{policy}: FP32 beam32 restored decoding was not accepted')
    rows = decode.get('records', [])
    examples = report.get('validation_samples')
    require(examples in (128, 256) and decode.get('examples') == examples and len(rows) == examples // 32,
            f'{policy}: fixed validation timing coverage is incomplete')
    seen = []
    for index, row in enumerate(rows):
        require(row.get('batch_size') == 32 and row.get('beam') == 32 and row.get('precision') == 'fp32',
                f'{policy}: decode timing batch/beam/precision mismatch')
        ids = row.get('sample_ids', [])
        require(len(ids) == 32 and all(isinstance(item, str) and item for item in ids),
                f'{policy}: timed validation identities are missing')
        seen.extend(ids)
        _timing(row, f'{policy}/decode/{index + 1}')
    require(len(set(seen)) == examples, f'{policy}: repeated validation requests bias the timing')
    require(_digest(decode.get('first_batch_output_sha256')), f'{policy}: output reset signature missing')
    number(decode.get('catalog_timing', {}).get('seconds'), f'{policy}/catalog build')
    subset = report.get('validation_subset', {})
    require(subset.get('population_count') == FULL_VALIDATION_ROWS and subset.get('count') == examples
            and subset.get('split') == 'validation'
            and subset.get('selection') == 'sha256_utf8_sample_id_then_sample_id'
            and _digest(subset.get('rows_sha256')), f'{policy}: fixed label-blind validation subset binding missing')
    catalog = report.get('catalog_diagnostic', {})
    require(catalog.get('full_state_unchanged') is True and catalog.get('hard_sids_match_parent') is True
            and catalog.get('metrics', {}).get('num_items') == 10728,
            f'{policy}: full-catalog diagnostic acceptance missing')
    number(catalog.get('seconds'), f'{policy}/catalog diagnostic')
    return cfg


def validate_pair(decay, freeze, *, allow_synthetic=False):
    for policy, report in (('decay', decay), ('freeze', freeze)):
        validate_report(report, policy, allow_synthetic=allow_synthetic)
    for key in ('parent_sha256', 'source_binding', 'backend', 'gpu', 'torch_version', 'cuda_version', 'validation_subset'):
        require(decay.get(key) == freeze.get(key), f'Paired {key} identity differs')
    for key in ('parent_hashes', 'hard_sid_sha256'):
        require(decay['initial'][key] == freeze['initial'][key], f'Paired initial {key} differs')
    require(decay['decode_profile']['first_batch_output_sha256'] == freeze['decode_profile']['first_batch_output_sha256'],
            'The same restored R20 model produced different paired initial outputs')
    if 'protocol' in decay or 'protocol' in freeze:
        require('protocol' in decay and 'protocol' in freeze, 'Only one report has explicit recovery protocol metadata')
        protocols = []
        for policy, report in (('decay', decay), ('freeze', freeze)):
            protocol = dict(report['protocol'])
            require(protocol.pop('policy', None) == policy, 'Explicit protocol policy differs')
            require(protocol.get('parent_sha256') == PARENT_SHA256
                    and protocol.get('start_phase') == 25 and protocol.get('stop_phase') == 33,
                    'Explicit protocol parent/phase range differs')
            protocols.append(protocol)
        require(protocols[0] == protocols[1], 'Explicit recovery protocols differ beyond policy')
        protocol_mode = 'explicit_recovery_protocol_and_identical_full_source_binding'
    else:
        protocol_mode = 'identical_full_source_binding_and_locked_H_R20_configuration'
    return dict(parent_sha256=PARENT_SHA256, source_binding_sha256=json_sha256(decay['source_binding']),
                protocol_validation=protocol_mode, workload=WORKLOAD)


def representative(rows, statistic, label):
    require(statistic in ('max', 'median'), 'statistic must be max or median')
    values = [_timing(row, f'{label}/{index + 1}') for index, row in enumerate(rows)]
    require(len(values) >= 3, label + ': need at least two timed updates after discarding the first')
    retained = values[1:]
    typical = max(retained) if statistic == 'max' else statistics.median(retained)
    return dict(seconds=typical, statistic=statistic + '_after_first', measured_seconds=values,
                excluded_first_seconds=values[0], retained_count=len(retained),
                first_step_extra_seconds=max(0.0, values[0] - typical),
                field='end_to_end_seconds')


def startup_from_events(path, policy):
    path = Path(path)
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    required = ('authenticated_R20_fork', 'load_fixed_validation_subset')
    parts = {}
    for stage in required:
        found = [row for row in events if row.get('event') == 'stage_pass' and row.get('stage') == stage]
        require(len(found) == 1, f'{policy}: require exactly one completed startup stage {stage}')
        parts[stage] = number(found[0].get('seconds'), f'{policy}/startup/{stage}', zero=True)
    complete = [row for row in events if row.get('event') == 'preflight_complete' and row.get('policy') == policy]
    require(len(complete) == 1, f'{policy}: events are not from one completed matching preflight')
    return dict(seconds=sum(parts.values()), basis='measured_startup_stage_pass_sum',
                evidence=str(path.resolve()), evidence_sha256=sha256(path), components=parts)


def estimate(decay, freeze, *, spent_gpu_seconds, price_per_gpu_hour, spent_evidence,
             price_evidence, startups, statistic='max', allow_synthetic=False):
    identity = validate_pair(decay, freeze, allow_synthetic=allow_synthetic)
    spent = number(spent_gpu_seconds, 'platform already-spent GPU seconds', zero=True)
    price = number(price_per_gpu_hour, 'current price per GPU hour', zero=True)
    require(isinstance(spent_evidence, str) and spent_evidence.strip(), 'Platform usage evidence reference is required')
    require(isinstance(price_evidence, str) and price_evidence.strip(), 'Current price evidence reference is required')
    require(set(startups) == {'decay', 'freeze'}, 'Explicit measured or provided startup overhead is required for both arms')
    arms = {}
    for policy, report in (('decay', decay), ('freeze', freeze)):
        startup = startups[policy]
        startup_seconds = number(startup.get('seconds'), policy + '/startup_seconds', zero=True)
        require(startup.get('basis') and startup.get('evidence'), policy + ': startup basis/evidence must be explicit')
        r = representative(report['profile']['R']['records'], statistic, policy + '/R')
        t = representative(report['profile']['T']['records'], statistic, policy + '/T') if policy == 'decay' else None
        decode = representative(report['decode_profile']['records'], statistic, policy + '/decode')
        catalog_build = number(report['decode_profile']['catalog_timing']['seconds'], policy + '/catalog_build')
        catalog_diagnostic = number(report['catalog_diagnostic']['seconds'], policy + '/catalog_diagnostic')
        work = WORKLOAD[policy]
        full_batches, subset_batches = math.ceil(FULL_VALIDATION_ROWS / EVAL_BATCH), SUBSET_ROWS // EVAL_BATCH
        # Each evaluation creates a new SIDCatalog. Charge the observed first
        # batch excess once per call, and round the final short batch upward.
        full_each = full_batches * decode['seconds'] + decode['first_step_extra_seconds'] + catalog_build
        subset_each = subset_batches * decode['seconds'] + decode['first_step_extra_seconds'] + catalog_build
        training_first_extra = r['first_step_extra_seconds'] + (t['first_step_extra_seconds'] if t else 0.0)
        components = dict(startup=startup_seconds,
            recommender_updates=work['r_updates'] * r['seconds'],
            tokenizer_updates=work['t_updates'] * t['seconds'] if t else 0.0,
            training_first_step_excess=training_first_extra,
            full_validation=work['full_validations'] * full_each,
            extra_subset_evaluation=work['extra_subset_evaluations'] * subset_each,
            catalog_diagnostics=work['catalog_diagnostics'] * catalog_diagnostic)
        total = sum(components.values())
        arms[policy] = dict(workload=dict(work, full_validation_rows=FULL_VALIDATION_ROWS, subset_rows=SUBSET_ROWS,
            full_validation_batches_each=full_batches, subset_batches_each=subset_batches,
            evaluation_calls=work['full_validations'] + work['extra_subset_evaluations']),
            measured_rates=dict(R=r, T=t, decode_batch=decode, catalog_build_seconds=catalog_build,
                                catalog_diagnostic_seconds=catalog_diagnostic),
            startup=startup, seconds_by_component=components, remaining_gpu_seconds_raw=total,
            remaining_gpu_hours_raw=total / 3600,
            remaining_gpu_seconds_with_margin=total * SAFETY_FACTOR,
            remaining_gpu_hours_with_margin=total * SAFETY_FACTOR / 3600,
            single_gpu_wall_seconds_with_margin=total * SAFETY_FACTOR)
    remaining = sum(arm['remaining_gpu_seconds_raw'] for arm in arms.values())
    remaining_margin = remaining * SAFETY_FACTOR
    parallel_wall = max(arm['remaining_gpu_seconds_raw'] for arm in arms.values()) * SAFETY_FACTOR
    total_gpu_hours = (spent + remaining_margin) / 3600
    shared_allocation = 2 * parallel_wall
    shared_total_gpu_hours = (spent + shared_allocation) / 3600
    synthetic = bool(decay.get('synthetic_fixture') or freeze.get('synthetic_fixture'))
    return dict(format='stage09_h1_recovery_budget_v1',
        status='synthetic_estimate_not_for_authorization' if synthetic else 'estimate_only_no_submission',
        synthetic_fixture=synthetic, source_identity=identity, arms=arms,
        assumptions=dict(statistic=statistic + '_after_first', safety_factor=SAFETY_FACTOR,
            evaluation_batch_size=EVAL_BATCH, beam=32, precision='fp32',
            full_validation_tail='charge a full batch for the final 9 samples',
            subset_reuse='R23/R26 subset metrics extracted from full predictions; no duplicate inference',
            first_step_excess='positive observed first-step excess charged once per R/T phase type and once per evaluation call',
            parallel_execution='two independent GPUs, one arm per GPU, both start together',
            primary_gpu_hours='sum of arm times assumes each GPU is released when its arm completes',
            shared_two_gpu_job='if both GPUs stay allocated until the slower arm completes, use the separately reported two-times-max allocation estimate',
            unmeasured_overhead='metric computation, gzip/checkpoint I/O, SID refresh, interpreter/context startup and release delay are not separately timed; covered only by the 25% contingency'),
        platform_usage=dict(already_spent_gpu_seconds=spent, evidence=spent_evidence,
            source='caller_supplied_platform_allocation_evidence', preflight_worker_seconds_used=False),
        totals=dict(remaining_gpu_seconds_raw=remaining, remaining_gpu_hours_raw=remaining / 3600,
            remaining_gpu_seconds_with_margin=remaining_margin, remaining_gpu_hours_with_margin=remaining_margin / 3600,
            already_spent_gpu_hours=spent / 3600,
            total_gpu_hours_including_spent=total_gpu_hours,
            two_gpu_parallel_remaining_wall_seconds=parallel_wall,
            two_gpu_parallel_remaining_wall_hours=parallel_wall / 3600,
            shared_two_gpu_job_remaining_allocated_gpu_seconds=shared_allocation,
            shared_two_gpu_job_remaining_allocated_gpu_hours=shared_allocation / 3600,
            shared_two_gpu_job_idle_tail_gpu_seconds=shared_allocation - remaining_margin,
            shared_two_gpu_job_total_gpu_hours_including_spent=shared_total_gpu_hours),
        price=dict(per_gpu_hour=price, unit='points', evidence=price_evidence,
            already_spent_points_estimate=spent / 3600 * price,
            remaining_points_estimate=remaining_margin / 3600 * price,
            total_points_estimate=total_gpu_hours * price, settlement_verified=False,
            shared_two_gpu_job_total_points_estimate=shared_total_gpu_hours * price,
            note='Current supplied rate applied as an estimate; actual platform settlement is unverified.'),
        submission_performed=False, budget_limit_changed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decay-report', required=True)
    parser.add_argument('--freeze-report', required=True)
    parser.add_argument('--decay-events')
    parser.add_argument('--freeze-events')
    parser.add_argument('--decay-startup-seconds', type=float)
    parser.add_argument('--freeze-startup-seconds', type=float)
    parser.add_argument('--decay-startup-evidence')
    parser.add_argument('--freeze-startup-evidence')
    parser.add_argument('--spent-gpu-seconds', required=True, type=float)
    parser.add_argument('--spent-evidence', required=True)
    parser.add_argument('--price-per-gpu-hour', required=True, type=float)
    parser.add_argument('--price-evidence', required=True)
    parser.add_argument('--statistic', choices=('max', 'median'), default='max')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    reports, sources, startups = {}, {}, {}
    for policy in ('decay', 'freeze'):
        path = Path(getattr(args, policy + '_report'))
        reports[policy] = json.loads(path.read_text(encoding='utf-8'))
        sources[policy] = dict(path=str(path.resolve()), sha256=sha256(path))
        override = getattr(args, policy + '_startup_seconds')
        if override is None:
            events = getattr(args, policy + '_events') or path.with_name('events.jsonl')
            startups[policy] = startup_from_events(events, policy)
        else:
            startups[policy] = dict(seconds=override, basis='explicit_caller_provided_startup_overhead',
                evidence=getattr(args, policy + '_startup_evidence'))
    result = estimate(reports['decay'], reports['freeze'], spent_gpu_seconds=args.spent_gpu_seconds,
        price_per_gpu_hour=args.price_per_gpu_hour, spent_evidence=args.spent_evidence,
        price_evidence=args.price_evidence, startups=startups, statistic=args.statistic)
    result.update(input_reports=sources, created_utc=datetime.now(timezone.utc).isoformat())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(dict(status=result['status'], totals=result['totals'], price=result['price'],
                          output=str(output.resolve())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
