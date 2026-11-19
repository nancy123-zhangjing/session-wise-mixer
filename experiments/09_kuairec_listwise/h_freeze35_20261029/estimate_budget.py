"""Offline, stdlib-only freeze35 budget arithmetic; no submission or authorization.

Accepted GPU preflight timings are measured inputs. Missing host/I/O overheads
must be supplied as explicit measured ranges or labelled operator allowances.
Worker/supervisor duration is never substituted for actual platform GPU usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


FORMAT = 'stage09_freeze35_budget_v1'
SAFETY_FACTOR = 1.25
PARENTS = {
    'A': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
    'B': '8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e',
    'C': 'b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166',
}
STARTS = {'A': (33, 18018, 200), 'B': (41, 22176, 360), 'C': (5, 3465, 0)}
EVALUATIONS = {'A': (3, 2), 'B': (1, 1), 'C': (10, 2)}
EXTRA_KEYS = (
    'package_verification_platform_startup_release_per_job',
    'phase_indices_logging_per_phase',
    'evaluation_output_and_unprofiled_cpu_per_call',
    'checkpoint_coverage_validation_per_file',
    'c_replay_trajectory_and_prediction_audit_total',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), name + ': finite nonnegative seconds/value required')
    return float(value)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def workload(branch):
    # Original plan: five R phases, then ten unchanged [T,R,R,R] cycles.
    plan = [('R', 693)] * 5 + [('T', 40), ('R', 693), ('R', 693), ('R', 693)] * 10
    phase_start = STARTS[branch][0]
    rows = []
    for phase in range(phase_start, 45):
        kind, original_updates = plan[phase]
        active_updates = original_updates if kind == 'R' or (branch == 'C' and phase < 25) else 0
        rows.append(dict(phase=phase, kind=kind, optimizer_updates=active_updates,
                         periodic_last_writes=active_updates // 100,
                         boundary_last_writes=3, immutable_phase_writes=1))
    full_val, max_test = EVALUATIONS[branch]
    return dict(r_updates=sum(row['optimizer_updates'] for row in rows if row['kind'] == 'R'),
        t_updates=sum(row['optimizer_updates'] for row in rows if row['kind'] == 'T'),
        r_epochs=sum(row['kind'] == 'R' for row in rows), phases=len(rows), phase_rows=rows,
        full_validations=full_val, maximum_new_tests=max_test,
        validation_batches=math.ceil(71657 / 32), test_batches=math.ceil(53194 / 32),
        initial_and_phase_slots=1 + len(rows), r35_boundary_writes=1, r20_pre_freeze_writes=int(branch == 'C'),
        immutable_writes=2 + len(rows) + int(branch == 'C'),
        rolling_writes=sum(row['periodic_last_writes'] + 3 for row in rows) + 1 + int(branch == 'C'),
        covered_checkpoint_reads=1 + len(rows) + int(branch == 'C'),
        model_builds=1 + max_test, selected_test_checkpoint_loads=max_test,
        active_t_catalog_refreshes=5 if branch == 'C' else 0)


def range_statistics(values, *, warm=False):
    values = [number(value, 'timing', positive=True) for value in values]
    require(values, 'No timing samples')
    retained = values[1:] if warm else values
    require(retained, 'At least two records required to distinguish first and warm timing')
    median, maximum = statistics.median(retained), max(retained)
    return dict(low=median, high=maximum, samples=values, retained_samples=len(retained),
                statistic='median/max after first' if warm else 'median/max all measured samples',
                first_extra_low=max(0, values[0] - median) if warm else 0,
                first_extra_high=max(0, values[0] - maximum) if warm else 0)


def record_timings(rows, label, *, decode=False):
    require(len(rows) >= (2 if decode else 3), label + ': insufficient real records')
    values = []
    for row in rows:
        compute = number(row.get('seconds'), label + '/seconds', positive=True)
        prepare = number(row.get('data_prepare_seconds'), label + '/data_prepare_seconds')
        if 'end_to_end_seconds' in row:
            require(math.isclose(row['end_to_end_seconds'], compute + prepare, rel_tol=1e-8, abs_tol=1e-8),
                    label + ': end-to-end timing does not equal compute plus data preparation')
        if decode:
            require((row.get('batch_size'), row.get('beam'), row.get('inference_precision')) == (32, 32, 'fp32'),
                    label + ': require FP32 beam32 batch32')
        else:
            require((row.get('effective_lists'), row.get('microbatch'), row.get('accumulation')) == (1024, 256, 4),
                    label + ': require a real effective1024 update')
        values.append(compute + prepare)
    return range_statistics(values, warm=True)


def validate(report, supervisor, spent, *, allow_synthetic=False):
    require(report.get('format') == 'stage09_recovery35_preflight_v1' and report.get('status') == 'passed',
            'Need a passed recovery35 GPU preflight report')
    require(not report.get('synthetic_fixture') or allow_synthetic, 'Synthetic fixture cannot estimate an actual budget')
    require(report.get('preflight_only') is True and report.get('formal_training_started') is False,
            'Preflight identity/isolated status missing')
    require(supervisor.get('status') == 'passed' and supervisor.get('preflight_only') is True
            and supervisor.get('returncode') == 0 and supervisor.get('reason') is None, 'Supervisor did not pass')
    supervised = number(supervisor.get('allocated_gpu_seconds_during_script'), 'supervised allocated seconds', positive=True)
    require(spent * 3600 + 1e-6 >= supervised, 'Actual cumulative platform GPU time is below supervised script time')
    gpu = report.get('gpu', {})
    require(gpu.get('visible_gpu_count') == 1 and 'h100' in gpu.get('name', '').lower()
            and gpu.get('mig_mode') == 'disabled' and gpu.get('total_memory_bytes', 0) >= 75_000_000_000,
            'Need timings from one full physical H100')
    require(set(report.get('branches', {})) == set('ABC'), 'Missing A/B/C accepted profiles')
    shared = None
    for branch, row in report['branches'].items():
        require(row.get('format') == 'stage09_recovery35_branch_preflight_v1' and row.get('branch') == branch
                and row.get('status') == 'passed' and row.get('parent_sha256') == PARENTS[branch], branch + ': wrong source/status')
        require(row.get('preflight_only') is True and row.get('formal_training_started') is False
                and row.get('finished_at_exact_parent_state') is True
                and row.get('formal_resume_from_preflight_forbidden') is True, branch + ': updates were not discarded')
        phase, r, t = STARTS[branch]
        require(row.get('initial_counts') == dict(R=r, T=t, phase=phase), branch + ': wrong parent clocks')
        binding = row['source_binding']
        identity = {key: binding[key] for key in ('config', 'data', 'asset_manifest', 'code', 'engine')}
        if shared is not None:
            require(shared == identity, 'A/B/C scientific/source/input bindings differ')
        shared = identity
        cfg = binding['config']
        require(cfg['architecture'] == 'H' and cfg['seed'] == 2024 and cfg['list_length'] == 5, branch + ': protocol differs')
        require(cfg['training']['effective_batch_size'] == 1024 and cfg['training']['microbatch_size'] == 256
                and cfg['training']['checkpoint_every_updates'] == 100, branch + ': batch/checkpoint interval differs')
        require(cfg['evaluation']['batch_size'] == 32 and cfg['evaluation']['val_beam'] == 32
                and cfg['evaluation']['test_beam'] == 32, branch + ': evaluation protocol differs')
        training = row['training_profile']
        require(('T' in training) == (branch == 'C'), branch + ': frozen T profile or missing C profile')
        if branch == 'C':
            require(len(training['T']['records']) == 40, 'C must profile its complete original T40 block')
        for kind in ('R', 'T') if branch == 'C' else ('R',):
            replay = training[kind + '_disk_replay']
            require(all(replay.get(key) is True for key in ('exact_loaded_state', 'exact_next_update',
                    'reusable_snapshot_unchanged', 'formal_restore_rejected')), branch + '/' + kind + ': replay acceptance missing')
        decode = row['decode_profile']
        require(decode.get('exact_output_repeat') is True
                and decode.get('full_training_state_rng_modes_gradients_unchanged') is True
                and decode.get('quality_metric') is False, branch + ': decode preservation gate missing')
    return supervised


def allowance(value, name):
    require(isinstance(value, dict), name + ': provide an explicit seconds range')
    low, high = number(value.get('low'), name + '/low'), number(value.get('high'), name + '/high')
    require(low <= high, name + ': reversed range')
    require(value.get('basis') in ('measured', 'operator_allowance') and str(value.get('evidence', '')).strip(),
            name + ': measured/operator_allowance basis and evidence required; zero also needs justification')
    return dict(low=low, high=high, basis=value['basis'], evidence=value['evidence'])


def estimate(report, supervisor, extras, *, spent_gpu_hours, spent_evidence, allow_synthetic=False):
    spent = number(spent_gpu_hours, 'actual cumulative platform GPU hours')
    require(isinstance(spent_evidence, str) and spent_evidence.strip(), 'Actual usage evidence must include failed attempts/startup/release')
    supervised = validate(report, supervisor, spent, allow_synthetic=allow_synthetic)
    require(extras.get('format') == 'stage09_freeze35_budget_overheads_v1', 'Unknown overhead template')
    overheads = {key: allowance(extras.get(key), key) for key in EXTRA_KEYS}
    arm_extra = {branch: allowance(extras.get('extra_gpu_allocated_cpu_audit', {}).get(branch),
                                  branch + '/extra_gpu_allocated_cpu_audit') for branch in 'ABC'}
    offline = extras.get('offline_cpu_audit_total')
    offline = allowance(offline, 'offline_cpu_audit_total') if offline and offline.get('low') is not None else None
    arms = {}
    for branch in 'ABC':
        row, counts = report['branches'][branch], workload(branch)
        training = row['training_profile']
        r = record_timings(training['R']['records'], branch + '/R')
        t = record_timings(training['T']['records'], branch + '/T') if branch == 'C' else None
        decode = record_timings(row['decode_profile']['records'], branch + '/decode', decode=True)
        saves = [row['initial_checkpoint'], training['R_disk_replay']['save']]
        loads = [training['R_disk_replay']['load']]
        restores = [training['R_disk_replay']['restore']]
        if branch == 'C':
            saves += [training['T_disk_replay']['save'], training['T40_boundary_checkpoint']]
            loads += [training['T_disk_replay']['load']]
            restores += [training['T_disk_replay']['restore']]
        for saved in saves:
            require(saved.get('fsync_completed') is True, branch + ': checkpoint write did not include fsync')
        io = {key: range_statistics([saved[key] for saved in saves]) for key in
              ('seconds', 'state_materialization_seconds', 'payload_fingerprint_seconds', 'file_sha256_seconds')}
        io['read_seconds'] = range_statistics([value['seconds'] for value in loads])
        io['restore_seconds'] = range_statistics([value['seconds'] for value in restores])
        components = {}

        def add(name, timing, units=1, *, first_extras=0, evidence=None):
            components[name] = dict(low=timing['low'] * units + timing.get('first_extra_low', 0) * first_extras,
                high=timing['high'] * units + timing.get('first_extra_high', 0) * first_extras,
                units=units, unit_seconds=timing, evidence=evidence)

        def single(value, label):
            value = number(value, label, positive=True)
            return dict(low=value, high=value, statistic='single measured timing; no repeated-startup distribution')

        all_evals = counts['full_validations'] + counts['maximum_new_tests']
        all_writes = counts['rolling_writes'] + counts['immutable_writes']
        add('R_compute_and_data', r, counts['r_updates'], first_extras=1)
        if t:
            add('T_compute_and_data', t, counts['t_updates'], first_extras=1)
            add('T_catalog_refresh', single(training['T40_refresh']['seconds'], 'T40 refresh'), 5)
        add('full_validation_decode_and_data', decode, counts['full_validations'] * 2240,
            first_extras=counts['full_validations'])
        add('maximum_test_decode_and_data', decode, counts['maximum_new_tests'] * 1663,
            first_extras=counts['maximum_new_tests'])
        add('evaluation_catalog_build', single(row['decode_profile']['catalog_timing']['seconds'], 'catalog'), all_evals)
        add('model_fork_data_hash_load_startup', single(row['build_timing']['seconds'], 'model build'), counts['model_builds'])
        add('checkpoint_write_and_fsync', io['seconds'], all_writes,
            evidence='fsync-profiled disposable bundle used conservatively also for rolling writes')
        add('checkpoint_state_materialization', io['state_materialization_seconds'], all_writes)
        add('checkpoint_payload_fingerprint', io['payload_fingerprint_seconds'],
            counts['immutable_writes'] + counts['covered_checkpoint_reads'])
        add('checkpoint_file_sha256', io['file_sha256_seconds'],
            counts['immutable_writes'] + counts['covered_checkpoint_reads'] + counts['maximum_new_tests'] + 1)
        add('checkpoint_readback', io['read_seconds'], counts['covered_checkpoint_reads'] + counts['selected_test_checkpoint_loads'])
        add('selected_test_checkpoint_restore', io['restore_seconds'], counts['selected_test_checkpoint_loads'])
        add('package_verification_platform_startup_release', overheads[EXTRA_KEYS[0]])
        add('phase_indices_logging_cpu', overheads[EXTRA_KEYS[1]], counts['phases'])
        add('evaluation_output_and_other_cpu', overheads[EXTRA_KEYS[2]], all_evals)
        add('checkpoint_coverage_validation_cpu', overheads[EXTRA_KEYS[3]], counts['covered_checkpoint_reads'])
        if branch == 'C':
            add('C_replay_trajectory_prediction_cpu', overheads[EXTRA_KEYS[4]])
        add('extra_gpu_allocated_cpu_audit', arm_extra[branch])
        total = {key: sum(value[key] for value in components.values()) for key in ('low', 'high')}
        checkpoint_bytes = max(number(saved['bytes'], 'checkpoint bytes', positive=True) for saved in saves)
        arms[branch] = dict(workload=counts, measured_profile=dict(R=r, T=t, decode=decode, checkpoint=io),
            components_seconds=components, unbuffered_wall_seconds=total,
            buffered_wall_seconds={key: value * SAFETY_FACTOR for key, value in total.items()},
            buffered_gpu_hours={key: value * SAFETY_FACTOR / 3600 for key, value in total.items()},
            suggested_job_authorized_seconds=math.ceil(total['high'] * SAFETY_FACTOR),
            checkpoint_write_traffic_bytes=checkpoint_bytes * all_writes,
            checkpoint_retained_capacity_bytes=checkpoint_bytes * (counts['immutable_writes'] + 1),
            checkpoint_bytes_basis='maximum observed bundle bytes; write traffic is not extra retained capacity')
    remaining = {key: sum(arm['buffered_gpu_hours'][key] for arm in arms.values()) for key in ('low', 'high')}
    cumulative = {key: spent + remaining[key] for key in ('low', 'high')}
    return dict(format=FORMAT, status='estimate_for_review_no_authorization', synthetic_fixture=bool(report.get('synthetic_fixture')),
        statistic_policy='Warm median is the central scenario; warm maximum is the conservative scenario. '
            'First-step excess is added once per training process and once per complete evaluation. '
            'All observed checkpoint I/O samples are retained. Conservative total receives an additional 25% margin.',
        safety_factor=SAFETY_FACTOR, arms=arms,
        actual_spent=dict(gpu_hours=spent, evidence=spent_evidence, supervised_script_gpu_hours_lower_bound=supervised / 3600,
            worker_seconds_used_as_platform_accounting=False),
        remaining_gpu_hours=remaining, cumulative_gpu_hours=cumulative,
        three_card_parallel_wall_hours={key: max(arm['buffered_gpu_hours'][key] for arm in arms.values()) for key in ('low', 'high')},
        suggested_cumulative_budget_integer_gpu_hours=math.ceil(cumulative['high']),
        suggested_additional_budget_integer_gpu_hours=math.ceil(remaining['high']),
        price=dict(points_per_gpu_hour=1, individual_settlement_verified=False,
                   cumulative_estimated_points=cumulative, basis='User-specified 1 point/GPU hour; not an invoice'),
        offline_cpu_audit_seconds_excluded_from_gpu=offline,
        excluded_work='Post-release CPU analysis/archive audit is outside GPU time. Any such work executed while a GPU '
            'remains allocated must instead be included in extra_gpu_allocated_cpu_audit. No retries or failed formal attempts forecast.',
        totals=dict(full_validations=14, maximum_tests=5, r_updates=29106, t_updates=200,
                    initial_and_phase_slots=59, immutable_writes=63, rolling_writes=424,
                    checkpoint_covered_reads=60, full_validation_batches=31360, maximum_test_batches=8315),
        automatic_submission=False, authorization_granted=False)


def templates(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    blank = lambda text: dict(low=None, high=None, basis=None, evidence=text)
    value = dict(format='stage09_freeze35_budget_overheads_v1', units='seconds',
        instructions='Fill every GPU-allocated range with measured or operator_allowance basis; null is not zero. '
        'An operator allowance is an explicit conservative assumption, never an observed timing.')
    descriptions = (
        'Per job: package verification, historical-reference authentication, process imports and platform startup/release. Exclude model build_timing already counted.',
        'Per phase: index generation, training JSONL/console logging and remaining host work. Exclude checkpoint I/O and C replay audits.',
        'Per full validation/test: dataset load, metrics, prediction/gzip output, full-state preservation checks and SHA receipts. Exclude profiled batch preparation/decode/catalog.',
        'Per coverage checkpoint: validate_clocks and receipt processing. Exclude separately profiled read/hash/fingerprint.',
        'C total: 20 trajectory audits, 5 replay prediction checks plus R20 recheck/gate. Does not include inference, checkpoint I/O, or offline final bootstrap.',
    )
    value.update({key: blank(text) for key, text in zip(EXTRA_KEYS, descriptions)})
    value['extra_gpu_allocated_cpu_audit'] = {branch: blank('Additional actual-restore/CPU audit while this GPU remains allocated; justify zero if none.') for branch in 'ABC'}
    value['offline_cpu_audit_total'] = blank('Optional post-GPU-release CPU-only work estimate; excluded from GPU budget.')
    write_new(directory / 'BUDGET_INPUT_TEMPLATE.json', value)
    write_new(directory / 'BUDGET_RESULT_TEMPLATE.json', dict(format=FORMAT, status='template_not_measured',
        actual_spent_gpu_hours=None, measured_preflight_report_sha256=None, supervisor_sha256=None,
        branches={branch: dict(workload=workload(branch), wall_seconds=None, gpu_hours=None) for branch in 'ABC'},
        cumulative_gpu_hours=None, suggested_integer_budget_gpu_hours=None,
        price_points_per_gpu_hour=1, individual_settlement_verified=False, authorization_granted=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    template = commands.add_parser('template')
    template.add_argument('--directory', required=True)
    command = commands.add_parser('estimate')
    for name in ('report', 'supervisor', 'overheads', 'spent-evidence', 'output'):
        command.add_argument('--' + name, required=True)
    command.add_argument('--spent-gpu-hours', required=True, type=float)
    args = parser.parse_args(argv)
    if args.command == 'template':
        templates(args.directory)
        return
    result = estimate(read(args.report), read(args.supervisor), read(args.overheads),
                      spent_gpu_hours=args.spent_gpu_hours, spent_evidence=args.spent_evidence)
    result['inputs'] = {name: dict(path=str(Path(getattr(args, name)).resolve()), sha256=digest(getattr(args, name)))
                        for name in ('report', 'supervisor', 'overheads')}
    write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'cumulative_gpu_hours',
         'three_card_parallel_wall_hours', 'suggested_cumulative_budget_integer_gpu_hours')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
