"""Bound the repaired OCN campaign on one H100, with persistent live logs.

Profile runs the two necessary arms sequentially and never resumes or tests.
Formal defaults to sequential execution; same-GPU concurrency requires a
separate passed, SHA-bound profile gate. This launcher never resubmits a Job.
--prior-allocated-seconds must include every earlier allocation in the Goal,
including the original 11273 seconds, profiles, and interrupted repair jobs.
Platform Scheduled-to-Pod-deletion accounting remains a separate requirement.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from item_local_gpu_job import single_process_environment


CAMPAIGN = 'stage07_ocn_repair_seed2024_20261020_r1'
TOTAL_SECONDS = 28 * 3600
ORIGINAL_SECONDS = 11273
HISTORICAL = '/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized'
ARMS = (
    ('control', 'ocn_repair_control_seed2024.json', 'stage07_ocn_repair_control_seed2024_20261020_r1'),
    ('item_local', 'ocn_repair_item_local_seed2024.json', 'stage07_ocn_repair_item_local_seed2024_20261020_r1'),
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_receipt(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def budget(prior, maximum, platform_cap, mode):
    need(all(math.isfinite(value) for value in (prior, maximum, platform_cap)), 'Budget values must be finite')
    need(ORIGINAL_SECONDS <= prior < TOTAL_SECONDS, 'Prior allocation must include at least the original 11273 seconds and remain below 28 GPU hours')
    need(maximum > 90 and platform_cap >= maximum + 150, 'Leave at least 150 seconds between the process limit and declared platform cap')
    need(platform_cap <= TOTAL_SECONDS - prior, 'Declared platform cap would exceed the cumulative 28-GPU-hour budget')
    if mode == 'profile':
        need(platform_cap <= 1800 and maximum <= 1650, 'The complete two-arm profile allocation is capped at 1800 seconds')


def git(directory, *arguments):
    return subprocess.check_output(['git', '-C', str(directory), *arguments], text=True).strip()


def source_binding(goal):
    binding = {}
    for label in ('code', 'engine'):
        directory = goal / label
        need(directory.is_dir(), f'Missing frozen {label} checkout')
        need(git(directory, 'status', '--porcelain') == '', f'Frozen {label} checkout is dirty')
        binding[label + '_commit'] = git(directory, 'rev-parse', 'HEAD')
    binding['runner_sha256'] = sha(goal / 'code/scripts/train/run_item_local_attention.py')
    binding['wrapper_sha256'] = sha(Path(__file__).resolve())
    binding['launcher_helper_sha256'] = sha(Path(__file__).with_name('item_local_gpu_job.py'))
    binding['engine_files'] = {name: sha(goal / 'engine/genrec/models/DIFF_GRM' / name)
                               for name in ('model.py', 'item_local_attention.py', 'trainer.py', 'training_state.py')}
    return binding


def studies(goal):
    directory = goal / 'code/configs/07_item_local_attention'
    campaign = directory / 'ocn_repair_campaign_seed2024.json'
    campaign_value = load(campaign)
    need(campaign_value.get('campaign_id') == CAMPAIGN, 'Campaign configuration identity differs')
    limits = campaign_value['budget']
    need(limits['cumulative_allocated_gpu_hours_limit'] == 28 and limits['legacy_allocated_gpu_seconds'] == ORIGINAL_SECONDS
         and limits['profile_platform_max_seconds'] == 1800 and limits['profile_process_max_seconds'] == 1650
         and limits['gpu_count'] == 1 and limits['gpu_type'] == 'H100', 'Campaign resource limits differ')
    rows = []
    for arm, filename, run_id in ARMS:
        config = directory / filename
        study_value = load(config)
        need(study_value.get('run_id') == run_id and study_value.get('seed') == 2024, f'{arm}: study identity differs')
        rows.append({'arm': arm, 'run_id': run_id, 'config': str(config),
                     'study_sha256': sha(config), 'output': str(goal / 'runs' / run_id)})
    return rows, {'path': str(campaign), 'sha256': sha(campaign), 'engine_commit': campaign_value['engine_commit']}


def gpu_sampler(path, environment, stopped, summary):
    """Read the assigned GPU's utilisation; never create artificial load."""
    command = ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used', '--format=csv,noheader,nounits']
    for key in ('NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        selected = environment.get(key, '').strip()
        if selected and selected.lower() not in ('all', 'none', 'void') and ',' not in selected:
            command += ['--id', selected]
            break
    values = []
    summary.update(status='sampling', interval_seconds=5, command=command, valid_samples=0, failed_samples=0,
                   csv=str(path), scope='Read-only instantaneous five-second samples during the short profile; does not establish three-hour utilisation or actual allocation/billing')
    try:
        with path.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['observed_utc', 'gpu_timestamp', 'utilization_gpu_percent', 'memory_used_mib', 'status', 'error'])
            while not stopped.is_set():
                beginning = time.monotonic()
                try:
                    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                            timeout=4, env=environment, check=False)
                    need(result.returncode == 0, result.stderr.strip()[:500] or 'nvidia-smi returned nonzero')
                    rows = list(csv.reader(result.stdout.strip().splitlines()))
                    need(len(rows) == 1 and len(rows[0]) == 3, 'Need exactly one assigned GPU; refusing to attribute other GPU samples')
                    gpu_time, utilisation, memory = [item.strip() for item in rows[0]]
                    utilisation, memory = float(utilisation), float(memory)
                    need(math.isfinite(utilisation) and 0 <= utilisation <= 100 and math.isfinite(memory) and memory >= 0,
                         'nvidia-smi did not return finite utilisation/memory measurements')
                    writer.writerow([utc(), gpu_time, utilisation, memory, 'measured', ''])
                    values.append(utilisation)
                    summary['valid_samples'] += 1
                except Exception as error:
                    writer.writerow([utc(), '', '', '', 'sampling_failed', type(error).__name__ + ': ' + str(error)])
                    summary['failed_samples'] += 1
                handle.flush()
                stopped.wait(max(0, 5 - (time.monotonic() - beginning)))
    except Exception as error:
        summary['error'] = type(error).__name__ + ': ' + str(error)
    summary['status'] = 'passed' if values and not summary['failed_samples'] else 'partial' if values else 'unavailable'
    if values:
        summary.update(mean_utilization_gpu_percent=sum(values) / len(values), minimum_utilization_gpu_percent=min(values),
                       maximum_utilization_gpu_percent=max(values), fraction_samples_below_40_percent=sum(value < 40 for value in values) / len(values))
    if path.is_file():
        summary['csv_sha256'] = sha(path)


def positive_seconds(value, label):
    need(not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0,
         f'Invalid measured duration: {label}')
    return float(value)


def digest(value):
    need(isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value),
         'Profile identity must be a lowercase SHA256')
    return value


def profile_results(child):
    work = Path(child['output']) / 'profile'
    path = work / 'profile_results.json'
    value, complete = load(path), load(work / 'COMPLETE.json')
    need(value.get('status') == 'profile_completed' and value.get('repair_campaign') == CAMPAIGN
         and value.get('variant') == child['arm'] and value.get('test_data_evaluated') is False,
         'Profile results have the wrong campaign/variant or evaluated final test data')
    need(not (work / 'TEST_STARTED.json').exists(), 'A profile contains a final-test marker')
    need(complete.get('mode') == 'profile' and complete.get('status') == 'profile_completed', 'Missing profile completion status')
    need(complete.get('repair_campaign') == CAMPAIGN and complete.get('variant') == child['arm']
         and complete.get('result_file') == 'profile_results.json' and complete.get('result_sha256') == sha(path),
         'Profile COMPLETE is not bound to these result bytes')
    expected_count = 84 if child['arm'] == 'control' else 90
    need(type(value['all_parameter_tensors']) is int and value['all_parameter_tensors'] == expected_count
         and type(value['optimizer_state_count']) is int and value['optimizer_state_count'] == expected_count
         and value.get('all_gradient_coverage') is True
         and value.get('actual_changed_parameter_tensors') == expected_count, 'Profile parameter/gradient/Adam coverage failed')
    integrity, recovery = value['training_integrity'], value['recovery_check']
    need(integrity.get('passed') is True and integrity.get('all_gradient_coverage') is True
         and integrity.get('all_observed_gradients_finite') is True and integrity.get('all_optimizer_steps_match') is True
         and integrity.get('optimizer_state_count') == integrity.get('all_parameter_tensors') == expected_count
         and integrity.get('cumulative_optimizer_updates') == 104, 'Profile training integrity was not accepted')
    hooks = integrity['gradient_hook_calls']
    need(len(hooks) == expected_count and all(type(count) is int and count > 0 for count in hooks.values()),
         'Profile gradient hooks do not cover every parameter')
    need(recovery.get('status') == 'passed' and recovery.get('strict_model_optimizer_scheduler_RNG_restore') is True
         and recovery.get('optimizer_updates_executed') == 1 and recovery.get('micro_batches') == 16
         and recovery.get('updates_before') == 104 and recovery.get('updates_after') == 105
         and recovery.get('actual_changed_parameter_tensors') == expected_count
         and recovery.get('all_gradient_coverage') is True
         and recovery.get('terminal_profile_state') is True and recovery.get('manual_one_update_probe') is True
         and recovery.get('trainer_fit_resume_not_exercised') is True and recovery.get('cuda_bitwise_claim') is False,
         'Profile actual manual recovery/update checks failed')
    coverage = recovery['gradient_integrity']
    need(coverage.get('passed') is True and coverage.get('all_gradient_coverage') is True
         and coverage.get('all_parameter_tensors') == expected_count and coverage.get('all_observed_gradients_finite') is True,
         'Recovered update has incomplete/nonfinite gradients')
    original = recovery['original_checkpoint_hashes_unchanged']
    need(digest(original['last']) == sha(work / 'training_state/last.pt')
         and digest(original['best']) == sha(work / 'training_state/best_model.bin')
         == value['checkpoint_sha256'] == complete['checkpoint_sha256'], 'Profile original checkpoint byte bindings differ')
    need(value.get('formal_train_batches_per_epoch') == 1652 and value.get('formal_eval_batches') == 2796,
         'Profile timing used truncated/unexpected train or validation data')
    result_sha = sha(path)
    if child.get('profile_results_sha256') is not None:
        need(child['profile_results_sha256'] == result_sha, 'Profile results changed after the completed Job receipt')
    return {'profile_results_sha256': result_sha,
            'initial_base_state_sha256': digest(value['initial_base_state_sha256']),
            'initial_rng_sha256': digest(value['initial_rng_sha256']),
            'train_epoch_seconds': positive_seconds(value['training_loop_seconds'], 'training loop'),
            'full_val_seconds': positive_seconds(value['full_val_seconds'], 'full validation'),
            'full_test_seconds_estimate': positive_seconds(value['beam256_val_2_batches_seconds'], 'beam256 validation benchmark') * 2796 / 2,
            'all_parameter_tensors': expected_count, 'all_gradient_and_optimizer_checks_passed': True,
            'manual_recovery_update_passed': True}


def paired_profile_checks(children):
    summaries = {child['arm']: profile_results(child) for child in children}
    need(set(summaries) == {'control', 'item_local'} and len(children) == 2, 'Need exactly two accepted profile arms')
    for field in ('initial_base_state_sha256', 'initial_rng_sha256'):
        need(summaries['control'][field] == summaries['item_local'][field], f'Paired profile {field} differs')
    projected = sum(200 * row['train_epoch_seconds'] + 37 * row['full_val_seconds']
                    + row['full_test_seconds_estimate'] for row in summaries.values())
    return {'status': 'passed', 'arms': summaries, 'common_initial_base_and_rng_verified': True,
            'formal_200_epoch_pair_seconds_with_margin': projected * 1.25 + 300,
            'projection_basis': 'Two individual full train/validation profiles, 200 epochs each, 37 validations each, beam256 validation extrapolation for final test, 25 percent margin plus 300 seconds startup/cleanup; margin is a reference, not a hard requirement',
            'projection_scope': 'Runtime planning only; does not predict convergence or prove the platform resource policy permits a long run'}


def profile_gate(path, binding, rows):
    value = load(path)
    need(value.get('campaign_id') == CAMPAIGN and value.get('mode') == 'profile'
         and value.get('status') == 'profile_completed', 'Formal execution needs a completed two-arm profile receipt')
    need(value.get('source_binding') == binding, 'Profile and formal frozen source bindings differ')
    expected = {row['arm']: (row['run_id'], row['study_sha256'], str(Path(row['output']).resolve())) for row in rows}
    observed = {row['arm']: (row['run_id'], row['study_sha256'], str(Path(row['output']).resolve())) for row in value.get('children', [])}
    need(observed == expected and len(value['children']) == len(rows), 'Profile receipt does not bind exactly the two authorized studies')
    for child in value['children']:
        need(child.get('returncode') == 0 and child.get('status') == 'completed', 'A profile arm failed or was not completed')
        completion = load(Path(child['output']) / 'profile/COMPLETE.json')
        need(completion.get('mode') == 'profile' and completion.get('status') == 'profile_completed', 'Profile completion artifact is missing or incomplete')
        need(sha(Path(child['output']) / 'profile/COMPLETE.json') == child['complete_sha256'], 'Profile completion artifact changed')
    checks = paired_profile_checks(value['children'])
    need(checks == value.get('paired_profile_checks'), 'Passed phase receipt omits or changes explicit paired profile checks')
    sampling = value.get('gpu_sampling', {})
    need(sampling.get('status') in ('passed', 'partial') and sampling.get('valid_samples', 0) > 0
         and sha(Path(sampling['csv'])) == sampling.get('csv_sha256'),
         'Formal policy assessment requires actual SHA-bound GPU utilisation samples, not a sampling failure')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'paired_profile_checks': checks}


def concurrency_gate(path, profile_evidence):
    value = load(path)
    need(value.get('schema') == 'stage07_ocn_repair_same_gpu_concurrency_gate_v1'
         and value.get('campaign_id') == CAMPAIGN and value.get('status') == 'passed'
         and value.get('formal_same_gpu_concurrency_safe') is True
         and value.get('visible_gpu_count') == 1
         and value.get('profile_receipt_sha256') == profile_evidence['sha256'],
         'Concurrent formal execution requires explicit, matching passed profile evidence')
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def formal_projection(children, checks):
    """Conservative remaining-work plan; checkpoint validity stays the runner's job."""
    plans, seconds = [], 0.0
    for child in children:
        output = Path(child['output'])
        measured = checks['arms'][child['arm']]
        completed = 0
        if child['status'] == 'skipped_complete':
            plans.append({'arm': child['arm'], 'already_complete': True, 'remaining_seconds': 0.0})
            continue
        if (output / 'FIT_COMPLETE.json').is_file():
            completed = 200
            basis = 'FIT_COMPLETE; child strictly requires its matching terminal last.pt before skipping fit'
        elif child['resume']:
            timings = load(output / 'training_state/epoch_timings.json')
            need(isinstance(timings, list) and [row['epoch'] for row in timings] == list(range(1, len(timings) + 1))
                 and len(timings) <= 200, 'Recovery timing history is missing, gapped, or outside the fixed protocol')
            # Each epoch is committed before the next starts. The latest loop
            # timing can precede its boundary commit; subtract one, never count
            # that potentially uncommitted epoch as completed for budgeting.
            completed = max(0, len(timings) - 1)
            basis = 'Conservative epoch-timing prefix minus one potentially uncommitted epoch; actual state is validated by the child'
        else:
            basis = 'Fresh repaired run, all 200 epochs budgeted'
        previous_val = 0 if completed < 20 else (completed - 20) // 5 + 1
        remaining_train, remaining_val = 200 - completed, 37 - previous_val
        final_seconds = 0.0 if (output / 'TEST_RESULT.json').exists() else measured['full_test_seconds_estimate']
        arm_seconds = remaining_train * measured['train_epoch_seconds'] + remaining_val * measured['full_val_seconds'] + final_seconds
        seconds += arm_seconds
        plans.append({'arm': child['arm'], 'training_completed_lower_bound': completed,
                      'remaining_epochs_upper_bound': remaining_train, 'remaining_validations_upper_bound': remaining_val,
                      'remaining_seconds_before_margin': arm_seconds, 'basis': basis})
    final_evaluation = next((row['full_test_seconds_estimate'] for child in reversed(children)
                             for row in [checks['arms'][child['arm']]]
                             if child['status'] != 'skipped_complete' and not (Path(child['output']) / 'TEST_RESULT.json').exists()), 0.0)
    reserve_extra = max(0.0, 1200.0 - final_evaluation) if final_evaluation else 0.0
    return {'remaining_seconds_reference': seconds,
            'last_arm_evaluation_reserve_extra_seconds': reserve_extra,
            'remaining_seconds_with_last_arm_reserve': seconds + reserve_extra + 150,
            'remaining_seconds_with_margin': seconds * 1.25 + 300,
            'arms': plans, 'margin': '25 percent plus 300 seconds, recorded as a planning reference rather than a hard launch requirement',
            'scope': 'Planning estimate from the accepted repair profile, not convergence or runtime guarantee; the phase deadline remains enforced'}


def child_guard(row, mode, allow_resume):
    output = Path(row['output'])
    work = output / 'profile' if mode == 'profile' else output
    if mode == 'profile':
        need(not any((work / name).exists() for name in ('STARTED.json', 'COMPLETE.json', 'training_state/last.pt')),
             f"{row['arm']}: profile already started; do not duplicate or resume it")
        return False, False
    need(not ((work / 'TEST_STARTED.json').exists() and not (work / 'TEST_RESULT.json').exists()),
         f"{row['arm']}: final test started without a result; refusing a duplicate test")
    if (work / 'COMPLETE.json').exists():
        complete, identity = load(work / 'COMPLETE.json'), load(work / 'identity.json')
        need(complete.get('mode') == 'formal' and complete.get('status') == 'formal_completed'
             and identity.get('run_id') == row['run_id'] and identity.get('study_sha256') == row['study_sha256'],
             f"{row['arm']}: completed artifacts have a different identity")
        need((work / 'TEST_RESULT.json').is_file()
             and sha(work / 'training_state/best_model.bin') == complete['checkpoint_sha256'],
             f"{row['arm']}: completed test/checkpoint evidence is incomplete")
        return False, True
    existing = (work / 'STARTED.json').exists() or (work / 'training_state/last.pt').exists()
    if existing:
        need(allow_resume and (work / 'training_state/last.pt').is_file(),
             f"{row['arm']}: existing execution requires explicit --resume and its own last.pt")
        identity = load(work / 'identity.json')
        need(identity.get('run_id') == row['run_id'] and identity.get('study_sha256') == row['study_sha256'],
             f"{row['arm']}: recovery state is outside the same campaign run")
    return existing, False


def execute(args):
    goal = args.goal_root.resolve()
    need(os.name == 'posix', 'GPU wrapper requires POSIX process-group cleanup')
    need(Path(__file__).resolve() == goal / 'code/scripts/train/ocn_repair_gpu_job.py', 'Run the wrapper from its frozen campaign checkout')
    budget(args.prior_allocated_seconds, args.max_seconds, args.platform_max_seconds, args.mode)
    need(args.attempt >= 1 and (not args.resume or args.attempt > 1), 'Recovery requires a new attempt number after prior Job release')
    need(args.mode != 'profile' or not args.resume, 'Profile cannot resume')
    need(args.mode == 'formal' or not args.concurrent, 'Profile must run sequentially')
    reserve = args.training_reserve_seconds if args.training_reserve_seconds is not None else (60.0 if args.mode == 'profile' else 1200.0)
    need(math.isfinite(reserve) and 0 <= reserve < args.max_seconds - 90, 'Invalid training/evaluation reserve')
    phase = goal / 'delivery' / f'{args.mode}_attempt{args.attempt}'
    need(phase.is_relative_to(goal), 'Phase output escaped the campaign root')
    phase.mkdir(parents=True, exist_ok=False)
    receipt_path = phase / 'receipt.json'
    started = time.time()
    deadline = started + args.max_seconds - 90
    record = {'schema': 'stage07_ocn_repair_job_phase_v1', 'campaign_id': CAMPAIGN,
              'mode': args.mode, 'attempt': args.attempt, 'status': 'starting', 'pid': os.getpid(),
              'started_unix': started, 'started_utc': utc(), 'deadline_unix': deadline,
              'allocated_gpu_count': 1, 'max_process_seconds': args.max_seconds,
              'declared_platform_max_seconds': args.platform_max_seconds,
              'prior_allocated_seconds_declared': args.prior_allocated_seconds,
              'cumulative_budget_seconds': TOTAL_SECONDS,
              'remaining_seconds_after_declared_platform_cap': TOTAL_SECONDS - args.prior_allocated_seconds - args.platform_max_seconds,
              'concurrent_same_gpu': args.concurrent, 'training_reserve_seconds': reserve,
              'console': str(phase / 'console.log'), 'children': [], 'goal_complete': False,
              'time_scope': 'Process wall time and caller-declared prior/platform caps; actual GPU allocation and release require platform lifecycle receipts'}
    write_receipt(receipt_path, record)
    processes, readers, owned_groups = {}, [], set()
    sampling_stop, sampling_thread = threading.Event(), None
    stop = {'reason': None, 'at': None}
    lock = threading.Lock()
    signals_before = {}

    def live_groups():
        for pid in list(owned_groups):
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                owned_groups.discard(pid)
        return list(owned_groups)

    def signal_groups(number):
        for pid in live_groups():
            try:
                os.killpg(pid, number)
            except ProcessLookupError:
                owned_groups.discard(pid)

    def terminate(reason):
        if stop['reason'] is None:
            stop.update(reason=reason, at=time.time())
        signal_groups(signal.SIGTERM)

    def on_signal(number, frame):
        terminate('received_signal_' + str(number))

    with (phase / 'console.log').open('x', encoding='utf-8') as console:
        def emit(label, line):
            message = '[' + label + '] ' + line.rstrip('\r\n') + '\n'
            with lock:
                console.write(message)
                console.flush()
                try:
                    sys.stdout.write(message)
                    sys.stdout.flush()
                except (BrokenPipeError, OSError):
                    pass

        def stream(arm, process):
            for line in process.stdout:
                emit(arm, line)
            process.stdout.close()

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                signals_before[number] = signal.signal(number, on_signal)
            record['source_binding'] = source_binding(goal)
            rows, record['campaign_config'] = studies(goal)
            need(record['source_binding']['engine_commit'] == record['campaign_config']['engine_commit'],
                 'Frozen engine differs from the campaign-approved repair commit')
            environment, record['platform_launcher_env'] = single_process_environment(os.environ)
            environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
            if args.mode == 'profile':
                # Keep the key set stable while the read-only sampler updates
                # values and the main thread persists progress receipts.
                record['gpu_sampling'] = {key: None for key in (
                    'status', 'interval_seconds', 'command', 'valid_samples', 'failed_samples', 'csv', 'scope', 'error',
                    'mean_utilization_gpu_percent', 'minimum_utilization_gpu_percent', 'maximum_utilization_gpu_percent',
                    'fraction_samples_below_40_percent', 'csv_sha256')}
                sampling_thread = threading.Thread(target=gpu_sampler,
                    args=(phase / 'gpu_samples.csv', environment, sampling_stop, record['gpu_sampling']), daemon=True)
                sampling_thread.start()
            if args.mode == 'formal':
                need(args.profile_evidence is not None, 'Formal execution requires --profile-evidence')
                record['profile_evidence'] = profile_gate(args.profile_evidence, record['source_binding'], rows)
            if args.concurrent:
                need(args.concurrent_profile_evidence is not None, 'Concurrent formal execution requires --concurrent-profile-evidence')
                record['concurrency_evidence'] = concurrency_gate(args.concurrent_profile_evidence, record['profile_evidence'])
            elif args.concurrent_profile_evidence is not None:
                raise ValueError('Concurrency evidence requires an explicit --concurrent flag')
            for row in rows:
                resume, skip = child_guard(row, args.mode, args.resume)
                command = [sys.executable, '-u', '-B', str(goal / 'code/scripts/train/run_item_local_attention.py'),
                           '--mode', args.mode, '--config', row['config'], '--output', row['output'],
                           '--project-root', HISTORICAL, '--engine-root', str(goal / 'engine'),
                           '--deadline-unix', str(deadline), '--training-deadline-reserve-seconds', str(reserve)]
                if resume:
                    command.append('--resume')
                record['children'].append({**row, 'argv': command, 'resume': resume,
                                           'status': 'skipped_complete' if skip else 'pending',
                                           'returncode': 0 if skip else None})
            if args.mode == 'formal':
                record['remaining_work_projection'] = formal_projection(
                    record['children'], record['profile_evidence']['paired_profile_checks'])
                required = record['remaining_work_projection']['remaining_seconds_with_last_arm_reserve']
                if args.concurrent:
                    # A sequential single-arm profile cannot certify concurrent
                    # throughput; the separately required gate must measure it.
                    evidence = load(args.concurrent_profile_evidence)
                    required = positive_seconds(evidence['formal_remaining_seconds_with_margin'], 'measured concurrent formal projection')
                    record['remaining_work_projection']['concurrent_gate_seconds_with_margin'] = required
                need(required <= args.max_seconds - 90, 'Measured remaining-work forecast does not fit the requested phase budget; do not launch either formal arm')
            record['status'] = 'running'
            write_receipt(receipt_path, record)
            emit('wrapper', f"{args.mode} attempt {args.attempt}: one H100, {'concurrent' if args.concurrent else 'sequential'}, deadline={deadline}")
            while True:
                active = [row for row in record['children'] if row['status'] == 'running']
                pending = [row for row in record['children'] if row['status'] == 'pending']
                if time.time() >= deadline + 30:
                    terminate('process_wall_deadline')
                if stop['reason'] is None and pending and (args.concurrent or not active):
                    for row in pending if args.concurrent else pending[:1]:
                        need(time.time() < deadline - reserve, 'Insufficient remaining phase budget to start another arm')
                        process = subprocess.Popen(row['argv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, encoding='utf-8', errors='replace', bufsize=1,
                                                   env=environment, start_new_session=True)
                        processes[row['arm']] = process
                        owned_groups.add(process.pid)
                        row.update(status='running', pid=process.pid, started_unix=time.time(), started_utc=utc())
                        reader = threading.Thread(target=stream, args=(row['arm'], process), daemon=True)
                        reader.start()
                        readers.append(reader)
                        write_receipt(receipt_path, record)
                for row in record['children']:
                    if row['status'] != 'running':
                        continue
                    code = processes[row['arm']].poll()
                    if code is None:
                        continue
                    row.update(returncode=code, ended_unix=time.time(), ended_utc=utc())
                    row['status'] = 'failed_or_incomplete'
                    if code == 0:
                        artifact = Path(row['output']) / ('profile/COMPLETE.json' if args.mode == 'profile' else 'COMPLETE.json')
                        completion = load(artifact)
                        need(completion.get('mode') == args.mode and completion.get('status') == args.mode + '_completed',
                             f"{row['arm']}: child exited without its expected completion receipt")
                        row.update(complete_path=str(artifact), complete_sha256=sha(artifact))
                        if args.mode == 'profile':
                            row['profile_results_sha256'] = profile_results(row)['profile_results_sha256']
                        row['status'] = 'completed'
                    else:
                        terminate('child_failure_' + row['arm'])
                    write_receipt(receipt_path, record)
                alive = bool(live_groups())
                if stop['reason'] is not None:
                    if alive and time.time() >= stop['at'] + 15:
                        signal_groups(signal.SIGKILL)
                    if not alive:
                        break
                elif all(row['status'] in ('completed', 'skipped_complete') for row in record['children']):
                    if args.mode == 'profile':
                        record['paired_profile_checks'] = paired_profile_checks(record['children'])
                    record['status'] = args.mode + '_completed'
                    break
                time.sleep(.2)
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            terminate('wrapper_exception')
        finally:
            if live_groups():
                terminate(stop['reason'] or 'wrapper_cleanup')
            cleanup_end = time.time() + 15
            for process in processes.values():
                if process.poll() is None:
                    try:
                        process.wait(timeout=max(.1, cleanup_end - time.time()))
                    except subprocess.TimeoutExpired:
                        signal_groups(signal.SIGKILL)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            record['cleanup_error'] = 'An owned child did not exit after SIGKILL'
            # A child leader can exit while one of its workers still holds its
            # process group/stdout. Clean that group too, without touching the
            # platform job, other jobs, or any external process group.
            if live_groups():
                signal_groups(signal.SIGKILL)
            for row in record['children']:
                if row['arm'] in processes and row['status'] == 'running':
                    row.update(status='failed_or_incomplete', returncode=processes[row['arm']].returncode,
                               ended_unix=time.time(), ended_utc=utc())
            for reader in readers:
                reader.join(timeout=5)
            sampling_stop.set()
            if sampling_thread is not None:
                sampling_thread.join(timeout=6)
                if sampling_thread.is_alive():
                    record['gpu_sampling'].update(status='unavailable', error='GPU sampler did not finish before receipt closure')
            record['owned_process_groups_alive_after_cleanup'] = live_groups()
            record['stdout_reader_threads_alive_after_cleanup'] = sum(reader.is_alive() for reader in readers)
            if record['owned_process_groups_alive_after_cleanup'] or record['stdout_reader_threads_alive_after_cleanup']:
                record['status'] = 'failed_or_incomplete'
            if record['status'] not in ('profile_completed', 'formal_completed'):
                record['status'] = 'failed_or_incomplete'
            record.update(stop_reason=stop['reason'], ended_unix=time.time(), ended_utc=utc(),
                          process_wall_seconds=time.time() - started)
            write_receipt(receipt_path, record)
            emit('wrapper', json.dumps({'status': record['status'], 'receipt': str(receipt_path),
                                       'children': [{key: row.get(key) for key in ('arm', 'status', 'returncode')}
                                                    for row in record['children']]}, ensure_ascii=False))
            for number, previous in signals_before.items():
                signal.signal(number, previous)
    return 0 if record['status'] in ('profile_completed', 'formal_completed') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('profile', 'formal'), required=True)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--platform-max-seconds', type=float, required=True)
    parser.add_argument('--prior-allocated-seconds', type=float, required=True)
    parser.add_argument('--training-reserve-seconds', type=float)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile-evidence', type=Path)
    parser.add_argument('--concurrent', action='store_true')
    parser.add_argument('--concurrent-profile-evidence', type=Path)
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
