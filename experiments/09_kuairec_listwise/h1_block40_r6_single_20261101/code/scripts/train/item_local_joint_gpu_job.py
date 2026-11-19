"""Bound the alternating joint campaign on one H100; never submit a Job.

Smoke runs control then item_local inside one <=1100-second allocation (process
<=900 seconds). Formal runs exactly one complete variant per Job and requires
SHA-bound paired smoke and full-Job budget/resource-policy evidence. Recovery
uses only that run's own last.pt after caller-verified previous Job release.
This wrapper records process time, not platform allocation or GPU release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from item_local_gpu_job import single_process_environment
from ocn_repair_gpu_job import gpu_sampler


CAMPAIGN = 'stage07_item_local_alternating_joint_seed2024_20261020_r1'
VARIANTS = ('control', 'item_local')
PRIOR_SECONDS = 11637
TOTAL_SECONDS = 28 * 3600
HISTORICAL = '/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized'
STUDY = 'configs/07_item_local_attention/joint_alternating_seed2024.json'
ENGINE_FILES = (
    'genrec/models/DIFF_GRM/model.py', 'genrec/models/DIFF_GRM/trainer.py',
    'genrec/models/DIFF_GRM/training_state.py', 'genrec/models/DIFF_GRM/joint_cross_pse.py',
    'genrec/models/DIFF_GRM/cross_pse.py', 'genrec/models/DIFF_GRM/item_local_attention.py',
    'genrec/models/DIFF_GRM/tokenizer.py', 'genrec/models/DIFF_GRM/collate.py',
    'genrec/models/DIFF_GRM/config.yaml', 'genrec/default.yaml',
    'genrec/datasets/AmazonReviews2014/config.yaml', 'genrec/utils.py', 'genrec/pipeline.py',
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


def digest(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Expected a lowercase SHA256')
    return value


def load(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write_receipt(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def positive(value):
    need(not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0, 'Expected finite positive duration/count')
    return float(value)


def run_id(variant):
    need(variant in VARIANTS, 'Unknown joint variant')
    return 'stage07_item_local_joint_' + variant + '_seed2024_20261020_r1'


def budget(prior, maximum, platform_cap, mode):
    need(all(not isinstance(value, bool) and math.isfinite(value) for value in (prior, maximum, platform_cap)), 'Budgets must be finite')
    need(PRIOR_SECONDS <= prior < TOTAL_SECONDS, 'Prior allocation must include all earlier allocations, at least 11637 seconds')
    need(maximum > 90 and platform_cap >= maximum + 150, 'Leave at least 150 seconds of allocation margin above the process cap')
    need(prior + platform_cap <= TOTAL_SECONDS, 'The declared allocation would exceed the cumulative 28-GPU-hour limit')
    if mode == 'smoke':
        need(maximum <= 900 and platform_cap <= 1100 and (prior - 11273) + platform_cap <= 1800,
             'The complete acceptance allocations since 11273 prior seconds must fit 1800 seconds, with process<=900/platform<=1100')


def git(directory, *arguments):
    return subprocess.check_output(['git', '-C', str(directory), *arguments], text=True).strip()


def source_binding(goal, expected_root, expected_engine):
    result = {}
    for label, expected in (('code', expected_root), ('engine', expected_engine)):
        need(re.fullmatch('[0-9a-f]{40}', expected), 'Both expected commits must be full 40-character SHAs')
        directory = goal / label
        need(git(directory, 'rev-parse', 'HEAD') == expected and git(directory, 'status', '--porcelain') == '',
             'Frozen ' + label + ' checkout differs or is dirty')
        result[label + '_commit'] = expected
    need(Path(__file__).resolve() == goal / 'code/scripts/train/item_local_joint_gpu_job.py',
         'Execute the wrapper from its frozen joint campaign checkout')
    for field, name in (('runner_sha256', 'scripts/train/run_item_local_joint.py'),
                        ('wrapper_sha256', 'scripts/train/item_local_joint_gpu_job.py'),
                        ('launcher_helper_sha256', 'scripts/train/item_local_gpu_job.py'),
                        ('sampler_helper_sha256', 'scripts/train/ocn_repair_gpu_job.py'),
                        ('stage05_helper_sha256', 'scripts/repro/stage05_repro.py')):
        result[field] = sha(goal / 'code' / name)
    result['engine_files'] = {name: sha(goal / 'engine' / name) for name in ENGINE_FILES}
    return result


def study_rows(goal):
    path = goal / 'code' / STUDY
    value = load(path)
    need(value['campaign_id'] == value['protocol_id'] == CAMPAIGN and value['seed'] == 2024
         and value['recipe_id'] == 'refined_joint' and value['max_gpu_hours'] == 28
         and value['prior_allocated_gpu_seconds'] == PRIOR_SECONDS
         and value['gpu_count'] == 1 and value['gpu_type'] == 'H100'
         and value['global_codebook_mixer_enabled'] is False and value['tokenizer_codebooks_trainable'] is True
         and value['recommendation_gradient_bridge'] == 'diger_st', 'Joint study/protocol/resource identity differs')
    need(set(value['variants']) == set(VARIANTS), 'Need exactly the authorized control/item_local variants')
    rows = []
    for variant in VARIANTS:
        specification = value['variants'][variant]
        need(specification['run_id'] == run_id(variant) and specification['item_local_attention'] ==
             {'enabled': variant == 'item_local', 'num_heads': 4, 'dropout': 0.0}, 'Wrong joint variant controls')
        rows.append({'variant': variant, 'run_id': run_id(variant), 'config': str(path),
                     'study_sha256': sha(path), 'output': str(goal / 'runs' / run_id(variant))})
    return rows


def completion(row, mode, binding):
    work = Path(row['output']) / 'smoke' if mode == 'smoke' else Path(row['output'])
    complete = load(work / 'COMPLETE.json')
    name = 'smoke_results.json' if mode == 'smoke' else 'results.json'
    result, identity = load(work / name), load(work / 'identity.json')
    actual_id = row['run_id'] + ('_smoke' if mode == 'smoke' else '')
    expected_status = 'smoke_passed_not_formal' if mode == 'smoke' else 'formal_completed'
    need(complete['mode'] == mode and complete['status'] == result['status'] == expected_status,
         'Child has no valid completion status')
    for value in (identity, complete, result):
        need(value['campaign_id'] == CAMPAIGN and value['variant'] == row['variant'] and value['run_id'] == actual_id,
             'Completion/run/variant identity differs')
    need(identity['formal_run_id'] == row['run_id'] and identity['seed'] == 2024
         and identity['study_sha256'] == row['study_sha256'] and identity['engine_commit'] == binding['engine_commit']
         and identity['engine_source_sha256'] == binding['engine_files'], 'Child executed different study/engine sources')
    need(complete['result_file'] == name and complete['result_sha256'] == sha(work / name), 'COMPLETE does not bind result bytes')
    best_sha = sha(work / 'training_state/best_model.bin')
    need(best_sha == complete['checkpoint_sha256'] == result['checkpoint_sha256'], 'Selected checkpoint byte binding differs')
    reload = load(work / 'checkpoint_reload.json')
    need(reload['strict_all_tensor_match'] is True and reload['checkpoint_sha256'] == best_sha
         and reload['matching_committed_SID_sha256'] == digest(result['selected_SID_sha256'])
         and reload['mapping_synchronized_without_requantization'] is True, 'Selected checkpoint/SID reload evidence differs')
    code = load(work / 'code_identity.json')
    need(code['engine']['head'] == binding['engine_commit'] and code['engine']['tracked_dirty'] is False
         and code['engine_sources'] == binding['engine_files'] and code['runner_sha256'] == binding['runner_sha256']
         and code['stage05_helper_sha256'] == binding['stage05_helper_sha256'], 'Child execution source hashes differ')
    if mode == 'formal':
        start, test = load(work / 'TEST_STARTED.json'), load(work / 'TEST_RESULT.json')
        need(result['test_data_evaluated'] is True and test['started_receipt_sha256'] == sha(work / 'TEST_STARTED.json')
             and test['checkpoint_sha256'] == start['checkpoint_sha256'] == best_sha
             and test['selected_SID_sha256'] == start['selected_SID_sha256'] == result['selected_SID_sha256'],
             'Final test is not bound to its once-only marker and selected weights/SID')
    return work, result, {'complete_sha256': sha(work / 'COMPLETE.json'), 'result_sha256': sha(work / name),
                          'checkpoint_sha256': best_sha}


def smoke_result(row, binding):
    work, value, hashes = completion(row, 'smoke', binding)
    need(value['test_data_evaluated'] is False and not (work / 'TEST_STARTED.json').exists()
         and not (work / 'TEST_RESULT.json').exists(), 'Smoke evaluated test data')
    initial = load(work / 'initial_state_identity.json')
    need(initial['campaign_id'] == CAMPAIGN and initial['variant'] == row['variant'] and initial['resume_requested'] is False,
         'Smoke was not a fresh shared warm start')
    for field in ('initial_base_state_sha256', 'initial_rng_sha256'):
        need(digest(value[field]) == initial[field], 'Smoke initial summary differs from result: ' + field)
    digest(initial['initial_SID_sha256'])
    integrity = load(work / 'training_integrity.json')
    need(value['training_integrity'] == integrity and integrity['status'] == 'passed'
         and integrity['global_codebook_mixer_enabled'] is False
         and integrity['local_block_parameter_side'] == 'recommender', 'Joint training integrity is not passed')
    optimizer = integrity['optimizer']
    need(optimizer['status'] == 'passed' and optimizer['all_active_optimizer_steps_match'] is True
         and optimizer['recommender_optimizer_updates'] == 4 and optimizer['tokenizer_optimizer_updates'] == 2,
         'Smoke did not retain both alternating Adam clocks')
    covered = optimizer['covered_parameter_names']
    need(len(covered) == len(set(covered)) and optimizer['optimizer_state_count'] == len(covered), 'Adam names/state counts differ')
    tokenizer_names = {'joint_alignment.mixer.base_codebooks'}
    recommender_names = set(covered) - tokenizer_names
    need(tokenizer_names <= set(covered) and recommender_names
         and not any(name.startswith(('joint_alignment.', 'joint_sequence_adapter.')) for name in recommender_names),
         'Unexpected active tokenizer/global Mixer/SIA parameters')
    local_names = {name for name in recommender_names if name.startswith('item_local_attention.')}
    need(bool(local_names) == (row['variant'] == 'item_local'), 'Local attention belongs to wrong variant')
    phases = load(work / 'phase_integrity.json')
    need(phases == value['phase_integrity'] and sha(work / 'phase_integrity.json') == integrity['phase_records_sha256']
         and [phase['phase'] for phase in phases] == ['recommender', 'tokenizer', 'recommender']
         and [phase['epoch'] for phase in phases] == [1, 2, 3], 'Smoke phase history is incomplete or changed')
    for phase in phases:
        active = tokenizer_names if phase['phase'] == 'tokenizer' else recommender_names
        calls, changed = phase['active_gradient_hook_calls'], phase['changed_parameters']
        need(phase['optimizer_updates'] == 2 and phase['frozen_parameters_unchanged'] is True
             and phase['all_active_gradients_finite'] is True and phase['active_parameter_tensors'] == len(active)
             and set(calls) == active and all(type(count) is int and count > 0 for count in calls.values())
             and len(changed) == len(set(changed)) == phase['changed_parameter_tensors']
             and set(changed) <= active and changed, 'A smoke phase lacks actual finite gradients/updates or changed its frozen side')
        sid = phase['SID_integrity']
        need(sid['catalogue_mapping_matches'] is True and digest(sid['hard_codes_sha256']) == phase['sid_sha256'], 'Phase SID mapping is stale')
        if phase['phase'] == 'tokenizer':
            positive(phase['direct_recommendation_gradient_norm'])
            need(sid['nearest_assignments_verified'] is True, 'Tokenizer refresh did not verify nearest assignments')
        elif local_names:
            need(bool(set(changed) & local_names), 'Recommender did not actually update local attention')
    progress = value['fit']['joint_progress']
    need(progress['tokenizer_optimizer_updates'] == 2 and progress['sid_refresh_count'] == 1
         and progress['tokenizer_epochs_completed'] == 1 and value['fit']['completed_epochs'] == 3
         and value['fit']['cumulative_recommender_optimizer_updates'] == 4
         and phases[0]['sid_sha256'] == initial['initial_SID_sha256']
         and phases[2]['sid_sha256'] == phases[1]['sid_sha256'], 'Wrong phase/optimizer/SID refresh counts')
    recovery = value['recovery_check']
    probe = load(work / 'recovery_probe/recovery_check.json')
    need(recovery == probe and recovery['status'] == 'passed'
         and recovery['strict_model_optimizer_scheduler_RNG_joint_runtime_restore'] is True
         and recovery['source_phase'] == 'tokenizer' and recovery['optimizer_updates_executed'] == 1
         and recovery['micro_batches'] == 16 and recovery['tokenizer_parameters_and_SID_unchanged'] is True
         and recovery['manual_one_update_probe'] is True and recovery['trainer_fit_resume_not_exercised'] is True
         and recovery['cuda_bitwise_claim'] is False, 'Smoke actual tokenizer-boundary recovery/update evidence failed')
    original = {'last': sha(work / 'training_state/last.pt'), 'best': hashes['checkpoint_sha256'],
                'source': sha(work / 'smoke_tokenizer_boundary.pt')}
    need(recovery['original_checkpoint_hashes_unchanged'] == original and recovery['source_last_sha256'] == original['source'],
         'Recovery modified original smoke checkpoint bytes')
    need(recovery['actual_changed_recommender_parameters'] and recovery['optimizer_clock_check']['all_active_optimizer_steps_match'] is True
         and recovery['optimizer_clock_check']['recommender_optimizer_updates'] == 3
         and recovery['optimizer_clock_check']['tokenizer_optimizer_updates'] == 2,
         'Recovered real optimizer update was not accepted')
    identity = load(work / 'input_identities.json')
    inputs = identity['artifacts']
    package = Path(row['config']).parents[2] / 'experiments/05_crosspse_joint'
    recipe = load(package / 'reproducibility/recipes/refined_joint.json')
    lock_path = package / 'reproducibility/assets.lock.json'
    lock = load(lock_path)['artifacts']
    need(load(work / 'recipe.json') == recipe and identity['assets_lock_sha256'] == sha(lock_path)
         and len(inputs) == 13 and {record['path'] for record in inputs} == set(recipe['required_artifacts']),
         'Missing/duplicate/different Refined warm-start input recipe')
    for record in inputs:
        need(record['sha256'] == lock[record['path']]['sha256'] and record['size_bytes'] == lock[record['path']]['size_bytes'],
             'Warm-start input identity differs from the locked source artifact')
    return {**hashes, 'initial_base_state_sha256': value['initial_base_state_sha256'],
            'initial_rng_sha256': value['initial_rng_sha256'], 'initial_SID_sha256': initial['initial_SID_sha256'],
            'input_artifacts': inputs, 'active_recommender_parameter_tensors': len(recommender_names),
            'active_tokenizer_parameter_tensors': len(tokenizer_names), 'phase_checks_passed': True,
            'actual_recovery_update_passed': True, 'smoke_process_seconds': positive(value['process_wall_seconds'])}


def paired_smoke(rows, binding):
    summaries = {row['variant']: smoke_result(row, binding) for row in rows}
    need(set(summaries) == set(VARIANTS), 'Both smoke variants must pass')
    for field in ('initial_base_state_sha256', 'initial_rng_sha256', 'initial_SID_sha256', 'input_artifacts'):
        need(summaries['control'][field] == summaries['item_local'][field], 'Paired smoke shared initialization/RNG/inputs differ: ' + field)
    return {'status': 'passed', 'common_warm_start_and_rng_verified': True, 'variants': summaries,
            'scope': 'Short rec/tok/rec gradient/recovery acceptance; not a full-run forecast or utilisation-policy proof'}


def smoke_gate(path, rows, binding):
    value = load(path)
    need(value.get('schema') == 'stage07_item_local_joint_job_phase_v1' and value.get('campaign_id') == CAMPAIGN
         and value.get('mode') == 'smoke' and value.get('status') == 'smoke_completed'
         and value.get('source_binding') == binding and value['owned_process_groups_alive_after_cleanup'] == []
         and value['stdout_reader_alive_after_cleanup'] is False, 'Formal requires a successful same-source cleaned-up two-arm smoke receipt')
    children = value['children']
    need(len(children) == 2 and [child['variant'] for child in children] == list(VARIANTS), 'Smoke was not sequential control/item_local')
    for child, row in zip(children, rows):
        need(all(child[key] == row[key] for key in ('variant', 'run_id', 'output', 'study_sha256'))
             and child['status'] == 'completed' and child['returncode'] == 0 and child['resume'] is False,
             'A smoke child was not actually completed with the authorized identity')
        _, _, hashes = completion(row, 'smoke', binding)
        need(all(child[key] == value for key, value in hashes.items()), 'Smoke child artifacts changed after wrapper completion')
    need(children[0]['ended_unix'] <= children[1]['started_unix'], 'Smoke arms were not sequential')
    checks = paired_smoke(rows, binding)
    need(checks == value['paired_smoke_checks'], 'Smoke wrapper omitted/changed the actual paired checks')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'paired_smoke_checks': checks}


def full_job_gate(path, variant, smoke, available_seconds):
    value = load(path)
    need(value.get('schema') == 'stage07_item_local_joint_budget_gate_v1' and value.get('campaign_id') == CAMPAIGN
         and value.get('variant') == variant and value.get('status') == 'passed' and value.get('full_job') is True
         and value.get('smoke_receipt_sha256') == smoke['sha256'] and value.get('resource_policy_compatible') is True,
         'Formal requires a passed full-Job budget/resource-policy gate bound to the accepted smoke')
    need(positive(value['required_process_seconds']) <= available_seconds, 'Predicted complete Job does not fit this process budget')
    digest(value['policy_sha256'])
    need(isinstance(value['reason'], str) and value['reason'].strip(), 'Parent policy/forecast reasoning is missing')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'evidence': value,
            'scope': 'Parent full-Job forecast and current-policy assessment; wrapper does not certify GPU utilisation'}


def child_guard(row, mode, resume, binding):
    work = Path(row['output']) / 'smoke' if mode == 'smoke' else Path(row['output'])
    if mode == 'smoke':
        need(not resume and not any((work / name).exists() for name in ('STARTED.json', 'COMPLETE.json', 'training_state/last.pt')),
             'Smoke is a unique fresh acceptance, never a resumed or duplicated probe')
        return False
    need(not ((work / 'TEST_STARTED.json').exists() and not (work / 'TEST_RESULT.json').exists()),
         'Final test started without a result; refusing a duplicate test')
    if (work / 'COMPLETE.json').exists():
        completion(row, mode, binding)
        return True
    exists = (work / 'STARTED.json').exists() or (work / 'training_state/last.pt').exists()
    need(not exists or (resume and (work / 'training_state/last.pt').is_file()), 'Existing formal execution requires explicit own-last recovery')
    need(not resume or (work / 'training_state/last.pt').is_file(), 'Recovery has no own committed last.pt')
    if exists:
        identity = load(work / 'identity.json')
        need(identity['run_id'] == row['run_id'] and identity['campaign_id'] == CAMPAIGN
             and identity['variant'] == row['variant'] and identity['study_sha256'] == row['study_sha256'], 'Recovery identity is outside this campaign run')
    return False


def formal_matches_smoke(row, result, smoke):
    expected = smoke['paired_smoke_checks']['variants'][row['variant']]
    need(all(result[field] == expected[field] for field in ('initial_base_state_sha256', 'initial_rng_sha256')),
         'Formal initialization/RNG differs from accepted smoke')
    initial = load(Path(row['output']) / 'initial_state_identity.json')
    need(initial['initial_SID_sha256'] == expected['initial_SID_sha256']
         and load(Path(row['output']) / 'input_identities.json')['artifacts'] == expected['input_artifacts'],
         'Formal initial SID/warm-start input identities differ from accepted smoke')


def release_gate(path, goal, variant, attempt, binding):
    value = load(path)
    previous = value['previous_attempt']
    need(type(previous) is int and 1 <= previous < attempt, 'Recovery needs a new attempt after an identified previous attempt')
    old_path = goal / 'delivery' / f'formal_{variant}_attempt{previous}' / 'receipt.json'
    old = load(old_path)
    need(value['campaign_id'] == CAMPAIGN and value['variant'] == variant and value['status'] == 'passed'
         and value['job_terminal'] is True and value['gpu_released'] is True
         and value['previous_receipt_sha256'] == sha(old_path)
         and old['source_binding'] == binding and old['mode'] == 'formal'
         and old['children'][0]['variant'] == variant, 'Prior Job release evidence is missing or bound to a different execution')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'previous_attempt': previous,
            'scope': 'Caller platform lifecycle attestation; process exit alone does not prove release'}


def execute(args):
    goal = args.goal_root.resolve()
    budget(args.prior_allocated_seconds, args.max_seconds, args.platform_max_seconds, args.mode)
    need(args.attempt >= 1 and (not args.resume or args.attempt > 1), 'Recovery requires a new explicit attempt')
    need((args.mode == 'formal') == (args.variant is not None), 'Formal selects one variant; smoke always runs both in order')
    need(args.mode == 'formal' or not args.resume, 'Smoke cannot resume')
    reserve = args.training_reserve_seconds if args.training_reserve_seconds is not None else (60.0 if args.mode == 'smoke' else 1200.0)
    need(math.isfinite(reserve) and 0 <= reserve < args.max_seconds - 90, 'Invalid training/evaluation reserve')
    phase_name = f'smoke_attempt{args.attempt}' if args.mode == 'smoke' else f'formal_{args.variant}_attempt{args.attempt}'
    phase = goal / 'delivery' / phase_name
    phase.mkdir(parents=True, exist_ok=False)
    receipt_path = phase / 'receipt.json'
    started = time.time(); deadline = started + args.max_seconds - 90
    record = {'schema': 'stage07_item_local_joint_job_phase_v1', 'campaign_id': CAMPAIGN, 'mode': args.mode,
              'variant': args.variant, 'attempt': args.attempt, 'status': 'starting', 'pid': os.getpid(),
              'started_utc': utc(), 'started_unix': started, 'deadline_unix': deadline,
              'allocated_gpu_count': 1, 'max_process_seconds': args.max_seconds,
              'declared_platform_max_seconds': args.platform_max_seconds,
              'prior_allocated_seconds_declared': args.prior_allocated_seconds, 'cumulative_budget_seconds': TOTAL_SECONDS,
              'training_reserve_seconds': reserve, 'children': [], 'console': str(phase / 'console.log'), 'goal_complete': False,
              'time_scope': 'Process wall time and caller-declared caps; actual allocation/release require separate platform lifecycle evidence',
              'execution_scope': 'Sequential two-arm smoke, or one complete formal model; no planned segmentation or automatic resubmission'}
    write_receipt(receipt_path, record)
    process, reader = None, None
    groups, stop = set(), {'reason': None, 'at': None}
    sample_stop = threading.Event(); sampler = None; previous_signals = {}; lock = threading.Lock()

    def live_groups():
        for pid in list(groups):
            try: os.killpg(pid, 0)
            except ProcessLookupError: groups.discard(pid)
        return list(groups)

    def send(number):
        for pid in live_groups():
            try: os.killpg(pid, number)
            except ProcessLookupError: groups.discard(pid)

    def terminate(reason):
        if stop['reason'] is None: stop.update(reason=reason, at=time.time())
        send(signal.SIGTERM)

    with (phase / 'console.log').open('x', encoding='utf-8') as console:
        def emit(label, line):
            message = '[' + label + '] ' + line.rstrip('\r\n') + '\n'
            with lock:
                console.write(message); console.flush()
                try: sys.stdout.write(message); sys.stdout.flush()
                except (BrokenPipeError, OSError): pass

        def stream(label, child):
            for line in child.stdout: emit(label, line)
            child.stdout.close()

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous_signals[number] = signal.signal(number, lambda number, _frame: terminate('received_signal_' + str(number)))
            binding = record['source_binding'] = source_binding(goal, args.expected_root_commit, args.expected_engine_commit)
            rows = study_rows(goal)
            environment, record['platform_launcher_env'] = single_process_environment(os.environ)
            environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
            if args.mode == 'formal':
                need(args.smoke_evidence is not None and args.budget_evidence is not None, 'Formal requires --smoke-evidence and --budget-evidence')
                record['smoke_evidence'] = smoke_gate(args.smoke_evidence, rows, binding)
                record['budget_evidence'] = full_job_gate(args.budget_evidence, args.variant, record['smoke_evidence'], args.max_seconds - 90)
                if args.resume:
                    need(args.release_evidence is not None, 'Recovery requires caller evidence that the previous Job terminated and GPU was released')
                    record['release_evidence'] = release_gate(args.release_evidence, goal, args.variant, args.attempt, binding)
                rows = [row for row in rows if row['variant'] == args.variant]
            for row in rows:
                skipped = child_guard(row, args.mode, args.resume, binding)
                need(args.mode != 'formal' or args.attempt == 1 or args.resume or skipped,
                     'A later formal attempt must recover a real interruption, never plan a split run')
                argv = [sys.executable, '-u', '-B', str(goal / 'code/scripts/train/run_item_local_joint.py'),
                        '--mode', args.mode, '--config', row['config'], '--variant', row['variant'], '--output', row['output'],
                        '--project-root', args.project_root, '--engine-root', str(goal / 'engine'),
                        '--expected-engine-commit', args.expected_engine_commit, '--deadline-unix', str(deadline),
                        '--training-deadline-reserve-seconds', str(reserve)]
                if args.resume: argv.append('--resume')
                record['children'].append({**row, 'argv': argv, 'resume': args.resume,
                                           'status': 'skipped_complete' if skipped else 'pending', 'returncode': 0 if skipped else None})
            record['gpu_sampling'] = {key: None for key in ('status', 'interval_seconds', 'command', 'valid_samples', 'failed_samples',
                'csv', 'scope', 'error', 'mean_utilization_gpu_percent', 'minimum_utilization_gpu_percent',
                'maximum_utilization_gpu_percent', 'fraction_samples_below_40_percent', 'csv_sha256')}
            sampler = threading.Thread(target=gpu_sampler, args=(phase / 'gpu_samples.csv', environment, sample_stop, record['gpu_sampling']), daemon=True)
            sampler.start(); record['status'] = 'running'; write_receipt(receipt_path, record)
            for row in record['children']:
                if row['status'] == 'skipped_complete':
                    _, result, hashes = completion(row, args.mode, binding)
                    row.update(hashes)
                    formal_matches_smoke(row, result, record['smoke_evidence'])
                    continue
                need(stop['reason'] is None and time.time() < deadline - reserve, 'Insufficient remaining budget to start the next child')
                process = subprocess.Popen(row['argv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1,
                                           env=environment, start_new_session=True)
                groups.add(process.pid)
                row.update(status='running', pid=process.pid, started_unix=time.time(), started_utc=utc())
                reader = threading.Thread(target=stream, args=(row['variant'], process), daemon=True); reader.start()
                write_receipt(receipt_path, record)
                while process.poll() is None:
                    if time.time() >= deadline + 30: terminate('process_wall_deadline')
                    if stop['at'] is not None and time.time() >= stop['at'] + 15: send(signal.SIGKILL)
                    need(stop['at'] is None or time.time() < stop['at'] + 22, 'Owned child did not exit after bounded TERM/KILL cleanup')
                    time.sleep(.2)
                row.update(returncode=process.returncode, ended_unix=time.time(), ended_utc=utc(), status='failed_or_incomplete')
                need(stop['reason'] is None and process.returncode == 0, 'Child failed or stopped incomplete')
                _, result, hashes = completion(row, args.mode, binding)
                row.update(hashes)
                if args.mode == 'formal':
                    formal_matches_smoke(row, result, record['smoke_evidence'])
                row['status'] = 'completed'; write_receipt(receipt_path, record)
                # A finished leader must not leave a worker group/stdout behind
                # before another sequential arm can start.
                if live_groups(): send(signal.SIGTERM)
                reader.join(timeout=2)
                if live_groups(): send(signal.SIGKILL)
                reader.join(timeout=2)
                need(not live_groups() and not reader.is_alive(), 'Child workers/stdout remain after completion')
            if args.mode == 'smoke': record['paired_smoke_checks'] = paired_smoke(rows, binding)
            record['status'] = args.mode + '_completed'
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            terminate('wrapper_exception')
        finally:
            if live_groups(): terminate(stop['reason'] or 'wrapper_cleanup')
            if process is not None and process.poll() is None:
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: record['cleanup_error'] = 'Owned child survived SIGKILL'
            if live_groups(): send(signal.SIGKILL)
            if reader is not None: reader.join(timeout=3)
            sample_stop.set()
            if sampler is not None: sampler.join(timeout=6)
            alive = live_groups()
            record['owned_process_groups_alive_after_cleanup'] = alive
            record['stdout_reader_alive_after_cleanup'] = bool(reader and reader.is_alive())
            if alive or record['stdout_reader_alive_after_cleanup'] or (sampler and sampler.is_alive()):
                record['status'] = 'failed_or_incomplete'
            if record['status'] not in ('smoke_completed', 'formal_completed'): record['status'] = 'failed_or_incomplete'
            for row in record['children']:
                if row['status'] == 'running': row.update(status='failed_or_incomplete', returncode=None if process is None else process.returncode)
            record.update(stop_reason=stop['reason'], ended_unix=time.time(), ended_utc=utc(), process_wall_seconds=time.time() - started)
            write_receipt(receipt_path, record)
            emit('wrapper', json.dumps({'status': record['status'], 'receipt': str(receipt_path)}, ensure_ascii=False))
            for number, previous in previous_signals.items(): signal.signal(number, previous)
    return 0 if record['status'] in ('smoke_completed', 'formal_completed') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, help='Formal only: one complete model per Job')
    parser.add_argument('--expected-root-commit', required=True)
    parser.add_argument('--expected-engine-commit', required=True)
    parser.add_argument('--max-seconds', type=float, help='Smoke default 900; formal requires an explicit full-Job cap')
    parser.add_argument('--platform-max-seconds', type=float, help='Smoke default 1100; formal requires explicit allocation cap')
    parser.add_argument('--prior-allocated-seconds', type=float, required=True)
    parser.add_argument('--training-reserve-seconds', type=float)
    parser.add_argument('--project-root', default=HISTORICAL)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-evidence', type=Path, help='Successful two-arm smoke wrapper receipt')
    parser.add_argument('--budget-evidence', type=Path, help='Passed parent full-Job forecast/current-resource-policy gate')
    parser.add_argument('--release-evidence', type=Path, help='Resume only: caller-attested prior same-variant Job terminal/GPU release receipt')
    args = parser.parse_args(argv)
    if args.mode == 'smoke':
        args.max_seconds = 900 if args.max_seconds is None else args.max_seconds
        args.platform_max_seconds = 1100 if args.platform_max_seconds is None else args.platform_max_seconds
    elif args.max_seconds is None or args.platform_max_seconds is None:
        parser.error('Formal requires explicit --max-seconds and --platform-max-seconds for the full Job')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
