"""CPU-only second-arm forecast from completed/released control and paired smoke.

Implements SECOND_FORMAL_TIMING_RULE.md without changing training or caps.
Requires the real frozen goal layout on a CPU host. No model is constructed,
no test is run, and no platform request is made. Incomplete or inadmissible
inputs fail without writing a passed budget. Parent reviews timing anomalies,
live policy freshness and the completeness of the supplied allocation ledger.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT_COMMIT = '842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'
CONTROL_JOB = 'nancy-stage07-item-local-joint-formal-control-20261020-r1-r2'
sys.dont_write_bytecode = True


def need(value, message):
    if not value:
        raise ValueError(message)


def positive(value):
    need(not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0, 'Invalid positive timing')
    return float(value)


def phase(epoch):
    return 'tokenizer' if epoch > 5 and (epoch - 6) % 4 == 0 else 'recommender'


def ratios(short):
    factors = {}
    for kind in ('recommender', 'tokenizer'):
        a, b = short['control']['epochs'], short['item_local']['epochs']
        need([r['epoch'] for r in a] == [r['epoch'] for r in b] == [1, 2, 3]
             and all(r['batches'] == 32 for r in a + b), 'Paired smoke epoch/batch geometry differs')
        indices = (1,) if kind == 'tokenizer' else (0, 2)
        factors[kind] = max(1., *(positive(b[i]['seconds']) / positive(a[i]['seconds']) for i in indices))
    a = {r['epoch']: r for r in short['control']['evaluation']}
    b = {r['epoch']: r for r in short['item_local']['evaluation']}
    need(set(a) == set(b) == {0, 1, 3} and all(r['split'] == 'val' for r in [*a.values(), *b.values()]),
         'Paired smoke evaluation calls differ')
    factors['validation'] = max(1., *(positive(b[e]['seconds']) / positive(a[e]['seconds']) for e in a))
    return factors


def complete_timings(result, epochs, evaluations):
    need(result['status'] == 'formal_completed' and result['test_data_evaluated'] is True,
         'Control must have real formal completion and its once-only final test')
    n = result['fit']['completed_epochs']
    need(type(n) is int and 1 <= n <= 45 and [r['epoch'] for r in epochs] == list(range(1, n + 1)),
         'Missing or duplicated completed control logical epochs')
    rec, tok, expected_vals = [], [], [0]
    for row in epochs:
        target = tok if phase(row['epoch']) == 'tokenizer' else rec
        need(row['batches'] == (96 if target is tok else 1652), 'Control phase was truncated or uses another recipe')
        target.append(positive(row['seconds']))
        if target is rec and len(rec) % 5 == 0:
            expected_vals.append(row['epoch'])
    validation = [r for r in evaluations if r['split'] == 'val']
    tests = [r for r in evaluations if r['split'] == 'test']
    need(rec and tok and len(tests) == 1 and evaluations[-1] == tests[0] and tests[0]['epoch'] is None
         and [r['epoch'] for r in validation] == expected_vals and len(evaluations) == len(validation) + 1,
         'Missing/repeated/unplanned full validation or once-only final test timing')
    return rec, tok, [positive(r['seconds']) for r in validation], positive(tests[0]['seconds'])


def forecast(rec, tok, val, test, wrapper_seconds, multiplier, test_prior):
    timed = sum(rec) + sum(tok) + sum(val) + test
    other = positive(wrapper_seconds) - timed
    need(other >= 0, 'Timed components exceed the full wrapper wall time; do not hide overlap or inconsistent clocks')
    other_bound = max(500., other * 1.25)
    components = {'recommender': 35 * max(rec) * multiplier['recommender'],
                  'tokenizer': 10 * max(tok) * multiplier['tokenizer'],
                  'validation': 8 * max(val) * multiplier['validation']}
    training = sum(components.values()) + other_bound
    testing = 2 * max(test, test_prior) * multiplier['validation']
    return {'recommender_epochs_max': 35, 'tokenizer_phases_max': 10, 'full_validation_calls_max': 8,
            'recommender_batches_each': 1652, 'tokenizer_batches_each': 96, 'full_evaluation_batches': 2796,
            'paired_smoke_ratios': multiplier, 'phase_upper_seconds': components,
            'control_observed_counts': {'recommender': len(rec), 'tokenizer': len(tok), 'validation': len(val), 'test': 1},
            'control_phase_seconds': {'recommender': rec, 'tokenizer': tok, 'validation': val, 'test': test},
            'control_wrapper_process_seconds': wrapper_seconds, 'control_timed_components_seconds': timed,
            'control_other_overhead_seconds': other, 'other_overhead_bound_seconds': other_bound,
            'training_including_full_validations_bound_seconds': training,
            'test_beam256_prior_seconds': test_prior, 'test_bound_seconds': testing,
            'required_process_seconds': math.ceil(training + testing),
            'training_window_slack_seconds': 9150 - training, 'test_reserve_slack_seconds': 1200 - testing}


def build(args):
    goal = args.goal_root.resolve(strict=True)
    # Fail on missing formal completion before importing even frozen helper modules.
    need((goal / 'runs/stage07_item_local_joint_control_seed2024_20261020_r1/COMPLETE.json').is_file(),
         'Control is not formally complete; second-arm budget is unavailable')
    path = goal / 'code/scripts/train/item_local_joint_gpu_job.py'
    spec = importlib.util.spec_from_file_location('_frozen_joint_budget_wrapper', path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    binding = wrapper.source_binding(goal, ROOT_COMMIT, ENGINE_COMMIT)
    rows = wrapper.study_rows(goal)
    smoke = wrapper.smoke_gate(goal / 'delivery/smoke_attempt1/receipt.json', rows, binding)
    control = next(r for r in rows if r['variant'] == 'control')
    local = next(r for r in rows if r['variant'] == 'item_local')
    work, result, hashes = wrapper.completion(control, 'formal', binding)
    need(wrapper.child_guard(local, 'formal', False, binding) is False, 'Item-local formal already exists')
    sources = {}
    def read(name, file):
        value = wrapper.load(file)
        sources[name] = {'path': str(Path(file).resolve()), 'sha256': wrapper.sha(file)}
        return value
    parent = read('completed_control_wrapper', args.control_wrapper_receipt)
    need(parent['status'] == 'formal_completed' and parent['mode'] == 'formal' and parent['variant'] == 'control'
         and parent['source_binding'] == binding and len(parent['children']) == 1
         and parent['owned_process_groups_alive_after_cleanup'] == [] and parent['stdout_reader_alive_after_cleanup'] is False,
         'Control wrapper is unfinished, from another source, or has live child processes')
    child = parent['children'][0]
    need(child['variant'] == 'control' and child['run_id'] == control['run_id'] and child['status'] == 'completed'
         and child['returncode'] == 0 and child['resume'] is False and all(child[k] == v for k, v in hashes.items())
         and parent['smoke_evidence'] == smoke, 'Need one complete fresh control Job bound to this paired smoke/results')
    need(abs((positive(parent['ended_unix']) - positive(parent['started_unix'])) - positive(parent['process_wall_seconds'])) < .01,
         'Control full process wall time differs from wrapper timestamps')
    config = read('control_parsed_config', work / 'parsed_config.json')
    need(config['epochs'] == 45 and config['train_batch_size'] == 64 and config['gradient_accumulation_steps'] == 16
         and config['eval_batch_size'] == 8 and config['evaluation_level'] == 'sid'
         and config.get('max_eval_batches') in (None, 0) and config.get('max_train_batches_per_epoch') in (None, 0)
         and config['vectorized_beam_search']['test'] == {'beam_act': 256, 'beam_max': 256},
         'Control full training/evaluation configuration differs')
    epoch_data = read('control_epoch_timings', work / 'epoch_timings.json')
    eval_data = read('control_evaluation_timings', work / 'evaluation_timings.json')
    need(len({r['attempt'] for r in eval_data}) == 1, 'Control timings include multiple execution attempts')
    rec, tok, val, test = complete_timings(result, epoch_data, eval_data)
    short = {}
    for row in rows:
        base = Path(row['output']) / 'smoke'
        short[row['variant']] = {'epochs': read(row['variant'] + '_smoke_epochs', base / 'epoch_timings.json'),
                                'evaluation': read(row['variant'] + '_smoke_evaluation', base / 'evaluation_timings.json')}
    historical = read('historical_validation_beam256_probe', args.historical_beam)
    need(historical['test_data_evaluated'] is False and historical['formal_eval_batches'] == 2796,
         'Historical beam256 reference is not the recorded same-shape validation-only probe')
    test_prior = positive(historical['test_beam256_on_validation_2_batches_seconds']) / 2 * 2796
    need(abs(test_prior - 511.46655656571966) < 1e-6, 'Historical test prior changed')
    calculated = forecast(rec, tok, val, test, parent['process_wall_seconds'], ratios(short), test_prior)
    ledger = read('updated_released_allocation_ledger', args.ledger)
    need(ledger['limit_gpu_hours'] == 28 and ledger['within_limit_for_accounted_jobs'] is True
         and ledger['all_accounted_allocations_released'] is True, 'Updated allocation ledger is not released/within the authorized budget')
    jobs = ledger['jobs']; names = [r['job_name'] for r in jobs]
    need(len(names) == len(set(names)) and abs(sum(positive(r['allocated_seconds']) for r in jobs) - ledger['allocated_seconds']) < 1e-6,
         'Allocation ledger has duplicate jobs or inconsistent totals')
    completed = [r for r in jobs if r['job_name'] == CONTROL_JOB]
    need(len(completed) == 1 and completed[0]['terminal_status'] == 'SUCCEEDED' and completed[0]['gpu_count'] == 1
         and completed[0]['released_by_pod_deletion'] is True, 'Missing successful control Job and actual Pod release in the current ledger')
    scheduled = datetime.fromisoformat(completed[0]['scheduled_local']).timestamp()
    released = datetime.fromisoformat(completed[0]['pod_deleted_local']).timestamp()
    need(scheduled - 1 <= parent['started_unix'] < parent['ended_unix'] <= released + 1
         and ledger['allocated_seconds'] >= 11946 + completed[0]['allocated_seconds'],
         'Released Job does not contain this complete control execution or the ledger omitted earlier allocation')
    wrapper.budget(ledger['allocated_seconds'], 10440, 10620, 'formal')
    policy = read('live_resource_policy', args.policy)
    jobs = [r for r in policy['data']['items'] if r['workload'] == 'job' and r['workspace'] == '分布式训练空间']
    need(policy['success'] is True and len(jobs) == 1 and jobs[0]['auto_reclaim'] is True
         and jobs[0]['reclaim_rule'] == 'gpu < 40% for 3h' and 10620 < 3 * 3600,
         'Current supplied resource policy differs; parent must reassess a full model Job')
    need(args.timing_rule.is_file(), 'Missing the prespecified second-arm timing rule')
    sources['prespecified_timing_rule'] = {'path': str(args.timing_rule.resolve()), 'sha256': wrapper.sha(args.timing_rule)}
    for name in ('COMPLETE.json', 'results.json', 'TEST_STARTED.json', 'TEST_RESULT.json'):
        read('control_' + name, work / name)
    need(calculated['training_including_full_validations_bound_seconds'] < 9150 and calculated['test_bound_seconds'] < 1200
         and calculated['required_process_seconds'] < 10350,
         'Second arm NOT admitted by the prespecified windows: ' + json.dumps(calculated, allow_nan=False))
    return {'schema': 'stage07_item_local_joint_budget_gate_v1', 'campaign_id': wrapper.CAMPAIGN,
            'variant': 'item_local', 'status': 'passed', 'full_job': True, 'smoke_receipt_sha256': smoke['sha256'],
            'policy_sha256': sources['live_resource_policy']['sha256'], 'resource_policy_compatible': True,
            'required_process_seconds': calculated['required_process_seconds'], 'source_binding': binding,
            'reason': 'Completed/released control and paired smoke; prespecified component maxima, measured overhead and conservative test bound fit the original complete-model Job windows. No assumed early stop, planned segmentation or artificial GPU work.',
            'forecast': calculated, 'caps': {'max_process_seconds': 10440, 'platform_max_seconds': 10620,
                'wrapper_deadline_margin_seconds': 90, 'training_reserve_seconds': 1200,
                'training_window_seconds': 9150, 'cumulative_prior_allocated_gpu_seconds': ledger['allocated_seconds'],
                'prior_plus_item_local_platform_gpu_hours': (ledger['allocated_seconds'] + 10620) / 3600},
            'sources': sources, 'tool_sha256': wrapper.sha(__file__),
            'created_utc': datetime.now(timezone.utc).isoformat(), 'goal_complete': False,
            'timer_scope': 'Pinned sequential engine loop timers and evaluate wrapper exclude each other; full wrapper remainder includes initialization, snapshots, SID refresh and cleanup.',
            'limitations': ['Forecast is not a future-runtime guarantee or utilisation improvement.',
                'Parent still checks live-inventory/policy freshness and explains late-phase slowdown/other overhead before submission.',
                'Same 35rec/10tok/8val maximum counts even if control legally stopped early; no cap, recipe or margin reduced.']}, wrapper, smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('goal-root', 'control-wrapper-receipt', 'ledger', 'policy', 'timing-rule', 'historical-beam', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    need(not os.path.lexists(args.output), 'Preserve existing evidence; choose a new output file')
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', ACCELERATE_USE_CPU='true')
    try:
        result, wrapper, smoke = build(args)
        # Exercise the frozen consumer contract before exposing the real passed receipt.
        with tempfile.TemporaryDirectory(prefix='joint_budget_contract_') as directory:
            path = Path(directory) / 'budget.json'; path.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
            wrapper.full_job_gate(path, 'item_local', smoke, 10350)
    except Exception as error:
        print(json.dumps({'status': 'SECOND_FORMAL_NOT_ADMITTED', 'error_type': type(error).__name__,
                          'error': str(error), 'passed_budget_written': False, 'goal_complete': False}))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': str(args.output), 'sha256': wrapper.sha(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
