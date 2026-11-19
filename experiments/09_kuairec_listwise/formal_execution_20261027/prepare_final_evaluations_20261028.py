"""Bind final evaluation templates to accepted real checkpoints and native usage."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
FORMAL = ROOT / 'experiments/09_kuairec_listwise/formal_execution_20261027'
PREP = ROOT / 'experiments/09_kuairec_listwise/preparation_20261027'
ARMS = ('J0', 'J1', 'H0', 'H1')
TZ = timezone(timedelta(hours=8))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def main():
    status_path = FORMAL / 'FORMAL_STATUS_20261028T090148Z.json'
    events_path = FORMAL / 'FORMAL_EVENTS_20261028T090305Z.json'
    freeze_path = FORMAL / 'FOUR_BEST_FREEZE_20261028.json'
    freeze_sha = '4d0a0a17be2fd087db30c3dbbfe815a1d9243983619e7c8d47f1ca6fcd681ae2'
    template_path = HERE / 'FINAL_EVAL_JOB_TEMPLATES.json'
    require(sha(freeze_path) == freeze_sha, 'Four-best freeze changed')
    require(sha(template_path) == '6ee4e5ad2cfc617f8e11179096f868bf7ae4834aaa4e0cfe34f150bd929f2546',
            'Reviewed evaluation template changed')
    freeze, native, events = read(freeze_path), read(status_path), read(events_path)
    require(freeze['status'] == 'passed' and freeze['distinct_full_validations'] == 42
            and all(freeze['arms'][a]['best_state']['stored_model_floating_dtype'] == 'torch.float32'
                    for a in ARMS), 'Four-best acceptance incomplete')
    require(native['success'] and native['data']['status'] == 'SUCCEEDED'
            and native['data']['name'] == 'nancy-stage09-four-arm-20261027-r1'
            and native['data']['resource']['gpu'] == 2 and native['data']['resource']['nodes'] == 1,
            'Formal native status identity differs')
    require(events['success'], 'Native event response failed')
    timeline = events['data']['items']
    parse = lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
    scheduled = sorted({parse(e['time']) for e in timeline if e.get('reason') == 'Scheduled'
                        and e.get('instance') == 'rank=0'})
    deleted = sorted({parse(e['time']) for e in timeline if e.get('reason') == 'SuccessfulDeletePod'})
    require(len(scheduled) == 1 and len(deleted) == 1
            and any(e.get('reason') == 'PyTorchJobSucceeded' for e in timeline), 'Ambiguous release timeline')
    require(int(scheduled[0].timestamp()) == 1793115998
            and datetime.fromtimestamp(int(native['data']['finished_at']) / 1000, TZ) <= deleted[0],
            'Scheduled/finished/deleted clocks inconsistent')
    seconds = int((deleted[0] - scheduled[0]).total_seconds()) * 2
    ledger_path = PREP / 'GPU_USAGE_LEDGER.json'
    ledger_before_sha = sha(ledger_path)
    ledger = read(ledger_path)
    previous = [j for j in ledger['jobs'] if j['name'] != native['data']['name']]
    require(len(previous) == 7 and all(j['resource_released'] for j in previous), 'Earlier resource state differs')
    previous_seconds = sum(j['allocated_gpu_seconds'] for j in previous)
    require(previous_seconds == 13535, 'Earlier verified accounting differs')
    total = previous_seconds + seconds
    audit = dict(schema='stage09_formal_release_accounting_v1', status='PASS',
        checked_at_utc=datetime.now(timezone.utc).isoformat(), job=native['data']['name'],
        native_status_source=status_path.name, native_status_sha256=sha(status_path),
        native_events_source=events_path.name, native_events_sha256=sha(events_path),
        event_query_scope='Exact named Job; merged controller and rank=0 pod events; duplicate timestamps deduplicated',
        scheduled_at=scheduled[0].isoformat(), pod_deleted_at=deleted[0].isoformat(), gpus=2, nodes=1,
        allocated_gpu_seconds=seconds, allocated_gpu_hours=seconds/3600,
        earlier_seven_jobs_gpu_seconds=previous_seconds, cumulative_gpu_seconds=total,
        cumulative_gpu_hours=total/3600, cumulative_authorized_gpu_seconds=270000,
        remaining_gpu_seconds=270000-total, resource_released=True, settlement_verified=False,
        points_per_gpu_hour=1.0, estimated_cumulative_points=total/3600,
        price_source='FINAL_LIVE_QUOTA_20261028T085040Z.json',
        price_source_sha256=sha(FORMAL/'FINAL_LIVE_QUOTA_20261028T085040Z.json'),
        ledger_before_sha256=ledger_before_sha)
    # Use observed full-validation cost without assuming benefit from ending GPU sharing.
    timings = {a: [] for a in ARMS}
    for path in FORMAL.glob('FORMAL_VALIDATION_R*.json'):
        for arm, row in read(path).get('arms', {}).items():
            s = row.get('summary')
            if isinstance(s, dict) and s.get('full_split') and s.get('examples') == 71657:
                timings[arm].append((s['seconds'], path.name))
    profiles_path = ROOT / 'experiments/09_kuairec_listwise/h100_preflight_r2/BUDGET_ESTIMATE.json'
    profiles = read(profiles_path)['measured_profiles']
    caps = dict(J0=2.0, J1=2.0, H0=1.5, H1=1.5)
    estimates = {}
    for a in ARMS:
        full, source = max(timings[a])
        beam256_batch = profiles[a]['decoding']['256']['variable_end_to_end_seconds_per_batch']
        baseline = full * (53194 + 1024) / 71657 + 2 * beam256_batch * 32 + 300
        estimates[a] = dict(observed_full_validation_seconds=full, observation_source=source,
            observation_sha256=sha(FORMAL/source), actual_test_examples=53194, fixed_subset_examples=1024,
            beam256_profile_seconds_per_batch=beam256_batch, beam256_profile_multiplier=2,
            startup_and_output_allowance_seconds=300, estimate_seconds_before_margin=baseline,
            estimate_with_25pct_margin_seconds=baseline*1.25, proposed_cap_hours=caps[a])
        require(baseline*1.25 < caps[a]*3600, 'Cap below conservative estimate')
    reserved = sum(caps.values()) * 3600
    require(total + reserved <= 270000 and reserved <= 8*3600, 'Final evaluations exceed approved/reserved budget')
    templates = read(template_path)
    remote_freeze = templates['root'] + '/delivery/' + freeze_path.name
    replacements = {'__ALL_FOUR_BEST_FREEZE_MANIFEST_ABSOLUTE_PATH__': remote_freeze,
                    '__ALL_FOUR_BEST_FREEZE_MANIFEST_SHA256__': freeze_sha}
    for a in ARMS:
        replacements['__FINAL_BEST_' + a + '_SHA256__'] = freeze['arms'][a]['best']['sha256']
        replacements['__RECHECKED_FINAL_EVAL_MAX_HOURS_' + a + '__'] = str(caps[a])
    def replace(value):
        if isinstance(value, str):
            for before, after in replacements.items():
                value = value.replace(before, after)
            return value
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k,v in value.items()}
        return value
    prepared = replace(templates)
    prepared.update(status='BOUND_NOT_SUBMITTED', all_four_training_complete_verified=True,
        all_four_best_frozen_verified=True, actual_final_checkpoint_dtypes_checked=True,
        specific_final_evaluation_caps_rechecked=True, bound_at_utc=datetime.now(timezone.utc).isoformat(),
        budget=dict(cumulative_used_gpu_seconds=total, reserved_evaluation_gpu_seconds=reserved,
                    cumulative_with_caps_gpu_seconds=total+reserved, remaining_uncommitted_gpu_seconds=270000-total-reserved),
        cost_estimates=estimates, profile_source=str(profiles_path.relative_to(ROOT)),
        profile_sha256=sha(profiles_path), limitations=templates['limitations'] + [
            'Caps are conservative estimates, not measured final duration or settlement.',
            'Beam256 uses a 2x short-profile allowance; trained checkpoint throughput may differ.',
            'Remote output absence and dry-runs must pass before any actual submission.'])
    for j in prepared['jobs']:
        j['status'] = 'BOUND_NOT_SUBMITTED'
        j['max_time_hours'] = caps[j['arm']]
        require(j['gpus'] == 1 and j['nodes'] == 1, 'Unexpected evaluation resource count')
        argv = j['cli_argv_template']
        require(float(argv[argv.index('--max-time') + 1]) == caps[j['arm']], 'Cap not bound in CLI')
        require(j['command_template'] == argv[argv.index('--command')+1], 'Command differs from CLI argv')
        require(len(j['stages']) == 3, 'Expected three evaluation stages')
    require(not re.search(r'__[A-Z0-9_]+__', json.dumps(prepared)), 'Unfilled evaluation placeholder')
    save(FORMAL / 'FORMAL_RELEASE_ACCOUNTING_20261028.json', audit)
    save(FORMAL / 'FINAL_EVAL_BOUND_PLAN_20261028.json', prepared)
    current = next(j for j in ledger['jobs'] if j['name'] == native['data']['name'])
    current.update(status='SUCCEEDED', pod_deleted_at=deleted[0].isoformat(),
        allocated_gpu_seconds=seconds, gpu_hours=seconds/3600, estimated_points=seconds/3600,
        resource_released=True, training_progress_snapshot=dict(all_four_complete=True,
            four_best_receipt='../formal_execution_20261027/' + freeze_path.name,
            four_best_receipt_sha256=freeze_sha, r_updates_per_arm=24255, t_updates_per_arm=400,
            accepted_distinct_validations=42))
    current['source_receipts_sha256'].update({
        '../formal_execution_20261027/'+status_path.name: sha(status_path),
        '../formal_execution_20261027/'+events_path.name: sha(events_path)})
    ledger.update(cumulative_allocated_gpu_seconds=total, completed_jobs_allocated_gpu_seconds=total,
        completed_jobs_gpu_hours=total/3600, remaining_total_gpu_seconds=270000-total,
        remaining_total_gpu_seconds_before_active_jobs=270000-total, active_gpu_jobs=[],
        status='formal_training_complete_final_evaluation_pending', updated_at_utc=datetime.now(timezone.utc).isoformat())
    require(sha(ledger_path) == ledger_before_sha, 'Concurrent ledger edit')
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='BOUND_NOT_SUBMITTED', used_gpu_hours=total/3600,
        final_evaluation_caps_hours=caps, total_with_caps_gpu_hours=(total+reserved)/3600,
        remaining_uncommitted_gpu_hours=(270000-total-reserved)/3600, estimates=estimates), ensure_ascii=False))


if __name__ == '__main__':
    main()
