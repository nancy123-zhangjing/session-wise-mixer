"""Account Stage07 allocations from preserved, single-job platform receipts.

Scheduled-to-Pod-deletion is the allocation proxy, not a billing statement.
Repeated formal receipt arguments account interrupted and recovered jobs in one
budget. The known cancelled first formal allocation is never silently omitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path


PREFIX = 'nancy-stage07-item-local-'
SUFFIX = '20261020-0045'
RUN_ID = 'stage07_item_local_seed2024_20261020_0045'
FORMAL_NAME = PREFIX + 'formal-' + SUFFIX
REPAIR_PREFIX = 'nancy-stage07-ocn-repair-'
JOINT_PREFIX = 'nancy-stage07-item-local-joint-'
LOCAL_ZONE = timezone(timedelta(hours=8))
SMOKES = (
    (PREFIX + 'smoke-' + SUFFIX, 'SMOKE_TERMINAL_20261020.json',
     'SMOKE_R1_EVENTS_FINAL_20261020.json'),
    (PREFIX + 'smoke-' + SUFFIX + '-r2', 'SMOKE_R2_TERMINAL_20261020.json',
     'SMOKE_R2_EVENTS_FINAL_20261020.json'),
)


def load(path):
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'))
    if value.get('success') is not True:
        raise ValueError(f'Unsuccessful platform receipt: {path}')
    return value['data'], {'path': str(path.resolve()),
                           'sha256': hashlib.sha256(raw).hexdigest()}


def stamp(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=LOCAL_ZONE)


def allocated(name, status_path, events_path, start_events_path=None):
    status, status_source = load(status_path)
    events, events_source = load(events_path)
    if status['name'] != name or status['status'] not in {'SUCCEEDED', 'FAILED', 'STOPPED', 'TERMINATED', 'CANCELLED'}:
        raise ValueError(f'Expected the exact terminal job: {name}')
    resource = status['resource']
    if resource.get('gpu') != 1 or resource.get('nodes') != 1 or 'H100' not in resource.get('gpu_type', ''):
        raise ValueError('This accounting applies only to the authorized single H100 jobs')
    event_sources = [events_source]
    items = list(events['items'])
    if start_events_path is not None:
        earlier, earlier_source = load(start_events_path)
        event_sources.append(earlier_source)
        items.extend(earlier['items'])
        # Kubernetes events can expire before a long formal run finishes.
        # Retain the separately captured start receipt and de-duplicate overlap.
        unique = {}
        for row in items:
            key = tuple(row.get(field) for field in ('time', 'instance', 'reason', 'message'))
            unique[key] = row
        items = list(unique.values())
    scheduled = [row for row in items if row['reason'] == 'Scheduled']
    deleted = [row for row in items if row['reason'] == 'SuccessfulDeletePod']
    if len(scheduled) != 1 or len(deleted) != 1:
        raise ValueError('Need one preserved Scheduled and one Pod-deletion event; do not guess missing intervals')
    if scheduled[0].get('instance') != 'rank=0':
        raise ValueError('Unexpected scheduled instance')
    start, end = stamp(scheduled[0]['time']), stamp(deleted[0]['time'])
    seconds = (end - start).total_seconds()
    created = datetime.fromtimestamp(int(status['created_at']) / 1000, timezone.utc)
    finished = datetime.fromtimestamp(int(status['finished_at']) / 1000, timezone.utc)
    # finished_at is the controller's terminal-state timestamp, not the moment
    # GPU allocation is released. It can precede Pod cleanup or lag cleanup
    # (the cancelled first formal job records terminal one second after delete).
    if not created <= start or start > finished or seconds <= 0:
        raise ValueError('Platform timestamps do not describe one coherent allocation')
    return {'job_name': name, 'terminal_status': status['status'], 'gpu_count': 1,
            'scheduled_local': start.isoformat(), 'pod_deleted_local': end.isoformat(),
            'controller_terminal_local': finished.astimezone(LOCAL_ZONE).isoformat(),
            'controller_terminal_minus_pod_deleted_seconds': (finished - end).total_seconds(),
            'allocated_seconds': seconds, 'allocated_gpu_hours': seconds / 3600,
            'created_to_terminal_seconds': (finished - created).total_seconds(),
            'released_by_pod_deletion': True,
            'sources': {'status': status_source, 'events': event_sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts-dir', type=Path,
                        default=Path('experiments/07_item_local_attention/execution_acceptance'))
    parser.add_argument('--formal-status', type=Path, action='append', default=[],
                        help='Terminal formal status receipt; repeat for recovery jobs in matching order')
    parser.add_argument('--formal-events', type=Path, action='append', default=[],
                        help='Terminal formal events receipt; repeat in status-receipt order')
    parser.add_argument('--formal-start-events', type=Path, action='append', default=[],
                        help='Optional preserved start receipts, one per explicit formal job; when omitted, the first job uses FORMAL_EVENTS_START_20261020.json and recovery jobs require Scheduled in their terminal events')
    parser.add_argument('--all-goal-jobs-accounted', action='store_true',
                        help='Declare the supplied inventory contains every Goal GPU job; this does not verify model/Goal completion')
    parser.add_argument('--approved-budget-receipt', type=Path,
                        help='Direct human approval receipt for the cumulative 28-hour repaired pair')
    parser.add_argument('--repair-status', type=Path, action='append', default=[])
    parser.add_argument('--repair-events', type=Path, action='append', default=[])
    parser.add_argument('--repair-start-events', type=Path, action='append', default=[])
    parser.add_argument('--joint-scope-receipt', type=Path)
    parser.add_argument('--joint-status', type=Path, action='append', default=[])
    parser.add_argument('--joint-events', type=Path, action='append', default=[])
    parser.add_argument('--joint-start-events', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.formal_status) != len(args.formal_events):
        parser.error('Provide one formal events receipt for each formal status receipt')
    if args.formal_start_events and len(args.formal_start_events) != len(args.formal_status):
        parser.error('When supplying start receipts, provide one per explicit formal status receipt')
    if len(args.repair_status) != len(args.repair_events):
        parser.error('Provide one repair events receipt per repair status receipt')
    if args.repair_start_events and len(args.repair_start_events) != len(args.repair_status):
        parser.error('Provide one start receipt per explicit repair job, or omit all')
    if len(args.joint_status) != len(args.joint_events):
        parser.error('Provide one joint events receipt per joint status receipt')
    if args.joint_start_events and len(args.joint_start_events) != len(args.joint_status):
        parser.error('Provide one start receipt per explicit joint job, or omit all')
    limit, approval_source = 8.0, None
    if args.approved_budget_receipt:
        raw = args.approved_budget_receipt.read_bytes()
        approval = json.loads(raw.decode('utf-8-sig'))
        if (approval.get('source') != 'Direct human reply to the concrete repair-and-budget question'
                or approval.get('answer') != '同意累计28 GPU小时，按此方案推进（推荐）'
                or approval.get('cumulative_allocated_gpu_hours_limit') != 28
                or approval.get('legacy_allocated_gpu_seconds_included') != 11273
                or approval.get('gpu_count_default') != 1
                or approval.get('new_repaired_control_and_item_local_training_authorized') is not True):
            raise ValueError('The larger budget requires the exact preserved human scope approval')
        limit = 28.0
        approval_source = {'path': str(args.approved_budget_receipt.resolve()),
                           'sha256': hashlib.sha256(raw).hexdigest()}
    if args.repair_status and not approval_source:
        parser.error('Repaired-pair allocation requires its explicit cumulative-budget approval receipt')
    joint_scope_source = None
    if args.joint_scope_receipt:
        raw = args.joint_scope_receipt.read_bytes()
        scope = json.loads(raw.decode('utf-8-sig'))
        if (scope.get('source') != 'Direct human correction in this conversation'
                or scope.get('instructions') != ['我要的是联合训练', '请你按照之前码本交互跑联合训练的方式帮我修改代码，我要联合训练，交替优化那种方式']
                or scope.get('cumulative_allocated_gpu_seconds') != 11637):
            raise ValueError('Joint accounting requires the preserved human scope correction')
        joint_scope_source = {'path': str(args.joint_scope_receipt.resolve()),
                              'sha256': hashlib.sha256(raw).hexdigest()}
    if args.joint_status and not (approval_source and joint_scope_source):
        parser.error('Joint allocation requires the retained cumulative budget and direct joint scope correction')
    rows = [allocated(name, args.receipts_dir / status, args.receipts_dir / events)
            for name, status, events in SMOKES]
    formals = []
    names = set()
    for index, (status_path, events_path) in enumerate(zip(args.formal_status, args.formal_events)):
        status, _ = load(status_path)
        name = status['name']
        if name != FORMAL_NAME and not name.startswith(FORMAL_NAME + '-'):
            raise ValueError('Formal receipt is outside the same authorized run/job-name family')
        if name in names:
            raise ValueError(f'Duplicate formal job would double count allocation: {name}')
        names.add(name)
        if args.formal_start_events:
            start_path = args.formal_start_events[index]
        elif name == FORMAL_NAME:
            start_path = args.receipts_dir / 'FORMAL_EVENTS_START_20261020.json'
        else:
            start_path = None
        formals.append((name, status_path, events_path, start_path))
    # A later successful recovery must not erase the first cancelled allocation.
    # Include its preserved terminal receipts automatically when available, even
    # in a partial/default audit or when only recovery receipts were supplied.
    first_status = args.receipts_dir / 'FORMAL_CANCELLED_STATUS_20261020.json'
    first_events = args.receipts_dir / 'FORMAL_CANCELLED_EVENTS_20261020.json'
    if FORMAL_NAME not in names and first_status.is_file() != first_events.is_file():
        raise ValueError('Known cancelled formal allocation has incomplete receipts; refusing to omit it')
    if FORMAL_NAME not in names and first_status.is_file() and first_events.is_file():
        formals.insert(0, (FORMAL_NAME, first_status, first_events,
                           args.receipts_dir / 'FORMAL_EVENTS_START_20261020.json'))
        names.add(FORMAL_NAME)
    if formals and FORMAL_NAME not in names:
        raise ValueError('Recovery accounting must include the terminal first formal job receipts')
    if args.all_goal_jobs_accounted and not formals:
        parser.error('A complete resource inventory must include the first formal allocation')
    for formal in formals:
        rows.append(allocated(*formal))
    for index, (status_path, events_path) in enumerate(zip(args.repair_status, args.repair_events)):
        status, _ = load(status_path)
        name = status['name']
        if not name.startswith(REPAIR_PREFIX) or '20261020-r1' not in name:
            raise ValueError('Repair receipt is outside this authorized campaign job family')
        if name in names:
            raise ValueError(f'Duplicate job would double count allocation: {name}')
        names.add(name)
        start_path = args.repair_start_events[index] if args.repair_start_events else None
        rows.append(allocated(name, status_path, events_path, start_path))
    # The stopped fixed-SID profile is real spent allocation, even though the
    # human subsequently changed the scientific protocol to joint training.
    profile_name = REPAIR_PREFIX + 'profile-20261020-r1'
    profile_dir = args.receipts_dir / 'ocn_repair_r1'
    profile_status = profile_dir / 'PROFILE_TERMINAL_FOR_JOINT_SCOPE.json'
    profile_events = profile_dir / 'PROFILE_EVENTS_STOP_FOR_JOINT_SCOPE.json'
    if profile_name not in names and profile_status.is_file() != profile_events.is_file():
        raise ValueError('Known stopped profile has incomplete receipts; refusing to omit its allocation')
    if profile_name not in names and profile_status.is_file():
        rows.append(allocated(profile_name, profile_status, profile_events,
                              profile_dir / 'PROFILE_EVENTS_SCHEDULED.json'))
        names.add(profile_name)
    if args.joint_status and profile_name not in names:
        raise ValueError('Joint accounting must include the stopped fixed-SID profile')
    for index, (status_path, events_path) in enumerate(zip(args.joint_status, args.joint_events)):
        status, _ = load(status_path)
        name = status['name']
        if not name.startswith(JOINT_PREFIX) or '20261020-r1' not in name:
            raise ValueError('Joint receipt is outside this authorized campaign job family')
        if name in names:
            raise ValueError(f'Duplicate job would double count allocation: {name}')
        names.add(name)
        start_path = args.joint_start_events[index] if args.joint_start_events else None
        rows.append(allocated(name, status_path, events_path, start_path))
    formal_accounted = bool(formals)
    total = sum(row['allocated_seconds'] for row in rows)
    within = total <= limit * 3600
    result = {
        'status': ('FINAL_ALLOCATION_ACCOUNTED' if args.all_goal_jobs_accounted
                   else 'PARTIAL_ALLOCATION_ACCOUNTED' if formal_accounted
                   else 'SMOKES_ACCOUNTED_FORMAL_PENDING'),
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': RUN_ID,
        'allocation_basis': 'Single-job Scheduled event to SuccessfulDeletePod event; timestamps UTC+08:00',
        'jobs': rows, 'allocated_seconds': total, 'allocated_gpu_hours': total / 3600,
        'limit_gpu_hours': limit, 'within_limit_for_accounted_jobs': within,
        'remaining_gpu_seconds_after_accounted_jobs': max(0, limit * 3600 - total),
        'remaining_gpu_hours_after_accounted_jobs': max(0, limit - total / 3600),
        'formal_job_accounted': formal_accounted,
        'formal_jobs_accounted': [row['job_name'] for row in rows if row['job_name'].startswith(FORMAL_NAME)],
        'repair_jobs_accounted': [row['job_name'] for row in rows if row['job_name'].startswith(REPAIR_PREFIX)],
        'joint_jobs_accounted': [row['job_name'] for row in rows if row['job_name'].startswith(JOINT_PREFIX)],
        'joint_scope_source': joint_scope_source,
        'budget_authorization_source': approval_source,
        'all_goal_gpu_jobs_accounted': args.all_goal_jobs_accounted,
        'complete_resource_inventory_declared_by_caller': args.all_goal_jobs_accounted,
        'all_accounted_allocations_released': True,
        'billing_or_settlement_verified': False, 'goal_complete': False,
    }
    if not formal_accounted:
        result['formal_cap_gpu_hours'] = 7.5
        result['smokes_plus_formal_cap_gpu_hours'] = total / 3600 + 7.5
        result['scope'] = 'Two completed smokes only; not final Goal resource acceptance'
    elif not args.all_goal_jobs_accounted:
        result['scope'] = 'Released jobs listed here only; all later allocations must be added to the same approved cumulative budget; not final Goal resource acceptance'
    else:
        result['scope'] = 'Caller-declared complete GPU-job inventory accounted from release events; model results and Goal completion are not verified'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'allocated_seconds': total,
                      'allocated_gpu_hours': total / 3600,
                      'formal_job_accounted': formal_accounted,
                      'formal_jobs_accounted': result['formal_jobs_accounted'],
                      'all_goal_gpu_jobs_accounted': args.all_goal_jobs_accounted,
                      'remaining_gpu_hours_after_accounted_jobs': result['remaining_gpu_hours_after_accounted_jobs'],
                      'within_limit_for_accounted_jobs': within}, ensure_ascii=False))
    if not within:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
