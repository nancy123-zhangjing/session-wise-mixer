"""Audit the four already-submitted GPU allocations using saved platform evidence.

Offline only. This is accounting/release evidence, not a billing or model audit.
The previous ledger and original raw receipts are preserved before updating.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_r1/delivery/final_lifecycle_r1/'
TZ = timezone(timedelta(hours=8))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def stamp(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)


def epoch(value):
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc)


def seconds(start, end):
    return int((end - start).total_seconds())


def has(events, reason, when):
    return any(e['reason'] == reason and e['time'] == when for e in events)


original = HERE / 'GPU_LEDGER_BEFORE_FINAL.json'
if not original.exists():
    shutil.copyfile(ROOT / 'GPU_LEDGER.json', original)
ledger = read(original)
for name in ['PREFLIGHT_EVENTS.json', 'PREFLIGHT_STATUS.json', 'FORMAL_ALLOCATION_EVENTS.json',
             'A_RELEASE_EVENTS.json', 'B_RELEASE_EVENTS.json', 'STATUS_20261029T1605.json',
             'STATUS_20261029T190527.json']:
    dest = HERE / name
    if not dest.exists():
        shutil.copyfile(ROOT / name, dest)
    assert sha(dest) == sha(ROOT / name), name

status = read(HERE / 'STATUS.json')
query = read(HERE / 'STATUS_QUERY.json')
events_doc = read(HERE / 'EVENTS.json')
assert status['success'] and events_doc['success']
assert not status['data']['unresolved']
current = {row['name']: row for row in status['data']['items']}
assert len(current) == 3
preflight_status = read(HERE / 'PREFLIGHT_STATUS.json')
assert preflight_status['success']
current[preflight_status['data']['name']] = preflight_status['data']
formal_alloc = read(HERE / 'FORMAL_ALLOCATION_EVENTS.json')['events']
all_events = events_doc['data']['items']
sources = {
    'preflight': 'PREFLIGHT_EVENTS.json',
    'A': 'A_RELEASE_EVENTS.json',
    'B': 'B_RELEASE_EVENTS.json',
    'C': 'EVENTS.json',
}
endpoints = {
    'preflight': ('2026-10-29 13:59:12', '2026-10-29 13:59:16'),
    'A': ('2026-10-29 19:03:01', '2026-10-29 19:03:01'),
    'B': ('2026-10-29 16:03:28', '2026-10-29 16:03:38'),
    'C': ('2026-10-30 05:00:43', '2026-10-30 05:00:50'),
}
roles = ['preflight', 'A', 'B', 'C']
review_rows = []
for role, job in zip(roles, ledger['jobs']):
    name = job['name']
    actual = current[name]
    assert actual['status'] == 'SUCCEEDED'
    assert actual['resource']['gpu'] == 1 and actual['resource']['nodes'] == 1
    original_events = read(HERE / sources[role])['data']['items']
    if role == 'C':
        original_events = [row for row in original_events if row['job'] == name]
    allocation = original_events if role == 'preflight' else formal_alloc[role]
    scheduled = [row for row in allocation if row['reason'] == 'Scheduled']
    assert len(scheduled) == 1
    scheduled_dt = stamp(scheduled[0]['time'])
    pod_end, teardown_end = endpoints[role]
    assert has(original_events, 'SuccessfulDeletePod', pod_end)
    assert has(original_events, 'JobTerminated', teardown_end)
    assert any(row['reason'] == 'ExitedWithCode' and 'exited with code 0' in row['message']
               for row in original_events)
    # Exactly one creation in the available lifetime evidence, and none after deletion.
    lifetime = original_events if role == 'preflight' else [row for row in all_events if row['job'] == name]
    creates = [row for row in lifetime if row['reason'] == 'SuccessfulCreatePod']
    assert len(creates) == 1
    assert not any(row['reason'] in ('SuccessfulCreatePod', 'Scheduled', 'Started')
                   and stamp(row['time']) > stamp(pod_end) for row in lifetime)
    ignored = [row for row in lifetime if row['reason'] == 'JobTerminated'
               and stamp(row['time']) > stamp(teardown_end)]
    gpu_sec = seconds(scheduled_dt, stamp(teardown_end))
    assert gpu_sec <= job['platform_max_hours'] * 3600
    pod_sec = seconds(scheduled_dt, stamp(pod_end))
    finished = epoch(actual['finished_at'])
    created = epoch(actual['created_at'])
    if role != 'C':
        assert gpu_sec == job['gpu_seconds'], (role, gpu_sec)
        assert job['release_event_sha256'] == sha(HERE / sources[role])
    else:
        job['prior_running_snapshot'] = {
            key: job.pop(key) for key in list(job)
            if key.startswith('observed_') or key.startswith('allocation_evidence')
        }
    job.update(
        status='SUCCEEDED_RELEASED', gpu_seconds=gpu_sec, gpu_hours=gpu_sec / 3600,
        scheduled_local=scheduled_dt.strftime('%Y-%m-%d %H:%M:%S'),
        scheduled_utc=scheduled_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        terminated_local=teardown_end,
        terminated_utc=stamp(teardown_end).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        successful_delete_pod_local=pod_end,
        scheduled_to_successful_delete_pod_gpu_seconds=pod_sec,
        teardown_accounting_margin_seconds=gpu_sec - pod_sec,
        allocation_basis='Scheduled to completion of the original contemporaneous teardown, multiplied by GPUs; later controller-only duplicate termination events do not extend the allocation.',
        release_evidence=f'SuccessfulDeletePod {pod_end}; original teardown JobTerminated {teardown_end}; Completed exit code 0; platform SUCCEEDED',
        release_event_source='final_lifecycle_r1/' + sources[role],
        release_event_sha256=sha(HERE / sources[role]),
        platform_status_source='final_lifecycle_r1/' + ('PREFLIGHT_STATUS.json' if role == 'preflight' else 'STATUS.json'),
        platform_status_sha256=sha(HERE / ('PREFLIGHT_STATUS.json' if role == 'preflight' else 'STATUS.json')),
        exit_code_zero_evidence=True,
        platform_created_utc=created.strftime('%Y-%m-%d %H:%M:%S UTC'),
        platform_finished_utc=finished.strftime('%Y-%m-%d %H:%M:%S UTC'),
        scheduled_to_platform_finished_gpu_seconds=seconds(scheduled_dt, finished),
        scheduled_to_platform_finished_gpu_hours=seconds(scheduled_dt, finished) / 3600,
        platform_created_to_finished_gpu_seconds=seconds(created, finished),
        platform_created_to_finished_gpu_hours=seconds(created, finished) / 3600,
        platform_finished_basis_note='Explicit alternative timestamp spans, neither verified billed GPU duration.',
        later_controller_only_termination_times_local=[row['time'] for row in ignored],
        no_post_delete_allocation_in_available_events=True,
    )
    review_rows.append(dict(role=role, name=name, scheduled_local=job['scheduled_local'],
        successful_delete_pod_local=pod_end, accounting_end_local=teardown_end,
        gpu_seconds=gpu_sec, gpu_hours=gpu_sec / 3600,
        scheduled_to_pod_delete_gpu_seconds=pod_sec,
        platform_created_utc=job['platform_created_utc'],
        platform_finished_utc=job['platform_finished_utc'],
        scheduled_to_platform_finished_gpu_seconds=job['scheduled_to_platform_finished_gpu_seconds'],
        created_to_platform_finished_gpu_seconds=job['platform_created_to_finished_gpu_seconds'],
        later_controller_only_termination_times_local=job['later_controller_only_termination_times_local']))

total = sum(row['gpu_seconds'] for row in ledger['jobs'])
assert total == 78518 and total <= 108000
assert ledger['jobs'][-1]['gpu_seconds'] == 53600
assert ledger['jobs'][2]['later_controller_only_termination_times_local'] == ['2026-10-29 18:27:11', '2026-10-30 04:18:00']
ledger['prior_incomplete_accounting_snapshot'] = {
    key: ledger[key] for key in ['accounting_snapshot_utc', 'accounting_query_end_utc',
        'cumulative_gpu_seconds', 'cumulative_gpu_hours', 'remaining_authorized_gpu_seconds_at_snapshot',
        'remaining_authorized_gpu_hours_at_snapshot', 'running_observed_allocation_gpu_seconds']
}
checked = datetime.fromisoformat(query['ended_utc'].replace('Z', '+00:00'))
ledger.update(
    format='freeze35_gpu_ledger_v2_final_allocation',
    accounting='Four known 1-H100 allocations including preflight. Conservative original Scheduled-to-teardown accounting; no later duplicate-controller extension. All four succeeded and pods were deleted. This is not verified platform billing.',
    cumulative_gpu_seconds=total, cumulative_gpu_hours=total / 3600,
    completed_allocation_gpu_seconds=total, completed_allocation_gpu_hours=total / 3600,
    running_observed_allocation_gpu_seconds=0, cumulative_is_final=True,
    all_goal_jobs_released=True,
    accounting_snapshot_utc=query['started_utc'], accounting_query_end_utc=query['ended_utc'],
    accounting_snapshot_local=datetime.fromisoformat(query['started_utc'].replace('Z', '+00:00')).astimezone(TZ).isoformat(),
    accounting_snapshot_precision='Actual final platform status query interval; event times are Asia/Shanghai seconds. Allocation totals use completed event spans, not query wall clock.',
    ledger_updated_utc=datetime.now(timezone.utc).isoformat(),
    remaining_authorized_gpu_seconds_at_snapshot=108000-total,
    remaining_authorized_gpu_hours_at_snapshot=(108000-total)/3600,
    remaining_note='Final arithmetic unused budget only; no new training, seed, inference, or GPU allocation is authorized by this remainder.',
    release_review_source='final_lifecycle_r1/ACCOUNTING_REVIEW.md',
    settlement_verified=False,
    evidence_scope='Only the four explicitly authorized allocations; receipts show one pod creation per job and no post-delete recreation. Not a platform-wide account inventory.',
)
for prefix, key in [('completed_platform_scheduled_to_finished', 'scheduled_to_platform_finished_gpu_seconds'),
                    ('completed_platform_created_to_finished', 'platform_created_to_finished_gpu_seconds'),
                    ('completed_scheduled_to_pod_delete', 'scheduled_to_successful_delete_pod_gpu_seconds')]:
    value = sum(job[key] for job in ledger['jobs'])
    ledger[prefix + '_gpu_seconds'] = value
    ledger[prefix + '_gpu_hours'] = value / 3600
ledger['price_estimate'] = dict(recorded_points_per_h100_gpu_hour=1,
    completed_allocation_estimated_points=total / 3600,
    final_allocation_estimated_points=total / 3600, settlement_verified=False,
    note='Historical recorded unit rate times completed accounting hours only; no attributable platform billing statement verified.')
save(ROOT / 'GPU_LEDGER.json', ledger)

evidence_names = ['STATUS.json', 'STATUS_QUERY.json', 'EVENTS.json', 'PREFLIGHT_EVENTS.json',
    'PREFLIGHT_STATUS.json', 'FORMAL_ALLOCATION_EVENTS.json', 'A_RELEASE_EVENTS.json',
    'B_RELEASE_EVENTS.json', 'STATUS_20261029T1605.json', 'STATUS_20261029T190527.json']
evidence = [dict(path=REMOTE+name, sha256=sha(HERE/name)) for name in evidence_names]
release = dict(format='freeze35_final_release_v1', status='verified',
    all_goal_jobs_released=True, checked_utc=query['ended_utc'],
    status_query_interval_utc=dict(start=query['started_utc'], end=query['ended_utc']),
    evidence_timezone='Asia/Shanghai (+08:00)', evidence=evidence,
    allocations=review_rows, known_allocation_count=4, formal_job_count=3,
    all_known_allocations_exit_code_zero=True, total_gpu_seconds=total,
    total_gpu_hours=total/3600, authorized_cumulative_gpu_hours=30,
    includes_preflight_gpu_seconds=219, budget_within_authorized=True,
    accounting_basis=ledger['accounting'], settlement_verified=False,
    resource_release_scope='Named original preflight plus A/B/C GPU job pods; CPU notebook is not a GPU allocation. Model/test/backup acceptance remains separate.',
    no_new_allocation_in_available_lifetime_events=True,
    duplicate_controller_event_note='B retained its originally proven 16:03:38 teardown boundary. The later 18:27:11 and next-day 04:18:00 JobTerminated events contain no new Pod creation/start/allocation. C uses the 05:00:50 completion of the same 05:00:43 teardown (7-second conservative margin).',
    remote_evidence_availability='Prepared locally; parent must upload exact bytes to each declared path and verify SHA before pack. This field does not assert upload has occurred.')
save(HERE/'RELEASE.json', release)

lines = ['# 最终 GPU 释放与用量复核', '',
    '原短测及 A/B/C 四次已知单卡 H100 分配均有 SUCCEEDED、退出码 0 和 SuccessfulDeletePod 证据。',
    f'累计保守分配口径为 **{total:,} GPU 秒 = {total/3600:.9f} GPU 小时**，含短测 219 秒，低于累计 30 GPU 小时上限；算术余量 {(108000-total)/3600:.9f} 小时不构成新作业授权。', '',
    f'最新正式作业状态查询区间：{query["started_utc"]} 至 {query["ended_utc"]}（北京时间 {checked.astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")} 返回）。事件文本时间按 Asia/Shanghai +08:00 解释。', '',
    '| 分配 | Scheduled（北京时间） | Pod 删除 | 原次收尾边界 | 计入 GPU 秒 | GPU 小时 |',
    '|---|---|---|---|---:|---:|']
for row in review_rows:
    lines.append(f'| {row["role"]} | {row["scheduled_local"]} | {row["successful_delete_pod_local"]} | {row["accounting_end_local"]} | {row["gpu_seconds"]:,} | {row["gpu_hours"]:.9f} |')
lines += ['', '口径说明：以保存的真实 Scheduled 事件作为分配起点，保留同一次退出、删 Pod 和收尾的保守边界。短测、A、B 既有秒数保持不变；C 为 53,600 秒。该口径包括 Scheduled 到退出收尾的全部间隔；未因中途网络不可见而减扣耗时。四次分配的证据均只出现一次建 Pod，没有可见的失败重建或重提分配；审计范围是这四个明确命名作业，不是整个平台账号的所有资源。', '',
    'B 在 10 月 29 日 16:03:19 成功退出，16:03:28 删除 Pod，16:03:38 已完成该次收尾。新查询中又出现 18:27:11 和次日 04:18:00 的 JobTerminated（后者伴随 PodGroup 删除），但没有新的 Pod 创建、Scheduled 或 Started，且 platform finished_at 仍是 16:03:19。它们不证明 GPU 重新分配，因此不把任意最新控制器事件当成计费终点。原台账“LAST JobTerminated”文字现已收紧为“原次同时段收尾完成”，以免误读。', '',
    'C 在 10 月 30 日 05:00:36 平台标记成功，05:00:43 删除 Pod，05:00:50 同次重复收尾结束。保守核算多计删除后的 7 秒；它与 B 数小时后没有新 Pod 的控制器重报不混为一谈。', '',
    '| 替代时间口径（四次合计） | 秒 | 小时 |', '|---|---:|---:|']
for label, prefix in [('Scheduled 到实际 Pod 删除', 'completed_scheduled_to_pod_delete'),
                      ('Scheduled 到原次收尾（主台账）', 'completed_allocation'),
                      ('Scheduled 到 platform finished_at', 'completed_platform_scheduled_to_finished'),
                      ('platform created_at 到 finished_at', 'completed_platform_created_to_finished')]:
    lines.append(f'| {label} | {ledger[prefix+"_gpu_seconds"]:,} | {ledger[prefix+"_gpu_hours"]:.9f} |')
lines += ['', '以上都不是已核实的账单。按先前记录的 1 点/H100 小时仅可估算约 21.810556 点；未获得可归属的实际结算记录，settlement_verified 保持 false。资源释放通过不等于模型、逐样本测试、完整备份或恢复已验收。', '',
    '原始证据按字节复制到本目录，上传清单为 UPLOAD_MANIFEST.json。RELEASE.json 的 evidence 使用指定远端绝对路径；当前只准备本地文件，根任务必须上传后逐 SHA 核验才能打包。GPU_LEDGER_BEFORE_FINAL.json 保留更新前原字节，最新台账另保留 prior_incomplete_accounting_snapshot。', '',
    '| 证据文件 | SHA-256 |', '|---|---|']
for name in evidence_names:
    lines.append(f'| {name} | {sha(HERE/name)} |')
lines += ['', f'更新后 GPU_LEDGER.json SHA-256：`{sha(ROOT / "GPU_LEDGER.json")}`。',
          f'RELEASE.json SHA-256：`{sha(HERE / "RELEASE.json")}`。', '']
(HERE/'ACCOUNTING_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
upload_names = evidence_names + ['RELEASE.json', 'ACCOUNTING_REVIEW.md', 'GPU_LEDGER_BEFORE_FINAL.json', 'review_accounting.py']
manifest = dict(format='freeze35_lifecycle_upload_manifest_v1', status='local_prepared_not_uploaded',
    remote_directory=REMOTE, files=[dict(local_path=str(HERE/name), remote_path=REMOTE+name,
        bytes=(HERE/name).stat().st_size, sha256=sha(HERE/name)) for name in upload_names])
manifest['files'].append(dict(local_path=str(ROOT/'GPU_LEDGER.json'), remote_path=REMOTE+'GPU_LEDGER.json',
    bytes=(ROOT/'GPU_LEDGER.json').stat().st_size, sha256=sha(ROOT/'GPU_LEDGER.json')))
save(HERE/'UPLOAD_MANIFEST.json', manifest)
print(json.dumps(dict(status='verified_local', gpu_seconds=total, gpu_hours=total/3600,
    release_sha256=sha(HERE/'RELEASE.json'), ledger_sha256=sha(ROOT/'GPU_LEDGER.json'),
    upload_files=len(manifest['files']), settlement_verified=False), ensure_ascii=False))
