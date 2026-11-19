"""Recompute released GPU allocation from platform status/events, not process exit.

Input inventories list the exact CLI evidence files for every own campaign Job.
The final audit additionally requires a fresh complete platform Job listing.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

CAMPAIGN = 'goal_20261020_long_mixer_v1'
PREFIX = 'nancy-stage05-long-'
TERMINAL = {'SUCCEEDED','FAILED','CANCELLED'}
BEIJING = timezone(timedelta(hours=8))


def need(value,message):
    if not value: raise ValueError(message)


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(root,name):
    path=(root / name).resolve()
    need(not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_relative_to(root.resolve()), 'Escaped evidence root')
    value=read(path)
    need(value['success'] is True,'Platform query did not succeed')
    return value['data'],sha(path)


def released_job(root,row):
    need(row['job_name'].startswith(PREFIX) and row['mode'] in ('smoke','train_arm','evaluate'), 'Unrecognized campaign Job')
    status,status_sha=evidence(root,row['status_file'])
    need(status['name']==row['job_name'] and status['status'] in TERMINAL,'Missing correct terminal platform status')
    resource=status['resource']
    need(resource['gpu']==resource['nodes']==1 and 'H100' in resource['gpu_type'],'Wrong actual allocation size')
    events=[]; hashes={row['status_file']:status_sha}
    for name in row['event_files']:
        data,digest=evidence(root,name); hashes[name]=digest; events.extend(data['items'])
    starts=[e['time'] for e in events if e['reason']=='Scheduled']
    ends=[e['time'] for e in events if e['reason']=='SuccessfulDeletePod']
    need(starts and ends,'Scheduled and Pod deletion proof are both required')
    parse=lambda value:datetime.strptime(value,'%Y-%m-%d %H:%M:%S').replace(tzinfo=BEIJING)
    start,end=min(map(parse,starts)),max(map(parse,ends))
    need(end>=start,'Deletion predates allocation')
    # Count through Pod deletion, including post-exit retention and any gaps.
    if status.get('finished_at'):
        finished=datetime.fromtimestamp(int(status['finished_at'])/1000,timezone.utc)
        need(end>=finished,'Pod deletion predates terminal process time')
    later_creations=[e for e in events if e['reason'] in ('Scheduled','SuccessfulCreatePod') and parse(e['time'])>end]
    need(not later_creations,'A new allocation appeared after the claimed final deletion')
    return dict(job_name=row['job_name'],mode=row['mode'],released=True,status=status['status'],gpu_count=1,
        started_at_beijing=start.isoformat(),released_at_beijing=end.isoformat(),
        allocated_gpu_seconds=(end-start).total_seconds(),time_basis='Scheduled through SuccessfulDeletePod, including retention',
        platform_evidence_sha256=hashes)


def audit(root,inventory,*,final):
    need(inventory['campaign_id']==CAMPAIGN and inventory['jobs'],'Wrong/empty campaign inventory')
    names=[r['job_name'] for r in inventory['jobs']]
    need(len(names)==len(set(names)),'Duplicate allocation would be double counted')
    jobs=[released_job(root,row) for row in inventory['jobs']]
    total=sum(row['allocated_gpu_seconds'] for row in jobs)
    acceptance=sum(row['allocated_gpu_seconds'] for row in jobs if row['mode']=='smoke')
    need(total<=43200 and acceptance<=1800,'Authorized cumulative GPU allocation exceeded')
    quota,quota_sha=evidence(root,inventory['quota_file'])
    rates=[r['points_per_hour'] for r in quota['items'] if r['workspace']=='分布式训练空间'
           and r['compute_group']=='开发区-H100-cuda13.2版本-183核' and r['quota']=='1,20,200']
    need(len(rates)==1 and rates[0]==1,'Expected actual single-H100 price row missing/different')
    listing_sha=None
    if final:
        listing,listing_sha=evidence(root,inventory['final_job_list_file'])
        actual={r['name']:r for r in listing['items'] if r['name'].startswith(PREFIX)}
        need(set(actual)==set(names),'Final platform inventory differs; an own Job may be omitted')
        need(all(actual[r['job_name']]['status']==r['status'] for r in jobs),'Current platform state is not the released terminal state')
        need(any(r['mode']=='evaluate' and r['status']=='SUCCEEDED' for r in jobs),'No successful formal evaluation allocation')
    return dict(status='RESOURCE_RELEASE_AND_ALLOCATION_AUDIT_PASS' if final else 'PARTIAL_RELEASED_ALLOCATION_AUDIT_NOT_FINAL',
        campaign_id=CAMPAIGN,utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(Path(__file__)),
        all_goal_jobs_released=bool(final),actual_allocated_gpu_seconds=total,actual_allocated_gpu_hours=total/3600,
        acceptance_allocated_gpu_seconds=acceptance,jobs=jobs,quota_evidence_sha256=quota_sha,final_job_list_sha256=listing_sha,
        rate_points_per_allocated_gpu_hour=rates[0],estimated_points_from_allocated_time=total/3600*rates[0],
        actual_goal_invoice_settlement_verified=False,goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    need(not args.output.exists(),'Use a new output receipt')
    result=audit(args.evidence_root.resolve(),read(args.inventory),final=args.final)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
