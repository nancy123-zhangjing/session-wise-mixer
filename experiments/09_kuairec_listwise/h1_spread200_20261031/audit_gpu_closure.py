"""Audit all five GPU attempts from final platform status/instances/events."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tz=timezone(timedelta(hours=8))
stamp=lambda s:datetime.fromisoformat(s).replace(tzinfo=tz).timestamp()
specs=[('nancy-s09-d-preflight-1031-r1','preflight','SUCCEEDED'),
       ('nancy-s09-d-1031-r1','formal','CANCELLED'),
       ('nancy-s09-d-prof-1031-r1','preflight','SUCCEEDED'),
       ('nancy-s09-d-concur-1031-r1','preflight','SUCCEEDED'),
       ('nancy-s09-d-async-1031-r2','formal_continuation','SUCCEEDED')]
rates={};inputs={}
for filename in ('GPU_RATE_BASIS_20261031_1620.json','GPU_RATE_BASIS_ORIGINAL_GROUP_1708.json'):
    path=here/filename;inputs[filename]=sha(path)
    for row in read(path)['result']['data']['items']:
        if row['quota']=='1,20,200':rates[row['compute_group']]=row['points_per_hour']
rows=[]
for name,kind,expected in specs:
    sources={}
    for suffix in ('status','instances','events'):
        path=here/'closure_platform_r1'/f'{name}_{suffix}.json'
        value=read(path)
        if not value['success']:raise ValueError('Platform query failed')
        sources[suffix]=value['data'];inputs[str(path.relative_to(here))]=sha(path)
    status=sources['status'];instances=sources['instances']['items'];events=sources['events']['items']
    if status['status']!=expected or len(instances)!=1:raise ValueError('Missing terminal single instance')
    if instances[0]['status']!=('instance_failed' if expected=='CANCELLED' else 'instance_success'):
        raise ValueError('Instance not terminal as expected')
    if status['resource']['gpu']!=1 or status['resource']['nodes']!=1:raise ValueError('Unexpected GPU allocation')
    created=int(status['created_at'])/1000;finished=int(status['finished_at'])/1000
    starts=[stamp(e['time']) for e in events if e['reason']=='Scheduled']
    deletions=[stamp(e['time']) for e in events if e['reason']=='SuccessfulDeletePod']
    if not starts or not deletions:raise ValueError('Actual scheduling/deletion evidence missing')
    start=min(starts);deleted=min(deletions)
    if not created<=start<=finished<=deleted:raise ValueError('Platform timestamps inconsistent')
    if any(stamp(e['time'])>deleted and e['reason'] in ('Scheduled','Started','SuccessfulCreatePod') for e in events):
        raise ValueError('Pod activity after claimed release')
    # A bounded extra minute is counted in the budget after the actual pod
    # deletion, even though no GPU pod exists then. Later repeated controller
    # cleanup events are recorded separately and do not imply a GPU allocation.
    upper=deleted-created+60
    rate=rates[status['compute_group']]
    rows.append(dict(name=name,kind=kind,status=expected,gpu_count=1,
        created_unix=created,scheduled_unix=start,pod_deleted_unix=deleted,
        platform_finished_unix=finished,scheduled_to_pod_deleted_seconds=deleted-start,
        created_to_pod_deleted_seconds=deleted-created,
        release_reserve_seconds=60,budget_upper_seconds=upper,
        points_per_instance_hour_basis=rate,estimated_points_upper=upper/3600*rate,
        latest_controller_event_unix=max(stamp(e['time']) for e in events)))
ordered=sorted(rows,key=lambda row:row['scheduled_unix'])
if any(a['pod_deleted_unix']>b['scheduled_unix'] for a,b in zip(ordered,ordered[1:])):
    raise ValueError('GPU allocation intervals overlap')
total=sum(r['budget_upper_seconds'] for r in rows)
short=sum(r['budget_upper_seconds'] for r in rows if r['kind']=='preflight')
if total>86400 or short>1800:raise ValueError('Authorized budget exceeded')
result=dict(status='passed_all_GPU_attempts_terminal_released_and_within_budget',jobs=rows,
    cumulative_budget_upper_seconds=total,cumulative_gpu_hours_upper=total/3600,
    scheduled_to_pod_deletion_hours=sum(r['scheduled_to_pod_deleted_seconds'] for r in rows)/3600,
    preflight_upper_seconds=short,max_concurrent_gpus=1,
    estimated_points_upper=sum(r['estimated_points_upper'] for r in rows),
    actual_settlement_verified=False,input_sha256=inputs,auditor_sha256=sha(Path(__file__)),
    timing_rule='Every attempt counts creation through actual Pod deletion plus 60 seconds conservative release reserve. Scheduled-to-deletion intervals prove no overlap.',
    limitation='Platform allocation windows and live listed point rates are evidence for a conservative estimate, not an invoice. Late controller-only cleanup after deleted Pods is not counted as GPU occupancy. CPU closure remains separately required.')
with (here/'FINAL_GPU_RESOURCE_AUDIT.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','input_sha256')}))
