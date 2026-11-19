"""Record verified final-run restore and native evaluation accounting evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).absolute().parents[3]
D=Path(__file__).absolute().parent
F=ROOT/'experiments/09_kuairec_listwise/formal_execution_20261027'
R=ROOT/'outputs/stage09_restore/goal_20261027_r1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def copied(p):
    target=F/p.name
    if target.exists():assert sha(target)==sha(p)
    else:shutil.copyfile(p,target)
    return dict(file=target.name,sha256=sha(target))

def main():
    receipts={}
    for a in ('J0','J1','H0','H1'):
        manifest=read(D/('STAGE09_RUN_'+a+'_R1.json'))
        archive=D/('stage09_run_'+a+'_r1.tar')
        assert archive.stat().st_size==manifest['archive']['bytes'] and sha(archive)==manifest['archive']['sha256']
        paths=[D/(a+'_REMOTE_BACKUP_PASS_20261028.json'),D/(a+'_LOCAL_DOWNLOAD_PASS_20261028.json'),D/('STAGE09_RUN_'+a+'_R1.json'),R/('RESTORE_'+a+'.json')]
        receipts[a]=[copied(p) for p in paths]
    for a in ('J1','H1'):
        p=R/'acceptance'/(a+'_CPU_RESTORE_20261028.json')
        assert read(p)['status']=='PASS';receipts[a].append(copied(p))
    for name in ('resume_run_download_blocks.py','recover_h1_final_eval_20261028.py','prepare_final_evaluations_20261028.py','final_eval_gate_and_dry_run.py','submit_final_evaluations_20261028.py','record_final_progress_20261028.py'):
        copied(D/name)
    backup=F/'FOUR_RUN_LOCAL_BACKUP_ACCEPTANCE_20261028.json'
    save(backup,dict(status='PASS',remote_root='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1',archive_root=str(D),restore_root=str(R),receipts=receipts,representative_cpu_restore_arms=['J1','H1'],gpu_used=False,final_evaluation_sidecar_complete=False,operations_sidecar_complete=False))
    lp=ROOT/'experiments/09_kuairec_listwise/preparation_20261027/GPU_USAGE_LEDGER.json'
    ledger=read(lp);before=sha(lp)
    assert not any(j['name'].startswith('nancy-stage09-final-eval') for j in ledger['jobs'])
    statuspath=F/'FINAL_EVAL_STATUS_20261028T095042Z.json'
    statuses={r['name']:r for r in read(statuspath)['data']['items']}
    for a in ('J0','J1','H0','H1','H1r2'):
        retry=a=='H1r2';arm='H1' if retry else a
        name='nancy-stage09-final-eval-'+arm.lower()+'-20261027-'+('r2' if retry else 'r1')
        event=F/('FINAL_EVAL_H1_RETRY_EVENTS_START_20261028.json' if retry else 'FINAL_EVAL_'+arm+'_EVENTS_START_20261028.json')
        failed=a=='H1'
        if failed:event=F/'FINAL_EVAL_H1_EVENTS_FAILURE_20261028.json'
        ev=read(event)['data']['items'];times={v['time'] for v in ev if v.get('reason')=='Scheduled'}
        assert len(times)==1
        start=datetime.fromisoformat(next(iter(times))+'+08:00')
        row=dict(name=name,purpose='Frozen-best final evaluation only',arm=arm,gpus=1,nodes=1,status='FAILED' if failed else statuses[name]['status'],scheduled_at=start.isoformat(),platform_max_hours=1.5 if arm.startswith('H') else 2.0,resource_released=failed,allocated_gpu_seconds=None,gpu_hours=None,points_per_gpu_hour=1.0,estimated_points=None,settlement_verified=False,source_receipts_sha256={'../formal_execution_20261027/'+event.name:sha(event)})
        if failed:
            ends={v['time'] for v in ev if v.get('reason')=='SuccessfulDeletePod'};assert len(ends)==1
            end=datetime.fromisoformat(next(iter(ends))+'+08:00');seconds=(end-start).total_seconds();assert seconds==522
            row.update(pod_deleted_at=end.isoformat(),allocated_gpu_seconds=seconds,gpu_hours=seconds/3600,estimated_points=seconds/3600,failure_evidence='NodeNotReady at 17:30:04; native FAILED; deleted 17:35:12; outputs retained; no model traceback in retained log')
            p=F/'FINAL_EVAL_H1_STATUS_FAILURE_20261028.json';row['source_receipts_sha256']['../formal_execution_20261027/'+p.name]=sha(p)
        else:row['source_receipts_sha256']['../formal_execution_20261027/'+statuspath.name]=sha(statuspath)
        if retry:row['replaces_failed_job']='nancy-stage09-final-eval-h1-20261027-r1'
        ledger['jobs'].append(row)
    complete=sum(j['allocated_gpu_seconds'] for j in ledger['jobs'] if j.get('resource_released'))
    active=[j for j in ledger['jobs'] if not j.get('resource_released')]
    reserve=sum(j['platform_max_hours']*3600*j['gpus'] for j in active)
    assert complete==138093 and reserve==25200 and complete+reserve<=270000
    ledger.update(status='four_training_runs_complete_final_evaluations_running_h1_infrastructure_retry',updated_at_utc=datetime.now(timezone.utc).isoformat(),active_gpu_jobs=[j['name'] for j in active],cumulative_allocated_gpu_seconds=complete,completed_jobs_allocated_gpu_seconds=complete,completed_jobs_gpu_hours=complete/3600,remaining_total_gpu_seconds=270000-complete,remaining_total_gpu_seconds_before_active_jobs=270000-complete,active_jobs_reserved_gpu_seconds=reserve,cumulative_with_active_caps_gpu_seconds=complete+reserve,uncommitted_gpu_seconds=270000-complete-reserve,accounting_note='Released jobs total includes failed H1 evaluation 522 seconds. Active Job durations are not settled; 25200 seconds reserves their full caps. No new training/tuning; H1 r2 uses the same frozen checkpoint and settings with separate output paths. Price estimate only; settlement not verified.')
    save(F/'FINAL_EVAL_ACCOUNTING_UPDATE_20261028.json',dict(ledger_before_sha256=before,completed_gpu_seconds=complete,active_reserved_gpu_seconds=reserve,worst_cumulative_gpu_hours=(complete+reserve)/3600,authorized_gpu_hours=75,active_jobs=[j['name'] for j in active]))
    lp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',four_run_backup='PASS',cpu_models=['J1','H1'],active_evaluations=4,failed_h1_gpu_hours=522/3600,worst_cumulative_gpu_hours=(complete+reserve)/3600)),flush=True)

if __name__=='__main__':main()
