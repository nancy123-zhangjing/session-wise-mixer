"""One bounded, identical H1 evaluation retry after native NodeNotReady failure.

Retains all failed outputs; only Job name, output paths and log path change.
No training, parameter edits, or model selection. Unknown create -> stop.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from inspire.services.utils.json_formatter import format_json

ROOT = Path(__file__).absolute().parents[3]
F = ROOT/'experiments/09_kuairec_listwise/formal_execution_20261027'
CLI = shutil.which('inspire') or 'inspire'
OLD = 'nancy-stage09-final-eval-h1-20261027-r1'
NEW = 'nancy-stage09-final-eval-h1-20261027-r2'
G = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
PY = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
P = 'FINAL_EVAL_H1_RETRY_'

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, obj):
    p=F/(P+name+'_20261028.json')
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
    return p

def invoke(argv, timeout=150, raw=False):
    r=subprocess.run([CLI,'--account','sii-shared',*argv],capture_output=True,encoding='utf-8',errors='replace',timeout=timeout)
    need(r.returncode==0,'CLI failure; private output withheld')
    if raw:return r.stdout
    v=json.loads(r.stdout);need(v.get('success') is True,'Native unsuccessful response');return v

def prepare():
    source=F/'FINAL_EVAL_BOUND_PLAN_20261028.json';plan=read(source)
    job=next(j for j in plan['jobs'] if j['arm']=='H1')
    need(job['name']==OLD and job['max_time_hours']==1.5,'Wrong original attempt')
    status=invoke(['--json','job','status',OLD,'--workspace','分布式训练空间'])
    events=invoke(['--json','job','events',OLD,'--workspace','分布式训练空间','--tail','100'])
    need(status['data']['name']==OLD and status['data']['status']=='FAILED','Old Job not FAILED')
    e=events['data']['items'];needed=('Scheduled','NodeNotReady','SuccessfulDeletePod')
    need(all(any(x.get('reason')==k for x in e) for k in needed),'Missing infrastructure failure/release evidence')
    starts={x['time'] for x in e if x.get('reason')=='Scheduled'}
    ends={x['time'] for x in e if x.get('reason')=='SuccessfulDeletePod'}
    need(len(starts)==len(ends)==1,'Ambiguous allocated interval')
    elapsed=(datetime.fromisoformat(next(iter(ends)))-datetime.fromisoformat(next(iter(starts)))).total_seconds()
    need(elapsed==522,'Unexpected failed-attempt accounting')
    listing=invoke(['--json','job','list','--workspace','分布式训练空间','--keyword',NEW,'--all'])
    need(not listing['data']['items'],'Retry name already exists')
    pairs={s['python_argv'][s['python_argv'].index('--output')+1]:s['python_argv'][s['python_argv'].index('--output')+1]+'_r2' for s in job['stages']}
    pairs[job['log']]=job['log'].replace('_r1.log','_r2.log')
    need(len(pairs)==4 and all(k.startswith(G+'/') and v.startswith(G+'/') for k,v in pairs.items()),'Unsafe paths')
    probe=f'''from pathlib import Path
import hashlib,json
g=Path({G!r});paths={list(pairs.values())!r}
assert all(not Path(p).exists() and not Path(p).is_symlink() for p in paths)
failed=g/'evaluations/H1_test_beam32'
assert failed.is_dir() and not failed.is_symlink() and not (failed/'summary.json').exists()
rows=[]
for p in sorted(failed.iterdir()):
 assert p.is_file() and not p.is_symlink()
 raw=p.read_bytes();rows.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
log=g/'logs/final_eval_h1_20261027_r1.log';raw=log.read_bytes()
assert b'Traceback' not in raw and b'CUDA out of memory' not in raw
print('H1_RETRY_PROBE '+json.dumps(dict(failed_outputs=rows,failed_log_sha256=hashlib.sha256(raw).hexdigest(),new_paths_absent=paths,old_outputs_retained=True)))
'''
    raw=invoke(['notebook','ssh','nancy-stage09-prep-20261027-r1','--workspace','CPU资源空间','--command-timeout','60','--timeout','60','--',PY,'-B','-c',probe],raw=True)
    rows=[json.loads(x[len('H1_RETRY_PROBE '):]) for x in raw.splitlines() if x.startswith('H1_RETRY_PROBE ')]
    need(len(rows)==1,'Missing retry probe')
    cmd=job['command_template']
    for a,b in pairs.items():need(a in cmd,'Missing replacement path');cmd=cmd.replace(a,b)
    argv=list(job['cli_argv_template']);argv[argv.index('--name')+1]=NEW
    argv[argv.index('--command')+1]=cmd
    changed=[(i,a,b) for i,(a,b) in enumerate(zip(job['cli_argv_template'],argv)) if a!=b]
    need(len(changed)==2,'Unexpected resource/parameter changes')
    shell=shlex.split(cmd);need(shell[:2]==['bash','-c'] and len(shell)==3,'Bad shell wrapper')
    r=subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n'],input=shell[2],capture_output=True,text=True)
    need(r.returncode==0,'Shell syntax differs')
    dry=invoke(argv+['--dry-run']);d=dry['data']
    display=json.loads(format_json({'command':cmd}))['data']['command']
    need(d['name']==NEW and d['command']==display and d['resource']['gpu']==1 and d['max_time_hours']==1.5 and not d['auto_fault_tolerance'],'Dry-run mismatch')
    ledger=read(ROOT/'experiments/09_kuairec_listwise/preparation_20261027/GPU_USAGE_LEDGER.json')
    # Prior seven jobs + actual released formal job, then failed attempt + active other caps + new H1 cap.
    released=sum(j['allocated_gpu_seconds'] for j in ledger['jobs'] if j.get('resource_released') and not j['name'].startswith('nancy-stage09-final-eval'))
    need(released==137571,'Earlier budget differs')
    worst=released+elapsed+(2+2+1.5+1.5)*3600
    need(worst<=270000,'75 GPU-hour budget exceeded')
    receipts={k:dict(file=(p:=save(k,v)).name,sha256=sha(p)) for k,v in [('STATUS_BEFORE',status),('EVENTS_BEFORE',events),('ABSENCE',listing),('OUTPUT_PROBE',rows[0]),('DRY_RUN',dry)]}
    save('PLAN',dict(status='GATED_NOT_SUBMITTED',name=NEW,replaces_failed_job=OLD,original_plan_sha256=sha(source),unchanged_scientific_settings=True,failed_attempt_gpu_seconds=elapsed,upper_bound_cumulative_gpu_seconds=worst,cli_argv=argv,command=cmd,path_replacements=pairs,receipts=receipts))
    print(json.dumps(dict(status='RETRY_GATES_PASS',name=NEW,worst_cumulative_gpu_hours=worst/3600,new_gpu_cap_hours=1.5)),flush=True)

def submit():
    pp=F/(P+'PLAN_20261028.json');p=read(pp)
    need(p['status']=='GATED_NOT_SUBMITTED' and p['upper_bound_cumulative_gpu_seconds']<=270000,'Invalid retry plan')
    for ref in p['receipts'].values():need(sha(F/ref['file'])==ref['sha256'],'Receipt modified')
    need(not (F/(P+'SUBMIT_INTENT_20261028.json')).exists(),'Existing attempt; inspect native status, do not resubmit')
    state=invoke(['--json','job','status',OLD,'--workspace','分布式训练空间'])
    need(state['data']['status']=='FAILED','Old status changed')
    absent=invoke(['--json','job','list','--workspace','分布式训练空间','--keyword',NEW,'--all'])
    need(not absent['data']['items'],'New Job already exists')
    save('SUBMIT_INTENT',dict(name=NEW,plan_sha256=sha(pp),attempted_at_utc=datetime.now(timezone.utc).isoformat(),uncertain_response_rule='Inspect exact named job before any retry'))
    try:
        r=invoke(p['cli_argv'],timeout=180);need(r['data']['name']==NEW,'Created identity differs');save('CREATE',r)
    except Exception as e:
        save('UNCERTAIN',dict(name=NEW,error_type=type(e).__name__,action='STOP_AND_INSPECT_EXACT_NAME'));raise RuntimeError('Uncertain retry response') from None
    print(json.dumps(dict(status='SUBMITTED',name=NEW,training_started=False,max_gpu_hours=1.5)),flush=True)

if __name__=='__main__':
    os.environ['NO_PROXY']='.sii.edu.cn,<institution-host>'
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);a=p.parse_args()
    prepare() if a.action=='prepare' else submit()
