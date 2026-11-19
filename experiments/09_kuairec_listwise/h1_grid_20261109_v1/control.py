"""Transport and single-card submissions; training source remains immutable."""
import base64, importlib.util, json, shlex, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('grid_transport', HERE.parent/'h1_no_code_tag_20261108_v1/control.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.HERE=HERE
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_grid_20261109_v1'

def py(code,receipt,timeout=120):
    encoded=base64.b64encode(code.encode()).decode()
    return c.remote(c.PYTHON+' -B -c '+shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))'),receipt,timeout)

def prepare():
    c.remote(f'mkdir {REMOTE}', 'DIRECTORY.json')
    names=['grid.py','protocol.py','experiment.py','config_B.json','config_C.json','README.md','LOCAL_CHECKS.json','PARENT_CPU_CHECK.json']
    for name in names:
        c.cli(['notebook','scp','env-zzl',str(HERE/name),REMOTE+'/'+name,'--workspace',c.CPU,'-t','120'],'UPLOAD_'+name+'.json')
    expected={n:c.digest(HERE/n) for n in names}
    code=f"""
from pathlib import Path
import hashlib,json
r=Path({REMOTE!r})
for name,digest in {expected!r}.items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
original=r.parent/'goal_20261027_r1'
assert (original/'runs/H1/initial.pt').is_file()
for arm in ('B','C'):
 cfg=json.loads((r/f'config_{{arm}}.json').read_text())['base_config']
 for key in ('dataset','assets'): assert Path(cfg[key]).is_dir(),key
assert Path({c.PYTHON!r}).is_file()
print(json.dumps(dict(status='passed',sha256={expected!r})))
"""
    print(py(code,'REMOTE_FILES_VERIFIED.json'))

def submit(arm):
    name=f'nancy-s09-grid-{arm.lower()}-1109-v1'
    if (HERE/f'{arm}_SUBMISSION_INTENT.json').exists(): raise ValueError('Inspect existing intent before retry')
    if c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',name,'--all'],arm+'_ABSENCE.json').get('items'): raise ValueError('Job exists')
    script=f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u experiment.py --arm {arm} --output {REMOTE}/runs/{arm} > {REMOTE}/{arm}.log 2>&1'
    req=['job','create','--name',name,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,'--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','24','--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',script])]
    c.cli(req+['--dry-run'],arm+'_DRY_RUN.json')
    c.write(arm+'_SUBMISSION_INTENT.json',dict(job=name,request=req,remote=REMOTE))
    print(c.cli(req,arm+'_SUBMISSION_RESULT.json'),flush=True)

def status():
    for arm in ('B','C'):
        state=c.cli(['job','status',f'nancy-s09-grid-{arm.lower()}-1109-v1','--workspace',c.WORKSPACE],arm+'_STATUS.json')
        print(json.dumps(dict(arm=arm,status=state['status'],nodes=state.get('nodes',[]))),flush=True)
    code=f"""
from pathlib import Path
import json,time
r=Path({REMOTE!r})
for a in ('B','C'):
 log=r/(a+'.log'); run=r/'runs'/a
 training=run/'training.jsonl'
 print(json.dumps(dict(arm=a,observed_unix=time.time(),log=log.read_text()[-2500:] if log.exists() else '',training_tail=training.read_text().splitlines()[-1] if training.exists() else None,checkpoints=[dict(name=p.name,bytes=p.stat().st_size) for p in run.glob('*.pt')],validations=[json.loads(p.read_text()) for p in sorted((run/'checkpoints').glob('R??.json'))])))
"""
    result=py(code,'PROGRESS.json')
    for line in result.splitlines():
        row=json.loads(line)
        latest=json.loads(row['training_tail']) if row['training_tail'] else None
        brief=dict(arm=row['arm'],observed_unix=row['observed_unix'],latest=latest,validations=row['validations'])
        for log_line in row['log'].splitlines():
            try:event=json.loads(log_line)
            except json.JSONDecodeError:continue
            if event.get('event')=='evaluation_progress':brief['evaluation_progress']=event
        if brief.get('evaluation_progress', {}).get('split') != 'test' and latest and (latest['phase']!='R' or latest['update']<693 or any(v['epoch']==latest['epoch'] for v in row['validations'])):
            brief.pop('evaluation_progress',None)
        with (HERE/'PROGRESS_HISTORY.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(brief)+'\n')
        if 'Traceback' in row['log'] or 'Error' in row['log']: brief['error_log']=row['log']
        print(json.dumps(brief),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='submit': submit(sys.argv[2])
    elif sys.argv[1]=='status': status()
