"""Recover timeout-interrupted evaluation using the unchanged experiment runner."""
import json
import runpy
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
ctl = runpy.run_path(str(HERE / 'control.py'))
c, remote = ctl['c'], ctl['REMOTE']

def inspect():
    for v in ctl['VARIANTS']:
        old = 'nancy-prefix-' + v.lower() + '-r20-1112-v1'
        status = c.cli(['job', 'status', old, '--workspace', c.WORKSPACE], 'RECOVERY_STATUS_'+v+'.json')
        assert status['status'] == 'CANCELLED', status
        events = c.cli(['job', 'events', old, '--workspace', c.WORKSPACE, '--tail', '100'], 'RECOVERY_EVENTS_'+v+'.json')
        assert 'time limit' in json.dumps(events).lower(), events
    code = f'''
import sys,json,pathlib
root=pathlib.Path({remote!r});sys.path.insert(0,str(root))
import experiment as e
import torch
e.verify_source()
rows={{}}
for v in e.VARIANTS:
 p=root/v/'train'/'last.pt';s=torch.load(p,map_location='cpu',weights_only=False)
 assert s['r_updates']==13860 and s['t_updates']==200
 progress=s['progress']
 assert (progress['phase'],progress['update']) in ((24,693),(25,0)),progress
 rows[v]={{'checkpoint':str(p),'sha256':e.sha(p),'progress':progress,'r_updates':s['r_updates'],'t_updates':s['t_updates']}}
 del s
print(json.dumps(rows))
'''
    result=json.loads(c.remote(c.PYTHON+' -B -c '+shlex.quote(code),'RECOVERY_CHECKPOINT_READBACK.json',180))
    c.write('RECOVERY_CHECKPOINTS.json',result)
    print(json.dumps(result),flush=True)

def submit():
    checkpoints=json.loads((HERE/'RECOVERY_CHECKPOINTS.json').read_text(encoding='utf-8'))
    c.resources()
    mapping={}
    for v in ctl['VARIANTS']:
        name='nancy-prefix-'+v.lower()+'-eval-1113-r1'
        existing=c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',name,'--all'],'RECOVERY_ABSENCE_'+v+'.json')
        if existing.get('items'):
            raise RuntimeError('Recovery job already exists: '+name)
        old='nancy-prefix-'+v.lower()+'-r20-1112-v1'
        assert c.cli(['job','status',old,'--workspace',c.WORKSPACE])['status']=='CANCELLED'
        code=f'''
import pathlib,hashlib,json
p=pathlib.Path({remote!r})/{v!r}/'train'
h=hashlib.sha256()
with (p/'last.pt').open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()=={checkpoints[v]['sha256']!r}
lock=p/'.running'
if lock.exists():
 archive=p/'.running.timeout_1113_r1'
 assert not archive.exists()
 lock.rename(archive)
log=pathlib.Path({remote!r})/({v!r}+'.log')
archive=log.with_suffix('.timeout_1113_r1.log')
assert not archive.exists()
log.rename(archive)
print('RECOVERY_READY')
'''
        c.remote(c.PYTHON+' -B -c '+shlex.quote(code),'RECOVERY_READY_'+v+'.json',90)
        command=f'cd {remote} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u experiment.py run --variant {v} > {v}.log 2>&1'
        request=['job','create','--name',name,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,'--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','12','--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',command])]
        c.write('RECOVERY_INTENT_'+v+'.json',dict(name=name,request=request))
        print(v,c.cli(request,'RECOVERY_SUBMIT_'+v+'.json'),flush=True)
        mapping[v]=name
        c.write('RECOVERY_JOBS.json',mapping)

if __name__=='__main__':
    import sys
    {'inspect':inspect,'submit':submit}[sys.argv[1] if len(sys.argv)>1 else 'inspect']()
