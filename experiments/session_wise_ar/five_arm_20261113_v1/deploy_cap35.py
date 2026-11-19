"""Explicit replacement of two authorized DIGER jobs, preserving checkpoints."""
import json
import shlex
from pathlib import Path
import control as c

REMOTE = c.REMOTE.replace('_v1','_r2')


def main():
    script = c.HERE/'cap35_worker.py'
    c.cli(['notebook','scp',c.CPU,str(script),REMOTE+'/delivery/cap35_worker.py',
           '--workspace','CPU资源空间'],'CAP35_UPLOAD.json')
    for arm in ('AR-D0','AR-D1'):
        old = 'nancy-five-'+arm.lower()+'-1113-r2'
        new = 'nancy-'+arm.lower()+'-cap35-1114-v1'
        listed = c.cli(['job','list','--workspace','分布式训练空间','--keyword',new,'--all'],arm+'_CAP_LOOKUP.json')
        if any(r['name']==new for r in listed['items']):
            print(new,'already submitted',flush=True);continue
        listed = c.cli(['job','list','--workspace','分布式训练空间','--keyword',old,'--all'],arm+'_OLD_LOOKUP.json')
        matches=[r for r in listed['items'] if r['name']==old]
        assert len(matches)==1
        if matches[0]['status']=='RUNNING':
            c.cli(['job','stop',old,'--workspace','分布式训练空间'],arm+'_CAP_STOP.json')
        listed = c.cli(['job','list','--workspace','分布式训练空间','--keyword',old,'--all'],arm+'_OLD_STOPPED.json')
        status = next(r['status'] for r in listed['items'] if r['name']==old)
        if status not in ('STOPPED','FAILED','SUCCEEDED','CANCELLED'):
            raise RuntimeError('Wait for old job terminal state: '+status)
        source=f'''from pathlib import Path
import torch,json,hashlib,time
p=Path({REMOTE!r})/'runs'/{arm!r}/'train'
s=torch.load(p/'last.pt',map_location='cpu',weights_only=False,mmap=True)
assert s['config']['training']['max_epochs']==120
assert not s['progress']['done']
assert s['progress']['step']<484995
h=hashlib.sha256()
with (p/'last.pt').open('rb') as f:
 for b in iter(lambda:f.read(4194304),b''):h.update(b)
r={{'unix':time.time(),'step':s['progress']['step'],'best_step':s['progress']['best_step'],'checkpoint_sha256':h.hexdigest(),'old_job_status':{status!r}}}
lock=p/'.running'
if lock.exists():
 r['old_lock']=lock.read_text();(p/'CAP35_OLD_RUNNING_LOCK.txt').write_text(r['old_lock']);lock.unlink()
(p/'CAP35_HANDOVER.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))
'''
        result=json.loads(c.python(source,arm+'_CAP_HANDOVER_RAW.json',timeout=60))
        (c.HERE/(arm+'_CAP_HANDOVER.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        intent=c.HERE/(arm+'_CAP_SUBMIT_INTENT.json')
        if intent.exists():raise RuntimeError('Resolve prior submission before retry')
        command=(f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 '
                 f'PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 '
                 f'TRANSFORMERS_OFFLINE=1 PYTHONPATH={REMOTE}/ar_deps:{REMOTE} && '
                 f'{c.PYTHON} -B -u delivery/cap35_worker.py --root {REMOTE} --arm {arm} '
                 f'> {REMOTE}/{arm}_cap35.log 2>&1')
        request=['job','create','--name',new,'--workspace','分布式训练空间','--project','公共科研项目',
                 '--group','开发区-H100-cuda12.8版本-183核','--quota','1,20,200',
                 '--image','ngc-pytorch:25.02-cuda12.8.0-py3','--nodes','1','--shm-size','32',
                 '--priority','4','--no-auto-fault-tolerance','--no-enable-notification',
                 '--command',shlex.join(['bash','-c',command])]
        intent.write_text(json.dumps({'name':new,'request':request},ensure_ascii=False,indent=2),encoding='utf-8')
        print(c.cli(request,arm+'_CAP_SUBMITTED.json'),flush=True)


if __name__=='__main__':main()
