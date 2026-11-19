"""Small account-bound CLI receipt helper for this execution only."""
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/session_wise_ar_five_20261113_v1'
PYTHON='/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
CPU='nancy-s09-e-prep-1101-r1'
ENV=dict(os.environ,NO_PROXY='.sii.edu.cn',HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',PYTHONUTF8='1')


def cli(args, receipt, timeout=180):
    p=subprocess.run(['inspire','--account','sii-shared','--json',*args],env=ENV,
                     capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    value=json.loads(p.stdout)
    (HERE/'raw').mkdir(exist_ok=True)
    (HERE/'raw'/receipt).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    if p.returncode or not value.get('success'): raise RuntimeError(str(value.get('error')))
    data=value.get('data',{})
    if isinstance(data,dict) and data.get('returncode'): raise RuntimeError(data.get('output','remote failed'))
    if isinstance(data,dict) and 'output' in data:
        data['output']=data['output'].split('Starting client (stdio mode)')[0].strip()
    return data


def remote(command,receipt,timeout=60):
    return cli(['notebook','exec',CPU,'--workspace','CPU资源空间','--timeout',str(timeout),command+' # '],
               receipt,timeout+120).get('output','')


def python(source,receipt,timeout=60):
    encoded=base64.b64encode(source.encode()).decode()
    code=f"import base64; exec(base64.b64decode({encoded!r}))"
    return remote(PYTHON+' -B -c '+shlex.quote(code),receipt,timeout)


def submit(arm):
    if arm not in ('AR-T0','AR-T1','AR-D0','AR-D1','CF-M0'): raise ValueError(arm)
    name='nancy-five-'+arm.lower()+'-1113-v1'
    listed=cli(['job','list','--workspace','分布式训练空间','--keyword',name,'--all'],arm+'_EXISTS.json')
    matches=[r for r in listed['items'] if r['name']==name]
    if len(matches)==1: return matches[0]
    intent=HERE/'raw'/(arm+'_INTENT.json')
    if matches or intent.exists(): raise RuntimeError('Resolve previous submission before retry')
    if arm!='CF-M0':
        gate=json.loads((HERE/'GPU_ACCEPTANCE.json').read_text())
        if not gate.get('passed'): raise RuntimeError('AR GPU gate not passed')
    environment='OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'
    if arm!='CF-M0': environment+=' PYTHONPATH='+REMOTE+'/ar_deps:'+REMOTE
    command=f'cd {REMOTE} && export {environment} && {PYTHON} -B -u delivery/worker.py {arm} > {arm}.log 2>&1'
    request=['job','create','--name',name,'--workspace','分布式训练空间','--project','公共科研项目',
             '--group','开发区-H100-cuda12.8版本-183核','--quota','1,20,200','--image','ngc-pytorch:25.02-cuda12.8.0-py3',
             '--nodes','1','--shm-size','32','--priority','4','--no-auto-fault-tolerance','--no-enable-notification',
             '--command',shlex.join(['bash','-c',command])]
    intent.write_text(json.dumps({'arm':arm,'request':request},ensure_ascii=False,indent=2),encoding='utf-8')
    return cli(request,arm+'_SUBMITTED.json')


if __name__=='__main__':
    source=Path(sys.argv[1]).read_text(encoding='utf-8-sig')
    print(python(source,sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 60))
