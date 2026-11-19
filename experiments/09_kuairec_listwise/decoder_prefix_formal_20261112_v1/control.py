"""Four independent one-H100 jobs; no duplicate submission on retry."""
import json
from pathlib import Path
import runpy
import shlex

HERE=Path(__file__).resolve().parent
c=runpy.run_path(str(HERE.parent/'multiseed_freeze_20261109_v1/control.py'))['c']
c.HERE=HERE
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'+HERE.name
VARIANTS=('H0-A','H1-A','H0-B','H1-B')

def prepare():
    c.remote('mkdir -p '+REMOTE,'REMOTE_DIRECTORY.json')
    files=('verified_code.tar.gz','SOURCE_MANIFEST.json','experiment.py','PLAN.md')
    for name in files:
        c.cli(['notebook','scp','env-zzl',(HERE/name).as_posix(),REMOTE+'/'+name,'--workspace',c.CPU,'-t','120'],'UPLOAD_'+name+'.json')
    expected={name:c.digest(HERE/name) for name in files}
    code=f"import pathlib,hashlib,tarfile; p=pathlib.Path({REMOTE!r}); expected={expected!r}; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in expected.items()); tarfile.open(p/'verified_code.tar.gz').extractall(p); print('STAGED_SHA_OK')"
    print(c.remote(c.PYTHON+' -B -c '+shlex.quote(code),'STAGE_READBACK.json',60),flush=True)
    c.write('DEPLOYMENT.json',dict(remote=REMOTE,files=expected))
    command=f'cd {REMOTE} && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nohup {c.PYTHON} -B -u experiment.py prepare > prepare.log 2>&1 < /dev/null &'
    print(c.remote(command,'CACHE_PREPARATION_START.json',30),flush=True)

def submit():
    print(c.remote(f'test -f {REMOTE}/CACHE_READY.json && cat {REMOTE}/CACHE_READY.json','CACHE_READY_READBACK.json',30),flush=True)
    for variant in VARIANTS:
        receipt='SUBMIT_'+variant+'.json'
        if (HERE/receipt).exists():continue
        name='nancy-prefix-'+variant.lower()+'-r20-1112-v1'
        found=c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',name,'--all'],'ABSENCE_'+variant+'.json')
        if found.get('items'):raise RuntimeError('Job already exists; inspect before retry: '+name)
        command=f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u experiment.py run --variant {variant} > {variant}.log 2>&1'
        request=['job','create','--name',name,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,'--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','12','--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',command])]
        c.write('INTENT_'+variant+'.json',dict(variant=variant,name=name,request=request))
        print(variant,c.cli(request,receipt),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'submit':submit}[sys.argv[1]]()
