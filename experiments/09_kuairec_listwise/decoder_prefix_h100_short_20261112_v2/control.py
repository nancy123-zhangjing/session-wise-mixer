"""One bounded H100 job; imports only the established platform transport."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
c=runpy.run_path(str(HERE.parent/'multiseed_freeze_20261109_v1/control.py'))['c']
c.HERE=HERE
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'+HERE.name
JOB='nancy-s09-prefix-short-1112-v2'


def prepare():
    files=list((ROOT/'scripts/listwise_diffgrm').glob('*.py'))
    files+=list((ROOT/'external/DiffGRM/genrec').rglob('*.py'))
    files+=list((ROOT/'configs/09_kuairec_listwise').glob('h[01]_prefix_*.json'))
    for p in (ROOT/'scripts/__init__.py',):
        if p.exists():files.append(p)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):c.digest(p) for p in files}
    c.write('SOURCE_MANIFEST.json',manifest)
    bundle=HERE/'short_test.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for p in files:tar.add(p,arcname='code/'+p.relative_to(ROOT).as_posix())
        tar.add(HERE/'benchmark.py',arcname='benchmark.py')
        tar.add(HERE/'SOURCE_MANIFEST.json',arcname='SOURCE_MANIFEST.json')
    c.remote('mkdir -p '+REMOTE,'REMOTE_DIRECTORY.json')
    c.cli(['notebook','scp','env-zzl',bundle.as_posix(),REMOTE+'/short_test.tar.gz','--workspace',c.CPU,'-t','120'],'UPLOAD.json')
    expected=c.digest(bundle)
    code=f"import hashlib,tarfile,pathlib; p=pathlib.Path({REMOTE!r}); assert hashlib.sha256((p/'short_test.tar.gz').read_bytes()).hexdigest()=={expected!r}; tarfile.open(p/'short_test.tar.gz').extractall(p); print('UPLOAD_SHA_OK')"
    print(c.remote(c.PYTHON+' -B -c '+shlex.quote(code),'STAGE_READBACK.json',60))


def submit():
    if (HERE/'SUBMISSION_INTENT.json').exists():raise RuntimeError('Submission exists; inspect first')
    if c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',JOB,'--all'],'ABSENCE.json').get('items'):raise RuntimeError('Job exists')
    command=f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u benchmark.py > benchmark.log 2>&1'
    req=['job','create','--name',JOB,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,
         '--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','0.5',
         '--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',command])]
    c.cli(req+['--dry-run'],'DRY_RUN.json')
    c.write('SUBMISSION_INTENT.json',{'job':JOB,'remote':REMOTE,'request':req})
    print(c.cli(req,'SUBMISSION_RESULT.json'))


def status():
    state=c.cli(['job','status',JOB,'--workspace',c.WORKSPACE],'STATUS.json')
    print(state,flush=True)
    if state['status'] not in ('QUEUING','PENDING','CREATED'):
        print(c.remote(f'if test -f {REMOTE}/benchmark.log; then tail -n 5 {REMOTE}/benchmark.log; fi','LOG_TAIL.json',30))


if __name__=='__main__':
    {'prepare':prepare,'submit':submit,'status':status}[sys.argv[1]]()
