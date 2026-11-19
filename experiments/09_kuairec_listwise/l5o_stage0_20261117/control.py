"""Stage-0 only. Public Inspire CLI, including in-process dispatch for long source payloads."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
from contextlib import redirect_stdout

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1'
PYTHON='/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
GPU_JOB='nancy-l5o-h100-check-1117-v1'


def cli(args,receipt):
    # Same public CLI argument validation, no direct platform API or secret access.
    # In-process avoids Windows' 32KB command-line limit for a source-only payload.
    from inspire.cli.main import main
    os.environ.update(NO_PROXY='.sii.edu.cn',HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',PYTHONUTF8='1')
    captured=io.StringIO()
    with redirect_stdout(captured):
        main.main(args=['--account','sii-shared','--json',*args],standalone_mode=False)
    raw=captured.getvalue()
    (HERE/'raw').mkdir(exist_ok=True)
    (HERE/'raw'/receipt).write_text(raw,encoding='utf-8')
    value=json.loads(raw)
    if not value.get('success'):
        raise RuntimeError(value.get('error'))
    return value['data']


def submit_gpu():
    if not (HERE/'REMOTE_PREPARED.json').exists():
        raise RuntimeError('Need read-back of PREPARED.json before allocating GPU')
    prepared=json.loads((HERE/'REMOTE_PREPARED.json').read_text(encoding='utf-8'))
    local=json.loads((HERE/'LOCAL_DATA_AUDIT.json').read_text(encoding='utf-8'))
    assert prepared['passed'] and prepared['normalized_sample_hashes']==local['normalized_sample_hashes']
    command=f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && {PYTHON} -u benchmark.py'
    request=['job','create','--name',GPU_JOB,'--workspace','分布式训练空间','--project','公共科研项目',
        '--group','开发区-H100-cuda12.8版本-183核','--quota','1,20,200','--image','ngc-pytorch:25.02-cuda12.8.0-py3',
        '--nodes','1','--shm-size','32','--max-time','2','--priority','4','--no-auto-fault-tolerance',
        '--no-enable-notification','--command',shlex.join(['bash','-c',command])]
    intent=HERE/'GPU_INTENT.json'
    if intent.exists(): raise RuntimeError('Prior GPU intent exists; inspect, do not duplicate')
    cli(request+['--dry-run'],'GPU_DRY_RUN.json')
    intent.write_text(json.dumps(dict(name=GPU_JOB,remote=REMOTE,max_hours=2)))
    print(json.dumps(cli(request,'GPU_SUBMITTED.json'),ensure_ascii=False))


if __name__=='__main__':
    {'submit_gpu':submit_gpu}[sys.argv[1]]()
