"""Submit the two versioned TIGER-AdamW GPU jobs (one H100 each). Refuses duplicates."""
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/session_wise_ar_tiger_adamw_20261114_v1'
PY = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
ENV = dict(os.environ, NO_PROXY='.sii.edu.cn', HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', PYTHONUTF8='1')


def cli(args):
    p = subprocess.run(['inspire', '--account', 'sii-shared', '--json', *args], env=ENV, capture_output=True, text=True, encoding='utf-8')
    return json.loads(p.stdout)


for arm in ('AR-T0A', 'AR-T1A'):
    name = f'nancy-tiger-adamw-{arm[3:].lower()}-1114-v1'
    listed = cli(['job', 'list', '--workspace', '分布式训练空间', '--keyword', name, '--all'])
    if any(r['name'] == name for r in listed['data']['items']):
        print(name, 'already exists; not resubmitting'); continue
    env = ('OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 '
           'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
           f'PYTHONPATH={ROOT}/ar_deps:{ROOT}')
    command = f'cd {ROOT} && export {env} && {PY} -B -u delivery/worker_adamw.py {arm} > {arm}.log 2>&1'
    request = ['job', 'create', '-n', name, '--workspace', '分布式训练空间', '-p', '公共科研项目',
               '--group', '开发区-H100-cuda12.8版本-183核', '-q', '1,20,200', '-i', 'ngc-pytorch:25.02-cuda12.8.0-py3',
               '--nodes', '1', '--shm-size', '32', '--priority', '4', '--max-time', '12', '--no-auto-fault-tolerance',
               '-c', shlex.join(['bash', '-c', command])]
    result = cli(request)
    (HERE / f'SUBMIT_{arm}.json').write_text(json.dumps(dict(request=request, result=result), ensure_ascii=False, indent=2), encoding='utf-8')
    print(name, result)
