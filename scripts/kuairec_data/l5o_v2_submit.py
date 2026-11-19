"""Build (and only on --submit, create) the protocol v2 Inspire jobs. Default is a dry run."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess

from .l5o_v2_job import JOBS

STAGE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
GROUP = '开发区-H100-cuda12.8版本-183核'
QUOTA = {1: '1,20,200', 2: '2,40,400', 4: '4,80,900', 8: '8,160,1800'}
# Generous caps over the estimates in PROTOCOL.md; every stage resumes from last.pt.
MAX_HOURS = {'h': 24, 'tiger': 48, 'cf': 36, 'sasrec': 12}


def request(job, code, root, tiger_batch=256, suffix='', extra=(), prefix='nancy-l5o-v2'):
    family = job.split('-')[0]
    inputs = dict(receipt=f'{STAGE}/l5o_sasrec_20261117_v1/run/COMPLETE.json',
                  dataset=f'{STAGE}/l5o_stage0_20261117_v1/dataset',
                  assets=f'{STAGE}/l5o_stage0_20261117_v1/assets',
                  cf_runtime=f'{code}/experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json')
    worker = [PYTHON, '-u', '-m', 'scripts.kuairec_data.l5o_v2_job', job, '--root', root,
              *[x for k, v in inputs.items() for x in ('--' + k.replace('_', '-'), v)],
              '--tiger-batch', str(tiger_batch), *extra]
    command = (f'cd {code} && export PYTHONPATH={code}/ar_deps:{code} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
               f'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && {shlex.join(worker)}')
    gpus = len(JOBS[job])
    return ['job', 'create', '--name', f'{prefix}-{job}{suffix}', '--workspace', '分布式训练空间',
            '--project', '公共科研项目', '--group', GROUP, '--quota', QUOTA[gpus],
            '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3', '--nodes', '1', '--shm-size', '32',
            '--max-time', str(MAX_HOURS[family]), '--priority', '4', '--no-auto-fault-tolerance',
            '--no-enable-notification', '--command', shlex.join(['bash', '-c', command])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs', nargs='+', choices=sorted(JOBS))
    p.add_argument('--code', required=True, help='commit-pinned code snapshot on Inspire (with ar_deps, external)')
    p.add_argument('--root', default=f'{STAGE}/l5o_v2_20261118')
    p.add_argument('--tiger-batch', type=int, default=256)
    p.add_argument('--suffix', default='', help='job-name suffix for a resubmission (e.g. -r2); log names keep the slot job')
    p.add_argument('--h-rhythm', choices=['pass', 'old693'], default='pass')
    p.add_argument('--h-split', choices=['l5o', 'old'], default='l5o')
    p.add_argument('--smoke', action='store_true', help='GPU smoke test of the job layout (never results)')
    p.add_argument('--prefix', default='nancy-l5o-v2', help='job-name prefix')
    p.add_argument('--submit', action='store_true', help='actually create the jobs (requires user approval)')
    a = p.parse_args()
    if (a.h_rhythm != 'pass' or a.h_split != 'l5o') and a.root == p.get_default('root'):
        p.error('a control run needs its own --root')
    if a.h_split != 'l5o' and any(not job.startswith('h-') for job in a.jobs):
        p.error('--h-split applies to H jobs only')
    extra = ((['--h-rhythm', a.h_rhythm] if a.h_rhythm != 'pass' else [])
             + (['--h-split', a.h_split] if a.h_split != 'l5o' else []) + (['--smoke'] if a.smoke else []))
    for job in a.jobs:
        args = ['inspire', '--account', 'sii-shared', *request(job, a.code, a.root, a.tiger_batch, a.suffix, extra, a.prefix)]
        if not a.submit:
            args.append('--dry-run')
        result = subprocess.run(args, capture_output=True, text=True)
        print(json.dumps(dict(job=job, submitted=a.submit, returncode=result.returncode,
                              output=result.stdout[-2000:], error=result.stderr[-2000:]), ensure_ascii=False))


if __name__ == '__main__':
    main()
