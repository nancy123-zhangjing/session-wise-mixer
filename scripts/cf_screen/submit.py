"""Build (and only with --submit, create) the two 4-GPU screening jobs. Default is a dry run."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess

from .variants import JOBS, RERUNS

QUOTA = {1: '1,20,200', 4: '4,80,900'}

STAGE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
GROUP = '开发区-H100-cuda12.8版本-183核'
ROOT = f'{STAGE}/cf_screen_20261118'
WAVE = f'{STAGE}/l5o_wave_20261117_v1'


def request(job, code, commit, smoke=False):
    worker = [PYTHON, '-u', '-m', 'scripts.cf_screen.job', job, '--root', ROOT, '--wave', WAVE,
              '--runtime', f'{code}/experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json',
              *(['--smoke'] if smoke else [])]
    command = (f'cd {code} && export PYTHONPATH={code}/ar_deps:{code} CODE_COMMIT={commit} OMP_NUM_THREADS=4 '
               f'MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {shlex.join(worker)}')
    name = f'nancy-cf-screen-{"smoke-" if smoke else ""}{job}-1118'
    return name, ['job', 'create', '--name', name, '--workspace', '分布式训练空间', '--project', '公共科研项目',
                  '--group', GROUP, '--quota', QUOTA[len({**JOBS, **RERUNS}[job])], '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3',
                  '--nodes', '1', '--shm-size', '32', '--max-time', '2' if smoke else '8', '--priority', '4',
                  '--no-auto-fault-tolerance', '--no-enable-notification',
                  '--command', shlex.join(['bash', '-c', command])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs', nargs='+', choices=sorted({**JOBS, **RERUNS}))
    p.add_argument('--code', required=True, help='commit-pinned code snapshot on Inspire (with ar_deps)')
    p.add_argument('--commit', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--submit', action='store_true', help='actually create the jobs')
    a = p.parse_args()
    for job in a.jobs:
        name, args = request(job, a.code, a.commit, a.smoke)
        args = ['inspire', '--account', 'sii-shared', *args, *([] if a.submit else ['--dry-run'])]
        result = subprocess.run(args, capture_output=True, text=True)
        print(json.dumps(dict(job=name, submitted=a.submit, returncode=result.returncode,
                              output=result.stdout[-2000:], error=result.stderr[-2000:]), ensure_ascii=False))


if __name__ == '__main__':
    main()
