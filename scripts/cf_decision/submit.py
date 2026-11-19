"""Build (and only with --submit, create) the one-GPU decision-rule job. Default is a dry run."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess

STAGE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
GROUP = '开发区-H100-cuda12.8版本-183核'
ROOT = f'{STAGE}/cf_decision_rules_20261118'


def request(code, commit, name, output='runs', extra=()):
    worker = [PYTHON, '-u', '-m', 'scripts.cf_decision.run', '--wave', f'{STAGE}/l5o_wave_20261117_v1',
              '--dataset', f'{STAGE}/l5o_stage0_20261117_v1/dataset', '--output', f'{ROOT}/{output}', '--device', 'cuda',
              *extra]
    command = (f'set -o pipefail && mkdir -p {ROOT}/logs && cd {code} && '
               f'export PYTHONPATH={code}/ar_deps:{code} CODE_COMMIT={commit} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 '
               f'CUBLAS_WORKSPACE_CONFIG=:4096:8 && {shlex.join(worker)} 2>&1 | tee -a {ROOT}/logs/{name}.log')
    return ['job', 'create', '--name', name, '--workspace', '分布式训练空间', '--project', '公共科研项目',
            '--group', GROUP, '--quota', '1,20,200', '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3', '--nodes', '1',
            '--shm-size', '32', '--max-time', '8', '--priority', '4', '--no-auto-fault-tolerance',
            '--no-enable-notification', '--command', shlex.join(['bash', '-c', command])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', required=True, help='commit-pinned code snapshot on Inspire (with ar_deps)')
    p.add_argument('--commit', required=True)
    p.add_argument('--name', default='nancy-cf-decision-rules-1118')
    p.add_argument('--output', default='runs', help='output directory under the decision-rule root')
    p.add_argument('--run-args', default='', help='extra arguments for scripts.cf_decision.run, e.g. "--ks 32 64 --lists 0"')
    p.add_argument('--submit', action='store_true', help='actually create the job')
    a = p.parse_args()
    args = ['inspire', '--account', 'sii-shared', *request(a.code, a.commit, a.name, a.output, shlex.split(a.run_args))]
    if not a.submit:
        args.append('--dry-run')
    result = subprocess.run(args, capture_output=True, text=True)
    print(json.dumps(dict(job=a.name, submitted=a.submit, returncode=result.returncode,
                          output=result.stdout[-2000:], error=result.stderr[-2000:]), ensure_ascii=False))


if __name__ == '__main__':
    main()
