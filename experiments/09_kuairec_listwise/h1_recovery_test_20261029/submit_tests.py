"""Explicit one-shot Inspire submissions for the user-authorized fixed-weight test."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
SOURCE = BASE + '/h1_recovery_20261028_r2'
ROOT = BASE + '/h1_recovery_test_20261029_r1'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python3.10'
PARENT = BASE + '/goal_20261027_r1/runs/H1/best.pt'
LAUNCHER_SHA = '10cfbc7483aab5238a37beeb8afae8379f78a41de506d5561e02a92d8cf87593'


def command(policy):
    launcher = ROOT + '/delivery/evaluate_test.py'
    output = ROOT + '/H1-' + policy
    logfile = ROOT + '/logs/' + policy + '.log'
    lines = [
        'set -euo pipefail',
        'set -o noclobber',
        'test ! -e ' + shlex.quote(output),
        'test ! -e ' + shlex.quote(logfile),
        'test "$(sha256sum ' + shlex.quote(launcher) + ' | cut -d " " -f1)" = ' + shlex.quote(LAUNCHER_SHA),
        'timeout --signal=TERM --kill-after=15s 2700s ' + shlex.join([
            PYTHON, '-u', '-B', launcher, '--root', SOURCE, '--parent', PARENT,
            '--policy', policy, '--output', output]) + ' > ' + shlex.quote(logfile) + ' 2>&1',
    ]
    return 'bash -c ' + shlex.quote('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', choices=('decay', 'freeze'), required=True)
    p.add_argument('--create', action='store_true', help='Actually submit; otherwise dry-run only')
    args = p.parse_args()
    mode = 'CREATE' if args.create else 'DRY_RUN'
    target = HERE / (args.policy.upper() + '_' + mode + '.json')
    if target.exists():
        raise FileExistsError('Inspect existing receipt; do not resubmit automatically: ' + str(target))
    name = 'nancy-h1-recovery-test-' + args.policy + '-20261029-r1'
    command_text = command(args.policy)
    cli = ['inspire', '--account', 'sii-shared', '--json', 'job', 'create',
        '--name', name, '--workspace', '分布式训练空间', '--project', '公共科研项目',
        '--group', '开发区-H100-cuda12.8版本-119核', '--quota', '1,10,200',
        '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3', '--nodes', '1', '--priority', '4',
        '--shm-size', '16', '--max-time', '0.8', '--no-auto-fault-tolerance',
        '--no-enable-notification', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '--env', 'OMP_NUM_THREADS=4',
        '--env', 'MKL_NUM_THREADS=4', '--env', 'OPENBLAS_NUM_THREADS=4',
        '--command', command_text]
    if not args.create:
        cli.append('--dry-run')
    if args.create:
        dry = json.loads((HERE / (args.policy.upper() + '_DRY_RUN.json')).read_text(encoding='utf-8'))
        plan = dry['data']
        if not dry['success'] or not plan['dry_run'] or plan['name'] != name or plan['command'] != command_text:
            raise ValueError('Existing dry-run does not match this submission')
    result = subprocess.run(cli, capture_output=True, text=True, encoding='utf-8', timeout=120)
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        payload = {'success': False, 'invalid_json': True, 'returncode': result.returncode}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / (args.policy.upper() + '_' + mode + '_REQUEST.json')).write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), argv=cli, returncode=result.returncode,
        output_root=ROOT, allocated_gpus=1, max_hours=0.8, unit_points_per_gpu_hour=1,
        user_authorization='2026-10-29: user requested evaluation of the three fixed models; two new tests only'),
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if result.returncode or not payload.get('success'):
        print(json.dumps(payload, ensure_ascii=False))
        raise SystemExit('Submission did not return success. Inspect existing Job before retrying.')
    print(json.dumps(payload if args.create else dict(status='dry_run_passed', name=name,
        resource=payload['data']['resource'], max_hours=payload['data']['max_time_hours']), ensure_ascii=False))


if __name__ == '__main__':
    main()
