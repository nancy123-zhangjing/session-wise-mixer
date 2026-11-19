"""Submit one dependency-ready L5O arm using a fresh remote readback."""
import argparse
import json
from pathlib import Path
import shlex

from control import PYTHON, REMOTE, cli


WAVE = REMOTE.replace('l5o_stage0_20261117_v1', 'l5o_wave_20261117_v1')
CALIBRATION = REMOTE.replace('l5o_stage0_20261117_v1', 'l5o_sasrec_20261117_v1') + '/run/COMPLETE.json'
WORKSPACE = '分布式训练空间'
DEPENDENCY = {'H1': 'h_warmup', 'T0A': 'tiger', 'T1A': 'tiger', 'CF-M1': 'cf_rq'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('H0', 'H1', 'T0A', 'T1A', 'CF-M0', 'CF-M1'))
    parser.add_argument('--readback', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cf-repeat-recovery', action='store_true')
    args = parser.parse_args()
    envelope = json.loads(args.readback.read_text(encoding='utf-8'))
    if envelope['returncode'] != 0:
        raise ValueError('Remote readback failed')
    ready = json.loads(envelope['output'].splitlines()[-1])
    gate = ready.get('sasrec', {})
    metrics = gate.get('test', {}).get('1.0', {})
    if (not ready.get('source') or not ready.get('calibration_artifacts_verified') or
            gate.get('passed') is not True or metrics.get('samples') != 7176 or
            not .25 <= metrics['imp_hit@5'] <= .45 or not .05 <= metrics['ev_ndcg@5'] <= .12):
        raise ValueError('Require verified remote source and passed full SASRec test')
    dependency = DEPENDENCY.get(args.arm)
    if dependency and not ready.get(dependency):
        raise ValueError(f'{args.arm} is waiting for {dependency}')
    if args.arm.startswith('H') and not ready.get('h_source'):
        raise ValueError('Require the verified H0/H1 warmup checkpoint fix')
    version, root = 'v1', WAVE
    code = WAVE + ('/code_c10c357' if args.arm.startswith('H') else '/code')
    if args.cf_repeat_recovery:
        if args.arm not in ('CF-M0', 'CF-M1') or not ready.get('cf_repeat_recovery_verified') or not ready.get('cf_rq'):
            raise ValueError('Recovery requires the verified CF overlay and shared RQ parent')
        previous = f'nancy-l5o-{args.arm.lower()}-1117-v1'
        old = cli(['job', 'list', '--workspace', WORKSPACE, '--keyword', previous], f'{previous}_terminal.json')
        if not any(item['name'] == previous and item['status'] == 'FAILED' for item in old['items']):
            raise ValueError('Previous CF attempt must be confirmed failed')
        version, root = 'v2', WAVE + '/cf_repeat_v2'
        code = WAVE + '/code_cf_l5o_repeat_v2'
    job = f'nancy-l5o-{args.arm.lower()}-1117-{version}'
    existing = cli(['job', 'list', '--workspace', WORKSPACE, '--keyword', job], f'{job}_lookup.json')
    if any(item['name'] == job for item in existing['items']):
        raise ValueError(f'{job} already exists; inspect its status before any retry')
    worker = shlex.join([PYTHON, '-u', '-m', 'scripts.kuairec_data.l5o_wave_worker', args.arm,
        '--root', root, '--calibration', CALIBRATION, '--dataset', REMOTE + '/dataset',
        '--assets', REMOTE + '/assets', '--cf-runtime',
        code + '/experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json'])
    command = f'cd {code} && export PYTHONPATH={code}/ar_deps:{code} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && {worker}'
    request = ['job', 'create', '--name', job, '--workspace', WORKSPACE, '--project', '公共科研项目',
        '--group', '开发区-H100-cuda12.8版本-183核', '--quota', '1,20,200',
        '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3', '--nodes', '1', '--shm-size', '32',
        '--max-time', '12', '--priority', '4', '--no-auto-fault-tolerance', '--no-enable-notification',
        '--command', shlex.join(['bash', '-c', command])]
    dry = cli(request + ['--dry-run'], f'{job}_dry_run.json')
    result = dry if args.dry_run else cli(request, f'{job}_submitted.json')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
