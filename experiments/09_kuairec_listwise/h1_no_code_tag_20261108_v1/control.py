"""Prepare the single GPU job, submit once, and read progress via env-zzl."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLI = Path.home() / 'AppData/Roaming/uv/tools/inspire-skill/Scripts/inspire.exe'
REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_no_code_tag_20261108_v1'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
JOB = 'nancy-s09-nocodetag-1108-v1'
WORKSPACE = '分布式训练空间'
GROUP = '开发区-H100-cuda12.8版本-183核'
CPU = 'CPU资源空间'
IMAGE = 'ngc-pytorch:25.02-cuda12.8.0-py3'
ENV = dict(os.environ, NO_PROXY='.sii.edu.cn', HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', PYTHONUTF8='1')


def write(name, value):
    target = HERE / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(target)


def cli(args, receipt=None, timeout=150):
    proc = subprocess.run([str(CLI), '--json', '--account', 'sii-shared', *args], env=ENV,
                          capture_output=True, encoding='utf-8', timeout=timeout)
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'{args[:2]} returned no platform JSON (exit {proc.returncode})') from exc
    data = value.get('data') or {}
    if isinstance(data, dict) and 'output' in data:
        data['output'] = data['output'].split('Starting client (stdio mode)')[0].strip()
    if receipt:
        write(receipt, value)
    if proc.returncode or not value.get('success') or isinstance(data, dict) and data.get('returncode'):
        raise RuntimeError(f'{args[:2]} failed: {value.get("error") or data}')
    return data


def remote(command, receipt=None, timeout=45):
    return cli(['notebook', 'exec', 'env-zzl', '--workspace', CPU, '--timeout', str(timeout),
                command + ' # '], receipt, timeout + 60)['output']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle():
    frozen = REPO / 'outputs/stage09_restore/goal_20261027_r1/code'
    paths = [f'scripts/listwise_diffgrm/{name}.py' for name in ('alignment', 'runtime', 'training')]
    paths += [f'external/DiffGRM/genrec/models/DIFF_GRM/{name}.py' for name in ('cross_pse', 'joint_cross_pse')]
    write('SOURCE_DELTA.json', {path: dict(before=digest(frozen / path), after=digest(REPO / path)) for path in paths})
    package = HERE / 'launch.tar.gz'
    with tarfile.open(package, 'w:gz') as tar:
        for path in paths:
            tar.add(REPO / path, arcname='overlay/' + path)
        for name in ('experiment.py', 'SOURCE_DELTA.json'):
            tar.add(HERE / name, arcname=name)
        tar.add(HERE.parent / 'h1_replay_r5_r20_20261105_r1/replay.py', arcname='replay_support.py')
    return package


def resources():
    queries = [
        (['job', 'quota', '--workspace', WORKSPACE, '--group', GROUP, '--all'], 'LIVE_QUOTA.json'),
        (['image', 'list', '--workspace', WORKSPACE, '--keyword', 'ngc-pytorch:25.02', '--limit', '5'], 'LIVE_IMAGE.json'),
        (['resources', 'availability', '--workspace', WORKSPACE, '--group', GROUP], 'LIVE_AVAILABILITY.json'),
        (['resources', 'policy', '--workspace', WORKSPACE], 'LIVE_POLICY.json'),
        (['project', 'detail', '公共科研项目'], 'LIVE_PROJECT.json'),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(cli, args, receipt) for args, receipt in queries]
        for job in jobs:
            job.result()
    print('Live resource receipts saved', flush=True)


def prepare():
    package = bundle()
    remote(f'mkdir {REMOTE}', 'DIRECTORY.json')
    cli(['notebook', 'scp', 'env-zzl', package.as_posix(), REMOTE + '/launch.tar.gz',
         '--workspace', CPU, '-t', '120'], 'UPLOAD.json')
    expected = digest(package)
    script = (f'cd {REMOTE} && sha256sum launch.tar.gz && '
              f'printf %s {shlex.quote(expected + "  launch.tar.gz")} | sha256sum -c - && '
              f'tar -xzf launch.tar.gz && {PYTHON} -B experiment.py stage')
    remote(script, 'STAGE.json', timeout=120)
    print(f'Staged and verified {expected}', flush=True)


def job_request():
    script = (f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
              'PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
              'CUBLAS_WORKSPACE_CONFIG=:4096:8 && '
              f'{PYTHON} -B -u experiment.py > training.log 2>&1')
    return ['job', 'create', '--name', JOB, '--workspace', WORKSPACE, '--project', '公共科研项目',
            '--group', GROUP, '--quota', '1,20,200', '--image', IMAGE, '--nodes', '1',
            '--shm-size', '32', '--max-time', '12', '--priority', '4', '--no-auto-fault-tolerance',
            '--no-enable-notification', '--command', shlex.join(['bash', '-c', script])]


def submit():
    if (HERE / 'SUBMISSION_INTENT.json').exists():
        raise RuntimeError('Prior submission intent exists. Inspect platform state before another attempt.')
    if cli(['job', 'list', '--workspace', WORKSPACE, '--keyword', JOB, '--all'], 'JOB_ABSENCE.json').get('items'):
        raise RuntimeError('Matching job exists')
    request = job_request()
    cli(request + ['--dry-run'], 'DRY_RUN.json')
    with (HERE / 'SUBMISSION_INTENT.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(job=JOB, request=request, unix=time.time()), stream, ensure_ascii=False, indent=2)
    print(json.dumps(cli(request, 'SUBMISSION_RESULT.json'), ensure_ascii=False), flush=True)


def status():
    with ThreadPoolExecutor(max_workers=2) as pool:
        job = pool.submit(cli, ['job', 'status', JOB, '--workspace', WORKSPACE], 'PLATFORM_LATEST.json')
        log = pool.submit(remote, f'cd {REMOTE} && if test -f training.log; then tail -n 7 training.log; fi; '
                          'if test -f runs/H1-NoCodeTag/training.jsonl; then tail -n 1 runs/H1-NoCodeTag/training.jsonl; fi; '
                          'if test -f audit.log; then tail -n 3 audit.log; fi; '
                          'for x in runs/H1-NoCodeTag/checkpoints/R??.json GPU_COMPLETE.json; '
                          'do if test -f "$x"; then cat "$x"; fi; done', 'PROGRESS_LATEST.json')
        errors = []
        for label, result in (('platform', job), ('shared log', log)):
            try:
                value = result.result()
                print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors.append(f'{label}: {exc}')
        if errors:
            raise RuntimeError('\n'.join(errors))


def stage_audit():
    core = HERE.parent / 'h1_r26_freeze_20261106_v1/audit_analysis_core.py'
    sources = (HERE / 'accept.py', core)
    for path in sources:
        cli(['notebook', 'scp', 'env-zzl', path.as_posix(), REMOTE + '/' + path.name,
             '--workspace', CPU, '-t', '120'], 'UPLOAD_' + path.stem + '.json')
    expected = '\n'.join(f'{digest(path)}  {path.name}' for path in sources)
    print(remote(f'cd {REMOTE} && printf %s {shlex.quote(expected)} | sha256sum -c -',
                 'AUDIT_SOURCE_CHECK.json'), flush=True)


def start_audit():
    job = cli(['job', 'status', JOB, '--workspace', WORKSPACE], 'PLATFORM_LATEST.json')
    if job['status'] != 'SUCCEEDED':
        raise ValueError('GPU job must be terminal and successful before CPU acceptance')
    command = (f'cd {REMOTE} && test -f GPU_COMPLETE.json && test -f accept.py && '
               f'test -f audit_analysis_core.py && mkdir audit_started && '
               f'(nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {PYTHON} -B -u accept.py '
               '> audit.log 2>&1 < /dev/null & echo $! > audit_started/pid)')
    remote(command, 'AUDIT_LAUNCH.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['resources', 'prepare', 'submit', 'status', 'stage_audit', 'start_audit'])
    args = parser.parse_args()
    globals()[args.action]()
