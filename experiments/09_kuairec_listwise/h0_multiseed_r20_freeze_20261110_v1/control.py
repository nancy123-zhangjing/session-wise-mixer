"""Isolated H0 Freeze submissions; reuse the existing platform transport."""
import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
c = runpy.run_path(str(HERE.parent / 'multiseed_freeze_20261109_v1/control.py'))['c']
c.HERE = HERE
REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/' + HERE.name


def prepare():
    c.resources()
    help_result = subprocess.run([str(c.CLI), 'job', 'create', '--help'], env=c.ENV,
                                 capture_output=True, encoding='utf-8', check=True)
    (HERE / 'CREATE_HELP.txt').write_text(help_result.stdout, encoding='utf-8')
    c.remote('mkdir -p ' + REMOTE)
    files = ('experiment.py', 'protocol.py', 'prepare_remote.py', 'accept_remote.py',
             'PLAN.md', 'config_seed2025.json', 'config_seed2026.json')
    for name in files:
        c.cli(['notebook', 'scp', 'env-zzl', (HERE / name).as_posix(), REMOTE + '/' + name,
               '--workspace', c.CPU, '-t', '120'], 'UPLOAD_' + name + '.json')
    value = json.loads(c.remote(c.PYTHON + ' -B ' + REMOTE + '/prepare_remote.py', 'PREPARE_READBACK.json', 60))
    for name, digest in value['runner_sha256'].items():
        assert c.digest(HERE / name) == digest
    for seed in (2025, 2026):
        assert c.digest(HERE / f'config_seed{seed}.json') == value['config_sha256'][str(seed)]
    c.write('MANIFEST.json', value)
    print('Parent metadata, config, source and uploads verified')


def name_for(seed):
    if seed not in (2025, 2026):
        raise ValueError('Unapproved seed')
    return f'nancy-s09-h0freeze-s{seed}-1110-v1'


def submit(seed):
    name = name_for(seed)
    intent = HERE / f'{seed}_SUBMISSION_INTENT.json'
    if intent.exists():
        raise ValueError('Submission intent exists; inspect instead of resubmitting')
    if c.cli(['job', 'list', '--workspace', c.WORKSPACE, '--keyword', name, '--all'], f'{seed}_ABSENCE.json').get('items'):
        raise ValueError('Job already exists')
    script = f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u experiment.py --seed {seed} > {seed}_H0.log 2>&1'
    request = ['job', 'create', '--name', name, '--workspace', c.WORKSPACE,
               '--project', '公共科研项目', '--group', c.GROUP, '--quota', '1,20,200',
               '--image', c.IMAGE, '--nodes', '1', '--shm-size', '32', '--max-time', '12',
               '--priority', '4', '--no-auto-fault-tolerance', '--no-enable-notification',
               '--command', shlex.join(['bash', '-c', script])]
    c.cli(request + ['--dry-run'], f'{seed}_DRY_RUN.json')
    c.write(intent.name, dict(job=name, request=request, remote=REMOTE))
    print(c.cli(request, f'{seed}_SUBMISSION_RESULT.json'), flush=True)


def status():
    for seed in (2025, 2026):
        if not (HERE / f'{seed}_SUBMISSION_INTENT.json').exists():
            continue
        state = c.cli(['job', 'status', name_for(seed), '--workspace', c.WORKSPACE], f'{seed}_STATUS.json')
        print(state, flush=True)
        if state['status'] in ('QUEUING', 'PENDING', 'CREATED'):
            continue
        try:
            print(c.remote(f'if test -f {REMOTE}/{seed}_H0.log; then tail -n 4 {REMOTE}/{seed}_H0.log; fi',
                           f'{seed}_LOG_TAIL.json', 30), flush=True)
        except RuntimeError as error:
            print(f'Log readback unavailable for {seed}: {error}', flush=True)


def accept():
    for seed in (2025, 2026):
        state = c.cli(['job', 'status', name_for(seed), '--workspace', c.WORKSPACE], f'{seed}_STATUS.json')
        assert state['status'] == 'SUCCEEDED'
        workers = c.cli(['job', 'instances', name_for(seed), '--workspace', c.WORKSPACE], f'{seed}_INSTANCES.json')['items']
        assert len(workers) == 1 and workers[0]['status'] == 'instance_success'
    value = json.loads(c.remote(c.PYTHON + ' -B ' + REMOTE + '/accept_remote.py', 'ACCEPT_READBACK.json', 600))
    value['all_gpu_released'] = True
    c.write('FINAL_ACCEPTANCE.json', value)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    elif sys.argv[1] == 'submit': submit(int(sys.argv[2]))
    elif sys.argv[1] == 'status': status()
    elif sys.argv[1] == 'accept': accept()
