"""Deploy committed source; submit CF now and Hybrid only after shared teacher is ready."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
CLI = shutil.which('inspire') or str(Path.home()/'AppData/Roaming/uv/tools/inspire-skill/Scripts/inspire.exe')
ENV = dict(os.environ, NO_PROXY='.sii.edu.cn', HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', PYTHONUTF8='1')
WORKSPACE, CPU = '分布式训练空间', 'CPU资源空间'
GROUP, IMAGE = '开发区-H100-cuda12.8版本-183核', 'ngc-pytorch:25.02-cuda12.8.0-py3'
JOBS = {mode: f'nancy-rqcd-{mode}-1113-v1' for mode in ('cf', 'hybrid')}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def cli(args, receipt, timeout=180):
    proc = subprocess.run([CLI, '--account', 'sii-shared', '--json', *args], env=ENV,
                          capture_output=True, encoding='utf-8', timeout=timeout)
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'{args[:2]} returned no JSON (exit {proc.returncode})') from exc
    write(HERE/'raw'/receipt, value)
    data = value.get('data') or {}
    if proc.returncode or not value.get('success') or isinstance(data, dict) and data.get('returncode'):
        raise RuntimeError(f'{args[:2]} failed: {value.get("error") or data}')
    if isinstance(data, dict) and 'output' in data:
        data['output'] = data['output'].split('Starting client (stdio mode)')[0].strip()
    return data


def remote(command, receipt, timeout=60):
    return cli(['notebook', 'exec', 'env-zzl', '--workspace', CPU, '--timeout', str(timeout),
                command+' # '], receipt, timeout+60)['output']


def stage():
    paths = sorted(REPO.glob('scripts/rqcd_mixer/*.py'))
    paths += sorted(REPO.glob('configs/rqcd_mixer/*'))
    paths += [REPO/'scripts/rqcd/data.py', REPO/'scripts/listwise_diffgrm/assets.py']
    for name in ('scripts/__init__.py', 'scripts/rqcd/__init__.py', 'scripts/listwise_diffgrm/__init__.py'):
        if (REPO/name).exists():
            paths.append(REPO/name)
    paths += [REPO/'docs/research/rqcd_mixer'/name for name in
              ('RQCD_MIXER_FULL_DESIGN_20261111.md', 'RQCD_TARGETS_DESIGN_20261113.md')]
    paths += [HERE/'worker.py', HERE/'runtime.locked.json']
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=REPO,
                       capture_output=True, check=True)
        if subprocess.check_output(['git', 'status', '--porcelain', '--', relative], cwd=REPO).strip():
            raise RuntimeError('Uncommitted deployment source: '+relative)
    mappings = {p.relative_to(REPO).as_posix(): p for p in paths}
    mappings.pop((HERE/'worker.py').relative_to(REPO).as_posix())
    mappings.pop((HERE/'runtime.locked.json').relative_to(REPO).as_posix())
    mappings['delivery/worker.py'] = HERE/'worker.py'
    mappings['runtime.locked.json'] = HERE/'runtime.locked.json'
    manifest = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'files': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in mappings.items()}}
    write(HERE/'SOURCE_MANIFEST.json', manifest)
    package = HERE/'code.tar.gz'
    with tarfile.open(package, 'w:gz') as tar:
        for name, path in mappings.items():
            tar.add(path, arcname=name)
        tar.add(HERE/'SOURCE_MANIFEST.json', arcname='SOURCE_MANIFEST.json')
    sha = hashlib.sha256(package.read_bytes()).hexdigest()
    remote('mkdir '+REMOTE, 'DIRECTORY.json')
    cli(['notebook', 'scp', 'env-zzl', package.as_posix(), REMOTE+'/code.tar.gz', '--workspace', CPU, '-t', '120'], 'UPLOAD.json')
    check = f'import hashlib; assert hashlib.sha256(open("code.tar.gz","rb").read()).hexdigest()=={sha!r}'
    result = remote(f'cd {REMOTE} && {PYTHON} -B -c {shlex.quote(check)} && tar -xzf code.tar.gz && {PYTHON} -B delivery/worker.py verify', 'READBACK.json')
    write(HERE/'DEPLOYMENT.json', dict(remote=REMOTE, code_commit=manifest['git_commit'], package_sha256=sha,
                                     readback=json.loads(result), jobs=JOBS))
    print(result, flush=True)


def snapshot():
    result = json.loads(remote(f'{PYTHON} -B {REMOTE}/delivery/worker.py snapshot', 'SNAPSHOT.json'))
    write(HERE/'LATEST.json', result)
    return result


def submit(mode):
    name = JOBS[mode]
    matches = cli(['job', 'list', '--workspace', WORKSPACE, '--keyword', name, '--all'], mode+'_EXISTS.json').get('items', [])
    exact = [row for row in matches if row['name'] == name]
    if len(exact) == 1:
        write(HERE/'raw'/f'{mode}_SUBMITTED.json', dict(reconciled=True, job=exact[0]))
        return exact[0]
    if exact or (HERE/'raw'/f'{mode}_INTENT.json').exists():
        raise RuntimeError('Ambiguous prior submission; inspect before another create: '+name)
    if mode == 'hybrid':
        state = snapshot()
        if not state.get('teacher_sha_verified'):
            raise RuntimeError('Teacher not ready')
    command = (f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 '
               'PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && '
               f'{PYTHON} -B -u delivery/worker.py {mode} > {mode}.log 2>&1')
    request = ['job', 'create', '--name', name, '--workspace', WORKSPACE, '--project', '公共科研项目',
               '--group', GROUP, '--quota', '1,20,200', '--image', IMAGE, '--nodes', '1', '--shm-size', '32',
               '--max-time', '24' if mode == 'cf' else '12', '--priority', '4', '--no-auto-fault-tolerance',
               '--no-enable-notification', '--command', shlex.join(['bash', '-c', command])]
    write(HERE/'raw'/f'{mode}_INTENT.json', dict(request=request, unix=time.time()))
    return cli(request, mode+'_SUBMITTED.json')


def watch():
    # Windows releases this lock automatically if the process dies. No stale PID guessing.
    import msvcrt
    lock = (HERE/'raw'/'dependency.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    deadline = time.time()+25*3600
    while time.time() < deadline:
        try:
            value = snapshot()
            write(HERE/'LIVE_DEPENDENCY.json', dict(pid=os.getpid(), unix=time.time(), stage='waiting_teacher',
                                                   teacher=value.get('teacher_progress')))
            if value.get('teacher_sha_verified'):
                result = submit('hybrid')
                write(HERE/'LIVE_DEPENDENCY.json', dict(pid=os.getpid(), unix=time.time(), stage='hybrid_submitted', result=result))
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return
            cf = cli(['job', 'status', JOBS['cf'], '--workspace', WORKSPACE], 'cf_STATUS.json')
            if cf.get('status') in ('FAILED', 'CANCELLED', 'STOPPED', 'SUCCEEDED'):
                write(HERE/'LIVE_DEPENDENCY.json', dict(pid=os.getpid(), unix=time.time(), stage='blocked_teacher_terminal', job=cf))
                return
        except Exception as exc:
            write(HERE/'LIVE_DEPENDENCY.json', dict(pid=os.getpid(), unix=time.time(), stage='retry_connection_or_submission', error=str(exc)))
            print(str(exc), flush=True)
        time.sleep(60)
    write(HERE/'LIVE_DEPENDENCY.json', dict(unix=time.time(), stage='deadline_reached'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'cf', 'hybrid', 'snapshot', 'watch'])
    args = p.parse_args()
    if args.action in JOBS:
        print(json.dumps(submit(args.action), ensure_ascii=False), flush=True)
    elif args.action == 'snapshot':
        print(json.dumps(snapshot(), ensure_ascii=False), flush=True)
    else:
        globals()[args.action]()
