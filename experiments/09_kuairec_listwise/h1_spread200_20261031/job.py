"""Bounded single-H100 launcher; never submits or retries a platform job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('root', 'python', 'manifest-sha', 'self-sha', 'parent-sha'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--mode', choices=('preflight', 'formal'), required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--resume')
    a = p.parse_args()
    start = time.monotonic()
    root = Path(a.root).resolve()
    code = root / 'code'
    if sha(__file__) != a.self_sha:
        raise ValueError('Launcher bytes changed')
    if not 120 <= a.max_seconds <= (1500 if a.mode == 'preflight' else 24*3600):
        raise ValueError('Unauthorized runtime')
    manifest = code / 'SPREAD200_EXECUTION_MANIFEST.json'
    if sha(manifest) != a.manifest_sha:
        raise ValueError('Execution manifest changed')
    content = json.loads(manifest.read_text(encoding='utf-8'))
    for name, expected in content['files'].items():
        file = code / name
        file.resolve().relative_to(code)
        if file.is_symlink() or file.stat().st_size != expected['bytes'] or sha(file) != expected['sha256']:
            raise ValueError('Scientific package changed: ' + name)
    inputs = json.loads((root / 'INPUTS.json').read_text(encoding='utf-8'))
    if sha(inputs['parent']) != a.parent_sha or content['parent_sha256'] != a.parent_sha:
        raise ValueError('Parent checkpoint changed')
    env = os.environ.copy()
    for key in ('NNODES', 'GROUP_WORLD_SIZE', 'SLURM_JOB_NUM_NODES'):
        if int(env.get(key, '1')) != 1:
            raise ValueError('Independent single-node process required')
    for key in list(env):
        if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT',
                   'NODE_RANK','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','NNODES','NPROC_PER_NODE',
                   'SLURM_PROCID','SLURM_LOCALID','SLURM_NTASKS'} or key.startswith(
                       ('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'ACCELERATE_', 'RDZV_')):
            env.pop(key)
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', DIFFGRM_ENGINE_ROOT=str(code / 'external/DiffGRM'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    jobs = root / 'jobs' / a.mode
    jobs.mkdir(parents=True, exist_ok=False)
    requested = {'signal': None, 'at': None}
    process = None
    def request(sig, _frame=None):
        if requested['signal'] is None:
            requested.update(signal=sig, at=time.monotonic())
        if process is not None and process.poll() is None:
            os.killpg(process.pid, sig)
    signal.signal(signal.SIGTERM, request)
    signal.signal(signal.SIGINT, request)
    deadline = start + a.max_seconds
    common = ['--parent', inputs['parent'], '--parent-sha', a.parent_sha,
              '--dataset', inputs['dataset'], '--assets', inputs['assets'],
              '--subset', inputs['subset'], '--warmup-predictions', inputs['warmup_predictions']]
    if a.mode == 'preflight':
        commands = [('preflight', [a.python, '-B', '-u', '-m', 'scripts.listwise_diffgrm.spread200_preflight'] +
            common + ['--output', str(root / 'preflight_r1'), '--reference-training', inputs['reference_training'],
                      '--max-seconds', str(a.max_seconds-90)])]
    else:
        gate = json.loads((root / 'PREFLIGHT_ACCEPTED.json').read_text(encoding='utf-8'))
        if gate['manifest_sha256'] != a.manifest_sha or gate['parent_sha256'] != a.parent_sha:
            raise ValueError('Preflight gate does not bind this execution')
        if gate['forecast_seconds'] > a.max_seconds-90:
            raise ValueError('Verified forecast exceeds formal allocation')
        commands = [('train', [a.python, '-B', '-u', '-m', 'scripts.listwise_diffgrm.spread200_run', 'train'] +
            common + ['--output', str(root / 'runs/H1-Spread200-R35')]),
                    ('test', [a.python, '-B', '-u', '-m', 'scripts.listwise_diffgrm.spread200_run', 'test'] +
            common + ['--output', str(root / 'runs/H1-Spread200-R35')])]
        if a.resume:
            commands[0][1].extend(['--resume', a.resume])
        (code / 'RUNNING_FORMAL').write_text('Scientific code is sealed for this run.\n', encoding='utf-8')
    result = {'mode': a.mode, 'status': 'running', 'automatic_retry': False, 'stages': [],
              'max_seconds': a.max_seconds, 'parent_sha256': a.parent_sha, 'manifest_sha256': a.manifest_sha}
    try:
        for label, command in commands:
            if requested['signal'] or time.monotonic() >= deadline-90:
                raise TimeoutError('Reserved shutdown time reached')
            with (jobs / (label + '.log')).open('xb') as log:
                process = subprocess.Popen(command, cwd=code, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    now = time.monotonic()
                    if requested['signal'] is None and now >= deadline-90:
                        request(signal.SIGTERM)
                    if requested['signal'] is not None and now >= min(deadline-15, requested['at']+60):
                        os.killpg(process.pid, signal.SIGKILL)
                    write(jobs / 'status.json', dict(result, stage=label, pid=process.pid, elapsed_seconds=now-start))
                    time.sleep(5)
                result['stages'].append({'stage': label, 'exit_code': process.returncode})
                if process.returncode:
                    raise RuntimeError(label + ' exited ' + str(process.returncode))
        result['status'] = 'preflight_complete' if a.mode == 'preflight' else 'training_validation_test_predictions_complete'
    except BaseException as exc:
        result.update(status='incomplete', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            request(signal.SIGTERM)
            try:
                process.wait(timeout=max(1,min(60,deadline-time.monotonic()-10)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        result['elapsed_seconds'] = time.monotonic()-start
        write(jobs / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
