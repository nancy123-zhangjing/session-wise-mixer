"""Single H100 execution only; resource submission and budget approval are external."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def environment():
    env = os.environ.copy()
    for key in ('NNODES', 'GROUP_WORLD_SIZE', 'SLURM_JOB_NUM_NODES', 'SLURM_NNODES'):
        if int(env.get(key, '1')) != 1:
            raise ValueError('Independent single-node process required: ' + key)
    for key in ('NODE_RANK', 'GROUP_RANK', 'SLURM_NODEID'):
        if int(env.get(key, '0')) != 0:
            raise ValueError('Unexpected additional node: ' + key)
    exact = {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
             'NODE_RANK', 'GROUP_RANK', 'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
             'NNODES', 'NPROC_PER_NODE', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS',
             'SLURM_NTASKS_PER_NODE', 'SLURM_STEP_NUM_TASKS'}
    for key in list(env):
        if key in exact or key.startswith(('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'MV2_', 'ACCELERATE_', 'RDZV_')):
            env.pop(key)
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--preparer-sha', required=True)
    parser.add_argument('--launcher-sha', required=True)
    parser.add_argument('--branch', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--authorized-seconds', required=True, type=int)
    args = parser.parse_args(argv)
    started = time.monotonic()
    if not 16 <= args.authorized_seconds <= 86400:
        raise ValueError('Runtime must include 15 reserved shutdown seconds')
    root, code = Path(args.root).resolve(), Path(args.root).resolve() / 'code'
    preparer = root / 'delivery/prepare.py'
    logs = root / 'jobs' / args.branch
    logs.mkdir(parents=True, exist_ok=False)
    deadline = started + args.authorized_seconds
    current = {'process': None, 'signal': None, 'signal_at': None}
    result = dict(branch=args.branch, status='starting', automatic_retry=False,
                  authorized_seconds=args.authorized_seconds, shutdown_seconds_reserved=15)

    def forward(signum):
        child = current['process']
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def stop(signum, _frame):
        if current['signal'] is None:
            current['signal'], current['signal_at'] = signum, time.monotonic()
        forward(signum)

    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)

    def run(label, command, seconds=None):
        now = time.monotonic()
        limit = min(deadline - 15 - now, seconds or args.authorized_seconds)
        if limit <= 0 or current['signal']:
            raise TimeoutError('Authorized runtime exhausted or shutdown requested')
        with (logs / (label + '.stdout.log')).open('xb') as out, (logs / (label + '.stderr.log')).open('xb') as err:
            child = subprocess.Popen(command, cwd=code, env=env, stdout=out, stderr=err, start_new_session=True)
            current['process'] = child
            stage_deadline = now + limit
            timed_out = False
            try:
                while True:
                    now = time.monotonic()
                    if current['signal'] is None and now >= stage_deadline:
                        timed_out = True
                        stop(signal.SIGTERM, None)
                    stopping = current['signal'] is not None
                    end = min(deadline, current['signal_at'] + 15) if stopping else stage_deadline
                    if stopping and now >= end:
                        forward(signal.SIGKILL)
                        # SIGKILL is already sent at the reserved deadline; a
                        # nonblocking reap cannot spend extra budget waiting.
                        rc = child.poll()
                        break
                    try:
                        rc = child.wait(timeout=min(.25, max(.001, end - now)))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                current['process'] = None
        if timed_out:
            raise TimeoutError('Bounded child exceeded authorized runtime: ' + label)
        if rc != 0 or current['signal']:
            raise RuntimeError(f'{label} failed: exit={rc}; signal={current["signal"]}')

    try:
        if os.name != 'posix':
            raise ValueError('Linux single-GPU bounded runtime required')
        if sha256(__file__) != args.launcher_sha or sha256(preparer) != args.preparer_sha:
            raise ValueError('Deployment launcher or package verifier bytes changed')
        env = environment()
        env['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
        # Keep the full source/input verification inside the same supervised
        # deadline and preserve its logs/failure receipt before any GPU query.
        run('verify', [args.python, '-B', str(preparer), 'verify', '--code-root', str(code),
                      '--manifest-sha', args.manifest_sha, '--references', 'remote', '--verify-results'])
        inputs = json.loads((code / 'FREEZE35_PARENTS.json').read_text(encoding='utf-8'))
        parent = inputs['parents'][args.branch]
        output = root / 'runs' / parent['name']
        result['output'] = str(output)
        if output.exists():
            raise FileExistsError('Automatic restart/reuse forbidden: ' + str(output))
        inventory = "import json,torch; print(json.dumps([dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory,uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]))"
        run('inventory', [args.python, '-B', '-c', inventory], 60)
        devices = json.loads((logs / 'inventory.stdout.log').read_text())
        if len(devices) != 1 or 'H100' not in devices[0]['name'] or devices[0]['memory'] < 75_000_000_000 or 'MIG' in devices[0]['uuid']:
            raise ValueError('Exactly one full H100 is required')
        request = dict(vars(args), parent=parent, devices=devices, threads=4, formal_budget_approval_required=True)
        write(logs / 'request.json', request)
        common = ['--branch', args.branch, '--parent', parent['remote'], '--dataset', inputs['dataset']['remote'],
                  '--assets', inputs['assets']['remote'], '--references', str(code / 'FREEZE35_REFERENCE_RESULTS.json'),
                  '--device', 'cuda:0', '--output', str(output)]
        if args.branch == 'A':
            common += ['--ancestor', inputs['ancestors']['A']['remote']]
        for stage in ('train', 'test'):
            result['status'] = stage + '_running'
            write(logs / 'status.json', result)
            run(stage, [args.python, '-B', '-m', 'scripts.listwise_diffgrm.recovery35_run', stage, *common])
            result[stage + '_process_complete'] = True
        result['status'] = 'predictions_complete_independent_audit_pending'
        return 0
    except BaseException as exc:
        result.update(status='failed_preserve_state_no_retry', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        result['signal'] = current['signal']
        write(logs / 'result.json', result)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
