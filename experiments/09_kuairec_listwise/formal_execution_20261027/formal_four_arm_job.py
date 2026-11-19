"""One platform Job, four independent frozen Stage09 training processes.

Local preparation only until explicitly deployed. No DDP, automatic resume,
scientific-code edits, checkpoint writes or additional evaluation here.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
ACCEPTANCE_SHA = '2c255470cdd58b2c13dd50fb130aa750f1609306720081ab0e3cd33050cb2831'
BINDING_SHA = '871e480d2a77d7e22bfed091d02065f0cdc42eb1db42cf20d43519634959e80a'
ARMS = ('J0', 'J1', 'H0', 'H1')
LAYOUTS = {'one_gpu_four_arms': {'J0': 0, 'J1': 0, 'H0': 0, 'H1': 0},
           'two_gpu_mixed_pairs': {'J0': 0, 'J1': 1, 'H0': 0, 'H1': 1}}
EXIT_GRACE = 60.0
CLEANUP_RESERVE = 15.0
TERM_GRACE = 10.0
KILL_SIGNAL = getattr(signal, 'SIGKILL', 9)  # Numeric fallback only for Windows CPU simulations.

# Separate bounded process: keeps the supervisor free of CUDA contexts. The
# inherited CVD restriction is respected; no physical index is guessed.
INVENTORY = r'''
import csv, json, os, subprocess, torch
rows = list(csv.reader(subprocess.run(
    ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
    check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()))
inventory = [{'uuid': r[0].strip(), 'name': r[1].strip()} for r in rows]
visible = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    u = str(getattr(p, 'uuid', ''))
    if u and not u.startswith('GPU-'): u = 'GPU-' + u
    visible.append({'uuid': u, 'name': p.name})
print(json.dumps({'status':'passed', 'inventory':inventory, 'torch_visible':visible,
    'inherited_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'nvidia_visible_devices':os.environ.get('NVIDIA_VISIBLE_DEVICES')}), flush=True)
'''


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def owned(relative):
    path = ROOT / relative
    need(path.is_relative_to(ROOT) and '..' not in path.parts, 'Path outside fixed Stage09 root')
    for candidate in (path, *path.parents):
        need(not candidate.is_symlink(), f'Symlink refused: {candidate}')
        if candidate == ROOT:
            break
    need(path.resolve(strict=False) == path, f'Path escapes fixed Stage09 root: {path}')
    return path


def single_environment():
    env = dict(os.environ)
    removed = []
    for name in list(env):
        if (name in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                     'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'NODE_RANK',
                     'MASTER_ADDR', 'MASTER_PORT'} or
                name.startswith(('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'MV2_'))):
            removed.append(name); env.pop(name)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=(ROOT / 'cache/huggingface').as_posix(), TMPDIR=(ROOT / 'tmp').as_posix(),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env, sorted(removed)


def training_argv(arm):
    need(arm in ARMS, 'Unknown arm')
    return [PYTHON, '-u', '-B', '-m', 'scripts.listwise_diffgrm.run',
        '--config', (ROOT / 'config/efficiency_r2.json').as_posix(), '--architecture', arm[0],
        '--seed', '2024', 'train', '--device', 'cuda:0', '--precision', 'bf16',
        '--arm', arm, '--shared-warmup', (ROOT / ('runs/warmup_' + arm[0] + '/last.pt')).as_posix(),
        '--output', (ROOT / ('runs/' + arm)).as_posix()]


def acceptance_argv(mode, architecture, *, arm=None, warmup_sha=None):
    argv = [PYTHON, '-u', '-B', (ROOT / 'delivery/formal_branch_acceptance.py').as_posix(), mode,
            '--architecture', architecture, '--manifest-sha256', BINDING_SHA]
    if mode == 'ready':
        argv += ['--arm', arm, '--accepted-warmup-sha256', warmup_sha]
    return argv


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def validate_inventory(devices, expected_count):
    visible, container = devices['torch_visible'], devices['inventory']
    ids = [row['uuid'] for row in visible]
    need(devices.get('status') == 'passed' and len(visible) == len(container) == expected_count and
         len(set(ids)) == expected_count and
         all(re.fullmatch('GPU-[0-9a-fA-F-]+', item) for item in ids) and
         set(ids) == {row['uuid'] for row in container} and
         all('H100' in row['name'] for row in visible + container),
         'Actual container/torch visible GPU UUIDs or H100 count differs from selected layout')
    return ids


@dataclass
class Child:
    label: str
    process: object
    streams: list
    stdout: Path
    started: float
    marker: object = None
    finished_check: object = None
    timeout: float | None = None
    marker_at: float | None = None
    exited_at: float | None = None
    done: bool = False


class Supervisor:
    """Owned-session lifecycle; injectable process/clock functions are CPU-tested."""
    def __init__(self, hard_deadline, *, clock=time.monotonic, sleep=time.sleep,
                 popen=subprocess.Popen, alive=group_alive, send=None):
        self.hard_deadline = hard_deadline
        self.clock, self.sleep, self.popen, self.alive = clock, sleep, popen, alive
        self.send = send or (lambda pid, sig: os.killpg(pid, sig))
        self.children = []
        self.stop_reason = None
        self.signal_history = []

    def launch(self, label, argv, env, cwd, stdout, *, marker=None, finished_check=None, timeout=None):
        self.pump()
        out = Path(stdout).open('xb')
        err = Path(str(stdout) + '.stderr').open('xb')
        try:
            process = self.popen(argv, env=env, cwd=str(cwd), stdout=out, stderr=err,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        except BaseException:
            out.close(); err.close(); raise
        child = Child(label, process, [out, err], Path(stdout), self.clock(), marker, finished_check, timeout)
        self.children.append(child)
        return child

    def pump(self):
        if self.stop_reason:
            raise RuntimeError(self.stop_reason)
        need(self.clock() < self.hard_deadline - CLEANUP_RESERVE, 'execution_deadline')
        for child in self.children:
            if child.done:
                continue
            code = child.process.poll()
            if code is not None and code != 0:
                raise RuntimeError(f'child_failed:{child.label}:exit={code}')
            now = self.clock()
            if child.marker is not None and child.marker() and child.marker_at is None:
                child.marker_at = now
            if code is None:
                if child.timeout is not None:
                    need(now - child.started < child.timeout, f'child_timeout:{child.label}')
                if child.marker_at is not None:
                    need(now - child.marker_at < EXIT_GRACE, f'exit_grace_expired:{child.label}')
                continue
            if child.exited_at is None:
                child.exited_at = now
                if child.finished_check is not None:
                    child.finished_check()
            if self.alive(child.process.pid):
                grace_start = child.marker_at if child.marker_at is not None else child.exited_at
                need(now - grace_start < EXIT_GRACE, f'process_group_did_not_drain:{child.label}')
                continue
            child.done = True
            for stream in child.streams:
                stream.close()

    def wait_until(self, predicate):
        while True:
            self.pump()
            if predicate():
                return
            self.sleep(min(.2, max(0, self.hard_deadline - CLEANUP_RESERVE - self.clock())))

    def wait(self, child):
        self.wait_until(lambda: child.done)

    def _live(self):
        # Reap exited leaders before checking their still-owned descendants.
        for child in self.children:
            child.process.poll()
        return [child for child in self.children if not child.done and self.alive(child.process.pid)]

    def _signal(self, children, number):
        for child in children:
            try:
                self.send(child.process.pid, number)
                self.signal_history.append(dict(label=child.label, signal=int(number), at=self.clock()))
            except ProcessLookupError:
                pass

    def cleanup(self):
        live = self._live()
        if live:
            self._signal(live, signal.SIGTERM)
            until = min(self.clock() + TERM_GRACE, self.hard_deadline - 5)
            while self._live() and self.clock() < until:
                self.sleep(min(.1, until - self.clock()))
            self._signal(self._live(), KILL_SIGNAL)
            while self._live() and self.clock() < self.hard_deadline:
                self.sleep(min(.1, self.hard_deadline - self.clock()))
        survivors = [child.label for child in self._live()]
        for child in self.children:
            child.process.poll()
            for stream in child.streams:
                stream.close()
        return survivors


def completion(arm):
    output = owned('runs/' + arm)
    need(not (output / '.running').exists(), f'{arm}: runner lock remains after exit')
    report = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
    need(report['arm'] == arm and report['r_updates'] == 24255 and report['t_updates'] == 400,
         f'{arm}: missing complete 35R/400T receipt')
    need(report['best_epoch'] in range(5, 36, 3) and math.isfinite(report['best_score']),
         f'{arm}: invalid selected-best metadata')


def gate(supervisor, directory, environment, mode, architecture, accepted, *, arm=None):
    label = mode + '_' + (arm or architecture)
    env = dict(environment, CUDA_VISIBLE_DEVICES='')
    child = supervisor.launch(label, acceptance_argv(mode, architecture, arm=arm,
        warmup_sha=accepted[architecture]), env, ROOT / 'code', directory / (label + '.json'))
    supervisor.wait(child)
    result = json.loads(child.stdout.read_text(encoding='utf-8'))
    need(result['status'] == 'passed' and result['mode'] == mode and
         result['architecture'] == architecture and result['manifest_sha256'] == BINDING_SHA,
         f'{label}: acceptance receipt differs')
    if mode == 'ready':
        need(result['arm'] == arm and result['accepted_warmup_sha256'] == accepted[architecture],
             f'{label}: warmup receipt differs')
    else:
        need(result['last']['sha256'] == accepted[architecture] and result['exact_state_equal_except_arm'],
             f'{label}: initial pair differs from accepted warmup')
    return dict(path=str(child.stdout), sha256=sha(child.stdout))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=list(LAYOUTS), required=True)
    parser.add_argument('--max-seconds', type=float, required=True,
        help='Whole wrapper allocation budget, including prechecks and cleanup; root must reserve remaining Goal work')
    parser.add_argument('--platform-deadline-unix', type=float, required=True,
        help='Actual Job allocation hard stop; no resetting this deadline on retries')
    parser.add_argument('--accepted-warmup-j-sha256', required=True)
    parser.add_argument('--accepted-warmup-h-sha256', required=True)
    parser.add_argument('--attempt', required=True, help='New receipt directory suffix, not an automatic resume')
    args = parser.parse_args(argv)
    if not (math.isfinite(args.max_seconds) and args.max_seconds > CLEANUP_RESERVE + EXIT_GRACE and
            math.isfinite(args.platform_deadline_unix)):
        parser.error('Finite positive total budget and actual platform deadline are required')
    if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}', args.attempt):
        parser.error('Unsafe attempt identifier')
    for value in (args.accepted_warmup_j_sha256, args.accepted_warmup_h_sha256):
        if not re.fullmatch('[0-9a-f]{64}', value):
            parser.error('Both CPU-accepted complete warmup SHA256 values are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    need(os.name == 'posix', 'Formal process-group wrapper is Linux-only; never run it locally on Windows')
    started, started_wall = time.monotonic(), time.time()
    remaining = min(args.max_seconds, args.platform_deadline_unix - started_wall)
    need(remaining > CLEANUP_RESERVE + EXIT_GRACE, 'Platform deadline leaves no safe execution window')
    supervisor = Supervisor(started + remaining)
    previous = {}
    directory = owned('delivery/four_arm_supervisor_' + args.attempt)
    need(not directory.exists(), 'Supervisor attempt already exists; no overwrite or automatic resume')
    need(directory.parent.is_dir(), 'Fixed Goal delivery directory missing')
    directory.mkdir()
    record = dict(schema='stage09_four_arm_job_v1', status='failed_or_incomplete', started_at_utc=utc(),
        layout=args.layout, gpu_slots=LAYOUTS[args.layout], allocated_gpus=1 + max(LAYOUTS[args.layout].values()),
        wrapper_max_seconds=args.max_seconds, platform_deadline_unix=args.platform_deadline_unix,
        effective_hard_deadline_unix=started_wall + remaining, cleanup_reserve_seconds=CLEANUP_RESERVE,
        normal_exit_grace_seconds=EXIT_GRACE, gates={}, children=[], no_ddp=True,
        no_automatic_resume=True, script_sha256=sha(Path(__file__)))
    accepted = {'J': args.accepted_warmup_j_sha256, 'H': args.accepted_warmup_h_sha256}
    record['accepted_warmup_sha256'] = accepted
    with (directory / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        def requested(number, _frame):
            supervisor.stop_reason = 'received_signal_' + str(number)
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, requested)
        need(sha(owned('delivery/formal_branch_acceptance.py')) == ACCEPTANCE_SHA, 'Acceptance helper changed')
        need(sha(owned('delivery/FORMAL_BRANCH_BINDING.json')) == BINDING_SHA, 'Formal binding changed')
        env, record['removed_distributed_environment_keys'] = single_environment()
        # All four CPU readiness gates complete before any training process.
        for arm in ARMS:
            record['gates']['ready_' + arm] = gate(supervisor, directory, env, 'ready', arm[0], accepted, arm=arm)
        inventory = supervisor.launch('gpu_inventory', [PYTHON, '-u', '-B', '-c', INVENTORY], env,
            ROOT / 'code', directory / 'gpu_inventory.json', timeout=120)
        supervisor.wait(inventory)
        devices = json.loads(inventory.stdout.read_text(encoding='utf-8'))
        expected_count = record['allocated_gpus']
        ids = validate_inventory(devices, expected_count)
        record['gpu_inventory'] = devices
        for arm in ARMS:
            child_env = dict(env, CUDA_VISIBLE_DEVICES=ids[LAYOUTS[args.layout][arm]])
            output = owned('runs/' + arm)
            need(not output.exists(), f'{arm}: output appeared after ready gate')
            child = supervisor.launch(arm, training_argv(arm), child_env, ROOT / 'code',
                directory / ('train_' + arm + '.log'),
                marker=lambda path=output: (path / 'complete.json').is_file(),
                finished_check=lambda selected=arm: completion(selected))
            record['children'].append(dict(arm=arm, pid=child.process.pid,
                gpu_uuid=child_env['CUDA_VISIBLE_DEVICES'], argv=training_argv(arm), output=str(output)))
            print(json.dumps(dict(event='training_child_started', **record['children'][-1])), flush=True)
        supervisor.wait_until(lambda: all(owned('runs/' + arm + '/initial.pt').is_file() for arm in ARMS))
        # Frozen run starts T immediately after atomic initial save. These CPU
        # checks run alongside training; they are not a pre-update barrier.
        for architecture in ('J', 'H'):
            record['gates']['pair_' + architecture] = gate(
                supervisor, directory, env, 'pair', architecture, accepted)
        supervisor.wait_until(lambda: all(child.done for child in supervisor.children))
        record['status'] = 'training_processes_completed'
    except BaseException as error:
        record['error'] = dict(type=type(error).__name__, message=str(error))
        supervisor.stop_reason = supervisor.stop_reason or str(error)
    finally:
        survivors = supervisor.cleanup()
        if survivors or supervisor.signal_history:
            record['status'] = 'failed_or_incomplete'
        elapsed = time.monotonic() - started
        record.update(ended_at_utc=utc(), elapsed_seconds=elapsed,
            allocated_gpu_seconds_during_wrapper=elapsed * record['allocated_gpus'],
            allocation_scope='Entire Job wall time including CPU gates; excludes allocation before wrapper start.',
            stop_reason=supervisor.stop_reason, cleanup_signals=supervisor.signal_history,
            owned_groups_alive_after_cleanup=survivors,
            process_exits={child.label: child.process.returncode for child in supervisor.children},
            final_model_acceptance_pending=True)
        with (directory / 'supervisor.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        for number, handler in previous.items():
            signal.signal(number, handler)
        print(json.dumps(dict(status=record['status'], receipt=str(directory / 'supervisor.json'))), flush=True)
    return 0 if record['status'] == 'training_processes_completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
