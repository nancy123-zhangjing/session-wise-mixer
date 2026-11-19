"""Bound one Stage07 GPU job and persist its process/log receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def single_process_environment(source):
    environment = dict(source)
    size_keys = ('WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'PMI_SIZE',
                 'OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_LOCAL_SIZE',
                 'MV2_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE')
    rank_keys = ('RANK', 'LOCAL_RANK', 'NODE_RANK', 'GROUP_RANK', 'ROLE_RANK',
                 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK',
                 'MV2_COMM_WORLD_RANK', 'MV2_COMM_WORLD_LOCAL_RANK', 'MPI_LOCALRANKID')
    original = {key: environment[key] for key in size_keys + rank_keys if key in environment}
    for key in size_keys:
        if int(environment.get(key, '1')) != 1:
            raise RuntimeError(f'This job requires a single process: {key}')
    for key in rank_keys:
        if int(environment.get(key, '0')) != 0:
            raise RuntimeError(f'This job requires sole rank zero: {key}')
    for key in ('ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_USE_FSDP', 'ACCELERATE_USE_MEGATRON_LM'):
        if environment.get(key, '').lower() in ('1', 'true', 'yes'):
            raise RuntimeError(f'This job does not authorize distributed plugins: {key}')
    for key in size_keys + rank_keys:
        environment.pop(key, None)
    return environment, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--training-reserve-seconds', type=float, default=1200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 8 * 3600:
        parser.error('The complete single-GPU allocation limit is eight hours')
    if args.attempt < 1 or (args.resume and args.attempt == 1):
        parser.error('Recovery requires an explicit new attempt number after confirming the previous job is terminal')
    goal = args.goal_root.resolve()
    delivery = goal / 'delivery'
    delivery.mkdir(parents=True, exist_ok=True)
    receipt_path = delivery / f'{args.mode}_attempt{args.attempt}_process_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Inspect the existing job receipt before authorizing a separate recovery attempt')
    started = time.time()
    # Leave startup/teardown margin inside the separately enforced platform cap.
    deadline = started + args.max_seconds - 90
    output = goal / 'runs' / 'stage07_item_local_seed2024_20261020_0045'
    command = [sys.executable, '-u', '-B', str(goal / 'code/scripts/train/run_item_local_attention.py'),
               '--mode', args.mode, '--project-root',
               '/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized',
               '--engine-root', str(goal / 'engine'), '--output', str(output),
               '--deadline-unix', str(deadline), '--training-deadline-reserve-seconds',
               str(args.training_reserve_seconds)]
    if args.resume:
        command.append('--resume')
    log_path = delivery / f'{args.mode}_attempt{args.attempt}_console.log'
    record = {'mode': args.mode, 'argv': command, 'pid': os.getpid(),
              'started_unix': started, 'deadline_unix': deadline,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'allocated_gpu_count': 1, 'max_process_seconds': args.max_seconds,
              'status': 'starting', 'console': str(log_path),
              'time_scope': 'process wall time; actual GPU allocation must be reconciled with platform lifecycle'}
    write_json(receipt_path, record)
    # Inspire injects WORLD_SIZE=1/LOCAL_RANK=0 even for this ordinary
    # single-process job. The historical init_device checks mere presence;
    # expose the intended single-process environment to Pipeline/Accelerate.
    environment, launcher_env = single_process_environment(os.environ)
    record['platform_launcher_env'] = launcher_env
    record['child_launch_mode'] = 'single_process_no_distributed_rank_environment'
    write_json(receipt_path, record)
    environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    with log_path.open('a', encoding='utf-8') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)

        def terminate(signum=None, frame=None):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        record.update(status='running', child_pid=child.pid)
        write_json(receipt_path, record)
        try:
            code = child.wait(timeout=max(1, deadline - time.time() + 30))
        except subprocess.TimeoutExpired:
            terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    record.update(status='succeeded' if code == 0 else 'failed_or_incomplete',
                  returncode=code, ended_unix=time.time(),
                  ended_utc=datetime.now(timezone.utc).isoformat(),
                  process_wall_seconds=time.time() - started)
    write_json(receipt_path, record)
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
