"""One bounded, evaluation-only process; never submits or retries a platform Job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--max-seconds', type=int, default=3400)
    args = parser.parse_args()
    if not 60 < args.max_seconds <= 3400:
        parser.error('Evaluation process is capped at 3400 seconds within the one-hour platform cap')
    goal, output, script = args.goal_root.resolve(), args.output.resolve(), args.script.resolve()
    if output == goal or goal in output.parents or script == goal or goal in script.parents:
        raise ValueError('New outputs and evaluator must be outside the old immutable campaign')
    if sha(script) != args.script_sha256:
        raise ValueError('Evaluator SHA256 mismatch')
    sys.path.insert(0, str(goal / 'code/scripts/train'))
    from item_local_gpu_job import single_process_environment
    environment, removed = single_process_environment(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       WANDB_MODE='offline', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'GPU_PROCESS_RECEIPT.json'
    with receipt_path.open('x', encoding='utf-8') as handle:
        handle.write('{}\n')
    started = time.time()
    command = [sys.executable, '-u', '-B', str(script), '--goal-root', str(goal),
               '--output', str(output), '--mode', 'evaluate',
               '--deadline-unix', str(started + args.max_seconds - 30)]
    receipt = dict(status='running', argv=command, started_unix=started,
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   max_process_seconds=args.max_seconds, platform_cap_seconds=3600,
                   normalized_single_process_variables=removed, evaluator_sha256=sha(script),
                   wrapper_sha256=sha(__file__), training_authorized=False,
                   test_evaluation_authorized=False, pid=os.getpid())
    write(receipt_path, receipt)
    print(json.dumps({'status':'starting_validation_grid', 'output':str(output)}), flush=True)
    process = None
    try:
        with (output / 'GPU_CONSOLE.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(output), env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            receipt['child_pid'] = process.pid
            write(receipt_path, receipt)
            try:
                returncode = process.wait(timeout=args.max_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                receipt['timed_out'] = True
                raise RuntimeError('Bounded evaluation process reached its time cap')
        receipt['returncode'] = returncode
        if returncode != 0:
            raise RuntimeError('Evaluation exited with code ' + str(returncode))
        complete = output / 'COMPLETE.json'
        if not complete.is_file():
            raise RuntimeError('Missing full four-cell completion evidence')
        receipt.update(status='process_completed_pending_independent_result_review',
                       complete_sha256=sha(complete))
    except BaseException as error:
        if process is not None:
            # The controller may have exited while a dataset worker is alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        receipt.update(finished_unix=time.time(), elapsed_process_seconds=time.time()-started)
        write(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
