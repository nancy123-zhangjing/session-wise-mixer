"""<20-second Linux-only real-process CPU smoke test of the prepared supervisor.

Writes only a new fixed Goal delivery subdirectory. No torch, GPU inventory,
training command, model import, platform call or runs-directory operation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
DELIVERY = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/delivery')
HELPER_SHA = '52b9837a1fa970577d43fba53fe84e71062c472f6ad428c8d0c3a76359327dfe'
LONG_SLEEP = r'''
import json, os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-u', '-B', '-c', 'import time; time.sleep(120)'])
def stop(number, _):
    # Same group receives TERM together; reap the descendant before exiting.
    child.wait(timeout=1)
    raise SystemExit(128 + number)
signal.signal(signal.SIGTERM, stop)
print(json.dumps({'child_pid':child.pid,'parent_pid':os.getpid(),'pgid':os.getpgrp()}),flush=True)
time.sleep(120)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,39}', args.attempt):
        parser.error('Unsafe attempt identifier')
    if os.name != 'posix':
        raise RuntimeError('Prepare locally; execute only in the Linux CPU Notebook')
    started = time.monotonic()
    hard_deadline = started + 18.0  # Whole-smoke watchdog, then <=0.5s cleanup.
    helper = DELIVERY / 'formal_four_arm_job.py'
    for path in (DELIVERY, *DELIVERY.parents, helper):
        if path.is_symlink():
            raise RuntimeError('Symlink refused: ' + str(path))
    if hashlib.sha256(helper.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Supervisor helper SHA differs')
    spec = importlib.util.spec_from_file_location('stage09_supervisor_cpu_smoke', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = DELIVERY / ('four_arm_cpu_smoke_' + args.attempt)
    output.mkdir()  # Refuse overwrite; no parent/source/runs directory creation.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    supervisors = []
    record = dict(schema='stage09_four_arm_linux_cpu_smoke_v1', status='failed',
        helper_sha256=HELPER_SHA, real_popen=True, real_killpg=True, gpu_used=False,
        training_started=False, cases=[], watchdog_seconds=18, cleanup_after_watchdog_seconds=.5,
        maximum_intended_seconds=19, executable=sys.executable)

    def supervisor(deadline=hard_deadline + module.CLEANUP_RESERVE):
        result = module.Supervisor(min(deadline, hard_deadline + module.CLEANUP_RESERVE))
        supervisors.append(result)
        return result

    def launch(owner, name, code, **kwargs):
        child = owner.launch(name, [sys.executable, '-u', '-B', '-c', code], environment,
                             output, output / (name + '.log'), **kwargs)
        if os.getpgid(child.process.pid) != child.process.pid:
            raise RuntimeError('Child does not own its process group')
        return child

    def descendant_ready(child):
        try:
            return json.loads(child.stdout.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return None

    def assert_gone(children):
        for child in children:
            if module.group_alive(child.process.pid) or child.process.poll() is None:
                raise RuntimeError('Owned process group remains: ' + child.label)
            item = descendant_ready(child)
            if item:
                try:
                    os.kill(item['child_pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('CPU descendant was not reaped')

    previous = {}
    try:
        def stop(number, _):
            for owner in supervisors:
                owner.stop_reason = 'cpu_smoke_signal_' + str(number)
                if number == signal.SIGALRM:
                    owner.hard_deadline = min(owner.hard_deadline, time.monotonic() + .5)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            previous[number] = signal.signal(number, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.01, hard_deadline - time.monotonic()))
        # A completed marker is visible immediately; true exit occurs after >1s.
        owner = supervisor()
        child = launch(owner, 'normal', 'import time; time.sleep(1.4)', marker=lambda: True)
        owner.wait(child)
        if child.process.returncode != 0 or owner.cleanup() or owner.signal_history:
            raise RuntimeError('Normal CPU exit was killed or did not finish')
        record['cases'].append(dict(case='normal_real_exit_after_one_second', status='passed',
            returncode=child.process.returncode, elapsed_seconds=time.monotonic()-started))

        owner = supervisor()
        sleepers = [launch(owner, name, LONG_SLEEP) for name in ('sleeper_a', 'sleeper_b')]
        owner.wait_until(lambda: all(descendant_ready(child) for child in sleepers))
        failed = launch(owner, 'failing', 'import time; time.sleep(.1); raise SystemExit(7)')
        try:
            owner.wait_until(lambda: False)
        except RuntimeError as error:
            if 'child_failed:failing:exit=7' not in str(error):
                raise
        else:
            raise RuntimeError('Nonzero child was not detected')
        if owner.cleanup():
            raise RuntimeError('Failure cleanup left a process group')
        assert_gone(sleepers + [failed])
        record['cases'].append(dict(case='nonzero_exit_cleans_other_groups_and_descendants',
            status='passed', original_exitcode=failed.process.returncode, signals=owner.signal_history))

        # Supervisor's production 15s cleanup reserve is retained unchanged.
        # Its execution cutoff occurs in <=0.5s; TERM then reaping must finish.
        owner = supervisor(time.monotonic() + module.CLEANUP_RESERVE + .5)
        child = launch(owner, 'deadline_sleeper', LONG_SLEEP)
        owner.wait_until(lambda: descendant_ready(child) is not None)
        try:
            owner.wait_until(lambda: False)
        except RuntimeError as error:
            if str(error) != 'execution_deadline':
                raise
        else:
            raise RuntimeError('Execution deadline was not enforced')
        if owner.cleanup():
            raise RuntimeError('Deadline cleanup left a process group')
        assert_gone([child])
        record['cases'].append(dict(case='deadline_cleans_real_group_and_descendant',
            status='passed', signals=owner.signal_history))
        record['status'] = 'passed'
    except BaseException as error:
        record['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        record['survivors'] = {str(i): owner.cleanup() for i, owner in enumerate(supervisors)}
        if any(record['survivors'].values()):
            record['status'] = 'failed'
        record['elapsed_seconds'] = time.monotonic() - started
        if record['elapsed_seconds'] >= 20:
            record['status'] = 'failed'
            record['budget_exceeded'] = True
        record['process_exits'] = {child.label: child.process.returncode
            for owner in supervisors for child in owner.children}
        record['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with (output / 'receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in previous.items():
            signal.signal(number, handler)
        print(json.dumps(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds'],
            receipt=str(output / 'receipt.json'))), flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
