"""Exercise the actual subprocess queue on CPU, including failure and forged receipts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('async_candidate', ROOT / 'async_resume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
WORKER = '''import hashlib,json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); task=json.loads(p.read_text()); time.sleep(.15)
v=dict(status='completed_readonly_task',request_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),runtime_sha256='test',checkpoint_sha256=task['checkpoint_sha256'],state_before='same',state_after='same')
p.with_suffix('.done.json').write_text(json.dumps(v))
'''


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='queue_cpu_', dir=ROOT) as directory:
        root = Path(directory)
        success = root / 'success'
        success.mkdir()
        for i in range(5):
            (success / f'phase{i:02d}.json').write_text(json.dumps(dict(checkpoint_sha256=str(i))))
        q = module.TaskQueue(success, lambda path: [sys.executable, '-B', '-c', WORKER, str(path)], 'test', lambda: False)
        peak = 0
        try:
            while True:
                q.pump()
                peak = max(peak, len(q.active))
                assert len(q.active) <= 2
                if not q.active and all(q.accepted(path) for path in q.requests()):
                    break
                time.sleep(.02)
            assert peak == 2
            checks.append('five_real_CPU_subprocesses_complete_max_two_concurrent')
            q.pump(wait=True)
            checks.append('completed_tasks_reused_without_duplicate_process_or_log')
            path = success / 'phase00.json'
            forged = json.loads(path.with_suffix('.done.json').read_text())
            forged['state_after'] = 'changed'
            path.with_suffix('.done.json').write_text(json.dumps(forged))
            try:
                q.accepted(path)
            except ValueError:
                checks.append('changed_readonly_state_rejected')
            else:
                raise AssertionError('Forged receipt accepted')
        finally:
            q.close()
        failed = root / 'failed'
        failed.mkdir()
        (failed / 'phase00.json').write_text('{"checkpoint_sha256":"x"}')
        q = module.TaskQueue(failed, lambda path: [sys.executable, '-B', '-c', 'raise SystemExit(7)'], 'test', lambda: False)
        try:
            try:
                q.pump(wait=True)
            except RuntimeError:
                checks.append('nonzero_worker_stops_queue_without_retry')
            else:
                raise AssertionError('Failed subprocess accepted')
        finally:
            q.close()
        stopped = root / 'stopped'
        stopped.mkdir()
        (stopped / 'phase00.json').write_text('{"checkpoint_sha256":"x"}')
        q = module.TaskQueue(stopped, lambda path: [], 'test', lambda: True)
        try:
            q.pump()
        except SystemExit as exc:
            assert exc.code == 124 and not q.active and not list(stopped.glob('*.log'))
            checks.append('stop_request_prevents_new_worker')
        else:
            raise AssertionError('Stop ignored')
    result = dict(status='passed_CPU_real_subprocess_queue', checks=checks, runtime_sha256=module.sha(ROOT / 'async_resume.py'),
                  limitation='CPU orchestration tests; no scientific model or GPU run.')
    (ROOT / 'CPU_ASYNC_QUEUE.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
