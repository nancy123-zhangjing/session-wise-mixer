"""Pure-stdlib supervisor tests with synthetic processes; never query a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('freeze35_job', Path(__file__).with_name('job.py'))
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)
TEST_TMP = Path(__file__).resolve().parents[3] / 'tmp'


class Clock:
    value = 0.0

    def monotonic(self):
        return self.value


class Signals:
    SIGTERM, SIGINT, SIGKILL = 15, 2, 9

    def __init__(self):
        self.handlers = {}

    def signal(self, number, handler):
        old = self.handlers.get(number)
        self.handlers[number] = handler
        return old


class Process:
    def __init__(self, factory, label, command):
        self.factory, self.label, self.command = factory, label, command
        self.pid, self.returncode = len(factory.calls), None
        self.injected = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.returncode is not None:
            return self.returncode
        if self.factory.mode == 'verify_failure' and self.label == 'verify':
            self.returncode = 7
            return 7
        if self.factory.mode in ('hang_verify', 'external_signal') and self.label == 'verify':
            self.factory.clock.value += timeout
            if self.factory.mode == 'external_signal' and not self.injected:
                self.injected = True
                self.factory.signals.handlers[15](15, None)
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = 0
        return 0


class Factory:
    def __init__(self, clock, signals, mode):
        self.clock, self.signals, self.mode = clock, signals, mode
        self.calls, self.children, self.sent = [], {}, []

    def popen(self, command, **kwargs):
        label = Path(kwargs['stdout'].name).name.split('.')[0]
        self.calls.append((label, command, kwargs['env']))
        child = Process(self, label, command)
        self.children[child.pid] = child
        if label == 'inventory':
            kwargs['stdout'].write(json.dumps([dict(name='NVIDIA H100', memory=80_000_000_000, uuid='GPU-synthetic')]).encode())
            kwargs['stdout'].flush()
        return child

    def killpg(self, pid, number):
        self.sent.append((self.clock.value, number))
        # Model an unresponsive child until escalation, without real processes.
        if number == 9:
            self.children[pid].returncode = -9


class JobTests(unittest.TestCase):
    def exercise(self, root, *, mode='ok', seconds=60, bad_hash=False):
        root = Path(root).resolve()
        assert root.is_relative_to(TEST_TMP.resolve()) and root != TEST_TMP.resolve()
        (root / 'code').mkdir()
        (root / 'delivery').mkdir()
        preparer = root / 'delivery/prepare.py'
        preparer.write_text('# synthetic verifier; never executed\n', encoding='utf-8')
        inputs = dict(parents={'B': dict(name='H0-Freeze32-R35', remote='/old/H0/best.pt')},
                      dataset={'remote': '/old/dataset'}, assets={'remote': '/old/assets'})
        (root / 'code/FREEZE35_PARENTS.json').write_text(json.dumps(inputs), encoding='utf-8')
        clock, signals = Clock(), Signals()
        factory = Factory(clock, signals, mode)
        fake_os = SimpleNamespace(name='posix', environ=dict(os.environ, CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='8'), killpg=factory.killpg)
        args = ['--root', str(root), '--python', 'synthetic-python', '--branch', 'B',
                '--manifest-sha', '0' * 64, '--preparer-sha', JOB.sha256(preparer),
                '--launcher-sha', '0' * 64 if bad_hash else JOB.sha256(JOB.__file__),
                '--authorized-seconds', str(seconds)]
        failure = None
        with patch.object(JOB, 'os', fake_os), patch.object(JOB, 'time', clock), patch.object(JOB, 'signal', signals), patch.object(JOB.subprocess, 'Popen', factory.popen):
            try:
                JOB.main(args)
            except (ValueError, RuntimeError, TimeoutError) as error:
                failure = error
        result = json.loads((root / 'jobs/B/result.json').read_text(encoding='utf-8'))
        return factory, result, failure

    def test_train_test_are_once_on_same_visible_card(self):
        with tempfile.TemporaryDirectory(prefix='freeze35_job_', dir=TEST_TMP) as root:
            factory, result, error = self.exercise(root)
            self.assertIsNone(error)
            self.assertEqual([row[0] for row in factory.calls], ['verify', 'inventory', 'train', 'test'])
            self.assertEqual(factory.calls[-2][2], factory.calls[-1][2])
            self.assertEqual(factory.calls[-1][2]['CUDA_VISIBLE_DEVICES'], '0')
            self.assertNotIn('WORLD_SIZE', factory.calls[-1][2])
            self.assertIn('cuda:0', factory.calls[-1][1])
            self.assertFalse(result['automatic_retry'])
            self.assertEqual(result['status'], 'predictions_complete_independent_audit_pending')

    def test_source_failure_has_receipt_without_starting_child(self):
        with tempfile.TemporaryDirectory(prefix='freeze35_job_', dir=TEST_TMP) as root:
            factory, result, error = self.exercise(root, bad_hash=True)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(factory.calls, [])
            self.assertEqual(result['status'], 'failed_preserve_state_no_retry')

    def test_verifier_failure_is_recorded_and_never_retried(self):
        with tempfile.TemporaryDirectory(prefix='freeze35_job_', dir=TEST_TMP) as root:
            factory, result, error = self.exercise(root, mode='verify_failure')
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual([row[0] for row in factory.calls], ['verify'])
            self.assertIn('verify failed', result['error'])

    def test_verifier_uses_total_deadline_and_reserved_grace(self):
        with tempfile.TemporaryDirectory(prefix='freeze35_job_', dir=TEST_TMP) as root:
            factory, result, error = self.exercise(root, mode='hang_verify', seconds=30)
            self.assertIsInstance(error, TimeoutError)
            self.assertEqual(factory.sent, [(15.0, 15), (30.0, 9)])
            self.assertEqual(result['seconds'], 30.0)
            self.assertEqual([row[0] for row in factory.calls], ['verify'])

    def test_external_signal_escalates_after_15_seconds_not_full_budget(self):
        with tempfile.TemporaryDirectory(prefix='freeze35_job_', dir=TEST_TMP) as root:
            factory, result, error = self.exercise(root, mode='external_signal', seconds=60)
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(factory.sent, [(.25, 15), (15.25, 9)])
            self.assertEqual(result['seconds'], 15.25)
            self.assertEqual(result['signal'], 15)
            self.assertEqual([row[0] for row in factory.calls], ['verify'])


if __name__ == '__main__':
    unittest.main()
