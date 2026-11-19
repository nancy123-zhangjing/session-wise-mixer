"""CPU-only deterministic process/clock simulation; no subprocess or GPU launch."""
import importlib.util
import json
from pathlib import Path
import shlex
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stage09_four_arm_test_subject', HERE / 'formal_four_arm_job.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class FakeWorld:
    def __init__(self):
        self.now = 0.0
        self.processes = {}
        self.next_plan = {}
        self.signals = []

    def sleep(self, seconds):
        self.now += seconds

    def popen(self, argv, **kwargs):
        if kwargs.get('start_new_session') is not True:
            raise AssertionError('Child must own a new session')
        world = self
        plan = dict(self.next_plan)
        class Process:
            pid = len(world.processes) + 5000
            returncode = None
            killed = False
            def poll(self):
                if self.returncode is None and world.now >= plan.get('exit_at', float('inf')):
                    self.returncode = plan.get('code', 0)
                return self.returncode
        process = Process()
        self.processes[process.pid] = (process, plan)
        return process

    def alive(self, pid):
        process, plan = self.processes[pid]
        return not process.killed and (process.poll() is None or self.now < plan.get('descendant_until', 0))

    def send(self, pid, number):
        self.signals.append((pid, number, self.now))
        process, plan = self.processes[pid]
        if number == m.KILL_SIGNAL or not plan.get('ignore_term', False):
            process.killed = True
            if process.returncode is None:
                process.returncode = -int(number)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu_process_fixture_', dir=HERE)
        self.path = Path(self.temp.name).resolve()
        self.assertTrue(self.path.is_relative_to(HERE))
        self.world = FakeWorld()
        self.supervisors = []

    def tearDown(self):
        for supervisor in self.supervisors:
            supervisor.cleanup()
        self.assertTrue(self.path.is_relative_to(HERE))
        self.temp.cleanup()

    def supervisor(self, deadline=200):
        result = m.Supervisor(deadline, clock=lambda: self.world.now, sleep=self.world.sleep,
            popen=self.world.popen, alive=self.world.alive, send=self.world.send)
        self.supervisors.append(result)
        return result

    def launch(self, supervisor, label, plan, **kwargs):
        self.world.next_plan = plan
        return supervisor.launch(label, ['CPU_SIMULATION_ONLY'], {}, self.path,
                                 self.path / (label + '.log'), **kwargs)

    def test_completed_markers_do_not_kill_slow_normal_exit(self):
        supervisor = self.supervisor()
        for index, arm in enumerate(m.ARMS):
            self.launch(supervisor, arm, {'exit_at': 2.4 + index}, marker=lambda: True)
        supervisor.wait_until(lambda: all(child.done for child in supervisor.children))
        self.assertGreater(self.world.now, 5)
        self.assertFalse(supervisor.cleanup())
        self.assertFalse(self.world.signals)
        self.assertTrue(all(child.process.returncode == 0 for child in supervisor.children))

    def test_child_failure_terminates_all_owned_groups_and_escalates(self):
        supervisor = self.supervisor()
        failed = self.launch(supervisor, 'J0', {'exit_at': .4, 'code': 7})
        others = [self.launch(supervisor, arm, {'ignore_term': arm == 'H1'})
                  for arm in ('J1', 'H0', 'H1')]
        with self.assertRaisesRegex(RuntimeError, 'child_failed:J0:exit=7'):
            supervisor.wait_until(lambda: False)
        self.assertFalse(supervisor.cleanup())
        self.assertEqual(failed.process.returncode, 7)
        term_pids = {pid for pid, sig, _ in self.world.signals if sig == signal.SIGTERM}
        self.assertEqual(term_pids, {child.process.pid for child in others})
        self.assertIn((others[-1].process.pid, m.KILL_SIGNAL), [(p, s) for p, s, _ in self.world.signals])

    def test_total_deadline_reserves_term_and_kill_time(self):
        supervisor = self.supervisor(deadline=20)
        self.launch(supervisor, 'J0', {'ignore_term': True})
        with self.assertRaisesRegex(RuntimeError, 'execution_deadline'):
            supervisor.wait_until(lambda: False)
        self.assertFalse(supervisor.cleanup())
        self.assertLessEqual(self.world.now, 20)
        self.assertEqual([sig for _, sig, _ in self.world.signals], [signal.SIGTERM, m.KILL_SIGNAL])
        self.assertLessEqual(self.world.signals[-1][2], 15.01)

    def test_complete_marker_allows_60_seconds_but_not_infinite_exit(self):
        supervisor = self.supervisor()
        self.launch(supervisor, 'J0', {'ignore_term': True}, marker=lambda: True)
        with self.assertRaisesRegex(RuntimeError, 'exit_grace_expired'):
            supervisor.wait_until(lambda: False)
        self.assertGreaterEqual(self.world.now, 60)
        self.assertLess(self.world.now, 60.3)
        self.assertFalse(supervisor.cleanup())

    def test_exited_leader_does_not_hide_lingering_owned_descendant(self):
        supervisor = self.supervisor()
        self.launch(supervisor, 'J0', {'exit_at': 2, 'descendant_until': 300, 'ignore_term': True},
                    marker=lambda: True)
        with self.assertRaisesRegex(RuntimeError, 'process_group_did_not_drain'):
            supervisor.wait_until(lambda: False)
        self.assertLess(self.world.now, 60.3)
        self.assertFalse(supervisor.cleanup())
        self.assertIn(m.KILL_SIGNAL, [sig for _, sig, _ in self.world.signals])

    def test_pair_failure_and_external_signal_clean_every_training_group(self):
        supervisor = self.supervisor()
        for arm in m.ARMS:
            self.launch(supervisor, arm, {})
        self.launch(supervisor, 'pair_J', {'exit_at': .2, 'code': 1})
        with self.assertRaisesRegex(RuntimeError, 'child_failed:pair_J'):
            supervisor.wait_until(lambda: False)
        supervisor.stop_reason = 'received_signal_15'
        self.assertFalse(supervisor.cleanup())
        self.assertEqual(len({pid for pid, sig, _ in self.world.signals if sig == signal.SIGTERM}), 4)

    def test_inventory_timeout_is_bounded_separately(self):
        supervisor = self.supervisor(deadline=500)
        self.launch(supervisor, 'gpu_inventory', {'ignore_term': True}, timeout=120)
        with self.assertRaisesRegex(RuntimeError, 'child_timeout:gpu_inventory'):
            supervisor.wait_until(lambda: False)
        self.assertLess(self.world.now, 120.3)
        self.assertFalse(supervisor.cleanup())

    def test_frozen_commands_layout_and_gpu_scope(self):
        templates = json.loads((HERE / 'BRANCH_JOB_TEMPLATES.json').read_text(encoding='utf-8'))
        for job in templates['jobs']:
            shell = shlex.split(job['command_template'])[2]
            command = next(line.strip() for line in shell.splitlines() if '-m scripts.listwise_diffgrm.run' in line)
            self.assertEqual(m.training_argv(job['arm']), shlex.split(command))
        self.assertEqual(m.LAYOUTS['two_gpu_mixed_pairs'], {'J0':0, 'H0':0, 'J1':1, 'H1':1})
        rows = [{'uuid':'GPU-1234-abcd','name':'NVIDIA H100'}, {'uuid':'GPU-abcd-5678','name':'NVIDIA H100'}]
        evidence = {'status':'passed','inventory':rows,'torch_visible':rows[::-1]}
        self.assertEqual(m.validate_inventory(evidence,2), ['GPU-abcd-5678','GPU-1234-abcd'])
        with self.assertRaises(RuntimeError): m.validate_inventory(evidence,1)
        with self.assertRaises(RuntimeError):
            m.validate_inventory({**evidence,'torch_visible':rows[:1]},2)
        with patch.dict(m.os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-1234-abcd','RANK':'0','WORLD_SIZE':'2','MASTER_PORT':'123'}, clear=True):
            env, removed = m.single_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-1234-abcd')
        self.assertEqual(set(removed), {'RANK','WORLD_SIZE','MASTER_PORT'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
