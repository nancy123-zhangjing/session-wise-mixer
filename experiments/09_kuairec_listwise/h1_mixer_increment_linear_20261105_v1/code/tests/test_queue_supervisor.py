"""Process lifecycle regressions for the external supervisor; no GPUs or training."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("queue_supervisor", ROOT / "scripts/train/continuous_queue_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
# Queue logic fixtures use a lock stand-in on Windows. Real Linux locking and
# process-group cleanup are exercised by the remote CPU run of this same suite.
if os.name != "posix":
    sys.modules.setdefault("fcntl", mock.Mock(LOCK_EX=2, LOCK_NB=4))


class SupervisorTests(unittest.TestCase):
    def stat(self, state="S", ticks="123"):
        return "42 (a name with ) inside) " + " ".join([state] + ["0"] * 18 + [ticks] + ["0"] * 6)

    def test_token_ignores_empty_or_changing_argv(self):
        def text(path, *args, **kwargs):
            return "boot-test" if str(path).endswith("boot_id") else self.stat()
        with mock.patch.object(Path, "read_text", text), mock.patch.object(supervisor.socket, "gethostname", return_value="host-test"):
            token = supervisor.process_token(42)
            self.assertEqual(token, {"pid": 42, "start_ticks": "123", "host": "host-test", "boot_id": "boot-test"})
            self.assertTrue(supervisor.same_process(token))
        self.assertNotIn("command", token)

    def test_zombie_or_exited_pid_is_not_live(self):
        with mock.patch.object(Path, "read_text", return_value=self.stat("Z")):
            self.assertIsNone(supervisor.process_token(42))
        with mock.patch.object(Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(supervisor.process_token(42))

    def test_changed_start_ticks_or_boot_refuses_identity(self):
        token = {"pid": 42, "start_ticks": "123", "host": "host-test", "boot_id": "boot-test"}
        for changed in ({**token, "start_ticks": "124"}, {**token, "boot_id": "other-boot"}):
            with mock.patch.object(supervisor, "process_token", return_value=changed):
                self.assertFalse(supervisor.same_process(token))

    def test_child_handle_drives_completion(self):
        child = mock.Mock()
        child.poll.return_value = None
        self.assertFalse(supervisor.child_finished(child))
        child.poll.return_value = 0
        self.assertTrue(supervisor.child_finished(child))
        child.poll.return_value = 1
        self.assertTrue(supervisor.child_finished(child))

    def test_exit_between_poll_and_proc_read_is_reaped(self):
        child = mock.Mock()
        child.poll.side_effect = [None,0]
        with mock.patch.object(supervisor, "same_process", return_value=False):
            self.assertFalse(supervisor.worker_running(child,{"pid":42}))
        child.poll.side_effect = [None,None]
        with mock.patch.object(supervisor, "same_process", return_value=False):
            with self.assertRaisesRegex(RuntimeError,"Stable worker identity lost"):
                supervisor.worker_running(child,{"pid":42})

    def test_cleanup_timeout_returned_without_masking_root_error(self):
        child = mock.Mock(pid=42)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("fixture", 0)
        record = {"process": child, "token": {"pid": 42}, "log": io.BytesIO()}
        with mock.patch.object(supervisor, "same_process", return_value=True), mock.patch.object(supervisor.os, "getpgid", return_value=42, create=True), mock.patch.object(supervisor.os, "killpg", create=True) as kill, mock.patch.object(supervisor.signal, "SIGKILL", 9, create=True):
            errors = supervisor.stop_children({0: record}, grace_seconds=0, kill_seconds=0)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "reap")
        self.assertEqual(kill.call_count, 2)
        self.assertTrue(record["log"].closed)

    def test_cleanup_never_signals_reused_pid(self):
        child = mock.Mock(pid=42)
        child.poll.return_value = None
        record = {"process": child, "token": {"pid": 42}, "log": io.BytesIO()}
        with mock.patch.object(supervisor, "same_process", return_value=False), mock.patch.object(supervisor.os, "killpg", create=True) as kill:
            self.assertEqual(supervisor.stop_children({0: record}, grace_seconds=0), [])
        kill.assert_not_called()

    def test_real_children_complete_and_are_reaped(self):
        records = {}
        try:
            for index in range(12):
                child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(.03)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                records[index] = child
            deadline = time.monotonic() + 10
            while any(not supervisor.child_finished(p) for p in records.values()) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(all(p.wait(timeout=1) == 0 for p in records.values()))
        finally:
            for child in records.values():
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)

    def test_linux_real_identity_and_group_cleanup(self):
        if os.name != "posix":
            # Local Windows performs the real child lifecycle above. The remote
            # CPU gate executes this additional Linux-only body, without skips.
            self.assertFalse(Path("/proc/sys/kernel/random/boot_id").exists())
            return
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        token = supervisor.process_token(child.pid)
        self.assertIsNotNone(token)
        record = {"process": child, "token": token, "log": io.BytesIO()}
        self.assertEqual(supervisor.stop_children({0: record}, grace_seconds=1), [])
        self.assertIsNotNone(child.returncode)

    def queue_fixture(self, temporary, count=89):
        import copy
        import hashlib
        import json
        from scripts.continuous.campaign import initialize_state
        base = Path(temporary)
        campaign, cache = base / "campaign", base / "cache"
        campaign.mkdir(); cache.mkdir()
        (cache / "cache.json").write_text("{}")
        state = initialize_state(supervisor.TRAINING_SOURCE)
        tasks = [copy.deepcopy(state["tasks"][0]) for _ in range(count)]
        for index, task in enumerate(tasks):
            task["task_id"] = f"cpu_logic_fixture_{index:03d}"
        state["tasks"] = tasks
        (campaign / "queue.json").write_text(json.dumps(state))
        gate = base / "gate.json"
        gate.write_text(json.dumps({"status": "PASS", "source_sha256": supervisor.TRAINING_SOURCE,
            "cache_manifest_sha256": hashlib.sha256((cache / "cache.json").read_bytes()).hexdigest()}))
        deployment = base / "deployment.json"
        deployment.write_text(json.dumps({"supervisor_sha256": hashlib.sha256(Path(supervisor.__file__).read_bytes()).hexdigest(),
                                         "training_source_sha256": supervisor.TRAINING_SOURCE}))
        return campaign, cache, gate, deployment

    def test_queue_refills_89_logic_tasks_without_changing_budget(self):
        import contextlib
        parent = ROOT / "tmp/queue_supervisor_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary, contextlib.ExitStack() as stack:
            campaign, cache, gate, deployment = self.queue_fixture(temporary)
            stack.enter_context(mock.patch.object(supervisor, "bind_training_root", return_value={}))
            stack.enter_context(mock.patch.object(supervisor, "process_token", side_effect=lambda pid: {"pid": pid}))
            stack.enter_context(mock.patch("scripts.continuous.gpu_smoke.gpu_inventory", return_value=[{"uuid":f"fixture-{i}"} for i in range(8)]))
            stack.enter_context(mock.patch("scripts.continuous.campaign.expand_dependencies"))
            stack.enter_context(mock.patch("scripts.continuous.campaign.freeze_validation_selection", return_value={"cpu_logic_fixture":True}))
            stack.enter_context(mock.patch("scripts.continuous.campaign.verify_result", return_value={"cpu_logic_fixture":True}))
            children = []
            def child(*args, **kwargs):
                process = mock.Mock(pid=len(children)+100, returncode=0)
                process.poll.return_value = 0
                children.append(process)
                return process
            stack.enter_context(mock.patch.object(supervisor.subprocess, "Popen", side_effect=child))
            result = supervisor.run_queue(ROOT, campaign, cache, gate, deployment, poll_seconds=0)
            self.assertEqual(result["status"], "TRAINING_COMPLETE")
            self.assertEqual(len(children), 89)
            self.assertFalse(result["workers"])
            self.assertTrue(all(t["status"] == "succeeded" for t in result["tasks"]))

    def test_queue_preserves_root_failure_and_interrupted_statuses(self):
        import contextlib
        import json
        parent = ROOT / "tmp/queue_supervisor_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary, contextlib.ExitStack() as stack:
            campaign, cache, gate, deployment = self.queue_fixture(temporary, count=8)
            stack.enter_context(mock.patch.object(supervisor, "bind_training_root", return_value={}))
            stack.enter_context(mock.patch.object(supervisor, "process_token", side_effect=lambda pid: {"pid": pid}))
            stack.enter_context(mock.patch("scripts.continuous.gpu_smoke.gpu_inventory", return_value=[{"uuid":f"fixture-{i}"} for i in range(8)]))
            stack.enter_context(mock.patch("scripts.continuous.campaign.expand_dependencies"))
            process = mock.Mock(pid=100, returncode=7)
            process.poll.return_value = 7
            stack.enter_context(mock.patch.object(supervisor.subprocess, "Popen", return_value=process))
            cleanup = stack.enter_context(mock.patch.object(supervisor, "stop_children", return_value=[{"cpu_cleanup_fixture":True}]))
            with self.assertRaisesRegex(RuntimeError, "Worker exited 7"):
                supervisor.run_queue(ROOT, campaign, cache, gate, deployment, poll_seconds=0)
            state = json.loads((campaign / "queue.json").read_text())
            self.assertEqual(state["status"], "STOPPED_FOR_AUDIT")
            self.assertIn("Worker exited 7", state["root_error"]["message"])
            self.assertEqual(sum(t["status"]=="interrupted" for t in state["tasks"]),7)
            self.assertEqual(sum(t["status"]=="failed" for t in state["tasks"]),1)
            for record in cleanup.call_args.args[0].values():
                record["log"].close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
