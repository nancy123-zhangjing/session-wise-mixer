"""CPU-only small fixtures; no real checkpoint, dataset, platform or GPU reads."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import types
import unittest
from unittest.mock import patch
import uuid

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sidecars', HERE / 'backup_final_sidecars.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TOKEN = 'sidecar_cpu_fixture_' + uuid.uuid4().hex[:12]
FIXTURE = HERE / TOKEN
FIXTURE.mkdir(exist_ok=False)


def native(name=m.TRAIN_JOB, status='SUCCEEDED'):
    return {'success': True, 'data': {'items': [{'name': name, 'status': status}]}}, {
        'success': True, 'data': {'items': [{'reason': 'SuccessfulDeletePod', 'time': 'fixture-time'}]}}, {
        'job_name': name, 'all_allocated_pods_released': True,
        'native_events_query_was_scoped_to_this_job': True, 'checked_at_utc': 'fixture-time', 'auditor': 'cpu-test'}


class Checks(unittest.TestCase):
    def test_allowlist(self):
        self.assertTrue(m.allowed('logs/four_arm_training_20261027_r1.log', 'operations'))
        self.assertTrue(m.allowed('evaluations/J0_test_beam32/per_sample.jsonl.gz', 'evaluations'))
        for name in ('delivery/stage09_bge_model.tar', 'inputs/dataset/train.parquet', 'runs/J0/best.pt', 'delivery/other.py'):
            self.assertFalse(m.allowed(name, 'operations'))
        for name in ('../file', '/etc/passwd', 'delivery\\file', 'delivery/a/../b'):
            with self.assertRaises(RuntimeError): m.allowed(name, 'operations')

    def test_native_status_and_release(self):
        status, events, audit = native()
        m.check_native_job(status, events, m.TRAIN_JOB, audit)
        single = {'success': True, 'data': status['data']['items'][0]}
        m.check_native_job(single, events, m.TRAIN_JOB, audit)
        for bad in ('RUNNING', 'FAILED', 'PENDING'):
            with self.assertRaises(RuntimeError): m.check_native_job(native(status=bad)[0], events, m.TRAIN_JOB, audit)
        with self.assertRaises(RuntimeError): m.check_native_job(status, {'success': True, 'data': {'items': []}}, m.TRAIN_JOB, audit)
        with self.assertRaises(RuntimeError): m.check_native_job(status, events, 'other-job', audit)
        with self.assertRaises(RuntimeError): m.check_native_job(status, events, m.TRAIN_JOB, dict(audit, all_allocated_pods_released=False))

    def test_four_completions_and_supervisor(self):
        complete = {'r_updates': 24255, 't_updates': 400}
        good = {'schema': 'stage09_four_arm_job_v1', 'status': 'training_processes_completed',
                'process_exits': dict.fromkeys(m.ARMS, 0), 'cleanup_signals': [], 'owned_groups_alive_after_cleanup': []}
        with patch.object(m.b, 'completion_guard', return_value=complete), patch.object(m, 'read_json', return_value=good):
            self.assertEqual(set(m.training_guard()), set(m.ARMS))
        with patch.object(m.b, 'completion_guard', return_value=dict(complete, t_updates=399)):
            with self.assertRaises(RuntimeError): m.training_guard()
        with patch.object(m.b, 'completion_guard', side_effect=RuntimeError('live .running')):
            with self.assertRaises(RuntimeError): m.training_guard()
        with patch.object(m.b, 'completion_guard', return_value=complete), patch.object(m, 'read_json', return_value=dict(good, cleanup_signals=['TERM'])):
            with self.assertRaises(RuntimeError): m.training_guard()

    def test_tar_rejects_links_duplicates_and_traversal(self):
        expected = {'logs/file.log': {'bytes': 1, 'sha256': '0' * 64}}
        for name, kind in (('../x', tarfile.REGTYPE), ('logs/file.log', tarfile.SYMTYPE), ('logs/file.log', tarfile.LNKTYPE)):
            member = tarfile.TarInfo(name); member.type = kind; member.size = 1
            with self.assertRaises(RuntimeError): m.b.validate_member(member, expected, set())
        member = tarfile.TarInfo('logs/file.log'); member.size = 1
        with self.assertRaises(RuntimeError): m.b.validate_member(member, expected, {'logs/file.log'})

    def test_evaluation_gate_all_twelve_and_partial_rejection(self):
        freeze_name = m.EVIDENCE + 'FOUR_BEST.json'
        freeze = {'schema': 'stage09_four_arm_final_selection_v1', 'status': 'passed', 'root': m.ROOT,
            'test_used_for_selection': False, 'four_best_sha256': dict.fromkeys(m.ARMS, '0' * 64),
            'four_best_bytes': dict.fromkeys(m.ARMS, 1)}
        observed = []
        current = {'name': '', 'partial': False}
        empty = FIXTURE / 'empty'; empty.write_bytes(b'')
        def fake_json(name, expected_sha=None):
            if name == freeze_name: return freeze
            if name == m.SUBSET: return {'rows': [{'sample_id': i} for i in range(1024)]}
            arm, variant = name.split('/')[1].split('_', 1)
            full = variant == 'test_beam32'
            observed.append(name)
            return {'split': 'test' if full else 'validation', 'examples': 53194 if full else 1024,
                    'full_split': full, 'beam': 256 if variant.endswith('256') else 32,
                    'architecture': arm[0], 'seed': 2024, 'inference_precision': 'fp32',
                    'diagnostic_prefix_limit': None, 'subset': None if full else {'manifest_sha256': m.SUBSET_SHA}}
        def fake_open(name):
            current['name'] = name
            return os.open(empty, os.O_RDONLY)
        class Rows:
            def __init__(self, **kwargs):
                self.count = 0
                self.total = 1 if current['partial'] else (53194 if '_test_' in current['name'] else 1024)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def readline(self, limit):
                if self.count == self.total: return b''
                self.count += 1
                return json.dumps({'sample_id': self.count - 1}).encode() + b'\n'
        with patch.object(m, 'read_json', side_effect=fake_json), \
             patch.object(m.b, 'hash_remote', return_value=({'sha256': '0' * 64, 'bytes': 1}, ())), \
             patch.object(m.b, 'open_remote', side_effect=fake_open), patch.object(m.gzip, 'GzipFile', Rows):
            self.assertEqual(m.evaluate_guard({'four_best_freeze': {'path': freeze_name, 'sha256': '0' * 64}}), freeze_name)
            self.assertEqual(len(observed), 12)
            current['partial'] = True
            with self.assertRaises(RuntimeError): m.evaluate_guard({'four_best_freeze': {'path': freeze_name, 'sha256': '0' * 64}})

    def test_real_small_tar_and_fresh_local_restore(self):
        # Simulate only remote filesystem I/O. Security of Linux no-follow opens
        # comes from unchanged base helper; this Windows test is not a Linux gate.
        remote = FIXTURE / 'remote'; remote.mkdir()
        names = sorted(set(m.OPERATIONS) | {m.SUP + n for n in m.SUP_REQUIRED}
                       | {m.index_name('operations')} | {'runs/' + a + '/complete.json' for a in m.ARMS})
        for name in names:
            path = remote.joinpath(*m.b.parts(name)); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture:' + name + '\n').encode())
        def fake_open(name, *, create=False, directory=False):
            path = remote.joinpath(*m.b.parts(name))
            return os.open(path, (os.O_WRONLY | os.O_CREAT | os.O_EXCL) if create else os.O_RDONLY,
                           0o600)
        fake_os = types.SimpleNamespace(name='posix', O_NOFOLLOW=1, fdopen=os.fdopen,
            fstat=os.fstat, fsync=os.fsync)
        def fake_json(name):
            return json.loads(remote.joinpath(*m.b.parts(name)).read_text(encoding='utf-8'))
        with patch.object(m, 'os', fake_os), patch.object(m.b, 'open_remote', side_effect=fake_open), \
             patch.object(m.b, 'remote_exists', side_effect=lambda n: remote.joinpath(*m.b.parts(n)).exists()), \
             patch.object(m, 'inventory', return_value=(names, {})), \
             patch.object(m, 'training_guard', return_value={}), patch.object(m, 'terminal_guard', return_value=({}, [])), \
             patch.object(m, 'read_json', side_effect=fake_json), contextlib.redirect_stdout(io.StringIO()):
            m.pack('operations', '0' * 64)
            with self.assertRaises(RuntimeError): m.pack('operations', '0' * 64)
        archive_name, manifest_name = m.output_names('operations')
        archive, manifest = remote / archive_name, remote / manifest_name
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result = json.loads(manifest.read_text(encoding='utf-8'))
        self.assertTrue(result['before_during_after_source_sha_equal'])
        self.assertEqual(result['file_count'], len(names))
        with self.assertRaises(RuntimeError): m.restore('operations', archive, manifest, '1' * 64, TOKEN)
        with contextlib.redirect_stdout(io.StringIO()): m.restore('operations', archive, manifest, sha, TOKEN)
        with self.assertRaises(RuntimeError): m.restore('operations', archive, manifest, sha, TOKEN)
        target = HERE.parents[2] / 'outputs/stage09_restore' / TOKEN / 'sidecars/operations'
        self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()), names)
        damaged = dict(result, files=result['files'] + [result['files'][0]])
        with self.assertRaises(RuntimeError): m.validate_manifest(damaged, 'operations')
        archive.write_bytes(archive.read_bytes()[:-1] + b'x')
        with self.assertRaises(RuntimeError): m.restore('operations', archive, manifest, sha, TOKEN + '_corrupt')
        # Real source-byte mutation after tar close must prevent publication of
        # any PASS manifest, even if the inventory names stay unchanged.
        remote = FIXTURE / 'moving_source'; remote.mkdir()
        for name in names:
            path = remote.joinpath(*m.b.parts(name)); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original fixture\n')
        calls = [0]
        def moving_inventory(*args):
            calls[0] += 1
            if calls[0] == 2:
                remote.joinpath(*m.b.parts(names[0])).write_bytes(b'changed fixture\n')
            return names, {}
        with patch.object(m, 'os', fake_os), patch.object(m.b, 'open_remote', side_effect=fake_open), \
             patch.object(m.b, 'remote_exists', side_effect=lambda n: remote.joinpath(*m.b.parts(n)).exists()), \
             patch.object(m, 'inventory', side_effect=moving_inventory), \
             patch.object(m, 'training_guard', return_value={}), patch.object(m, 'terminal_guard', return_value=({}, [])), \
             patch.object(m, 'read_json', side_effect=fake_json), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): m.pack('operations', '0' * 64)
        self.assertTrue((remote / archive_name).exists())
        self.assertFalse((remote / manifest_name).exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), fixture=str(FIXTURE),
        helper_sha256=hashlib.sha256((HERE / 'backup_final_sidecars.py').read_bytes()).hexdigest(),
        base_sha256=m.BASE_SHA, no_platform_model_dataset_or_gpu=True,
        limitation='Small mocked pack I/O and real local tar/restore only; Linux fd no-follow and actual final terminal/evaluation inputs not exercised.')
    with (FIXTURE / 'CPU_CHECKS.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
