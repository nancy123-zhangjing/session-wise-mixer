"""New preflight contract only: real tiny CPU updates, never real parents/GPU."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import test_listwise_recovery35 as fixtures
from scripts.listwise_diffgrm import recovery35_preflight as m
from scripts.listwise_diffgrm.assets import load_assets
from scripts.listwise_diffgrm.data import ListwiseCollator
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all


class ToyRows:
    def __init__(self, count):
        self.rows = [dict(sample_id=f'toy-{i:03d}', user_id=i,
                         history_item_ids=[1, 2, 3], history_timestamp_us=[1, 2, 3],
                         target_item_ids=[4, 5, 6, 7, 8], target_mask=[True] * 5,
                         target_timestamp_us=[4, 5, 6, 7, 8]) for i in range(count)]
        self.table = pa.table({'sample_id': [r['sample_id'] for r in self.rows]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def collator(self):
        return ListwiseCollator(3, 5, 9)


class Recovery35PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only fixture construction, not the old tests/suite.
        fixtures.Recovery35Tests.setUpClass()
        cls.factory = fixtures.Recovery35Tests()

    @classmethod
    def tearDownClass(cls):
        fixtures.Recovery35Tests.tearDownClass()

    def directory(self):
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name).resolve()
        self.assertTrue(directory.is_relative_to(ROOT / 'tmp'))
        self.addCleanup(temporary.cleanup)
        return directory

    def test_real_abc_next_update_disk_restore_exact_and_published_bundle_rejected(self):
        for branch in 'ABC':
            with self.subTest(branch=branch):
                controller, batches = self.factory.controller(branch)
                directory = self.directory()
                if branch in 'AB':
                    with patch.object(controller.trainer, 'refresh_catalog', side_effect=AssertionError('frozen refresh')):
                        skipped = m.skip_frozen_boundary(controller)
                    self.assertTrue(skipped['exact_non_cursor_state_unchanged'])
                kind = 'T' if branch == 'C' else 'R'
                record, proof = m.disk_replay_next(controller, batches, kind, directory)
                self.assertTrue(proof['exact_loaded_state'] and proof['exact_next_update'])
                self.assertTrue(proof['reusable_snapshot_unchanged'])
                for field in ('seconds', 'state_materialization_seconds', 'payload_fingerprint_seconds', 'file_sha256_seconds'):
                    self.assertGreaterEqual(proof['save'][field], 0)
                self.assertGreaterEqual(proof['load']['payload_fingerprint_seconds'], 0)
                disk = torch.load(proof['save']['path'], weights_only=False)
                self.assertEqual(disk['format'], m.BUNDLE_FORMAT)
                self.assertTrue(disk['preflight_only'])
                for override in (True, False):
                    with self.assertRaisesRegex(ValueError, 'Preflight'):
                        controller.restore(disk, allow_preflight=override)
                self.assertTrue(record['frozen_module_and_optimizer_unchanged'])
                if kind == 'R':
                    for layer in range(4):
                        for part in ('self_attn', 'cross_attn', 'mlp'):
                            self.assertGreater(record['gradients'][f'decoder_{layer}_{part}']['gradient_l1'], 0)
                else:
                    self.assertEqual(record['changed_parameters'], ['mixer.base_codebooks'])

    def test_c_executes_all_original_40_t_then_r_without_schedule_shortening(self):
        directory = self.directory()
        trainer, batches = fixtures.trainer_at(directory, False)
        cfg = copy.deepcopy(trainer.config)
        cfg['training']['t_updates_per_cycle'] = 40
        trainer = AlternatingTrainer(trainer.model, trainer.alignment, cfg, 16, precision='bf16')
        parent, _, execution = fixtures.bindings(trainer)
        for phase, row in enumerate(trainer.plan[:5]):
            seed_all(cfg['seed'] + 100003 * (phase + 1))
            for _ in range(row['updates']):
                trainer.update(batches, 'R')
        source = trainer.checkpoint(parent, dict(phase=5, update=0), 'H0')
        controller = fixtures.m.Recovery35Controller(trainer, 'C', parent, execution,
                                                     fixtures.m.BRANCHES['C']['parent_sha256'])
        controller.fork(source)
        before_plan = copy.deepcopy(trainer.plan)
        with redirect_stdout(io.StringIO()):
            result = m.profile_training(controller, ToyRows(16), directory, 3, m.StageLog(directory))
        self.assertEqual(trainer.plan, before_plan)
        self.assertEqual(len(result['T']['records']), 40)
        self.assertEqual(result['T']['updates'], 40)
        self.assertEqual(result['short_endpoint'], dict(phase=6, update=4, r_updates=24, t_updates=40))
        self.assertTrue(result['T_disk_replay']['exact_next_update'])
        self.assertTrue(result['R_disk_replay']['exact_next_update'])
        logs = [json.loads(line) for line in (directory / 'training.preflight.jsonl').read_text().splitlines()]
        self.assertEqual(len(logs), 44)
        self.assertEqual([(r['phase_index'], r['update']) for r in logs[:40]], [(5, i) for i in range(1, 41)])
        self.assertEqual(logs[40]['phase_index'], 6)
        self.assertTrue(all(r['preflight_only'] and 'gradient_norm' in r for r in logs))

    def test_bundle_tampering_and_overwrite_fail_before_restore(self):
        controller, _ = self.factory.controller('C')
        directory = self.directory()
        path = directory / 'preflight.bundle.pt'
        original, _ = m.save_bundle(controller, path)
        with self.assertRaises(FileExistsError):
            m.save_bundle(controller, path)
        bundle = torch.load(path, weights_only=False)
        bundle['snapshot']['inner']['r_updates'] += 1
        bad = directory / 'tampered.preflight.bundle.pt'
        torch.save(bundle, bad)
        with self.assertRaisesRegex(m.GateFailure, 'payload digest'):
            m.load_bundle(controller, bad)
        m.exact_equal(original, controller.checkpoint())

    def test_real_fp32_batch32_decode_preserves_training_rng_gradients_and_modes(self):
        controller, batches = self.factory.controller('A')
        m.skip_frozen_boundary(controller)
        m.checked_step(controller, batches, 'R')
        before = controller.checkpoint()
        modes = m.diagnostic_signature(controller.trainer)
        assets = load_assets(self.factory.directory / 'assets')
        directory = self.directory()
        # Only release lookup is synthetic; real catalog, beam decoding and
        # diagnostic restoration execute, using 32 toy visible-history rows.
        with patch('scripts.listwise_diffgrm.data.KuaiRecListDataset', return_value=ToyRows(32)), \
             patch.object(m, 'sha256', return_value='f' * 64), redirect_stdout(io.StringIO()):
            result = m.profile_decode(controller, assets, 32, directory, m.StageLog(directory))
        self.assertEqual((result['batch_size'], result['beam'], result['inference_precision']), (32, 32, 'fp32'))
        self.assertTrue(result['exact_output_repeat'])
        self.assertFalse(result['quality_metric'])
        self.assertEqual(len(result['records']), 1)
        m.exact_equal(before, controller.checkpoint())
        m.exact_equal(modes, m.diagnostic_signature(controller.trainer))

    def test_supervisor_terminates_hanging_cpu_worker_before_deadline(self):
        directory = self.directory()
        started = time.monotonic()
        with patch.object(m, 'SHUTDOWN_RESERVE', .4):
            result = m.run_bounded_process([sys.executable, '-c', 'import time; time.sleep(20)'],
                directory=directory, environment=os.environ.copy(), deadline=started + .7)
        self.assertEqual(result['reason'], 'overall_preflight_deadline')
        self.assertNotEqual(result['returncode'], 0)
        self.assertLess(time.monotonic() - started, 5)

    def test_gpu_identity_rejects_mig_small_memory_and_unverified_uuid(self):
        props = SimpleNamespace(name='NVIDIA H100 80GB HBM3', total_memory=85_000_000_000, uuid='abc-def')
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'get_device_properties', return_value=props), \
             patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
             patch.object(m.subprocess, 'check_output', return_value='GPU-abc-def, NVIDIA H100, 81920, Disabled\n'):
            self.assertEqual(m.full_h100_identity()['mig_mode'], 'disabled')
            props.total_memory = 40_000_000_000
            with self.assertRaisesRegex(m.GateFailure, '75 GB'):
                m.full_h100_identity()
            props.total_memory = 85_000_000_000
            props.uuid = 'MIG-abc'
            with self.assertRaisesRegex(m.GateFailure, 'MIG'):
                m.full_h100_identity()
            props.uuid = 'unknown'
            with self.assertRaisesRegex(m.GateFailure, 'Cannot prove'):
                m.full_h100_identity()

    def test_invalid_budget_is_rejected_before_supervisor_or_gpu(self):
        base = ['--parent-a', 'a', '--parent-b', 'b', '--parent-c', 'c', '--ancestor', 'd', '--output', 'out']
        with patch.object(m, 'supervise') as supervisor, patch.object(m, 'worker') as worker:
            for value in ('1621', 'nan', 'inf', '-1'):
                with self.subTest(value=value), redirect_stdout(io.StringIO()), \
                     patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                    m.main(base + ['--max-seconds', value])
            supervisor.assert_not_called()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
