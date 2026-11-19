"""CPU helper/gate regression; these tests never count as an H100 acceptance."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.listwise_diffgrm import preflight as gate
from test_listwise_training import fixture

TEMP = ROOT / 'tmp/listwise_preflight_tests'
TEMP.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


class PreflightTests(unittest.TestCase):
    def test_budget_accounts_for_allocated_cards_and_reserves_shutdown(self):
        self.assertEqual(gate.budget_seconds(3600, 2, 1700), 1700)
        for args in [(7201, 1, 1), (100, 0, 1), (100, 1, 71), (3600, 2, 1800)]:
            with self.assertRaises(ValueError):
                gate.budget_seconds(*args)

    def test_deadlines_distinguish_stage_and_global_limits(self):
        stage = dict(active=True, started=12, timeout_seconds=5, name='H1/T')
        self.assertIsNone(gate.deadline_reason(10, 30, stage, now=16))
        self.assertEqual(gate.deadline_reason(10, 30, stage, now=17), 'stage_timeout:H1/T')
        stage['active'] = False
        self.assertIsNone(gate.deadline_reason(10, 30, stage, now=39))
        self.assertEqual(gate.deadline_reason(10, 30, stage, now=40), 'overall_budget_deadline')

    def test_supervisor_terminates_worker_at_budget_and_leaves_receipt(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            output = Path(directory) / 'bounded'
            args = SimpleNamespace(output=str(output), remaining_gpu_seconds=100, allocated_gpus=1,
                                   max_seconds=1, worker=False)
            class Process:
                returncode = None
                terminated = False
                def poll(self): return self.returncode
                def terminate(self): self.terminated = True; self.returncode = -15
                def kill(self): self.returncode = -9
                def wait(self, timeout): return self.returncode
            process = Process()
            with patch.object(gate.subprocess, 'Popen', return_value=process), \
                    patch.object(gate.time, 'monotonic', side_effect=[0, 2, 3]), redirect_stdout(io.StringIO()):
                code = gate.supervise(args)
            self.assertEqual(code, 2)
            self.assertTrue(process.terminated)
            result = json.loads((output / 'supervisor.json').read_text())
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['allocated_gpu_seconds_during_script'], 3)

    def test_stage_log_records_failure_and_disarms_stage(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            log = gate.StageLog(directory, timeout_seconds=5)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'broken'):
                with log.stage('decoder'):
                    raise ValueError('broken')
            active = json.loads((Path(directory) / 'active_stage.json').read_text())
            self.assertFalse(active['active'])
            events = [json.loads(line) for line in (Path(directory) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in events], ['stage_start', 'stage_failed'])

    def test_tree_comparison_and_fingerprint_detect_optimizer_tensor_change(self):
        state = {'model': torch.arange(4).float(), 'rng': np.arange(5),
                 'optimizer': {'step': torch.tensor(1.), 'moment': torch.zeros(2)}}
        other = copy.deepcopy(state)
        gate.exact_equal(state, other)
        self.assertEqual(gate.fingerprint(state), gate.fingerprint(other))
        other['optimizer']['moment'][0] = 1
        with self.assertRaises(gate.GateFailure):
            gate.exact_equal(state, other)
        self.assertNotEqual(gate.fingerprint(state), gate.fingerprint(other))

    def test_generation_input_adapter_cannot_read_targets(self):
        class HistoryOnly(dict):
            def __getitem__(self, key):
                if key.startswith('target'):
                    raise AssertionError('Target leakage')
                return super().__getitem__(key)
        batch = HistoryOnly(history_item_ids=torch.tensor([[0, 2]]), history_mask=torch.tensor([[False, True]]))
        codes = torch.tensor([[-1] * 4, [0] * 4, [2] * 4])
        trainer = SimpleNamespace(device=torch.device('cpu'), alignment=SimpleNamespace(hard_codes=codes))
        sid, mask = gate.generation_inputs(trainer, batch)
        self.assertTrue(torch.equal(sid, codes[torch.tensor([[0, 2]])]))
        self.assertEqual(mask.tolist(), [[False, True]])

    def test_real_cpu_four_arm_gate_checks_partition_replay_and_refresh(self):
        # Tiny actual engine/JointCrossPSE tensors test gate logic only. The
        # production worker independently rejects toy assets and CPU devices.
        for route in ('J', 'H'):
            signatures = []
            for enabled in (False, True):
                with self.subTest(route=route, mixer=enabled), tempfile.TemporaryDirectory(dir=TEMP) as directory:
                    with redirect_stdout(io.StringIO()):
                        trainer, batch = fixture(directory, route, enabled)
                        signatures.append(gate.initial_signature(trainer))
                        result = gate.arm_gate(trainer, batch, directory, {'fixture': route},
                                              route + str(int(enabled)), gate.StageLog(directory))
                        from scripts.listwise_diffgrm.assets import load_assets
                        from scripts.listwise_diffgrm.run import make_catalog
                        assets = load_assets(Path(directory) / 'assets')
                        catalog = make_catalog(trainer.alignment, assets)
                        generated = gate.generate_target_free(trainer, batch, catalog, beam=2)
                        checked = gate.check_generation(generated, assets.n_items, expected_batch=2)
                    self.assertTrue(result['R']['tokenizer_unchanged'])
                    self.assertTrue(result['T']['recommender_unchanged'])
                    self.assertTrue(result['replay_R']['exact_next_update'])
                    self.assertTrue(result['replay_T']['exact_next_update'])
                    self.assertTrue(result['catalog_refresh']['verified_against_current_codebooks'])
                    self.assertEqual(checked['examples'], 2)
                    self.assertTrue(all(len(set(row)) == 5 for row in checked['best_lists']))
            gate.exact_equal(signatures[0], signatures[1])

    def test_formal_worker_rejects_cpu_instead_of_claiming_h100_pass(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            args = SimpleNamespace(output=directory, stage_timeout=5)
            with patch.dict(os.environ, {'LISTWISE_PREFLIGHT_SUPERVISED': '1'}), \
                    patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaisesRegex(gate.GateFailure, 'requires CUDA'):
                    gate.worker(args)

    def test_fixed_samples_are_real_indices_reproducible_and_unique(self):
        actual = gate.fixed_indices(2000, 1024, 2024)
        np.testing.assert_array_equal(actual, gate.fixed_indices(2000, 1024, 2024))
        self.assertEqual(len(np.unique(actual)), 1024)
        with self.assertRaises(ValueError):
            gate.fixed_indices(3, 4, 2024)

    def test_decode_cli_rejects_repeated_or_too_small_profile_before_launch(self):
        for extra in (['--decode-batches', '1'], ['--decode-batches', '4', '--eval-batch', '1']):
            with tempfile.TemporaryDirectory(dir=TEMP) as directory:
                output = Path(directory) / 'never_started'
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    gate.main(['--output', str(output), '--remaining-gpu-seconds', '3600', *extra])
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
