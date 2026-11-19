"""CPU restore portability checks; no production files, remote calls or GPU."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_listwise_h1_recovery import START, STOP, PARENT_SHA, make_parent, make_trainer
from scripts.analysis.audit_h1_recovery_restore import (
    PARENT_SHA256, cold_import, prove_relocation, require_final_state, restore_original_state, source_path,
)
from scripts.listwise_diffgrm import recovery


class RelocationTests(unittest.TestCase):
    def test_only_explicit_asset_paths_can_change(self):
        recorded = dict(config=dict(dataset='/recorded/data', assets='/recorded/assets', seed=2024),
                        code={'model.py': 'a'}, engine={'engine.py': 'b'}, data={'sha': 'c'}, asset_manifest='d')
        actual = copy.deepcopy(recorded)
        actual['config'].update(dataset='/local/data', assets='/local/assets')
        expected = copy.deepcopy(actual)
        proof = prove_relocation(recorded, actual)
        self.assertEqual(actual, expected)
        self.assertFalse(proof['recorded_binding_rewritten'])
        for field in ('code', 'engine', 'data', 'asset_manifest'):
            changed = copy.deepcopy(actual)
            changed[field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError):
                prove_relocation(recorded, changed)
        actual['config']['seed'] = 2025
        with self.assertRaises(ValueError):
            prove_relocation(recorded, actual)

    def test_final_production_receipt_and_cursor_are_required(self):
        saved = dict(format='stage09_h1_recovery_v1',
            protocol=dict(policy='decay', parent_sha256=PARENT_SHA256, start_phase=25, stop_phase=33),
            inner=dict(progress=dict(phase=33, update=0, boundary_prepared=False), r_updates=18018, t_updates=280))
        complete = dict(policy='decay', endpoint_epoch=26, phase=33, r_updates=18018, t_updates=280)
        require_final_state(saved, complete, 'decay')
        for item in ('preflight', 'not_final', 'receipt_count', 'policy'):
            changed, receipt = copy.deepcopy(saved), copy.deepcopy(complete)
            if item == 'preflight':
                changed['inner']['preflight_only'] = True
            elif item == 'not_final':
                changed['inner']['progress']['phase'] = 32
            elif item == 'receipt_count':
                receipt['r_updates'] -= 1
            else:
                changed['protocol']['policy'] = 'freeze'
            with self.subTest(item=item), self.assertRaises(ValueError):
                require_final_state(changed, receipt, 'decay')

    def test_cold_import_rejects_current_loaded_runtime(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh process'):
            cold_import(ROOT)

    def test_source_paths_cannot_escape(self):
        for value in ('../escape.py', '/absolute.py', 'dir\\other.py'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_path(ROOT, value)


class ActualTinyRestoreTests(unittest.TestCase):
    def test_both_real_tiny_endpoints_load_without_rewriting_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parent, parent_binding, execution_binding = make_parent(directory)
            for policy in ('decay', 'freeze'):
                with self.subTest(policy=policy):
                    trainer, batches = make_trainer(directory)
                    controller = recovery.RecoveryController(trainer, policy, parent_binding,
                        execution_binding, PARENT_SHA, start_phase=START, stop_phase=STOP)
                    controller.fork(parent)
                    while not controller.done:
                        skip = controller.row['kind'] == 'T' and policy == 'freeze'
                        while not skip and controller.progress['update'] < controller.row['updates']:
                            controller.step(batches, controller.row['kind'])
                        controller.prepare_boundary()
                        controller.complete_phase()
                    saved = controller.checkpoint()
                    # Explicitly exercise CPU inspection of a preserved CUDA-RNG field.
                    saved['inner']['rng']['cuda'] = [torch.get_rng_state().clone()]
                    before, parent_before = recovery.fingerprint(saved), recovery.fingerprint(parent)
                    restored_trainer, _ = make_trainer(directory)
                    actual = copy.deepcopy(execution_binding)
                    actual['config'].update(dataset='/read-only/local/data', assets='/read-only/local/assets')
                    restored, proof = restore_original_state(restored_trainer, parent, saved, actual, recovery)
                    self.assertTrue(restored.done)
                    self.assertTrue(proof['cpu_roundtrip_exact_except_cuda_rng'])
                    self.assertFalse(proof['cuda_exact_resume_verified'])
                    self.assertEqual(proof['restored_state_sha256'], proof['recorded_state_sha256'])
                    self.assertEqual(recovery.fingerprint(saved), before)
                    self.assertEqual(recovery.fingerprint(parent), parent_before)
                    self.assertEqual(restored.trainer.t_updates, 8 if policy == 'decay' else 4)
                    self.assertEqual(restored.trainer.r_updates, 5)


if __name__ == '__main__':
    unittest.main()
