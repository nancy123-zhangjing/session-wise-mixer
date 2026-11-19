"""CPU oracle and state-isolation tests; no production data, GPU or remote job."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm.recovery_diagnostics import diagnostic_context, diagnose_catalog
from scripts.listwise_diffgrm.runtime import engine_module
from scripts.listwise_diffgrm.training import rng_state

CrossPSECodebookMixer = engine_module('cross_pse').CrossPSECodebookMixer
JointCrossPSEAlignment = engine_module('joint_cross_pse').JointCrossPSEAlignment


class TinyAlignment(nn.Module):
    """Synthetic coordinates with the production mixer and real cache API."""
    set_recommender_cache = JointCrossPSEAlignment.set_recommender_cache

    def __init__(self):
        super().__init__()
        base = torch.tensor([[[0.], [10.], [20.], [30.]]] * 2)
        self.mixer = CrossPSECodebookMixer(base, n_heads=1,
                                          train_base_codebooks=True, mixer_enabled=False)
        self.register_buffer('item_slots', torch.tensor([
            [[-999.], [-999.]],  # PAD must never enter usage or reconstruction.
            [[0.], [0.]], [[1.], [1.]], [[10.], [0.]], [[10.], [10.]], [[20.], [10.]],
        ]))
        # Deliberately stale, proving the diagnostic does not read/publish this.
        self.register_buffer('hard_codes', torch.full((6, 2), -1, dtype=torch.long))
        self.register_buffer('_recommender_codebook_cache', torch.empty(0), persistent=False)
        self.register_buffer('diger_optimizer_step', torch.tensor(200))


def tiny_trainer():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(.3))
    model.register_buffer('inference_counter', torch.tensor(7))
    model[0].last_attention = torch.tensor([.1, .9])
    model[0].capture_attention = True
    model.inference_cache = {'items': [torch.tensor([4., 5.])], 'hits': 2}
    alignment = TinyAlignment()
    alignment.set_recommender_cache(True)
    model.train()
    model[0].eval()  # Mixed tree must be restored, not flattened to root.training.
    alignment.eval()
    alignment.mixer.input_norm.train()
    return SimpleNamespace(model=model, alignment=alignment, r_updates=13860, t_updates=200)


def assert_tree_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for lvalue, rvalue in zip(left, right):
            assert_tree_equal(test, lvalue, rvalue)
    else:
        test.assertEqual(left, right)


class RecoveryDiagnosticTests(unittest.TestCase):
    def test_catalog_statistics_match_hand_calculated_oracle_without_refresh(self):
        trainer = tiny_trainer()
        original = trainer.alignment.hard_codes.clone()
        metrics, snapshot = diagnose_catalog(trainer, batch_size=2)
        expected = torch.tensor([[0, 0], [0, 0], [1, 0], [1, 1], [2, 1]])
        self.assertTrue(torch.equal(snapshot['hard_codes'], expected))
        self.assertTrue(torch.equal(trainer.alignment.hard_codes, original))
        self.assertEqual(metrics['num_items'], 5)
        self.assertEqual(metrics['used_codes_per_slot'], [3, 2])
        self.assertEqual(metrics['per_slot'][0]['counts'], [2, 2, 1, 0])
        self.assertEqual(metrics['per_slot'][1]['counts'], [3, 2, 0, 0])
        self.assertEqual(metrics['per_slot'][0]['frequencies'], [.4, .4, .2, 0.])
        entropy = -(.4 * math.log(.4) * 2 + .2 * math.log(.2))
        self.assertAlmostEqual(metrics['per_slot'][0]['entropy_nats'], entropy)
        self.assertAlmostEqual(metrics['per_slot'][0]['effective_code_count'], math.exp(entropy))
        self.assertAlmostEqual(metrics['per_slot'][0]['normalized_entropy'], entropy / math.log(4))
        self.assertEqual(metrics['unique_sid_count'], 4)
        self.assertEqual(metrics['collision_item_count'], 2)
        self.assertEqual(metrics['collision_item_fraction'], .4)
        self.assertEqual(metrics['collision_item_percent'], 40.)
        self.assertEqual(metrics['duplicate_row_count'], 1)
        self.assertEqual(metrics['largest_bucket_size'], 2)
        self.assertAlmostEqual(metrics['mean_nearest_reconstruction_sse'], .4)
        self.assertEqual(metrics['mixer_delta_over_base_global_norm'], 0.)
        self.assertTrue(all(value is None for value in metrics['previous_change'].values()))
        json.dumps(metrics, allow_nan=False)
        self.assertTrue(all(tensor.device.type == 'cpu' and not tensor.requires_grad
                            for tensor in snapshot.values()))
        # Snapshots own their storage and cannot mutate training buffers/parameters.
        snapshot['effective_codebooks'].zero_()
        self.assertEqual(float(trainer.alignment.mixer.base_codebooks[0, 1, 0]), 10.)

    def test_previous_snapshot_change_and_codebook_displacement_are_correct(self):
        trainer = tiny_trainer()
        _, first = diagnose_catalog(trainer)
        before = {key: value.clone() for key, value in first.items()}
        with torch.no_grad():
            trainer.alignment.mixer.base_codebooks[0, 1, 0] = -10.
        metrics, second = diagnose_catalog(trainer, first, batch_size=1)
        change = metrics['previous_change']
        self.assertEqual(change['sid_item_change_fraction'], .4)
        self.assertEqual(change['sid_digit_change_fraction'], .2)
        self.assertEqual(change['sid_change_fraction_per_slot'], [.4, 0.])
        self.assertEqual(change['effective_displacement_norm'], 20.)
        self.assertAlmostEqual(change['effective_displacement_over_previous_global_norm'],
                               20. / math.sqrt(2800.))
        self.assertTrue(torch.equal(second['hard_codes'][:, 0], torch.tensor([0, 0, 0, 0, 2])))
        assert_tree_equal(self, first, before)
        self.assertAlmostEqual(metrics['mean_nearest_reconstruction_sse'], 40.4)

    def test_context_restores_mixed_modes_rng_buffers_caches_and_gradients_on_exception(self):
        for fail in (False, True):
            with self.subTest(exception=fail):
                trainer = tiny_trainer()
                roots = (trainer.model, trainer.alignment)
                modes = [(module, module.training) for root in roots for module in root.modules()]
                buffers = [(module, name, value, value.clone()) for root in roots
                           for module in root.modules() for name, value in module._buffers.items()]
                parameters = list(trainer.model.parameters()) + list(trainer.alignment.parameters())
                for index, parameter in enumerate(parameters):
                    parameter.grad = torch.full_like(parameter, index + .5)
                gradients = [(p, p.requires_grad, p.grad, p.grad.clone()) for p in parameters]
                old_attention = trainer.model[0].last_attention
                old_cache = trainer.alignment._recommender_codebook_cache
                nested_cache_tensor = trainer.model.inference_cache['items'][0]
                state = rng_state()

                def measure():
                    with diagnostic_context(trainer):
                        self.assertFalse(torch.is_grad_enabled())
                        self.assertTrue(all(not module.training for module, _ in modes))
                        random.random()
                        np.random.random(4)
                        torch.rand(5)
                        trainer.alignment.hard_codes.fill_(3)
                        trainer.model.inference_counter.add_(10)
                        trainer.alignment.set_recommender_cache(False)
                        trainer.alignment.register_buffer('temporary_buffer', torch.ones(1))
                        trainer.model[0].last_attention.add_(2)
                        trainer.model[0].last_attention = None
                        trainer.model[0].capture_attention = False
                        trainer.model.inference_cache['items'][0].zero_()
                        trainer.model.inference_cache['hits'] = 99
                        trainer.model.new_inference_cache = [123]
                        if fail:
                            raise RuntimeError('simulated evaluation failure')

                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'simulated evaluation failure'):
                        measure()
                else:
                    measure()
                assert_tree_equal(self, rng_state(), state)
                for module, mode in modes:
                    self.assertEqual(module.training, mode)
                for module, name, original, value in buffers:
                    self.assertIs(module._buffers[name], original)
                    self.assertTrue(torch.equal(module._buffers[name], value))
                self.assertFalse(hasattr(trainer.alignment, 'temporary_buffer'))
                self.assertFalse(hasattr(trainer.model, 'new_inference_cache'))
                self.assertIs(trainer.alignment._recommender_codebook_cache, old_cache)
                self.assertIs(trainer.model[0].last_attention, old_attention)
                self.assertTrue(torch.equal(old_attention, torch.tensor([.1, .9])))
                self.assertTrue(trainer.model[0].capture_attention)
                assert_tree_equal(self, trainer.model.inference_cache,
                                  {'items': [torch.tensor([4., 5.])], 'hits': 2})
                self.assertIs(trainer.model.inference_cache['items'][0], nested_cache_tensor)
                for parameter, requires_grad, gradient, values in gradients:
                    self.assertEqual(parameter.requires_grad, requires_grad)
                    self.assertIs(parameter.grad, gradient)
                    self.assertTrue(torch.equal(parameter.grad, values))

    def test_diagnostic_is_chunk_independent_nested_safe_and_ignores_autocast(self):
        trainer = tiny_trainer()
        before = rng_state()
        reference, first = diagnose_catalog(trainer, batch_size=1)
        with diagnostic_context(trainer), torch.autocast('cpu', dtype=torch.bfloat16):
            actual, second = diagnose_catalog(trainer, first, batch_size=5)
        self.assertEqual(actual['per_slot'], reference['per_slot'])
        self.assertEqual(actual['mean_nearest_reconstruction_sse'], reference['mean_nearest_reconstruction_sse'])
        self.assertEqual(actual['previous_change']['sid_item_change_fraction'], 0.)
        self.assertEqual(actual['previous_change']['effective_displacement_norm'], 0.)
        self.assertEqual(second['effective_codebooks'].dtype, torch.float32)
        assert_tree_equal(self, rng_state(), before)

    def test_invalid_inputs_fail_without_publishing_sids_or_modes(self):
        trainer = tiny_trainer()
        original = trainer.alignment.hard_codes.clone()
        modes = [module.training for module in trainer.model.modules()]
        for batch_size in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                diagnose_catalog(trainer, batch_size=batch_size)
        _, snapshot = diagnose_catalog(trainer)
        snapshot['hard_codes'] = snapshot['hard_codes'][:1]
        with self.assertRaisesRegex(ValueError, 'previous snapshot shape'):
            diagnose_catalog(trainer, snapshot)
        with torch.no_grad():
            trainer.alignment.mixer.base_codebooks[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            diagnose_catalog(trainer)
        self.assertTrue(torch.equal(trainer.alignment.hard_codes, original))
        self.assertEqual([module.training for module in trainer.model.modules()], modes)

    def test_real_noisy_next_update_and_optimizer_state_are_unchanged(self):
        # Existing tiny H architecture uses real assets, alignment, trainer and
        # DIGER-ST. It is unrelated to the production KuaiRec dataset.
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_listwise_training import fixture
        with tempfile.TemporaryDirectory() as directory:
            trainer, batch = fixture(directory, 'H', True)
            trainer.update([batch], 'R')
            trainer.update([batch], 'T')
            binding = {'fixture': 'recovery-diagnostics'}
            checkpoint = trainer.checkpoint(binding, {'phase': 1, 'update': 1}, 'H1')
            expected = trainer.update([batch], 'T')
            expected_alignment = copy.deepcopy(trainer.alignment.state_dict())
            expected_optimizer = copy.deepcopy(trainer.t_optimizer.state_dict())
            trainer.restore(checkpoint, binding, arm='H1')
            trainer.alignment.set_recommender_cache(True)
            cache = trainer.alignment._recommender_codebook_cache.clone()
            original_buffers = {name: value.clone() for name, value in trainer.alignment.named_buffers()}
            original_optimizer = copy.deepcopy(trainer.t_optimizer.state_dict())
            original_r_optimizer = copy.deepcopy(trainer.r_optimizer.state_dict())
            original_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
            original_rng = rng_state()
            original_grads = [(p, p.grad, None if p.grad is None else p.grad.clone())
                              for root in (trainer.model, trainer.alignment) for p in root.parameters()]
            metrics, snapshot = diagnose_catalog(trainer, batch_size=4)
            json.dumps(metrics, allow_nan=False)
            self.assertEqual(snapshot['hard_codes'].shape, (9, 4))
            assert_tree_equal(self, original_rng, rng_state())
            assert_tree_equal(self, original_buffers, dict(trainer.alignment.named_buffers()))
            assert_tree_equal(self, original_optimizer, trainer.t_optimizer.state_dict())
            assert_tree_equal(self, original_r_optimizer, trainer.r_optimizer.state_dict())
            assert_tree_equal(self, original_scheduler, trainer.scheduler.state_dict())
            self.assertTrue(torch.equal(cache, trainer.alignment._recommender_codebook_cache))
            for parameter, reference, value in original_grads:
                self.assertIs(parameter.grad, reference)
                if value is not None:
                    self.assertTrue(torch.equal(parameter.grad, value))
            actual = trainer.update([batch], 'T')
            self.assertEqual(actual, expected)
            assert_tree_equal(self, trainer.alignment.state_dict(), expected_alignment)
            assert_tree_equal(self, trainer.t_optimizer.state_dict(), expected_optimizer)


class RecoveryPreflightHelperTests(unittest.TestCase):
    def test_fixed_validation_order_uses_exact_unsalted_utf8_ids(self):
        import hashlib
        from scripts.listwise_diffgrm.recovery_preflight import fixed_validation_rows
        ids = ['sample-2', '样本一', 'sample-10', 'sample-1']
        expected = sorted(ids, key=lambda value: (hashlib.sha256(value.encode('utf-8')).hexdigest(), value))[:3]
        rows = fixed_validation_rows(ids, 3)
        self.assertEqual([row['sample_id'] for row in rows], expected)
        self.assertEqual([ids[row['index']] for row in rows], expected)
        with self.assertRaisesRegex(ValueError, 'unique'):
            fixed_validation_rows(['a', 'a'], 1)

    def test_real_cpu_preflight_t_replay_checks_controller_and_rejects_formal_resume(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_listwise_h1_recovery import make_parent, make_trainer, START, STOP, PARENT_SHA
        from scripts.listwise_diffgrm.recovery import RecoveryController
        from scripts.listwise_diffgrm.recovery_preflight import gradient_groups, replay_next_update
        with tempfile.TemporaryDirectory() as directory:
            parent, binding, execution = make_parent(directory)
            trainer, batches = make_trainer(directory)
            controller = RecoveryController(trainer, 'decay', binding, execution, PARENT_SHA,
                                            start_phase=START, stop_phase=STOP)
            controller.fork(parent)
            controller.step(batches, 'T')
            gradients = gradient_groups(trainer, 'T')
            self.assertTrue(all(row['gradient_l1'] > 0 for row in gradients.values()))
            result = replay_next_update(controller, batches, 'T', directory, START + 1)
            self.assertTrue(result['exact_next_update'])
            saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            self.assertTrue(saved['preflight_only'])
            self.assertTrue(saved['inner']['preflight_only'])
            self.assertNotIn('preflight_only', saved['inner']['progress'])
            with self.assertRaisesRegex(ValueError, '[Pp]reflight'):
                controller.restore(saved)

    def test_real_cpu_disposable_r_replay_preserves_freeze_side(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_listwise_h1_recovery import make_parent, make_trainer, START, STOP, PARENT_SHA
        from scripts.listwise_diffgrm.recovery import RecoveryController, fingerprint
        from scripts.listwise_diffgrm.recovery_preflight import gradient_groups, replay_next_update
        with tempfile.TemporaryDirectory() as directory:
            parent, binding, execution = make_parent(directory)
            trainer, batches = make_trainer(directory)
            controller = RecoveryController(trainer, 'freeze', binding, execution, PARENT_SHA,
                                            start_phase=START, stop_phase=STOP)
            controller.fork(parent)
            frozen = fingerprint(dict(alignment=trainer.alignment.state_dict(),
                                      optimizer=trainer.t_optimizer.state_dict()))
            result = replay_next_update(controller, batches, 'R', directory, START + 1)
            self.assertTrue(result['exact_next_update'])
            gradients = gradient_groups(trainer, 'R')
            self.assertTrue(all(row['gradient_l1'] > 0 for row in gradients.values()))
            self.assertEqual(frozen, fingerprint(dict(alignment=trainer.alignment.state_dict(),
                                                     optimizer=trainer.t_optimizer.state_dict())))
            saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            self.assertTrue(saved['preflight_only'])
            with self.assertRaisesRegex(ValueError, '[Pp]reflight'):
                trainer.restore(saved, execution, arm=saved['arm'])


if __name__ == '__main__':
    unittest.main()
