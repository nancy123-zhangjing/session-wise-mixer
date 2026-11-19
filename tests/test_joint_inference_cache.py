"""Shared CPU inference/cache checks for the joint DiffGRM path.

Uses the stage07 synthetic catalog fixture and real model implementation.
No stage08 module, dataset download, remote platform or CUDA work is needed.
"""
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    '_joint_inference_fixture', ROOT / 'tests/test_item_local_alternating_joint.py',
)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


@contextmanager
def shared_joint_model(item_local=False):
    original_config = fixture.fixture.tiny_config

    def config(_enabled=None):
        return original_config(item_local)

    with mock.patch.object(fixture.fixture, 'tiny_config', config):
        with fixture.joint_model(mixer_enabled=True) as model:
            yield model


class JointInferenceCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_inference_is_label_independent_and_training_requires_target(self):
        for item_local in (False, True):
            with self.subTest(item_local=item_local), shared_joint_model(item_local) as model:
                model.set_joint_phase('recommender')
                model.eval()
                batch = fixture.joint_batch()
                history = {key: value for key, value in batch.items()
                           if key != 'target_item_id'}
                expected = model(batch, return_loss=False).hidden_states.detach()
                changed_target = dict(batch, target_item_id=torch.tensor([6, 7]))
                self.assertTrue(torch.equal(
                    expected, model(changed_target, return_loss=False).hidden_states,
                ))
                self.assertTrue(torch.equal(
                    expected, model(history, return_loss=False).hidden_states,
                ))
                codes = model.joint_alignment.hard_codes.clone()
                with redirect_stdout(io.StringIO()):
                    result = model.generate(history, n_return_sequences=4)
                self.assertEqual(tuple(result.shape), (2, 2, 4))
                self.assertTrue(torch.equal(codes, model.joint_alignment.hard_codes))
                with self.assertRaises(KeyError):
                    model(history, return_loss=True)

    def test_ids_only_inference_derives_padding_mask_and_rejects_bad_shapes(self):
        for item_local in (False, True):
            with self.subTest(item_local=item_local), shared_joint_model(item_local) as model:
                model.set_joint_phase('recommender')
                model.eval()
                batch = fixture.joint_batch()
                explicit = {key: value for key, value in batch.items()
                            if key != 'target_item_id'}
                ids_only = {'history_item_ids': batch['history_item_ids']}
                expected = model(explicit, return_loss=False).hidden_states.detach()
                actual = model(ids_only, return_loss=False).hidden_states
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(int(torch.count_nonzero(actual[~batch['history_mask']])), 0)
                marked_pad = dict(ids_only, history_mask=torch.ones_like(batch['history_mask']))
                self.assertTrue(torch.equal(
                    expected, model(marked_pad, return_loss=False).hidden_states,
                ))
                malformed = dict(ids_only, history_mask=torch.ones(2, 2, dtype=torch.bool))
                with self.assertRaisesRegex(ValueError, 'same shape'):
                    model(malformed, return_loss=False)
                with self.assertRaisesRegex(ValueError, 'history_item_ids must have shape'):
                    model({'history_item_ids': batch['history_item_ids'][0]}, return_loss=False)
                with redirect_stdout(io.StringIO()):
                    a = model.generate(explicit, n_return_sequences=4)
                    b = model.generate(ids_only, n_return_sequences=4)
                self.assertTrue(torch.equal(a, b))

    def test_phase_change_and_checkpoint_reload_invalidate_effective_codebook_cache(self):
        with shared_joint_model() as model:
            model.set_joint_phase('recommender')
            model.eval()
            batch = fixture.joint_batch()
            expected = model(batch, return_loss=False).hidden_states.detach()
            saved = copy.deepcopy(model.state_dict())
            alignment = model.joint_alignment
            original_cache = alignment._recommender_codebook_cache.clone()
            self.assertGreater(original_cache.numel(), 0)
            model.set_joint_phase('tokenizer')
            self.assertEqual(alignment._recommender_codebook_cache.numel(), 0)
            with torch.no_grad():
                alignment.mixer.base_codebooks.add_(0.1)
            model.set_joint_phase('recommender')
            self.assertFalse(torch.equal(original_cache, alignment._recommender_codebook_cache))
            model.load_state_dict(saved, strict=True)
            self.assertEqual(alignment._recommender_codebook_cache.numel(), 0)
            model.set_joint_phase('recommender')
            model.eval()
            self.assertTrue(torch.equal(original_cache, alignment._recommender_codebook_cache))
            self.assertTrue(torch.equal(expected, model(batch, return_loss=False).hidden_states))

    def test_cache_creation_stays_fp32_inside_bf16_autocast(self):
        with shared_joint_model() as model:
            alignment = model.joint_alignment
            with torch.no_grad():
                alignment.mixer.attention.out_proj.weight.normal_(0.0, 0.03)
            model.set_joint_phase('recommender')
            model.eval()
            with torch.no_grad(), torch.autocast('cpu', enabled=False):
                expected, _ = alignment.mixer.crossed_codebooks(return_attention=False)
            expected = expected.detach().clone()
            self.assertTrue(torch.equal(expected, alignment._recommender_codebook_cache))
            alignment.set_recommender_cache(False)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                alignment.set_recommender_cache(True)
            actual = alignment._recommender_codebook_cache
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(expected, actual))
            self.assertFalse(actual.requires_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
