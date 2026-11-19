import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_session_ar_core import tiny
from scripts.session_wise_ar.catalog import Catalog, add_suffix
from scripts.session_wise_ar import runtime
from scripts.rqcd_mixer.common import read_state


class FakeSplit:
    def __init__(self, root, split, cfg):
        self.split = split
        self.sample_ids = [f'{split}-{i}' for i in range(cfg['split_counts'][split])]
        self.users = [i+1 for i in range(len(self.sample_ids))]
        self.history = torch.tensor([[0, 6, 7]]*len(self.sample_ids))
        self.target = torch.tensor([[1, 2, 3, 4, 5]]*len(self.sample_ids))
    def __len__(self): return len(self.sample_ids)
    def batch(self, idx, device): return self.history[idx].to(device), self.target[idx].to(device)


class Scores:
    def __init__(self): self.contexts = []
    def encode(self, context, sid, users):
        self.contexts.append(context.clone())
        return torch.zeros(len(context), 1, 1), torch.ones(len(context), 1)
    def next_logprobs(self, hidden, mask, prefix):
        # Prefer digit zero; deterministic ties exercise original-ID tie breaking.
        logits = torch.tensor([0., -1., -2., -3.]).expand(len(hidden), -1)
        return logits.log_softmax(-1)


class ExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_legal_item_beam_prefix_and_duplicates(self):
        semantic = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
        sid = add_suffix(semantic, 4)
        catalog = Catalog(sid, 4)
        rec = Scores()
        pred = catalog.generate(rec, torch.tensor([[0, 6]]), torch.tensor([0]), length=5, width=20)
        self.assertEqual(len(set(pred[0].tolist())), 5)
        self.assertEqual(pred[0, 0].item(), 1)
        self.assertEqual(rec.contexts[1][0].tolist(), [6, 1, 0])
        self.assertEqual(rec.contexts[-1][0, 1:5].tolist(), pred[0, :4].tolist())

    def test_suffix_overflow_fails(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            add_suffix(torch.zeros(9, 3, dtype=torch.long), 8)

    def test_pretrain_records_overflow_without_building_catalog(self):
        from scripts.session_wise_ar.tokenizer import TigerRQ
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = tiny('AR-T0')
            cfg['rq'].update(epochs=1, eval_interval=1, batch_size=8)
            content = root/'content.npy'
            values = np.random.default_rng(2).normal(size=(12, 8)).astype('float32')
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            np.save(content, values)
            with patch.object(runtime, 'bind_inputs', return_value={'synthetic': True}), \
                 patch.object(runtime, 'catalog_original_ids', return_value=list(range(1, 13))):
                runtime.prepare(cfg, root/'data', content, root/'parent', 'cpu')
                with patch.object(TigerRQ, 'indices', return_value=torch.zeros(12, 3, dtype=torch.long)):
                    result = runtime.pretrain(cfg, root/'parent', root/'data', content, root/'pre', 'cpu')
            stats = result['records'][-1]['encoding']
            self.assertTrue(result['done'])
            self.assertFalse(stats['suffix_capacity_ok'])
            self.assertEqual(stats['max_collision_bucket'], 12)
            self.assertAlmostEqual(stats['collision_redundancy'], 11/12)

    def assert_state_equal(self, left, right):
        for key in ('tokenizer', 'recommender'):
            for name, value in left[key].items():
                torch.testing.assert_close(value, right[key][name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(left['sid'], right['sid'], rtol=0, atol=0)
        self.assertEqual(left['progress']['step'], right['progress']['step'])

    def test_synthetic_stages_resume_and_best_final(self):
        # Real model/optimizer/beam on artificial data, never project data/GPU.
        for arm in ('AR-T0', 'AR-T1', 'AR-D0', 'AR-D1'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root/'dataset'
                cfg = tiny(arm)
                cfg['split_counts'] = {'train': 3, 'validation': 2, 'test': 2}
                cfg['rq'].update(epochs=2, warmup_epochs=0, eval_interval=1, batch_size=8)
                cfg['training'].update(batch_size=8, warmup_updates=0,
                                        max_steps=3 if arm=='AR-T0' else None,
                                        max_epochs=None if arm=='AR-T0' else 2)
                cfg['model']['dropout'] = 0.1
                rng = np.random.default_rng(2)
                values = rng.normal(size=(12, 8)).astype('float32')
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                content = root/'content.npy'; np.save(content, values)
                prepared, pre, full, resumed = [root/x for x in ('prepared', 'pre', 'full', 'resumed')]
                with patch.object(runtime, 'bind_inputs', return_value={'synthetic': True}), \
                     patch.object(runtime, 'catalog_original_ids', return_value=list(range(1,13))), \
                     patch.object(runtime, 'PackedSplit', FakeSplit):
                    runtime.prepare(cfg, dataset, content, prepared, 'cpu')
                    runtime.pretrain(cfg, prepared, dataset, content, pre, 'cpu', _stop_after=1)
                    runtime.pretrain(cfg, prepared, dataset, content, pre, 'cpu', resume=True)
                    runtime.train(cfg, prepared, pre, dataset, content, full, 'cpu')
                    # Stop exactly before the epoch-boundary validation.
                    runtime.train(cfg, prepared, pre, dataset, content, resumed, 'cpu', _stop_after=2)
                    runtime.train(cfg, prepared, pre, dataset, content, resumed, 'cpu', resume=True)
                    self.assert_state_equal(read_state(full/'last.pt'), read_state(resumed/'last.pt'))
                    result = runtime.test(cfg, prepared, pre, dataset, content, full, root/'test', 'cpu')
                    self.assertIn('ndcg@5', result['best']['metrics'])
                    self.assertEqual(result['final']['metrics']['completion_rate'], 1)
                    self.assertTrue(json.loads((full/'complete.json').read_text())['progress']['done'])
                    if arm.startswith('AR-T'):
                        frozen = read_state(pre/'selected.pt')['tokenizer']
                        final = read_state(full/'last.pt')['tokenizer']
                        for key in frozen: torch.testing.assert_close(frozen[key], final[key], atol=0, rtol=0)
                    if arm == 'AR-D1':
                        interrupted = root/'interrupted_validation'
                        original = runtime.evaluate
                        def crash_after_validation(*args, **kwargs):
                            original(*args, **kwargs)
                            raise RuntimeError('simulated failure after prediction write')
                        with patch.object(runtime, 'evaluate', side_effect=crash_after_validation):
                            with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                                runtime.train(cfg, prepared, pre, dataset, content, interrupted, 'cpu')
                        runtime.train(cfg, prepared, pre, dataset, content, interrupted, 'cpu', resume=True)
                        self.assert_state_equal(read_state(full/'last.pt'), read_state(interrupted/'last.pt'))


if __name__ == '__main__': unittest.main()
