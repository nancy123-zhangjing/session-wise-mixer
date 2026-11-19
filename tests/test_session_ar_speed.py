"""Opt-in speed paths for the TIGER/DIGER runtime keep FP32 semantics unchanged."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_session_ar_core import tiny
from test_session_ar_execution import FakeSplit
from scripts.session_wise_ar import runtime
from scripts.rqcd_mixer.common import read_state


def python_accumulate(values, count):
    """Former per-step accumulation: progress[k] = progress.get(k, 0) + v.item()*count."""
    total = 0
    for v in values:
        total = total+v.item()*count
    return total


class SpeedPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_device_meter_is_bit_identical_across_flushes(self):
        rng = torch.Generator().manual_seed(7)
        values = (torch.rand(50, generator=rng, dtype=torch.float32)*7).unbind()
        expected = python_accumulate(values, 256)
        progress = {'loss_sum': {}}
        meter = runtime.DeviceLossMeter()
        for i, v in enumerate(values, 1):
            meter.add(progress, {'total': v}, 256, torch.tensor([1.0]))
            if i in (3, 17, 18, 40):  # checkpoint / resume flushes at arbitrary points
                meter.flush(progress)
        meter.flush(progress)
        self.assertEqual(progress['loss_sum']['total'], expected)
        self.assertEqual(progress['last_gradient_norms'], [1.0])

    def test_finite_step_rejects_before_any_update(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.ones(2, 3)
        for bad in (float('nan'), float('inf')):
            loss = model(x).sum()*bad
            with self.assertRaises(FloatingPointError):
                runtime.finite_step(loss, [model], [optimizer], [None])
            for p, q in zip(model.parameters(), before):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
        # Finite update equals a plain SGD step (clip=None scales gradients by exactly 1).
        reference = torch.nn.Linear(3, 1); reference.load_state_dict(model.state_dict())
        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
        norms = runtime.finite_step(model(x).sum(), [model], [optimizer], [None])
        ref_opt.zero_grad(); reference(x).sum().backward(); ref_opt.step()
        self.assertEqual(norms.shape, (1,))
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_precision_option_is_validated(self):
        cfg = tiny('AR-T0')
        self.assertEqual(runtime.training_precision(cfg), cfg['execution']['precision'])
        cfg['execution']['precision'] = 'fp16'
        with self.assertRaisesRegex(ValueError, 'precision'):
            runtime.training_precision(cfg)

    def test_bf16_training_runs_resumes_and_keeps_tokenizer_frozen(self):
        for arm in ('AR-T0', 'AR-T1'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root/'dataset'
                cfg = tiny(arm)
                cfg['execution']['precision'] = 'bf16'
                cfg['split_counts'] = {'train': 3, 'validation': 2, 'test': 2}
                cfg['rq'].update(epochs=2, warmup_epochs=0, eval_interval=1, batch_size=8)
                cfg['training'].update(batch_size=8, warmup_updates=0, max_steps=3, max_epochs=None)
                cfg['model']['dropout'] = 0.1
                values = np.random.default_rng(2).normal(size=(12, 8)).astype('float32')
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                content = root/'content.npy'; np.save(content, values)
                prepared, pre, full, resumed = [root/x for x in ('prepared', 'pre', 'full', 'resumed')]
                with patch.object(runtime, 'bind_inputs', return_value={'synthetic': True}), \
                     patch.object(runtime, 'catalog_original_ids', return_value=list(range(1, 13))), \
                     patch.object(runtime, 'PackedSplit', FakeSplit):
                    runtime.prepare(cfg, dataset, content, prepared, 'cpu')
                    runtime.pretrain(cfg, prepared, dataset, content, pre, 'cpu')
                    runtime.train(cfg, prepared, pre, dataset, content, full, 'cpu')
                    runtime.train(cfg, prepared, pre, dataset, content, resumed, 'cpu', _stop_after=2)
                    runtime.train(cfg, prepared, pre, dataset, content, resumed, 'cpu', resume=True)
                a, b = read_state(full/'last.pt'), read_state(resumed/'last.pt')
                self.assertEqual(a['progress']['records'][-1]['loss'], b['progress']['records'][-1]['loss'])
                for key in a['recommender']:
                    torch.testing.assert_close(a['recommender'][key], b['recommender'][key], atol=0, rtol=0)
                    self.assertEqual(a['recommender'][key].dtype, torch.float32)
                frozen = read_state(pre/'selected.pt')['tokenizer']
                for key in frozen:
                    torch.testing.assert_close(frozen[key], a['tokenizer'][key], atol=0, rtol=0)
                self.assertTrue(all(np.isfinite(v) for v in a['progress']['records'][-1]['loss'].values()))


if __name__ == '__main__': unittest.main()
