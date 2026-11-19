"""Execution equivalence with real backbone, stochastic R/T, tail and resume."""
import copy
import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from tests.test_listwise_assets import make_dataset, make_assets
from scripts.listwise_diffgrm.data import KuaiRecListDataset
from scripts.listwise_diffgrm.execution import TrainingTensorCache, prefetched_updates, TrainingLog
from scripts.listwise_diffgrm.run import build_context, phase_indices, microbatches, training_updates, evaluate, train
from scripts.listwise_diffgrm.training import rng_state, restore_rng, seed_all, cpu_copy

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ExecutionTests(unittest.TestCase):
    def assert_exact(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_exact(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_exact(x, y)
        else:
            self.assertEqual(a, b)

    def config(self, root, mode, optimized):
        data = make_dataset(root / 'data')
        assets = make_assets(data, root / 'assets')
        cfg = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
        cfg.update(dataset=str(data), assets=str(assets.root), history_max=7)
        cfg['model'].update(codebook_size=8, n_embd=16, n_inner=32, h_prefix_mode=mode,
                            dropout=0.1, attn_pdrop=0.1, resid_pdrop=0.1)
        cfg['training'].update(effective_batch_size=2, microbatch_size=1, warmup_epochs=1,
                              cycles=1, r_epochs_per_cycle=1, t_updates_per_cycle=1,
                              decoder_warmup_updates=1, checkpoint_every_updates=1)
        cfg['evaluation'].update(val_beam=2, test_beam=2, candidate_lists=2, batch_size=2)
        cfg['execution'] = {'optimized_io': optimized}
        return cfg

    def test_cached_batches_match_collator_order_tail_duplicates_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root, 'asymmetric', True)
            data = KuaiRecListDataset(root / 'data', 'train')
            before = rng_state()
            cache = TrainingTensorCache(data, chunk_size=2)
            self.assert_exact(rng_state(), before)
            indices = np.array([2, 0, 2, 1, 0])
            original = list(microbatches(data, indices, 1))
            with prefetched_updates(cache, indices, 2, 1, 0, 'cpu') as batches:
                actual = [(u, values) for u, values in batches]
            self.assertEqual([u for u, _ in actual], [0, 1, 2])
            flat = [b for _, batch in actual for b in batch]
            for a, b in zip(flat, original):
                self.assert_exact(a, {k: b[k] for k in cache.fields})
            self.assert_exact(rng_state(), before)
            with prefetched_updates(cache, indices, 2, 1, 2, 'cpu') as batches:
                resumed = list(batches)
            self.assertEqual(len(resumed), 1)
            self.assert_exact(resumed[0][1], actual[2][1])
            # Original data validity checks still run during cache construction.
            class BadData:
                def __len__(self): return len(data)
                def collator(self): return data.collator()
                def __getitem__(self, i):
                    row = data[i]
                    row['target_item_ids'][0] = 0
                    return row
            with self.assertRaises(ValueError):
                TrainingTensorCache(BadData())

    def paired_run(self, mode, mixer, device='cpu'):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg = self.config(root, mode, False)
            reference, assets, data, _ = build_context(cfg, device, 'bf16', mixer)
            fast_cfg = copy.deepcopy(cfg)
            fast_cfg['execution']['optimized_io'] = True
            fast_cfg['execution']['optimized_model_checks'] = True
            fast, fast_assets, fast_data, fast_binding = build_context(fast_cfg, device, 'bf16', mixer)
            self.assert_exact(reference.model.state_dict(), fast.model.state_dict())
            self.assert_exact(reference.alignment.state_dict(), fast.alignment.state_dict())
            records = []
            for trainer, ds, config in ((reference, data, cfg), (fast, fast_data, fast_cfg)):
                events = []
                for phase, row in enumerate(trainer.plan):
                    seed_all(cfg['seed'] + 100003 * (phase + 1))
                    indices = phase_indices(config, len(ds), phase, row)
                    with training_updates(ds, indices, config, 0, device) as updates:
                        for u, batches in updates:
                            values = trainer.update(batches, row['kind'])
                            events.append(cpu_copy(dict(values=values, model=trainer.model.state_dict(),
                                alignment=trainer.alignment.state_dict(),
                                grads={k:p.grad for k,p in trainer.model.named_parameters()},
                                t_grads={k:p.grad for k,p in trainer.alignment.named_parameters()},
                                r_opt=trainer.r_optimizer.state_dict(), t_opt=trainer.t_optimizer.state_dict(),
                                scheduler=trainer.scheduler.state_dict(), rng=rng_state())))
                    if row['kind'] == 'T':
                        trainer.refresh_catalog()
                    if row['kind'] == 'T' and trainer is fast:
                        # Resume at the real phase boundary, including full RNG/optimizers.
                        saved = trainer.checkpoint(fast_binding, {'phase':phase + 1,'update':0}, 'H1' if mixer else 'H0')
                        replacement, _, _, binding = build_context(fast_cfg, device, 'bf16', mixer)
                        replacement.restore(saved, binding, arm='H1' if mixer else 'H0')
                        trainer = replacement
                records.append(events)
                if config['execution']['optimized_io']:
                    fast = trainer
            self.assert_exact(records[0], records[1])
            # Full toy evaluation is unchanged and caches ONLY static data.
            ref = evaluate(reference, assets, split='validation', destination=root/'ref')
            got = evaluate(fast, fast_assets, split='validation', destination=root/'fast')
            self.assert_exact(ref['metrics'], got['metrics'])
            import gzip
            def predictions(path):
                with gzip.open(path/'per_sample.jsonl.gz', 'rt', encoding='utf-8') as f:
                    return [json.loads(line) for line in f]
            self.assert_exact(predictions(root/'ref'), predictions(root/'fast'))
            cached = fast._evaluation_data.copy()
            evaluate(fast, fast_assets, split='validation', destination=root/'fast_again')
            self.assertEqual([id(v) for v in cached.values()], [id(v) for v in fast._evaluation_data.values()])

    def test_four_versions_stochastic_updates_gradients_rng_resume_and_predictions(self):
        for mode in ('asymmetric', 'cross_attention'):
            for mixer in (False, True):
                with self.subTest(mode=mode, mixer=mixer):
                    self.paired_run(mode, mixer)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable; no H100 equivalence claim')
    def test_cuda_four_versions(self):
        for mode in ('asymmetric', 'cross_attention'):
            for mixer in (False, True):
                self.paired_run(mode, mixer, 'cuda:0')

    def test_log_records_and_flush_boundary_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'legacy', Path(tmp)/'buffered'
            for path, buffered in ((a, False), (b, True)):
                with TrainingLog(path, buffered) as log:
                    for n in range(7):
                        log.write(json.dumps({'step':n,'loss':0.1*n})+'\n')
                    log.flush()
                    self.assertEqual(len(path.read_text().splitlines()), 7)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_real_runner_warmup_fork_log_checkpoint_and_resume_all_versions(self):
        for mode in ('asymmetric', 'cross_attention'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                cfg = self.config(root, mode, False)
                finals = {}
                for fast in (False, True):
                    cfg = copy.deepcopy(cfg)
                    cfg['execution']['optimized_io'] = fast
                    base = root / ('fast' if fast else 'reference')
                    common = dict(device='cpu', precision='bf16', resume=None)
                    warmup = base/'warmup'
                    train(argparse.Namespace(command='warmup', output=str(warmup), **common), cfg)
                    for arm in ('H0', 'H1'):
                        out = base/arm
                        args = argparse.Namespace(command='train', output=str(out), arm=arm,
                            shared_warmup=str(warmup/'last.pt'), **common)
                        train(args, cfg)
                        saved = torch.load(out/'last.pt', weights_only=False)
                        values = {k:saved[k] for k in ('model','alignment','r_optimizer','t_optimizer',
                                                       'scheduler','rng','r_updates','t_updates')}
                        values['log'] = [{k:v for k,v in json.loads(line).items() if k!='attempt'}
                                         for line in (out/'training.jsonl').read_text().splitlines()]
                        values['best_score'] = saved['progress']['best_score']
                        values['best_epoch'] = saved['progress']['best_epoch']
                        if fast:
                            self.assert_exact(finals[arm], values)
                            args.resume = str(out/'last.pt')
                            train(args, cfg)
                            after = torch.load(out/'last.pt', weights_only=False)
                            self.assert_exact(saved, after)
                        else:
                            finals[arm] = values


if __name__ == '__main__':
    unittest.main()
