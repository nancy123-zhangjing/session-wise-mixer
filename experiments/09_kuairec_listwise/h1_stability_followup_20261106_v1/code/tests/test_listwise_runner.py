"""End-to-end CPU fixture: actual Parquet, PSE bridge, four-layer model and beam."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_listwise_assets import make_assets, make_dataset
from scripts.listwise_diffgrm.run import build_context, evaluate, train

torch.set_num_threads(2)


class RunnerTests(unittest.TestCase):
    def test_shared_warmup_two_arms_resume_and_true_item_evaluation(self):
        temporary_root = ROOT / 'tmp/listwise_runner'
        temporary_root.mkdir(parents=True, exist_ok=True)
        for route in ('J', 'H'):
            with self.subTest(route=route), tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
                root = Path(tmp)
                data = make_dataset(root / 'dataset')
                assets = make_assets(data, root / 'assets')
                cfg = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
                cfg.update(architecture=route, dataset=str(data), assets=str(assets.root), history_max=7)
                cfg['model'].update(codebook_size=8, n_embd=16, n_inner=32,
                                    dropout=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
                cfg['training'].update(effective_batch_size=2, microbatch_size=1, warmup_epochs=1,
                    cycles=1, r_epochs_per_cycle=1, t_updates_per_cycle=1, decoder_warmup_updates=1,
                    checkpoint_every_updates=1)
                cfg['evaluation'].update(val_beam=2, test_beam=2, candidate_lists=2, batch_size=2)
                common = dict(device='cpu', precision='bf16', resume=None)
                warmup_dir = root / 'warmup'
                train(argparse.Namespace(command='warmup', output=str(warmup_dir), **common), cfg)
                shared = torch.load(warmup_dir / 'last.pt', weights_only=False)
                self.assertEqual(shared['r_updates'], 2)
                self.assertEqual(shared['t_updates'], 0)
                for arm in (route + '0', route + '1'):
                    output = root / arm
                    args = argparse.Namespace(command='train', output=str(output), arm=arm,
                                              shared_warmup=str(warmup_dir / 'last.pt'), **common)
                    train(args, cfg)
                    checkpoint = torch.load(output / 'last.pt', weights_only=False)
                    self.assertEqual((checkpoint['r_updates'], checkpoint['t_updates']), (4, 1))
                    self.assertTrue((output / 'best.pt').is_file())
                    initial = torch.load(output / 'initial.pt', weights_only=False)
                    self.assertTrue(all(torch.equal(v, shared['model'][k]) for k, v in initial['model'].items()))
                    # Resuming a completed run performs no duplicate update.
                    args.resume = str(output / 'last.pt')
                    train(args, cfg)
                    after = torch.load(output / 'last.pt', weights_only=False)
                    self.assertEqual(after['r_updates'], 4)
                    trainer, bound_assets, _, binding = build_context(cfg, 'cpu', 'bf16', arm.endswith('1'))
                    trainer.restore(after, binding, arm=arm)
                    result = evaluate(trainer, bound_assets, split='test', destination=output / 'test')
                    self.assertEqual(result['examples'], 3)
                    self.assertTrue(result['full_split'])
                    self.assertTrue(0 <= result['metrics']['list/ndcg@5'] <= 1)
                    self.assertIn('aggregate/ndcg@50', result['metrics'])
                    self.assertEqual(result['groups']['cold_target']['examples'], 3)


if __name__ == '__main__':
    unittest.main()
