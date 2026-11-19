"""Crash/replacement ownership cases for long-running listwise training."""
import json
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.listwise_diffgrm import run


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'tmp/listwise_recovery'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_crash_during_promotion_cannot_advance_last_past_saved_best(self):
        old = dict(best_score=0.1, phase=5, weight=torch.tensor([1.0]))
        new = dict(best_score=0.2, phase=9, weight=torch.tensor([2.0]))
        run.publish_checkpoint(self.root, old, save_best=True)
        actual_save = run.atomic_save

        def crash_before_last(path, state):
            if Path(path).name == 'last.pt':
                raise OSError('simulated interruption during cursor publication')
            actual_save(path, state)

        with patch.object(run, 'atomic_save', side_effect=crash_before_last):
            with self.assertRaisesRegex(OSError, 'simulated interruption'):
                run.publish_checkpoint(self.root, new, save_best=True)
        last = torch.load(self.root / 'last.pt', weights_only=False)
        best = torch.load(self.root / 'best.pt', weights_only=False)
        self.assertEqual(last['phase'], 5)
        self.assertEqual(best['phase'], 9)
        self.assertLessEqual(last['best_score'], best['best_score'])
        run.publish_checkpoint(self.root, new, save_best=True)
        self.assertEqual(torch.load(self.root / 'last.pt', weights_only=False)['phase'], 9)

    def test_existing_lock_blocks_resume_without_platform_evidence(self):
        directory = self.root / 'arm'
        with run.run_lock(directory):
            owner = json.loads((directory / '.running').read_text())
            self.assertTrue(owner['hostname'])
            self.assertGreater(owner['pid'], 0)
            with self.assertRaisesRegex(RuntimeError, 'platform job'):
                with run.run_lock(directory, resume=True):
                    self.fail('Second owner entered the same run')
        self.assertFalse((directory / '.running').exists())

    def test_old_owner_does_not_unlink_replacement_lock(self):
        directory = self.root / 'arm'
        with run.run_lock(directory):
            (directory / '.running').write_text(json.dumps(dict(token='new-owner')))
        self.assertEqual(json.loads((directory / '.running').read_text())['token'], 'new-owner')

    def test_validation_interruption_resumes_without_repeating_optimizer_updates(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'tests'))
        from test_listwise_assets import make_assets, make_dataset
        data = make_dataset(self.root / 'dataset')
        assets = make_assets(data, self.root / 'assets')
        cfg = json.loads((root / 'configs/09_kuairec_listwise/base.json').read_text())
        cfg.update(architecture='H', dataset=str(data), assets=str(assets.root), history_max=7)
        cfg['model'].update(codebook_size=8, n_embd=16, n_inner=32,
                            dropout=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        cfg['training'].update(effective_batch_size=2, microbatch_size=1, warmup_epochs=1,
            cycles=1, r_epochs_per_cycle=1, t_updates_per_cycle=1, decoder_warmup_updates=1,
            checkpoint_every_updates=100)
        cfg['evaluation'].update(val_beam=2, test_beam=2, candidate_lists=2, batch_size=2)
        output = self.root / 'interrupted_warmup'
        args = argparse.Namespace(command='warmup', output=str(output), device='cpu',
                                  precision='bf16', resume=None)
        with patch.object(run, 'evaluate', side_effect=OSError('validation interrupted')):
            with self.assertRaisesRegex(OSError, 'validation interrupted'):
                run.train(args, cfg)
        checkpoint = torch.load(output / 'last.pt', weights_only=False)
        self.assertEqual(checkpoint['r_updates'], 2)
        self.assertEqual(checkpoint['progress']['phase'], 0)
        self.assertEqual(checkpoint['progress']['update'], 2)
        args.resume = str(output / 'last.pt')
        with patch.object(run.AlternatingTrainer, 'update',
                          side_effect=AssertionError('Completed training must not be repeated')):
            run.train(args, cfg)
        final = torch.load(output / 'last.pt', weights_only=False)
        self.assertEqual(final['r_updates'], 2)
        self.assertEqual(final['progress']['phase'], 1)
        self.assertTrue((output / 'best.pt').exists())


if __name__ == '__main__':
    unittest.main()
