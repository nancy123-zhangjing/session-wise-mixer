"""Real CPU evaluation equivalence and label-blind recovery subset checks."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_listwise_assets import make_assets, make_dataset
from scripts.listwise_diffgrm.data import KuaiRecListDataset
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.recovery_evaluation import (
    evaluate_recovery, extract_subset, load_subset, subset_manifest,
)
from scripts.listwise_diffgrm.run import build_context, evaluate, microbatches
from scripts.listwise_diffgrm.training import rng_state

torch.set_num_threads(2)
TEMP_ROOT = ROOT / 'tmp/listwise_recovery_evaluation'
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def training_state(trainer):
    return fingerprint(dict(
        model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict(),
        r_optimizer=trainer.r_optimizer.state_dict(), t_optimizer=trainer.t_optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict(), rng=rng_state(),
        modes=[module.training for root in (trainer.model, trainer.alignment) for module in root.modules()],
        requires_grad=[parameter.requires_grad for root in (trainer.model, trainer.alignment)
                       for parameter in root.parameters()],
        cache=trainer.alignment._recommender_codebook_cache,
    ))


class RecoveryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_dataset(self.root / 'dataset')
        assets = make_assets(self.dataset, self.root / 'assets')
        config = json.loads((ROOT / 'configs/09_kuairec_listwise/base.json').read_text())
        config.update(architecture='H', dataset=str(self.dataset), assets=str(assets.root), history_max=7)
        config['model'].update(codebook_size=8, n_embd=16, n_inner=32,
                               dropout=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        config['training'].update(effective_batch_size=2, microbatch_size=1, warmup_epochs=1,
            cycles=1, r_epochs_per_cycle=1, t_updates_per_cycle=1,
            decoder_warmup_updates=1, checkpoint_every_updates=1)
        config['evaluation'].update(val_beam=2, test_beam=2, candidate_lists=2, batch_size=2)
        with redirect_stdout(io.StringIO()):
            self.trainer, self.assets, train, _ = build_context(config, 'cpu', 'bf16', True)
        # Exercise a learned H1 state and a refreshed catalog, not mocked beam
        # output or a no-op tokenizer. These are synthetic data only.
        batches = list(microbatches(train, [0, 1], 1))
        self.trainer.update(batches, 'R')
        self.trainer.update(batches, 'T')
        self.trainer.refresh_catalog()
        self.validation = KuaiRecListDataset(self.dataset, 'validation', assets=self.assets)

    def test_new_full_evaluation_exactly_matches_legacy_predictions_and_metrics(self):
        old_directory, new_directory = self.root / 'legacy', self.root / 'recovery'
        old = evaluate(self.trainer, self.assets, split='validation', destination=old_directory)
        new = evaluate_recovery(self.trainer, self.assets, self.validation, new_directory)
        old_rows = read_rows(old_directory / 'per_sample.jsonl.gz')
        new_rows = read_rows(new_directory / 'per_sample.jsonl.gz')
        self.assertEqual(len(new_rows), len(self.validation))
        # Includes targets, user/index identity, every prediction, scores,
        # complete/beam masks, and all per-sample metrics.
        self.assertEqual(new_rows, old_rows)
        for name in ('split', 'examples', 'full_split', 'dataset_binding', 'catalog_size',
                     'metric_protocol', 'beam', 'inference_precision', 'architecture',
                     'seed', 'r_updates', 't_updates', 'decode_counters', 'metrics', 'user_average'):
            self.assertEqual(new[name], old[name], name)

    def test_direct_subset_and_extraction_from_full_predictions_agree(self):
        full_directory, subset_directory = self.root / 'full', self.root / 'subset'
        full = evaluate_recovery(self.trainer, self.assets, self.validation, full_directory)
        manifest = subset_manifest(self.dataset, 2)
        indices = [row['index'] for row in manifest['rows']]
        direct = evaluate_recovery(self.trainer, self.assets, self.validation, subset_directory,
                                   indices=indices, subset={k: v for k, v in manifest.items() if k != 'rows'})
        with patch('scripts.listwise_diffgrm.recovery_evaluation.generate_lists',
                   side_effect=AssertionError('Extraction must not run model inference')):
            extracted = extract_subset(full_directory / 'per_sample.jsonl.gz', manifest)
        self.assertTrue(full['full_split'])
        self.assertFalse(direct['full_split'])
        self.assertEqual(extracted['examples'], 2)
        self.assertEqual(extracted['metrics'], direct['metrics'])
        full_rows = {row['sample_id']: row for row in read_rows(full_directory / 'per_sample.jsonl.gz')}
        subset_rows = read_rows(subset_directory / 'per_sample.jsonl.gz')
        self.assertEqual([row['sample_id'] for row in subset_rows], [row['sample_id'] for row in manifest['rows']])
        self.assertEqual([row['dataset_index'] for row in subset_rows], indices)
        for row in subset_rows:
            self.assertEqual(row['metrics'], full_rows[row['sample_id']]['metrics'])
            self.assertEqual(row['predictions'], full_rows[row['sample_id']]['predictions'])

    def test_evaluation_success_and_early_exception_restore_training_state(self):
        trainer = self.trainer
        trainer.model.train()
        trainer.alignment.train()
        # A mixed-mode subtree and populated cache must also survive.
        list(trainer.model.modules())[-1].eval()
        trainer.alignment.set_recommender_cache(True)
        cache_identity = id(trainer.alignment._recommender_codebook_cache)
        before = training_state(trainer)
        evaluate_recovery(trainer, self.assets, self.validation, self.root / 'success')
        self.assertEqual(training_state(trainer), before)
        self.assertEqual(id(trainer.alignment._recommender_codebook_cache), cache_identity)
        with patch('scripts.listwise_diffgrm.recovery_evaluation.make_catalog',
                   side_effect=RuntimeError('catalog construction failed')):
            with self.assertRaisesRegex(RuntimeError, 'catalog construction'):
                evaluate_recovery(trainer, self.assets, self.validation, self.root / 'interrupted')
        self.assertEqual(training_state(trainer), before)
        self.assertEqual(id(trainer.alignment._recommender_codebook_cache), cache_identity)
        self.assertFalse((self.root / 'interrupted/summary.json').exists())

    def test_test_split_and_invalid_index_selection_are_rejected(self):
        test = KuaiRecListDataset(self.dataset, 'test', assets=self.assets)
        with self.assertRaisesRegex(ValueError, 'test'):
            evaluate_recovery(self.trainer, self.assets, test, self.root / 'forbidden-test')
        invalid = ([], [0, 0], [-1], [len(self.validation)], [0.5], [True], [[0, 1]])
        for number, indices in enumerate(invalid):
            with self.subTest(indices=indices):
                before = training_state(self.trainer)
                with self.assertRaises((ValueError, IndexError)):
                    evaluate_recovery(self.trainer, self.assets, self.validation,
                                      self.root / f'invalid-{number}', indices=indices)
                self.assertEqual(training_state(self.trainer), before)

    def test_extraction_rejects_missing_or_duplicate_locked_samples(self):
        directory = self.root / 'full-for-extraction'
        evaluate_recovery(self.trainer, self.assets, self.validation, directory)
        rows = read_rows(directory / 'per_sample.jsonl.gz')
        manifest = subset_manifest(self.dataset, 2)
        selected = {row['sample_id'] for row in manifest['rows']}
        picked = next(row for row in rows if row['sample_id'] in selected)
        corruptions = {
            'duplicate': rows + [picked],
            'missing': [row for row in rows if row['sample_id'] != picked['sample_id']],
        }
        for name, corrupt in corruptions.items():
            path = self.root / f'{name}.jsonl.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                for row in corrupt:
                    stream.write(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                extract_subset(path, manifest)


class RecoverySubsetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = [f'用户/sample-{i:02d}' for i in range(30)]
        self.path = self.root / 'validation.parquet'
        pq.write_table(pa.table({'sample_id': self.ids, 'target_item_ids': [[1, 2, 3, 4, 5]] * 30}), self.path)

    def test_subset_is_pure_sha256_of_exact_utf8_ids_and_reads_no_labels(self):
        actual_read = pq.read_table
        with patch('pyarrow.parquet.read_table', wraps=actual_read) as read:
            manifest = subset_manifest(self.root, 7)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.args, (self.path,))
        self.assertEqual(read.call_args.kwargs, {'columns': ['sample_id']})
        expected_ids = sorted(self.ids, key=lambda value: (hashlib.sha256(value.encode('utf-8')).hexdigest(), value))[:7]
        self.assertEqual([row['sample_id'] for row in manifest['rows']], expected_ids)
        self.assertNotEqual([row['index'] for row in manifest['rows']], list(range(7)))
        for row in manifest['rows']:
            self.assertEqual(self.ids[row['index']], row['sample_id'])
            self.assertEqual(row['selection_sha256'], hashlib.sha256(row['sample_id'].encode('utf-8')).hexdigest())
        # Changing labels and physical row order cannot select different IDs.
        pq.write_table(pa.table({'sample_id': self.ids[::-1], 'target_item_ids': [[9, 8, 7, 6, 5]] * 30}), self.path)
        changed = subset_manifest(self.root, 7)
        self.assertEqual([row['sample_id'] for row in changed['rows']], expected_ids)
        self.assertNotEqual(changed['validation_sha256'], manifest['validation_sha256'])

    def test_loaded_manifest_rejects_tampered_rows_and_changed_source_bytes(self):
        manifest = subset_manifest(self.root, 7)
        path = self.root / 'subset.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        self.assertEqual(load_subset(path, self.root), manifest)
        bad = copy.deepcopy(manifest)
        bad['rows'][0]['index'] = (bad['rows'][0]['index'] + 1) % len(self.ids)
        path.write_text(json.dumps(bad), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_subset(path, self.root)
        path.write_text(json.dumps(manifest), encoding='utf-8')
        pq.write_table(pa.table({'sample_id': self.ids, 'new_column': list(range(30))}), self.path)
        with self.assertRaises(ValueError):
            load_subset(path, self.root)


if __name__ == '__main__':
    unittest.main()
