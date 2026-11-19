"""Label-blind hash selection and evaluation-entry integration on CPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.listwise_diffgrm.subset import (
    DEFAULT_SALT, load_subset_indices, validation_subset_manifest, write_validation_subset,
)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='subset_test_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.ids = [f'sample_{i:03}' for i in range(30)]
        pq.write_table(pa.table({'sample_id': self.ids, 'target_item_ids': [[1, 2, 3, 4, 5]] * 30}),
                       self.root / 'validation.parquet')
        self.manifest = self.root / 'subset.json'

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_hash_selection_is_not_prefix_and_only_reads_id_column(self):
        real_read = pq.read_table
        with patch('pyarrow.parquet.read_table', wraps=real_read) as read:
            result = write_validation_subset(self.root, self.manifest, size=7)
        self.assertEqual(read.call_args.kwargs, {'columns': ['sample_id']})
        self.assertEqual(read.call_args.args[0].name, 'validation.parquet')
        expected = sorted(self.ids, key=lambda value: (hashlib.sha256((DEFAULT_SALT + '\n' + value).encode()).hexdigest(), value))[:7]
        self.assertEqual([row['sample_id'] for row in result['rows']], expected)
        self.assertNotEqual([row['index'] for row in result['rows']], list(range(7)))
        self.assertFalse(result['test_split_opened'])
        self.assertEqual(result['label_columns_read'], [])
        before = self.manifest.read_bytes()
        write_validation_subset(self.root, self.manifest, size=7)
        self.assertEqual(self.manifest.read_bytes(), before)
        indices, verified = load_subset_indices(self.manifest, self.root)
        self.assertEqual([self.ids[i] for i in indices], expected)
        self.assertEqual(verified['count'], 7)

    def test_selection_identity_is_independent_of_physical_row_order(self):
        first = validation_subset_manifest(self.root, size=7)
        pq.write_table(pa.table({'sample_id': self.ids[::-1]}), self.root / 'validation.parquet')
        second = validation_subset_manifest(self.root, size=7)
        self.assertEqual([row['sample_id'] for row in first['rows']], [row['sample_id'] for row in second['rows']])
        self.assertNotEqual(first['dataset_files'], second['dataset_files'])

    def test_tampered_subset_and_changed_source_are_rejected(self):
        original = write_validation_subset(self.root, self.manifest, size=7)
        corrupt = dict(original, rows=list(original['rows']))
        corrupt['rows'][0] = dict(corrupt['rows'][0], index=29)
        self.manifest.write_text(json.dumps(corrupt), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_subset_indices(self.manifest, self.root)
        self.manifest.write_text(json.dumps(original), encoding='utf-8')
        pq.write_table(pa.table({'sample_id': self.ids, 'extra': list(range(30))}), self.root / 'validation.parquet')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_subset_indices(self.manifest, self.root)
        with self.assertRaises(FileExistsError):
            write_validation_subset(self.root, self.manifest, size=7)

    def test_test_split_duplicate_ids_and_numeric_id_coercion_are_rejected(self):
        write_validation_subset(self.root, self.manifest, size=7)
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            load_subset_indices(self.manifest, self.root, split='test')
        pq.write_table(pa.table({'sample_id': ['x', 'x']}), self.root / 'validation.parquet')
        with self.assertRaisesRegex(ValueError, 'unique'):
            validation_subset_manifest(self.root, size=1)
        pq.write_table(pa.table({'sample_id': [1, 2]}), self.root / 'validation.parquet')
        with self.assertRaisesRegex(ValueError, 'text'):
            validation_subset_manifest(self.root, size=1)

    def test_evaluate_uses_locked_indices_and_records_protocol(self):
        from scripts.listwise_diffgrm.data import ListwiseCollator
        from scripts.listwise_diffgrm.decoding import GenerationOutput
        from scripts.listwise_diffgrm.run import evaluate

        locked = write_validation_subset(self.root, self.manifest, size=7)
        ids = self.ids

        class Dataset:
            binding = {'identity': 'toy'}
            def __init__(self, *args, **kwargs): pass
            def __len__(self): return len(ids)
            def collator(self): return ListwiseCollator(history_max=1, list_length=5, n_items=10)
            def __getitem__(self, index):
                return dict(sample_id=ids[index], user_id=index // 3, history_item_ids=[1],
                            history_timestamp_us=[1], target_item_ids=[1, 2, 3, 4, 5],
                            target_timestamp_us=[2, 3, 4, 5, 6])

        def generate(model, history, masks, catalog, **kwargs):
            n = history.shape[0]
            return GenerationOutput(torch.tensor([[[1, 2, 3, 4, 5]]] * n), torch.zeros(n, 1, 5, 4),
                                    torch.zeros(n, 1), torch.ones(n, 1, dtype=torch.bool),
                                    torch.ones(n, 1, dtype=torch.bool), {})

        cfg = dict(dataset=str(self.root), architecture='J', seed=2024, list_length=5,
                   evaluation=dict(val_beam=32, test_beam=32, batch_size=3, aggregate_ks=[10, 20, 50],
                                   candidate_lists=32, aggregation='list_score_order_then_item_position_first_occurrence'))
        trainer = SimpleNamespace(config=cfg, model=torch.nn.Linear(1, 1),
                                  alignment=SimpleNamespace(eval=lambda: None, hard_codes=torch.zeros(11, 4, dtype=torch.long)),
                                  device=torch.device('cpu'), r_updates=0, t_updates=0)
        with patch('scripts.listwise_diffgrm.data.KuaiRecListDataset', Dataset), \
             patch('scripts.listwise_diffgrm.run.make_catalog', return_value=None), \
             patch('scripts.listwise_diffgrm.decoding.generate_lists', side_effect=generate):
            summary = evaluate(trainer, SimpleNamespace(n_items=10), split='validation',
                               destination=self.root / 'eval', subset_manifest=self.manifest)
        import gzip
        with gzip.open(self.root / 'eval/per_sample.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        self.assertEqual([row['sample_id'] for row in rows], [row['sample_id'] for row in locked['rows']])
        self.assertEqual([row['dataset_index'] for row in rows], [row['index'] for row in locked['rows']])
        self.assertFalse(summary['full_split'])
        self.assertEqual(summary['examples'], 7)
        self.assertEqual(summary['subset']['selected_rows_sha256'], locked['selected_rows_sha256'])
        self.assertEqual(summary['catalog_size'], 10)
        with self.assertRaises(ValueError):
            evaluate(trainer, None, split='validation', destination=self.root / 'bad', limit=3, subset_manifest=self.manifest)
        with self.assertRaises(ValueError):
            evaluate(trainer, None, split='test', destination=self.root / 'bad', subset_manifest=self.manifest)

    def test_source_binding_uses_posix_paths_and_binds_engine_base_classes(self):
        from scripts.listwise_diffgrm.run import bindings
        root, engine, assets = self.root / 'repo', self.root / 'engine', self.root / 'assets'
        (root / 'scripts/listwise_diffgrm').mkdir(parents=True)
        (root / 'scripts/listwise_diffgrm/model.py').write_text('local = 1\n')
        (engine / 'genrec/models/DIFF_GRM').mkdir(parents=True)
        (engine / 'genrec/models/DIFF_GRM/model.py').write_text('engine = 1\n')
        for name in ('model', 'dataset', 'tokenizer'):
            (engine / f'genrec/{name}.py').write_text('base = 1\n')
        assets.mkdir()
        (assets / 'manifest.json').write_text('{}')
        modules = {f'genrec.{name}': None for name in ('model', 'dataset', 'tokenizer')}
        with patch('scripts.listwise_diffgrm.run.ROOT', root), \
             patch('scripts.listwise_diffgrm.run.ENGINE', engine), patch.dict(sys.modules, modules):
            first = bindings({}, SimpleNamespace(root=assets), SimpleNamespace(binding={}))
            self.assertIn('scripts/listwise_diffgrm/model.py', first['code'])
            self.assertIn('genrec/model.py', first['engine'])
            self.assertIn('genrec/models/DIFF_GRM/model.py', first['engine'])
            self.assertTrue(all('\\' not in key for key in first['code'] | first['engine']))
            (root / 'README.md').write_text('Documentation-only change')
            self.assertEqual(first, bindings({}, SimpleNamespace(root=assets), SimpleNamespace(binding={})))
            (engine / 'genrec/model.py').write_text('base = 2\n')
            self.assertNotEqual(first, bindings({}, SimpleNamespace(root=assets), SimpleNamespace(binding={})))
            sys.modules['genrec.model'] = SimpleNamespace(__file__=str(root / 'wrong.py'))
            with self.assertRaisesRegex(ValueError, 'outside'):
                bindings({}, SimpleNamespace(root=assets), SimpleNamespace(binding={}))

    def test_formal_evaluation_rejects_preflight_and_non_r_boundaries(self):
        from scripts.listwise_diffgrm.run import validate_evaluation_checkpoint
        plan = [{'kind': 'R'}, {'kind': 'T'}, {'kind': 'R'}]
        for phase in (1, 3):
            validate_evaluation_checkpoint({'progress': {'phase': phase, 'update': 0}}, plan)
        for progress in ({'phase': 0, 'update': 0}, {'phase': 1, 'update': 3},
                         {'phase': 2, 'update': 0}, {'phase': 4, 'update': 0},
                         {'phase': 1, 'update': 0, 'preflight_only': True}):
            with self.assertRaises(ValueError):
                validate_evaluation_checkpoint({'progress': progress}, plan)


if __name__ == '__main__':
    unittest.main()
