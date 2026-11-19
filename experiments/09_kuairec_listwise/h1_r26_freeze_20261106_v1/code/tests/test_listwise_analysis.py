"""Independent metric replay, source identity, and clustered sampling checks."""
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.listwise_diffgrm.analysis import audit_predictions, cluster_bootstrap, paired_comparison, recompute_row
from scripts.listwise_diffgrm.decoding import GenerationOutput
from scripts.listwise_diffgrm.metrics import evaluate_generation
from scripts.listwise_diffgrm.subset import file_sha256


def prediction(sample='a', user=1, hit=False):
    return dict(sample_id=sample, user_id=user, dataset_index=0, cold_target=False,
                targets=[1, 2, 3, 4, 5], predictions=[[1, 2, 3, 4, 5] if hit else [6, 7, 8, 9, 10]],
                scores=[-1.0], beam_mask=[True], complete_mask=[True])


def production_metrics(row):
    width = len(row['predictions'])
    output = GenerationOutput(torch.tensor([row['predictions']]), torch.zeros(1, width, 5, 4),
        torch.tensor([[s if s is not None else -math.inf for s in row['scores']]]),
        torch.tensor([row['beam_mask']]), torch.tensor([row['complete_mask']]), {})
    values = evaluate_generation(output, torch.tensor([row['targets']]), catalog_item_ids=torch.arange(1, 21))
    return {key: float(value[0]) for key, value in values.items()}


def write_evaluation(root, rows):
    root.mkdir()
    sums, users, groups = {}, {}, {}
    for index, row in enumerate(rows):
        row.setdefault('dataset_index', index)
        row['metrics'] = production_metrics(row)
        user = users.setdefault(row['user_id'], [])
        user.append(row['metrics'])
        group = groups.setdefault('cold_target' if row['cold_target'] else 'seen_targets', [])
        group.append(row['metrics'])
        for key, value in row['metrics'].items():
            sums[key] = sums.get(key, 0) + value
    mean = {key: value / len(rows) for key, value in sums.items()}
    user_means = {key: sum(sum(r[key] for r in values) / len(values) for values in users.values()) / len(users) for key in sums}
    summary = dict(examples=len(rows), split='validation', full_split=True, subset=None, diagnostic_prefix_limit=None,
        beam=32, architecture='J', seed=2024, catalog_size=20, dataset_binding={}, metrics=mean, user_average=user_means,
        groups={group: dict(examples=len(values), metrics={key: sum(r[key] for r in values) / len(values) for key in sums})
                for group, values in groups.items()},
        metric_protocol=dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                             aggregation='list_score_order_then_item_position_first_occurrence'))
    (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    path = root / 'per_sample.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return path


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='analysis_test_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_independent_replay_matches_production_on_partial_and_aggregate_lists(self):
        row = prediction()
        row.update(predictions=[[3, 99, 1, 1, 0], [2, 3, 4, 5, 10], [0, 0, 0, 0, 0]],
                   scores=[-.1, -2.0, None], beam_mask=[True, True, False], complete_mask=[False, True, False])
        actual = recompute_row(row, catalog_size=20)
        expected = production_metrics(row)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=6, msg=key)
        ideal = sum(1 / math.log2(i + 2) for i in range(5))
        self.assertAlmostEqual(actual['list/ndcg@5'], 1.5 / ideal)
        self.assertEqual(actual['aggregate/candidate_count'], 5)
        self.assertEqual(actual['aggregate/recall@50'], .8)

    def test_corrupted_completion_flags_and_sid_collision_credit_are_rejected(self):
        row = prediction()
        row['predictions'] = [[7, 7, 7, 7, 7]]
        with self.assertRaisesRegex(ValueError, 'completion'):
            recompute_row(row, catalog_size=20)
        row['complete_mask'] = [False]
        self.assertEqual(recompute_row(row, catalog_size=20)['list/recall@5'], 0)

    def test_replay_detects_saved_metric_tampering(self):
        path = write_evaluation(self.root / 'a', [prediction(hit=True), prediction('b', 2)])
        audit = audit_predictions(path)
        self.assertTrue(audit['passed'])
        self.assertLess(audit['max_absolute_error'], 2e-6)
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        rows[0]['metrics']['list/ndcg@5'] = .123
        with gzip.open(path, 'wt') as stream:
            for row in rows: stream.write(json.dumps(row) + '\n')
        audit = audit_predictions(path)
        self.assertFalse(audit['passed'])
        self.assertEqual(audit['mismatches'], 1)

    def test_replay_verifies_source_user_labels_index_and_full_coverage(self):
        rows = [prediction(hit=True), prediction('b', 2)]
        rows[1]['dataset_index'] = 1
        path = write_evaluation(self.root / 'a', rows)
        source = self.root / 'source'
        source.mkdir()
        pq.write_table(pa.Table.from_pylist([dict(sample_id=r['sample_id'], user_id=r['user_id'], target_item_ids=r['targets']) for r in rows]),
                       source / 'validation.parquet')
        summary_path = path.parent / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['dataset_binding'] = {'files': {'validation.parquet': {'sha256': file_sha256(source / 'validation.parquet')}}}
        summary_path.write_text(json.dumps(summary))
        self.assertTrue(audit_predictions(path, dataset_dir=source)['source_labels_verified'])
        rows[1]['dataset_index'] = 0
        with gzip.open(path, 'wt') as stream:
            for row in rows: stream.write(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'source data'):
            audit_predictions(path, dataset_dir=source)

    def test_pairing_uses_sample_ids_and_separates_sample_and_user_means(self):
        a_rows = [prediction(str(i), 1 if i < 3 else 2) for i in range(4)]
        b_rows = [prediction(str(i), 1 if i < 3 else 2, hit=i == 3) for i in range(4)][::-1]
        a = write_evaluation(self.root / 'a', a_rows)
        b = write_evaluation(self.root / 'b', b_rows)
        result = paired_comparison(a, b, repeats=200, seed=2024)
        self.assertEqual(result['examples'], 4)
        self.assertEqual(result['users'], 2)
        self.assertAlmostEqual(result['primary']['list/ndcg@5']['delta'], .25)
        self.assertAlmostEqual(result['user_average']['list/ndcg@5']['delta'], .5)
        self.assertIsNone(result['primary']['list/ndcg@5']['relative_delta'])
        self.assertIn('does not establish cross-seed', result['interpretation'])
        self.assertEqual(result, paired_comparison(a, b, repeats=200, seed=2024))
        b_rows[-1]['sample_id'] = 'unmatched'
        with gzip.open(b, 'wt') as stream:
            for row in b_rows: stream.write(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            paired_comparison(a, b, repeats=200)

    def test_pairing_rejects_different_beam_or_training_seed(self):
        a = write_evaluation(self.root / 'a', [prediction()])
        b = write_evaluation(self.root / 'b', [prediction(hit=True)])
        p = b.parent / 'summary.json'
        summary = json.loads(p.read_text())
        summary['beam'] = 256
        p.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'beam'):
            paired_comparison(a, b, repeats=10)
        summary['beam'], summary['seed'] = 32, 2025
        p.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'seed'):
            paired_comparison(a, b, repeats=10)


class BootstrapTests(unittest.TestCase):
    def test_user_resampling_retains_all_samples_and_random_denominator(self):
        generator = unittest.mock.Mock()
        generator.integers.return_value = np.array([[0, 0], [0, 1], [1, 1]])
        with patch('numpy.random.default_rng', return_value=generator):
            result = cluster_bootstrap([3, 1], [[0], [0]], [[0], [1]], repeats=3, confidence=.5)
        self.assertAlmostEqual(result['sample_delta'][0], .25)
        self.assertAlmostEqual(result['user_delta'][0], .5)
        # Draws: sample means [0, 1/4, 1], user means [0, 1/2, 1].
        np.testing.assert_allclose(result['sample_ci'][0], [.125, .625])
        np.testing.assert_allclose(result['user_ci'][0], [.25, .75])

    def test_identical_arms_have_zero_interval(self):
        result = cluster_bootstrap([2, 1, 5], [[1, 0], [0, 1], [3, 2]], [[1, 0], [0, 1], [3, 2]], repeats=100)
        np.testing.assert_array_equal(result['sample_ci'], np.zeros((2, 2)))
        np.testing.assert_array_equal(result['user_ci'], np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
