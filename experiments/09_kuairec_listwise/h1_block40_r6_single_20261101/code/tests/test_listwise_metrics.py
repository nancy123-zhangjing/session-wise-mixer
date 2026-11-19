"""Hand-computed multi-positive, collision-safe item recommendation metrics."""
import math
import unittest

import torch

from scripts.listwise_diffgrm.decoding import GenerationOutput
from scripts.listwise_diffgrm.metrics import evaluate_generation, ranking_metrics


class MetricsTests(unittest.TestCase):
    def test_five_relevant_targets_use_idcg_five_and_recall_denominator_five(self):
        predictions = torch.tensor([[3, 99, 1, 1, 0]])
        targets = torch.tensor([[1, 2, 3, 4, 5]])
        got = ranking_metrics(predictions, targets, catalog_item_ids=torch.arange(1, 11))
        ideal = sum(1 / math.log2(r + 1) for r in range(1, 6))
        self.assertAlmostEqual(float(got['recall@5'][0]), 2 / 5, places=6)
        self.assertAlmostEqual(float(got['ndcg@5'][0]), (1 + .5) / ideal, places=6)
        self.assertEqual(float(got['hr@5'][0]), 1)
        self.assertAlmostEqual(float(got['duplicate_rate'][0]), .2, places=6)
        self.assertAlmostEqual(float(got['invalid_rate'][0]), .2, places=6)
        self.assertAlmostEqual(float(got['missing_rate'][0]), .2, places=6)
        self.assertEqual(float(got['completion_rate'][0]), 0)

    def test_relevance_set_order_is_distinct_from_sequence_reconstruction(self):
        got = ranking_metrics(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertEqual(float(got['ndcg@5'][0]), 1)
        self.assertEqual(float(got['recall@5'][0]), 1)
        self.assertEqual(float(got['exact_match'][0]), 0)
        self.assertAlmostEqual(float(got['position_accuracy'][0]), .2, places=6)

    def test_oversized_k_is_not_silently_assigned_to_a_five_item_output(self):
        with self.assertRaisesRegex(ValueError, 'aggregation'):
            ranking_metrics(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 5]]), ks=(10,))

    def test_zero_hits_short_output_and_duplicate_targets(self):
        got = ranking_metrics(torch.tensor([[8, 0, 0, 0, 0]]), torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertEqual(float(got['ndcg@5'][0]), 0)
        self.assertEqual(float(got['hr@5'][0]), 0)
        self.assertEqual(float(got['duplicate_rate'][0]), 0)
        with self.assertRaises(ValueError):
            ranking_metrics(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[1, 1, 3, 4, 5]]))

    def test_catalog_collision_never_credits_another_item(self):
        # Predicted 7 and target 8 could share a SID, but only IDs matter here.
        got = ranking_metrics(torch.tensor([[7, 0, 0, 0, 0]]), torch.tensor([[8, 9, 10, 11, 12]]))
        self.assertEqual(float(got['recall@5'][0]), 0)

    def test_single_list_and_multi_list_aggregation_are_separate(self):
        output = GenerationOutput(
            torch.tensor([[[1, 6, 7, 8, 9], [2, 3, 4, 5, 10]]]),
            torch.zeros((1, 2, 5, 4), dtype=torch.long),
            torch.tensor([[-1.0, -2.0]]), torch.tensor([[True, True]]),
            torch.tensor([[True, True]]), {},
        )
        got = evaluate_generation(output, torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertAlmostEqual(float(got['list/recall@5'][0]), .2, places=6)
        self.assertEqual(float(got['aggregate/recall@10'][0]), 1)
        self.assertEqual(float(got['aggregate/candidate_count'][0]), 10)
        self.assertAlmostEqual(float(got['aggregate/coverage@50'][0]), .2, places=6)
        self.assertLess(float(got['aggregate/ndcg@10'][0]), 1)

    def test_single_target_reduces_to_first_hit_metrics(self):
        got = ranking_metrics(torch.tensor([[4, 2, 2, 0]]), torch.tensor([[2]]), ks=(1, 4))
        self.assertEqual(float(got['recall@1'][0]), 0)
        self.assertEqual(float(got['recall@4'][0]), 1)
        self.assertAlmostEqual(float(got['ndcg@4'][0]), 1 / math.log2(3), places=6)


if __name__ == '__main__':
    unittest.main()
