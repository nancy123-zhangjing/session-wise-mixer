import math
import unittest

import numpy as np

from scripts.kuairec_data.l5o import user_slates
from scripts.kuairec_data.l5o_metrics import slate_metrics


class SlateMetricsTest(unittest.TestCase):
    def test_empty_ev_and_repeated_targets(self):
        m = slate_metrics([[1,2,3,4,5]], [[1,1,1,2,2]], [[0]*5])
        self.assertIsNone(m['ev_ndcg@5'])
        self.assertEqual(m['imp_ndcg@5'], 1)
        self.assertLess(m['legacy_fixed5_ndcg@5'], 1)

    def test_hand_example(self):
        target = [[1, 2, 3, 4, 5]]
        wr = [[0.2, 1.5, 0.9, 2.5, 0.0]]          # EV(>=1): {2, 4}
        pred = [[2, 9, 4, 1, 8]]
        m = slate_metrics(pred, target, wr, 1.0)
        d = [1 / math.log2(k + 2) for k in range(5)]
        self.assertAlmostEqual(m['imp_hit@5'], 1.0)
        self.assertAlmostEqual(m['imp_recall@5'], 3 / 5)
        self.assertAlmostEqual(m['imp_ndcg@5'], (d[0] + d[2] + d[3]) / sum(d))
        self.assertAlmostEqual(m['ev_recall@5'], 1.0)
        self.assertAlmostEqual(m['ev_ndcg@5'], (d[0] + d[2]) / (d[0] + d[1]))
        m2 = slate_metrics(pred, target, wr, 2.0)    # EV(>=2): {4}
        self.assertAlmostEqual(m2['ev_ndcg@5'], d[2] / d[0])

    def test_no_ev_samples_excluded_and_duplicates(self):
        target = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        wr = [[0.1] * 5, [1.0, 0, 0, 0, 0]]
        pred = [[1, 1, 1, 1, 1], [6, 6, 0, 0, 0]]
        m = slate_metrics(pred, target, wr, 1.0)
        self.assertEqual(m['ev_samples'], 1)
        self.assertAlmostEqual(m['imp_recall@5'], (1 / 5 + 1 / 5) / 2)  # duplicate preds count once
        self.assertAlmostEqual(m['ev_ndcg@5'], 1.0)


if __name__ == '__main__':
    unittest.main()
