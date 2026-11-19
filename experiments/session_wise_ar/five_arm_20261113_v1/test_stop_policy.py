"""No training: reject premature stops, wrong best selection and changed budgets."""
import copy
import unittest
from audit_ar_nodes import audit_stop_policy


class StopPolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {'family': 'diger', 'split_counts': {'train': 709443},
                    'list_length': 5, 'training': {'batch_size': 256,
                    'max_steps': None, 'max_epochs': 120, 'patience': 15}}
        self.policy = {'max_execution_epochs': 35, 'scheduler_epochs': 120, 'patience': 15}

    def progress(self, scores):
        records = [{'step': (i+1)*13857, 'validation': {'metrics': {'ndcg@5': s}}}
                   for i, s in enumerate(scores)]
        best_i = max(range(len(scores)), key=scores.__getitem__)
        return {'records': records, 'step': records[-1]['step'], 'done': True,
                'best_step': records[best_i]['step'], 'best_score': scores[best_i],
                'bad_epochs': len(scores)-best_i-1}

    def test_cap_keeps_scheduler_horizon(self):
        result = audit_stop_policy(self.cfg, self.progress(list(range(35))), self.policy)
        self.assertEqual(result['execution_limit'], 484995)
        self.assertEqual(result['scheduler_epochs'], 120)
        self.assertTrue(result['budget_reached'])

    def test_original_early_stop_remains_valid(self):
        result = audit_stop_policy(self.cfg, self.progress([1]+[0]*15), self.policy)
        self.assertTrue(result['patience_reached'])
        self.assertFalse(result['budget_reached'])

    def test_reject_invalid_completions(self):
        cases = [self.progress([1, 2]), self.progress(list(range(36))),
                 self.progress([1]+[0]*16)]
        wrong_best = self.progress(list(range(35)))
        wrong_best['best_step'] = 13857
        cases.append(wrong_best)
        for p in cases:
            with self.subTest(step=p['step']), self.assertRaises(ValueError):
                audit_stop_policy(self.cfg, p, self.policy)

    def test_reject_changed_scheduler(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['training']['max_epochs'] = 35
        with self.assertRaises(ValueError):
            audit_stop_policy(cfg, self.progress(list(range(35))), self.policy)


if __name__ == '__main__':
    unittest.main()
