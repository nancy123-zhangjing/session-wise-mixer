"""Regression checks for idempotent process artifacts and measurement isolation."""
from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_listwise_spread200 import make_case
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm import spread200_run as m


class RunnerTests(unittest.TestCase):
    def test_full_prediction_extraction_is_reused_but_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _, _ = make_case(root)
            source = root / 'source.jsonl.gz'
            with gzip.open(source, 'wt', encoding='utf-8') as f:
                for i in range(3):
                    f.write(json.dumps(dict(sample_id=str(i), metrics={'list/ndcg@5': i/10})) + '\n')
            subset = {'count': 2, 'rows': [{'sample_id':'0'}, {'sample_id':'2'}]}
            state = fingerprint(c.checkpoint())
            with patch.object(m, 'evaluate_recovery', side_effect=AssertionError('Must not infer again')):
                first = m.diagnostic(c, None, None, subset, root, 'R05', source=source)
                second = m.diagnostic(c, None, None, subset, root, 'R05', source=source)
            self.assertEqual(first, second)
            self.assertEqual(first['result']['examples'], 2)
            self.assertEqual(fingerprint(c.checkpoint()), state)
            with source.open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaises(ValueError):
                m.diagnostic(c, None, None, subset, root, 'R05', source=source)

    def test_catalog_resume_reuses_authenticated_snapshot_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _, _ = make_case(root)
            state = fingerprint(c.checkpoint())
            m.catalog(c, root, 'initial')
            (root / 'catalog_previous.pt').unlink()
            with patch.object(m, 'diagnose_catalog', side_effect=AssertionError('Should reuse')):
                m.catalog(c, root, 'initial')
            self.assertTrue((root / 'catalog_previous.pt').is_file())
            self.assertEqual(fingerprint(c.checkpoint()), state)
            with (root / 'catalog/initial.pt').open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaises(ValueError):
                m.catalog(c, root, 'initial')

    def test_process_measurement_that_changes_rng_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _, _ = make_case(root)
            subset = {'rows': [{'index':0, 'sample_id':'0'}]}
            def mutate(*args, **kwargs):
                torch.rand(1)
                path = Path(args[3]); path.mkdir()
                (path / 'per_sample.jsonl.gz').write_bytes(b'toy')
                return {'examples':1, 'metrics':{}}
            with patch.object(m, 'evaluate_recovery', side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, 'changed training state'):
                    m.diagnostic(c, None, None, subset, root, 'after_T')


if __name__ == '__main__':
    unittest.main()
