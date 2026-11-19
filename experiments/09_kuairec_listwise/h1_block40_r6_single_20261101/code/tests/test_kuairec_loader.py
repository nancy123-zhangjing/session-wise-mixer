"""CPU batch-loader checks using small, explicit Parquet fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kuairec_data.loader import iter_batches


class LoaderTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / 'outputs' / 'kuairec_loader_tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='loader-', dir=temp_root)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text(json.dumps({'history_max': 50}), encoding='utf-8')
        (self.root / 'train_seen.json').write_text(json.dumps({'item_ids': [1, 2, 3, 4, 5, 11, 12], 'user_ids': [1]}), encoding='utf-8')
        self.rows = [
            self.row('first', 1, [1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 100),
            self.row('second', 2, [2, 2, 7, 8, 1, 9, 10], [1, 11, 12, 20, 21], 200),
            self.row('third', 1, [i % 7 + 1 for i in range(50)], [21, 22, 23, 24, 25], 300),
        ]
        # Small row groups deliberately differ from the requested batch size.
        pq.write_table(pa.Table.from_pylist(self.rows), self.root / 'validation.parquet', row_group_size=1)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def row(name, user_id, history, target, start):
        return {'sample_id': name, 'user_id': user_id, 'history_item_ids': history,
                'history_mask': [True] * len(history),
                'history_timestamp_us': list(range(start, start + len(history))),
                'target_item_ids': target, 'target_mask': [True] * 5,
                'target_timestamp_us': list(range(start + 100, start + 105))}

    def batches(self, size=2):
        return list(iter_batches(self.root, 'validation', batch_size=size))

    def test_short_history_is_left_padded_with_zero_and_false_masks(self):
        batch = self.batches()[0]
        np.testing.assert_array_equal(batch['history_item_ids'][0, :45], np.zeros(45, dtype=np.int64))
        np.testing.assert_array_equal(batch['history_item_ids'][0, 45:], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(batch['history_timestamp_us'][0, :45], 0)
        np.testing.assert_array_equal(batch['history_timestamp_us'][0, 45:], [100, 101, 102, 103, 104])
        self.assertFalse(batch['history_mask'][0, :45].any())
        self.assertTrue(batch['history_mask'][0, 45:].all())
        self.assertFalse(batch['history_cold_start'][0, :45].any())
        np.testing.assert_array_equal(batch['history_mask'], batch['history_item_ids'] != 0)

    def test_full_history_has_no_padding_and_repeated_items_keep_their_positions(self):
        batch = self.batches()[1]
        self.assertEqual(batch['history_item_ids'].shape, (1, 50))
        np.testing.assert_array_equal(batch['history_item_ids'][0], self.rows[2]['history_item_ids'])
        np.testing.assert_array_equal(batch['history_timestamp_us'][0], self.rows[2]['history_timestamp_us'])
        self.assertTrue(batch['history_mask'].all())
        self.assertLess(len(set(batch['history_item_ids'][0])), 50)

    def test_cold_start_uses_only_the_saved_training_reference(self):
        batch = self.batches()[0]
        np.testing.assert_array_equal(batch['user_cold_start'], [False, True])
        np.testing.assert_array_equal(batch['target_cold_start'][0], [False, False, True, True, True])
        np.testing.assert_array_equal(batch['target_cold_start'][1], [False, False, False, True, True])
        np.testing.assert_array_equal(batch['history_cold_start'][1, -7:], [False, False, True, True, False, True, True])
        # PAD positions do not represent unseen real items.
        self.assertFalse(batch['history_cold_start'][1, :-7].any())

    def test_batch_boundaries_preserve_every_sample_once_in_order(self):
        batches = self.batches(2)
        self.assertEqual([len(b['sample_id']) for b in batches], [2, 1])
        self.assertEqual([item for b in batches for item in b['sample_id']], ['first', 'second', 'third'])
        for size in (1, 2, 8):
            with self.subTest(batch_size=size):
                batches = self.batches(size)
                targets = np.concatenate([b['target_item_ids'] for b in batches])
                np.testing.assert_array_equal(targets, [row['target_item_ids'] for row in self.rows])

    def test_output_types_targets_and_history_times_are_preserved(self):
        for batch in self.batches():
            self.assertEqual(batch['user_id'].dtype, np.dtype('int64'))
            self.assertEqual(batch['history_item_ids'].dtype, np.dtype('int64'))
            self.assertEqual(batch['history_timestamp_us'].dtype, np.dtype('int64'))
            self.assertEqual(batch['history_mask'].dtype, np.dtype('bool'))
            self.assertEqual(batch['target_item_ids'].shape[1], 5)
            self.assertTrue(batch['target_mask'].all())
            self.assertTrue((batch['history_timestamp_us'].max(axis=1) < batch['target_timestamp_us'][:, 0]).all())
            self.assertTrue((np.diff(batch['target_timestamp_us'], axis=1) > 0).all())

    def test_requested_split_reads_its_own_file(self):
        train_row = self.row('train-only', 1, [1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 1)
        pq.write_table(pa.Table.from_pylist([train_row]), self.root / 'train.parquet')
        self.assertEqual(next(iter_batches(self.root, 'train'))['sample_id'], ['train-only'])
        self.assertEqual(self.batches()[0]['sample_id'], ['first', 'second'])


if __name__ == '__main__':
    unittest.main()
