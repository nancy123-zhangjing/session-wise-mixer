"""CPU NumPy batches; left padding is explicit and PAD=0 is never an item."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def iter_batches(dataset, split='train', batch_size=64):
    dataset = Path(dataset)
    config = json.loads((dataset / 'config.json').read_text(encoding='utf-8'))
    seen = json.loads((dataset / 'train_seen.json').read_text(encoding='utf-8'))
    item_seen = set(seen['item_ids'])
    user_seen = set(seen['user_ids'])
    width = config['history_max']
    for record_batch in pq.ParquetFile(dataset / f'{split}.parquet').iter_batches(batch_size=batch_size):
        rows = record_batch.to_pylist()
        size = len(rows)
        result = {'sample_id': [r['sample_id'] for r in rows],
                  'user_id': np.array([r['user_id'] for r in rows], dtype=np.int64),
                  'history_item_ids': np.zeros((size, width), dtype=np.int64),
                  'history_mask': np.zeros((size, width), dtype=bool),
                  'history_timestamp_us': np.zeros((size, width), dtype=np.int64),
                  'target_item_ids': np.array([r['target_item_ids'] for r in rows], dtype=np.int64),
                  'target_mask': np.array([r['target_mask'] for r in rows], dtype=bool),
                  'target_timestamp_us': np.array([r['target_timestamp_us'] for r in rows], dtype=np.int64),
                  'user_cold_start': np.array([r['user_id'] not in user_seen for r in rows], dtype=bool),
                  'target_cold_start': np.array([[i not in item_seen for i in r['target_item_ids']] for r in rows], dtype=bool),
                  'history_cold_start': np.zeros((size, width), dtype=bool)}
        for i, row in enumerate(rows):
            n = len(row['history_item_ids'])
            result['history_item_ids'][i, -n:] = row['history_item_ids']
            result['history_timestamp_us'][i, -n:] = row['history_timestamp_us']
            result['history_mask'][i, -n:] = True
            result['history_cold_start'][i, -n:] = [v not in item_seen for v in row['history_item_ids']]
        yield result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    batch = next(iter_batches(args.dataset, args.split, args.batch_size))
    assert batch['target_item_ids'].shape[1] == 5
    assert np.array_equal(batch['history_item_ids'] != 0, batch['history_mask'])
    assert (batch['history_timestamp_us'].max(axis=1) < batch['target_timestamp_us'][:, 0]).all()
    print(json.dumps({'split': args.split, 'batch_size': len(batch['sample_id']),
                      'history_shape': list(batch['history_item_ids'].shape),
                      'target_shape': list(batch['target_item_ids'].shape),
                      'real_history_lengths': batch['history_mask'].sum(axis=1).tolist(),
                      'first_sample_id': batch['sample_id'][0], 'passed': True}))


if __name__ == '__main__':
    main()
