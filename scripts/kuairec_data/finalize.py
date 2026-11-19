"""Bind validated outputs, cold-start catalogs, versions and reproducibility hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .prepare import dump, sha, sample_schema
from .loader import iter_batches


def canonical_array(array):
    """Remove reader-only validity bitmaps without expanding scalar objects.

    This normalizer supports the fixed primitive/string/list schema used here.
    All-null columns are rebuilt to avoid unspecified bytes in null slots.
    """
    if array.null_count == len(array):
        return pa.nulls(len(array), type=array.type)
    is_list = pa.types.is_list(array.type)
    buffers = array.buffers()[:2] if is_list else array.buffers()
    if array.null_count == 0:
        buffers[0] = None
    children = {'children': [canonical_array(array.values)]} if is_list else {}
    return pa.Array.from_buffers(array.type, len(array), buffers,
                                 null_count=array.null_count, offset=array.offset, **children)


def normalized_hash(path):
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        # Reconstruct canonical buffers; Parquet readers may materialize an
        # all-valid bitmap where the writer originally had no bitmap.
        table = pa.Table.from_arrays([canonical_array(a) for a in batch.columns], schema=sample_schema())
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, sample_schema()) as stream:
            stream.write_table(table)
        digest.update(sink.getvalue())
    return digest.hexdigest()


def finalize(dataset):
    validation = json.loads((dataset / 'validation.json').read_text(encoding='utf-8'))
    if validation.get('passed') is not True:
        raise ValueError('Full validation must pass before binding the delivery')
    if (dataset / 'trace_validation.json').exists():
        if json.loads((dataset / 'trace_validation.json').read_text(encoding='utf-8')).get('passed') is not True:
            raise ValueError('Existing original-source trace failed')
    config = json.loads((dataset / 'config.json').read_text(encoding='utf-8'))
    seen = json.loads((dataset / 'train_seen.json').read_text(encoding='utf-8'))
    seen_items, seen_users = set(seen['item_ids']), set(seen['user_ids'])
    texts = {r['video_id']: r for r in map(json.loads, (dataset / 'item_text.jsonl').read_text(encoding='utf-8').splitlines())}
    item_rows = []
    for entry in json.loads((dataset / 'item_map.json').read_text(encoding='utf-8'))['entries']:
        original = entry['original_id']
        item_rows.append({'item_id': entry['id'], 'original_item_id': original,
                          'text': texts.get(original, {}).get('text', ''),
                          'text_missing': texts.get(original, {}).get('text_missing', True),
                          'metadata_missing': original not in texts,
                          'train_seen': entry['id'] in seen_items})
    user_rows = [{'user_id': e['id'], 'original_user_id': e['original_id'], 'train_seen': e['id'] in seen_users}
                 for e in json.loads((dataset / 'user_map.json').read_text(encoding='utf-8'))['entries']]
    pq.write_table(pa.Table.from_pylist(item_rows), dataset / 'item_catalog.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(user_rows), dataset / 'user_catalog.parquet', compression='zstd')
    loader_checks = {}
    for split in ('train', 'validation', 'test'):
        batch = next(iter_batches(dataset, split, batch_size=64))
        assert batch['target_item_ids'].shape == (len(batch['sample_id']), 5)
        assert np.array_equal(batch['history_mask'], batch['history_item_ids'] != 0)
        assert (batch['history_timestamp_us'].max(axis=1) < batch['target_timestamp_us'][:, 0]).all()
        loader_checks[split] = {'batch_rows': len(batch['sample_id']), 'passed': True}
    dump(dataset / 'loader_validation.json', loader_checks)
    construction = json.loads((dataset / 'construction.json').read_text(encoding='utf-8'))
    # Rebinding a report need not deserialize unchanged hundreds of millions of
    # scalar values again. Reuse only hashes already verified against these
    # exact Parquet bytes, configuration and Arrow version.
    prior = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8')) if (dataset / 'manifest.json').exists() else {}
    prior_files = {entry['path']: entry['sha256'] for entry in prior.get('artifacts', [])}
    hashes = {}
    for split in ('train', 'validation', 'test'):
        name = f'{split}.parquet'
        if (prior.get('config_sha256') == sha(dataset / 'config.json')
                and prior.get('versions', {}).get('pyarrow') == pa.__version__
                and prior_files.get(name) == sha(dataset / name)):
            hashes[split] = prior['normalized_sample_hashes'][split]
        else:
            hashes[split] = normalized_hash(dataset / name)
    if hashes != construction['normalized_sample_hashes']:
        raise AssertionError('Readback normalized data hashes differ from construction')
    for raw, expected in [(config['input_archive'], config['archive_sha256']), (config['input_text'], config['text_sha256'])]:
        if sha(Path(raw)) != expected:
            raise AssertionError('Raw source changed during preprocessing')
    files = []
    for path in sorted(dataset.rglob('*')):
        if path.is_file() and path.name != 'manifest.json' and 'staging' not in path.parts and path.name != 'clean_times.bin':
            files.append({'path': path.relative_to(dataset).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {'schema_version': 1, 'config_sha256': sha(dataset / 'config.json'),
                'normalized_sample_hashes': hashes, 'raw_input_sha256': {'archive': config['archive_sha256'], 'text': config['text_sha256']},
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pa.__version__},
                'code_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'code_file_hashes': {p.as_posix(): sha(p) for p in sorted(Path('scripts/kuairec_data').glob('*.py'))},
                'artifacts': files}
    dump(dataset / 'manifest.json', manifest)
    return manifest


def compare(left, right):
    a = json.loads((left / 'manifest.json').read_text(encoding='utf-8'))
    b = json.loads((right / 'manifest.json').read_text(encoding='utf-8'))
    assert a['config_sha256'] == b['config_sha256']
    assert a['normalized_sample_hashes'] == b['normalized_sample_hashes']
    for filename in ('user_map.json', 'item_map.json', 'train_seen.json', 'item_text.jsonl', 'text_repairs.jsonl', 'ingestion.json', 'construction.json', 'validation.json', 'split_statistics.json'):
        assert sha(left / filename) == sha(right / filename), filename
    result = {'passed': True, 'method': 'two full raw-ingestion and construction runs; same config and dependencies',
              'left': str(left), 'right': str(right), 'config_sha256': a['config_sha256'],
              'normalized_sample_hashes': a['normalized_sample_hashes'], 'mapping_split_text_equal': True}
    dump(left / 'reproducibility.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    finalize(args.dataset)
    if args.compare:
        print(json.dumps(compare(args.dataset, args.compare)))
        finalize(args.dataset)  # Include comparison receipt in the manifest.
    else:
        print(json.dumps({'dataset': str(args.dataset), 'manifest_written': True}))


if __name__ == '__main__':
    main()
