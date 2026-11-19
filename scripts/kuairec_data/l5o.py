"""Build the per-user leave-five-out KuaiRec protocol (kuairec_l5o_v1).

Run from repository root:
    python -m scripts.kuairec_data.l5o --output outputs/kuairec_l5o_v1/run1

Input is the verified cleaned event store of kuairec_listwise_v1 (read-only).
The output keeps the column names consumed by the existing loaders
(listwise_diffgrm, rqcd PackedSplit) so every model reads it unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ('train', 'validation', 'test')
SCHEMA = pa.schema([
    ('sample_id', pa.string()), ('user_id', pa.int64()), ('original_user_id', pa.int64()),
    ('split', pa.string()), ('slate_index', pa.int64()),
    ('history_item_ids', pa.list_(pa.int64())), ('history_timestamp_us', pa.list_(pa.int64())),
    ('target_item_ids', pa.list_(pa.int64())), ('target_original_item_ids', pa.list_(pa.int64())),
    ('target_timestamp_us', pa.list_(pa.int64())), ('target_source_event_id', pa.list_(pa.int64())),
    ('target_play_duration', pa.list_(pa.int64())), ('target_watch_ratio', pa.list_(pa.float64())),
    ('target_mask', pa.list_(pa.bool_())),
])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def user_slates(items, length, history_min):
    """Consecutive records, newest first; repetitions never move split boundaries.

    Last five are test, previous five validation; earlier stride-five blocks
    are training. Users with 15..19 events have an observed training prefix but
    no supervised training block (history_min=5); they remain in evaluation.
    """
    if length < 1 or history_min < 1:
        raise ValueError('Positive slate length and history minimum required')
    return [(list(range(end - length, end)), end - length)
            for end in range(len(items), history_min + length - 1, -length)]


def list_array(values, offsets, kind):
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(values, kind))


def build(config, output):
    src = Path(config['source_dataset'])
    length, hmax, hmin = config['list_length'], config['history_max'], config['history_min']
    user_map = {e['original_id']: e['id'] for e in json.loads((src / 'user_map.json').read_text(encoding='utf-8'))['entries']}
    item_map = {e['original_id']: e['id'] for e in json.loads((src / 'item_map.json').read_text(encoding='utf-8'))['entries']}
    item_lookup = np.zeros(max(item_map) + 1, dtype=np.int64)
    for k, v in item_map.items():
        item_lookup[k] = v
    writers = {s: pq.ParquetWriter(output / f'{s}.parquet', SCHEMA, compression='zstd') for s in SPLITS}
    digests = {s: hashlib.sha256() for s in SPLITS}
    stats = Counter()
    users_missing = []
    train_items, train_users = set(), set()
    visited_users = set()
    for path in sorted((src / 'events').glob('*.parquet')):
        frame = pq.read_table(path, columns=['source_event_id', 'user_id', 'video_id', 'timestamp_us', 'play_duration',
                                             'watch_ratio', 'event_valid', 'ambiguous']).to_pandas()
        stats['events_in'] += len(frame)
        usable = frame.event_valid & ~frame.ambiguous
        stats['events_dropped_invalid_or_ambiguous'] += int((~usable).sum())
        frame = frame.loc[usable].sort_values(['user_id', 'timestamp_us', 'source_event_id'], kind='mergesort')
        cols = {s: [] for s in SPLITS}
        for uid, g in frame.groupby('user_id', sort=True):
            if int(uid) in visited_users:
                raise ValueError('Source partitions split a user across files')
            visited_users.add(int(uid))
            orig = g.video_id.to_numpy()
            items = item_lookup[orig]
            ts = g.timestamp_us.to_numpy()
            if np.any(np.diff(ts) <= 0):
                raise ValueError(f'Non-increasing timestamps for user {uid}')
            slates = user_slates(items.tolist(), length, hmin)
            if len(slates) < 2:
                users_missing.append(int(uid))
                stats['events_excluded_short_users'] += len(g)
                continue
            stats['users_retained'] += 1
            stats['users_without_train_sample'] += int(len(slates) == 2)
            # Prefix records are observed training data even for users without
            # a complete supervised block; held-out targets are never included.
            train_users.add(user_map[int(uid)])
            train_items.update(items[:-2 * length].tolist())
            stats['events_prefix_not_targets'] += slates[-1][1]
            for k, (kept, start) in enumerate(slates):
                split = 'test' if k == 0 else 'validation' if k == 1 else 'train'
                end = kept[-1] + 1
                stats[f'target_events_{split}'] += length
                stats[f'repeated_slates_{split}'] += int(len(set(items[kept])) < length)
                h0 = max(0, start - hmax)
                cols[split].append((uid, k, h0, start, kept, g))
                if split == 'train':
                    train_users.add(user_map[int(uid)])
                    train_items.update(items[h0:start].tolist())
                    train_items.update(items[kept].tolist())
        for split in SPLITS:
            rows = cols[split]
            if not rows:
                continue
            batch = {name: [] for name in SCHEMA.names}
            for uid, k, h0, start, kept, g in rows:
                orig = g.video_id.to_numpy(); ts = g.timestamp_us.to_numpy()
                batch['sample_id'].append(f'u{int(uid)}:{split}:{k}')
                batch['user_id'].append(user_map[int(uid)]); batch['original_user_id'].append(int(uid))
                batch['split'].append(split); batch['slate_index'].append(k)
                batch['history_item_ids'].append(item_lookup[orig[h0:start]].tolist())
                batch['history_timestamp_us'].append(ts[h0:start].tolist())
                batch['target_item_ids'].append(item_lookup[orig[kept]].tolist())
                batch['target_original_item_ids'].append(orig[kept].tolist())
                batch['target_timestamp_us'].append(ts[kept].tolist())
                batch['target_source_event_id'].append(g.source_event_id.to_numpy()[kept].tolist())
                batch['target_play_duration'].append(g.play_duration.to_numpy()[kept].tolist())
                batch['target_watch_ratio'].append(g.watch_ratio.to_numpy()[kept].tolist())
                batch['target_mask'].append([True] * length)
            table = pa.Table.from_pydict(batch, schema=SCHEMA)
            writers[split].write_table(table)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, SCHEMA) as w:
                w.write_table(table)
            digests[split].update(sink.getvalue())
            stats[f'samples_{split}'] += len(rows)
        print(json.dumps({'bucket': path.stem, **{f'samples_{s}': stats[f'samples_{s}'] for s in SPLITS}}), flush=True)
    for w in writers.values():
        w.close()
    accounted = (stats['events_dropped_invalid_or_ambiguous'] + stats['events_excluded_short_users']
                 + stats['events_prefix_not_targets'] + sum(stats[f'target_events_{s}'] for s in SPLITS))
    if accounted != stats['events_in']:
        raise AssertionError('Event conservation failed')
    stats['events_accounted'] = accounted
    return stats, users_missing, train_items, train_users, {s: d.hexdigest() for s, d in digests.items()}


def finalize(config, output, stats, users_missing, train_items, train_users, hashes):
    src = Path(config['source_dataset'])
    for name in ('item_map.json', 'user_map.json', 'item_text.jsonl', 'text_repairs.jsonl'):
        shutil.copyfile(src / name, output / name)
    cat = pq.read_table(src / 'item_catalog.parquet').to_pandas()
    cat['train_seen'] = cat.item_id.isin(train_items)
    pq.write_table(pa.Table.from_pandas(cat, preserve_index=False), output / 'item_catalog.parquet', compression='zstd')
    ucat = pq.read_table(src / 'user_catalog.parquet').to_pandas()
    ucat['train_seen'] = ucat.user_id.isin(train_users)
    pq.write_table(pa.Table.from_pandas(ucat, preserve_index=False), output / 'user_catalog.parquet', compression='zstd')
    dump(output / 'train_seen.json', {'reference': 'all retained user prefixes before validation/test (l5o)',
                                      'item_ids': sorted(int(i) for i in train_items),
                                      'user_ids': sorted(int(u) for u in train_users)})
    dump(output / 'construction.json', {'counts': dict(stats), 'users_without_test_and_validation': users_missing,
                                        'normalized_sample_hashes': hashes,
                                        'source_manifest_sha256': sha(src / 'manifest.json')})
    artifacts = []
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name != 'manifest.json':
            artifacts.append({'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    code = {f: sha(Path(f)) for f in ('scripts/kuairec_data/l5o.py',)}
    dump(output / 'manifest.json', {'schema_version': 1, 'artifacts': artifacts, 'code_file_hashes': code,
                                    'config_sha256': sha(output / 'config.json'), 'normalized_sample_hashes': hashes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/kuairec_l5o_v1.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=False)
    dump(a.output / 'config.json', config)
    result = build(config, a.output)
    finalize(config, a.output, *result)
    print(json.dumps({'complete': str(a.output), **{k: v for k, v in result[0].items()}}), flush=True)


if __name__ == '__main__':
    main()
