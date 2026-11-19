"""Independent source-to-split audit, streamed to bound memory use."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .l5o import sha


def validate(root):
    root = Path(root)
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    source = Path(cfg['source_dataset'])
    h, length, hmin = cfg['history_max'], cfg['list_length'], cfg['history_min']
    mapping = json.loads((source / 'item_map.json').read_text(encoding='utf-8'))['entries']
    lookup = np.zeros(max(r['original_id'] for r in mapping) + 1, dtype=np.int64)
    for r in mapping:
        lookup[r['original_id']] = r['id']
    sequences, source_counts, source_hashes = {}, Counter(), {}
    for file in sorted((source / 'events').glob('*.parquet')):
        source_hashes[file.name] = sha(file)
        frame = pq.read_table(file, columns=['user_id', 'video_id', 'timestamp_us', 'source_event_id',
            'watch_ratio', 'play_duration', 'event_valid', 'ambiguous']).to_pandas()
        source_counts['all'] += len(frame)
        keep = frame.event_valid & ~frame.ambiguous
        source_counts['excluded_invalid_or_ambiguous'] += int((~keep).sum())
        frame = frame[keep].sort_values(['user_id', 'timestamp_us', 'source_event_id'], kind='mergesort')
        for uid, g in frame.groupby('user_id', sort=True):
            if int(uid) in sequences:
                raise AssertionError('User spans source partitions')
            sequences[int(uid)] = dict(item=lookup[g.video_id.to_numpy()], ts=g.timestamp_us.to_numpy(),
                event=g.source_event_id.to_numpy(), wr=g.watch_ratio.to_numpy(), play=g.play_duration.to_numpy())
    seen_items = set(json.loads((root / 'train_seen.json').read_text())['item_ids'])
    seen_nodes, report = defaultdict(set), {}
    for split in ('train', 'validation', 'test'):
        counts = Counter()
        for batch in pq.ParquetFile(root / f'{split}.parquet').iter_batches(batch_size=2048):
            values = batch.to_pydict()
            for i, uid in enumerate(values['original_user_id']):
                seq, k = sequences[uid], values['slate_index'][i]
                n = len(seq['item'])
                assert n >= 2 * length + hmin
                assert k not in seen_nodes[uid], 'duplicate user/slate node'
                seen_nodes[uid].add(k)
                assert split == ('test' if k == 0 else 'validation' if k == 1 else 'train')
                end = n - k * length
                start = end - length
                assert start >= hmin
                assert values['sample_id'][i] == f'u{uid}:{split}:{k}'
                for column, key in [('target_item_ids','item'), ('target_timestamp_us','ts'),
                    ('target_source_event_id','event'), ('target_watch_ratio','wr'), ('target_play_duration','play')]:
                    if not np.array_equal(values[column][i], seq[key][start:end]):
                        raise AssertionError(f'Source mismatch: {column} {uid} {k}')
                for column, key in [('history_item_ids','item'), ('history_timestamp_us','ts')]:
                    if not np.array_equal(values[column][i], seq[key][max(0,start-h):start]):
                        raise AssertionError(f'History mismatch: {column} {uid} {k}')
                assert np.all(np.diff(seq['ts'][max(0,start-h):end]) > 0)
                target, wr = values['target_item_ids'][i], np.asarray(values['target_watch_ratio'][i])
                counts['samples'] += 1
                counts['history_events'] += min(start,h)
                counts['repeated_slates'] += int(len(set(target)) < length)
                counts['cold_target_slates'] += int(any(x not in seen_items for x in target))
                counts['zero_play_events'] += sum(x <= 0 for x in values['target_play_duration'][i])
                for threshold in (1.0, 2.0):
                    counts[f'ev{threshold}_slates'] += int((wr >= threshold).any())
                    counts[f'ev{threshold}_events'] += int((wr >= threshold).sum())
        report[split] = dict(counts)
        print(json.dumps({'audited_split':split, **dict(counts)}), flush=True)
    target_events, prefix_events, excluded_short = 0, 0, 0
    for uid, seq in sequences.items():
        n = len(seq['item'])
        if n < 2 * length + hmin:
            assert not seen_nodes[uid]
            excluded_short += n
        else:
            expected_count = (n - hmin) // length
            assert seen_nodes[uid] == set(range(expected_count)), 'missing/extra target blocks'
            target_events += expected_count * length
            prefix_events += n - expected_count * length
    assert source_counts['all'] == source_counts['excluded_invalid_or_ambiguous'] + excluded_short + target_events + prefix_events
    report.update(source_counts=dict(source_counts), source_event_sha256=source_hashes,
        excluded_short_user_events=excluded_short, target_events=target_events,
        prefix_only_events=prefix_events, all_records_accounted=True,
        exact_source_targets_and_histories=True, users=sum(bool(v) for v in seen_nodes.values()))
    (root / 'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, type=Path)
    validate(p.parse_args().dataset)
