"""Prepare all KuaiRec big_matrix events using bounded-memory disk buckets.

Run from repository root: python -m scripts.kuairec_data.prepare --output ...
Raw files are opened read-only. Existing final output is never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .protocol import user_samples
from .text import repair_text

FIELDS = ['user_id', 'video_id', 'play_duration', 'video_duration', 'time', 'date', 'timestamp', 'watch_ratio']
EVENT_SCHEMA = pa.schema([
    ('source_event_id', pa.int64()), ('user_id', pa.int64()), ('video_id', pa.int64()),
    ('timestamp_us', pa.int64()), ('play_duration', pa.int64()), ('video_duration', pa.int64()),
    ('watch_ratio', pa.float64()), ('event_valid', pa.bool_()),
    ('timestamp_raw', pa.string()), ('watch_ratio_raw', pa.string()),
    ('time_raw', pa.string()), ('date_raw', pa.string()), ('date_mismatch', pa.bool_()),
])


def sample_schema():
    fields = [('sample_id', pa.string()), ('user_id', pa.int64()), ('original_user_id', pa.int64()),
              ('session_id', pa.string()), ('session_start_us', pa.int64()), ('session_end_us', pa.int64()),
              ('block_offset', pa.int64()), ('split', pa.string()), ('target_span_us', pa.int64()),
              ('target_total_play_ms', pa.int64()), ('train_optional_ge60s', pa.bool_())]
    for p in ('history', 'target'):
        for key in ('item_ids', 'original_item_ids', 'source_event_id', 'timestamp_us', 'play_duration', 'video_duration'):
            fields.append((p + '_' + key, pa.list_(pa.int64())))
        fields += [(p + '_watch_ratio', pa.list_(pa.float64())), (p + '_mask', pa.list_(pa.bool_()))]
    return pa.schema(fields)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def log(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)


def parse_chunk(raw, offset):
    """Exact decimal timestamp parsing, never float seconds -> integer time."""
    timestamp = raw.timestamp.str.extract(r'^(\d+)(?:\.(\d{1,6}))?$')
    valid_time = timestamp[0].notna()
    ts = pd.to_numeric(timestamp[0], errors='coerce').fillna(0).astype('int64') * 1_000_000
    ts += pd.to_numeric(timestamp[1].fillna('').str.pad(6, side='right', fillchar='0'), errors='coerce').fillna(0).astype('int64')
    data = pd.DataFrame({'source_event_id': np.arange(offset + 1, offset + len(raw) + 1), 'timestamp_us': ts})
    valid = valid_time.copy()
    for col in ('user_id', 'video_id', 'play_duration', 'video_duration'):
        good = raw[col].str.fullmatch(r'\d+').fillna(False)
        data[col] = pd.to_numeric(raw[col].where(good, '0')).astype('int64')
        valid &= good
    if not raw.user_id.str.fullmatch(r'\d+').all():
        raise ValueError('Invalid user ID cannot be assigned a safe sequence; inspect quarantine before changing protocol')
    data['watch_ratio'] = pd.to_numeric(raw.watch_ratio, errors='coerce')
    valid &= np.isfinite(data.watch_ratio) & (data.watch_ratio >= 0) & (data.video_duration > 0)
    # Local time/date are redundant raw fields. Verify rather than dedup away conflicts.
    local = pd.to_datetime(raw.time, errors='coerce', format='mixed')
    expected = pd.to_datetime(ts, unit='us') + pd.Timedelta(hours=8)
    clock_ok = local.eq(expected)
    # Some released date labels disagree with BOTH matching precise clocks.
    # Preserve/report them; sequence order follows the two agreeing time fields.
    data['date_mismatch'] = ~raw.date.eq(expected.dt.strftime('%Y%m%d'))
    valid &= clock_ok
    data['event_valid'] = valid
    for src, dst in [('timestamp', 'timestamp_raw'), ('watch_ratio', 'watch_ratio_raw'), ('time', 'time_raw'), ('date', 'date_raw')]:
        data[dst] = raw[src]
    return data, valid_time, clock_ok


def ingest(config, output):
    archive, text = Path(config['input_archive']), Path(config['input_text'])
    for path, expected in [(archive, config['archive_sha256']), (text, config['text_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Raw source hash mismatch: {path}')
    staging, clean = output / 'staging', output / 'events'
    staging.mkdir(parents=True, exist_ok=False)
    clean.mkdir()
    writers = {}
    stats, invalid_users, all_users, all_items = Counter(), set(), set(), set()
    duplicate_writer = pq.ParquetWriter(output / 'duplicates.parquet', pa.schema([('removed_event_id', pa.int64()), ('canonical_event_id', pa.int64()), ('removed_date_label', pa.string()), ('canonical_date_label', pa.string())]), compression='zstd')
    with (output / 'quarantine.jsonl').open('w', encoding='utf-8') as quarantine:
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.endswith('/big_matrix.csv')]
            if len(names) != 1:
                raise ValueError('Expected exactly one big_matrix.csv')
            with z.open(names[0]) as source:
                for raw in pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=config['chunk_rows']):
                    if list(raw.columns) != FIELDS:
                        raise ValueError('Unexpected input schema')
                    data, valid_time, clock_ok = parse_chunk(raw, stats['raw_events'])
                    stats['raw_events'] += len(raw)
                    stats['clock_mismatch_rows'] += int((~clock_ok).sum())
                    stats['date_label_mismatch_rows'] += int(data.date_mismatch.sum())
                    stats['invalid_timestamp_rows'] += int((~valid_time).sum())
                    stats['invalid_field_rows'] += int((~data.event_valid).sum())
                    stats['watch_ratio_gt1_raw'] += int((data.watch_ratio > 1).sum())
                    stats['max_timestamp_fraction_digits'] = max(stats['max_timestamp_fraction_digits'], int(raw.timestamp.str.split('.').str[1].fillna('').str.len().max()))
                    all_users.update(data.user_id.tolist())
                    all_items.update(data.video_id.tolist())
                    invalid_users.update(data.loc[~valid_time, 'user_id'].tolist())
                    for i in data.index[~data.event_valid]:
                        quarantine.write(json.dumps({'source_event_id': int(data.loc[i, 'source_event_id']), 'reason': 'invalid_timestamp' if not valid_time[i] else 'invalid_field_or_clock', 'raw': raw.loc[i].to_dict()}, sort_keys=True, ensure_ascii=False) + '\n')
                    for bucket, frame in data.loc[valid_time].groupby(data.loc[valid_time, 'user_id'] % config['user_buckets'], sort=True):
                        if bucket not in writers:
                            writers[bucket] = pq.ParquetWriter(staging / f'{bucket:03d}.parquet', EVENT_SCHEMA, compression='zstd')
                        writers[bucket].write_table(pa.Table.from_pandas(frame, schema=EVENT_SCHEMA, preserve_index=False))
                    if stats['raw_events'] % 1_000_000 == 0:
                        log(stage='parse', rows=stats['raw_events'])
        for writer in writers.values():
            writer.close()
        with (output / 'clean_times.bin').open('wb') as times:
            for bucket in sorted(writers):
                frame = pq.read_table(staging / f'{bucket:03d}.parquet').to_pandas()
                frame.sort_values('source_event_id', inplace=True)
                # Once precise time and timestamp agree, date is only a noisy
                # redundant label, not a distinct observation. Preserve label
                # differences in the duplicate ledger rather than creating ties.
                frame['_invalid_clock_guard'] = np.where(frame.event_valid, '', frame.time_raw + '|' + frame.date_raw)
                keys = ['user_id', 'video_id', 'timestamp_us', 'play_duration', 'video_duration', 'watch_ratio_raw', 'event_valid', '_invalid_clock_guard']
                duplicated = frame.duplicated(keys)
                if duplicated.any():
                    canonical = frame.groupby(keys, dropna=False, sort=False).source_event_id.transform('min')
                    canonical_date = frame.groupby(keys, dropna=False, sort=False).date_raw.transform('first')
                    pair = pd.DataFrame({'removed_event_id': frame.loc[duplicated, 'source_event_id'], 'canonical_event_id': canonical[duplicated], 'removed_date_label': frame.loc[duplicated, 'date_raw'], 'canonical_date_label': canonical_date[duplicated]})
                    stats['duplicate_rows_with_different_date_label'] += int((pair.removed_date_label != pair.canonical_date_label).sum())
                    duplicate_writer.write_table(pa.Table.from_pandas(pair, preserve_index=False))
                stats['exact_duplicates_removed'] += int(duplicated.sum())
                frame = frame.loc[~duplicated].copy()
                frame.drop(columns=['_invalid_clock_guard'], inplace=True)
                frame.sort_values(['user_id', 'timestamp_us', 'source_event_id'], inplace=True)
                frame['ambiguous'] = frame.duplicated(['user_id', 'timestamp_us'], keep=False)
                stats['ambiguous_events'] += int(frame.ambiguous.sum())
                stats['clean_structural_events'] += len(frame)
                usable = frame.event_valid & ~frame.ambiguous
                stats['clean_unambiguous_events'] += int(usable.sum())
                for row in frame.loc[frame.ambiguous].itertuples():
                    quarantine.write(json.dumps({'source_event_id': row.source_event_id, 'user_id': row.user_id, 'timestamp_us': row.timestamp_us, 'reason': 'ambiguous_user_timestamp'}, sort_keys=True) + '\n')
                frame.loc[usable, 'timestamp_us'].to_numpy(dtype='<i8').tofile(times)
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), clean / f'{bucket:03d}.parquet', compression='zstd')
                log(stage='clean', bucket=int(bucket), events=len(frame))
    duplicate_writer.close()
    timestamps = np.memmap(output / 'clean_times.bin', dtype='<i8', mode='r+')
    ranks = [math.ceil(q * len(timestamps)) - 1 for q in config['split_quantiles']]
    timestamps.partition(ranks)
    cuts = [int(timestamps[i]) for i in ranks]
    timestamps.flush()
    del timestamps
    if cuts[0] >= cuts[1]:
        raise ValueError('Time cuts collapse; do not silently change protocol')
    meta = {'counts': dict(stats), 'cuts_us': cuts, 'cut_order_indices_zero_based': ranks,
            'invalid_timestamp_users': sorted(invalid_users), 'users': sorted(all_users), 'items': sorted(all_items)}
    dump(output / 'ingestion.json', meta)
    return meta


class SampleWriter:
    def __init__(self, path):
        self.schema = sample_schema()
        self.writer = pq.ParquetWriter(path, self.schema, compression='zstd')
        self.rows = []
        self.digest = hashlib.sha256()

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= 512:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self.schema) as writer:
            writer.write_table(table)
        self.digest.update(sink.getvalue())
        self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()
        return self.digest.hexdigest()


def build(config, output, meta):
    text_stats = repair_text(Path(config['input_text']), output)
    texts = [json.loads(line) for line in (output / 'item_text.jsonl').read_text(encoding='utf-8').splitlines()]
    users, items = meta['users'], sorted(set(meta['items']) | {row['video_id'] for row in texts})
    user_map = {v: i + 1 for i, v in enumerate(users)}
    item_map = {v: i + 1 for i, v in enumerate(items)}
    dump(output / 'user_map.json', {'pad_id': 0, 'entries': [{'original_id': v, 'id': user_map[v]} for v in users]})
    dump(output / 'item_map.json', {'pad_id': 0, 'entries': [{'original_id': v, 'id': item_map[v]} for v in items]})
    writers = {s: SampleWriter(output / f'{s}.parquet') for s in ('train', 'validation', 'test')}
    stats = Counter()
    for path in sorted((output / 'events').glob('*.parquet')):
        frame = pq.read_table(path).to_pandas()
        for uid, group in frame.groupby('user_id', sort=True):
            for result in user_samples(group.to_dict('records'), config, meta['cuts_us'], user_map, item_map, uid in meta['invalid_timestamp_users']):
                if '_stats' in result:
                    stats.update(result['_stats'])
                else:
                    writers[result['split']].add(result)
        log(stage='lists', bucket=path.stem, retained=stats['retained_blocks'])
    hashes = {s: writer.close() for s, writer in writers.items()}
    if any(stats['retained_' + s] == 0 for s in writers):
        raise ValueError('Empty split; protocol must not be relaxed automatically')
    dump(output / 'construction.json', {'counts': dict(stats), 'text': text_stats, 'normalized_sample_hashes': hashes,
                                      'hash_method': 'fixed schema Arrow IPC streams of consecutive 512 rows, concatenated into SHA256'})
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/kuairec_listwise_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'ingest', 'build'], default='all')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.stage != 'build':
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / 'config.json', config)
        meta = ingest(config, args.output)
    else:
        if (args.output / 'train.parquet').exists():
            raise FileExistsError('Refusing to overwrite an existing dataset')
        if json.loads((args.output / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Config changed since ingestion')
        meta = json.loads((args.output / 'ingestion.json').read_text(encoding='utf-8'))
    if args.stage != 'ingest':
        build(config, args.output, meta)
    log(stage='complete', output=str(args.output))


if __name__ == '__main__':
    main()
