"""Audit complete KuaiRec big-matrix histories for 512 seeded users, CPU only."""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def checksum(path, name='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, name).hexdigest()


def text_audit(path, observed_items):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    valid_id = [r for r in rows if str(r.get('video_id', '')).isdigit()]
    complete = [r for r in valid_id if None not in r and all(v is not None for v in r.values())]
    bad = {'', 'UNKNOWN', 'unknown', 'nan', '[]'}
    numeric_ids = {int(r['video_id']) for r in valid_id}
    complete_ids = {int(r['video_id']) for r in complete}
    caption_ids = {int(r['video_id']) for r in complete if r['caption'].strip() not in bad}
    return {'bytes': path.stat().st_size, 'md5': checksum(path, 'md5'),
            'parsed_rows': len(rows), 'numeric_id_rows': len(valid_id),
            'unique_numeric_ids': len({r['video_id'] for r in valid_id}),
            'complete_rows': len(complete),
            'incomplete_numeric_ids': [r['video_id'] for r in valid_id if any(v is None for v in r.values())],
            'nonempty_complete_rows': {key: sum(r[key].strip() not in bad for r in complete) for key in complete[0] if key != 'video_id'},
            'rows_chinese_caption': sum(any('\u4e00' <= c <= '\u9fff' for c in r['caption']) for r in complete),
            'sample_items_without_numeric_metadata': sorted(observed_items - numeric_ids),
            'sample_items_without_complete_metadata': sorted(observed_items - complete_ids),
            'sample_items_with_nonempty_caption': len(observed_items & caption_ids),
            'note': 'Malformed lines are reported, not silently repaired or included in coverage denominators.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / 'KuaiRec.zip'
    md5 = checksum(path, 'md5')
    if md5 != '261550d472c48eff4990fb13c0e5bcf7':
        raise ValueError('Archive does not match author-published MD5')
    sample_ids = set(np.random.default_rng(2024).choice(7176, 512, replace=False).tolist())
    chunks, total_rows, all_users, all_items = [], 0, set(), set()
    columns = ['user_id', 'video_id', 'timestamp', 'play_duration', 'video_duration', 'watch_ratio']
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/big_matrix.csv'))
        with archive.open(name) as stream:
            for chunk in pd.read_csv(stream, usecols=columns, chunksize=300000):
                total_rows += len(chunk)
                all_users.update(chunk.user_id.unique().tolist())
                all_items.update(chunk.video_id.unique().tolist())
                chunks.append(chunk.loc[chunk.user_id.isin(sample_ids)].copy())
    frame = pd.concat(chunks, ignore_index=True)
    sample_rows = len(frame)
    missing = int(frame[columns].isna().any(axis=1).sum())
    frame = frame.dropna(subset=columns)
    exact_duplicates = int(frame.duplicated(columns).sum())
    frame = frame.drop_duplicates(columns)
    key = ['user_id', 'video_id', 'timestamp']
    conflict = frame.duplicated(key, keep=False)
    conflicting_rows = int(conflict.sum())
    # Audit conservatively excludes ambiguous events, preserving their count.
    frame = frame.loc[~conflict].sort_values(['user_id', 'timestamp', 'video_id'], kind='stable').reset_index(drop=True)
    frame['played'] = frame.play_duration > 0
    frame['observed_gap'] = frame.groupby('user_id').timestamp.diff()
    frame['prior_events'] = frame.groupby('user_id').cumcount()
    times = pd.to_datetime(frame.timestamp, unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
    result = {'schema_version': 1, 'mode': 'CPU-only', 'archive_md5': md5, 'archive_bytes': path.stat().st_size,
              'full_big_matrix_rows': total_rows, 'full_users': len(all_users), 'full_items': len(all_items),
              'sample_seed': 2024, 'requested_sample_users': 512, 'observed_sample_users': int(frame.user_id.nunique()),
              'sample_users': sorted(sample_ids), 'sample_raw_rows': sample_rows,
              'rows_missing_required_fields': missing, 'exact_duplicate_events_removed': exact_duplicates,
              'ambiguous_key_rows_excluded': conflicting_rows, 'clean_sample_rows': len(frame),
              'sample_items': int(frame.video_id.nunique()), 'timestamp_tie_rows': int((frame.observed_gap == 0).sum()),
              'time_min_shanghai': str(times.min()), 'time_max_shanghai': str(times.max()),
              'nonpositive_video_duration_rows': int((frame.video_duration <= 0).sum()),
              'text': text_audit(args.directory / 'kuairec_caption_category.csv', set(frame.video_id.unique().tolist())),
              'rules': {},
              'scope': 'Sample users retain all observed big-matrix records; big_matrix itself is filtered and is not a complete natural browsing log. No train/validation/test split is certified.'}
    for minutes in [10, 30, 60]:
        start = frame.observed_gap.isna() | (frame.observed_gap > minutes * 60)
        session = start.cumsum()
        groups = frame.groupby(session, sort=False)
        aggregate = groups.agg(events=('video_id', 'size'), items=('video_id', 'nunique'), played=('played', 'sum'))
        # Consecutive, non-overlapping groups of five within each observed session.
        position = groups.cumcount()
        block = frame.assign(session=session, block=position // 5).groupby(['session', 'block'], sort=False)
        blocks = block.agg(events=('video_id', 'size'), items=('video_id', 'nunique'),
                           unique_times=('timestamp', 'nunique'), played=('played', 'sum'),
                           prior=('prior_events', 'min'), total_play_ms=('play_duration', 'sum'),
                           user=('user_id', 'first'), time_min=('timestamp', 'min'), time_max=('timestamp', 'max'))
        eligible = (blocks.events == 5) & (blocks['items'] == 5) & (blocks.unique_times == 5) & (blocks.played == 5) & (blocks.prior >= 5)
        span = blocks.time_max - blocks.time_min
        quality = eligible & (blocks.total_play_ms >= 60000)
        result['rules'][str(minutes)] = {'sessions': len(aggregate),
            'session_median_events': float(aggregate.events.median()),
            'session_p90_events': float(aggregate.events.quantile(.9)),
            'sessions_unique_items_ge3': int((aggregate['items'] >= 3).sum()),
            'sessions_unique_items_ge5': int((aggregate['items'] >= 5).sum()),
            'nonoverlap_five_blocks_with_distinct_items_times_positive_play_and_prior5': int(eligible.sum()),
            'same_blocks_total_play_ge60000ms': int(quality.sum()),
            'candidate_users': int(blocks.loc[eligible, 'user'].nunique()),
            'total_play_ge60000ms_users': int(blocks.loc[quality, 'user'].nunique()),
            'candidate_span_seconds_p50': float(span[eligible].median()),
            'candidate_span_seconds_p90': float(span[eligible].quantile(.9)),
            'candidate_span_le600s': int((eligible & (span <= 600)).sum()),
            'total_play_ge60000ms_and_span_le600s': int((quality & (span <= 600)).sum())}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['sample_users', 'text']}, ensure_ascii=True), flush=True)
    print(json.dumps(result['text'], ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
