"""Independently reconstruct complete histories of 32 users from original CSV.

This verifier deliberately does not import the preparation or protocol modules.
Run: python -m scripts.kuairec_data.trace_verify --dataset PATH
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SEED = 2024
USER_COUNT = 32
SCALE = 1_000_000
RATIO_RTOL = 1e-14
RATIO_ATOL = 1e-15
SPLITS = ('train', 'validation', 'test')
FIELDS = ('user_id', 'video_id', 'play_duration', 'video_duration',
          'time', 'date', 'timestamp', 'watch_ratio')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_event(raw, event_id):
    """Return exact semantic identity and parsed fields; retain invalid barriers."""
    uid = int(raw['user_id'])
    try:
        number = Decimal(raw['timestamp'])
        scaled = number * SCALE
        if not number.is_finite() or scaled != scaled.to_integral_value() or number < 0:
            raise ValueError('Timestamp cannot be represented as nonnegative integer microseconds')
        timestamp = int(scaled)
    except (InvalidOperation, ValueError, OverflowError):
        return None

    valid = True
    parsed = {}
    for name in ('video_id', 'play_duration', 'video_duration'):
        if re.fullmatch(r'[0-9]+', raw[name]):
            parsed[name] = int(raw[name])
        else:
            valid = False
            parsed[name] = 0
    try:
        ratio_decimal = Decimal(raw['watch_ratio'])
        ratio = float(ratio_decimal)
        valid &= ratio_decimal.is_finite() and math.isfinite(ratio) and ratio >= 0
    except (InvalidOperation, ValueError, OverflowError):
        ratio_decimal, ratio, valid = None, float('nan'), False
    valid &= parsed['video_duration'] > 0
    try:
        expected_local = datetime(1970, 1, 1) + timedelta(microseconds=timestamp, hours=8)
        clock_ok = datetime.fromisoformat(raw['time']) == expected_local
        date_mismatch = raw['date'] != expected_local.strftime('%Y%m%d')
    except (ValueError, OverflowError):
        clock_ok, date_mismatch = False, True
    valid &= clock_ok
    # Decimal parses time exactly. Preserve original decimal feedback so float
    # conversion cannot merge subtly different observations. Valid time is already
    # verified against timestamp; auxiliary date labels do not identify events.
    identity = (uid, parsed['video_id'] if re.fullmatch(r'[0-9]+', raw['video_id']) else raw['video_id'],
                timestamp,
                parsed['play_duration'] if re.fullmatch(r'[0-9]+', raw['play_duration']) else raw['play_duration'],
                parsed['video_duration'] if re.fullmatch(r'[0-9]+', raw['video_duration']) else raw['video_duration'],
                raw['watch_ratio'],
                '' if valid else raw['time'] + '|' + raw['date'])
    return {'source_event_id': event_id, 'user_id': uid, 'timestamp_us': timestamp,
            **parsed, 'watch_ratio': ratio, 'event_valid': bool(valid),
            'clock_ok': bool(clock_ok), 'date_mismatch': bool(date_mismatch), 'identity': identity}


def deduplicate_and_mark(records):
    seen, unique = set(), []
    removed = 0
    for record in sorted(records, key=lambda x: x['source_event_id']):
        if record['identity'] in seen:
            removed += 1
        else:
            seen.add(record['identity'])
            unique.append(record)
    unique.sort(key=lambda x: (x['timestamp_us'], x['source_event_id']))
    frequencies = Counter(record['timestamp_us'] for record in unique)
    for record in unique:
        record['ambiguous'] = frequencies[record['timestamp_us']] > 1
    return unique, removed


def self_check():
    """Check semantic duplicates independently before touching a real dataset."""
    raw = {'user_id': '0', 'video_id': '1', 'play_duration': '1000',
           'video_duration': '2000', 'time': '2020-07-05 00:08:23.438',
           'date': '20200705', 'timestamp': '1593878903.438', 'watch_ratio': '0.5'}
    date_only = {**raw, 'date': '20200706'}
    feedback = {**raw, 'play_duration': '1200', 'watch_ratio': '0.6'}
    later = {**raw, 'timestamp': '1593878904.438', 'time': '2020-07-05 00:08:24.438'}
    records = [parse_event(row, i + 1) for i, row in enumerate((raw, date_only, feedback, later))]
    if not all(record['event_valid'] for record in records):
        raise AssertionError('Verifier self-check: synthetic valid event unexpectedly quarantined')
    unique, removed = deduplicate_and_mark(records)
    if removed != 1 or [record['source_event_id'] for record in unique] != [1, 3, 4]:
        raise AssertionError('Verifier self-check: date-label duplicate or distinct feedback/time handled incorrectly')
    if [record['ambiguous'] for record in unique] != [True, True, False]:
        raise AssertionError('Verifier self-check: distinct same-time feedback must remain ambiguous')
    return True


def recover_users(archive_path, chosen):
    selected = set(chosen)
    frames = []
    scanned = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith('/big_matrix.csv')]
        if len(members) != 1:
            raise ValueError('Exactly one raw big_matrix member is required')
        with archive.open(members[0]) as stream:
            for chunk in pd.read_csv(stream, dtype=str, keep_default_na=False, chunksize=100_000):
                if tuple(chunk.columns) != FIELDS:
                    raise ValueError('Original CSV schema differs from the audited input')
                # pandas retains the global zero-based input record index per chunk.
                ids = pd.to_numeric(chunk['user_id'], errors='raise')
                part = chunk.loc[ids.isin(selected)].copy()
                part['source_event_id'] = part.index.to_numpy() + 1
                frames.append(part)
                scanned += len(chunk)
                if scanned % 2_000_000 == 0:
                    print(json.dumps({'trace_stage': 'raw_scan', 'rows': scanned}), flush=True)
    raw_selected = pd.concat(frames, ignore_index=True)
    by_user = defaultdict(list)
    invalid_time_users = set()
    counts = Counter(raw_rows_scanned=scanned, selected_raw_rows=len(raw_selected))
    for raw in raw_selected.to_dict('records'):
        record = parse_event(raw, int(raw.pop('source_event_id')))
        if record is None:
            invalid_time_users.add(int(raw['user_id']))
            counts['selected_invalid_timestamp_rows'] += 1
            continue
        by_user[record['user_id']].append(record)
        counts['selected_invalid_field_rows'] += int(not record['event_valid'])
        counts['selected_clock_mismatch_rows'] += int(not record['clock_ok'])
        counts['selected_date_label_mismatch_rows'] += int(record['date_mismatch'])
    for uid, records in by_user.items():
        unique, removed = deduplicate_and_mark(records)
        counts['selected_exact_duplicates_removed'] += removed
        counts['selected_ambiguous_events'] += sum(record['ambiguous'] for record in unique)
        counts['selected_clean_structural_events'] += len(unique)
        by_user[uid] = unique
    return by_user, invalid_time_users, counts


def reconstructed_samples(records, config, cuts, user_map, item_map, suppressed):
    """Rebuild each whole session; history selection is independent of block walk."""
    if suppressed or not records:
        return []
    gap = int(config['session_gap_seconds']) * SCALE
    block_length = int(config['list_length'])
    played = [r for r in records if r['event_valid'] and not r['ambiguous'] and r['play_duration'] > 0]
    played_times = [r['timestamp_us'] for r in played]
    boundaries = [0] + [i for i in range(1, len(records))
                        if records[i]['timestamp_us'] - records[i - 1]['timestamp_us'] > gap] + [len(records)]
    expected = []
    for begin, end in zip(boundaries, boundaries[1:]):
        session = records[begin:end]
        first, last = session[0]['timestamp_us'], session[-1]['timestamp_us']
        if any(first < cut <= last for cut in cuts):
            continue
        session_id = 'u{}:e{}'.format(session[0]['user_id'], min(r['source_event_id'] for r in session))
        split = SPLITS[sum(first >= cut for cut in cuts)]
        for offset in range(0, len(session) - block_length + 1, block_length):
            target = session[offset:offset + block_length]
            if (any(not r['event_valid'] or r['ambiguous'] or r['play_duration'] <= 0 for r in target)
                    or len({r['video_id'] for r in target}) != block_length
                    or target[-1]['timestamp_us'] - target[0]['timestamp_us'] > config['max_target_span_seconds'] * SCALE):
                continue
            pos = bisect_left(played_times, target[0]['timestamp_us'])
            history = played[max(0, pos - config['history_max']):pos]
            if len(history) < config['history_min']:
                continue
            total_play = sum(r['play_duration'] for r in target)
            sample = {
                'sample_id': session_id + ':b' + str(offset // block_length),
                'user_id': user_map[session[0]['user_id']],
                'original_user_id': session[0]['user_id'], 'session_id': session_id,
                'session_start_us': first, 'session_end_us': last, 'block_offset': offset,
                'split': split, 'target_span_us': target[-1]['timestamp_us'] - target[0]['timestamp_us'],
                'target_total_play_ms': total_play,
                'train_optional_ge60s': total_play >= config['train_optional_total_play_ms'] if split == 'train' else None,
            }
            for prefix, group in [('history', history), ('target', target)]:
                sample[prefix + '_item_ids'] = [item_map[r['video_id']] for r in group]
                sample[prefix + '_original_item_ids'] = [r['video_id'] for r in group]
                for field in ('source_event_id', 'timestamp_us', 'play_duration', 'video_duration', 'watch_ratio'):
                    sample[prefix + '_' + field] = [r[field] for r in group]
                sample[prefix + '_mask'] = [True] * len(group)
            expected.append(sample)
    return expected


def verify(dataset):
    self_check()
    config = read_json(dataset / 'config.json')
    ingestion = read_json(dataset / 'ingestion.json')
    user_data, item_data = read_json(dataset / 'user_map.json'), read_json(dataset / 'item_map.json')
    if user_data['pad_id'] != 0 or item_data['pad_id'] != 0:
        raise AssertionError('PAD must be zero')
    user_map = {entry['original_id']: entry['id'] for entry in user_data['entries']}
    item_map = {entry['original_id']: entry['id'] for entry in item_data['entries']}
    for mapping in (user_map, item_map):
        if [mapping[k] for k in sorted(mapping)] != list(range(1, len(mapping) + 1)):
            raise AssertionError('ID map is not sorted, deterministic, contiguous, and PAD-exclusive')
    if len(user_map) < USER_COUNT:
        raise ValueError('The full dataset must have at least 32 users')
    chosen = sorted(int(x) for x in np.random.default_rng(SEED).choice(sorted(user_map), USER_COUNT, replace=False))
    archive_path = Path(config['input_archive'])
    if not archive_path.is_absolute():
        archive_path = Path(__file__).resolve().parents[2] / archive_path
    if file_sha256(archive_path) != config['archive_sha256']:
        raise AssertionError('Original archive SHA256 has changed')
    by_user, blocked_users, counts = recover_users(archive_path, chosen)
    expected = {}
    for uid in chosen:
        for row in reconstructed_samples(by_user[uid], config, ingestion['cuts_us'], user_map, item_map, uid in blocked_users):
            if row['sample_id'] in expected:
                raise AssertionError('Reconstructed duplicate sample ID')
            expected[row['sample_id']] = row
    if not expected:
        raise AssertionError('No independently reconstructed sample')
    actual = {}
    for split in SPLITS:
        table = pq.read_table(dataset / f'{split}.parquet', filters=[('original_user_id', 'in', chosen)])
        counts['verified_' + split + '_samples'] = table.num_rows
        for row in table.to_pylist():
            if row['sample_id'] in actual:
                raise AssertionError('Output duplicates a sample within or across splits')
            if row['split'] != split:
                raise AssertionError('Sample split field differs from its file')
            actual[row['sample_id']] = row
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise AssertionError(f'Complete-user sample sets differ: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}')
    maximum_ratio_error = 0.0
    for sid, wanted in expected.items():
        got = actual[sid]
        for key, value in wanted.items():
            if key.endswith('_watch_ratio'):
                if len(got[key]) != len(value):
                    raise AssertionError(f'{sid}: {key} length differs')
                for lhs, rhs in zip(got[key], value):
                    maximum_ratio_error = max(maximum_ratio_error, abs(lhs - rhs))
                    if not math.isclose(lhs, rhs, rel_tol=RATIO_RTOL, abs_tol=RATIO_ATOL):
                        raise AssertionError(f'{sid}: original watch_ratio feedback differs')
            elif got.get(key) != value:
                raise AssertionError(f'{sid}: {key} differs from the original-event reconstruction')
    return {
        'schema_version': 1, 'passed': True, 'selection_seed': SEED, 'selected_users': chosen,
        'semantic_duplicate_self_check_passed': True,
        'source_archive_sha256': config['archive_sha256'],
        'verifier_sha256': file_sha256(Path(__file__)),
        'config_sha256': file_sha256(dataset / 'config.json'),
        'counts': dict(counts), 'verified_samples': len(actual),
        'selected_users_with_invalid_timestamp': sorted(blocked_users),
        'ratio_float_comparison': {'relative_tolerance': RATIO_RTOL, 'absolute_tolerance': RATIO_ATOL,
                                   'maximum_observed_absolute_difference': maximum_ratio_error},
        'checks': [
            'Original ZIP scanned without using prepared events or main transformation functions',
            'All original records and all output samples for 32 fixed users reconstructed',
            'Decimal time arithmetic, semantic deduplication and ambiguity barriers checked',
            'Whole-session split boundaries and unfiltered nonoverlap block positions checked',
            'Strict-past nearest played history independently selected with binary search',
            'Exact source event IDs, original IDs, ordering, times, durations, masks and map IDs checked',
            'All three split sample sets and optional train-only flag checked',
        ],
        'scope': 'Independent complete-history trace for 32 users, not a replacement for full-dataset invariant validation or quantile verification.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.dataset)
    except Exception as error:
        result = {'schema_version': 1, 'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
        (args.dataset / 'trace_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    (args.dataset / 'trace_validation.json').write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
