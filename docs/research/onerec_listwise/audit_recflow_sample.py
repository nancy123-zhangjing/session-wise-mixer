"""Read complete Feather files from a bounded archive prefix; CPU only."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_prefix(prefix, directory):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {'prefix_bytes': prefix.stat().st_size, 'prefix_sha256': sha(prefix),
               'whole_archive_verified': False, 'complete_members': []}
    try:
        with tarfile.open(prefix, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.feather'):
                    continue
                if member.size > 300_000_000:
                    raise ValueError('Unexpectedly large member')
                target = directory / Path(member.name).name
                partial = target.with_suffix('.feather.partial')
                try:
                    with archive.extractfile(member) as source, partial.open('wb') as out:
                        shutil.copyfileobj(source, out, 1024 * 1024)
                    if partial.stat().st_size != member.size:
                        raise EOFError('Incomplete member')
                    partial.replace(target)
                except (EOFError, tarfile.ReadError) as error:
                    receipt['incomplete_member'] = member.name
                    receipt['stop_reason'] = str(error)
                    partial.unlink(missing_ok=True)
                    break
                receipt['complete_members'].append({'name': member.name, 'bytes': member.size, 'sha256': sha(target)})
    except (EOFError, tarfile.ReadError) as error:
        receipt['stop_reason'] = str(error)
    return receipt


def hist(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def audit(path):
    table = feather.read_table(path)
    schema = {field.name: str(field.type) for field in table.schema}
    frame = table.to_pandas()
    del table
    frame.columns = frame.columns.str.strip()
    raw_rows = len(frame)
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    observed = frame.loc[frame.realshow == 1].copy()
    observed['played'] = observed.playing_time > 0
    social = [c for c in ['like', 'follow', 'forward', 'comment'] if c in observed]
    observed['social'] = observed[social].gt(0).any(axis=1)
    group = observed.groupby('request_id', sort=False)
    req = group.agg(items=('video_id', 'nunique'), rows=('video_id', 'size'),
                    users=('user_id', 'nunique'), times=('request_timestamp', 'nunique'),
                    played=('played', 'sum'), effective=('effective_view', 'sum'),
                    total_play_raw=('playing_time', 'sum'), social=('social', 'max'))
    times = pd.to_datetime(observed.request_timestamp, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    out = {'file': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
           'raw_columns': schema, 'raw_rows': raw_rows, 'exact_duplicate_rows': duplicates,
           'realshow_rows': len(observed), 'non_realshow_rows': int((frame.realshow != 1).sum()),
           'users': int(observed.user_id.nunique()), 'items': int(observed.video_id.nunique()),
           'requests': len(req), 'duplicate_request_item_rows': int(observed.duplicated(['request_id', 'video_id']).sum()),
           'requests_multiple_users': int((req.users != 1).sum()),
           'requests_multiple_timestamps': int((req.times != 1).sum()),
           'time_min_shanghai': str(times.min()), 'time_max_shanghai': str(times.max()),
           'date_rows': hist(times.dt.strftime('%Y-%m-%d')),
           'items_per_request': hist(req['items']), 'played_per_request': hist(req.played),
           'effective_per_request': hist(req.effective), 'requests_social': int(req.social.sum()),
           'requests_played_ge5_effective_lt5': int(((req.played >= 5) & (req.effective < 5)).sum()),
           'negative_play_rows': int((observed.playing_time < 0).sum()),
           'nonpositive_duration_rows': int((observed.duration <= 0).sum()),
           'quality_counts': {}, 'position_diagnostics': {},
           'playing_time_unit': 'Unconfirmed by author documentation; millisecond scale is an inference. Thresholds below are raw values.'}
    for k in [2, 3, 5]:
        counts = {'items_ge_k': int((req['items'] >= k).sum()),
                  'played_ge_k': int((req.played >= k).sum()),
                  'effective_ge_k': int((req.effective >= k).sum()),
                  'played_ge_k_and_social': int(((req.played >= k) & req.social).sum())}
        for raw_threshold in [30000, 60000, 120000]:
            condition = (req.played >= k) & (req.total_play_raw >= raw_threshold)
            counts[f'played_ge_k_total_raw_ge_{raw_threshold}'] = int(condition.sum())
            counts[f'played_ge_k_total_raw_ge_{raw_threshold}_social'] = int((condition & req.social).sum())
        out['quality_counts'][str(k)] = counts
    for column in ['index', 'rank_index', 'rerank_index']:
        if column in observed:
            out['position_diagnostics'][column] = {
                'min': int(observed[column].min()), 'max': int(observed[column].max()),
                'negative_rows': int((observed[column] < 0).sum()),
                'requests_with_ties': int((group[column].nunique() < req.rows).sum())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = extract_prefix(args.directory / 'realshow.prefix128MiB.tar.gz', args.directory / 'archive_complete_days')
    files = [args.directory / 'archive_complete_days' / Path(m['name']).name for m in receipt['complete_members']]
    files += sorted(args.directory.glob('learning_*.feather'))
    result = {'schema_version': 1, 'mode': 'CPU-only', 'archive_extraction': receipt,
              'scope': 'Complete sampled files only; not full-dataset coverage or verified final display order.', 'files': []}
    for path in files:
        item = audit(path)
        result['files'].append(item)
        print(json.dumps({k: item[k] for k in ['file', 'raw_rows', 'requests', 'time_min_shanghai', 'time_max_shanghai', 'quality_counts']}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
