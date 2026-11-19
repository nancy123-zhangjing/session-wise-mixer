"""Read-only: Mixer difference by the session structure of the L5O test slate (dose-response)."""
import json
import numpy as np, pyarrow.parquet as pq
from scripts.mixer_reversal import analyze as A
from scripts.kuairec_data.l5o_metrics import load_predictions
S = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
D = f'{S}/l5o_stage0_20261117_v1/dataset'
t = pq.read_table(f'{D}/test.parquet', columns=['sample_id', 'target_item_ids', 'target_watch_ratio', 'target_timestamp_us', 'target_play_duration'])
ids = t['sample_id'].to_pylist()
tgt = np.array(t['target_item_ids'].to_pylist(), dtype=np.int64); wr = np.array(t['target_watch_ratio'].to_pylist(), float)
ts = np.array(t['target_timestamp_us'].to_pylist(), dtype=np.int64)
pdur = np.array(t['target_play_duration'].to_pylist(), dtype=np.int64)
breaks = (np.diff(ts, axis=1) > 1800 * 10**6).sum(1)
span_min = (ts[:, -1] - ts[:, 0]) / 6e7
distinct = np.array([len(set(r)) == 5 for r in tgt]); played = (pdur > 0).all(1)
groups = {f'breaks_{k}': breaks == k for k in range(5)}
groups.update({'span_le10min': span_min <= 10, 'span_10_60min': (span_min > 10) & (span_min <= 60),
               'span_1_24h': (span_min > 60) & (span_min <= 1440), 'span_gt_1d': span_min > 1440,
               'one_session': breaks == 0, 'multi_session': breaks > 0,
               'one_session_distinct_played': (breaks == 0) & distinct & played,
               'oldlike': (breaks == 0) & distinct & played & (span_min <= 10)})
sizes = {g: int(m.sum()) for g, m in groups.items()}
roots = {'v2': f'{S}/l5o_v2_20261118', 'r693': f'{S}/l5o_v2_rhythm693_20261119'}
res = {}
for name, root in roots.items():
    for n in ('smoothed', 'best', 'final'):
        for s in (2024, 2025, 2026):
            ps = {arm: A.per_sample(load_predictions(f'{root}/h/seed{s}/{arm}/test/{n}/per_sample.jsonl.gz', ids), tgt, wr)
                  for arm in ('H0', 'H1')}
            row = {}
            for k in ('ev1_ndcg', 'imp_ndcg'):
                d = ps['H1'][k] - ps['H0'][k]
                row[k] = {g: A.user_boot(d, m, draws=1000) for g, m in groups.items()}
                row[k]['contrast_one_minus_multi_session'] = A.contrast_boot(d, groups['one_session'], groups['multi_session'], draws=1000)
                row[k]['slope_per_break'] = A.slope_boot(d, breaks.astype(float), draws=1000)
            res[f'{name}/{n}/{s}'] = row
print(json.dumps(dict(sizes=sizes, results=res)))
