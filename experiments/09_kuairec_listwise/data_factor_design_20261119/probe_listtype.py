"""Read-only design probe: L5O slates that satisfy the old block rules; training sizes under a time cut;
Mixer difference on old-like versus other test slates (existing v2 / rhythm predictions)."""
import json, sys
import numpy as np, pyarrow.parquet as pq
from scripts.mixer_reversal import analyze as A
from scripts.kuairec_data.l5o_metrics import load_predictions
S = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
D = f'{S}/l5o_stage0_20261117_v1/dataset'
CUT80, CUT90 = 1598609167593000, 1598881785624000
def oldlike(t):
    ts = np.array(t['target_timestamp_us'].to_pylist(), dtype=np.int64)
    it = np.array(t['target_item_ids'].to_pylist(), dtype=np.int64)
    pdur = np.array(t['target_play_duration'].to_pylist(), dtype=np.int64)
    distinct = np.array([len(set(r)) == 5 for r in it])
    played = (pdur > 0).all(1)
    span = (ts[:, -1] - ts[:, 0]) <= 600 * 10**6
    nogap = (np.diff(ts, axis=1) <= 1800 * 10**6).all(1)
    return dict(distinct=distinct, played=played, span=span, nogap=nogap, all=distinct & played & span & nogap), ts
out = {}
for split in ('train', 'validation', 'test'):
    t = pq.read_table(f'{D}/{split}.parquet', columns=['target_timestamp_us', 'target_item_ids', 'target_play_duration'])
    m, ts = oldlike(t)
    row = {k: int(v.sum()) for k, v in m.items()}
    row['n'] = len(ts)
    if split == 'train':
        last = ts[:, -1]
        row['before_cut80'] = int((last < CUT80).sum()); row['before_cut90'] = int((last < CUT90).sum())
        row['oldlike_and_before_cut80'] = int((m['all'] & (last < CUT80)).sum())
        row['oldlike_and_before_cut90'] = int((m['all'] & (last < CUT90)).sum())
    out[split] = row
print(json.dumps(out))
# E1: Mixer difference on old-like vs other test slates
t = pq.read_table(f'{D}/test.parquet', columns=['sample_id', 'target_item_ids', 'target_watch_ratio', 'target_timestamp_us', 'target_play_duration'])
m, _ = oldlike(t)
ids = t['sample_id'].to_pylist()
tgt = np.array(t['target_item_ids'].to_pylist(), dtype=np.int64); wr = np.array(t['target_watch_ratio'].to_pylist(), float)
roots = {'v2': f'{S}/l5o_v2_20261118', 'r693': f'{S}/l5o_v2_rhythm693_20261119'}
res = {}
for name, root in roots.items():
    for s in (2024, 2025, 2026):
        for n in ('smoothed', 'best', 'final'):
            ps = {}
            for arm in ('H0', 'H1'):
                ps[arm] = A.per_sample(load_predictions(f'{root}/h/seed{s}/{arm}/test/{n}/per_sample.jsonl.gz', ids), tgt, wr)
            row = {}
            for k in ('ev1_ndcg', 'imp_ndcg'):
                d = ps['H1'][k] - ps['H0'][k]
                row[k] = dict(oldlike=A.user_boot(d, m['all']), other=A.user_boot(d, ~m['all']),
                              contrast=A.contrast_boot(d, m['all'], ~m['all']))
            res[f'{name}/{s}/{n}'] = row
print(json.dumps(res))
