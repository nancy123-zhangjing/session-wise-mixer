"""Read-only size probe for design X: per user, held-out = the last old-rule block (session gap 30 min,
blocks at session offsets 0,5,..., 5 distinct played items within 10 min), validation = the previous one;
training = records before the validation block (X-all: consecutive stride-5 slates anchored at the
validation start; X-old: old-rule blocks entirely before it). Same cleaning as L5O (invalid/ambiguous dropped)."""
import json
from collections import Counter
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
S = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
src = Path(json.load(open(f'{S}/l5o_stage0_20261117_v1/dataset/config.json'))['source_dataset'])
GAP, SPAN, L, HMIN = 1800 * 10**6, 600 * 10**6, 5, 5
st = Counter(); test_ts = []; xall = xold = 0; l5o_users = 0
for path in sorted((src / 'events').glob('*.parquet')):
    f = pq.read_table(path, columns=['source_event_id', 'user_id', 'video_id', 'timestamp_us', 'play_duration', 'event_valid', 'ambiguous']).to_pandas()
    f = f.loc[f.event_valid & ~f.ambiguous].sort_values(['user_id', 'timestamp_us', 'source_event_id'], kind='mergesort')
    for uid, g in f.groupby('user_id', sort=True):
        ts = g.timestamp_us.to_numpy(); vid = g.video_id.to_numpy(); pdur = g.play_duration.to_numpy()
        n = len(ts)
        if n >= 15: l5o_users += 1
        starts = [0] + [i for i in range(1, n) if ts[i] - ts[i - 1] > GAP]
        ends = starts[1:] + [n]
        blocks = []
        for a, b in zip(starts, ends):
            for o in range(a, b - L + 1, L):
                k = slice(o, o + L)
                if len(set(vid[k])) == L and (pdur[k] > 0).all() and ts[o + L - 1] - ts[o] <= SPAN and o >= HMIN:
                    blocks.append(o)
        st['users'] += 1
        if len(blocks) < 2:
            st['users_lt2_blocks'] += 1; continue
        vstart, tstart = blocks[-2], blocks[-1]
        nx_all = len(range(vstart, HMIN + L - 1, -L)) - 1   # stride-5 slates ending at vstart, history >= 5
        nx_old = sum(1 for o in blocks[:-2] if o + L <= vstart)
        if nx_all <= 0:
            st['users_no_train'] += 1
        st['users_ok'] += 1; xall += max(nx_all, 0); xold += nx_old
        st['after_test_records'] += n - (tstart + L)
        test_ts.append(ts[tstart])
test_ts = np.array(test_ts)
CUT80, CUT90 = 1598609167593000, 1598881785624000
print(json.dumps(dict(counts=dict(st), l5o_users_ge15=l5o_users, xall_train_slates=xall, xold_train_slates=xold,
                      test_start_quantiles_days=((np.quantile(test_ts, [.05, .25, .5, .75, .95]) - CUT90) / 8.64e10).round(2).tolist(),
                      test_in_old_window=float((test_ts >= CUT90).mean()))))
