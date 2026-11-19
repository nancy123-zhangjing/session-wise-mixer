"""Read-only profile of the old global-time split and the L5O split, side by side (train / validation / test).

For each split: sample and user counts, target-list structure (time span, 30-min breaks, repeats, zero plays),
watch ratios (EV targets), target time relative to the old cuts, history length and gap, targets already in the
model's history, and whether test targets appear among training targets on the same day.
Train history statistics use every 10th row. No training, no inference; CPU only.
"""
import json
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

S = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
SETS = {'old': f'{S}/goal_20261027_r1/inputs/dataset', 'l5o': f'{S}/l5o_stage0_20261117_v1/dataset'}
CUT80, CUT90 = 1598609167593000, 1598881785624000
MIN = 60 * 10**6
COLS = ['user_id', 'target_item_ids', 'target_timestamp_us', 'target_play_duration', 'target_watch_ratio',
        'history_item_ids', 'history_timestamp_us']


def fixed(col, k=5):
    lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    assert (lens == k).all(), np.unique(lens)
    return pc.list_flatten(col).to_numpy(zero_copy_only=False).reshape(-1, k)


def day(ts_us):  # Asia/Shanghai calendar day index since the Unix epoch
    return (ts_us // 10**6 + 8 * 3600) // 86400


def profile(path, split, hist_every):
    f = pq.ParquetFile(f'{path}/{split}.parquet')
    c = Counter()
    users, hist_len, gap_min, day_first = [], [], [], []
    pairs = []  # (day, item) of every target record
    in_hist_items = in_hist_lists = hist_rows = 0
    row0 = 0
    for b in f.iter_batches(batch_size=50_000, columns=COLS):
        n = b.num_rows
        it = fixed(b.column('target_item_ids')); ts = fixed(b.column('target_timestamp_us')).astype(np.int64)
        pd_ = fixed(b.column('target_play_duration')); wr = fixed(b.column('target_watch_ratio')).astype(float)
        assert (it > 0).all()
        users.append(b.column('user_id').to_numpy(zero_copy_only=False))
        span = (ts[:, -1] - ts[:, 0]) / MIN
        breaks = (np.diff(ts, axis=1) > 30 * MIN).sum(1)
        distinct = np.array([len(set(r)) == 5 for r in it]); played = (pd_ > 0).all(1)
        c['n'] += n
        c['span_le10m'] += int((span <= 10).sum()); c['span_10_60m'] += int(((span > 10) & (span <= 60)).sum())
        c['span_1_24h'] += int(((span > 60) & (span <= 1440)).sum()); c['span_gt1d'] += int((span > 1440).sum())
        for k in range(5):
            c[f'breaks_{k}'] += int((breaks == k).sum())
        c['distinct'] += int(distinct.sum()); c['all_played'] += int(played.sum())
        c['zero_play_items'] += int((pd_ <= 0).sum())
        c['oldlike_content'] += int((distinct & played & (span <= 10)).sum())
        c['ev1_items'] += int((wr >= 1).sum()); c['ev1_lists'] += int((wr >= 1).any(1).sum())
        c['ev2_items'] += int((wr >= 2).sum()); c['wr_sum'] += float(np.clip(wr, 0, 10).sum())
        first = ts[:, 0]
        c['first_before_cut80'] += int((first < CUT80).sum()); c['first_cut80_90'] += int(((first >= CUT80) & (first < CUT90)).sum())
        c['first_after_cut90'] += int((first >= CUT90).sum())
        day_first.append(day(first))
        pairs.append(np.unique(day(ts).ravel() * 100_000 + it.ravel()))
        # history (every hist_every-th row of the split, by global row index)
        idx = np.arange(row0, row0 + n)
        sel = np.flatnonzero(idx % hist_every == 0)
        hi = b.column('history_item_ids').take(sel).to_pylist()
        ht = b.column('history_timestamp_us').take(sel).to_pylist()
        for j, r in enumerate(sel):
            h = [x for x in hi[j] if x and x > 0]
            t = [x for x, y in zip(ht[j], hi[j]) if y and y > 0]
            hist_len.append(len(h))
            gap_min.append((ts[r, 0] - max(t)) / MIN if t else np.nan)
            hs = set(h)
            m = sum(int(x) in hs for x in it[r])
            in_hist_items += m; in_hist_lists += int(m > 0); hist_rows += 1
        row0 += n
    users = np.concatenate(users)
    _, per_user = np.unique(users, return_counts=True)
    per_user_sorted = np.sort(per_user)[::-1]
    top10 = per_user_sorted[:max(1, len(per_user_sorted) // 10)].sum() / per_user.sum()
    hist_len = np.array(hist_len); gap = np.array(gap_min, float)
    d = np.concatenate(day_first); dd, dn = np.unique(d, return_counts=True)
    n = c['n']
    out = dict(
        samples=n, users=int(len(per_user)),
        samples_per_user=dict(mean=float(per_user.mean()), median=float(np.median(per_user)), max=int(per_user.max()),
                              top10pct_user_share=float(top10)),
        list_structure={k: c[k] / n for k in ('span_le10m', 'span_10_60m', 'span_1_24h', 'span_gt1d', 'breaks_0', 'breaks_1',
                                              'breaks_2', 'breaks_3', 'breaks_4', 'distinct', 'all_played', 'oldlike_content')},
        zero_play_item_share=c['zero_play_items'] / (5 * n),
        watch=dict(mean_wr_clip10=c['wr_sum'] / (5 * n), ev1_item_share=c['ev1_items'] / (5 * n),
                   ev2_item_share=c['ev2_items'] / (5 * n), ev1_list_share=c['ev1_lists'] / n),
        time=dict(first_before_cut80=c['first_before_cut80'] / n, first_cut80_90=c['first_cut80_90'] / n,
                  first_after_cut90=c['first_after_cut90'] / n,
                  first_day_counts={str(int(k)): int(v) for k, v in zip(dd, dn)}),
        history=dict(rows=hist_rows, len_mean=float(hist_len.mean()), len_median=float(np.median(hist_len)),
                     len_quantiles=np.quantile(hist_len, [.1, .25, .75, .9]).tolist(),
                     gap_le30m=float(np.nanmean(gap <= 30)), gap_30m_1d=float(np.nanmean((gap > 30) & (gap <= 1440))),
                     gap_gt1d=float(np.nanmean(gap > 1440)), gap_median_min=float(np.nanmedian(gap)),
                     target_item_in_history_share=in_hist_items / (5 * hist_rows),
                     list_with_target_in_history_share=in_hist_lists / hist_rows),
        distinct_target_items=int(len(np.unique(np.concatenate(pairs) % 100_000))),
    )
    return out, np.unique(np.concatenate(pairs))


res = {}
for name, path in SETS.items():
    res[name] = {}
    tr, tr_pairs = profile(path, 'train', 10)
    res[name]['train'] = tr
    tr_items = np.unique(tr_pairs % 100_000)
    pair_set = tr_pairs
    for split in ('validation', 'test'):
        p, te_pairs = profile(path, split, 1)
        # same-day pool: is the (day, item) of each distinct test target pair also a training target pair?
        dy, item = te_pairs // 100_000, te_pairs % 100_000
        same = np.isin(te_pairs, pair_set)
        near = same | np.isin((dy - 1) * 100_000 + item, pair_set) | np.isin((dy + 1) * 100_000 + item, pair_set)
        p['target_pairs_vs_train'] = dict(distinct_day_item_pairs=int(len(te_pairs)),
                                          item_seen_as_train_target=float(np.isin(item, tr_items).mean()),
                                          same_day_in_train=float(same.mean()), within_1day_in_train=float(near.mean()))
        res[name][split] = p
print(json.dumps(res, indent=1))
