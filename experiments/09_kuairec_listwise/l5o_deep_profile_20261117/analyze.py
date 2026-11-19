"""Read-only deep profile: L5O first wave versus the old global-time split.

Runs on a CPU notebook next to Inspire storage. Uses existing datasets and
existing per-sample test predictions only; no training, inference or writes
outside --output.

Data level (both protocols): item classes with the old Stage09 rule (train-target
rank: head top 10 %, mid next 40 %, tail rest; never_seen / history_only),
target composition, repeat consumption, how much of each target item's training
signal lies after the target time (other users' future), and history-filtered
popularity baselines (global, causal past-7-day, contemporaneous +/-1 day).

Model level (L5O, 10 prediction sets): recommendation profile, EV-NDCG@5
contribution by item class (sums to EV-NDCG@5), per-class target hit rates,
slot hit rates, breakdown by test-time and user-activity terciles, and paired
comparisons (Mixer and best->final) with a user bootstrap (2000 draws, seed 2024).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.kuairec_data.l5o_metrics import DISCOUNT, _relevance, load_predictions, slate_metrics  # noqa: E402

K = 5
NITEM = 10728
DAY = 86_400_000_000
G = ['never_seen', 'history_only', 'head', 'mid', 'tail']
DRAWS = 2000
TOPN = 600


def flat(col):
    return col.flatten().to_numpy(zero_copy_only=False).astype(np.int64)


def lengths(col):
    off = col.offsets.to_numpy()
    return np.diff(off)


def train_stats(path):
    """Item classes, per-day target counts and per-user prior item keys."""
    pf = pq.ParquetFile(path)
    tc = np.zeros(NITEM + 1, np.int64)
    hc = np.zeros(NITEM + 1, np.int64)
    days_items, keys, user_targets = [], [], {}
    for b in pf.iter_batches(batch_size=200_000, columns=['user_id', 'history_item_ids', 'target_item_ids',
                                                          'target_timestamp_us']):
        users = b.column('user_id').to_numpy().astype(np.int64)
        t_col, h_col, ts_col = b.column('target_item_ids'), b.column('history_item_ids'), b.column('target_timestamp_us')
        t, h, ts = flat(t_col), flat(h_col), flat(ts_col)
        tu = np.repeat(users, lengths(t_col))
        hu = np.repeat(users, lengths(h_col))
        tc += np.bincount(t[t > 0], minlength=NITEM + 1)
        hc += np.bincount(h[h > 0], minlength=NITEM + 1)
        ok = t > 0
        days_items.append((ts[ok], t[ok]))
        keys.append(np.unique(tu[ok] * (NITEM + 1) + t[ok]))
        keys.append(np.unique(hu[h > 0] * (NITEM + 1) + h[h > 0]))
        cnt = np.bincount(tu[ok])
        for u in np.flatnonzero(cnt):
            user_targets[int(u)] = user_targets.get(int(u), 0) + int(cnt[u])
    seen = (tc + hc) > 0
    seen[0] = False
    ranked = sorted((np.flatnonzero(tc[1:]) + 1).tolist(), key=lambda i: (-tc[i], i))
    cls = np.full(NITEM + 1, -1, np.int8)
    cls[1:] = 0
    cls[seen] = 1
    h, m = math.ceil(.1 * len(ranked)), math.ceil(.5 * len(ranked))
    cls[ranked[:h]] = 2
    cls[ranked[h:m]] = 3
    cls[ranked[m:]] = 4
    cls[0] = -1
    ts_all = np.concatenate([x[0] for x in days_items])
    it_all = np.concatenate([x[1] for x in days_items])
    prior = np.unique(np.concatenate(keys))
    pct = np.zeros(NITEM + 1)
    order = np.argsort(-tc[1:], kind='stable') + 1
    pct[order] = 1.0 - np.arange(len(order)) / len(order)          # 1.0 = most popular train target
    return dict(tc=tc, cls=cls, ts=ts_all, items=it_all, prior=prior, pct=pct, user_targets=user_targets)


def load_eval(path):
    t = pq.read_table(path, columns=['sample_id', 'user_id', 'history_item_ids', 'target_item_ids',
                                     'target_watch_ratio', 'target_timestamp_us'])
    hist = [np.asarray([x for x in row if x > 0], np.int64) for row in t['history_item_ids'].to_pylist()]
    tgt = np.asarray(t['target_item_ids'].to_pylist(), np.int64)
    wr = np.asarray(t['target_watch_ratio'].to_pylist(), float)
    tts = np.asarray(t['target_timestamp_us'].to_pylist(), np.int64)
    return dict(ids=t['sample_id'].to_pylist(), users=np.asarray(t['user_id'].to_pylist(), np.int64),
                hist=hist, tgt=tgt, wr=wr, tts=tts)


def composition(E, S):
    cls = S['cls']
    out = {}
    imp = E['tgt'] > 0
    ev = imp & (E['wr'] >= 1.0)
    for name, mask in (('impressions', imp), ('ev1', ev)):
        c = np.bincount(cls[E['tgt'][mask]], minlength=5)
        out[name] = dict(targets=int(mask.sum()), share=dict(zip(G, (c / c.sum()).round(6).tolist())))
    in_window = np.array([[x in set(h.tolist()) if x > 0 else False for x in row]
                          for row, h in zip(E['tgt'], E['hist'])])
    keys = E['users'][:, None] * (NITEM + 1) + E['tgt']
    ever = np.isin(keys, S['prior']) & imp
    out['repeat_in_history_window'] = float(in_window[imp].mean())
    out['repeat_in_user_train_prior'] = float(ever[imp].mean())
    out['ev1_repeat_in_history_window'] = float(in_window[ev].mean())
    out['ev1_repeat_in_user_train_prior'] = float(ever[ev].mean())
    out['samples_with_ev1'] = float(ev.any(1).mean())
    out['history_len_mean'] = float(np.mean([len(h) for h in E['hist']]))
    return out


class Days:
    def __init__(self, S, t0):
        self.t0 = t0
        d = (S['ts'] - t0) // DAY
        self.n = int(d.max()) + 1
        self.mat = np.zeros((self.n, NITEM + 1), np.int64)
        np.add.at(self.mat, (d, S['items']), 1)
        self.cum = np.cumsum(self.mat, 0)
        self.total = self.cum[-1]
        self.cache = {}

    def day(self, ts):
        return (np.asarray(ts) - self.t0) // DAY

    def counts(self, lo, hi):
        lo, hi = max(lo, 0), min(hi, self.n - 1)
        if hi < lo:
            return np.zeros(NITEM + 1, np.int64)
        return self.cum[hi] - (self.cum[lo - 1] if lo > 0 else 0)

    def ranked(self, key, lo, hi):
        """Top items by training-target count in days [lo, hi]; None when the window is empty."""
        if key not in self.cache:
            c = self.counts(lo, hi).astype(float)
            if c[1:].sum() == 0:
                self.cache[key] = None
            else:
                c[0] = -1
                self.cache[key] = np.argsort(-c, kind='stable')[:TOPN]
        return self.cache[key]

    def future_share(self, items, days):
        """Share of each item's training target events strictly after `days`."""
        days = np.clip(days, -1, self.n - 1)
        after = self.total[items] - np.where(days >= 0, self.cum[np.maximum(days, 0), items], 0)
        tot = self.total[items]
        return np.where(tot > 0, after / np.maximum(tot, 1), np.nan)


def popularity_lists(E, D, S, kind):
    preds = np.zeros((len(E['ids']), K), np.int64)
    d0 = D.day(E['tts'][:, 0])
    glob = np.argsort(-np.where(np.arange(NITEM + 1) > 0, S['tc'], -1), kind='stable')[:TOPN]
    fallback = 0
    for i, (h, d) in enumerate(zip(E['hist'], d0)):
        d = int(d)
        if kind == 'global':
            cand = glob
        elif kind == 'past7':
            # seven most recent days of training data strictly before the target day
            last = min(d - 1, D.n - 1)
            cand = D.ranked(('past7', last), last - 6, last)
        else:
            cand = D.ranked(('win1', d), d - 1, d + 1)
        if cand is None:
            cand = glob
            fallback += 1
        hs = set(h.tolist())
        pick = []
        for x in cand:
            x = int(x)
            if x > 0 and x not in hs:
                pick.append(x)
                if len(pick) == K:
                    break
        while len(pick) < K:
            pick.append(0)
        preds[i] = pick
    return preds, fallback / len(E['ids'])


def score(preds, E):
    m1, per = slate_metrics(preds, E['tgt'], E['wr'], 1.0, return_per_sample=True)
    keep = ('imp_hit@5', 'imp_ndcg@5', 'legacy_fixed5_ndcg@5', 'ev_hit@5', 'ev_ndcg@5')
    return {k: m1[k] for k in keep}, per


def terciles(values):
    q = np.quantile(values, [1 / 3, 2 / 3])
    return np.digitize(values, q), q.tolist()


def protocol_data(name, root, splits):
    S = train_stats(root / 'train.parquet')
    E = {s: load_eval(root / f'{s}.parquet') for s in splits}
    t0 = min(int(S['ts'].min()), *(int(e['tts'][e['tts'] > 0].min()) for e in E.values()))
    D = Days(S, t0)
    out = dict(class_counts=np.bincount(S['cls'][1:], minlength=5).tolist(), train_days=D.n, splits={})
    per_base = {}
    for s, e in E.items():
        row = composition(e, S)
        d0 = D.day(e['tts'][:, 0])
        row['target_day_quantiles'] = np.quantile(d0, [0, .1, .5, .9, 1]).tolist()
        imp = e['tgt'] > 0
        tday = D.day(e['tts'])
        fut = D.future_share(np.where(imp, e['tgt'], 0), np.where(imp, tday, 0))
        row['target_future_train_share_mean'] = float(np.nanmean(fut[imp]))
        row['target_future_train_share_ge_half'] = float(np.nanmean((fut[imp] >= .5).astype(float)))
        base = {}
        for kind in ('global', 'past7', 'win1'):
            pl, fb = popularity_lists(e, D, S, kind)
            m, per = score(pl, e)
            m['empty_window_fallback_rate'] = fb
            base[kind] = m
            per_base[(s, kind)] = per
        row['popularity_baselines'] = base
        out['splits'][s] = row
    return out, S, E, D, per_base


def class_contrib(pred, E, S):
    cls = S['cls']
    imp_sets = [set(int(x) for x in r if x > 0) for r in E['tgt']]
    ev_sets = [set(int(x) for x, w in zip(r, wr) if x > 0 and w >= 1.0) for r, wr in zip(E['tgt'], E['wr'])]
    rel_i, rel_e = _relevance(pred, imp_sets), _relevance(pred, ev_sets)
    ev_size = np.array([len(s) for s in ev_sets])
    has = ev_size > 0
    idcg = np.array([DISCOUNT[:min(K, s)].sum() if s > 0 else 1.0 for s in ev_size])
    pc = cls[np.clip(pred, 0, NITEM)]
    contrib = np.stack([((rel_e & (pc == k)) * DISCOUNT).sum(1) / idcg for k in range(5)], 1)
    return dict(rel_i=rel_i, rel_e=rel_e, has=has, contrib=contrib, imp_sets=imp_sets, ev_sets=ev_sets)


def model_profile(pred, E, S, D, C, tert):
    cls, pct = S['cls'], S['pct']
    f = pred.reshape(-1)
    f = f[(f > 0) & (f <= NITEM)]
    ic = np.bincount(f, minlength=NITEM + 1)
    has = C['has']
    out = dict(
        ev_ndcg5=float(C['contrib'][has].sum(1).mean()),
        imp_hit5=float(C['rel_i'].any(1).mean()),
        distinct_items=int((ic > 0).sum()),
        top10_share=float(np.sort(ic)[::-1][:10].sum() / len(f)),
        rec_share=dict(zip(G, (np.bincount(cls[f], minlength=5) / len(f)).round(6).tolist())),
        rec_popularity_pct_mean=float(pct[f].mean()),
        ev_contribution=dict(zip(G, C['contrib'][has].mean(0).tolist())),
        slot_hit_imp=C['rel_i'].mean(0).tolist(),
        slot_hit_ev=C['rel_e'][has].mean(0).tolist())
    d0 = D.day(E['tts'][:, 0])
    in_glob, in_trend = [], []
    glob = set(np.argsort(-np.where(np.arange(NITEM + 1) > 0, S['tc'], -1), kind='stable')[:100].tolist())
    for row, d in zip(pred, d0):
        r = D.ranked(('win1', int(d)), int(d) - 1, int(d) + 1)
        trend = set(r[:100].tolist()) if r is not None else set()
        for x in row:
            if x > 0:
                in_glob.append(x in glob)
                in_trend.append(x in trend)
    out['rec_in_global_top100'] = float(np.mean(in_glob))
    out['rec_in_day_window_top100'] = float(np.mean(in_trend))
    hit = {}
    for name, sets in (('impressions', C['imp_sets']), ('ev1', C['ev_sets'])):
        hits, totals = np.zeros(5), np.zeros(5)
        for row, s in zip(pred, sets):
            ps = set(int(x) for x in row)
            for x in s:
                totals[cls[x]] += 1
                hits[cls[x]] += x in ps
        hit[name] = {g: (float(hits[k] / totals[k]) if totals[k] else None) for k, g in enumerate(G)}
    out['target_hit_rate'] = hit
    per = C['contrib'].sum(1)
    out['ev_ndcg5_by_tercile'] = {t: [float(per[has & (lab == k)].mean()) for k in range(3)]
                                  for t, lab in tert.items()}
    return out


def boot(diff, has, rng):
    d = diff[has]
    n = len(d)
    means = []
    for start in range(0, DRAWS, 100):
        idx = rng.integers(n, size=(min(100, DRAWS - start), n))
        means.append(d[idx].mean(1))
    means = np.concatenate(means)
    return np.quantile(means, [.025, .975], axis=0)


def pair(a, b, CA, CB, preds, tert):
    has = CA['has']
    assert np.array_equal(has, CB['has'])
    diff = np.concatenate([CA['contrib'] - CB['contrib'], (CA['contrib'] - CB['contrib']).sum(1, keepdims=True)], 1)
    ci = boot(diff, has, np.random.default_rng(2024))
    keys = G + ['total']
    ha, hb = CA['rel_e'].any(1)[has], CB['rel_e'].any(1)[has]
    pa, pb = preds[a], preds[b]
    jac = np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(pa, pb)])
    tot = diff[:, -1]
    return dict(variant=a, base=b,
                diff={k: float(diff[has, i].mean()) for i, k in enumerate(keys)},
                ci95={k: ci[:, i].tolist() for i, k in enumerate(keys)},
                ev_hit_users=dict(both=float((ha & hb).mean()), only_variant=float((ha & ~hb).mean()),
                                  only_base=float((~ha & hb).mean())),
                list_jaccard_mean=float(jac),
                total_diff_by_tercile={t: [float(tot[has & (lab == k)].mean()) for k in range(3)]
                                       for t, lab in tert.items()})


MODELS = {
    'H0_best': 'runs/H0/test/best/per_sample.jsonl.gz', 'H0_final': 'runs/H0/test/final/per_sample.jsonl.gz',
    'H1_best': 'runs/H1/test/best/per_sample.jsonl.gz', 'H1_final': 'runs/H1/test/final/per_sample.jsonl.gz',
    'CF-M0_best': 'cf_repeat_v2/runs/CF-M0/test/best/predictions.jsonl.gz',
    'CF-M0_final': 'cf_repeat_v2/runs/CF-M0/test/final/predictions.jsonl.gz',
    'CF-M1_best': 'cf_repeat_v2/runs/CF-M1/test/best/predictions.jsonl.gz',
    'CF-M1_final': 'cf_repeat_v2/runs/CF-M1/test/final/predictions.jsonl.gz',
    'T0A_best': 'runs/T0A/test/step_0166284/predictions.jsonl.gz',
    'T0A_final': 'runs/T0A/test/step_0200000/predictions.jsonl.gz',
    'T1A_best': 'runs/T1A/test/step_0193998/predictions.jsonl.gz',
    'T1A_final': 'runs/T1A/test/step_0200000/predictions.jsonl.gz',
}
PAIRS = [('H1_best', 'H0_best'), ('H1_final', 'H0_final'), ('CF-M1_best', 'CF-M0_best'),
         ('CF-M1_final', 'CF-M0_final'), ('T1A_best', 'T0A_best'), ('T1A_final', 'T0A_final'),
         ('H0_final', 'H0_best'), ('CF-M1_final', 'CF-M1_best'), ('T0A_final', 'T0A_best'),
         ('T1A_final', 'T1A_best')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--l5o', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--wave', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {}
    old, *_ = protocol_data('old', a.old, ('validation', 'test'))
    result['old_data'] = old
    new, S, E, D, per_base = protocol_data('l5o', a.l5o, ('validation', 'test'))
    result['l5o_data'] = new
    e = E['test']
    tert_time, q_time = terciles(D.day(e['tts'][:, 0]))
    act = np.array([S['user_targets'].get(int(u), 0) for u in e['users']])
    tert_act, q_act = terciles(act)
    tert = {'test_time': tert_time, 'user_activity': tert_act}
    result['terciles'] = dict(test_day_cuts=q_time, activity_cuts=q_act)
    has = np.array([any(w >= 1.0 for x, w in zip(r, wr) if x > 0) for r, wr in zip(e['tgt'], e['wr'])])
    result['l5o_baselines_by_tercile'] = {
        kind: {t: [float(np.asarray(per_base[('test', kind)]['ev_ndcg@5'])[has & (lab == k)].mean())
                   for k in range(3)] for t, lab in tert.items()}
        for kind in ('global', 'past7', 'win1')}
    preds, C, prof = {}, {}, {}
    for name, rel in MODELS.items():
        preds[name] = load_predictions(a.wave / rel, e['ids'])
        C[name] = class_contrib(preds[name], e, S)
        prof[name] = model_profile(preds[name], e, S, D, C[name], tert)
        prof[name]['metrics_check'] = score(preds[name], e)[0]
    result['models'] = prof
    result['pairs'] = [pair(x, y, C[x], C[y], preds, tert) for x, y in PAIRS]
    result['scope'] = 'Read-only; existing datasets and test predictions; no training or inference.'
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
