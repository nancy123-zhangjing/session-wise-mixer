"""Read-only checks: why does the Mixer effect differ between the old global-time split and L5O v2?

Runs on a CPU notebook next to Inspire storage. Reads existing datasets, per-sample
predictions and validation summaries only; no training, decoding, or writes outside --output.

Stages (one JSON each):
  old   Old split. H freeze pairs (3 seeds x best/R35) and the TIGER pairs re-scored with the old
        exposure NDCG (gate: equals the stored per-sample list/ndcg@5 for H and the summary for
        TIGER), EV-NDCG@5 at watch ratio >= 1 and >= 2, and EV Hit@5, on the stored (unfiltered)
        lists and on history-shifted lists (approximate filter). EV contribution by item class,
        watch ratio of hit targets. User-cluster bootstrap (the old test has many samples per user).
  v2    L5O protocol v2, family h or tiger. Per seed and selection node (smoothed/best/final):
        Mixer minus no-Mixer for EV(1.0), EV(2.0), exposure NDCG and Hit; overall and by subgroup
        (test-start day <=71 / >=72, users whose test starts inside the old test window, users whose
        targets are / are not among the same-day hot training targets, test-time and activity
        terciles), the pre-registered contrast Delta(>=72) - Delta(<=71), the same-day-volume and
        day-hot-share trends, EV contribution by item class,
        recommendation profiles, hits on watched (>=1) versus not-watched (<1) targets.
  traj  Validation trajectories of Mixer minus no-Mixer: v2 family h (per pass) or tiger (per
        validation step); the old H freeze trajectories from the accepted CSV.

Intervals: 95 % bootstrap over users (2,000 draws, seed 2024), within one training seed.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.kuairec_data.l5o_metrics import load_predictions, slate_metrics

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/09_kuairec_listwise'
K = 5
DRAWS = 2000
METRICS = ('ev1_ndcg', 'ev2_ndcg', 'ev1_hit', 'imp_ndcg', 'imp_hit')
SEEDS = (2024, 2025, 2026)
ARMS = {'h': ('H0', 'H1'), 'tiger': ('T0A', 'T1A')}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def profile_module():
    return module(EXP / 'l5o_deep_profile_20261117/analyze.py', 'l5o_deep_profile')


def old_module():
    return module(EXP / 'history_repeat_old_20261117/analyze.py', 'history_repeat_old')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def per_sample(pred, tgt, wr):
    """Per-sample metric arrays; NaN where a user has no eligible EV target."""
    _, p1 = slate_metrics(pred, tgt, wr, 1.0, return_per_sample=True)
    _, p2 = slate_metrics(pred, tgt, wr, 2.0, return_per_sample=True)
    return {'ev1_ndcg': p1['ev_ndcg@5'], 'ev2_ndcg': p2['ev_ndcg@5'], 'ev1_hit': p1['ev_hit@5'],
            'imp_ndcg': p1['imp_ndcg@5'], 'imp_hit': p1['imp_hit@5'], 'legacy_ndcg': p1['legacy_fixed5_ndcg@5']}


def cluster_boot(diff, uidx, nu, mask=None, seed=2024, draws=None):
    """Mean of diff over finite (and masked) samples; user-cluster bootstrap interval."""
    draws = draws or DRAWS
    ok = np.isfinite(diff) if mask is None else (np.isfinite(diff) & mask)
    if not ok.any():
        return None
    S = np.bincount(uidx[ok], weights=diff[ok], minlength=nu)
    N = np.bincount(uidx[ok], minlength=nu).astype(float)
    rng = np.random.default_rng(seed)
    est = np.empty(draws)
    for d in range(draws):
        w = np.bincount(rng.integers(0, nu, nu), minlength=nu)
        den = w @ N
        est[d] = (w @ S) / den if den else np.nan
    return dict(delta=float(diff[ok].mean()), ci95=np.nanquantile(est, [.025, .975]).tolist(), n=int(ok.sum()))


def user_boot(diff, mask=None, seed=2024, draws=None):
    """One sample per user (L5O): plain bootstrap over eligible users."""
    draws = draws or DRAWS
    ok = np.isfinite(diff) if mask is None else (np.isfinite(diff) & mask)
    d = diff[ok]
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(len(d), size=(min(200, draws - s), len(d)))].mean(1)
                            for s in range(0, draws, 200)])
    return dict(delta=float(d.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), n=int(len(d)))


def contrast_boot(diff, a, b, seed=2024, draws=None):
    """mean(diff | a) - mean(diff | b), resampling users jointly."""
    draws = draws or DRAWS
    ok = np.isfinite(diff)
    ia, ib = np.flatnonzero(ok & a), np.flatnonzero(ok & b)
    if not len(ia) or not len(ib):
        return None
    idx = np.flatnonzero(ok & (a | b))
    rng = np.random.default_rng(seed)
    est = np.empty(draws)
    for d in range(draws):
        s = idx[rng.integers(len(idx), size=len(idx))]
        sa, sb = s[a[s]], s[b[s]]
        est[d] = diff[sa].mean() - diff[sb].mean() if len(sa) and len(sb) else np.nan
    return dict(contrast=float(diff[ia].mean() - diff[ib].mean()), ci95=np.nanquantile(est, [.025, .975]).tolist(),
                n_a=int(len(ia)), n_b=int(len(ib)))


def slope_boot(diff, x, seed=2024, draws=None):
    draws = draws or DRAWS
    ok = np.isfinite(diff) & np.isfinite(x)
    d, xv = diff[ok], x[ok]
    if len(d) < 3:
        return None

    def fit(dd, xx):
        xc = xx - xx.mean()
        den = (xc * xc).sum()
        return float((xc * (dd - dd.mean())).sum() / den) if den else np.nan
    rng = np.random.default_rng(seed)
    est = np.array([fit(d[i], xv[i]) for i in (rng.integers(len(d), size=len(d)) for _ in range(draws))])
    return dict(slope=fit(d, xv), ci95=np.nanquantile(est, [.025, .975]).tolist(), n=int(len(d)))


def hit_watch(pred, tgt, wr):
    """Watch ratio of the targets each list hits (exposure hits)."""
    hits_wr = []
    per_user_low, per_user_high = np.zeros(len(pred)), np.zeros(len(pred))
    for i, (row, t, w) in enumerate(zip(pred, tgt, wr)):
        ps = set(int(x) for x in row if x > 0)
        for x, v in zip(t, w):
            if x > 0 and int(x) in ps:
                hits_wr.append(v)
                if v >= 1.0:
                    per_user_high[i] += 1
                else:
                    per_user_low[i] += 1
    h = np.asarray(hits_wr, float)
    return dict(hits=int(len(h)), mean_wr=float(h.mean()) if len(h) else None,
                share_ge1=float((h >= 1).mean()) if len(h) else None,
                share_ge2=float((h >= 2).mean()) if len(h) else None), per_user_low, per_user_high


def shifted(pred, hist):
    out = np.zeros_like(pred)
    for i, (row, hs) in enumerate(zip(pred, hist)):
        keep = [int(x) for x in row if int(x) not in hs][:K]
        out[i, :len(keep)] = keep
    return out


# ----------------------------------------------------------------------------------- old split

OLD_REPORTED = {  # accepted FINAL_REPORT.md (old freeze), test NDCG@5
    ('2024', 'R35'): (0.008256, 0.008956), ('2025', 'R35'): (0.004996, 0.007113), ('2026', 'R35'): (0.006905, 0.008336),
    ('2024', 'best'): (0.008471, 0.009431), ('2025', 'best'): (0.005890, 0.007079), ('2026', 'best'): (0.008112, 0.008604)}


def stage_old(a):
    P, O = profile_module(), old_module()
    data = a.base / 'goal_20261027_r1/inputs/dataset'
    S = P.train_stats(data / 'train.parquet')
    t = pq.read_table(data / 'test.parquet', columns=['sample_id', 'user_id', 'history_item_ids', 'history_mask',
                                                       'target_item_ids', 'target_watch_ratio']).to_pydict()
    ids = t['sample_id']
    tgt = np.asarray(t['target_item_ids'], np.int64)
    wr = np.asarray(t['target_watch_ratio'], float)
    hist = [set(int(i) for i, m in zip(h, mk) if m and i > 0) for h, mk in zip(t['history_item_ids'], t['history_mask'])]
    _, uidx = np.unique(np.asarray(t['user_id'], np.int64), return_inverse=True)
    nu = int(uidx.max()) + 1
    E = dict(tgt=tgt, wr=wr)
    rerun = getattr(a, 'rerun_root', None)
    nodes = O.NODES
    if rerun is not None:
        # Step-1 control (2026-11-19): current code on the old split; best = validation best, R35 = final.
        nodes = {f'H{s}_H{k}F_{n}_test': ('h', 'test', str(rerun / f'h/seed{s}/H{k}/test/{d}'))
                 for s in SEEDS for k in (0, 1) for n, d in (('best', 'best'), ('R35', 'final'))}
    res = dict(scope='read-only; old global-time split test; stored (unfiltered) lists and a history-shift approximation'
               + ('' if rerun is None else f'; step-1 rerun with current code under {rerun}'),
               dataset=str(data), test_sha256=sha(data / 'test.parquet'), samples=len(ids), users=nu,
               target_wr_share_ge1=float((wr[tgt > 0] >= 1).mean()), target_wr_share_ge2=float((wr[tgt > 0] >= 2).mean()),
               samples_with_ev1=float(((wr >= 1) & (tgt > 0)).any(1).mean()), nodes={}, pairs={})
    cache = {}

    def node(name):
        if name in cache:
            return cache[name]
        kind, split, rel = nodes[name]
        assert split == 'test'
        path = a.base / rel / ('per_sample.jsonl.gz' if kind == 'h' else 'predictions.jsonl.gz')
        pred = load_predictions(path, ids)
        ps = per_sample(pred, tgt, wr)
        if kind == 'h':
            stored = np.full(len(ids), np.nan)
            index = {s: i for i, s in enumerate(ids)}
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    stored[index[r['sample_id']]] = r['metrics']['list/ndcg@5']
            gate = float(np.max(np.abs(ps['legacy_ndcg'] - stored)))
        else:
            ref = json.loads((a.base / rel / 'summary.json').read_text())['metrics']['ndcg@5']
            gate = abs(float(ps['legacy_ndcg'].mean()) - float(ref))
        if gate > 1e-6:
            raise ValueError(f'{name}: old NDCG not reproduced ({gate})')
        sh = shifted(pred, hist)
        pss = per_sample(sh, tgt, wr)
        C = P.class_contrib(pred, dict(tgt=tgt, wr=wr), S)
        hw, low, high = hit_watch(pred, tgt, wr)
        res['nodes'][name] = dict(file=str(path), sha256=sha(path), gate_max_abs_err=gate,
                                  stored={k: float(np.nanmean(v)) for k, v in ps.items()},
                                  shifted={k: float(np.nanmean(v)) for k, v in pss.items()},
                                  ev1_contribution={g: float(x) for g, x in zip(P.G, C['contrib'][C['has']].mean(0))},
                                  hit_watch=hw)
        cache[name] = (ps, pss, C, low, high)
        return cache[name]

    pairs = [(f'H{s} {n}', f'H{s}_H1F_{n}_test', f'H{s}_H0F_{n}_test') for s in map(str, SEEDS)
             for n in ('best', 'R35')]
    pairs += [] if rerun is not None else [('TIGER best', 'TIGER_T1A_final_test', 'TIGER_T0A_best_test'),
              ('TIGER final', 'TIGER_T1A_final_test', 'TIGER_T0A_final_test')]
    for label, mx, base in pairs:
        (pm, pms, Cm, lm, hm), (pb, pbs, Cb, lb, hb) = node(mx), node(base)
        row = {'stored': {}, 'shifted': {}}
        for k in METRICS + ('legacy_ndcg',):
            row['stored'][k] = cluster_boot(pm[k] - pb[k], uidx, nu)
            row['shifted'][k] = cluster_boot(pms[k] - pbs[k], uidx, nu)
        has = Cm['has']
        cdiff = (Cm['contrib'] - Cb['contrib'])
        row['ev1_contribution_diff'] = {g: cluster_boot(np.where(has, cdiff[:, j], np.nan), uidx, nu)
                                        for j, g in enumerate(P.G)}
        row['hits_on_wr_lt1_diff'] = cluster_boot(lm - lb, uidx, nu)
        row['hits_on_wr_ge1_diff'] = cluster_boot(hm - hb, uidx, nu)
        if label.startswith('H') and rerun is not None:
            s, n = label[1:5], label.split()[1]
            row['accepted_old_run'] = dict(H0F=OLD_REPORTED[(s, n)][0], H1F=OLD_REPORTED[(s, n)][1],
                                           delta=OLD_REPORTED[(s, n)][1] - OLD_REPORTED[(s, n)][0])
        elif label.startswith('H'):
            s, n = label[1:5], label.split()[1]
            rep = OLD_REPORTED[(s, n)]
            row['reported_check'] = dict(reported=rep, recomputed=[res['nodes'][base]['stored']['legacy_ndcg'],
                                                                   res['nodes'][mx]['stored']['legacy_ndcg']])
            if max(abs(rep[0] - row['reported_check']['recomputed'][0]),
                   abs(rep[1] - row['reported_check']['recomputed'][1])) > 6e-7:
                raise ValueError(f'{label}: does not match the accepted report')
        res['pairs'][label] = row
    if rerun is not None:
        res['validation'] = rerun_validation(rerun)
    return res


def rerun_validation(rerun):
    """Validation exposure NDCG@5 per R of the step-1 rerun (shared warmup + each arm) and the
    accepted old freeze curves of the same seeds, for the trajectory comparison."""
    old = {}
    with open(EXP / 'h0_multiseed_r20_freeze_20261110_v1/VALIDATION_COMPARISON.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            old.setdefault(r['seed'], {})[int(r['epoch'])] = float(r['H1_freeze']) - float(r['H0_freeze'])
    out = {}
    for s in SEEDS:
        curves = {}
        for arm in ('H0', 'H1'):
            pts = {}
            for d in [*(rerun / f'h/seed{s}/warmup').glob('validation_R*'), *(rerun / f'h/seed{s}/{arm}').glob('validation_R*')]:
                if '_retry' in d.name or not (d / 'summary.json').is_file():
                    continue
                pts[int(d.name.split('_R')[1][:2])] = json.loads((d / 'summary.json').read_text())['metrics']['list/ndcg@5']
            curves[arm] = pts
        common = sorted(set(curves['H0']) & set(curves['H1']))
        out[str(s)] = [dict(R=r, H0F=curves['H0'][r], H1F=curves['H1'][r], delta=curves['H1'][r] - curves['H0'][r],
                            accepted_old_delta=old.get(str(s), {}).get(r)) for r in common]
    return out


# ------------------------------------------------------------------------------------------ v2

def v2_nodes(root, family, seed, arm):
    """{node: predictions file} for one arm; plus the recorded test EV-NDCG@5 for the gate."""
    base = root / family / f'seed{seed}' / arm
    if family == 'h':
        out = {}
        for n in ('smoothed', 'best', 'final'):
            d = base / 'test' / n
            out[n] = (d / 'per_sample.jsonl.gz', json.loads((d / 'summary.json').read_text())['metrics']['ev_ndcg@5'])
        return out
    complete = json.loads((base / 'test' / 'complete.json').read_text())
    by_sha = {}
    for d in sorted((base / 'test').glob('step_*')):
        s = json.loads((d / 'summary.json').read_text())
        by_sha[s['prediction_sha256']] = d
    out = {}
    for n in ('smoothed', 'best', 'final'):
        d = by_sha[complete[n]['prediction_sha256']]
        out[n] = (d / 'predictions.jsonl.gz', complete[n]['metrics']['ev_ndcg@5'])
    return out


def stage_v2(a):
    P = profile_module()
    data = a.base / 'l5o_stage0_20261117_v1/dataset'
    old_data = a.base / 'goal_20261027_r1/inputs/dataset'
    S = P.train_stats(data / 'train.parquet')
    E = {s: P.load_eval(data / f'{s}.parquet') for s in ('validation', 'test')}
    t0 = min(int(S['ts'].min()), *(int(e['tts'][e['tts'] > 0].min()) for e in E.values()))
    D = P.Days(S, t0)
    e = E['test']
    tgt, wr = e['tgt'], e['wr']
    d0 = D.day(e['tts'][:, 0])
    first_ts = e['tts'][:, 0]
    old_ts = pq.read_table(old_data / 'test.parquet', columns=['target_timestamp_us'])['target_timestamp_us']
    old_flat = old_ts.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.int64)
    old_flat = old_flat[old_flat > 0]
    old_lo, old_hi = int(old_flat.min()), int(old_flat.max())
    hist_len = np.array([len(h) for h in e['hist']])
    act = np.array([S['user_targets'].get(int(u), 0) for u in e['users']])
    tert_time, q_time = P.terciles(d0)
    tert_act, q_act = P.terciles(act)
    tert = {'test_time': tert_time, 'user_activity': tert_act}
    # Share of each user's five test targets that are among the top 100 training targets of the
    # +/-1 day window around the test start (other users' same-day interactions are in L5O training).
    hot_share = np.zeros(len(tgt))
    for i, (row, d) in enumerate(zip(tgt, d0)):
        r = D.ranked(('win1', int(d)), int(d) - 1, int(d) + 1)
        top = set(r[:100].tolist()) if r is not None else set()
        items = [int(x) for x in row if x > 0]
        hot_share[i] = sum(x in top for x in items) / len(items)
    groups = {'day_le71': d0 <= 71, 'day_ge72': d0 >= 72,
              'old_test_window': (first_ts >= old_lo) & (first_ts <= old_hi),
              'outside_old_window': ~((first_ts >= old_lo) & (first_ts <= old_hi)),
              'targets_dayhot_none': hot_share == 0, 'targets_dayhot_some': hot_share > 0,
              'targets_dayhot_ge40pct': hot_share >= 0.4,
              **{f'test_time_t{k + 1}': tert_time == k for k in range(3)},
              **{f'activity_t{k + 1}': tert_act == k for k in range(3)}}
    volume = np.log1p(D.mat[np.clip(d0, 0, D.n - 1)].sum(1).astype(float))
    res = dict(scope='read-only; L5O protocol v2 test predictions', family=a.family, dataset=str(data),
               day_origin_us=int(t0), old_test_window_us=[old_lo, old_hi],
               group_sizes={g: int(m.sum()) for g, m in groups.items()},
               ev1_users=int(np.isfinite(per_sample(np.zeros_like(tgt), tgt, wr)['ev1_ndcg']).sum()),
               cuts=dict(test_day_terciles=q_time, activity_terciles=q_act),
               history_length=dict(min=int(hist_len.min()), quantiles=np.quantile(hist_len, [.05, .25, .5, .75]).tolist(),
                                   note='every L5O test user has more than 50 history items; length strata are not testable here'),
               target_dayhot_share=dict(mean=float(hot_share.mean()), users_none=int((hot_share == 0).sum())),
               seeds={})
    base_arm, mix_arm = ARMS[a.family]
    for seed in SEEDS:
        nb, nm = v2_nodes(a.root, a.family, seed, base_arm), v2_nodes(a.root, a.family, seed, mix_arm)
        res['seeds'][str(seed)] = {}
        for n in ('smoothed', 'best', 'final'):
            pb, pm = load_predictions(nb[n][0], e['ids']), load_predictions(nm[n][0], e['ids'])
            sb, sm = per_sample(pb, tgt, wr), per_sample(pm, tgt, wr)
            for arm, s, rec in ((base_arm, sb, nb[n][1]), (mix_arm, sm, nm[n][1])):
                if abs(float(np.nanmean(s['ev1_ndcg'])) - float(rec)) > 1e-9:
                    raise ValueError(f'{a.family} {seed} {arm} {n}: EV-NDCG@5 not reproduced')
            row = dict(files={base_arm: str(nb[n][0]), mix_arm: str(nm[n][0])},
                       base={k: float(np.nanmean(v)) for k, v in sb.items()},
                       mixer={k: float(np.nanmean(v)) for k, v in sm.items()},
                       overall={k: user_boot(sm[k] - sb[k]) for k in METRICS}, groups={}, contrasts={})
            for g, m in groups.items():
                row['groups'][g] = {k: user_boot(sm[k] - sb[k], m) for k in ('ev1_ndcg', 'imp_ndcg', 'imp_hit')}
            for k in ('ev1_ndcg', 'imp_ndcg'):
                diff = sm[k] - sb[k]
                row['contrasts'][k] = dict(
                    day_ge72_minus_le71=contrast_boot(diff, groups['day_ge72'], groups['day_le71']),
                    old_window_minus_outside=contrast_boot(diff, groups['old_test_window'], groups['outside_old_window']),
                    dayhot_some_minus_none=contrast_boot(diff, groups['targets_dayhot_some'], groups['targets_dayhot_none']),
                    same_day_volume_slope=slope_boot(diff, volume),
                    dayhot_share_slope=slope_boot(diff, hot_share))
            Cb, Cm = P.class_contrib(pb, e, S), P.class_contrib(pm, e, S)
            row['ev1_class_pair'] = P.pair('mixer', 'base', Cm, Cb, {'mixer': pm, 'base': pb}, tert)
            row['profiles'] = {base_arm: P.model_profile(pb, e, S, D, Cb, tert), mix_arm: P.model_profile(pm, e, S, D, Cm, tert)}
            hb, lb, hhb = hit_watch(pb, tgt, wr)
            hm, lm, hhm = hit_watch(pm, tgt, wr)
            row['hit_watch'] = {base_arm: hb, mix_arm: hm,
                                'hits_on_wr_lt1_diff': user_boot(lm - lb), 'hits_on_wr_ge1_diff': user_boot(hhm - hhb)}
            res['seeds'][str(seed)][n] = row
    res['three_seed'] = {}
    for n in ('smoothed', 'best', 'final'):
        summary = {}
        keys = [('overall', k) for k in METRICS] + [('groups', g, k) for g in groups for k in ('ev1_ndcg', 'imp_ndcg')]
        for key in keys:
            vals = []
            for seed in SEEDS:
                x = res['seeds'][str(seed)][n]
                for part in key:
                    x = x[part]
                vals.append(x['delta'] if x else np.nan)
            v = np.asarray(vals, float)
            summary['/'.join(key)] = dict(per_seed=vals, mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                                          positive=int((v > 0).sum()))
        for k in ('ev1_ndcg', 'imp_ndcg'):
            for c in ('day_ge72_minus_le71', 'old_window_minus_outside', 'dayhot_some_minus_none'):
                got = [res['seeds'][str(s)][n]['contrasts'][k][c] for s in SEEDS]
                vals = [g['contrast'] if g else np.nan for g in got]
                summary[f'contrast/{k}/{c}'] = dict(per_seed=vals, mean=float(np.nanmean(vals)) if any(g for g in got) else None,
                                                    positive=int(sum(v > 0 for v in vals if np.isfinite(v))))
        res['three_seed'][n] = summary
    return res


# -------------------------------------------------------------------------------- trajectories

def stage_traj(a):
    res = dict(scope='read-only; validation summaries', family=a.family, seeds={})
    base_arm, mix_arm = ARMS[a.family]
    keys = ('ev_ndcg@5', 'imp_ndcg@5', 'imp_hit@5')
    for seed in SEEDS:
        curves = {}
        for arm in (base_arm, mix_arm):
            pts = {}
            if a.family == 'h':
                for d in sorted((a.root / 'h' / f'seed{seed}' / arm).glob('validation_R*')):
                    pts[int(d.name.split('_R')[1])] = json.loads((d / 'summary.json').read_text())['metrics']
            else:
                for d in sorted((a.root / 'tiger' / f'seed{seed}' / arm / 'train' / 'validation').glob('step_*')):
                    pts[int(d.name.split('_')[1])] = json.loads((d / 'summary.json').read_text())['metrics']
            curves[arm] = pts
        common = sorted(set(curves[base_arm]) & set(curves[mix_arm]))
        rows = [dict(node=x, **{k: curves[mix_arm][x][k] - curves[base_arm][x][k] for k in keys},
                     base_ev=curves[base_arm][x]['ev_ndcg@5'], base_imp_ndcg=curves[base_arm][x]['imp_ndcg@5'])
                for x in common]
        out = dict(points=rows)
        if a.family == 'h':
            for name, lo, hi in (('joint_R6_20', 6, 20), ('frozen_R21_35', 21, 35)):
                sel = [r for r in rows if lo <= r['node'] <= hi]
                out[name] = {k: dict(mean=float(np.mean([r[k] for r in sel])) if sel else None,
                                     positive=sum(r[k] > 0 for r in sel), n=len(sel)) for k in keys}
            post = [r for r in rows if r['node'] >= 20]
            if len(post) >= 3:
                x = np.array([r['node'] for r in post], float)
                out['frozen_slope_per_pass'] = {k: float(np.polyfit(x, [r[k] for r in post], 1)[0]) for k in keys}
        else:
            n = len(rows)
            for name, sel in (('first_third', rows[:n // 3]), ('middle_third', rows[n // 3:2 * n // 3]),
                              ('last_third', rows[2 * n // 3:])):
                out[name] = {k: dict(mean=float(np.mean([r[k] for r in sel])) if sel else None,
                                     positive=sum(r[k] > 0 for r in sel), n=len(sel)) for k in keys}
        res['seeds'][str(seed)] = out
    if a.family == 'h':
        old = {}
        with open(EXP / 'h0_multiseed_r20_freeze_20261110_v1/VALIDATION_COMPARISON.csv', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                old.setdefault(r['seed'], []).append(dict(R=int(r['epoch']),
                                                          delta=float(r['H1_freeze']) - float(r['H0_freeze']),
                                                          relative_pct=float(r['H1_freeze_vs_H0_freeze_pct'])))
        res['old_split_exposure_ndcg'] = old
    return res


def clean(o):
    """JSON-safe: NaN/inf become null; numpy scalars become Python numbers."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def main():
    global DRAWS, SEEDS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['old', 'v2', 'traj'])
    p.add_argument('--base', type=Path, default=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'))
    p.add_argument('--root', type=Path, help='protocol v2 root (stages v2, traj)')
    p.add_argument('--rerun-root', type=Path, help='stage old only: score the step-1 rerun (current code, old split) instead')
    p.add_argument('--family', choices=['h', 'tiger'])
    p.add_argument('--draws', type=int, default=2000, help='bootstrap draws (smaller only for code tests)')
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS), help='training seeds (fewer only for code tests)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    DRAWS, SEEDS = a.draws, tuple(a.seeds)
    if a.stage in ('v2', 'traj') and (a.root is None or a.family is None):
        p.error('v2 and traj need --root and --family')
    res = {'old': stage_old, 'v2': stage_v2, 'traj': stage_traj}[a.stage](a)
    res['draws'], res['seeds_used'] = DRAWS, list(SEEDS)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(res), indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(a.output)
    print(json.dumps(dict(stage=a.stage, family=a.family, output=str(a.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
