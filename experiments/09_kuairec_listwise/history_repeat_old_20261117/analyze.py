"""Read-only: how often old-protocol (unfiltered) predictions recommend items from the
user's own input history, and how a history filter would move the Mixer delta.

Runs on a CPU notebook next to Inspire storage. Reads the old global-time dataset and
existing per-sample predictions only; no training, inference or writes outside --output.

Old protocol evaluated without a history filter. For every stored prediction set:
  * repeat rate: share of the main 5-item list (rank-0 list) that lies in the sample's
    input history (history_item_ids where history_mask), overall and per slot;
  * which item classes the repeated items belong to (Stage09 rule);
  * NDCG@5 (old formula, fixed IDCG of 5) and the part earned by history items;
  * two filtered approximations, not a substitute for re-running constrained decoding:
      - shift: drop history items from the list and move the remaining items up
        (no replacement; a lower-bound style estimate);
      - backfill (H only, 32 stored lists): walk the stored lists in score order,
        dedupe, skip history items, take the first 5.
Paired Mixer comparisons report the delta under each variant and the
difference-in-differences (filtered delta - unfiltered delta), with a user-cluster
bootstrap (2000 draws, seed 2024). The interval covers user sampling within one
training seed only.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

K = 5
NITEM = 10728
DISC = 1 / np.log2(np.arange(K) + 2)
IDCG = DISC.sum()
G = ['never_seen', 'history_only', 'head', 'mid', 'tail']
DRAWS = 2000

TG = 'session_wise_ar_tiger_adamw_20261114_v1/runs/'
HF = 'h_freeze35_20261029_r1/runs/'
MS = 'multiseed_freeze_20261109_v1/'
H0M = 'h0_multiseed_r20_freeze_20261110_v1/'
CF0 = 'session_wise_ar_five_20261113_v1/runs/CF-M0/train/'
CF1 = 'rqcd_targets_formal_20261113_v1/cf/'
CF0F = 'cf_freeze_r9_20261114_v1/runs/M0/train/'
CF1F = 'cf_freeze_r9_20261114_v1/runs/M1/train/'

# name: (kind, split, relative directory). Nodes follow the accepted reports
# (best chosen on validation; TIGER T1A best == final == step 200000).
NODES = {
    'TIGER_T0A_best_test': ('tiger', 'test', TG + 'AR-T0A/test/step_0193998'),
    'TIGER_T0A_final_test': ('tiger', 'test', TG + 'AR-T0A/test/step_0200000'),
    'TIGER_T1A_final_test': ('tiger', 'test', TG + 'AR-T1A/test/step_0200000'),
    'TIGER_T0A_best_val': ('tiger', 'validation', TG + 'AR-T0A/train/validation/step_0193998'),
    'TIGER_T0A_final_val': ('tiger', 'validation', TG + 'AR-T0A/train/validation/step_0200000'),
    'TIGER_T1A_final_val': ('tiger', 'validation', TG + 'AR-T1A/train/validation/step_0200000'),
    'H2024_H0F_R35_test': ('h', 'test', HF + 'H0-Freeze20-R35/evaluations/test_R35/attempt1'),
    'H2024_H1F_R35_test': ('h', 'test', HF + 'H1-Freeze20-R35/evaluations/test_R35/attempt1'),
    'H2024_H0F_R35_val': ('h', 'validation', HF + 'H0-Freeze20-R35/evaluations/validation_R35/attempt1'),
    'H2024_H1F_R35_val': ('h', 'validation', HF + 'H1-Freeze20-R35/evaluations/validation_R35/attempt1'),
    'H2024_H0F_best_test': ('h', 'test', HF + 'H0-Freeze20-R35/evaluations/test_R23/attempt1'),
    'H2024_H1F_best_test': ('h', 'test', HF + 'H1-Freeze20-R35/evaluations/test_R32/attempt1'),
    'H2024_H0F_best_val': ('h', 'validation', HF + 'H0-Freeze20-R35/evaluations/validation_R23/attempt1'),
    'H2024_H1F_best_val': ('h', 'validation', HF + 'H1-Freeze20-R35/evaluations/validation_R32/attempt1'),
    'H2025_H0F_R35_test': ('h', 'test', H0M + 'seed2025_H0/test_R35'),
    'H2025_H1F_R35_test': ('h', 'test', MS + 'seed2025_H1/test_R35'),
    'H2025_H0F_R35_val': ('h', 'validation', H0M + 'seed2025_H0/validation_R35'),
    'H2025_H1F_R35_val': ('h', 'validation', MS + 'seed2025_H1/validation_R35'),
    'H2025_H0F_best_test': ('h', 'test', H0M + 'seed2025_H0/test_best'),
    'H2025_H1F_best_test': ('h', 'test', MS + 'seed2025_H1/test_best'),
    'H2025_H0F_best_val': ('h', 'validation', H0M + 'seed2025_H0/validation_R26'),
    'H2025_H1F_best_val': ('h', 'validation', MS + 'seed2025_H1/validation_R32'),
    'H2026_H0F_R35_test': ('h', 'test', H0M + 'seed2026_H0/test_R35'),
    'H2026_H1F_R35_test': ('h', 'test', MS + 'seed2026_H1/test_R35'),
    'H2026_H0F_R35_val': ('h', 'validation', H0M + 'seed2026_H0/validation_R35'),
    'H2026_H1F_R35_val': ('h', 'validation', MS + 'seed2026_H1/validation_R35'),
    'H2026_H0F_best_test': ('h', 'test', H0M + 'seed2026_H0/test_best'),
    'H2026_H1F_best_test': ('h', 'test', MS + 'seed2026_H1/test_best'),
    'H2026_H0F_best_val': ('h', 'validation', MS + 'seed2026_H0/validation_R20'),
    'H2026_H1F_best_val': ('h', 'validation', MS + 'seed2026_H1/validation_R29'),
    'CF_M0F_R35_test': ('cf', 'test', CF0F + 'test/final'),
    'CF_M1F_R35_test': ('cf', 'test', CF1F + 'test/final'),
    'CF_M0F_R35_val': ('cf', 'validation', CF0F + 'validation/R35'),
    'CF_M1F_R35_val': ('cf', 'validation', CF1F + 'validation/R35'),
    'CF_M0_best_test': ('cf', 'test', CF0 + 'test/best'),
    'CF_M1_best_test': ('cf', 'test', CF1 + 'test/best'),
    'CF_M0_best_val': ('cf', 'validation', CF0 + 'validation/R09'),
    'CF_M1_best_val': ('cf', 'validation', CF1 + 'validation/R09'),
}
PAIRS = [  # (label, mixer node, no-mixer node)
    ('TIGER best test', 'TIGER_T1A_final_test', 'TIGER_T0A_best_test'),
    ('TIGER final test', 'TIGER_T1A_final_test', 'TIGER_T0A_final_test'),
    ('TIGER best val', 'TIGER_T1A_final_val', 'TIGER_T0A_best_val'),
    ('TIGER final val', 'TIGER_T1A_final_val', 'TIGER_T0A_final_val'),
] + [(f'H{s} {n} {sp}', f'H{s}_H1F_{n}_{sp}', f'H{s}_H0F_{n}_{sp}')
     for s in ('2024', '2025', '2026') for n in ('R35', 'best') for sp in ('test', 'val')] + [
    ('CF freeze R35 test', 'CF_M1F_R35_test', 'CF_M0F_R35_test'),
    ('CF freeze R35 val', 'CF_M1F_R35_val', 'CF_M0F_R35_val'),
    ('CF best(R9) test', 'CF_M1_best_test', 'CF_M0_best_test'),
    ('CF best(R9) val', 'CF_M1_best_val', 'CF_M0_best_val'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def item_classes(path):
    tc = np.zeros(NITEM + 1, np.int64)
    hc = np.zeros(NITEM + 1, np.int64)
    for b in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=['history_item_ids', 'target_item_ids']):
        t = b.column('target_item_ids').flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        h = b.column('history_item_ids').flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        tc += np.bincount(t[t > 0], minlength=NITEM + 1)
        hc += np.bincount(h[h > 0], minlength=NITEM + 1)
    seen = (tc + hc) > 0
    seen[0] = False
    ranked = sorted((np.flatnonzero(tc[1:]) + 1).tolist(), key=lambda i: (-tc[i], i))
    cls = np.zeros(NITEM + 1, np.int8)
    cls[seen] = 1
    h, m = math.ceil(.1 * len(ranked)), math.ceil(.5 * len(ranked))
    cls[ranked[:h]] = 2
    cls[ranked[h:m]] = 3
    cls[ranked[m:]] = 4
    return cls, np.bincount(cls[1:], minlength=5).tolist()


def load_split(path):
    t = pq.read_table(path, columns=['sample_id', 'user_id', 'history_item_ids', 'history_mask',
                                     'target_item_ids']).to_pydict()
    n = len(t['sample_id'])
    hist = [set(int(i) for i, m in zip(h, mk) if m and i > 0) for h, mk in zip(t['history_item_ids'], t['history_mask'])]
    tgt = np.asarray(t['target_item_ids'], np.int64)
    users = np.asarray(t['user_id'], np.int64)
    _, uidx = np.unique(users, return_inverse=True)
    return dict(n=n, index={s: i for i, s in enumerate(t['sample_id'])}, hist=hist, targets=tgt, uidx=uidx,
                nu=int(uidx.max()) + 1,
                target_in_hist=float(np.mean([sum(int(x) in hs for x in row) / K for row, hs in zip(tgt, hist)])))


def ndcg(lst, targets):
    s = set(int(x) for x in targets)
    return sum(DISC[p] for p, it in enumerate(lst[:K]) if it in s) / IDCG


def load_node(kind, path, T):
    n = T['n']
    main = [None] * n
    lists = [None] * n if kind == 'h' else None
    stored = np.full(n, np.nan)
    fname = 'per_sample.jsonl.gz' if kind == 'h' else 'predictions.jsonl.gz'
    p = Path(path) / fname
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            i = T['index'][r['sample_id']]
            tg = r['targets'] if kind == 'h' else r['target']
            assert list(tg) == T['targets'][i].tolist(), (path, r['sample_id'])
            if kind == 'h':
                main[i] = [int(x) for x in r['predictions'][0]]
                keep = [l for l, c in zip(r['predictions'], r['complete_mask']) if c]
                lists[i] = [[int(x) for x in l] for l in keep]
                stored[i] = r['metrics']['list/ndcg@5']
            else:
                main[i] = [int(x) for x in r['prediction']]
                if kind == 'cf':
                    stored[i] = r['metrics']['ndcg@5']
    assert all(m is not None for m in main), path
    if kind == 'tiger':
        summ = json.loads((Path(path) / 'summary.json').read_text())
        ref_mean = float(summ['metrics']['ndcg@5'])
    else:
        ref_mean = None
    return p, main, lists, stored, ref_mean


def per_sample(kind, main, lists, T, cls):
    n = T['n']
    out = {k: np.zeros(n) for k in ('ndcg', 'ndcg_hist_part', 'ndcg_shift', 'ndcg_backfill', 'rep_frac', 'any_rep')}
    slot = np.zeros((n, K))
    rep_cls = np.zeros(5, np.int64)
    for i in range(n):
        hs, tg, m = T['hist'][i], T['targets'][i], main[i]
        s = set(int(x) for x in tg)
        flags = [it in hs for it in m[:K]]
        slot[i] = flags
        out['rep_frac'][i] = sum(flags) / K
        out['any_rep'][i] = float(any(flags))
        out['ndcg'][i] = ndcg(m, tg)
        out['ndcg_hist_part'][i] = sum(DISC[p] for p, it in enumerate(m[:K]) if it in s and it in hs) / IDCG
        for it, fl in zip(m[:K], flags):
            if fl and 0 < it <= NITEM:
                rep_cls[cls[it]] += 1
        out['ndcg_shift'][i] = ndcg([it for it in m[:K] if it not in hs], tg)
        if kind == 'h':
            seen, fill = set(), []
            for l in lists[i]:
                for it in l:
                    if it > 0 and it not in seen and it not in hs:
                        seen.add(it)
                        fill.append(it)
                    if len(fill) == K:
                        break
                if len(fill) == K:
                    break
            out['ndcg_backfill'][i] = ndcg(fill, tg)
        else:
            out['ndcg_backfill'][i] = np.nan
    return out, slot, rep_cls


def user_boot(values, uidx, nu, rng_seed=2024):
    """Mean over samples with user-cluster resampling; returns (point, lo, hi)."""
    S = np.bincount(uidx, weights=values, minlength=nu)
    N = np.bincount(uidx, minlength=nu).astype(np.float64)
    rng = np.random.default_rng(rng_seed)
    est = np.empty(DRAWS)
    for d in range(DRAWS):
        w = np.bincount(rng.integers(0, nu, nu), minlength=nu)
        est[d] = (w @ S) / (w @ N)
    return float(values.mean()), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data = a.base / 'goal_20261027_r1/inputs/dataset'
    res = {'scope': 'read-only; old global-time split; unfiltered stored predictions; '
                    'shift/backfill are approximations of a history filter, not re-decoding',
           'inputs': {f: sha(data / f) for f in ('train.parquet', 'validation.parquet', 'test.parquet')}}
    cls, counts = item_classes(data / 'train.parquet')
    res['class_counts'] = dict(zip(G, counts))
    T = {s: load_split(data / f'{s}.parquet') for s in ('validation', 'test')}
    res['target_in_history'] = {s: T[s]['target_in_hist'] for s in T}
    cache, res['nodes'] = {}, {}
    for name, (kind, split, rel) in NODES.items():
        p, main_l, lists, stored, ref_mean = load_node(kind, a.base / rel, T[split])
        v, slot, rep_cls = per_sample(kind, main_l, lists, T[split], cls)
        if kind in ('h', 'cf'):
            err = float(np.nanmax(np.abs(v['ndcg'] - stored)))
            assert err < 1e-6, (name, err)
        else:
            err = abs(float(v['ndcg'].mean()) - ref_mean)
            assert err < 1e-9, (name, err)
        cache[name] = v
        res['nodes'][name] = dict(
            file=str(p), sha256=sha(p), recompute_max_abs_err=err, samples=T[split]['n'],
            ndcg=float(v['ndcg'].mean()), ndcg_hist_part=float(v['ndcg_hist_part'].mean()),
            ndcg_shift=float(v['ndcg_shift'].mean()),
            ndcg_backfill=None if kind != 'h' else float(v['ndcg_backfill'].mean()),
            rep_frac=float(v['rep_frac'].mean()), any_rep=float(v['any_rep'].mean()),
            rep_by_slot=slot.mean(0).tolist(),
            rep_item_class_share=dict(zip(G, (rep_cls / max(rep_cls.sum(), 1)).tolist())))
        print(name, json.dumps({k: res['nodes'][name][k] for k in ('ndcg', 'ndcg_shift', 'ndcg_backfill', 'rep_frac')}),
              flush=True)
    res['pairs'] = []
    for label, mx, base in PAIRS:
        split = NODES[mx][1]
        assert NODES[base][1] == split
        A, B, t = cache[mx], cache[base], T[split]
        row = {'label': label, 'mixer': mx, 'no_mixer': base}
        variants = ['ndcg', 'ndcg_shift', 'rep_frac'] + (['ndcg_backfill'] if NODES[mx][0] == 'h' else [])
        for k in variants:
            row['delta_' + k] = user_boot(A[k] - B[k], t['uidx'], t['nu'])
        row['did_shift'] = user_boot((A['ndcg_shift'] - B['ndcg_shift']) - (A['ndcg'] - B['ndcg']), t['uidx'], t['nu'])
        if NODES[mx][0] == 'h':
            row['did_backfill'] = user_boot((A['ndcg_backfill'] - B['ndcg_backfill']) - (A['ndcg'] - B['ndcg']),
                                            t['uidx'], t['nu'])
        res['pairs'].append(row)
        print(label, json.dumps({k: [round(x, 6) for x in v] for k, v in row.items() if isinstance(v, tuple)}), flush=True)
    out = a.output / 'RESULTS.json'
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print('wrote', out, sha(out))


if __name__ == '__main__':
    main()
