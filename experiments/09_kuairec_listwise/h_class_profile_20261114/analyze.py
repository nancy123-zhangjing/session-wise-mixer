"""H0/H1 (PSE + DiffGRM) item-class profile, mirroring the CF analysis.

Question: in the original framework, does the frozen-codebook Mixer gain (H1 Freeze
vs H0 Freeze, both frozen at R20/T200) also come from mid-popularity items?

Read-only: existing per-sample predictions (highest-scoring single 5-item list, i.e.
predictions[0], the main-metric list), no training/inference, no remote writes.
Item classes identical to the Stage09 audit / CF analysis. Per seed and node we report
recommendation class shares, per-target hit rates by class, NDCG contribution by
class (sums to NDCG@5), and a paired user-cluster bootstrap (2000 draws, seed 2024)
for H1F - H0F contribution differences per class.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'session_wise_ar/cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

B = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'
HF = B + 'h_freeze35_20261029_r1/runs/'
MS = B + 'multiseed_freeze_20261109_v1/'
H0M = B + 'h0_multiseed_r20_freeze_20261110_v1/'
# node -> {split: path}; best nodes as selected by validation in the accepted reports
NODES = {
    '2024_H0F_R35': {'test': HF + 'H0-Freeze20-R35/evaluations/test_R35/attempt1', 'validation': HF + 'H0-Freeze20-R35/evaluations/validation_R35/attempt1'},
    '2024_H1F_R35': {'test': HF + 'H1-Freeze20-R35/evaluations/test_R35/attempt1', 'validation': HF + 'H1-Freeze20-R35/evaluations/validation_R35/attempt1'},
    '2025_H0F_R35': {'test': H0M + 'seed2025_H0/test_R35', 'validation': H0M + 'seed2025_H0/validation_R35'},
    '2025_H1F_R35': {'test': MS + 'seed2025_H1/test_R35', 'validation': MS + 'seed2025_H1/validation_R35'},
    '2026_H0F_R35': {'test': H0M + 'seed2026_H0/test_R35', 'validation': H0M + 'seed2026_H0/validation_R35'},
    '2026_H1F_R35': {'test': MS + 'seed2026_H1/test_R35', 'validation': MS + 'seed2026_H1/validation_R35'},
    '2024_H0F_best': {'test': HF + 'H0-Freeze20-R35/evaluations/test_R23/attempt1', 'validation': HF + 'H0-Freeze20-R35/evaluations/validation_R23/attempt1'},
    '2024_H1F_best': {'test': HF + 'H1-Freeze20-R35/evaluations/test_R32/attempt1', 'validation': HF + 'H1-Freeze20-R35/evaluations/validation_R32/attempt1'},
    '2025_H0F_best': {'test': H0M + 'seed2025_H0/test_best', 'validation': H0M + 'seed2025_H0/validation_R26'},
    '2025_H1F_best': {'test': MS + 'seed2025_H1/test_best', 'validation': MS + 'seed2025_H1/validation_R32'},
    '2026_H0F_best': {'test': H0M + 'seed2026_H0/test_best', 'validation': MS + 'seed2026_H0/validation_R20'},
    '2026_H1F_best': {'test': MS + 'seed2026_H1/test_best', 'validation': MS + 'seed2026_H1/validation_R29'},
}
PAIRS = [(s, n) for s in ('2024', '2025', '2026') for n in ('R35', 'best')]

REMOTE = r'''
import gzip, json, math, hashlib
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
DATA = Path(BASE + 'goal_20261027_r1/inputs/dataset'); NITEM = 10728; CUT = 1598745441188500
G = ['never_seen', 'history_only', 'head', 'mid', 'tail']
DISC = 1 / np.log2(np.arange(5) + 2); IDCG = DISC.sum()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tc = np.zeros(NITEM + 1, np.int64); hc = np.zeros(NITEM + 1, np.int64)
for b in pq.ParquetFile(DATA / 'train.parquet').iter_batches(batch_size=65536, columns=['history_item_ids', 'target_item_ids']):
    b = b.to_pydict()
    tc += np.bincount(np.asarray(b['target_item_ids'], np.int64).reshape(-1), minlength=NITEM + 1)
    hc += np.bincount(np.concatenate([np.asarray(x, np.int64) for x in b['history_item_ids']]), minlength=NITEM + 1)
seen = (tc + hc) > 0; seen[0] = False
ranked = sorted(np.flatnonzero(tc[1:]) + 1, key=lambda i: (-tc[i], i))
cls = np.full(NITEM + 1, -1, np.int8); cls[1:] = 0; cls[seen] = 1
h, m = math.ceil(.1 * len(ranked)), math.ceil(.5 * len(ranked))
cls[ranked[:h]] = 2; cls[ranked[h:m]] = 3; cls[ranked[m:]] = 4; cls[0] = -1
assert np.bincount(cls[1:], minlength=5).tolist() == [1914, 799, 802, 3206, 4007]
truth = {}
for split in ('validation', 'test'):
    t = pq.read_table(DATA / (split + '.parquet'), columns=['sample_id', 'user_id', 'target_item_ids', 'target_timestamp_us']).to_pydict()
    truth[split] = dict(index={str(s): i for i, s in enumerate(t['sample_id'])}, users=np.asarray(t['user_id']),
                        targets=np.asarray(t['target_item_ids'], np.int64), start=np.asarray([x[0] for x in t['target_timestamp_us']], np.int64))
def per_sample(pred, targets):
    # first occurrence, valid id; relevant if in target set (same list-metric semantics)
    valid = (pred > 0) & (pred <= NITEM)
    dup = np.zeros_like(valid)
    for j in range(1, 5): dup[:, j] = (pred[:, :j] == pred[:, j:j + 1]).any(1)
    rel = (pred[:, :, None] == targets[:, None, :]).any(-1) & valid & ~dup
    pc = cls[np.clip(pred, 0, NITEM)]
    contrib = np.stack([((rel & (pc == k)) * DISC).sum(1) / IDCG for k in range(5)], 1)   # [n,5]
    hit_t = ((pred[:, :, None] == targets[:, None, :]) & (valid & ~dup)[:, :, None]).any(1)   # [n,5 targets]
    return contrib, hit_t
def profile(pred, targets, contrib, hit_t, mask):
    p, t, c, ht = pred[mask], targets[mask], contrib[mask], hit_t[mask]
    flat = p.reshape(-1); flat = flat[(flat > 0) & (flat <= NITEM)]
    ic = np.bincount(flat, minlength=NITEM + 1); tcl = cls[t]
    return dict(samples=int(mask.sum()), ndcg5=float(c.sum(1).mean()),
                rec_share=dict(zip(G, (np.bincount(cls[flat], minlength=5) / len(flat)).round(6).tolist())),
                distinct_items=int((ic > 0).sum()), top10_share=float(np.sort(ic)[::-1][:10].sum() / len(flat)),
                ndcg_contribution=dict(zip(G, c.mean(0).tolist())),
                target_hit_rate={g: (float(ht[tcl == k].mean()) if (tcl == k).any() else None) for k, g in enumerate(G)})
out = {'class_counts': [1914, 799, 802, 3206, 4007], 'nodes': {}, 'pairs': []}
cache = {}
for name, splits in NODES.items():
    out['nodes'][name] = {}
    for split, path in splits.items():
        T = truth[split]; n = len(T['users'])
        pred = np.zeros((n, 5), np.int64); filled = np.zeros(n, bool)
        p = Path(path) / 'per_sample.jsonl.gz'
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); i = T['index'][str(r['sample_id'])]
                assert list(r['targets']) == T['targets'][i].tolist(); pred[i] = r['predictions'][0]; filled[i] = True
        assert filled.all(), (name, split)
        c, ht = per_sample(pred, T['targets']); cache[(name, split)] = c
        row = {'file': str(p), 'sha256': sha(p), 'all': profile(pred, T['targets'], c, ht, np.ones(n, bool))}
        if split == 'validation':
            row['early'] = profile(pred, T['targets'], c, ht, T['start'] < CUT)
            row['late'] = profile(pred, T['targets'], c, ht, T['start'] >= CUT)
        out['nodes'][name][split] = row
rng_seed = 2024
for seed, node in PAIRS:
    for split in ('validation', 'test'):
        a = cache[(f'{seed}_H1F_{node}', split)]; b = cache[(f'{seed}_H0F_{node}', split)]
        d = a - b; users, inv = np.unique(truth[split]['users'], return_inverse=True)
        cnt = np.bincount(inv, minlength=len(users)).astype(float)
        us = np.stack([np.bincount(inv, weights=d[:, k], minlength=len(users)) for k in range(5)] + [np.bincount(inv, weights=d.sum(1), minlength=len(users))], 1)
        rng = np.random.default_rng(rng_seed); boots = []
        for _ in range(2000):
            w = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)).astype(float)
            boots.append((w @ us) / (w @ cnt))
        boots = np.asarray(boots); keys = G + ['total']
        out['pairs'].append(dict(seed=seed, node=node, split=split,
            diff={k: float(d[:, i].mean()) if i < 5 else float(d.sum(1).mean()) for i, k in enumerate(keys)},
            ci95={k: np.quantile(boots[:, i], [.025, .975]).tolist() for i, k in enumerate(keys)}))
print('@@RESULT@@' + json.dumps(out))
'''


def main():
    src = REMOTE.replace('BASE + ', repr(B) + ' + ').replace('NODES.items()', repr(NODES) + '.items()').replace('PAIRS:', repr(PAIRS) + ':')
    text = jexec.python(src, timeout=2400)
    (HERE / 'RAW_OUTPUT.txt').write_text(text, encoding='utf-8')
    line = [x for x in text.splitlines() if x.startswith('@@RESULT@@')][-1]
    data = json.loads(line[len('@@RESULT@@'):])
    data['protocol'] = __doc__
    (HERE / 'RESULTS.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', len(data['nodes']), 'nodes', len(data['pairs']), 'pairs')


if __name__ == '__main__':
    main()
