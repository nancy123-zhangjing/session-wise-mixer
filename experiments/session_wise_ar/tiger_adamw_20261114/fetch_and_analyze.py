"""Fetch TIGER-AdamW (AR-T0A, AR-T1A) full results, paired user-cluster bootstrap,
and 5-class item popularity decomposition.

Read-only on remote shared storage: loads completed JSON summaries, catalogs, and
per-sample prediction files, writes nothing remotely, and saves structured local
results to experiments/session_wise_ar/tiger_adamw_20261114/RESULTS.json.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ['NO_PROXY'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
urllib.request.getproxies = lambda: {}

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cf_freeze_r9_20261114_v1' / 'paired_bootstrap_20261114'))
import jexec  # noqa: E402

REMOTE_SCRIPT = r'''
import gzip, json, math, hashlib
import numpy as np, pyarrow.parquet as pq, torch
from pathlib import Path

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
ROOT = BASE / 'session_wise_ar_tiger_adamw_20261114_v1'
DATA = BASE / 'goal_20261027_r1/inputs/dataset'
NITEM = 10728
G = ['never_seen', 'history_only', 'head', 'mid', 'tail']
DISC = 1.0 / np.log2(np.arange(5) + 2)
IDCG = float(DISC.sum())
REPS, SEED = 10000, 2024

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

out = {'arms': {}, 'bootstrap': [], 'nodes': {}, 'class_bootstrap': [], 'catalogs': {}}

# 1. Summarize pretrain, train curve, and test complete.json
for arm in ('AR-T0A', 'AR-T1A'):
    arm_dir = ROOT / 'runs' / arm
    pre = json.loads((arm_dir / 'pretrain' / 'complete.json').read_text())
    tr = json.loads((arm_dir / 'train' / 'complete.json').read_text())
    te = json.loads((arm_dir / 'test' / 'complete.json').read_text())
    pre_records = pre['progress']['records']
    tr_records = []
    for r in tr['progress']['records']:
        tr_records.append({
            'step': r['step'],
            'session_epochs': r['session_epochs'],
            'metrics': r['validation']['metrics'],
            'position_hit_rate': r['validation'].get('position_hit_rate'),
            'checkpoint_sha256': r['validation']['identity']['checkpoint_sha256'],
        })
    out['arms'][arm] = {
        'pretrain': {
            'selected_sha256': pre['selected_sha256'],
            'last_sha256': pre['last_sha256'],
            'parent_sha256': pre['parent_sha256'],
            'train_seconds': pre['progress']['train_seconds'],
            'first_record': pre_records[0],
            'final_record': pre_records[-1],
        },
        'train': {
            **{k: v for k, v in tr.items() if k != 'progress'},
            'best_step': tr['progress'].get('best_step'),
            'best_score': tr['progress'].get('best_score'),
            'train_seconds': tr['progress'].get('train_seconds'),
            'records': tr_records,
        },
        'test': te,
    }

# 2. Item classes (identical to Stage09 audit)
tc = np.zeros(NITEM + 1, np.int64)
hc = np.zeros(NITEM + 1, np.int64)
for b in pq.ParquetFile(DATA / 'train.parquet').iter_batches(batch_size=65536, columns=['history_item_ids', 'target_item_ids']):
    b = b.to_pydict()
    tc += np.bincount(np.asarray(b['target_item_ids'], np.int64).reshape(-1), minlength=NITEM + 1)
    hc += np.bincount(np.concatenate([np.asarray(x, np.int64) for x in b['history_item_ids']]), minlength=NITEM + 1)
seen = (tc + hc) > 0
seen[0] = False
ranked = sorted(np.flatnonzero(tc[1:]) + 1, key=lambda i: (-tc[i], i))
cls = np.full(NITEM + 1, -1, np.int8)
cls[1:] = 0
cls[seen] = 1
h, m = math.ceil(0.1 * len(ranked)), math.ceil(0.5 * len(ranked))
cls[ranked[:h]] = 2
cls[ranked[h:m]] = 3
cls[ranked[m:]] = 4
cls[0] = -1
counts = np.bincount(cls[1:], minlength=5).tolist()
assert counts == [1914, 799, 802, 3206, 4007], counts
out['class_counts'] = dict(zip(G, counts))

# 3. Catalogs & collision breakdown (semantic 3-level code from train/last.pt['sid'][1:, :3])
for arm in ('AR-T0A', 'AR-T1A'):
    cpath = ROOT / 'runs' / arm / 'train' / 'last.pt'
    s = torch.load(cpath, map_location='cpu', weights_only=False)
    sid_full = s['sid'].long().numpy()
    sid = sid_full[1:, :3]
    _, inv, size = np.unique(sid, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    item_bucket = size[inv]
    item_cls = cls[1:]
    coll = item_bucket > 1
    big = np.argsort(-size)[:5]
    out['catalogs'][arm] = {
        'file': str(cpath),
        'sha256': sha(cpath),
        'unique_codes': int(len(size)),
        'collision_items': int(coll.sum()),
        'collision_fraction': float(coll.mean()),
        'collision_redundancy': float(1.0 - len(size) / NITEM),
        'max_bucket': int(size.max()),
        'collision_items_by_class': {g: int((coll & (item_cls == k)).sum()) for k, g in enumerate(G)},
        'collision_rate_within_class': {g: float(coll[item_cls == k].mean()) for k, g in enumerate(G)},
        'largest_buckets': [
            {'size': int(size[b]), 'by_class': {g: int(((inv == b) & (item_cls == k)).sum()) for k, g in enumerate(G)}}
            for b in big
        ],
    }

# 4. Load truth splits
truth = {}
for split in ('validation', 'test'):
    t = pq.read_table(DATA / f'{split}.parquet', columns=['sample_id', 'user_id', 'target_item_ids']).to_pydict()
    sid_list = [str(x) for x in t['sample_id']]
    truth[split] = {
        'index': {s: i for i, s in enumerate(sid_list)},
        'users': np.asarray([str(u) for u in t['user_id']]),
        'targets': np.asarray(t['target_item_ids'], np.int64),
    }

# 5. Load predictions for key nodes
NODES = {
    'T0A_best_test': ('test', ROOT / 'runs/AR-T0A/test/step_0193998'),
    'T0A_final_test': ('test', ROOT / 'runs/AR-T0A/test/step_0200000'),
    'T1A_best_test': ('test', ROOT / 'runs/AR-T1A/test/step_0200000'),
    'T1A_final_test': ('test', ROOT / 'runs/AR-T1A/test/step_0200000'),
    'T0A_best_val': ('validation', ROOT / 'runs/AR-T0A/train/validation/step_0193998'),
    'T0A_final_val': ('validation', ROOT / 'runs/AR-T0A/train/validation/step_0200000'),
    'T1A_step193998_val': ('validation', ROOT / 'runs/AR-T1A/train/validation/step_0193998'),
    'T1A_best_val': ('validation', ROOT / 'runs/AR-T1A/train/validation/step_0200000'),
}

loaded = {}
for name, (split, pdir) in NODES.items():
    T = truth[split]
    n = len(T['users'])
    pred = np.zeros((n, 5), np.int64)
    filled = np.zeros(n, bool)
    p = Path(pdir) / 'predictions.jsonl.gz'
    summary = json.loads((Path(pdir) / 'summary.json').read_text())
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            i = T['index'][str(r['sample_id'])]
            pred[i] = r['prediction']
            filled[i] = True
    assert filled.all(), name
    # Compute per-sample metrics and class NDCG contributions directly from pred & targets
    hit = pred[:, :, None] == T['targets'][:, None, :]
    gain_pos = hit.any(-1) * DISC[None, :]
    metrics = {
        'ndcg@5': gain_pos.sum(axis=1) / IDCG,
        'recall@5': hit.any(1).mean(axis=1),
        'hr@5': hit.any((1, 2)).astype(np.float64),
    }
    for k in ('ndcg@5', 'recall@5', 'hr@5'):
        assert abs(float(metrics[k].mean()) - float(summary['metrics'][k])) < 1e-9, (name, k, metrics[k].mean(), summary['metrics'][k])
    pos_cls = cls[np.clip(pred, 0, NITEM)]
    cls_ndcg = {g: (gain_pos * (pos_cls == k)).sum(axis=1) / IDCG for k, g in enumerate(G)}
    flat = pred.reshape(-1)
    flat_valid = flat[(flat > 0) & (flat <= NITEM)]
    share = np.bincount(cls[flat_valid], minlength=5) / len(flat_valid)
    ic = np.bincount(flat_valid, minlength=NITEM + 1)
    top10 = float(np.sort(ic)[::-1][:10].sum() / len(flat_valid))
    tgt_hit = hit.any(1)
    tgt_cls = cls[T['targets']]
    hit_rate = {g: (float(tgt_hit[tgt_cls == k].mean()) if (tgt_cls == k).any() else None) for k, g in enumerate(G)}
    out['nodes'][name] = {
        'split': split,
        'file': str(p),
        'sha256': sha(p),
        'samples': int(n),
        'ndcg@5': float(metrics['ndcg@5'].mean()),
        'recall@5': float(metrics['recall@5'].mean()),
        'hr@5': float(metrics['hr@5'].mean()),
        'rec_share': dict(zip(G, share.round(6).tolist())),
        'distinct_items': int((ic > 0).sum()),
        'top10_share': top10,
        'ndcg_contribution': {g: float(cls_ndcg[g].mean()) for g in G},
        'target_hit_rate': hit_rate,
    }
    loaded[name] = {'split': split, 'metrics': metrics, 'cls_ndcg': cls_ndcg, 'file': str(p), 'sha256': out['nodes'][name]['sha256']}

# 6. Paired user-cluster bootstrap
COMPS = [
    ('best_mixer_vs_none', 'test', 'T1A_best_test', 'T0A_best_test'),
    ('final_mixer_vs_none', 'test', 'T1A_final_test', 'T0A_final_test'),
    ('best_mixer_vs_none', 'validation', 'T1A_best_val', 'T0A_best_val'),
    ('final_mixer_vs_none', 'validation', 'T1A_best_val', 'T0A_final_val'),
    ('step193998_mixer_vs_none', 'validation', 'T1A_step193998_val', 'T0A_best_val'),
]

for label, split, na, nb in COMPS:
    A, B = loaded[na], loaded[nb]
    T = truth[split]
    users, inv = np.unique(T['users'], return_inverse=True)
    counts_u = np.bincount(inv, minlength=len(users)).astype(np.float64)
    rng = np.random.default_rng(SEED)
    w = np.empty((REPS, len(users)), dtype=np.float32)
    for s in range(0, REPS, 500):
        d = rng.integers(0, len(users), size=(min(500, REPS - s), len(users)))
        for j, dj in enumerate(d):
            w[s + j] = np.bincount(dj, minlength=len(users))
    w = w.astype(np.float64)
    den = w @ counts_u

    row = {'label': label, 'split': split, 'A': na, 'B': nb, 'samples': int(len(T['users'])), 'users': int(len(users)), 'metrics': {}}
    for k in ('ndcg@5', 'recall@5', 'hr@5'):
        d = A['metrics'][k] - B['metrics'][k]
        usum = np.bincount(inv, weights=d, minlength=len(users))
        boot = (w @ usum) / den
        ma, mb = float(A['metrics'][k].mean()), float(B['metrics'][k].mean())
        p_two = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        row['metrics'][k] = {
            'mean_A': ma, 'mean_B': mb, 'diff': ma - mb,
            'relative': (ma - mb) / mb if mb else None,
            'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
            'p_bootstrap_two_sided': p_two,
            'A_better_samples': int((d > 0).sum()),
            'B_better_samples': int((d < 0).sum()),
        }
    out['bootstrap'].append(row)

    crow = {'label': label, 'split': split, 'A': na, 'B': nb, 'classes': {}}
    for g in G:
        d = A['cls_ndcg'][g] - B['cls_ndcg'][g]
        usum = np.bincount(inv, weights=d, minlength=len(users))
        boot = (w @ usum) / den
        ma, mb = float(A['cls_ndcg'][g].mean()), float(B['cls_ndcg'][g].mean())
        p_two = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        crow['classes'][g] = {
            'mean_A': ma, 'mean_B': mb, 'diff': ma - mb,
            'relative': (ma - mb) / mb if mb else None,
            'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
            'p_bootstrap_two_sided': p_two,
        }
    out['class_bootstrap'].append(crow)

print('@@TIGER_ADAMW_RESULT@@' + json.dumps(out))
'''


def main():
    text = jexec.python(REMOTE_SCRIPT, timeout=1200)
    line = [x for x in text.splitlines() if x.startswith('@@TIGER_ADAMW_RESULT@@')][-1]
    data = json.loads(line[len('@@TIGER_ADAMW_RESULT@@'):])
    out_path = HERE / 'RESULTS.json'
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {out_path}')
    for r in data['bootstrap']:
        m = r['metrics']['ndcg@5']
        print(f"{r['split']:10s} {r['label']:26s} {m['mean_A']:.6f} vs {m['mean_B']:.6f} "
              f"rel={m['relative']:+.2%} CI=[{m['ci95'][0]:+.6f},{m['ci95'][1]:+.6f}] p={m['p_bootstrap_two_sided']:.4f}")
    print('--- Per-class NDCG contributions ---')
    for c in data['class_bootstrap']:
        print(f"=== {c['split']} {c['label']} ===")
        for g in ('head', 'mid', 'tail'):
            cg = c['classes'][g]
            rel_s = f"{cg['relative']:+.2%}" if cg['relative'] is not None else 'N/A'
            print(f"  {g:8s}: {cg['mean_A']:.6f} vs {cg['mean_B']:.6f} diff={cg['diff']:+.6f} ({rel_s}) CI=[{cg['ci95'][0]:+.6f},{cg['ci95'][1]:+.6f}] p={cg['p_bootstrap_two_sided']:.4f}")


if __name__ == '__main__':
    main()
