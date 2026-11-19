"""Paired user-clustered bootstrap for the four completed CF arms.

Read-only: loads existing per-sample prediction files on shared storage, runs no
model, writes nothing remotely. Results are printed as JSON and saved locally.
Protocol (fixed before looking at results):
- unit = user cluster; a sampled user contributes all of their samples;
- estimate = sample-weighted mean difference (same as reported NDCG means);
- 10000 resamples, seed 2024, 95% percentile interval;
- Holm adjustment is reported across the test-set comparison family only.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jexec  # noqa: E402

BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
ROOTS = {
    'M0': BASE + '/session_wise_ar_five_20261113_v1/runs/CF-M0/train',
    'M1': BASE + '/rqcd_targets_formal_20261113_v1/cf',
    'M0F': BASE + '/cf_freeze_r9_20261114_v1/runs/M0/train',
    'M1F': BASE + '/cf_freeze_r9_20261114_v1/runs/M1/train',
}
# (label, split, A arm, A relative file, B arm, B relative file); diff = A - B
COMPARISONS = [
    ('best_R9_mixer_vs_none', 'test', 'M1', 'test/best', 'M0', 'test/best'),
    ('R35_frozen_mixer_vs_frozen_none', 'test', 'M1F', 'test/final', 'M0F', 'test/final'),
    ('R35_mixer_frozen_vs_continued', 'test', 'M1F', 'test/final', 'M1', 'test/final'),
    ('R35_none_frozen_vs_continued', 'test', 'M0F', 'test/final', 'M0', 'test/final'),
    ('R35_continued_mixer_vs_none', 'test', 'M1', 'test/final', 'M0', 'test/final'),
    ('best_R9_mixer_vs_none', 'validation', 'M1', 'validation/R09', 'M0', 'validation/R09'),
    ('R35_frozen_mixer_vs_frozen_none', 'validation', 'M1F', 'validation/R35', 'M0F', 'validation/R35'),
    ('R35_mixer_frozen_vs_continued', 'validation', 'M1F', 'validation/R35', 'M1', 'validation/R35'),
    ('R35_none_frozen_vs_continued', 'validation', 'M0F', 'validation/R35', 'M0', 'validation/R35'),
    ('R35_continued_mixer_vs_none', 'validation', 'M1', 'validation/R35', 'M0', 'validation/R35'),
]
REPS, SEED = 10000, 2024

REMOTE_SOURCE = r'''
import gzip, json, hashlib, numpy as np
from pathlib import Path
roots = ROOTS; comps = COMPARISONS; REPS = REPS_; SEED = SEED_
cache = {}
def load(arm, rel):
    key = (arm, rel)
    if key in cache: return cache[key]
    p = Path(roots[arm]) / rel / 'predictions.jsonl.gz'
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    sid, uid, m = [], [], {'ndcg@5': [], 'recall@5': [], 'hr@5': []}
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); sid.append(r['sample_id']); uid.append(str(r['user_id']))
            for k in m: m[k].append(float(r['metrics'][k]))
    order = np.argsort(np.asarray(sid, dtype=object).astype(str), kind='stable')
    out = {'path': str(p), 'sha256': h, 'n': len(sid),
           'sid': np.asarray(sid, dtype=object)[order].astype(str),
           'uid': np.asarray(uid, dtype=object)[order].astype(str),
           'm': {k: np.asarray(v, dtype=np.float64)[order] for k, v in m.items()}}
    cache[key] = out; return out
results = []
for label, split, a, ra, b, rb in comps:
    A, B = load(a, ra), load(b, rb)
    assert A['n'] == B['n'] and (A['sid'] == B['sid']).all() and (A['uid'] == B['uid']).all()
    users, inv = np.unique(A['uid'], return_inverse=True)
    counts = np.bincount(inv, minlength=len(users)).astype(np.float64)
    rng = np.random.default_rng(SEED)
    w = np.empty((REPS, len(users)), dtype=np.float32)
    for s in range(0, REPS, 500):
        d = rng.integers(0, len(users), size=(min(500, REPS - s), len(users)))
        for j, dj in enumerate(d): w[s + j] = np.bincount(dj, minlength=len(users))
    w = w.astype(np.float64)
    den = w @ counts
    row = {'label': label, 'split': split, 'A': a, 'A_file': A['path'], 'A_sha256': A['sha256'],
           'B': b, 'B_file': B['path'], 'B_sha256': B['sha256'], 'samples': int(A['n']),
           'users': int(len(users)), 'metrics': {}}
    for k in A['m']:
        d = A['m'][k] - B['m'][k]
        usum = np.bincount(inv, weights=d, minlength=len(users))
        boot = (w @ usum) / den
        ma, mb = float(A['m'][k].mean()), float(B['m'][k].mean())
        p_two = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        row['metrics'][k] = {'mean_A': ma, 'mean_B': mb, 'diff': ma - mb,
                             'relative': (ma - mb) / mb if mb else None,
                             'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
                             'p_bootstrap_two_sided': p_two,
                             'A_better_samples': int((d > 0).sum()), 'B_better_samples': int((d < 0).sum())}
    results.append(row)
print(json.dumps({'reps': REPS, 'seed': SEED, 'results': results}))
'''


def holm(rows):
    tests = [r for r in rows if r['split'] == 'test']
    ps = sorted((r['metrics']['ndcg@5']['p_bootstrap_two_sided'], i) for i, r in enumerate(tests))
    m, running = len(ps), 0.0
    for rank, (p, i) in enumerate(ps):
        running = max(running, min(1.0, (m - rank) * p))
        tests[i]['metrics']['ndcg@5']['p_holm_test_family'] = running


def main():
    source = (REMOTE_SOURCE.replace('ROOTS', repr(ROOTS), 1).replace('COMPARISONS', repr(COMPARISONS), 1)
              .replace('REPS_', str(REPS)).replace('SEED_', str(SEED)))
    text = jexec.python(source, timeout=1200)
    (HERE / 'BOOTSTRAP_RAW_OUTPUT.txt').write_text(text, encoding='utf-8')
    line = [x for x in text.splitlines() if x.strip().startswith('{"reps"')][-1]
    data = json.loads(line)
    holm(data['results'])
    data['protocol'] = __doc__
    (HERE / 'RESULTS.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in data['results']:
        n = r['metrics']['ndcg@5']
        print(f"{r['split']:10s} {r['label']:34s} {n['mean_A']:.6f} vs {n['mean_B']:.6f} "
              f"rel={n['relative']:+.2%} CI=[{n['ci95'][0]:+.6f},{n['ci95'][1]:+.6f}] "
              f"p={n['p_bootstrap_two_sided']:.4f} holm={n.get('p_holm_test_family', '')}")


if __name__ == '__main__':
    main()
