"""Steps 2-4 of the CF follow-up: read-only statistics on existing artifacts.

Step 2 (prediction profile): what kinds of items each CF node recommends (popularity
class shares, distinct items, top-10 concentration) at best R9 vs R35.
Step 3 (catalog collisions): which items sit in multi-item SID buckets.
Step 4 (validation vs test): per-item-class target composition, hit rate and NDCG
contribution for CF and TIGER on validation (early/late) and test.

No training, inference, or remote writes. Item classes follow the earlier Stage09
audit exactly: never_seen (absent from all training rows), history_only (only in
training histories), and training-target items split by target count into
head (top 10%), mid (next 40%) and tail (rest); ties by internal item ID.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'paired_bootstrap_20261114'))
import jexec  # noqa: E402

BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
M0 = BASE + '/session_wise_ar_five_20261113_v1/runs/CF-M0/train'
M1 = BASE + '/rqcd_targets_formal_20261113_v1/cf'
M0F = BASE + '/cf_freeze_r9_20261114_v1/runs/M0/train'
M1F = BASE + '/cf_freeze_r9_20261114_v1/runs/M1/train'
T0 = BASE + '/session_wise_ar_five_20261113_r2/runs/AR-T0'
PRED = {  # name: (split, path)
    'M0_best_test': ('test', M0 + '/test/best'), 'M1_best_test': ('test', M1 + '/test/best'),
    'M0_R35_test': ('test', M0 + '/test/final'), 'M1_R35_test': ('test', M1 + '/test/final'),
    'M0F_R35_test': ('test', M0F + '/test/final'), 'M1F_R35_test': ('test', M1F + '/test/final'),
    'T0_best_test': ('test', T0 + '/test/step_0193998'), 'T0_final_test': ('test', T0 + '/test/step_0200000'),
    'M0_best_val': ('validation', M0 + '/validation/R09'), 'M1_best_val': ('validation', M1 + '/validation/R09'),
    'M0_R35_val': ('validation', M0 + '/validation/R35'), 'M1_R35_val': ('validation', M1 + '/validation/R35'),
    'M0F_R35_val': ('validation', M0F + '/validation/R35'), 'M1F_R35_val': ('validation', M1F + '/validation/R35'),
    'T0_best_val': ('validation', T0 + '/train/validation/step_0193998'),
    'T0_final_val': ('validation', T0 + '/train/validation/step_0200000'),
}
CATALOGS = {'M0_R09': M0 + '/checkpoints/R09_T040.pt', 'M1_R09': M1 + '/checkpoints/R09_T040.pt',
            'M0_R35': M0 + '/checkpoints/R35_T300.pt', 'M1_R35': M1 + '/checkpoints/R35_T300.pt'}

REMOTE = r'''
import gzip, json, math, hashlib
import numpy as np, pyarrow.parquet as pq, torch
from pathlib import Path
DATA = Path(BASE_ + '/goal_20261027_r1/inputs/dataset'); NITEM = 10728; CUT = 1598745441188500
G = ['never_seen', 'history_only', 'head', 'mid', 'tail']
DISC = 1 / np.log2(np.arange(5) + 2); IDCG = DISC.sum()
out = {'inputs': {}}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('train.parquet', 'validation.parquet', 'test.parquet', 'train_seen.json'):
    out['inputs'][f] = sha(DATA / f)
# ---- item classes (same rule as Stage09 audit) ----
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
counts = np.bincount(cls[1:], minlength=5).tolist(); assert counts == [1914, 799, 802, 3206, 4007], counts
out['class_counts'] = dict(zip(G, counts))
# ---- splits ----
truth = {}
for split in ('validation', 'test'):
    t = pq.read_table(DATA / (split + '.parquet'), columns=['sample_id', 'user_id', 'target_item_ids', 'target_timestamp_us']).to_pydict()
    sid = [str(x) for x in t['sample_id']]
    truth[split] = dict(index={s: i for i, s in enumerate(sid)}, users=np.asarray(t['user_id']),
                        targets=np.asarray(t['target_item_ids'], np.int64),
                        start=np.asarray([x[0] for x in t['target_timestamp_us']], np.int64))
def comp(targets):
    c = np.bincount(cls[targets].reshape(-1), minlength=5); return dict(zip(G, (c / c.sum()).round(6).tolist()))
out['target_composition'] = {}
for split, T in truth.items():
    out['target_composition'][split] = comp(T['targets'])
    if split == 'validation':
        out['target_composition']['validation_early'] = comp(T['targets'][T['start'] < CUT])
        out['target_composition']['validation_late'] = comp(T['targets'][T['start'] >= CUT])
# ---- per-node profile ----
def profile(pred, targets, mask):
    pred, targets = pred[mask], targets[mask]
    n = len(pred)
    flat = pred.reshape(-1); flat = flat[(flat > 0) & (flat <= NITEM)]
    share = np.bincount(cls[flat], minlength=5) / len(flat)
    ic = np.bincount(flat, minlength=NITEM + 1)
    top10 = np.sort(ic)[::-1][:10].sum() / len(flat)
    hit = pred[:, :, None] == targets[:, None, :]          # [n, pos, target]
    gain_pos = (hit.any(-1) * DISC[None, :])                # NDCG gain per predicted position
    pos_cls = cls[np.clip(pred, 0, NITEM)]
    contrib = {g: float((gain_pos * (pos_cls == k)).sum() / IDCG / n) for k, g in enumerate(G)}
    tgt_hit = hit.any(1)                                    # each target hit?
    tgt_cls = cls[targets]
    hit_rate = {g: (float(tgt_hit[tgt_cls == k].mean()) if (tgt_cls == k).any() else None) for k, g in enumerate(G)}
    return dict(samples=int(n), ndcg5=float(gain_pos.sum() / IDCG / n),
                rec_share=dict(zip(G, share.round(6).tolist())), distinct_items=int((ic > 0).sum()),
                top10_share=float(top10), mean_log_train_count=float(np.log1p(tc[flat]).mean()),
                ndcg_contribution=contrib, target_hit_rate=hit_rate)
out['nodes'] = {}
for name, (split, path) in PRED.items():
    T = truth[split]; n = len(T['users'])
    pred = np.zeros((n, 5), np.int64); filled = np.zeros(n, bool)
    p = Path(path) / 'predictions.jsonl.gz'
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); i = T['index'][str(r['sample_id'])]
            assert list(r['target']) == T['targets'][i].tolist(); pred[i] = r['prediction']; filled[i] = True
    assert filled.all(), name
    row = {'file': str(p), 'sha256': sha(p), 'all': profile(pred, T['targets'], np.ones(n, bool))}
    if split == 'validation':
        row['early'] = profile(pred, T['targets'], T['start'] < CUT)
        row['late'] = profile(pred, T['targets'], T['start'] >= CUT)
    out['nodes'][name] = row
# ---- catalog collisions ----
out['catalogs'] = {}; ref_ids = None
for name, path in CATALOGS.items():
    s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cat = s['catalog']; sid = cat['sid'].long().numpy(); ids = np.asarray(cat['original_ids'])
    if ref_ids is None: ref_ids = ids
    assert np.array_equal(ids, ref_ids) and sid.shape == (NITEM, 4)
    _, inv, size = np.unique(sid, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1); item_bucket = size[inv]            # row r -> item r+1
    item_cls = cls[1:]
    coll = item_bucket > 1
    big = np.argsort(-size)[:5]
    out['catalogs'][name] = dict(
        file=path, collision_items=int(coll.sum()), collision_fraction=float(coll.mean()), max_bucket=int(size.max()),
        collision_items_by_class={g: int((coll & (item_cls == k)).sum()) for k, g in enumerate(G)},
        collision_rate_within_class={g: float(coll[item_cls == k].mean()) for k, g in enumerate(G)},
        largest_buckets=[dict(size=int(size[b]), by_class={g: int(((inv == b) & (item_cls == k)).sum()) for k, g in enumerate(G)}) for b in big])
print('@@RESULT@@' + json.dumps(out))
'''


def main():
    source = (REMOTE.replace('BASE_', repr(BASE)).replace('PRED.items()', repr(PRED) + '.items()')
              .replace('CATALOGS.items()', repr(CATALOGS) + '.items()'))
    text = jexec.python(source, timeout=1500)
    (HERE / 'RAW_OUTPUT.txt').write_text(text, encoding='utf-8')
    line = [x for x in text.splitlines() if x.startswith('@@RESULT@@')][-1]
    data = json.loads(line[len('@@RESULT@@'):])
    data['protocol'] = __doc__
    (HERE / 'RESULTS.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', len(data['nodes']), 'nodes', len(data['catalogs']), 'catalogs')


if __name__ == '__main__':
    main()
