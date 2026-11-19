"""Slate metrics aligned with OGR (arXiv:2608.17613) and HiGR (arXiv:2512.24787).

Columns: Impressions Hit@5 / Recall@5, Effective-View Hit@5 / Recall@5 / NDCG@5,
plus Impression NDCG@5. EV metrics are averaged only over samples whose target
slate contains at least one effective view (watch_ratio >= threshold).

CLI (any model's predictions.jsonl.gz with sample_id + prediction):
    python -m scripts.kuairec_data.l5o_metrics --dataset outputs/kuairec_l5o_v1/run1 \
        --split test --predictions path/predictions.jsonl.gz
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

K = 5
DISCOUNT = 1.0 / np.log2(np.arange(K) + 2.0)


def _relevance(pred, members):
    """pred [N,K], members boolean mask builder -> rel [N,K]; duplicate preds count once."""
    n = len(pred)
    rel = np.zeros((n, K), dtype=bool)
    for i in range(n):
        seen = set()
        for j in range(K):
            p = int(pred[i, j])
            if p > 0 and p not in seen and p in members[i]:
                rel[i, j] = True
            seen.add(p)
    return rel


def _ndcg(rel, sizes):
    dcg = (rel * DISCOUNT).sum(1)
    idcg = np.array([DISCOUNT[:min(K, int(s))].sum() if s > 0 else 1.0 for s in sizes])
    return dcg / idcg


def slate_metrics(pred, target, watch_ratio, ev_min=1.0, return_per_sample=False):
    pred, target, watch_ratio = np.asarray(pred), np.asarray(target), np.asarray(watch_ratio, dtype=float)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape[1] != K or target.shape != watch_ratio.shape or len(pred) != len(target) or not len(pred):
        raise ValueError('Shape mismatch')
    if not np.isfinite(watch_ratio).all() or not np.isfinite(ev_min):
        raise ValueError('Watch ratios and EV threshold must be finite')
    imp_sets = [set(int(x) for x in row if x > 0) for row in target]
    ev_sets = [set(int(x) for x, w in zip(row, wr) if x > 0 and w >= ev_min) for row, wr in zip(target, watch_ratio)]
    imp_size = np.array([len(s) for s in imp_sets]); ev_size = np.array([len(s) for s in ev_sets])
    if np.any(imp_size == 0):
        raise ValueError('Empty target slate')
    rel_i, rel_e = _relevance(pred, imp_sets), _relevance(pred, ev_sets)
    has_ev = ev_size > 0
    per = {
        'imp_hit@5': (rel_i.sum(1) > 0).astype(float),
        'imp_recall@5': rel_i.sum(1) / imp_size,
        'imp_ndcg@5': _ndcg(rel_i, imp_size),
        'legacy_fixed5_ndcg@5': (rel_i * DISCOUNT).sum(1) / DISCOUNT.sum(),
        'ev_hit@5': np.where(has_ev, rel_e.sum(1) > 0, np.nan).astype(float),
        'ev_recall@5': np.where(has_ev, rel_e.sum(1) / np.maximum(ev_size, 1), np.nan),
        'ev_ndcg@5': np.where(has_ev, _ndcg(rel_e, ev_size), np.nan),
    }
    # JSON null, not NaN or a fabricated zero, when no eligible EV sample exists.
    out = {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in per.items()}
    out.update(samples=int(len(pred)), ev_samples=int(has_ev.sum()), ev_min=float(ev_min))
    return (out, per) if return_per_sample else out


def load_split(dataset, split):
    t = pq.read_table(Path(dataset) / f'{split}.parquet',
                      columns=['sample_id', 'history_item_ids', 'target_item_ids', 'target_watch_ratio'])
    return (t['sample_id'].to_pylist(), np.array(t['target_item_ids'].to_pylist()),
            np.array(t['target_watch_ratio'].to_pylist()), t['history_item_ids'])


def load_predictions(path, sample_ids):
    index = {s: i for i, s in enumerate(sample_ids)}
    pred = np.zeros((len(sample_ids), K), dtype=np.int64)
    filled = np.zeros(len(sample_ids), dtype=bool)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            i = index[r['sample_id']]
            if filled[i]:
                raise ValueError(f'Duplicate prediction sample: {r["sample_id"]}')
            values = r.get('prediction', r.get('predictions', [None])[0])
            if values is None or len(values) < K:
                raise ValueError('Missing five-item prediction')
            pred[i] = values[:K]
            filled[i] = True
    if not filled.all():
        raise ValueError(f'{int((~filled).sum())} samples lack predictions')
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', default='test', choices=['validation', 'test'])
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--ev-min', type=float, nargs='+', default=[1.0, 2.0])
    a = p.parse_args()
    ids, target, wr, _ = load_split(a.dataset, a.split)
    pred = load_predictions(a.predictions, ids)
    print(json.dumps({f'ev>={t}': slate_metrics(pred, target, wr, t) for t in a.ev_min}, indent=2))


if __name__ == '__main__':
    main()
