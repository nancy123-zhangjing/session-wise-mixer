"""Compare full-test L5O predictions with a paired user bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np

from .l5o_metrics import load_predictions, load_split, slate_metrics


def paired_delta(base, variant, rng, draws):
    if base.shape != variant.shape or not np.array_equal(np.isfinite(base), np.isfinite(variant)):
        raise ValueError('Paired metrics must use the same eligible users')
    delta = (variant - base)[np.isfinite(base)]
    if not len(delta):
        raise ValueError('No eligible users')
    sampled = []
    for start in range(0, draws, 32):
        indices = rng.integers(len(delta), size=(min(32, draws - start), len(delta)))
        sampled.extend(delta[indices].mean(axis=1))
    return dict(users=len(delta), delta=float(delta.mean()),
                ci95=np.quantile(sampled, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'base', 'variant', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    import pyarrow.parquet as pq

    users = pq.read_table(args.dataset/'test.parquet', columns=['user_id'])[0].to_pylist()
    ids, target, ratios, _ = load_split(args.dataset, 'test')
    if len(ids) != 7176 or len(set(users)) != len(ids):
        raise ValueError('Expected the complete L5O test with one slate per user')
    base = load_predictions(args.base, ids)
    variant = load_predictions(args.variant, ids)
    results = {}
    for threshold in (1.0, 2.0):
        _, a = slate_metrics(base, target, ratios, threshold, return_per_sample=True)
        _, b = slate_metrics(variant, target, ratios, threshold, return_per_sample=True)
        results[str(threshold)] = {metric: paired_delta(a[metric], b[metric], np.random.default_rng(2024), args.draws)
                                   for metric in ('imp_hit@5', 'imp_recall@5', 'imp_ndcg@5',
                                                  'ev_hit@5', 'ev_recall@5', 'ev_ndcg@5')}
    result = dict(direction='variant-minus-base', draws=args.draws, seed=2024,
                  interpretation='Paired user sampling for a single training seed', thresholds=results)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
