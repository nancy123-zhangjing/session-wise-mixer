"""Independent prediction replay and paired user-cluster bootstrap.

This module deliberately does not import decoding.py, metrics.py, or torch.
Primary estimates average samples. A bootstrap draw samples users with
replacement and carries all of each drawn user's samples (including repeats).
User-average estimates are reported separately. Neither is a multi-seed test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .subset import canonical_sha256, file_sha256


def _ranking(predictions, targets, ks, catalog_size):
    seen, relevant, valid, duplicate = set(), [], [], []
    for item in predictions:
        is_valid = isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= catalog_size
        repeat = item > 0 and item in seen
        seen.add(item)
        valid.append(is_valid)
        duplicate.append(repeat)
        relevant.append(is_valid and not repeat and item in targets)
    length = len(predictions)
    metrics = {}
    for k in ks:
        if not 1 <= k <= length:
            raise ValueError('Metric cutoff exceeds supplied recommendation width')
        hits = sum(relevant[:k])
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(targets))))
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(relevant[:k]))
        metrics.update({f'recall@{k}': hits / len(targets), f'hr@{k}': float(hits > 0),
                        f'hit_rate@{k}': float(hits > 0), f'ndcg@{k}': dcg / ideal,
                        f'precision@{k}': hits / k})
    metrics.update(completion_rate=float(all(v and not d for v, d in zip(valid, duplicate))),
                   duplicate_rate=sum(duplicate) / length,
                   missing_rate=predictions.count(0) / length,
                   invalid_rate=sum(item != 0 and not v for item, v in zip(predictions, valid)) / length,
                   distinct_valid_items=float(sum(v and not d for v, d in zip(valid, duplicate))))
    if length == len(targets):
        metrics['position_accuracy'] = sum(a == b for a, b in zip(predictions, targets)) / length
        metrics['exact_match'] = float(predictions == targets)
    return metrics


def recompute_row(row, *, catalog_size, list_length=5, aggregate_ks=(10, 20, 50), max_candidate_lists=32):
    """Recompute item-level relevance from IDs, never trust saved metric values."""
    targets = row['targets']
    if (len(targets) != list_length or len(set(targets)) != list_length
            or any(not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= catalog_size for i in targets)):
        raise ValueError('Targets must be distinct true item IDs in the locked catalog')
    predictions, scores = row['predictions'], row['scores']
    beams, complete = row['beam_mask'], row['complete_mask']
    if not predictions or any(len(values) != len(predictions) for values in (scores, beams, complete)):
        raise ValueError('Prediction/score/beam lengths differ')
    for items, score, present, declared_complete in zip(predictions, scores, beams, complete):
        if len(items) != list_length or any(not isinstance(i, int) or isinstance(i, bool) for i in items):
            raise ValueError('Each beam must hold exactly list_length integer item IDs')
        finite = score is not None and isinstance(score, (int, float)) and math.isfinite(score)
        if type(present) is not bool or type(declared_complete) is not bool or present != finite:
            raise ValueError('Beam mask must agree with finite path scores')
        derived = present and len(set(items)) == list_length and all(1 <= i <= catalog_size for i in items)
        if declared_complete != derived:
            raise ValueError('Saved completion flag disagrees with true distinct item IDs')
    metrics = {'list/' + key: value for key, value in
               _ranking(predictions[0], targets, (list_length,), catalog_size).items()}
    if aggregate_ks:
        maximum = max(aggregate_ks)
        if maximum < 1 or max_candidate_lists < 1:
            raise ValueError('Candidate budgets must be positive')
        order = sorted(range(len(predictions)), key=lambda i: (-(scores[i] if scores[i] is not None else -math.inf), i))
        collected, seen, used = [], set(), 0
        for index in order:
            if not (beams[index] and complete[index]):
                continue
            if used == max_candidate_lists:
                break
            used += 1
            for item in predictions[index]:
                if item not in seen:
                    seen.add(item)
                    collected.append(item)
                    if len(collected) == maximum:
                        break
            if len(collected) == maximum:
                break
        count = len(collected)
        collected.extend([0] * (maximum - count))
        metrics.update({'aggregate/' + key: value for key, value in
                        _ranking(collected, targets, aggregate_ks, catalog_size).items()})
        metrics['aggregate/candidate_count'], metrics['aggregate/lists_used'] = float(count), float(used)
        for k in aggregate_ks:
            metrics[f'aggregate/coverage@{k}'] = min(count, k) / k
    return metrics


def _rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Blank prediction record at line {number}')
            row = json.loads(line)
            if not isinstance(row.get('sample_id'), str) or not row['sample_id']:
                raise ValueError('sample_id must be exact nonempty text')
            if not isinstance(row.get('user_id'), int) or isinstance(row['user_id'], bool):
                raise ValueError('user_id must be integer')
            yield row


def _metadata(predictions_path):
    path = Path(predictions_path).parent / 'summary.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    protocol = meta['metric_protocol']
    if meta['catalog_size'] < 1 or protocol['pad_id'] != 0:
        raise ValueError('Unsupported catalog or PAD convention')
    if protocol['aggregation'] != 'list_score_order_then_item_position_first_occurrence':
        raise ValueError('Unsupported candidate aggregation protocol')
    return meta, dict(catalog_size=meta['catalog_size'], list_length=protocol['list_length'],
                      aggregate_ks=tuple(protocol['aggregate_ks']), max_candidate_lists=protocol['candidate_lists'])


def _truth(dataset_dir, meta):
    import pyarrow.parquet as pq
    root = Path(dataset_dir)
    split = meta['split']
    if split not in ('validation', 'test'):
        raise ValueError('Unsupported evaluation split')
    name = split + '.parquet'
    expected = meta['dataset_binding']['files'][name]['sha256']
    if file_sha256(root / name) != expected:
        raise ValueError('Ground-truth split hash differs from evaluation binding')
    table = pq.read_table(root / name, columns=['sample_id', 'user_id', 'target_item_ids'])
    result = {}
    for index, row in enumerate(table.to_pylist()):
        key = row['sample_id']
        if key in result:
            raise ValueError('Duplicate ground-truth sample ID')
        result[key] = (index, row['user_id'], row['target_item_ids'])
    return result


def audit_predictions(predictions_path, *, dataset_dir=None, tolerance=2e-6):
    """Replay every metric and optionally bind every label/index to source data."""
    meta, arguments = _metadata(predictions_path)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    truth = _truth(dataset_dir, meta) if dataset_dir is not None else None
    ids, sums, users, groups = set(), defaultdict(float), {}, {}
    mismatch_count, max_error, count = 0, 0.0, 0
    selected_rows = []
    for row in _rows(predictions_path):
        key, user = row['sample_id'], row['user_id']
        if key in ids:
            raise ValueError('Duplicate prediction sample ID')
        ids.add(key)
        if meta.get('subset') is not None:
            digest = hashlib.sha256((meta['subset']['salt'] + '\n' + key).encode('utf-8')).hexdigest()
            selected_rows.append(dict(index=row['dataset_index'], sample_id=key, selection_sha256=digest))
        if truth is not None:
            expected = truth.get(key)
            if expected != (row.get('dataset_index'), user, row['targets']):
                raise ValueError('Saved sample index, user, or targets disagree with source data')
        metrics = recompute_row(row, **arguments)
        if set(row['metrics']) != set(metrics):
            raise ValueError('Stored metric names differ from independently recomputed protocol')
        count += 1
        user_row = users.setdefault(user, [0, defaultdict(float)])
        group = 'cold_target' if row['cold_target'] else 'seen_targets'
        group_row = groups.setdefault(group, [0, defaultdict(float)])
        user_row[0] += 1
        group_row[0] += 1
        for name, value in metrics.items():
            stored = row['metrics'][name]
            error = abs(value - stored) if isinstance(stored, (int, float)) and math.isfinite(stored) else math.inf
            max_error = max(max_error, error)
            mismatch_count += error > tolerance
            sums[name] += value
            user_row[1][name] += value
            group_row[1][name] += value
    if not count or count != meta['examples']:
        raise ValueError('Prediction count differs from completed evaluation summary')
    if selected_rows and canonical_sha256(selected_rows) != meta['subset']['selected_rows_sha256']:
        raise ValueError('Evaluated sample IDs/order differ from the locked hash subset')
    if truth is not None and meta['full_split'] and ids != truth.keys():
        raise ValueError('Claimed full-split evaluation omitted or added source samples')
    means = {name: value / count for name, value in sums.items()}
    user_means = {name: sum(values[name] / n for n, values in users.values()) / len(users) for name in sums}
    group_means = {group: dict(examples=n, metrics={name: value / n for name, value in values.items()})
                   for group, (n, values) in groups.items()}
    for saved, actual in ((meta['metrics'], means), (meta['user_average'], user_means)):
        if set(saved) != set(actual):
            raise ValueError('Summary metric names differ')
        for name, value in actual.items():
            error = abs(value - saved[name]) if math.isfinite(saved[name]) else math.inf
            max_error = max(max_error, error)
            mismatch_count += error > tolerance
    for group, values in group_means.items():
        saved = meta['groups'][group]
        if saved['examples'] != values['examples'] or set(saved['metrics']) != set(values['metrics']):
            raise ValueError('Summary group membership or metrics differ')
        for name, value in values['metrics'].items():
            error = abs(value - saved['metrics'][name]) if math.isfinite(saved['metrics'][name]) else math.inf
            max_error = max(max_error, error)
            mismatch_count += error > tolerance
    return dict(format='stage09_independent_metric_audit_v1', passed=mismatch_count == 0,
                examples=count, users=len(users), source_labels_verified=truth is not None,
                predictions_sha256=file_sha256(predictions_path), summary_sha256=file_sha256(Path(predictions_path).parent / 'summary.json'),
                tolerance=tolerance, mismatches=mismatch_count,
                max_absolute_error=max_error if math.isfinite(max_error) else None,
                nonfinite_saved_metric=not math.isfinite(max_error),
                metrics=means, user_average=user_means, groups=group_means,
                implementation='Independent Python math/set replay; no production metrics/decoding import.')


def cluster_bootstrap(user_counts, baseline_sums, treatment_sums, *, repeats=10000, seed=2024, confidence=.95):
    """Paired users, preserving all their samples and the random denominator."""
    counts = np.asarray(user_counts, dtype=np.int64)
    a, b = np.asarray(baseline_sums, dtype=np.float64), np.asarray(treatment_sums, dtype=np.float64)
    if counts.ndim != 1 or not len(counts) or np.any(counts <= 0) or a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(counts):
        raise ValueError('Each bootstrap cluster needs positive count and aligned metric sums')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or repeats < 2 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap inputs')
    delta = b - a
    per_user = delta / counts[:, None]
    sample_draws = np.empty((repeats, a.shape[1]), dtype=np.float64)
    user_draws = np.empty_like(sample_draws)
    generator = np.random.default_rng(seed)
    # At most 1e6 user indices plus small [repeats, metrics] result buffers.
    chunk = max(1, min(64, 1000000 // len(counts)))
    for start in range(0, repeats, chunk):
        end = min(repeats, start + chunk)
        selected = generator.integers(0, len(counts), size=(end - start, len(counts)))
        denominator = counts[selected].sum(axis=1)
        for metric in range(a.shape[1]):
            sample_draws[start:end, metric] = delta[:, metric][selected].sum(axis=1) / denominator
            user_draws[start:end, metric] = per_user[:, metric][selected].mean(axis=1)
    bounds = ((1 - confidence) / 2, 1 - (1 - confidence) / 2)
    return dict(sample_baseline=a.sum(axis=0) / counts.sum(), sample_treatment=b.sum(axis=0) / counts.sum(),
                sample_delta=delta.sum(axis=0) / counts.sum(), sample_ci=np.quantile(sample_draws, bounds, axis=0).T,
                user_baseline=(a / counts[:, None]).mean(axis=0), user_treatment=(b / counts[:, None]).mean(axis=0),
                user_delta=per_user.mean(axis=0), user_ci=np.quantile(user_draws, bounds, axis=0).T)


def paired_comparison(baseline_path, treatment_path, *, repeats=10000, seed=2024, confidence=.95, metric_names=None):
    """Compare one training-seed pair; all values are independently replayed."""
    a_meta, arguments = _metadata(baseline_path)
    b_meta, b_arguments = _metadata(treatment_path)
    for field in ('split', 'beam', 'architecture', 'seed', 'catalog_size', 'metric_protocol', 'dataset_binding', 'subset', 'diagnostic_prefix_limit'):
        if a_meta.get(field) != b_meta.get(field):
            raise ValueError(f'Paired evaluation protocol differs: {field}')
    if arguments != b_arguments:
        raise ValueError('Paired metric settings differ')
    if metric_names is None:
        metric_names = [f'list/{name}@{arguments["list_length"]}' for name in ('ndcg', 'recall', 'hr')]
        metric_names += [f'aggregate/{name}@{k}' for k in arguments['aggregate_ks'] for name in ('ndcg', 'recall', 'hr')]
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError('Metrics must be nonempty and distinct')
    baseline = {}
    for row in _rows(baseline_path):
        key = row['sample_id']
        if key in baseline:
            raise ValueError('Duplicate baseline sample ID')
        values = recompute_row(row, **arguments)
        baseline[key] = (row['user_id'], row['targets'], row['cold_target'], np.array([values[name] for name in metric_names]))
    expected = len(baseline)
    users, groups = {}, {}
    for row in _rows(treatment_path):
        reference = baseline.pop(row['sample_id'], None)
        if reference is None:
            raise ValueError('Treatment has duplicate/unmatched sample ID')
        user, targets, cold, a_values = reference
        if (user, targets, cold) != (row['user_id'], row['targets'], row['cold_target']):
            raise ValueError('Paired sample user, labels, or cold-start group differ')
        values = recompute_row(row, **arguments)
        b_values = np.array([values[name] for name in metric_names])
        accum = users.setdefault(user, [0, np.zeros(len(metric_names)), np.zeros(len(metric_names))])
        accum[0] += 1
        accum[1] += a_values
        accum[2] += b_values
        group = groups.setdefault('cold_target' if cold else 'seen_targets', [0, np.zeros(len(metric_names)), np.zeros(len(metric_names))])
        group[0] += 1
        group[1] += a_values
        group[2] += b_values
    if baseline or not expected or expected != a_meta['examples'] or expected != b_meta['examples']:
        raise ValueError('Paired sample sets/counts are not identical')
    ordered = [users[user] for user in sorted(users)]
    estimates = cluster_bootstrap([v[0] for v in ordered], [v[1] for v in ordered], [v[2] for v in ordered],
                                  repeats=repeats, seed=seed, confidence=confidence)

    def report(kind):
        result = {}
        for i, name in enumerate(metric_names):
            a, b, delta = (float(estimates[f'{kind}_{field}'][i]) for field in ('baseline', 'treatment', 'delta'))
            result[name] = dict(baseline=a, treatment=b, delta=delta,
                                relative_delta=delta / a if a else None,
                                confidence_interval=estimates[f'{kind}_ci'][i].tolist())
        return result

    return dict(format='stage09_paired_user_bootstrap_v1', examples=expected, users=len(users),
                bootstrap=dict(repeats=repeats, seed=seed, confidence=confidence, quantile='linear_percentile',
                    unit='user; draw U users with replacement, carrying every sample of each selected user'),
                interpretation='One training-seed comparison. User bootstrap does not establish cross-seed stability.',
                primary_estimand='sample mean; sum resampled user metric sums divided by sum resampled user sample counts',
                secondary_estimand='mean of per-user sample means; each user has equal weight',
                primary=report('sample'), user_average=report('user'),
                groups={group: dict(examples=n, metrics={name: dict(baseline=float(a[i] / n), treatment=float(b[i] / n),
                              delta=float((b[i] - a[i]) / n)) for i, name in enumerate(metric_names)})
                        for group, (n, a, b) in groups.items()},
                inputs=dict(baseline_sha256=file_sha256(baseline_path), treatment_sha256=file_sha256(treatment_path)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('recompute')
    replay.add_argument('--predictions', required=True)
    replay.add_argument('--dataset', help='Also verify each sample/index/target against the bound original split')
    replay.add_argument('--tolerance', type=float, default=2e-6)
    compare = commands.add_parser('compare')
    compare.add_argument('--baseline', required=True)
    compare.add_argument('--treatment', required=True)
    compare.add_argument('--repeats', type=int, default=10000)
    compare.add_argument('--seed', type=int, default=2024)
    for subparser in (replay, compare):
        subparser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'recompute':
        result = audit_predictions(args.predictions, dataset_dir=args.dataset, tolerance=args.tolerance)
    else:
        result = paired_comparison(args.baseline, args.treatment, repeats=args.repeats, seed=args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(path), examples=result['examples'], users=result['users'], passed=result.get('passed')), ensure_ascii=False))
    if result.get('passed') is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
