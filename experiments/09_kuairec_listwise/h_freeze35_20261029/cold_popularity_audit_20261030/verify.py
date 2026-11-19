"""Independent CPU audit of saved cold/popularity diagnostic outputs.

Does not import the analysis script, production metrics/decoding, or torch.
Recomputes all array-based statistics and bootstrap draws; inspects a fixed
subset of original prediction rows for score-order aggregation provenance.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(2)
pa.set_io_thread_count(2)

HERE = Path(__file__).resolve().parent
GROUPS = ['never_seen', 'history_only', 'target_head', 'target_mid', 'target_tail']
CLASSES = {name: [i] for i, name in enumerate(GROUPS)} | {'target_seen': [2, 3, 4]}
KVALUES = (5, 20, 50)
TOL = 2e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)


def close(actual, expected, label, tol=TOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape, label + ': shape')
    check(np.isfinite(actual).all() and np.isfinite(expected).all(), label + ': finite')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    check(error <= tol, f'{label}: error={error}')
    return error


def rank_and_credit(predictions, targets, k):
    # Search each target's first prediction rank independently. Later duplicates
    # never earn additional credit, and PAD cannot equal a positive target.
    ranks = np.zeros(targets.shape, dtype=np.int16)
    for pos in range(predictions.shape[1] - 1, -1, -1):
        ranks[targets == predictions[:, pos, None]] = pos + 1
    hit = (ranks > 0) & (ranks <= k)
    gain = np.zeros(targets.shape, dtype=np.float64)
    gain[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return ranks, hit, gain


def raw_rank50(row, catalog_size):
    candidates = []
    n = len(row['predictions'])
    check(all(len(row[key]) == n for key in ('scores', 'beam_mask', 'complete_mask')), 'raw candidate lengths')
    for idx, (items, score, present, complete) in enumerate(zip(
            row['predictions'], row['scores'], row['beam_mask'], row['complete_mask'])):
        check(len(items) == 5 and all(type(x) is int and 0 <= x <= catalog_size for x in items), 'raw item IDs')
        finite = score is not None and type(score) in (int, float) and math.isfinite(score)
        check(type(present) is bool and type(complete) is bool and present == finite, 'raw finite mask')
        valid = present and len(set(items)) == 5 and all(1 <= x <= catalog_size for x in items)
        check(complete == valid, 'raw complete mask')
        if valid:
            candidates.append((-float(score), idx, items))
    candidates.sort(key=lambda x: (x[0], x[1]))
    result, present = [], set()
    for _, _, items in candidates[:32]:
        for item in items:
            if item not in present:
                result.append(item)
                present.add(item)
                if len(result) == 50:
                    return result
    return result + [0] * (50 - len(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=HERE / 'initial_r1')
    parser.add_argument('--output', type=Path, default=HERE / 'VERIFY.json')
    args = parser.parse_args()
    root = args.result_dir.resolve()
    check(not args.output.exists(), 'Do not overwrite an existing audit')
    complete = read(root / 'COMPLETE.json')
    check(complete['status'] == 'passed', 'analysis completion')
    expected_files = complete['files']
    for name, digest in expected_files.items():
        check(sha(root / name) == digest, 'output hash: ' + name)
    request = read(root / 'REQUEST.json')
    check(sha(HERE / 'analyze.py') == complete['script_sha256'], 'analysis script hash')
    for path, digest in request['source_sha256'].items():
        check(sha(path) == digest, 'input hash: ' + path)
    names, dataset = request['models'], Path(request['dataset'])
    data = np.load(root / 'shared_arrays.npz', allow_pickle=False)
    check(data['group_names'].tolist() == GROUPS and request['class_order'] == GROUPS, 'class order')
    seen_json = read(dataset / 'train_seen.json')
    size = len(data['fine_group']) - 1
    train, hist = np.zeros(size + 1, dtype=np.int64), np.zeros(size + 1, dtype=np.int64)
    train_rows = 0
    for batch in pq.ParquetFile(dataset / 'train.parquet').iter_batches(
            batch_size=16384, columns=['target_item_ids', 'history_item_ids']):
        train_rows += batch.num_rows
        for key, counts in [('target_item_ids', train), ('history_item_ids', hist)]:
            # Arrow flatten preserves repeats in the actual training examples.
            ids = batch.column(key).flatten().to_numpy(zero_copy_only=False)
            counts += np.bincount(ids, minlength=size + 1)
    check(train_rows == request['train_rows'] == 709443, 'training row count')
    check(int(train.sum()) == 5 * train_rows, 'training target count')
    check(np.array_equal(train, data['train_count']) and np.array_equal(hist, data['history_count']), 'training frequencies')
    seen = (train + hist) > 0
    check(not seen[0], 'PAD absent from train')
    check(np.flatnonzero(seen).tolist() == seen_json['item_ids'] and np.array_equal(seen, data['seen']), 'seen reference')
    ranked = sorted([i for i in range(1, size + 1) if train[i]], key=lambda i: (-int(train[i]), i))
    nhead, nheadmid = math.ceil(len(ranked) / 10), math.ceil(len(ranked) / 2)
    fine = np.full(size + 1, 0, dtype=np.int8)
    fine[seen] = 1
    for group, ids in [(2, ranked[:nhead]), (3, ranked[nhead:nheadmid]), (4, ranked[nheadmid:])]:
        fine[ids] = group
    fine[0] = -1
    check(np.array_equal(fine, data['fine_group']), 'five item groups')
    check(np.bincount(fine[1:], minlength=5).tolist() == [1914, 799, 802, 3206, 4007], 'class catalog counts')
    with (root / 'train_item_counts.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    check(len(rows) == size, 'training CSV rows')
    for i, row in enumerate(rows, 1):
        check(int(row['item_id']) == i and int(row['train_target_occurrences']) == train[i]
              and int(row['train_history_occurrences']) == hist[i] and row['fine_class'] == GROUPS[fine[i]], 'training CSV row')
        check(row['coarse_class'] == ('target_seen' if fine[i] >= 2 else GROUPS[fine[i]]), 'coarse CSV group')

    truth = pq.read_table(dataset / 'validation.parquet', columns=['sample_id', 'user_id', 'target_item_ids', 'target_timestamp_us']).to_pydict()
    targets, users = np.asarray(truth['target_item_ids']), np.asarray(truth['user_id'])
    starts = np.asarray([v[0] for v in truth['target_timestamp_us']])
    sample_ids = np.asarray(truth['sample_id'])
    check(np.array_equal(targets, data['targets']) and np.array_equal(users, data['users'])
          and np.array_equal(starts, data['starts']) and np.array_equal(sample_ids, data['sample_id']), 'full validation source identities')
    n = len(users)
    check(n == 71657 and len(set(sample_ids)) == n, 'validation population')
    check(np.all(np.diff(np.sort(targets, axis=1), axis=1) > 0), 'unique five targets')
    check(request['cutoff_us'] == 1598745441188500, 'locked temporal cutoff')
    masks = {'early': starts < request['cutoff_us'], 'late': starts >= request['cutoff_us'], 'full': np.ones(n, bool)}
    check([int(m.sum()) for m in masks.values()] == [38644, 33013, 71657], 'time populations')
    groups = fine[targets]
    check(np.array_equal(groups, data['target_group']), 'target groups')
    uids, inverse = np.unique(users, return_inverse=True)
    check(np.array_equal(uids, data['user_ids']) and np.array_equal(inverse, data['inverse_users']), 'user identity arrays')
    counts = np.stack([np.bincount(inverse[masks[s]], minlength=len(uids)) for s in ('early', 'late')])
    check(np.array_equal(counts, data['user_segment_count']), 'user segment counts')
    composition = read(root / 'DATA_COMPOSITION.json')
    for segment, mask in masks.items():
        view = composition[segment]
        check(view['samples'] == mask.sum() and view['users'] == len(np.unique(users[mask])), 'composition population')
        shares = []
        for label, ids in CLASSES.items():
            selected = np.isin(groups[mask], ids)
            group = view['classes'][label]
            check(group['target_occurrences'] == selected.sum() and group['unique_target_items'] == len(np.unique(targets[mask][selected])), 'composition group count')
            close(group['target_share'], selected.mean(), 'composition group share')
            if label in GROUPS:
                shares.append(group['target_share'])
        close(sum(shares), 1.0, 'composition share closure')

    metric_keys = request['metric_keys']
    feature_keys = request['feature_keys']
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, 6))
    close(ideal, request['idcg5'], 'IDCG')
    picks = set(np.random.default_rng(20241030).choice(n, 1024, replace=False).tolist())
    boundary = int(np.argmax(starts >= request['cutoff_us']))
    picks.update([0, n - 1, boundary, max(boundary - 1, 0)])
    results, sums_all, features_all, gains_all = {}, [], [], []
    max_metric_error, max_bootstrap_error = 0., 0.
    for name in names:
        array = np.load(root / f'arrays_{name}.npz', allow_pickle=False)
        report = read(root / f'MODEL_{name}.json')
        pred5, pred50 = array['pred5'], array['pred50']
        check(pred5.shape == (n, 5) and pred50.shape == (n, 50), 'prediction array shape')
        check(np.all((pred5 >= 0) & (pred5 <= size)) and np.all((pred50 >= 0) & (pred50 <= size)), 'prediction ID range')
        raw_checked, raw_total = 0, 0
        with gzip.open(request['points'][name]['predictions'], 'rt', encoding='utf-8') as file:
            for index, line in enumerate(file):
                raw_total += 1
                if index not in picks:
                    continue
                row = json.loads(line)
                check(row['dataset_index'] == index and row['sample_id'] == sample_ids[index]
                      and row['user_id'] == int(users[index]) and row['targets'] == targets[index].tolist(), 'raw sampled identity')
                check(row['predictions'][0] == pred5[index].tolist(), 'raw sampled best list')
                check(raw_rank50(row, size) == pred50[index].tolist(), 'raw sampled score aggregation')
                raw_checked += 1
        check(raw_total == n and raw_checked == len(picks), 'raw prediction population and sample')
        ranks5, hits5, gain5 = rank_and_credit(pred5, targets, 5)
        ranks50, _, gain50 = rank_and_credit(pred50, targets, 50)
        check(np.array_equal(ranks5, array['rank5']) and np.array_equal(ranks50, array['rank50']), 'all target ranks')
        metrics, gain_by_k = [], {}
        for k, pred in [(5, pred5), (10, pred50), (20, pred50), (50, pred50)]:
            _, hit, gain = rank_and_credit(pred, targets, k)
            metrics.extend([gain.sum(axis=1) / ideal, np.any(hit, axis=1).astype(float), hit.sum(axis=1) / 5])
            gain_by_k[k] = gain
        metrics = np.stack(metrics, axis=1)
        max_metric_error = max(max_metric_error, close(metrics, array['metric_values'], 'all sample metrics'))
        for j, key in enumerate(metric_keys):
            close(float(metrics[:, j].mean()), request['points'][name]['metrics'][key], 'original independently audited metric')
        contributions = np.stack([np.stack([(gain_by_k[k] * (groups == g)).sum(axis=1) / ideal for g in range(5)], axis=1) for k in KVALUES], axis=1)
        close(contributions, array['class_ndcg'], 'all class NDCG contributions')
        features = np.concatenate([np.column_stack([contributions[:, ki].sum(axis=1), contributions[:, ki], contributions[:, ki, 2:].sum(axis=1)]) for ki in range(3)], axis=1)
        model_sums = np.zeros((2, len(uids), len(feature_keys)))
        for si, seg in enumerate(('early', 'late')):
            for j in range(len(feature_keys)):
                model_sums[si, :, j] = np.bincount(inverse[masks[seg]], weights=features[masks[seg], j], minlength=len(uids))
        close(model_sums, array['user_segment_sums'], 'all user feature sums')
        for segment, mask in masks.items():
            view = report['segments'][segment]
            for j, key in enumerate(metric_keys):
                close(metrics[mask, j].mean(), view['metrics'][key], 'segment metric')
            for label, ids in CLASSES.items():
                selected = np.isin(groups[mask], ids)
                occurrences = int(selected.sum())
                check(view['classes'][label]['target_occurrences'] == occurrences, 'class occurrence count')
                for k in KVALUES:
                    v = view['classes'][label]['metrics'][str(k)]
                    gain = gain_by_k[k][mask]
                    total = float(gain[selected].sum())
                    hits = int((gain[selected] > 0).sum())
                    check(v['target_hits'] == hits, 'class target hits')
                    close(v['discounted_gain_sum'], total, 'class gain sum', tol=2e-9)
                    close(v['ndcg_contribution'], total / (mask.sum() * ideal), 'class NDCG mean contribution')
                    if occurrences:
                        close(v['target_hit_rate'], hits / occurrences, 'class target hit rate')
                        close(v['mean_discounted_gain_per_target'], total / occurrences, 'class mean gain')
                    else:
                        check(v['target_hit_rate'] is None and v['mean_discounted_gain_per_target'] is None, 'empty class missingness')
            for ki, k in enumerate(KVALUES):
                key = f'{"list" if k == 5 else "aggregate"}/ndcg@{k}'
                close(sum(view['classes'][g]['metrics'][str(k)]['ndcg_contribution'] for g in GROUPS), view['metrics'][key], 'five-class NDCG closure')
            for label, predicted in [('list5', pred5), ('aggregate50', pred50)]:
                v = view['predictions'][label]
                slots = predicted[mask].reshape(-1)
                valid = slots[slots > 0]
                freq = np.bincount(valid, minlength=size + 1)
                order = sorted(np.flatnonzero(freq), key=lambda i: (-freq[i], i))[:10]
                check(v['real_item_slots'] == len(valid) and v['total_slots'] == len(slots), 'prediction slot counts')
                check(v['unique_recommended_items'] == np.count_nonzero(freq), 'prediction unique count')
                close(v['coverage'], len(valid) / len(slots), 'prediction coverage')
                close(v['top10_concentration'], freq[order].sum() / len(valid), 'top10 prediction concentration')
                check([x['item_id'] for x in v['top10_items']] == order, 'top10 identities')
                close(v['hhi'], np.square(freq / len(valid)).sum(), 'prediction HHI')
                close(v['mean_log1p_train_target_frequency'], np.log1p(train[valid]).mean(), 'prediction train frequency')
                for group, ids in CLASSES.items():
                    close(v['class_shares'][group], np.isin(fine[valid], ids).mean(), 'prediction class share')
        results[name] = {'array_rows_checked': n, 'raw_rows_checked': raw_checked, 'raw_rows_counted': raw_total,
                         'class_contribution_closure_passed': True, 'prediction_bias_metrics_passed': True}
        sums_all.append(model_sums)
        features_all.append(features)
        gains_all.append(gain_by_k)
        print(json.dumps({'event': 'verified_model', 'model': name, **results[name]}), flush=True)

    paired = read(root / 'PAIRED_BOOTSTRAP.json')
    saved_draws = np.load(root / 'bootstrap_draws.npz', allow_pickle=False)
    check(saved_draws['model_names'].tolist() == names and saved_draws['feature_keys'].tolist() == feature_keys, 'bootstrap axis identity')
    pair_indices = {}
    for label in paired['full']:
        possibilities = [(i, j) for i, baseline in enumerate(names) for j, treatment in enumerate(names) if treatment + '_minus_' + baseline == label]
        check(len(possibilities) == 1, 'unique model pair label')
        pair_indices[label] = possibilities[0]
    for segment in ('early', 'late', 'full', 'late_minus_early'):
        rng = np.random.default_rng(2024)
        if segment == 'late_minus_early':
            selected_users = np.arange(len(uids))
            selected_counts = counts
            observed = np.stack([f[masks['late']].mean(axis=0) - f[masks['early']].mean(axis=0) for f in features_all])
        else:
            si = 0 if segment == 'early' else 1
            pool_counts = counts.sum(axis=0) if segment == 'full' else counts[si]
            selected_users = np.flatnonzero(pool_counts)
            selected_counts = pool_counts[selected_users]
            observed = np.stack([f[masks[segment]].mean(axis=0) for f in features_all])
        draws = np.empty((2000, len(names), len(feature_keys)))
        for iteration in range(2000):
            # This implementation sums repeated sampled-user records directly,
            # unlike the analysis script's multinomial weight matrix products.
            selection = rng.integers(0, len(selected_users), size=len(selected_users))
            picked = selected_users[selection]
            if segment == 'late_minus_early':
                early_den, late_den = counts[0, picked].sum(), counts[1, picked].sum()
                check(early_den > 0 and late_den > 0, 'joint bootstrap denominators')
                for mi, ss in enumerate(sums_all):
                    draws[iteration, mi] = ss[1, picked].sum(axis=0) / late_den - ss[0, picked].sum(axis=0) / early_den
            else:
                denominator = selected_counts[selection].sum()
                for mi, ss in enumerate(sums_all):
                    combined = ss.sum(axis=0) if segment == 'full' else ss[si]
                    draws[iteration, mi] = combined[picked].sum(axis=0) / denominator
        max_bootstrap_error = max(max_bootstrap_error, close(draws, saved_draws[segment], 'independent bootstrap draws'))
        for label, (bi, ti) in pair_indices.items():
            bounds = np.quantile(draws[:, ti] - draws[:, bi], [.025, .975], axis=0)
            for j, feature in enumerate(feature_keys):
                row = paired[segment][label][feature]
                close([row['baseline'], row['treatment'], row['delta'], row['ci_low'], row['ci_high']],
                      [observed[bi, j], observed[ti, j], observed[ti, j] - observed[bi, j], bounds[0, j], bounds[1, j]], 'paired bootstrap summary')
        print(json.dumps({'event': 'verified_bootstrap', 'segment': segment}), flush=True)

    decompositions = read(root / 'DECOMPOSITION.json')
    for label, (bi, ti) in pair_indices.items():
        for k in KVALUES:
            report = decompositions[label][str(k)]
            composition_total, within_total = 0., 0.
            for gi, group in enumerate(GROUPS):
                values = []
                for seg in ('early', 'late'):
                    mask = masks[seg]
                    take = groups[mask] == gi
                    count = int(take.sum())
                    total = float((gains_all[ti][k][mask] - gains_all[bi][k][mask])[take].sum())
                    values.append((count / (5 * mask.sum()), total / count if count else None))
                (we, de), (wl, dl) = values
                de0, dl0 = 0. if de is None else de, 0. if dl is None else dl
                composition_effect = (wl - we) * (dl0 + de0) * 2.5 / ideal
                within_effect = (wl + we) * (dl0 - de0) * 2.5 / ideal
                r = report['classes'][gi]
                check(r['fine_class'] == group and r['both_periods_observed'] == (de is not None and dl is not None), 'decomposition class identity/support')
                close([r['weight_early'], r['weight_late'], r['composition_ndcg_change'], r['within_class_ndcg_change']],
                      [we, wl, composition_effect, within_effect], 'decomposition class formula')
                for key, val in [('delta_mean_raw_gain_early', de), ('delta_mean_raw_gain_late', dl)]:
                    check(r[key] is None, 'empty effect support') if val is None else close(r[key], val, 'conditional raw effect')
                composition_total += composition_effect
                within_total += within_effect
            feature = f'{"list" if k == 5 else "aggregate"}/ndcg@{k}/overall'
            observed = paired['late_minus_early'][label][feature]['delta']
            close([report['composition_change'], report['within_class_change'], report['observed_advantage_change']],
                  [composition_total, within_total, observed], 'decomposition totals')
            close(composition_total + within_total, observed, 'symmetric decomposition closure')
            check(report['causal'] is False, 'descriptive interpretation boundary')

    for name, digest in expected_files.items():
        check(sha(root / name) == digest, 'output unchanged after audit: ' + name)
    audit = {'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
             'result_dir': str(root), 'analysis_complete_sha256': sha(root / 'COMPLETE.json'),
             'verifier_sha256': sha(__file__), 'analysis_script_sha256': complete['script_sha256'],
             'full_source_validation_rows_checked': n, 'train_rows_independently_counted': train_rows,
             'catalog_class_counts': np.bincount(fine[1:], minlength=5).tolist(), 'models': results,
             'max_array_metric_abs_error': max_metric_error, 'max_bootstrap_draw_abs_error': max_bootstrap_error,
             'bootstrap': {'repeats': 2000, 'seed': 2024, 'all_model_feature_draws_independently_recomputed': True,
                           'temporal_contrast_same_global_user_draw': True},
             'target_contribution_and_symmetric_decomposition_closure': True,
             'read_only_source_hashes_and_output_hashes_verified': True,
             'scope': 'All validation source rows, train frequencies, saved-array metrics and classes, output concentration, user sums, bootstrap draws and symmetric decomposition; sampled raw prediction aggregation.',
             'limitations': ['Only a fixed subset of original raw prediction rows was re-parsed for aggregation; all saved prediction arrays were recomputed.',
                             'No model loading, forward inference, optimization, checkpoint restore, remote access, or GPU use.',
                             'Post-hoc selected-model validation diagnosis; confidence intervals exclude training seed and model selection uncertainty.'],
             'numpy_version': np.__version__, 'pyarrow_version': pa.__version__}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(audit, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'event': 'independent_verify_passed', 'output': str(args.output), 'max_metric_error': max_metric_error,
                      'max_bootstrap_error': max_bootstrap_error}), flush=True)


if __name__ == '__main__':
    main()
