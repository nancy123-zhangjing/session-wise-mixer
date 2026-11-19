"""CPU-only target-level cold/popularity diagnosis of saved validation predictions.

No model loading, inference, optimization, remote access, or protocol changes.
Every output directory must be new. Extra models use the same audited-point schema
as temporal_validation_r32_20261029/REQUEST.json['points'] (a JSON mapping).
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(2)
pa.set_io_thread_count(2)

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = ROOT / 'outputs/kuairec_listwise_v1/run1'
CUT = 1598745441188500
NITEM = 10728
N = 71657
GROUPS = ['never_seen', 'history_only', 'target_head', 'target_mid', 'target_tail']
CLASS_IDS = {name: [i] for i, name in enumerate(GROUPS)}
CLASS_IDS['target_seen'] = [2, 3, 4]
KS = (5, 20, 50)
DISCOUNT = 1 / np.log2(np.arange(50) + 2)
IDCG = float(DISCOUNT[:5].sum())
METRIC_KEYS = ['list/' + m + '@5' for m in ('ndcg', 'hr', 'recall')]
METRIC_KEYS += ['aggregate/' + m + '@' + str(k) for k in (10, 20, 50) for m in ('ndcg', 'hr', 'recall')]
FEATURE_KEYS = [f'{"list" if k == 5 else "aggregate"}/ndcg@{k}/{g}' for k in KS for g in ['overall'] + GROUPS + ['target_seen']]
BOUND = {}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def bind(p, expected=None):
    p = Path(p).resolve()
    actual = sha(p)
    if expected is not None and actual != expected:
        raise ValueError(f'Input hash mismatch: {p}')
    BOUND[str(p)] = actual
    return actual

def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def csvsave(p, rows):
    with Path(p).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def log(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False, allow_nan=False), flush=True)

def rank_targets(pred, targets):
    matched = targets[:, :, None] == pred[:, None, :]
    return np.where(matched.any(axis=2), matched.argmax(axis=2) + 1, 0).astype(np.uint8)

def gains(ranks, k):
    valid = (ranks > 0) & (ranks <= k)
    return np.where(valid, DISCOUNT[np.maximum(ranks.astype(np.int64) - 1, 0)], 0.)

def feature_matrix(contrib):
    columns = []
    for ki in range(3):
        columns.extend([contrib[:, ki].sum(axis=1), *[contrib[:, ki, gi] for gi in range(5)], contrib[:, ki, 2:].sum(axis=1)])
    return np.stack(columns, axis=1)

def bootstrap(sums, counts, repeats=2000, seed=2024):
    """Same user draw for all models/features; denominator carries sampled rows."""
    rng = np.random.default_rng(seed)
    u = len(counts)
    all_sums = np.concatenate(sums, axis=1)
    draws = np.empty((repeats, all_sums.shape[1]), dtype=np.float64)
    for start in range(0, repeats, 32):
        size = min(32, repeats - start)
        weights = np.zeros((size, u), dtype=np.float64)
        selected = rng.integers(0, u, size=(size, u))
        for j in range(size):
            weights[j] = np.bincount(selected[j], minlength=u)
        denominator = weights @ counts
        assert np.all(denominator > 0)
        draws[start:start + size] = (weights @ all_sums) / denominator[:, None]
    return draws.reshape(repeats, len(sums), len(FEATURE_KEYS))

def shift_bootstrap(user_sums, counts, repeats=2000, seed=2024):
    """Global validation users resampled jointly across early and late."""
    rng = np.random.default_rng(seed)
    u = counts.shape[1]
    draw = np.empty((repeats, len(user_sums), len(FEATURE_KEYS)), dtype=np.float64)
    flat = [np.concatenate([s[seg] for s in user_sums], axis=1) for seg in range(2)]
    for start in range(0, repeats, 32):
        size = min(32, repeats - start)
        weights = np.zeros((size, u), dtype=np.float64)
        selected = rng.integers(0, u, size=(size, u))
        for j in range(size):
            weights[j] = np.bincount(selected[j], minlength=u)
        den = [weights @ counts[seg] for seg in range(2)]
        assert all(np.all(d > 0) for d in den)
        delta = (weights @ flat[1]) / den[1][:, None] - (weights @ flat[0]) / den[0][:, None]
        draw[start:start + size] = delta.reshape(size, len(user_sums), len(FEATURE_KEYS))
    return draw

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE / 'initial_r1')
    parser.add_argument('--extra-models', type=Path)
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bind(__file__)
    temporal_path = TASK / 'temporal_validation_r32_20261029/REQUEST.json'
    bind(temporal_path)
    temporal = read(temporal_path)
    old_results_path = temporal_path.parent / 'RESULTS.json'
    bind(old_results_path)
    old_results = read(old_results_path)
    points = dict(temporal['points'])
    if args.extra_models:
        bind(args.extra_models)
        extra = read(args.extra_models)
        extra = extra.get('points', extra)
        assert not (set(points) & set(extra)), 'Extra model names may not overwrite locked models'
        points.update(extra)
    names = list(points)
    metas = {}
    for name, p in points.items():
        assert all(c.isalnum() or c in '_-' for c in name)
        for field in ('predictions', 'summary', 'checkpoint', 'audit_path'):
            bind(p[field], p['audit_sha256'] if field == 'audit_path' else p[field + '_sha256'])
        audit = read(p['audit_path'])
        assert audit['passed'] and audit['source_labels_verified'] and audit['mismatches'] == 0
        assert audit['predictions_sha256'] == p['predictions_sha256']
        assert audit['summary_sha256'] == p['summary_sha256']
        assert audit['metrics'] == p['metrics']
        meta = read(p['summary'])
        assert meta['split'] == 'validation' and meta['full_split'] and meta['examples'] == N
        assert meta['beam'] == 32 and meta['inference_precision'] == 'fp32'
        assert meta['metric_protocol']['aggregation'] == 'list_score_order_then_item_position_first_occurrence'
        assert meta['metric_protocol']['list_length'] == 5 and meta['metric_protocol']['candidate_lists'] == 32
        assert meta['catalog_size'] == NITEM
        metas[name] = meta
    ref = metas[names[0]]
    for meta in metas.values():
        assert all(meta[f] == ref[f] for f in ('dataset_binding', 'metric_protocol', 'catalog_size'))
        assert meta.get('subset') is None and meta.get('diagnostic_prefix_limit') is None
    for file in ('train.parquet', 'validation.parquet', 'train_seen.json'):
        bind(DATA / file, ref['dataset_binding']['files'][file]['sha256'])
    bind(DATA / 'manifest.json', ref['dataset_binding']['manifest_sha256'])
    seen = np.zeros(NITEM + 1, dtype=bool)
    seen[read(DATA / 'train_seen.json')['item_ids']] = True
    train_count = np.zeros(NITEM + 1, dtype=np.int64)
    history_count = np.zeros(NITEM + 1, dtype=np.int64)
    train_rows = 0
    for batch in pq.ParquetFile(DATA / 'train.parquet').iter_batches(batch_size=32768, columns=['history_item_ids', 'target_item_ids']):
        b = batch.to_pydict()
        train_rows += len(b['target_item_ids'])
        train_count += np.bincount(np.asarray(b['target_item_ids'], dtype=np.int64).reshape(-1), minlength=NITEM + 1)
        h = np.concatenate([np.asarray(x, dtype=np.int64) for x in b['history_item_ids']])
        history_count += np.bincount(h, minlength=NITEM + 1)
    derived_seen = (history_count + train_count) > 0
    derived_seen[0] = False
    assert np.array_equal(derived_seen, seen)
    assert seen.sum() == 8814 and np.count_nonzero(train_count[1:]) == 8015
    ranked = sorted(np.flatnonzero(train_count[1:]) + 1, key=lambda i: (-train_count[i], i))
    fine = np.zeros(NITEM + 1, dtype=np.int8)
    fine[seen] = 1
    cut_head, cut_mid = math.ceil(.1 * len(ranked)), math.ceil(.5 * len(ranked))
    fine[ranked[:cut_head]] = 2
    fine[ranked[cut_head:cut_mid]] = 3
    fine[ranked[cut_mid:]] = 4
    fine[0] = -1
    assert np.bincount(fine[1:], minlength=5).tolist() == [1914, 799, 802, 3206, 4007]
    truth = pq.read_table(DATA / 'validation.parquet', columns=['sample_id', 'user_id', 'target_item_ids', 'target_timestamp_us']).to_pydict()
    sample_id = np.asarray(truth['sample_id'])
    users = np.asarray(truth['user_id'], dtype=np.int64)
    targets = np.asarray(truth['target_item_ids'], dtype=np.int32)
    starts = np.asarray([x[0] for x in truth['target_timestamp_us']], dtype=np.int64)
    assert len(users) == N and len(set(sample_id)) == N and targets.shape == (N, 5)
    assert np.all(np.diff(np.sort(targets, axis=1), axis=1) > 0)
    target_group = fine[targets]
    masks = {'early': starts < CUT, 'late': starts >= CUT, 'full': np.ones(N, dtype=bool)}
    assert [int(x.sum()) for x in masks.values()] == [38644, 33013, 71657]
    uid, inverse = np.unique(users, return_inverse=True)
    assert len(uid) == 4733
    user_count = np.stack([np.bincount(inverse[masks[s]], minlength=len(uid)) for s in ('early', 'late')])
    request = dict(format='target_cold_popularity_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        models=names, points=points, dataset=str(DATA), cutoff_us=CUT, rule='first_target_timestamp < cutoff: early; otherwise late',
        class_order=GROUPS, class_ids=CLASS_IDS, train_rows=train_rows,
        popularity=dict(source='retained train target occurrence count', ordering='count descending, internal item ID ascending',
                        target_seen_items=len(ranked), head_count=cut_head, mid_count=cut_mid-cut_head, tail_count=len(ranked)-cut_mid),
        list_length=5, idcg5=IDCG, metric_keys=METRIC_KEYS, feature_keys=FEATURE_KEYS,
        bootstrap=dict(repeats=2000, seed=2024, unit='paired user clusters', estimand='sample mean',
                       segment_rule='sample users appearing in each segment', temporal_contrast='same global-user draw for both segments'),
        source_sha256=dict(BOUND), only_cpu_statistics=True, training_or_inference=False,
        python=sys.version, numpy=np.__version__, pyarrow=pa.__version__, argv=sys.argv)
    save(out / 'REQUEST.json', request)
    np.savez_compressed(out / 'shared_arrays.npz', sample_id=sample_id, users=users, targets=targets,
                        starts=starts, fine_group=fine, target_group=target_group, train_count=train_count,
                        history_count=history_count, seen=seen, user_ids=uid, inverse_users=inverse,
                        user_segment_count=user_count, group_names=np.asarray(GROUPS), feature_keys=np.asarray(FEATURE_KEYS), metric_keys=np.asarray(METRIC_KEYS))
    csvsave(out / 'train_item_counts.csv', [dict(item_id=i, train_target_occurrences=int(train_count[i]), train_history_occurrences=int(history_count[i]),
        seen_in_training=bool(seen[i]), fine_class=GROUPS[fine[i]], coarse_class=('target_seen' if fine[i] >= 2 else GROUPS[fine[i]])) for i in range(1, NITEM + 1)])
    composition = {}
    for seg, mask in masks.items():
        composition[seg] = dict(samples=int(mask.sum()), users=int(np.unique(users[mask]).size), target_occurrences=int(mask.sum()*5), classes={})
        for label, groupids in CLASS_IDS.items():
            target_mask = np.isin(target_group[mask], groupids)
            items = targets[mask][target_mask]
            composition[seg]['classes'][label] = dict(unique_target_items=int(np.unique(items).size), target_occurrences=int(items.size),
                target_share=float(items.size/(mask.sum()*5)), catalog_class_items=int(np.isin(fine[1:], groupids).sum()))
    save(out / 'DATA_COMPOSITION.json', composition)
    log(event='data_bound', train_rows=train_rows, population={s:composition[s]['samples'] for s in masks})
    model_results, model_sums, model_features, raw_gains = {}, [], [], []
    for name, p in points.items():
        pred5 = np.empty((N, 5), dtype=np.int32)
        pred50 = np.zeros((N, 50), dtype=np.int32)
        count = 0
        with gzip.open(p['predictions'], 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                assert i < N
                row = json.loads(line)
                assert row['dataset_index'] == i and row['sample_id'] == sample_id[i] and row['user_id'] == int(users[i])
                assert row['targets'] == targets[i].tolist() and row['cold_target'] == bool(np.any(~seen[targets[i]]))
                predictions, scores = row['predictions'], row['scores']
                beam, complete = row['beam_mask'], row['complete_mask']
                assert len(predictions) == len(scores) == len(beam) == len(complete)
                for items, score, present, done in zip(predictions, scores, beam, complete):
                    assert len(items) == 5 and all(type(v) is int and 0 <= v <= NITEM for v in items)
                    finite = score is not None and isinstance(score, (int, float)) and math.isfinite(score)
                    assert type(present) is bool and type(done) is bool and present == finite
                    assert done == (present and len(set(items)) == 5 and all(1 <= v <= NITEM for v in items))
                pred5[i] = predictions[0]
                order = sorted(range(len(scores)), key=lambda j: (-(scores[j] if scores[j] is not None else -math.inf), j))
                collected, already, used = [], set(), 0
                for j in order:
                    if not (beam[j] and complete[j]):
                        continue
                    if used == 32:
                        break
                    used += 1
                    for item in predictions[j]:
                        if item not in already:
                            already.add(item)
                            collected.append(item)
                            if len(collected) == 50:
                                break
                    if len(collected) == 50:
                        break
                pred50[i, :len(collected)] = collected
                count += 1
                if count % 20000 == 0:
                    log(event='prediction_progress', model=name, examples=count)
        assert count == N
        rank5, rank50 = rank_targets(pred5, targets), rank_targets(pred50, targets)
        gs = np.stack([gains(rank5 if k == 5 else rank50, k) for k in KS], axis=1)
        contribution = np.stack([(gs * (target_group == gi)[:, None, :]).sum(axis=2)/IDCG for gi in range(5)], axis=2)
        feature = feature_matrix(contribution)
        metrics = []
        for ranks, k in [(rank5, 5)] + [(rank50, k) for k in (10, 20, 50)]:
            hit = (ranks > 0) & (ranks <= k)
            metrics.extend([gains(ranks, k).sum(axis=1)/IDCG, hit.any(axis=1).astype(float), hit.mean(axis=1)])
        metric_values = np.stack(metrics, axis=1)
        means = metric_values.mean(axis=0)
        errors = {k:abs(float(means[j]) - p['metrics'][k]) for j, k in enumerate(METRIC_KEYS)}
        assert max(errors.values()) < 2e-12
        sums = np.zeros((2, len(uid), len(FEATURE_KEYS)), dtype=np.float64)
        for si, seg in enumerate(('early', 'late')):
            np.add.at(sums[si], inverse[masks[seg]], feature[masks[seg]])
        np.savez_compressed(out / f'arrays_{name}.npz', pred5=pred5, pred50=pred50, rank5=rank5, rank50=rank50,
                            metric_values=metric_values, class_ndcg=contribution, user_segment_sums=sums)
        result = dict(model=name, baseline_reproduction_absolute_error=errors, segments={})
        for seg, mask in masks.items():
            n = int(mask.sum())
            metrics_dict = {k:float(metric_values[mask, j].mean()) for j, k in enumerate(METRIC_KEYS)}
            if name in old_results['summaries'][seg]:
                assert max(abs(v-old_results['summaries'][seg][name][k]) for k,v in metrics_dict.items()) < 2e-12
            classes = {}
            for label, gids in CLASS_IDS.items():
                pick = np.isin(target_group[mask], gids)
                occurrences = int(pick.sum())
                classes[label] = dict(target_occurrences=occurrences, metrics={})
                for ki, k in enumerate(KS):
                    rr = (rank5 if k == 5 else rank50)[mask]
                    hits = int(((rr > 0) & (rr <= k) & pick).sum())
                    gain_sum = float((gs[mask, ki] * pick).sum())
                    classes[label]['metrics'][str(k)] = dict(target_hits=hits, target_hit_rate=hits/occurrences if occurrences else None,
                        discounted_gain_sum=gain_sum, mean_discounted_gain_per_target=gain_sum/occurrences if occurrences else None,
                        ndcg_contribution=gain_sum/(n*IDCG))
            predictions_stats = {}
            for label, predictions in [('list5', pred5), ('aggregate50', pred50)]:
                flat = predictions[mask].reshape(-1)
                valid = flat[flat > 0]
                c = np.bincount(valid, minlength=NITEM+1)
                order = sorted(np.flatnonzero(c), key=lambda item:(-c[item], item))[:10]
                den = len(valid)
                predictions_stats[label] = dict(real_item_slots=den, total_slots=len(flat), coverage=den/len(flat),
                    unique_recommended_items=int(np.count_nonzero(c)), top10_concentration=float(c[order].sum()/den) if den else None,
                    top10_items=[dict(item_id=int(item), count=int(c[item]), train_target_occurrences=int(train_count[item]), fine_class=GROUPS[fine[item]]) for item in order],
                    mean_log1p_train_target_frequency=float(np.log1p(train_count[valid]).mean()) if den else None,
                    hhi=float(np.sum((c/den)**2)) if den else None,
                    class_shares={g:float(np.isin(fine[valid], gids).sum()/den) if den else None for g,gids in CLASS_IDS.items()})
            result['segments'][seg] = dict(metrics=metrics_dict, classes=classes, predictions=predictions_stats)
        save(out / f'MODEL_{name}.json', result)
        model_results[name] = result
        model_sums.append(sums)
        model_features.append(feature)
        raw_gains.append(gs)
        log(event='model_finished', model=name, ndcg5=means[0], reproduction_max_error=max(errors.values()))
    pairs = [('H0_best_R32', 'H1_freeze_R32'), ('H1_R20', 'H1_freeze_R32'), ('H0_best_R32', 'H1_R20')]
    for name in names:
        if name not in temporal['points']:
            pairs.append((name, 'H1_freeze_R32'))
    paired = {}
    all_draws = {}
    for seg in ('early','late','full','late_minus_early'):
        if seg == 'late_minus_early':
            draws = shift_bootstrap(model_sums, user_count)
            observed = np.stack([f[masks['late']].mean(axis=0)-f[masks['early']].mean(axis=0) for f in model_features])
        else:
            mask = masks[seg]
            counts = user_count.sum(axis=0) if seg=='full' else user_count[0 if seg=='early' else 1]
            selected = counts > 0
            sums = [s.sum(axis=0) if seg=='full' else s[0 if seg=='early' else 1] for s in model_sums]
            draws = bootstrap([s[selected] for s in sums], counts[selected])
            observed = np.stack([f[mask].mean(axis=0) for f in model_features])
        all_draws[seg] = draws
        paired[seg] = {}
        for baseline,treatment in pairs:
            bi, ti = names.index(baseline), names.index(treatment)
            ci = np.quantile(draws[:,ti]-draws[:,bi], [.025,.975], axis=0)
            paired[seg][treatment+'_minus_'+baseline] = {k:dict(baseline=float(observed[bi,j]), treatment=float(observed[ti,j]),
                delta=float(observed[ti,j]-observed[bi,j]), ci_low=float(ci[0,j]), ci_high=float(ci[1,j])) for j,k in enumerate(FEATURE_KEYS)}
        log(event='bootstrap_finished', segment=seg)
    save(out/'PAIRED_BOOTSTRAP.json', paired)
    np.savez_compressed(out/'bootstrap_draws.npz', model_names=np.asarray(names), feature_keys=np.asarray(FEATURE_KEYS), **all_draws)
    decomposition = {}
    for baseline,treatment in pairs:
        bi, ti = names.index(baseline), names.index(treatment)
        label = treatment+'_minus_'+baseline
        decomposition[label] = {}
        for ki,k in enumerate(KS):
            rows = []
            for gi,g in enumerate(GROUPS):
                values = []
                for seg in ('early','late'):
                    mask = masks[seg]
                    pick = target_group[mask] == gi
                    occurrence = int(pick.sum())
                    gain_delta = float(((raw_gains[ti][mask,ki]-raw_gains[bi][mask,ki])*pick).sum())
                    values.append((occurrence/(mask.sum()*5), gain_delta/occurrence if occurrence else None))
                (we,de),(wl,dl) = values
                identified = de is not None and dl is not None
                # Zero-mass class convention provides algebraic closure only, never an identified mechanism.
                de0, dl0 = (0. if de is None else de), (0. if dl is None else dl)
                composition_effect = .5*(wl-we)*(dl0+de0)*5/IDCG
                within_effect = .5*(wl+we)*(dl0-de0)*5/IDCG
                rows.append(dict(fine_class=g, weight_early=we, weight_late=wl, delta_mean_raw_gain_early=de,
                    delta_mean_raw_gain_late=dl, both_periods_observed=identified,
                    composition_ndcg_change=composition_effect, within_class_ndcg_change=within_effect))
            total_comp = sum(r['composition_ndcg_change'] for r in rows)
            total_within = sum(r['within_class_ndcg_change'] for r in rows)
            observed = paired['late_minus_early'][label][f'{"list" if k==5 else "aggregate"}/ndcg@{k}/overall']['delta']
            assert abs(total_comp+total_within-observed)<2e-12
            decomposition[label][str(k)] = dict(classes=rows, composition_change=total_comp, within_class_change=total_within,
                observed_advantage_change=observed, closure_error=abs(total_comp+total_within-observed), scale_from_raw_target_gain=5/IDCG,
                causal=False, zero_mass_convention='zero conditional effect only for algebra; unidentified if absent in either period')
    save(out/'DECOMPOSITION.json', decomposition)
    csvrows = []
    for name,result in model_results.items():
        for seg,r in result['segments'].items():
            csvrows.append(dict(model=name, segment=seg, **r['metrics']))
    csvsave(out/'METRICS.csv', csvrows)
    for path,digest in BOUND.items():
        assert sha(path)==digest, 'Input changed during analysis: '+path
    save(out/'RESULTS.json', dict(status='passed', models=names, population={s:composition[s]['samples'] for s in masks},
        split='validation', input_hashes_rechecked=True, source_sha256=BOUND,
        limitations=['Post-hoc diagnosis on the validation set used for model selection.',
        'User-cluster intervals do not include seed or checkpoint-selection uncertainty.',
        'The two H1 models have T200; the old H0 R32 has T360 and is not a matched causal Mixer comparison.',
        'Aggregate metrics combine score-ordered candidate lists; they do not measure one generated 50-item list.',
        'Training unseen denotes behavior unseen in retained train samples; catalog/content may already be known.',
        'Descriptive target-composition decomposition cannot identify a unique causal mechanism.']))
    save(out/'COMPLETE.json', dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        only_cpu_statistics=True, training_or_inference=False, full_goal_complete=False,
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}, script_sha256=sha(__file__)))
    log(event='complete', output=str(out))

if __name__ == '__main__':
    main()
