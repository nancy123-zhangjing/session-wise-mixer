"""Recovery validation uses the unchanged Stage09 decoder and metric functions."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import KuaiRecListDataset
from .decoding import generate_lists
from .metrics import evaluate_generation
from .recovery_diagnostics import diagnostic_context
from .run import make_catalog, microbatches, resolved
from .training import sha256


def subset_manifest(dataset_dir, size=8192):
    import pyarrow.parquet as pq
    path = Path(dataset_dir) / 'validation.parquet'
    ids = pq.read_table(path, columns=['sample_id'])['sample_id'].to_pylist()
    if not 0 < size <= len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid diagnostic size or duplicate sample IDs')
    ranked = sorted((hashlib.sha256(value.encode('utf-8')).hexdigest(), value, i)
                    for i, value in enumerate(ids))[:size]
    return dict(format='stage09_h1_recovery_subset_v1', count=size, population=len(ids),
                validation_sha256=sha256(path), selection='sha256_utf8_sample_id_then_sample_id',
                rows=[dict(index=i, sample_id=value, selection_sha256=digest)
                      for digest, value, i in ranked])


def load_subset(path, dataset_dir):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = subset_manifest(dataset_dir, saved['count'])
    if saved != expected:
        raise ValueError('Recovery diagnostic subset identity mismatch')
    return saved


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def extract_subset(source, manifest):
    """Only extract existing full predictions, not another model forward pass."""
    wanted = {row['sample_id'] for row in manifest['rows']}
    seen, sums = set(), {}
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['sample_id'] not in wanted:
                continue
            if row['sample_id'] in seen:
                raise ValueError('Repeated diagnostic sample in full evaluation')
            seen.add(row['sample_id'])
            for name, value in row['metrics'].items():
                sums[name] = sums.get(name, 0.0) + value
    if seen != wanted:
        raise ValueError('Full evaluation does not contain the locked diagnostic samples')
    return dict(examples=len(seen), metrics={k: v / len(seen) for k, v in sums.items()})


def evaluate_recovery(trainer, assets, data, destination, *, indices=None, subset=None):
    """Item-ID evaluation, matching original run.evaluate; validation only."""
    if data.split != 'validation':
        raise ValueError('Recovery experiments must not evaluate test')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    selected = np.arange(len(data)) if indices is None else np.asarray(indices)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or len(selected) == 0 or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(data))):
        raise ValueError('Evaluation indices must be unique in-range integers')
    selected = selected.astype(np.int64, copy=False)
    cfg = trainer.config
    sums, users, counters = {}, {}, {}
    count = 0
    started = time.perf_counter()
    with diagnostic_context(trainer):
        catalog = make_catalog(trainer.alignment, assets)
        with gzip.open(destination / 'per_sample.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for batch in microbatches(data, selected, cfg['evaluation']['batch_size']):
                ids = batch['history_item_ids'].to(trainer.device)
                generated = generate_lists(trainer.model, trainer.alignment.hard_codes[ids],
                    batch['history_mask'].to(trainer.device), catalog, route=cfg['architecture'],
                    list_length=cfg['list_length'], beam_size=cfg['evaluation']['val_beam'])
                metrics = evaluate_generation(generated, batch['target_item_ids'],
                    catalog_item_ids=torch.arange(1, assets.n_items + 1),
                    aggregate_ks=cfg['evaluation']['aggregate_ks'],
                    max_candidate_lists=cfg['evaluation']['candidate_lists'])
                rows = {k: v.cpu().tolist() for k, v in metrics.items()}
                predictions = generated.item_ids.cpu().tolist()
                scores = generated.scores.cpu().tolist()
                beam_masks = generated.beam_mask.cpu().tolist()
                complete_masks = generated.complete_mask.cpu().tolist()
                for name, value in generated.counters.items():
                    if isinstance(value, (int, float)):
                        counters[name] = counters.get(name, 0) + value
                for i, sample_id in enumerate(batch['sample_id']):
                    values = {k: v[i] for k, v in rows.items()}
                    user = int(batch['user_id'][i])
                    user_row = users.setdefault(user, dict(count=0, sums={}))
                    user_row['count'] += 1
                    for name, value in values.items():
                        sums[name] = sums.get(name, 0.0) + value
                        user_row['sums'][name] = user_row['sums'].get(name, 0.0) + value
                    stream.write(json.dumps(dict(sample_id=sample_id, dataset_index=int(selected[count]),
                        user_id=user, cold_target=bool(batch['target_cold_start'][i].any()),
                        targets=batch['target_item_ids'][i].tolist(), predictions=predictions[i],
                        scores=[v if np.isfinite(v) else None for v in scores[i]],
                        beam_mask=beam_masks[i], complete_mask=complete_masks[i], metrics=values), allow_nan=False) + '\n')
                    count += 1
                if count % 1024 < len(batch['sample_id']):
                    print(json.dumps(dict(event='recovery_validation', processed=count, total=len(selected),
                                          seconds=time.perf_counter() - started)), flush=True)
    result = dict(split='validation', examples=count, full_split=indices is None, subset=subset,
                  dataset_binding=data.binding, catalog_size=assets.n_items,
                  metric_protocol=dict(list_length=cfg['list_length'], aggregate_ks=cfg['evaluation']['aggregate_ks'],
                      candidate_lists=cfg['evaluation']['candidate_lists'], pad_id=0,
                      target_relevance='unordered_set_of_five_future_distinct_items',
                      aggregation=cfg['evaluation']['aggregation']),
                  beam=cfg['evaluation']['val_beam'], inference_precision='fp32', architecture=cfg['architecture'],
                  seed=cfg['seed'], r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                  seconds=time.perf_counter() - started, decode_counters=counters,
                  metrics={k: v / count for k, v in sums.items()},
                  user_average={k: sum(row['sums'][k] / row['count'] for row in users.values()) / len(users)
                                for k in sums})
    if count != len(selected):
        raise RuntimeError('Incomplete validation')
    write_json(destination / 'summary.json', result)
    return result
