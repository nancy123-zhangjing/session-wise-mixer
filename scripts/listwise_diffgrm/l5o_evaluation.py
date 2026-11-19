"""L5O adapter: the shared impression/EV evaluator selects the checkpoint."""
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.kuairec_data.l5o_metrics import slate_metrics
from .data import KuaiRecListDataset
from .decoding import generate_lists
from .training import rng_state, restore_rng


@torch.no_grad()
def evaluate_l5o(trainer, assets, *, split, destination, beam=None, limit=None, subset_manifest=None):
    from .run import resolved, make_catalog, microbatches
    if split not in ('validation', 'test') or subset_manifest is not None:
        raise ValueError('L5O evaluation requires validation/test; legacy subset manifests are incompatible')
    cfg, state = trainer.config, rng_state()
    filter_history = cfg['evaluation']['filter_history']
    data = KuaiRecListDataset(resolved(cfg['dataset']), split, assets=assets)
    if data.config.get('protocol_id') != 'kuairec_l5o_records_v1':
        raise ValueError('L5O metrics require the record-preserving L5O dataset')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    count = len(data) if limit is None else min(int(limit), len(data))
    if count < 1:
        raise ValueError('Empty evaluation')
    width = beam if beam is not None else cfg['evaluation']['val_beam' if split == 'validation' else 'test_beam']
    preds, targets, ratios = [], [], []
    started = time.perf_counter()
    try:
        trainer.model.eval()
        trainer.alignment.eval()
        catalog = make_catalog(trainer.alignment, assets)
        with gzip.open(destination / 'per_sample.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for batch in microbatches(data, np.arange(count), cfg['evaluation']['batch_size']):
                hs = trainer.alignment.hard_codes[batch['history_item_ids'].to(trainer.device)]
                out = generate_lists(trainer.model, hs, batch['history_mask'].to(trainer.device),
                    catalog, route=cfg['architecture'], list_length=cfg['list_length'], beam_size=width,
                    batched_frontier=cfg.get('execution', {}).get('batched_frontier', False),
                    excluded_item_ids=batch['history_item_ids'] if filter_history else None)
                p = out.best_item_ids.cpu().numpy()
                y, wr = batch['target_item_ids'].numpy(), batch['target_watch_ratio'].numpy()
                _, per = slate_metrics(p, y, wr, cfg['evaluation']['ev_min'], return_per_sample=True)
                preds.append(p); targets.append(y); ratios.append(wr)
                for i, sample in enumerate(batch['sample_id']):
                    row = dict(sample_id=sample, user_id=int(batch['user_id'][i]), prediction=p[i].tolist(),
                               targets=y[i].tolist(), target_watch_ratio=wr[i].tolist(),
                               metrics={k: float(v[i]) if np.isfinite(v[i]) else None for k, v in per.items()})
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
        p, y, wr = np.concatenate(preds), np.concatenate(targets), np.concatenate(ratios)
        metrics = slate_metrics(p, y, wr, cfg['evaluation']['ev_min'])
        result = dict(split=split, full_split=limit is None, diagnostic_prefix_limit=limit,
            examples=count, dataset_binding=data.binding, beam=width, inference_precision='fp32',
            r_updates=trainer.r_updates, t_updates=trainer.t_updates, metrics=metrics,
            sensitivity_ev2=slate_metrics(p, y, wr, 2.0), seconds=time.perf_counter() - started,
            metric_protocol=dict(name='l5o', selection='ev_ndcg@5', filter_history=filter_history,
                target_relevance='deduplicated target set; EV by record watch_ratio',
                ev_denominator='slates with nonempty EV target set', idcg='min(5, unique relevant items)'))
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        return result
    finally:
        restore_rng(state)
