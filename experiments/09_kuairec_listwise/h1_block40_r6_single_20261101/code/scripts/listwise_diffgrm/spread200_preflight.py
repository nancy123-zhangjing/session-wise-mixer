"""Bounded real-H100 acceptance; products are never formal continuation inputs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .data import KuaiRecListDataset
from .recovery import fingerprint
from .recovery35_artifacts import write_once
from .recovery_evaluation import evaluate_recovery, load_subset, write_json
from .run import microbatches
from .spread200 import build_spread, index_sha, paired_indices
from .training import atomic_save, sha256


def check(value, message):
    if not value:
        raise RuntimeError(message)


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('parent', 'parent-sha', 'dataset', 'assets', 'subset', 'output',
              'warmup-predictions', 'reference-training'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--validation-samples', type=int, default=128)
    p.add_argument('--max-seconds', type=int, default=1500)
    a = p.parse_args(argv)
    started = time.monotonic()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'request.json', dict(vars(a), preflight_only=True))
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA device required')
    check('H100' in torch.cuda.get_device_name(0), 'Only the authorized H100 is accepted')
    c, assets, data = build_spread(a.parent, a.parent_sha, 'cuda:0', a.dataset, a.assets)
    subset = load_subset(a.subset, a.dataset)
    validation = KuaiRecListDataset(a.dataset, 'validation', assets=assets)
    selected = np.asarray([r['index'] for r in subset['rows'][:a.validation_samples]], dtype=np.int64)
    check(len(selected) == a.validation_samples, 'Insufficient locked diagnostic samples')
    initial = c.checkpoint()
    check(all(fingerprint(initial['inner'][k]) == v for k, v in c.parent_hashes.items()), 'R5 state changed at fork')
    # Verify original R5 item outputs, with no optimizer update and FP32 decode.
    summary, eval_seconds = timed(lambda: evaluate_recovery(c.trainer, assets, validation,
        out / 'initial_validation', indices=selected, subset={'preflight_only': True}))
    wanted = {r['sample_id'] for r in subset['rows'][:a.validation_samples]}
    reference = {}
    with gzip.open(a.warmup_predictions, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['sample_id'] in wanted:
                reference[row['sample_id']] = row
    check(set(reference) == wanted, 'Original R5 predictions lack locked samples')
    agreement = 0
    with gzip.open(out / 'initial_validation/per_sample.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            old = reference[row['sample_id']]
            check(row['targets'] == old['targets'] and row['user_id'] == old['user_id'], 'R5 sample identity changed')
            agreement += row['predictions'][0] == old['predictions'][0]
    check(agreement == len(selected), 'Original R5 Top5 output mismatch')
    check(fingerprint(c.checkpoint()) == fingerprint(initial), 'Startup evaluation changed state')
    reference_updates = {}
    with Path(a.reference_training).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('phase') == 'T' and row.get('phase_index') == 5:
                reference_updates.setdefault(row['t_updates'], row)
    r_seconds, t_seconds, replay = [], [], []
    cfg = c.trainer.config
    indices = paired_indices(cfg, len(data), 5, c.row)
    for update in range(20):
        check(time.monotonic() - started < a.max_seconds - 60, 'Short-acceptance time reserve reached')
        begin = update * cfg['training']['effective_batch_size']
        batches = list(microbatches(data, indices[begin:begin + cfg['training']['effective_batch_size']],
                                   cfg['training']['microbatch_size']))
        snapshot = c.checkpoint() if update == 3 else None
        before = fingerprint(c.trainer.model.state_dict()) if update == 0 else None
        result, seconds = timed(lambda: c.step(batches, 'T'))
        t_seconds.append(seconds)
        if before is not None:
            check(fingerprint(c.trainer.model.state_dict()) == before, 'T changed recommender parameters')
        old = reference_updates.get(update + 1)
        check(old is not None, 'Original first-20T reference missing')
        deltas = {}
        for key in ('loss', 'recommendation', 'reconstruction', 'balance', 'anchor', 'gradient_norm'):
            check(key in old and key in result, 'Missing replay scalar ' + key)
            delta = abs(result[key] - old[key])
            tolerance = (1e-6 + (1e-3 if key == 'gradient_norm' else 1e-4) * abs(old[key]))
            check(delta <= tolerance, f'First-20T numerical drift: update={update+1} {key} delta={delta} tol={tolerance}')
            deltas[key] = dict(delta=delta, tolerance=tolerance)
        replay.append(dict(update=update+1, deltas=deltas))
        if snapshot is not None:
            expected = fingerprint(c.checkpoint())
            c.restore(snapshot)
            c.step(batches, 'T')
            check(fingerprint(c.checkpoint()) == expected, 'CUDA next-T restore is not exact')
        print(json.dumps(dict(event='spread_preflight_T', update=update+1, seconds=seconds)), flush=True)
    c.prepare_boundary()
    c.complete_phase()
    indices = paired_indices(cfg, len(data), c.progress['phase'], c.row)
    for update in range(3):
        begin = update * cfg['training']['effective_batch_size']
        batches = list(microbatches(data, indices[begin:begin + cfg['training']['effective_batch_size']],
                                   cfg['training']['microbatch_size']))
        frozen = fingerprint(c.trainer.alignment.state_dict())
        snapshot = c.checkpoint()
        result, seconds = timed(lambda: c.step(batches, 'R'))
        r_seconds.append(seconds)
        check(fingerprint(c.trainer.alignment.state_dict()) == frozen, 'R changed Tokenizer state')
        if update == 1:
            expected = fingerprint(c.checkpoint())
            c.restore(snapshot)
            c.step(batches, 'R')
            check(fingerprint(c.checkpoint()) == expected, 'CUDA next-R restore is not exact')
    tagged = c.checkpoint()
    tagged['preflight_only'] = True
    atomic_save(out / 'preflight_only.pt', tagged)
    try:
        c.restore(tagged)
    except ValueError:
        pass
    else:
        raise RuntimeError('Short-test checkpoint accepted as formal')
    # Conservative forecast, explicit count of all requested decoding work.
    eval_examples = 10*71657 + 20*8192 + 2*53194
    estimate = 20790 * max(r_seconds[1:]) + 200 * max(t_seconds[1:]) + eval_examples * eval_seconds / len(selected)
    report = dict(status='passed', preflight_only=True, seconds=time.monotonic()-started,
        device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        parent_sha256=a.parent_sha, original_top5_agreement=agreement/len(selected),
        paired_first_T_indices_sha256=index_sha(paired_indices(cfg,len(data),5,c.trainer.plan[5])),
        replay=replay, next_T_exact_restore=True, next_R_exact_restore=True,
        r_update_seconds=r_seconds, t_update_seconds=t_seconds, evaluation_seconds=eval_seconds,
        evaluation_examples=len(selected), forecast_unbuffered_hours=estimate/3600,
        forecast_with_25pct_margin_hours=estimate*1.25/3600,
        checkpoint_parent_reuse_required=True, gpu_budget_hours=24)
    write_once(out / 'report.json', report)
    print(json.dumps(dict(event='spread_preflight_finished', status='passed',
                         forecast_hours=report['forecast_with_25pct_margin_hours'])), flush=True)


if __name__ == '__main__':
    main()
