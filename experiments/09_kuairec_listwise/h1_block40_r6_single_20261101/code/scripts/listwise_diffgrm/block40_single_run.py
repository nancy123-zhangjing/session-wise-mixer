"""Train/evaluate the single pre-registered spread200 branch; no submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .data import KuaiRecListDataset
from .recovery import fingerprint
from .recovery35_artifacts import save_boundary, write_once
from .recovery35_run import evaluation
from .recovery_diagnostics import diagnose_catalog
from .recovery_evaluation import evaluate_recovery, extract_subset, load_subset, write_json
from .run import microbatches, run_lock
from .block40_single import BEST_EPOCHS, NAME, VALIDATION_EPOCHS, build_spread, index_sha, paired_indices, original_phase
from .training import atomic_save, sha256


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def last(c, output):
    atomic_save(Path(output) / 'last.pt', c.checkpoint())


def catalog(c, output, label):
    """Idempotent per-boundary diagnostic, preserving all training state."""
    path = Path(output) / 'catalog' / (label + '.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        snapshot_path = path.with_suffix('.pt')
        if (saved['state'] != fingerprint(c.checkpoint()) or
                sha256(snapshot_path) != saved['snapshot_sha256']):
            raise ValueError('Cached catalog diagnostic changed or belongs to another state')
        atomic_save(Path(output) / 'catalog_previous.pt',
                    torch.load(snapshot_path, map_location='cpu', weights_only=False))
        return
    before = fingerprint(c.checkpoint())
    previous_path = Path(output) / 'catalog_previous.pt'
    previous = torch.load(previous_path, map_location='cpu', weights_only=False) if previous_path.exists() else None
    metrics, snapshot = diagnose_catalog(c.trainer, previous)
    if fingerprint(c.checkpoint()) != before:
        raise RuntimeError('Catalog diagnostic changed training state')
    snapshot_path = Path(output) / 'catalog' / (label + '.pt')
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(snapshot_path, snapshot)
    write_once(path, dict(metrics, state=before, r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates,
                         snapshot_sha256=sha256(snapshot_path)))
    atomic_save(previous_path, snapshot)


def diagnostic(c, assets, validation, subset, output, label, source=None):
    root = Path(output) / 'diagnostics' / label
    root.mkdir(parents=True, exist_ok=True)
    done = root / 'complete.json'
    before = fingerprint(c.checkpoint())
    state = fingerprint(dict(model=c.trainer.model.state_dict(), alignment=c.trainer.alignment.state_dict()))
    if done.exists():
        saved = json.loads(done.read_text(encoding='utf-8'))
        if saved['state'] != state:
            raise ValueError('Diagnostic belongs to a different checkpoint')
        file_key, hash_key = ('reused', 'source_sha256') if 'reused' in saved else ('predictions', 'predictions_sha256')
        if sha256(saved[file_key]) != saved[hash_key]:
            raise ValueError('Cached process predictions changed')
        return saved
    if source is not None:
        result = extract_subset(source, subset)
        receipt = dict(state=state, result=result, reused=str(source), source_sha256=sha256(source))
    else:
        attempt = 1
        while (root / f'attempt{attempt}').exists():
            attempt += 1
        destination = root / f'attempt{attempt}'
        indices = np.asarray([r['index'] for r in subset['rows']], dtype=np.int64)
        result = evaluate_recovery(c.trainer, assets, validation, destination, indices=indices, subset=subset)
        receipt = dict(state=state, result=result, predictions=str(destination / 'per_sample.jsonl.gz'),
                       predictions_sha256=sha256(destination / 'per_sample.jsonl.gz'))
    if fingerprint(c.checkpoint()) != before:
        raise RuntimeError('Process evaluation changed training state')
    write_once(done, receipt)
    return receipt


def context(args):
    return build_spread(args.parent, args.parent_sha, args.device, args.dataset, args.assets)


def train(args):
    output = Path(args.output)
    c, assets, data = context(args)
    subset = load_subset(args.subset, c.trainer.config['dataset'])
    validation = KuaiRecListDataset(c.trainer.config['dataset'], 'validation', assets=assets)
    request = dict(parent_sha256=args.parent_sha, subset_sha256=sha256(args.subset),
                   protocol=c.protocol, execution_binding=c.execution_binding,
                   warmup_predictions_sha256=sha256(args.warmup_predictions))
    with run_lock(output, resume=bool(args.resume)):
        stop = {'requested': False}
        def request_stop(_sig, _frame):
            stop['requested'] = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        write_once(output / 'request.json', request)
        if args.resume:
            c.restore(torch.load(args.resume, map_location='cpu', weights_only=False))
        else:
            save_boundary(c, output, 'initial_R05_T000.pt')
            last(c, output)
        if c.progress['phase'] == 5 and c.progress['update'] == 0:
            catalog(c, output, 'initial_R05_T000')
            diagnostic(c, assets, validation, subset, output, 'initial_R05_T000', source=args.warmup_predictions)
        while not c.done:
            phase, row = c.progress['phase'], dict(c.row)
            indices = paired_indices(c.trainer.config, len(data), phase, row)
            write_once(output / 'sample_orders' / f'phase{phase:02d}.json', dict(
                phase=phase, kind=row['kind'], epoch=row['epoch'], lists=len(indices),
                indices_sha256_int64_le=index_sha(indices),
                original_t_phase=original_phase(row),
                original_t_half=None))
            while c.progress['update'] < row['updates']:
                if stop['requested']:
                    last(c, output)
                    raise SystemExit(124)
                begin = c.progress['update'] * c.trainer.config['training']['effective_batch_size']
                selected = indices[begin:begin + c.trainer.config['training']['effective_batch_size']]
                started = time.perf_counter()
                values = c.step(microbatches(data, selected, c.trainer.config['training']['microbatch_size']), row['kind'])
                values.update(epoch=row['epoch'], seconds=time.perf_counter() - started,
                              sample_indices_sha256=index_sha(selected))
                append(output / 'training.jsonl', values)
                if c.progress['update'] % (10 if row['kind'] == 'T' else 100) == 0:
                    last(c, output)
                    print(json.dumps(dict(event='spread_progress', **values)), flush=True)
            if not c.progress['boundary_prepared']:
                c.prepare_boundary()
                last(c, output)
            label = f'phase{phase:02d}_{row["kind"]}_R{row["epoch"]:02d}_T{c.trainer.t_updates:03d}'
            record = None
            if row['kind'] == 'T':
                catalog(c, output, label)
                diagnostic(c, assets, validation, subset, output, label)
            elif row['epoch'] in VALIDATION_EPOCHS:
                record = evaluation(c, assets, output, 'validation', row['epoch'])
                diagnostic(c, assets, validation, subset, output, label, source=record['predictions'])
            elif (row['epoch'] - 6) % 6 == 0:
                diagnostic(c, assets, validation, subset, output, label)
            c.complete_phase()
            snapshot = save_boundary(c, output, label + '.pt')
            if record is not None:
                record.update(checkpoint=str(output / (label + '.pt')), checkpoint_sha256=snapshot['sha256'])
                write_once(output / 'validation_records' / f'R{row["epoch"]:02d}.json', record)
            last(c, output)
            print(json.dumps(dict(event='spread_boundary', epoch=row['epoch'], kind=row['kind'],
                                 r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates)), flush=True)
        candidates = [json.loads((output / 'validation_records' / f'R{e:02d}.json').read_text(encoding='utf-8'))
                      for e in BEST_EPOCHS]
        selected = sorted(candidates, key=lambda r: (-r['ndcg5'], r['epoch']))[0]
        write_once(output / 'selection.json', dict(criterion='validation/list/ndcg@5; tie earlier',
                                                   candidates=candidates, selected=selected))
        write_once(output / 'complete.json', dict(status='training_and_validation_complete',
            r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates,
            last_sha256=sha256(output / 'last.pt'), selection_sha256=sha256(output / 'selection.json')))


def test(args):
    output = Path(args.output)
    complete = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
    if (complete['status'] != 'training_and_validation_complete' or
            complete['selection_sha256'] != sha256(output / 'selection.json')):
        raise ValueError('Training selection is not locked')
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    targets = [json.loads((output / 'validation_records/R35.json').read_text(encoding='utf-8'))]
    if selection['selected']['epoch'] != 35:
        targets.append(selection['selected'])
    c, assets, _ = context(args)
    results = []
    for target in targets:
        if sha256(target['checkpoint']) != target['checkpoint_sha256']:
            raise ValueError('Selected checkpoint bytes changed')
        c.restore(torch.load(target['checkpoint'], map_location='cpu', weights_only=False))
        results.append(evaluation(c, assets, output, 'test', target['epoch'], checkpoint=target['checkpoint']))
    write_once(output / 'test_predictions_complete.json', dict(results=results,
        selection_sha256=sha256(output / 'selection.json'), independent_audit_pending=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'test'))
    for k in ('parent', 'parent-sha', 'dataset', 'assets', 'subset', 'output', 'warmup-predictions'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume')
    a = p.parse_args(argv)
    if a.command == 'test' and a.resume:
        p.error('Test does not accept a training resume override')
    return train(a) if a.command == 'train' else test(a)


if __name__ == '__main__':
    main()
