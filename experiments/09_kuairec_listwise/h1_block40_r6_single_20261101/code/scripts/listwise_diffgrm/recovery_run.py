"""Two H1 R20 forks; no platform submission or resource allocation on import."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from .data import KuaiRecListDataset
from .recovery import PARENT_SHA256, RecoveryController, fingerprint
from .recovery_diagnostics import diagnose_catalog
from .recovery_evaluation import (evaluate_recovery, extract_subset, load_subset,
                                  subset_manifest, write_json)
from .run import build_context, microbatches, phase_indices, run_lock
from .training import atomic_save, seed_all, sha256


def build_recovery(parent_path, policy, device, dataset=None, assets=None, inspection=False):
    parent_path = Path(parent_path)
    digest = sha256(parent_path)
    if digest != PARENT_SHA256:
        raise ValueError('H1 R20 parent SHA does not match authorized checkpoint')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    if (parent['r_updates'], parent['t_updates'], parent['progress']['phase']) != (13860, 200, 25):
        raise ValueError('Parent counters are not R20/T200')
    config = copy.deepcopy(parent['binding']['config'])
    if dataset:
        config['dataset'] = str(dataset)
    if assets:
        config['assets'] = str(assets)
    trainer, loaded_assets, data, current = build_context(config, device, parent['precision'], True)
    controller = RecoveryController(trainer, policy, parent['binding'], current, digest)
    controller.fork(parent, exact_rng=not inspection)
    return controller, loaded_assets, data


def save(controller, output, name='last.pt'):
    atomic_save(Path(output) / name, controller.checkpoint())


def catalog_event(controller, output, label):
    event = 'catalog/' + label
    if event in controller.diagnostic_events:
        return
    metrics, snapshot = diagnose_catalog(controller.trainer, controller.diagnostic_snapshot)
    metrics.update(event=event, phase_index=controller.progress['phase'], update=controller.progress['update'])
    directory = Path(output) / 'catalog'
    directory.mkdir(exist_ok=True)
    write_json(directory / f'{label}.json', metrics)
    controller.diagnostic_snapshot = snapshot
    controller.diagnostic_events.append(event)
    save(controller, output)


def evaluation_event(controller, assets, data, manifest, output, label, *, full=False):
    event = 'validation/' + label
    trainer = controller.trainer
    binding = dict(state=fingerprint(dict(model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict())),
                   source=fingerprint(controller.execution_binding), subset=fingerprint(manifest), full=full,
                   r_updates=trainer.r_updates, t_updates=trainer.t_updates)
    root = Path(output) / 'evaluations' / label
    root.mkdir(parents=True, exist_ok=True)
    completed = root / 'complete.json'
    if completed.is_file():
        receipt = json.loads(completed.read_text(encoding='utf-8'))
        artifact = root / receipt['attempt']
        if (receipt['binding'] != binding or sha256(artifact / 'per_sample.jsonl.gz') != receipt['predictions_sha256']
                or sha256(artifact / 'summary.json') != receipt['summary_sha256']):
            raise ValueError('Existing evaluation receipt does not match this state')
        summary = json.loads((artifact / 'summary.json').read_text(encoding='utf-8'))
        if summary['examples'] != (len(data) if full else manifest['count']):
            raise ValueError('Existing evaluation has wrong sample count')
    else:
        attempt = 1
        while (root / f'attempt{attempt}').exists():
            attempt += 1
        artifact = root / f'attempt{attempt}'
        indices = None if full else [row['index'] for row in manifest['rows']]
        evaluate_recovery(trainer, assets, data, artifact, indices=indices,
                          subset=None if full else {k: v for k, v in manifest.items() if k != 'rows'})
        if full:
            write_json(artifact / 'diagnostic_subset.json', extract_subset(artifact / 'per_sample.jsonl.gz', manifest))
        write_json(completed, dict(binding=binding, attempt=artifact.name,
                                  predictions_sha256=sha256(artifact / 'per_sample.jsonl.gz'),
                                  summary_sha256=sha256(artifact / 'summary.json')))
    if event not in controller.diagnostic_events:
        controller.diagnostic_events.append(event)
    save(controller, output)


def audit_evaluation_files(controller, output, manifest):
    expected = {'initial_R20', *(f'R{epoch:02d}' for epoch in range(21, 27))}
    if controller.policy == 'decay':
        expected |= {'after_T240', 'after_T280'}
    if {event.split('/', 1)[1] for event in controller.diagnostic_events if event.startswith('validation/')} != expected:
        raise ValueError('Required validation events are incomplete')
    receipts = {}
    for label in sorted(expected):
        root = Path(output) / 'evaluations' / label
        receipt = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
        artifact = root / receipt['attempt']
        summary = json.loads((artifact / 'summary.json').read_text(encoding='utf-8'))
        count = manifest['population'] if label in ('R23', 'R26') else manifest['count']
        if label == 'initial_R20':
            expected_clocks = controller.start_counts
        elif label.startswith('after_T'):
            target_t, total_t, boundary = int(label[7:]), 0, None
            for index, row in enumerate(controller.trainer.plan):
                if row['kind'] == 'T':
                    total_t += row['updates']
                    if total_t == target_t:
                        boundary = index + 1
                        break
            expected_clocks = controller.counts_at(dict(phase=boundary, update=0, boundary_prepared=False))
        else:
            epoch = int(label[1:])
            boundary = next(index + 1 for index, row in enumerate(controller.trainer.plan)
                            if row['kind'] == 'R' and row['epoch'] == epoch)
            expected_clocks = controller.counts_at(dict(phase=boundary, update=0, boundary_prepared=False))
        if (summary['examples'] != count or summary['split'] != 'validation' or summary['beam'] != 32
                or summary['full_split'] != (label in ('R23', 'R26'))
                or summary['r_updates'] != expected_clocks['R'] or summary['t_updates'] != expected_clocks['T']
                or receipt['binding']['r_updates'] != expected_clocks['R']
                or receipt['binding']['t_updates'] != expected_clocks['T']
                or receipt['binding']['full'] != (label in ('R23', 'R26'))
                or receipt['binding']['source'] != fingerprint(controller.execution_binding)
                or receipt['binding']['subset'] != fingerprint(manifest)
                or sha256(artifact / 'per_sample.jsonl.gz') != receipt['predictions_sha256']
                or sha256(artifact / 'summary.json') != receipt['summary_sha256']):
            raise ValueError(f'Final evaluation receipt verification failed: {label}')
        receipts[label] = dict(examples=count, predictions_sha256=receipt['predictions_sha256'])
    return receipts


def train(args):
    if not args.device.startswith('cuda'):
        raise ValueError('Formal continuation requires the authorized H100 runtime')
    controller, assets, data = build_recovery(args.parent, args.policy, args.device, args.dataset, args.assets)
    trainer = controller.trainer
    output = Path(args.output)
    manifest = load_subset(args.subset, trainer.config['dataset'])
    if manifest['count'] != 8192 or manifest['population'] != 71657:
        raise ValueError('Formal diagnostic/validation size changed')
    with run_lock(output, resume=bool(args.resume)):
        if args.resume:
            controller.restore(torch.load(args.resume, map_location='cpu', weights_only=False))
        else:
            write_json(output / 'fork.json', dict(protocol=controller.protocol, parent_hashes=controller.parent_hashes,
                                                execution_binding=controller.execution_binding))
            save(controller, output, 'initial.pt')
            save(controller, output)
        validation = KuaiRecListDataset(trainer.config['dataset'], 'validation', assets=assets)
        if not args.resume:
            catalog_event(controller, output, 'initial_R20')
            evaluation_event(controller, assets, validation, manifest, output, 'initial_R20')
        elif controller.progress['phase'] == controller.start_phase and controller.progress['update'] == 0:
            catalog_event(controller, output, 'initial_R20')
            evaluation_event(controller, assets, validation, manifest, output, 'initial_R20')
        started = time.monotonic()
        while not controller.done:
            phase_id = controller.progress['phase']
            row = controller.row
            skip = row['kind'] == 'T' and controller.policy == 'freeze'
            indices = phase_indices(trainer.config, len(data), phase_id, row)
            if controller.progress['update'] == 0 and not controller.progress['boundary_prepared']:
                seed_all(trainer.config['seed'] + 100003 * (phase_id + 1))
            while not skip and controller.progress['update'] < row['updates']:
                start = controller.progress['update'] * trainer.config['training']['effective_batch_size']
                selected = indices[start:start + trainer.config['training']['effective_batch_size']]
                before = time.perf_counter()
                values = controller.step(microbatches(data, selected, trainer.config['training']['microbatch_size']), row['kind'])
                values.update(epoch=row['epoch'], update=controller.progress['update'],
                              seconds=time.perf_counter() - before, cumulative_seconds=time.monotonic() - started)
                with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(values, allow_nan=False) + '\n')
                if row['kind'] == 'T' and trainer.t_updates % 10 == 0:
                    catalog_event(controller, output, f'T{trainer.t_updates:03d}')
                elif controller.progress['update'] % trainer.config['training']['checkpoint_every_updates'] == 0:
                    save(controller, output)
                if controller.progress['update'] % 20 == 0:
                    print(json.dumps(dict(event='recovery_training', policy=controller.policy, **values)), flush=True)
            # Persist an update-complete cursor before any boundary work.
            save(controller, output)
            controller.prepare_boundary()
            save(controller, output)
            boundary = f'phase{phase_id:02d}_{row["kind"]}_R{row["epoch"]:02d}_T{trainer.t_updates:03d}'
            if not skip:
                if row['kind'] == 'R':
                    label = f'R{row["epoch"]:02d}'
                    evaluation_event(controller, assets, validation, manifest, output, label, full=row['epoch'] in (23, 26))
                else:
                    evaluation_event(controller, assets, validation, manifest, output, f'after_T{trainer.t_updates:03d}')
            save(controller, output, boundary + '.pt')
            controller.complete_phase()
            save(controller, output)
        receipts = audit_evaluation_files(controller, output, manifest)
        write_json(output / 'complete.json', dict(policy=controller.policy, r_updates=trainer.r_updates,
            t_updates=trainer.t_updates, phase=controller.progress['phase'], endpoint_epoch=26,
            endpoint_checkpoint_sha256=sha256(output / 'last.pt'), events=controller.diagnostic_events,
            evaluation_audit=receipts, attempt_seconds=time.monotonic() - started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'inspect', 'subset'])
    parser.add_argument('--parent')
    parser.add_argument('--policy', choices=['decay', 'freeze'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dataset')
    parser.add_argument('--assets')
    parser.add_argument('--output', required=True)
    parser.add_argument('--subset')
    parser.add_argument('--resume')
    args = parser.parse_args(argv)
    if args.command == 'subset':
        destination = Path(args.output)
        result = subset_manifest(args.dataset)
        if destination.exists():
            if json.loads(destination.read_text(encoding='utf-8')) != result:
                raise FileExistsError('Refusing to replace different subset')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_json(destination, result)
        print(json.dumps(dict(count=result['count'], manifest_sha256=sha256(destination))))
    elif args.command == 'inspect':
        controller, _, data = build_recovery(args.parent, args.policy, args.device, args.dataset, args.assets, inspection=True)
        result = dict(protocol=controller.protocol, parent_hashes=controller.parent_hashes,
                      examples=len(data), r_updates=controller.trainer.r_updates, t_updates=controller.trainer.t_updates,
                      cuda_exact_resume_verified=False)
        write_json(args.output, result)
        print(json.dumps(result))
    else:
        train(args)


if __name__ == '__main__':
    main()
