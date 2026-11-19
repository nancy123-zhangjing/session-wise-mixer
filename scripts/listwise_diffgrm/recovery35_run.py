"""Three independent continuations with immutable snapshots and early milestones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .recovery import fingerprint, RecoveryController as LegacyController, PARENT_SHA256
from .recovery35 import build_recovery35, verify_binding
from .recovery35_artifacts import (BEST_EPOCHS, REPLAY_RULE, VALIDATION_EPOCHS,
    audit_checkpoints, compare_replay_predictions, reference_record, save_boundary,
    select_best, write_once)
from .recovery35_trajectory import audit_phase, record_phase_order, RULE as TRAJECTORY_RULE
from .recovery_diagnostics import diagnostic_context
from .recovery_evaluation import write_json
from .run import build_context, evaluate, microbatches, phase_indices, run_lock
from .training import atomic_save, seed_all, sha256


def save_last(controller, output):
    atomic_save(Path(output) / 'last.pt', controller.checkpoint())


def milestone(output, name, value):
    value = dict(value, event=name)
    write_once(Path(output) / 'milestones' / (name + '.json'), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def evaluation(controller, assets, output, split, epoch, *, checkpoint=None):
    """Exactly one successful artifact for a fixed state; retries never overwrite."""
    root = Path(output) / 'evaluations' / f'{split}_R{epoch:02d}'
    root.mkdir(parents=True, exist_ok=True)
    before = fingerprint(controller.checkpoint())
    binding = dict(branch=controller.branch, split=split, epoch=epoch,
                   model=fingerprint(dict(model=controller.trainer.model.state_dict(),
                                          alignment=controller.trainer.alignment.state_dict())),
                   execution=fingerprint(controller.execution_binding),
                   r_updates=controller.trainer.r_updates, t_updates=controller.trainer.t_updates)
    done = root / 'complete.json'
    expected_count = 71657 if split == 'validation' else 53194
    if done.exists():
        receipt = json.loads(done.read_text(encoding='utf-8'))
        if receipt['binding'] != binding:
            raise ValueError('Completed evaluation belongs to a different state')
        destination = root / receipt['attempt']
        for name, key in (('summary.json', 'summary_sha256'), ('per_sample.jsonl.gz', 'predictions_sha256')):
            if sha256(destination / name) != receipt[key]:
                raise ValueError('Completed evaluation artifact changed')
    else:
        attempt = 1
        while (root / f'attempt{attempt}').exists():
            attempt += 1
        destination = root / f'attempt{attempt}'
        with diagnostic_context(controller.trainer), torch.autocast(controller.trainer.device.type, enabled=False):
            evaluate(controller.trainer, assets, split=split, destination=destination, beam=32)
        if fingerprint(controller.checkpoint()) != before:
            raise RuntimeError('Evaluation changed training state')
        receipt = dict(binding=binding, attempt=destination.name,
                       summary_sha256=sha256(destination / 'summary.json'),
                       predictions_sha256=sha256(destination / 'per_sample.jsonl.gz'))
    summary = json.loads((destination / 'summary.json').read_text(encoding='utf-8'))
    if (summary['examples'] != expected_count or summary['split'] != split
            or not summary['full_split'] or summary['beam'] != 32
            or summary['inference_precision'] != 'fp32'
            or summary['seed'] != 2024 or summary['architecture'] != 'H'
            or summary['r_updates'] != binding['r_updates'] or summary['t_updates'] != binding['t_updates']):
        raise ValueError('Evaluation count, clock or protocol mismatch')
    if fingerprint(controller.checkpoint()) != before:
        raise RuntimeError('Evaluation/reuse changed training state')
    write_once(done, receipt)
    return dict(kind='new', epoch=epoch, summary=str(destination / 'summary.json'),
                predictions=str(destination / 'per_sample.jsonl.gz'),
                summary_sha256=receipt['summary_sha256'], predictions_sha256=receipt['predictions_sha256'],
                checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                ndcg5=summary['metrics']['list/ndcg@5'])


def replay_check(controller, output, epoch, candidate, references):
    reference, _ = reference_record(references, 'C', epoch)
    result = compare_replay_predictions(reference['predictions'], candidate['predictions'])
    result.update(branch='C', epoch=epoch,
                  validation_sha256=sha256(Path(controller.trainer.config['dataset']) / 'validation.parquet'),
                  candidate_summary_sha256=candidate['summary_sha256'])
    write_once(Path(output) / 'replay' / f'R{epoch:02d}.json', result)
    if result['status'] != 'passed':
        milestone(output, f'replay_review_R{epoch:02d}', result)
        raise RuntimeError('H0 replay exceeded the pre-registered engineering gate; retain state, no automatic retrain')
    return result


def authenticate_references(controller, manifest):
    branch = controller.branch
    epochs = {'A': (20, 23, 26), 'B': (32,), 'C': (5, 8, 11, 14, 17, 20)}[branch]
    for epoch in epochs:
        record, summary = reference_record(manifest, branch, epoch)
        # Parent-stage counts are explicitly checked; C references are the old
        # unfrozen trajectory, not new best-selection candidates.
        if summary['r_updates'] != epoch * 693:
            raise ValueError('Reference R epoch/clock mismatch')
        expected_data = controller.execution_binding['data']['files']['validation.parquet']['sha256']
        if summary['dataset_binding']['files']['validation.parquet']['sha256'] != expected_data:
            raise ValueError('Reference validation release differs')


def train(args):
    if not args.device.startswith('cuda'):
        raise ValueError('Formal training requires the reviewed H100 runtime')
    controller, assets, data = build_recovery35(args.parent, args.branch, args.device,
        args.dataset, args.assets, ancestor_path=args.ancestor)
    output = Path(args.output)
    refs = json.loads(Path(args.references).read_text(encoding='utf-8'))
    authenticate_references(controller, refs)
    trainer = controller.trainer
    request = dict(branch=args.branch, parent=args.parent, parent_sha256=sha256(args.parent),
                   ancestor=args.ancestor, reference_manifest_sha256=sha256(args.references),
                   protocol=controller.protocol, replay_rule=REPLAY_RULE,
                   trajectory_rule=TRAJECTORY_RULE,
                   validation_epochs=list(VALIDATION_EPOCHS[args.branch]))
    with run_lock(output, resume=bool(args.resume)):
        if args.resume:
            if json.loads((output / 'request.json').read_text(encoding='utf-8')) != request:
                raise ValueError('Resume request/source/reference manifest changed')
            controller.restore(torch.load(args.resume, map_location='cpu', weights_only=False))
        else:
            save_boundary(controller, output, 'initial.pt')
            write_once(output / 'request.json', request)
            save_last(controller, output)
        started = time.monotonic()
        while not controller.done:
            if controller.needs_replay_gate:
                save_boundary(controller, output, 'R20_pre_freeze.pt')
                checks = [json.loads((output / 'replay' / f'R{e:02d}.json').read_text(encoding='utf-8'))
                          for e in (8, 11, 14, 17, 20)]
                if any(c['status'] != 'passed' or c['rule'] != REPLAY_RULE for c in checks):
                    raise ValueError('All pre-freeze replay checks must pass')
                candidate = json.loads((output / 'validation_records' / 'R20.json').read_text(encoding='utf-8'))
                # Recheck actual files, rather than trusting a previous pass flag.
                latest = replay_check(controller, output, 20, candidate, refs)
                receipt = dict(latest, checkpoint_state_sha256=fingerprint(controller.checkpoint()['inner']),
                               protocol_sha256=controller.protocol_sha256,
                               checks={str(c['epoch']): c for c in checks})
                write_once(output / 'replay_gate.json', receipt)
                controller.accept_replay_gate(receipt)
                save_last(controller, output)
                milestone(output, 'R20_freeze_accepted', dict(branch='C', r_updates=13860, t_updates=200,
                          gate_sha256=sha256(output / 'replay_gate.json')))
            phase = controller.progress['phase']
            row = controller.row
            skipped = row['kind'] == 'T' and controller.frozen
            if not controller.progress['boundary_prepared']:
                indices = phase_indices(trainer.config, len(data), phase, row) if not skipped else None
                if args.branch == 'C' and phase < 25:
                    record_phase_order(controller, output, phase, indices)
                if not skipped and controller.progress['update'] == 0:
                    seed_all(trainer.config['seed'] + 100003 * (phase + 1))
                while not skipped and controller.progress['update'] < row['updates']:
                    begin = controller.progress['update'] * trainer.config['training']['effective_batch_size']
                    selected = indices[begin:begin + trainer.config['training']['effective_batch_size']]
                    clock = time.monotonic()
                    values = controller.step(microbatches(data, selected, trainer.config['training']['microbatch_size']), row['kind'])
                    values.update(epoch=row['epoch'], branch=args.branch, phase_index=phase,
                                  update=controller.progress['update'], seconds=time.monotonic() - clock)
                    with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(values, allow_nan=False) + '\n')
                    if controller.progress['update'] % trainer.config['training']['checkpoint_every_updates'] == 0:
                        save_last(controller, output)
                    if controller.progress['update'] % 20 == 0:
                        print(json.dumps(dict(event='training_progress', **values)), flush=True)
                save_last(controller, output)
                controller.prepare_boundary()
                save_last(controller, output)
            if args.branch == 'C' and phase < 25:
                audit_phase(controller, output, phase)
            if row['kind'] == 'R' and row['epoch'] == 35:
                save_boundary(controller, output, 'R35_training_boundary.pt')
                milestone(output, 'training_finished', dict(branch=args.branch, epoch=35,
                          r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                          checkpoint='R35_training_boundary.pt', validation_pending=True))
            candidate = None
            if row['kind'] == 'R' and row['epoch'] in VALIDATION_EPOCHS[args.branch]:
                candidate = evaluation(controller, assets, output, 'validation', row['epoch'])
                if args.branch == 'C' and row['epoch'] <= 20:
                    replay_check(controller, output, row['epoch'], candidate, refs)
            controller.complete_phase()
            name = f'phase{phase:02d}_{row["kind"]}_R{row["epoch"]:02d}_T{trainer.t_updates:03d}.pt'
            snapshot = save_boundary(controller, output, name)
            if candidate is not None:
                candidate.update(checkpoint=str(output / name), checkpoint_sha256=snapshot['sha256'])
                write_once(output / 'validation_records' / f'R{row["epoch"]:02d}.json', candidate)
            save_last(controller, output)
        coverage = audit_checkpoints(controller, output)
        write_once(output / 'CHECKPOINT_COVERAGE.json', coverage)
        records = {}
        if args.branch in ('A', 'C'):
            for epoch in BEST_EPOCHS:
                record_path = output / 'validation_records' / f'R{epoch:02d}.json'
                if record_path.is_file():
                    records[str(epoch)] = json.loads(record_path.read_text(encoding='utf-8'))
                else:
                    record, summary = reference_record(refs, args.branch, epoch)
                    records[str(epoch)] = dict(record, kind='reference', epoch=epoch,
                                               ndcg5=summary['metrics']['list/ndcg@5'])
            selected = select_best(records)
        else:
            selected = json.loads((output / 'validation_records/R35.json').read_text(encoding='utf-8'))
        selection = dict(branch=args.branch, criterion='validation/list/ndcg@5; tie earlier',
                         candidates=records, selected=selected, fixed_endpoint=35)
        write_once(output / 'selection.json', selection)
        milestone(output, 'validation_finished', dict(branch=args.branch, selected_epoch=selected['epoch'],
                   selection_sha256=sha256(output / 'selection.json')))
        write_once(output / 'complete.json', dict(status='training_and_validation_complete', branch=args.branch,
            r_updates=trainer.r_updates, t_updates=trainer.t_updates, phase=controller.progress['phase'],
            last_sha256=sha256(output / 'last.pt'), checkpoint_coverage_sha256=sha256(output / 'CHECKPOINT_COVERAGE.json'),
            selection_sha256=sha256(output / 'selection.json')))
        print(json.dumps(dict(event='arm_train_complete', branch=args.branch, seconds=time.monotonic() - started)), flush=True)


def historical_h1_controller(args, target):
    """Only evaluate an authenticated historical candidate; never continue it."""
    import copy
    checkpoint = Path(target['checkpoint'])
    if sha256(checkpoint) != target['checkpoint_sha256'] or sha256(args.ancestor) != PARENT_SHA256:
        raise ValueError('Historical H1 checkpoint or ancestor changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    ancestor = torch.load(args.ancestor, map_location='cpu', weights_only=False)
    inner = saved.get('inner', saved)
    if inner['r_updates'] != target['epoch'] * 693 or inner['t_updates'] != 200:
        raise ValueError('Historical H1 candidate clock mismatch')
    config = copy.deepcopy(inner['binding']['config'])
    if args.dataset:
        config['dataset'] = args.dataset
    if args.assets:
        config['assets'] = args.assets
    trainer, assets, _, current = build_context(config, args.device, inner['precision'], True)
    sealed = saved['execution_binding'] if 'inner' in saved else current
    verify_binding(inner['binding'], current)
    controller = LegacyController(trainer, 'freeze', ancestor['binding'], sealed, PARENT_SHA256)
    controller.fork(ancestor)
    if 'inner' in saved:
        controller.restore(saved)
    elif target['epoch'] != 20 or sha256(checkpoint) != PARENT_SHA256:
        raise ValueError('Unsupported historical raw H1 candidate')
    controller.branch = 'A'
    return controller, assets


def test(args):
    """Only fixed R35 plus a prelocked A/C best; never enumerate epochs on test."""
    output = Path(args.output)
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    complete = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
    if (complete['status'] != 'training_and_validation_complete' or complete['branch'] != args.branch
            or sha256(output / 'selection.json') != complete['selection_sha256']):
        raise ValueError('Training/selection is not locked')
    refs = json.loads(Path(args.references).read_text(encoding='utf-8'))
    request = json.loads((output / 'request.json').read_text(encoding='utf-8'))
    if (request['reference_manifest_sha256'] != sha256(args.references)
            or request['parent_sha256'] != sha256(args.parent)
            or request['branch'] != args.branch or request['ancestor'] != args.ancestor):
        raise ValueError('Test inputs differ from the locked training request')
    fixed = json.loads((output / 'validation_records/R35.json').read_text(encoding='utf-8'))
    targets = [fixed]
    if args.branch in ('A', 'C') and selection['selected']['epoch'] != 35:
        targets.append(selection['selected'])
    results = []
    for target in targets:
        epoch = target['epoch']
        if target['kind'] == 'reference':
            if str(epoch) in refs.get('test', {}).get(args.branch, {}):
                reused, _ = reference_record(refs, args.branch, epoch, split='test')
                if reused['checkpoint_sha256'] != target['checkpoint_sha256']:
                    raise ValueError('Cached test belongs to a different selected model')
                results.append(dict(reused, epoch=epoch, reused=True))
            else:
                if args.branch != 'A':
                    raise ValueError('Unexpected uncached historical test candidate')
                controller, assets = historical_h1_controller(args, target)
                results.append(evaluation(controller, assets, output, 'test', epoch,
                                          checkpoint=target['checkpoint']))
            continue
        checkpoint = Path(target['checkpoint'])
        if sha256(checkpoint) != target['checkpoint_sha256']:
            raise ValueError('Selected model bytes changed')
        controller, assets, _ = build_recovery35(args.parent, args.branch, args.device,
            args.dataset, args.assets, ancestor_path=args.ancestor)
        controller.restore(torch.load(checkpoint, map_location='cpu', weights_only=False))
        if controller.trainer.r_updates != epoch * 693:
            raise ValueError('Selected checkpoint epoch mismatch')
        result = evaluation(controller, assets, output, 'test', epoch, checkpoint=checkpoint)
        results.append(result)
    # Metrics are audited independently before the user-facing test report.
    write_once(output / 'test_predictions_complete.json', dict(branch=args.branch, results=results,
               selection_sha256=complete['selection_sha256'], independently_audited=False))
    milestone(output, 'test_predictions_finished', dict(branch=args.branch,
               epochs=[r['epoch'] for r in results], independent_audit_pending=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'test'))
    parser.add_argument('--branch', required=True, choices=('A', 'B', 'C'))
    parser.add_argument('--parent', required=True)
    parser.add_argument('--ancestor')
    parser.add_argument('--dataset')
    parser.add_argument('--assets')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args(argv)
    if args.command == 'test' and args.resume:
        parser.error('Test does not accept a training resume override')
    return train(args) if args.command == 'train' else test(args)


if __name__ == '__main__':
    main()
