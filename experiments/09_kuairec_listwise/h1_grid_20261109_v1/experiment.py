"""Explicit single-card B/C runner. Importing this file launches nothing."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_ORIGINAL = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def fresh_destination(base):
    dest, retry = base, 0
    while dest.exists():
        retry += 1
        dest = base.with_name(base.name + f'_retry{retry}')
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['B', 'C'], required=True)
    parser.add_argument('--original-root', type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    original = args.original_root.resolve()
    code = original / 'code'
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(code))
    import torch
    from scripts.listwise_diffgrm import run, training
    from protocol import GridController, PARENT_SHA256, fingerprint, skipped
    if Path(run.__file__).resolve() != (code / 'scripts/listwise_diffgrm/run.py').resolve():
        raise RuntimeError('Wrong historical trainer imported')
    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('This protocol requires one H100')
    config_path = ROOT / f'config_{args.arm}.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    cfg = config['base_config']
    if config['arm'] != args.arm or cfg['seed'] != 2024:
        raise ValueError('Wrong arm/seed config')
    expected_grid = dict(mode='row' if args.arm == 'B' else 'row_column', heads=4,
                         norm_eps=1e-5, initialization_seed=2024)
    if config['grid'] != expected_grid or (config['start_epoch'], config['freeze_epoch'], config['end_epoch']) != (5, 20, 35):
        raise ValueError('Grid or freeze settings differ from the locked protocol')
    parent_path = original / 'runs/H1/initial.pt'
    if training.sha256(parent_path) != PARENT_SHA256:
        raise ValueError('Original H1 R5 checkpoint changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    spec = {key: value for key, value in config.items() if key != 'base_config'}
    spec.update(parent_sha256=PARENT_SHA256, config_sha256=training.sha256(config_path),
                extension_sha256={name: training.sha256(ROOT / name)
                                  for name in ('grid.py', 'protocol.py', 'experiment.py')})
    output = args.output.resolve()
    with run.run_lock(output, resume=args.resume):
        trainer, assets, data, binding = run.build_context(cfg, 'cuda:0', 'bf16', True)
        if len(data) != 709443 or len(trainer.plan) != 45:
            raise ValueError('Original data/schedule changed')
        controller = GridController(trainer, parent, binding, spec)
        if args.resume:
            controller.restore(torch.load(output / 'last.pt', map_location='cpu', weights_only=False))
        else:
            write_json(output / 'resolved_config.json', config)
            write_json(output / 'binding.json', binding)
            write_json(output / 'experiment_spec.json', spec)
            initial = controller.checkpoint()
            training.atomic_save(output / 'initial.pt', initial)
            training.atomic_save(output / 'best.pt', initial)
            training.atomic_save(output / 'last.pt', initial)
            write_json(output / 'R5_FORK.json', dict(
                parent=str(parent_path), parent_sha256=PARENT_SHA256,
                inherited_state_preserved=True, grid_parameters=sum(p.numel() for p in controller.grid.parameters()),
                grid_mode=spec['grid']['mode'], r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                grid_adam_updates=0, r_lr=trainer.r_optimizer.param_groups[0]['lr']))
        del parent
        t = cfg['training']
        while controller.progress['phase'] < len(trainer.plan):
            progress = controller.progress
            phase = progress['phase']
            row = trainer.plan[phase]
            if skipped(row):
                controller.assert_frozen()
                controller.finish_phase()
                training.atomic_save(output / 'last.pt', controller.checkpoint())
                print(json.dumps(dict(event='T_skipped_frozen',epoch=row['epoch'],T=trainer.t_updates)), flush=True)
                continue
            indices = run.phase_indices(cfg, len(data), phase, row)
            if progress['update'] == 0:
                training.seed_all(cfg['seed'] + 100003 * (phase + 1))
            while progress['update'] < row['updates']:
                start = progress['update'] * t['effective_batch_size']
                selected = indices[start:start + t['effective_batch_size']]
                values = controller.update(run.microbatches(data, selected, t['microbatch_size']), row['kind'])
                record = dict(phase_index=phase, epoch=row['epoch'], update=progress['update'], **values)
                with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                if progress['update'] % t['checkpoint_every_updates'] == 0:
                    training.atomic_save(output / 'last.pt', controller.checkpoint())
                    print(json.dumps(record), flush=True)
            if row['kind'] == 'T':
                trainer.refresh_catalog()
            save_best = False
            if row['validate']:
                training.atomic_save(output / 'last.pt', controller.checkpoint())
                dest = fresh_destination(output / f"validation_R{row['epoch']:02d}")
                result = run.evaluate(trainer, assets, split='validation', destination=dest)
                score = result['metrics']['list/ndcg@5']
                progress['validations'].append(str(dest))
                if progress['best_score'] is None or score > progress['best_score']:
                    progress['best_score'], progress['best_epoch'] = score, row['epoch']
                    save_best = True
            controller.finish_phase()
            saved = controller.checkpoint()
            if row['validate']:
                checkpoints = output / 'checkpoints'
                checkpoints.mkdir(exist_ok=True)
                cp = checkpoints / f"R{row['epoch']:02d}_T{trainer.t_updates:03d}.pt"
                if cp.exists():
                    previous = torch.load(cp, map_location='cpu', weights_only=False)
                    # A crash before publishing last may replay this boundary.
                    # Never overwrite the retained checkpoint; require same state.
                    for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler'):
                        if fingerprint(previous[key]) != fingerprint(saved[key]):
                            raise ValueError('Retained boundary differs from resumed execution')
                else:
                    training.atomic_save(cp, saved)
                write_json(checkpoints / f"R{row['epoch']:02d}.json", dict(
                    epoch=row['epoch'], T=trainer.t_updates, checkpoint=str(cp), sha256=training.sha256(cp),
                    validation=str(dest), ndcg5=score))
                print(json.dumps(dict(event='validation_complete', arm=args.arm,
                                      epoch=row['epoch'], T=trainer.t_updates, ndcg5=score)), flush=True)
            if row['epoch'] == 20 and row['kind'] == 'R':
                write_json(output / 'FREEZE_R20.json', dict(epoch=20, T=trainer.t_updates,
                           frozen_state_sha256=controller.freeze_hash, grid_continues_training=True))
            if save_best:
                training.atomic_save(output / 'best.pt', saved)
            training.atomic_save(output / 'last.pt', saved)
        if (trainer.r_updates, trainer.t_updates) != (24255, 200):
            raise ValueError('Wrong final update budget')
        controller.assert_frozen()
        best_epoch = controller.progress['best_epoch']
        write_json(output / 'TRAINING_COMPLETE.json', dict(arm=args.arm, seed=2024,
                   R=35, T=200, best_epoch=best_epoch, best_score=controller.progress['best_score']))
        receipts_path = output / 'TEST_RESULTS.json'
        receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
        for label, checkpoint in [('best', output / 'best.pt'), ('R35', output / 'last.pt')]:
            if label == 'R35' and best_epoch == 35:
                if not any(r['label'] == label for r in receipts):
                    receipts.append(dict(label=label, reused='best'))
                continue
            checkpoint_hash = training.sha256(checkpoint)
            existing = next((r for r in receipts if r['label'] == label), None)
            if existing:
                if existing['checkpoint_sha256'] != checkpoint_hash:
                    raise ValueError('Test receipt is for a different checkpoint')
                for artifact in ('summary', 'predictions'):
                    if training.sha256(Path(existing[artifact])) != existing[artifact + '_sha256']:
                        raise ValueError('Accepted evaluation artifact changed')
                continue
            controller.restore(torch.load(checkpoint, map_location='cpu', weights_only=False))
            dest = fresh_destination(output / f'test_{label}')
            before = fingerprint(dict(model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict()))
            result = run.evaluate(trainer, assets, split='test', destination=dest, beam=32)
            after = fingerprint(dict(model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict()))
            if before != after:
                raise ValueError('Evaluation changed model or Tokenizer state')
            receipt = dict(label=label, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                           metrics=result['metrics'])
            for name, filename in [('summary', 'summary.json'), ('predictions', 'per_sample.jsonl.gz')]:
                receipt[name] = str(dest / filename)
                receipt[name + '_sha256'] = training.sha256(dest / filename)
            receipts.append(receipt)
            write_json(receipts_path, receipts)
        write_json(receipts_path, receipts)
        write_json(output / 'GPU_COMPLETE.json', dict(status='passed', arm=args.arm, seed=2024, tests=receipts))


if __name__ == '__main__':
    main()
