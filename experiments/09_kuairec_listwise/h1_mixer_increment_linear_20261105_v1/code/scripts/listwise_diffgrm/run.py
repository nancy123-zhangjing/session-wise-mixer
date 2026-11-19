"""Explicit local runner. No job submission or training occurs on import/inspect."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

import numpy as np
import torch

from .runtime import ROOT, ENGINE, build_model, read_config
from .training import AlternatingTrainer, atomic_save, rng_state, restore_rng, seed_all, sha256


def resolved(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def bindings(config, assets, data):
    source = {p.relative_to(ROOT).as_posix(): sha256(p)
              for p in sorted((ROOT / 'scripts/listwise_diffgrm').glob('*.py'))}
    # genrec.model/dataset/tokenizer are imported by the real DiffGRM model.
    # Bind their bytes too, and reject an accidentally preloaded other checkout.
    for name in ('genrec.model', 'genrec.dataset', 'genrec.tokenizer'):
        expected = ENGINE / (name.replace('.', '/') + '.py')
        if not expected.is_file():
            raise FileNotFoundError(f'Missing engine base module: {expected}')
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != expected.resolve():
            raise ValueError(f'Loaded {name} is outside the selected DiffGRM engine')
    engine_sources = list((ENGINE / 'genrec').glob('*.py'))
    engine_sources += list((ENGINE / 'genrec/models/DIFF_GRM').glob('*.py'))
    engine = {p.relative_to(ENGINE).as_posix(): sha256(p) for p in sorted(engine_sources)}
    return dict(config=config, asset_manifest=sha256(assets.root / 'manifest.json'),
                data=data.binding, code=source, engine=engine)


@contextmanager
def run_lock(output, *, resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f'Output exists; choose a new directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.running'
    owner = dict(format='listwise_run_lock_v1', token=uuid.uuid4().hex,
                 pid=os.getpid(), hostname=socket.gethostname(), created_at=time.time())
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f'Run lock exists: {lock}. A lock alone does not prove a live or dead job. '
            'Inspect its owner and the platform job; only after confirming the previous '
            'job is terminal, archive the stale lock and resume the last checkpoint.') from exc
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        # Never remove a replacement owner's lock if an operator recovered this
        # path while the old process was still unwinding.
        if lock.exists():
            try:
                current = json.loads(lock.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                current = {}
            if current.get('token') == owner['token']:
                lock.unlink(missing_ok=True)


def publish_checkpoint(output, state, *, save_best):
    """Publish a promoted best before the cursor that says promotion completed.

    On interruption between writes, last may replay deterministic work but can
    never advance its best_score beyond the checkpoint actually saved as best.
    """
    output = Path(output)
    if save_best:
        atomic_save(output / 'best.pt', state)
    atomic_save(output / 'last.pt', state)


def build_context(config, device, precision, mixer):
    from .assets import load_assets
    from .alignment import build_alignment
    from .data import KuaiRecListDataset
    deterministic = bool(config['training'].get('deterministic_algorithms', True))
    if device.startswith('cuda') and deterministic:
        required_workspace = ':4096:8'
        current_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if current_workspace not in (None, required_workspace):
            raise ValueError('CUBLAS_WORKSPACE_CONFIG must match the locked :4096:8 setting')
        if torch.cuda.is_initialized() and current_workspace is None:
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before initializing CUDA')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = required_workspace
    torch.use_deterministic_algorithms(deterministic)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable')
    # Catalog assignment is sensitive to small distance changes; match the
    # earlier precision audit rather than silently enabling H100 TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    dataset_dir = resolved(config['dataset'])
    assets = load_assets(resolved(config['assets']), dataset_dir=dataset_dir)
    if (assets.n_digit, assets.codebook_size) != (config['model']['n_digit'], config['model']['codebook_size']):
        raise ValueError('Asset codebook layout differs from model config')
    data = KuaiRecListDataset(dataset_dir, 'train', assets=assets)
    if data.config['history_max'] != config['history_max'] or data.config['list_length'] != config['list_length']:
        raise ValueError('Model/data history or target length mismatch')
    seed_all(config['seed'])
    model = build_model(config, assets.make_tokenizer()).to(device)
    alignment = build_alignment(assets, mixer_enabled=mixer, config_overrides=config['alignment']).to(device)
    trainer = AlternatingTrainer(model, alignment, config, len(data), precision=precision)
    return trainer, assets, data, bindings(config, assets, data)


def phase_indices(config, data_count, phase_number, row):
    """Independent, paired data RNG; no global torch/dropout state is consumed."""
    generator = np.random.default_rng(np.random.SeedSequence([config['seed'], 7919, phase_number]))
    order = generator.permutation(data_count)
    if row['kind'] == 'R':
        return order
    needed = row['updates'] * config['training']['effective_batch_size']
    pieces = [order]
    while sum(len(p) for p in pieces) < needed:
        pieces.append(generator.permutation(data_count))
    return np.concatenate(pieces)[:needed]


def microbatches(data, indices, size):
    collator = data.collator()
    for start in range(0, len(indices), size):
        yield collator(data[int(i)] for i in indices[start:start + size])


@torch.no_grad()
def make_catalog(alignment, assets):
    from .decoding import SIDCatalog
    alignment.eval()
    books, _ = alignment.mixer.crossed_codebooks(False)
    errors = []
    for start in range(1, assets.n_items + 1, 1024):
        slots = alignment.item_slots[start:start + 1024]
        codes = alignment.hard_codes[start:start + 1024]
        reconstructed = alignment.mixer.gather_codes(books, codes)
        errors.append((slots - reconstructed).square().sum((-1, -2)).cpu())
    return SIDCatalog(item_ids=torch.arange(1, assets.n_items + 1),
                      sids=alignment.hard_codes[1:].cpu(), reconstruction_errors=torch.cat(errors),
                      original_item_ids=assets.original_item_ids[1:], codebook_size=assets.codebook_size)


def validate_evaluation_checkpoint(state, plan):
    """Only complete R boundaries have the formal SID/decoder pairing."""
    progress = state.get('progress', {})
    if state.get('preflight_only') or progress.get('preflight_only'):
        raise ValueError('Preflight-only checkpoints cannot be used for formal evaluation')
    phase, update = progress.get('phase'), progress.get('update')
    if (type(phase) is not int or type(update) is not int or update != 0
            or not 1 <= phase <= len(plan) or plan[phase - 1]['kind'] != 'R'):
        raise ValueError('Evaluation requires a complete R-phase boundary with update=0; intermediate/T checkpoints are rejected')


@torch.no_grad()
def evaluate(trainer, assets, *, split, destination, beam=None, limit=None, subset_manifest=None):
    from .data import KuaiRecListDataset
    from .decoding import generate_lists
    from .metrics import evaluate_generation
    cfg = trainer.config
    if split not in ('validation', 'test'):
        raise ValueError('Evaluation split must be validation or test')
    if subset_manifest is not None and limit is not None:
        raise ValueError('Use either a locked hash subset or an explicit diagnostic prefix limit, not both')
    if subset_manifest is not None and split != 'validation':
        raise ValueError('The locked sensitivity subset is validation-only')
    state = rng_state()
    trainer.model.eval()
    trainer.alignment.eval()
    data = KuaiRecListDataset(resolved(cfg['dataset']), split, assets=assets)
    subset = None
    if subset_manifest is not None:
        from .subset import load_subset_indices
        selected, subset = load_subset_indices(resolved(subset_manifest), resolved(cfg['dataset']), split=split)
        indices = np.asarray(selected, dtype=np.int64)
    else:
        total = len(data) if limit is None else min(int(limit), len(data))
        indices = np.arange(total)
    catalog = make_catalog(trainer.alignment, assets)
    width = cfg['evaluation']['val_beam' if split == 'validation' else 'test_beam'] if beam is None else beam
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    count, sums, users, groups = 0, {}, {}, {'cold_target': {}, 'seen_targets': {}}
    counters = {}
    group_counts = dict.fromkeys(groups, 0)
    started = time.perf_counter()
    total = len(indices)
    if total <= 0 or width <= 0:
        raise ValueError('Evaluation limit and beam must be positive')
    try:
        with gzip.open(destination / 'per_sample.jsonl.gz', 'wt', encoding='utf-8') as f:
            for batch in microbatches(data, indices, cfg['evaluation']['batch_size']):
                ids = batch['history_item_ids'].to(trainer.device)
                hsid = trainer.alignment.hard_codes[ids]
                output = generate_lists(trainer.model, hsid, batch['history_mask'].to(trainer.device),
                                        catalog, route=cfg['architecture'], list_length=cfg['list_length'],
                                        beam_size=width)
                per_sample = evaluate_generation(output, batch['target_item_ids'],
                    catalog_item_ids=torch.arange(1, assets.n_items + 1),
                    aggregate_ks=cfg['evaluation']['aggregate_ks'],
                    max_candidate_lists=cfg['evaluation']['candidate_lists'])
                metric_rows = {k: v.cpu().tolist() for k, v in per_sample.items()}
                predictions = output.item_ids.cpu().tolist()
                scores = output.scores.cpu().tolist()
                beam_mask = output.beam_mask.cpu().tolist()
                complete_mask = output.complete_mask.cpu().tolist()
                for k, v in output.counters.items():
                    if isinstance(v, (int, float)):
                        counters[k] = counters.get(k, 0) + v
                for i, sample_id in enumerate(batch['sample_id']):
                    metrics = {k: v[i] for k, v in metric_rows.items()}
                    user = int(batch['user_id'][i])
                    cold = bool(batch['target_cold_start'][i].any())
                    group = 'cold_target' if cold else 'seen_targets'
                    group_counts[group] += 1
                    user_row = users.setdefault(user, {'count': 0, 'sums': {}})
                    user_row['count'] += 1
                    for k, v in metrics.items():
                        sums[k] = sums.get(k, 0.0) + v
                        user_row['sums'][k] = user_row['sums'].get(k, 0.0) + v
                        groups[group][k] = groups[group].get(k, 0.0) + v
                    f.write(json.dumps(dict(sample_id=sample_id, dataset_index=int(indices[count]),
                                            user_id=user, cold_target=cold,
                                            targets=batch['target_item_ids'][i].tolist(),
                                            predictions=predictions[i],
                                            scores=[s if np.isfinite(s) else None for s in scores[i]],
                                            beam_mask=beam_mask[i], complete_mask=complete_mask[i],
                                            metrics=metrics), allow_nan=False) + '\n')
                    count += 1
                if count % 1000 < len(batch['sample_id']):
                    print(json.dumps(dict(event='evaluation_progress', split=split, processed=count,
                                          total=total, seconds=time.perf_counter() - started)), flush=True)
        result = dict(split=split, examples=count, full_split=limit is None and subset is None,
                      subset=subset, diagnostic_prefix_limit=limit, dataset_binding=data.binding,
                      catalog_size=assets.n_items,
                      metric_protocol=dict(list_length=cfg['list_length'],
                          aggregate_ks=cfg['evaluation']['aggregate_ks'],
                          candidate_lists=cfg['evaluation']['candidate_lists'], pad_id=0,
                          target_relevance='unordered_set_of_five_future_distinct_items',
                          aggregation=cfg['evaluation']['aggregation']),
                      beam=width, inference_precision='fp32', architecture=cfg['architecture'], seed=cfg['seed'], r_updates=trainer.r_updates,
                      t_updates=trainer.t_updates, seconds=time.perf_counter() - started,
                      decode_counters=counters,
                      metrics={k: v / count for k, v in sums.items()},
                      user_average={k: sum(u['sums'][k] / u['count'] for u in users.values()) / len(users)
                                    for k in sums},
                      groups={g: dict(examples=group_counts[g], metrics={
                          k: v / group_counts[g] for k, v in values.items()}) for g, values in groups.items()})
        (destination / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        restore_rng(state)


def train(args, config):
    warmup = args.command == 'warmup'
    arm = 'warmup' if warmup else args.arm
    mixer = arm.endswith('1')
    if not warmup and arm[0] != config['architecture']:
        raise ValueError('Arm architecture differs from config')
    output = resolved(args.output)
    with run_lock(output, resume=bool(args.resume)):
        trainer, assets, data, binding = build_context(config, args.device, args.precision, mixer)
        attempt = uuid.uuid4().hex
        progress = dict(phase=0, update=0, best_score=None, best_epoch=None, validations=[])
        if args.resume or not warmup:
            source = resolved(args.resume or args.shared_warmup)
            state = torch.load(source, map_location='cpu', weights_only=False)
            progress = trainer.restore(state, binding, arm=arm, fork=not bool(args.resume))
            if not args.resume:
                # Shared warmup's best is a valid common candidate, but must be
                # saved under the actual arm identity for later evaluation.
                atomic_save(output / 'best.pt', trainer.checkpoint(binding, progress, arm))
        (output / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (output / 'binding.json').write_text(json.dumps(binding, indent=2), encoding='utf-8')
        with (output / 'attempts.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(attempt=attempt, restored_from=args.resume or
                (None if warmup else args.shared_warmup), resume_progress=progress,
                r_updates=trainer.r_updates, t_updates=trainer.t_updates)) + '\n')
        atomic_save(output / 'initial.pt', trainer.checkpoint(binding, progress, arm)) if not args.resume else None
        end = config['training']['warmup_epochs'] if warmup else len(trainer.plan)
        t = config['training']
        while progress['phase'] < end:
            phase_id = progress['phase']
            row = trainer.plan[phase_id]
            indices = phase_indices(config, len(data), phase_id, row)
            if progress['update'] == 0:
                # Keep arm-paired stochastic streams even if extra validation
                # or a restart happened in one arm. Mixer dropout is fixed zero.
                seed_all(config['seed'] + 100003 * (phase_id + 1))
            while progress['update'] < row['updates']:
                start = progress['update'] * t['effective_batch_size']
                selected = indices[start:start + t['effective_batch_size']]
                values = trainer.update(microbatches(data, selected, t['microbatch_size']), row['kind'])
                progress['update'] += 1
                with (output / 'training.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(attempt=attempt, phase_index=phase_id, epoch=row['epoch'],
                        update=progress['update'], **values)) + '\n')
                if progress['update'] % t['checkpoint_every_updates'] == 0:
                    atomic_save(output / 'last.pt', trainer.checkpoint(binding, progress, arm))
            if row['kind'] == 'T':
                trainer.refresh_catalog()
            if row['validate']:
                # Full validation can be much longer than an update. Persist
                # the completed training boundary first so interruption there
                # resumes validation without repeating the epoch's tail.
                atomic_save(output / 'last.pt', trainer.checkpoint(binding, progress, arm))
                destination = output / f"validation_R{row['epoch']:02d}"
                # Interrupted evaluation is an incomplete artifact, never an
                # accepted best score. Use a fresh retry directory on resume.
                retry = 0
                while destination.exists():
                    retry += 1
                    destination = output / f"validation_R{row['epoch']:02d}_retry{retry}"
                result = evaluate(trainer, assets, split='validation', destination=destination)
                score = result['metrics']['list/ndcg@5']
                progress['validations'].append(str(destination))
                if progress['best_score'] is None or score > progress['best_score']:
                    progress['best_score'], progress['best_epoch'] = score, row['epoch']
                    # Save after advancing phase below, so exact continuation
                    # cannot replay the already completed epoch.
                    save_best = True
                else:
                    save_best = False
            else:
                save_best = False
            progress['phase'] += 1
            progress['update'] = 0
            saved = trainer.checkpoint(binding, progress, arm)
            publish_checkpoint(output, saved, save_best=save_best)
        (output / 'complete.json').write_text(json.dumps(dict(arm=arm, r_updates=trainer.r_updates,
            t_updates=trainer.t_updates, best_epoch=progress['best_epoch'], best_score=progress['best_score']), indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/09_kuairec_listwise/base.json')
    parser.add_argument('--architecture', choices=['J', 'H'])
    parser.add_argument('--seed', type=int)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    for name in ('warmup', 'train', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--device', default='cpu')
        p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
        p.add_argument('--output', required=True)
        if name != 'evaluate':
            p.add_argument('--resume')
        if name == 'train':
            p.add_argument('--arm', choices=['J0', 'J1', 'H0', 'H1'], required=True)
            p.add_argument('--shared-warmup')
        if name == 'evaluate':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--split', choices=['validation', 'test'], default='validation')
            p.add_argument('--beam', type=int)
            p.add_argument('--limit', type=int)
            p.add_argument('--subset-manifest', help='Verified hash-selected validation subset; cannot combine with --limit')
    args = parser.parse_args(argv)
    config = read_config(resolved(args.config), args.architecture, args.seed)
    if args.command == 'inspect':
        print(json.dumps(dict(config=config, assets_ready=resolved(config['assets']).is_dir(),
                              action='read_only_inspection'), indent=2))
    elif args.command == 'evaluate':
        state = torch.load(resolved(args.checkpoint), map_location='cpu', weights_only=False)
        if state.get('preflight_only') or state.get('progress', {}).get('preflight_only'):
            raise ValueError('Preflight-only checkpoints cannot be used for formal evaluation')
        trainer, assets, data, binding = build_context(config, args.device, args.precision, state['arm'].endswith('1'))
        validate_evaluation_checkpoint(state, trainer.plan)
        trainer.restore(state, binding, arm=state['arm'])
        print(json.dumps(evaluate(trainer, assets, split=args.split, destination=resolved(args.output),
                                  beam=args.beam, limit=args.limit, subset_manifest=args.subset_manifest), indent=2))
    else:
        if args.command == 'train' and not (args.shared_warmup or args.resume):
            parser.error('train requires --shared-warmup or --resume')
        train(args, config)


if __name__ == '__main__':
    main()
