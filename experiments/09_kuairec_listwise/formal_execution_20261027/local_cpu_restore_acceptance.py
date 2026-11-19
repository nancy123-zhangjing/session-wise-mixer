"""CPU-only forward restoration of a real, SHA-locked Stage09 checkpoint.

Requires an independently restored code/assets/checkpoint tree and the existing
hash-bound dataset. Preserves the original checkpoint/binding and CUDA RNG;
does not call the CUDA-dependent training restore entry or resume training.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).absolute().parents[3]
RESTORE_PARENT = WORKSPACE / 'outputs/stage09_restore'
REMOTE_ROOT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
SOURCE_COMMIT = '1c423662d9a25a926328ee6466c30c0f2857b4f6'
ENGINE_COMMIT = '98b834cc5ce4428d1a95b3e4f9651833b5fbd5cd'
BINDING_PATH = Path(__file__).absolute().with_name('FORMAL_BRANCH_BINDING.json')
BINDING_SHA = '871e480d2a77d7e22bfed091d02065f0cdc42eb1db42cf20d43519634959e80a'
SUBSET_PATH = 'experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json'
SUBSET_SHA = 'c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def safe_path(value):
    path = Path(value).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            continue
        info = current.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not
                (getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
                f'Symlink/junction refused: {current}')
    require(path.resolve(strict=False) == path, f'Path resolves elsewhere: {path}')
    return path


def file_sha(path):
    path = safe_path(path)
    require(path.is_file(), f'Required file is absent: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f'File changed during read: {path}')
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='Complete real checkpoint inside the independent restore tree')
    parser.add_argument('--checkpoint-sha256', required=True, help='Expected SHA from the frozen delivery receipt')
    parser.add_argument('--source-root', required=True, type=Path, help='Restored frozen main checkout, containing external/DiffGRM')
    parser.add_argument('--assets', required=True, type=Path, help='Restored assets/pse_v1 directory')
    parser.add_argument('--dataset', required=True, type=Path, help='Existing hash-bound KuaiRec run1 directory')
    parser.add_argument('--output', required=True, type=Path, help='New JSON receipt outside restored code/assets')
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{64}', args.checkpoint_sha256):
        parser.error('--checkpoint-sha256 must be 64 lowercase hexadecimal characters')
    return args


def locations(args):
    paths = {name: safe_path(getattr(args, name)) for name in
             ('checkpoint', 'source_root', 'assets', 'dataset', 'output')}
    source, assets, output = paths['source_root'], paths['assets'], paths['output']
    require(source.is_relative_to(RESTORE_PARENT) and source.parent != RESTORE_PARENT,
            'Source must be in a dedicated outputs/stage09_restore/<run>/code tree')
    restore_root = source.parent
    require(all(paths[name].is_relative_to(restore_root) for name in ('checkpoint', 'assets', 'output')),
            'Checkpoint, assets and output must belong to the same independent restore tree')
    require(not assets.is_relative_to(source) and not paths['checkpoint'].is_relative_to(source),
            'Restore artifacts must be outside the frozen code tree')
    require(not output.is_relative_to(source) and not output.is_relative_to(assets),
            'Receipt must not modify frozen source or assets')
    require(paths['dataset'].is_relative_to(WORKSPACE), 'Dataset must be an explicit local workspace reference')
    require(not os.path.lexists(output), 'Output receipt already exists')
    return paths


def verify_source_files(source, binding):
    engine = source / 'external/DiffGRM'
    sources = {path.relative_to(source).as_posix() for path in (source / 'scripts/listwise_diffgrm').glob('*.py')}
    engine_sources = {path.relative_to(engine).as_posix() for directory in
        (engine / 'genrec', engine / 'genrec/models/DIFF_GRM') for path in directory.glob('*.py')}
    for root, expected, actual in ((source, binding['code'], sources), (engine, binding['engine'], engine_sources)):
        require(set(expected) == actual, 'Restored executable source-file set differs from checkpoint binding')
        for name, digest in expected.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and str(relative) == name and '\\' not in name and
                    ':' not in name and all(part not in ('', '.', '..') for part in relative.parts),
                    'Unsafe source binding path')
            require(file_sha(root / name) == digest, f'Restored source bytes differ: {name}')
    return engine


def accept(args):
    paths = locations(args)
    source, checkpoint_path = paths['source_root'], paths['checkpoint']
    require(file_sha(checkpoint_path) == args.checkpoint_sha256, 'Checkpoint SHA differs from expected delivery identity')
    for name in ('source_root', 'assets', 'dataset'):
        require(paths[name].is_dir(), f'Required directory is absent: {name}')
    # Process-local isolation only; no packages or persistent environment change.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    require(torch.version.cuda is None, 'Use the already-verified CPU-only PyTorch interpreter')
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    torch.set_num_threads(4)

    def fingerprint(value):
        digest = hashlib.sha256()

        def visit(item):
            if torch.is_tensor(item):
                require(item.device.type == 'cpu', 'Loaded tensor is not on CPU')
                tensor = item.detach().contiguous()
                require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()), 'Nonfinite stored tensor')
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, np.ndarray):
                digest.update(str((item.dtype, item.shape)).encode()); digest.update(item.tobytes())
            elif isinstance(item, dict):
                digest.update(b'dict')
                for key in sorted(item, key=repr):
                    digest.update(repr(key).encode()); visit(item[key])
            elif isinstance(item, (tuple, list)):
                digest.update(type(item).__name__.encode())
                for part in item:
                    visit(part)
            else:
                digest.update(repr(item).encode())
        visit(value)
        return digest.hexdigest()

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(file_sha(checkpoint_path) == args.checkpoint_sha256, 'Checkpoint changed while loading')
    require(state['format'] == 'kuairec_listwise_v1' and state['arm'] in ('J0', 'J1', 'H0', 'H1') and
            state['precision'] == 'bf16', 'Not a formal Stage09 BF16-identity arm checkpoint')
    require(not state.get('preflight_only') and not state['progress'].get('preflight_only'),
            'Preflight checkpoints cannot substitute for the formal model recovery')
    original_binding = state['binding']
    original_config = original_binding['config']
    require(original_config['architecture'] == state['arm'][0] and original_config['seed'] == 2024,
            'Checkpoint arm/architecture/seed differs')
    require(file_sha(BINDING_PATH) == BINDING_SHA, 'Pinned formal source/config/data/asset binding differs')
    formal = json.loads(BINDING_PATH.read_text(encoding='utf-8'))
    require(formal['execution_source'] == SOURCE_COMMIT and formal['root'] == REMOTE_ROOT and
            original_binding == formal['expected_bindings'][state['arm'][0]],
            'Checkpoint binding is not the accepted frozen formal experiment')
    mapping = dict(dataset=dict(original=REMOTE_ROOT + '/inputs/dataset', local=str(paths['dataset'])),
                   assets=dict(original=REMOTE_ROOT + '/assets/pse_v1', local=str(paths['assets'])))
    require(all(original_config[key] == value['original'] for key, value in mapping.items()),
            'Unexpected original physical paths; do not silently rewrite a different experiment')
    engine = verify_source_files(source, original_binding)
    require(file_sha(source / SUBSET_PATH) == SUBSET_SHA, 'Fixed validation-history subset manifest differs')
    prefixes = ('scripts.listwise_diffgrm', 'genrec', '_kuairec_listwise_engine')
    require('scripts' not in sys.modules and not any(
            name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in prefixes),
            'Start a fresh process: executable project modules were already imported')
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(engine)
    sys.path.insert(0, str(source))
    modules = {name: importlib.import_module('scripts.listwise_diffgrm.' + name) for name in
               ('assets', 'runtime', 'alignment', 'training', 'subset', 'run')}
    config = copy.deepcopy(original_config)
    for key, value in mapping.items():
        config[key] = value['local']
    modules['runtime'].validate_config(config)
    assets = modules['assets'].load_assets(paths['assets'], dataset_dir=paths['dataset'])
    model = modules['runtime'].build_model(config, assets.make_tokenizer()).to('cpu')
    alignment = modules['alignment'].build_alignment(assets, state['arm'].endswith('1'),
        config_overrides=config['alignment']).to('cpu')
    actual_binding = modules['run'].bindings(config, assets, SimpleNamespace(binding=assets.manifest['dataset_binding']))
    require(set(actual_binding) == set(original_binding), 'Binding fields differ')
    for key in original_binding:
        if key != 'config':
            require(actual_binding[key] == original_binding[key], f'Restored binding differs: {key}')
    require(set(config) == set(original_config) and
            all(config[key] == value for key, value in original_config.items() if key not in mapping),
            'Configuration changed beyond the two declared physical path mappings')
    imported = {}
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        actual = safe_path(filename)
        expected_root = source / 'scripts/listwise_diffgrm' if name.startswith('scripts.listwise_diffgrm') else engine / 'genrec'
        require(actual.is_relative_to(expected_root), f'Imported module is outside restored source: {name}')
        imported[name] = str(actual)
    require(Path(modules['runtime'].ROOT) == source and Path(modules['runtime'].ENGINE) == engine,
            'Runtime root/engine mapping differs')
    train_count = pq.read_metadata(paths['dataset'] / 'train.parquet').num_rows
    require(train_count == 709443 and assets.n_items == 10728, 'Training/catalog cardinality differs')
    trainer = modules['training'].AlternatingTrainer(model, alignment, config, train_count, precision='bf16')
    modules['run'].validate_evaluation_checkpoint(state, trainer.plan)
    counts = modules['training'].expected_update_counts(trainer.plan, state['progress'])
    for kind in ('R', 'T'):
        count = state[kind.lower() + '_updates']
        require(type(count) is int and count == counts[kind], 'Stored updates disagree with phase cursor')
        modules['training'].validate_optimizer_clock(state[kind.lower() + '_optimizer'], count, kind)
    require(state['scheduler']['last_epoch'] == state['r_updates'] and
            int(state['alignment']['diger_optimizer_step']) == state['t_updates'] and
            int(state['alignment']['diger_total_optimizer_steps']) == 400, 'Stored scheduler/T clock differs')
    require(set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'} and
            state['rng']['cuda'] is not None and len(state['rng']['cuda']) == 1, 'Saved RNG snapshot incomplete')
    # Intentionally use strict state loading, not trainer.restore: its final
    # restore_rng call requires CUDA and is unrelated to an FP32 CPU forward.
    model.load_state_dict(state['model'], strict=True)
    alignment.load_state_dict(state['alignment'], strict=True)
    trainer.r_optimizer.load_state_dict(state['r_optimizer'])
    trainer.t_optimizer.load_state_dict(state['t_optimizer'])
    trainer.scheduler.load_state_dict(state['scheduler'])
    trainer.r_updates, trainer.t_updates = state['r_updates'], state['t_updates']
    loaded = dict(model=model.state_dict(), alignment=alignment.state_dict(),
        r_optimizer=trainer.r_optimizer.state_dict(), t_optimizer=trainer.t_optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict())
    fingerprints = {}
    for key, value in loaded.items():
        require(fingerprint(value) == fingerprint(state[key]), f'CPU loaded state differs: {key}')
        fingerprints[key] = fingerprint(value)
    floating = [value for value in model.state_dict().values() if value.is_floating_point()]
    require(floating and all(value.dtype == torch.float32 and value.device.type == 'cpu' for value in floating),
            'Restored floating model weights must be FP32 on CPU')
    sid = alignment.hard_codes
    require(sid.dtype == torch.long and tuple(sid.shape) == (10729, 4) and bool((sid[0] == -1).all()) and
            bool(((sid[1:] >= 0) & (sid[1:] < 256)).all()), 'Restored SID shape/range/PAD differs')
    require(torch.equal(alignment.item_slots, assets.residual_features), 'Frozen content slots differ from restored assets')
    indices, subset = modules['subset'].load_subset_indices(source / SUBSET_PATH, paths['dataset'], split='validation')
    index = indices[0]
    table = pq.read_table(paths['dataset'] / 'validation.parquet', columns=['sample_id', 'history_item_ids'])
    sample_id = table['sample_id'][index].as_py()
    history = table['history_item_ids'][index].as_py()
    subset_rows = json.loads((source / SUBSET_PATH).read_text(encoding='utf-8'))['rows']
    require(sample_id == subset_rows[0]['sample_id'] and index == subset_rows[0]['index'], 'Selected history identity differs')
    require(1 <= len(history) <= 50 and all(type(item) is int and 1 <= item <= 10728 for item in history),
            'Invalid real validation history')
    history_ids = torch.zeros((1, 50), dtype=torch.long)
    history_ids[0, -len(history):] = torch.tensor(history, dtype=torch.long)
    history_mask = history_ids != 0
    history_sid = alignment.hard_codes[history_ids]
    architecture = original_config['architecture']
    positions = 20 if architecture == 'J' else 4
    partial = torch.full((1, positions), -1, dtype=torch.long)
    model.eval(); alignment.eval(); alignment.set_recommender_cache(False)
    with torch.inference_mode():
        logits = model.decode_logits(history_sid, history_mask, partial, item_index=None if architecture == 'J' else 0)
    require(tuple(logits.shape) == (1, positions, 256) and logits.dtype == torch.float32 and
            logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()), 'CPU target-free logits check failed')
    require(not torch.cuda.is_initialized(), 'CPU restoration unexpectedly initialized CUDA')
    require(file_sha(checkpoint_path) == args.checkpoint_sha256, 'Original checkpoint changed')
    return dict(schema='stage09_local_cpu_model_restore_v1', status='PASS',
        checked_at_utc=datetime.now(timezone.utc).isoformat(), arm=state['arm'], architecture=architecture,
        checkpoint=dict(path=str(checkpoint_path), bytes=checkpoint_path.stat().st_size, sha256=args.checkpoint_sha256),
        source_root=str(source), engine_root=str(engine), expected_source_commit=SOURCE_COMMIT,
        expected_engine_commit=ENGINE_COMMIT, source_verified_by='exact checkpoint-bound source bytes; not inferred from cwd HEAD',
        frozen_formal_binding_sha256=BINDING_SHA,
        physical_path_mapping=mapping, unchanged_original_binding_sha256=hashlib.sha256(
            json.dumps(original_binding, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
        config_changes_limited_to_physical_paths=True, imported_module_paths=imported,
        state_fingerprints=fingerprints, strict_model_alignment_load=True, optimizer_scheduler_load_exact=True,
        r_updates=trainer.r_updates, t_updates=trainer.t_updates, progress=state['progress'],
        checkpoint_precision='bf16', stored_model_floating_dtype='torch.float32', inference_precision='fp32',
        sid=dict(shape=list(sid.shape), real_code_range=[0, 255], pad_row=-1, fingerprint=fingerprint(sid)),
        rng=dict(saved_snapshot_fingerprint=fingerprint(state['rng']), cuda_streams_preserved=1,
                 cuda_rng_restored=False, exact_gpu_training_resume_verified=False),
        forward=dict(method='decode_logits', true_target_argument=False, target_columns_read=False,
            sample_id=sample_id, dataset_index=index, history_length=len(history),
            subset_manifest_sha256=SUBSET_SHA, subset_selected_rows_sha256=subset['selected_rows_sha256'],
            logits_shape=list(logits.shape), logits_dtype=str(logits.dtype), finite=True,
            logits_fingerprint=fingerprint(logits), minimum=float(logits.min()), maximum=float(logits.max())),
        environment=dict(python=sys.version, executable=sys.executable,
                         versions={name:importlib.metadata.version(name) for name in ('torch', 'numpy', 'pyarrow')}),
        gpu_used=False, network_used=False, checkpoint_modified=False, script_sha256=file_sha(Path(__file__)),
        limitations=['CPU forward restoration only; no CUDA RNG replay or cross-device bitwise equality claim.',
                     'No optimizer update, model training, full-list beam evaluation, or recommendation-quality result is produced.'])


def main(argv=None):
    args = parse_args(argv)
    result = accept(args)
    output = safe_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with output.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    require(output.read_bytes() == payload, 'Restore receipt readback differs')
    print(json.dumps(dict(status='PASS', arm=result['arm'], output=str(output),
                         report_sha256=hashlib.sha256(payload).hexdigest()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
