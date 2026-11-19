"""Describe saved SID collisions after four-arm final selection; CPU only.

Requires the real SHA-bound final selection receipt and independently restored
best/last checkpoints. Does not construct a model, infer, reassign SIDs, or read
evaluation labels. Collision statistics do not establish a cause of low coverage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).absolute().parents[3]
RESTORE_PARENT = WORKSPACE / 'outputs/stage09_restore'
REMOTE_ROOT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
SOURCE = '1c423662d9a25a926328ee6466c30c0f2857b4f6'
BINDING_SHA = '871e480d2a77d7e22bfed091d02065f0cdc42eb1db42cf20d43519634959e80a'
BRANCH_HELPER_SHA = '2c255470cdd58b2c13dd50fb130aa750f1609306720081ab0e3cd33050cb2831'
FINAL_HELPER_SHA = 'd5b981a6077dead491289965d559a235082d63e6bf73fbca4cd4a42896f09935'
ARMS = ('J0', 'J1', 'H0', 'H1')
ITEMS = 10728
CHUNK = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def safe_path(value):
    path = Path(value).absolute()
    for entry in reversed((path, *path.parents)):
        if os.path.lexists(entry):
            info = entry.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400), 'Link/reparse path refused: ' + str(entry))
    require(path.resolve(strict=False) == path, 'Path resolves elsewhere: ' + str(path))
    return path


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(stream):
    value, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(CHUNK), b''):
        value.update(block)
        size += len(block)
    return dict(bytes=size, sha256=value.hexdigest())


def file_identity(path):
    path = safe_path(path)
    require(path.is_file(), 'Required regular file absent: ' + str(path))
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        result = digest(stream)
        require(identity(before) == identity(os.fstat(stream.fileno())), 'File changed during read')
    return result


def read_bound_json(path, expected_sha):
    require(re.fullmatch('[0-9a-f]{64}', expected_sha or ''), 'Expected SHA must be lowercase hex')
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size <= 8 * 1024 * 1024, 'Missing/oversized JSON receipt')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(8 * 1024 * 1024 + 1)
        require(identity(before) == identity(os.fstat(stream.fileno())), 'JSON changed during read')
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'JSON SHA mismatch')
    return json.loads(raw)


def clocks(epoch):
    require(type(epoch) is int and epoch in range(5, 36, 3), 'Invalid accepted validation epoch')
    cycle = (epoch - 5) // 3
    return dict(r_updates=epoch * 693, t_updates=cycle * 40, phase=5 + cycle * 4, update=0)


def validate_selection(receipt):
    require(receipt.get('schema') == 'stage09_four_arm_final_selection_v1' and receipt.get('status') == 'passed'
            and receipt.get('root') == REMOTE_ROOT and receipt.get('execution_source') == SOURCE
            and receipt.get('branch_binding_sha256') == BINDING_SHA
            and receipt.get('branch_helper_sha256') == BRANCH_HELPER_SHA
            and receipt.get('script_sha256') == FINAL_HELPER_SHA
            and receipt.get('distinct_full_validations') == 42
            and receipt.get('test_used_for_selection') is False and receipt.get('gpu_used') is False,
            'Not the accepted final four-arm selection receipt')
    for key in ('arms', 'four_best_sha256', 'four_best_bytes'):
        require(set(receipt.get(key, {})) == set(ARMS), 'Final receipt must cover exactly four arms')
    for arm in ARMS:
        row = receipt['arms'][arm]
        selection = row['selection']
        require(selection['metric'] == 'list/ndcg@5' and selection['split'] == 'validation'
                and selection['beam'] == 32 and selection['tie_break'] == 'earlier'
                and type(selection['score']) in (int, float) and math.isfinite(selection['score'])
                and 0 <= selection['score'] <= 1 and row['shared_initial_exact'] is True,
                'Selection protocol differs: ' + arm)
        require(selection['best_t_updates'] == clocks(selection['best_epoch'])['t_updates'], 'Best T clock differs')
        for role in ('best', 'last'):
            ref = row[role]
            epoch = selection['best_epoch'] if role == 'best' else 35
            saved = row['best_state' if role == 'best' else 'final_state']
            require(ref['path'] == REMOTE_ROOT + '/runs/' + arm + '/' + role + '.pt'
                    and type(ref['bytes']) is int and ref['bytes'] > 0
                    and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Unexpected bound checkpoint')
            require(saved['epoch'] == epoch and all(saved[k] == v for k, v in clocks(epoch).items())
                    and saved['checkpoint_precision'] == 'bf16'
                    and saved['stored_model_floating_dtype'] == 'torch.float32', 'Accepted state identity differs')
            sid = saved['sid_snapshot']
            require(sid['field'] == 'alignment.hard_codes' and sid['shape'] == [ITEMS + 1, 4]
                    and sid['pad_row'] == -1 and sid['real_code_range'] == [0, 255]
                    and re.fullmatch('[0-9a-f]{64}', sid['fingerprint']), 'Accepted SID identity differs')
        require(row['best']['sha256'] == receipt['four_best_sha256'][arm]
                and row['best']['bytes'] == receipt['four_best_bytes'][arm], 'Four-best identity differs')


def cpu_libraries():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_initialized(), 'Use the CPU-only PyTorch interpreter')
    torch.set_num_threads(1)
    return torch


def load_statistics(source, bindings):
    """Import only the bound package initializer/assets/diagnostic modules."""
    sources = {}
    for name in ('__init__.py', 'assets.py', 'asset_diagnostics.py'):
        relative = 'scripts/listwise_diffgrm/' + name
        expected = bindings['J']['code'][relative]
        require(expected == bindings['H']['code'][relative], 'J/H statistics source differs')
        path = safe_path(source / relative)
        actual = file_identity(path)
        require(actual['sha256'] == expected, 'Frozen statistics source SHA differs: ' + relative)
        sources[relative] = dict(path=str(path), **actual)
    package = '_stage09_saved_sid_statistics'
    require(not any(name == package or name.startswith(package + '.') for name in sys.modules), 'Statistics package already loaded')
    directory = source / 'scripts/listwise_diffgrm'
    for suffix, filename in (('', '__init__.py'), ('.assets', 'assets.py'), ('.asset_diagnostics', 'asset_diagnostics.py')):
        name = package + suffix
        options = {'submodule_search_locations': [str(directory)]} if not suffix else {}
        spec = importlib.util.spec_from_file_location(name, directory / filename, **options)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        raw = (directory / filename).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sources['scripts/listwise_diffgrm/' + filename]['sha256'],
                'Statistics source changed before import')
        # Execute verified source bytes, never a pre-existing bytecode cache.
        exec(compile(raw, str(directory / filename), 'exec'), module.__dict__)
    return module.collision_statistics, sources


def sid_fingerprint(sid):
    # Exact tensor branch of formal_branch_acceptance.fingerprint (pinned above).
    tensor = sid.detach().contiguous()
    value = hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode())
    value.update(tensor.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return value.hexdigest()


def summarize_sid(sid, statistics, *, expected_items=ITEMS):
    import torch
    require(torch.is_tensor(sid) and sid.device.type == 'cpu' and sid.dtype == torch.int64
            and tuple(sid.shape) == (expected_items + 1, 4), 'Wrong saved SID dtype/device/shape')
    require(bool((sid[0] == -1).all()) and bool(((sid[1:] >= 0) & (sid[1:] < 256)).all()),
            'Wrong PAD row or real SID code range')
    # Exclude exactly internal item zero, never all-zero SID tuples of real items.
    return dict(fingerprint=sid_fingerprint(sid), shape=list(sid.shape), dtype=str(sid.dtype),
                excluded_pad_item=0, real_all_zero_sid_items=int((sid[1:] == 0).all(dim=1).sum()),
                collision=statistics(sid[1:].numpy(), 256))


def inspect_checkpoint(path, ref, saved, *, arm, selection, binding, statistics):
    torch = cpu_libraries()
    expected = dict(bytes=ref['bytes'], sha256=ref['sha256'])
    require(file_identity(path) == expected, 'Checkpoint SHA/bytes mismatch: ' + str(path))
    with safe_path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        state = torch.load(stream, map_location='cpu', weights_only=False)
        stream.seek(0)
        require(digest(stream) == expected and identity(before) == identity(os.fstat(stream.fileno())), 'Checkpoint changed while loading')
    require(state['format'] == 'kuairec_listwise_v1' and state['arm'] == arm
            and state['precision'] == 'bf16' and state['binding'] == binding
            and binding['config']['seed'] == 2024 and binding['config']['architecture'] == arm[0]
            and not state.get('preflight_only') and not state['progress'].get('preflight_only'), 'Checkpoint identity differs')
    clock = clocks(saved['epoch'])
    require(all(state[k] == clock[k] and type(state[k]) is int for k in ('r_updates', 't_updates'))
            and all(state['progress'][k] == clock[k] and type(state['progress'][k]) is int for k in ('phase', 'update'))
            and state['progress']['best_epoch'] == selection['best_epoch']
            and state['progress']['best_score'] == selection['score'], 'Checkpoint/selection clock differs')
    result = summarize_sid(state['alignment']['hard_codes'], statistics)
    require(result['fingerprint'] == saved['sid_snapshot']['fingerprint'], 'Saved SID fingerprint differs from final receipt')
    return dict(local_path=str(path), remote_path=ref['path'], **expected, epoch=saved['epoch'],
                **clock, sid=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection-receipt', 'selection-sha256', 'restore-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    receipt_path, restored, output = (safe_path(value) for value in (args.selection_receipt, args.restore_root, args.output))
    require(restored.parent == RESTORE_PARENT and restored.is_dir(), 'Use a dedicated local stage09_restore directory')
    require(output.is_relative_to(WORKSPACE) and not output.is_relative_to(restored)
            and output.parent.is_dir() and not os.path.lexists(output), 'Output must be new, outside restored inputs, in an existing workspace directory')
    receipt = read_bound_json(receipt_path, args.selection_sha256)
    validate_selection(receipt)
    manifest = read_bound_json(Path(__file__).with_name('FORMAL_BRANCH_BINDING.json'), BINDING_SHA)
    require(manifest['root'] == REMOTE_ROOT and manifest['execution_source'] == SOURCE, 'Formal binding identity differs')
    torch = cpu_libraries()
    statistics, source_files = load_statistics(restored / 'code', manifest['expected_bindings'])
    arms = {}
    for arm in ARMS:
        row = receipt['arms'][arm]
        arms[arm] = {role: inspect_checkpoint(restored / 'runs' / arm / (role + '.pt'), row[role],
            row['best_state' if role == 'best' else 'final_state'], arm=arm, selection=row['selection'],
            binding=manifest['expected_bindings'][arm[0]], statistics=statistics) for role in ('best', 'last')}
    require(read_bound_json(receipt_path, args.selection_sha256) == receipt, 'Selection receipt changed')
    for row in arms.values():
        for item in row.values():
            require(file_identity(item['local_path']) == {key: item[key] for key in ('bytes', 'sha256')}, 'Checkpoint changed during diagnostics')
    for item in source_files.values():
        require(file_identity(item['path']) == {key: item[key] for key in ('bytes', 'sha256')}, 'Statistics source changed')
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    result = dict(schema='stage09_checkpoint_sid_diagnostics_v1', status='PASS', created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_receipt=dict(path=str(receipt_path), sha256=args.selection_sha256), remote_root=REMOTE_ROOT,
        restore_root=str(restored), execution_source=SOURCE, branch_binding_sha256=BINDING_SHA,
        script_sha256=file_identity(__file__)['sha256'], statistics_function='asset_diagnostics.collision_statistics',
        statistics_source_files=source_files, sid_fingerprint_algorithm_source_sha256=BRANCH_HELPER_SHA,
        torch_version=torch.__version__, gpu_used=False, model_constructed=False, model_forward=False,
        sid_reassigned=False, source_files_changed=False, arms=arms,
        limitation='Descriptive saved-SID bucket statistics only. They do not establish that collisions caused candidate-coverage shortages; output metrics and causal evidence are separate.')
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with output.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(output.read_bytes() == raw, 'Output readback differs')
    print(json.dumps(dict(status='PASS', output=str(output), sha256=hashlib.sha256(raw).hexdigest(), checkpoint_roles=8)))


if __name__ == '__main__':
    main()
