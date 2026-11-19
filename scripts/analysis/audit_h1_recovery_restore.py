"""Read-only CPU restoration of a final H1 recovery backup using its frozen code.

Run this file in a fresh process. It verifies the execution manifest before
importing archived runtime modules, then records dataset/assets relocation as a
separate proof. Original checkpoint bindings and files are never rewritten.
This is a loading/portability check, not CUDA stochastic-continuation evidence.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
MANIFEST_SHA256 = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
SOURCE_COMMIT = '8b5acc8853101975a0ba09fe904f990549643dcc'
STATE_KEYS = ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_path(code, relative):
    parsed = PurePosixPath(relative)
    if not relative or '\\' in relative or parsed.is_absolute() or '..' in parsed.parts:
        raise ValueError(f'Invalid relative source path: {relative!r}')
    path = code / relative
    if path.is_symlink() or not path.resolve().is_relative_to(code) or not path.is_file():
        raise ValueError(f'Missing or unsafe source: {relative}')
    return path


def verify_execution(code, parent):
    """Authenticate the cold-import tree and parent independently of last.pt."""
    code = Path(code).resolve()
    manifest_path = code / 'RECOVERY_EXECUTION_MANIFEST.json'
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError('Execution manifest differs from the accepted H100 package')
    manifest = read_json(manifest_path)
    if (manifest.get('format') != 'stage09_h1_recovery_execution_package_v1'
            or manifest['workspace'].get('package_sources_uncommitted') is not False
            or manifest['workspace'].get('commit') != SOURCE_COMMIT):
        raise ValueError('Execution source is not the reviewed committed package')
    files = manifest['execution_files']
    if len(files) != manifest['execution_file_count']:
        raise ValueError('Execution file count mismatch')
    for relative, record in files.items():
        path = source_path(code, relative)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Execution bytes differ: {relative}')
    metadata = {'RECOVERY_EXECUTION_MANIFEST.json', 'RECOVERY_PARENT_BINDING.json'}
    observed = set()
    for path in code.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed in restored code: {path}')
        if path.is_file():
            observed.add(path.relative_to(code).as_posix())
    if observed != set(files) | metadata:
        raise ValueError('Restored code contains missing or unbound files')
    binding_file = code / 'RECOVERY_PARENT_BINDING.json'
    if sha256(binding_file) != manifest['parent_binding_json_sha256']:
        raise ValueError('Parent binding metadata differs')
    binding = read_json(binding_file)
    for section, prefix in (('code', ''), ('engine', 'external/DiffGRM/')):
        for relative, expected in binding[section].items():
            if sha256(source_path(code, prefix + relative)) != expected:
                raise ValueError(f'Original {section} binding differs: {relative}')
    if sha256(parent) != PARENT_SHA256 or manifest['parent']['sha256'] != PARENT_SHA256:
        raise ValueError('Parent checkpoint is not the authorized R20 source')
    return manifest, binding


def prove_relocation(recorded, actual):
    """Only two filesystem paths can differ; every scientific byte must match."""
    if set(recorded) != set(actual):
        raise ValueError('Binding schema differs after relocation')
    old, new = copy.deepcopy(recorded['config']), copy.deepcopy(actual['config'])
    relocation = {}
    for name in ('dataset', 'assets'):
        before, after = old.pop(name), new.pop(name)
        relocation[name] = dict(recorded=before, local=after)
    if old != new:
        raise ValueError('Scientific configuration changed during CPU relocation')
    for name in set(recorded) - {'config'}:
        if recorded[name] != actual[name]:
            raise ValueError(f'CPU relocation changed bound {name}')
    return dict(allowed_path_fields=['dataset', 'assets'], paths=relocation,
                scientific_config_equal=True, code_data_assets_equal=True,
                recorded_binding_rewritten=False)


def restore_original_state(trainer, parent, saved, actual_binding, recovery_module):
    """Restore the unmodified recorded identity after independent local proof.

    The trainer opens relocated byte-identical assets. The controller retains
    the recorded experiment identity, with all normal state checks enabled.
    """
    if trainer.device.type != 'cpu':
        raise ValueError('Backup audit must use CPU only')
    proof = prove_relocation(saved['execution_binding'], actual_binding)
    if saved['parent_binding'] != parent['binding']:
        raise ValueError('Final checkpoint names a different parent binding')
    fingerprint = recovery_module.fingerprint
    before = fingerprint(saved)
    parent_before = fingerprint(parent)
    protocol = saved['protocol']
    controller = recovery_module.RecoveryController(
        trainer, protocol['policy'], parent['binding'], saved['execution_binding'],
        protocol['parent_sha256'], start_phase=protocol['start_phase'], stop_phase=protocol['stop_phase'])
    controller.fork(parent, exact_rng=False)
    controller.restore(saved, exact_rng=False)
    restored = controller.checkpoint()
    expected = copy.deepcopy(saved)
    expected['inner']['rng']['cuda'] = None
    if fingerprint(restored) != fingerprint(expected):
        raise ValueError('CPU-restored state differs beyond the disclosed CUDA RNG omission')
    if fingerprint(saved) != before or fingerprint(parent) != parent_before:
        raise ValueError('Restoration changed the in-memory input checkpoint')
    proof.update(cpu_roundtrip_exact_except_cuda_rng=True,
        input_checkpoint_unchanged=True, input_parent_unchanged=True,
        cuda_rng_preserved_in_source=saved['inner']['rng'].get('cuda') is not None,
        cuda_exact_resume_verified=False,
        restored_state_sha256={key: fingerprint(restored['inner'][key]) for key in STATE_KEYS},
        recorded_state_sha256={key: fingerprint(saved['inner'][key]) for key in STATE_KEYS})
    return controller, proof


def require_final_state(saved, complete, policy):
    inner = saved['inner']
    if (saved.get('format') != 'stage09_h1_recovery_v1'
            or saved.get('preflight_only') or inner.get('preflight_only')
            or saved['protocol']['policy'] != policy
            or saved['protocol']['parent_sha256'] != PARENT_SHA256
            or saved['protocol']['start_phase'] != 25 or saved['protocol']['stop_phase'] != 33
            or inner['progress'] != dict(phase=33, update=0, boundary_prepared=False)
            or inner['r_updates'] != 18018 or inner['t_updates'] != (280 if policy == 'decay' else 200)
            or complete.get('policy') != policy or complete.get('endpoint_epoch') != 26
            or complete.get('phase') != 33 or complete.get('r_updates') != inner['r_updates']
            or complete.get('t_updates') != inner['t_updates']):
        raise ValueError('Checkpoint/receipt is not the authorized final R26 endpoint')


def cold_import(code):
    prefixes = ('scripts', 'genrec', '_kuairec_listwise_engine')
    if any(name == root or name.startswith(root + '.') for name in sys.modules for root in prefixes):
        raise RuntimeError('Run this audit file in a fresh process, not python -m or an imported workspace runtime')
    if 'torch' in sys.modules:
        raise RuntimeError('CPU visibility must be fixed before importing torch')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    importlib.invalidate_caches()
    torch = importlib.import_module('torch')
    torch.set_num_threads(2)
    runtime = importlib.import_module('scripts.listwise_diffgrm.runtime')
    run = importlib.import_module('scripts.listwise_diffgrm.run')
    recovery = importlib.import_module('scripts.listwise_diffgrm.recovery')
    if runtime.ROOT.resolve() != code or runtime.ENGINE != (code / 'external/DiffGRM').resolve():
        raise ValueError('Runtime imports escaped the verified archived source')
    return torch, run, recovery


def audit(args):
    code, run_dir = Path(args.code).resolve(), Path(args.run).resolve()
    parent_path = Path(args.parent).resolve()
    dataset, assets = Path(args.dataset).resolve(), Path(args.assets).resolve()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite audit output: {destination}')
    if any(destination.is_relative_to(directory)
           for directory in (code, run_dir, dataset, assets, parent_path.parent)):
        raise ValueError('Audit output must be outside all read-only input trees')
    manifest, parent_binding = verify_execution(code, parent_path)
    checkpoint, complete_path = run_dir / 'last.pt', run_dir / 'complete.json'
    complete = read_json(complete_path)
    original_hashes = {str(path): sha256(path) for path in (parent_path, checkpoint, complete_path)}
    if original_hashes[str(checkpoint)] != complete['endpoint_checkpoint_sha256']:
        raise ValueError('Endpoint checkpoint SHA differs from completion receipt')
    torch, runtime_run, recovery = cold_import(code)
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require_final_state(saved, complete, args.policy)
    if (parent['binding'] != parent_binding or parent['arm'] != 'H1'
            or parent['r_updates'] != 13860 or parent['t_updates'] != 200
            or parent['progress']['phase'] != 25 or parent['progress']['update'] != 0):
        raise ValueError('Independent parent metadata is not H1 R20/T200')
    config = copy.deepcopy(saved['execution_binding']['config'])
    config.update(dataset=str(dataset), assets=str(assets))
    trainer, loaded_assets, data, actual_binding = runtime_run.build_context(
        config, 'cpu', saved['inner']['precision'], True)
    if (len(data), loaded_assets.n_items) != (709443, 10728):
        raise ValueError('Restored data/catalog counts differ')
    controller, proof = restore_original_state(trainer, parent, saved, actual_binding, recovery)
    if not controller.done or torch.cuda.is_initialized():
        raise ValueError('Expected a completed CPU-only restored controller')
    after_hashes = {path: sha256(path) for path in original_hashes}
    if after_hashes != original_hashes:
        raise ValueError('An input artifact changed during CPU restoration')
    # The runtime may import many files; verify the source tree again after use.
    verify_execution(code, parent_path)
    fingerprint = recovery.fingerprint
    frozen_proof = {key: fingerprint(saved['inner'][key]) == fingerprint(parent[key])
                    for key in ('alignment', 't_optimizer')}
    if args.policy == 'freeze' and not all(frozen_proof.values()):
        raise ValueError('Frozen endpoint modified tokenizer state')
    result = dict(format='stage09_h1_recovery_cpu_restore_audit_v1', status='passed',
        created_utc=datetime.now(timezone.utc).isoformat(), policy=args.policy,
        source_commit=SOURCE_COMMIT, execution_manifest_sha256=MANIFEST_SHA256,
        execution_file_count=manifest['execution_file_count'], archived_code=str(code),
        input_sha256=original_hashes, input_bytes_unchanged=True,
        local_dependency_paths=dict(dataset=str(dataset), assets=str(assets)),
        recorded_execution_binding=saved['execution_binding'], local_execution_binding=actual_binding,
        restoration=proof, r_updates=trainer.r_updates, t_updates=trainer.t_updates,
        phase=controller.progress['phase'], endpoint_epoch=26, r_scheduler_step=trainer.scheduler.last_epoch,
        r_learning_rate=trainer.r_optimizer.param_groups[0]['lr'],
        t_learning_rates={group['group_name']: group['lr'] for group in trainer.t_optimizer.param_groups},
        frozen_state_equal_to_parent=frozen_proof,
        cuda_initialized=False, cuda_exact_resume_verified=False,
        evidence_scope='Actual CPU model/optimizer/scheduler/state loading from the restored archive; '
                       'no optimizer update, no decoding, no GPU allocation, no CUDA RNG replay.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'run', 'parent', 'dataset', 'assets', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--policy', required=True, choices=('decay', 'freeze'))
    result = audit(parser.parse_args(argv))
    print(json.dumps({key: result[key] for key in ('status', 'policy', 'r_updates', 't_updates',
                                                  'phase', 'cuda_exact_resume_verified')}, indent=2))


if __name__ == '__main__':
    main()
