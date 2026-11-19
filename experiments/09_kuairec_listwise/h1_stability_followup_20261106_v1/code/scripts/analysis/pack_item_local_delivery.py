"""CPU-only, hash-bound delivery of a completed Stage07 run.

Call only after the owning GPU job is terminal and its allocation is released.
The package contains the selected model, full last training state, curves,
configuration/provenance, three light input files, and exact source snapshots.
The other historical inputs are verified remotely without copying them again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time


REPAIR_CAMPAIGN = 'stage07_ocn_repair_seed2024_20261020_r1'
REPAIR_ENGINE = '8a48a300daa6621d6c705966ff73fa837ef8f12e'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def repair_identity(identity, run_id, engine_commit):
    campaign = identity.get('repair_campaign')
    if campaign is None:
        if run_id.startswith('stage07_ocn_repair_') or identity.get('variant') is not None or identity.get('repaired_engine_commit') is not None:
            raise ValueError('Repair artifacts cannot bypass their exact campaign identity')
        return None
    variant = identity.get('variant')
    if (campaign != REPAIR_CAMPAIGN or variant not in ('control', 'item_local')
            or run_id != f'stage07_ocn_repair_{variant}_seed2024_20261020_r1'
            or identity.get('run_id') != run_id
            or identity.get('repaired_engine_commit') != REPAIR_ENGINE
            or engine_commit != REPAIR_ENGINE):
        raise ValueError('Repaired delivery campaign/variant/run/engine identity differs')
    return {'repair_campaign': campaign, 'variant': variant, 'repaired_engine_commit': REPAIR_ENGINE}


def repair_evidence(run, identity, complete, binding):
    """Bind repair evidence bytes; model/metric acceptance stays independent."""
    profile = run / 'profile'
    required = ['training_integrity.json', 'initial_state_identity.json', 'study.json', 'code_identity.json',
                'profile/identity.json', 'profile/study.json', 'profile/code_identity.json',
                'profile/COMPLETE.json', 'profile/profile_results.json', 'profile/checkpoint_reload.json',
                'profile/training_integrity.json', 'profile/initial_state_identity.json',
                'profile/training_state/last.pt', 'profile/training_state/best_model.bin',
                'profile/recovery_probe/recovery_check.json', 'profile/recovery_probe/source_last.pt',
                'profile/recovery_probe/best_model.bin']
    for name in required:
        path = run / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'Missing/nonregular repair evidence: {path}')
    for work, mode, expected_id, result_name in (
            (run, 'formal', identity['run_id'], 'results.json'),
            (profile, 'profile', identity['run_id'] + '_profile', 'profile_results.json')):
        own_identity = identity if mode == 'formal' else read_json(work / 'identity.json')
        if (own_identity.get('run_id') != expected_id or own_identity.get('repair_campaign') != REPAIR_CAMPAIGN
                or own_identity.get('variant') != binding['variant']
                or own_identity.get('repaired_engine_commit') != REPAIR_ENGINE
                or own_identity.get('study_sha256') != identity['study_sha256']
                or own_identity.get('sid_sha256') != identity['sid_sha256']):
            raise ValueError('Formal/profile identities do not belong to the same repaired study and SID')
        own_complete = complete if mode == 'formal' else read_json(work / 'COMPLETE.json')
        if (own_complete.get('mode') != mode or own_complete.get('status') != mode + '_completed'
                or own_complete.get('repair_campaign') != REPAIR_CAMPAIGN or own_complete.get('variant') != binding['variant']
                or own_complete.get('result_file') != result_name
                or own_complete.get('result_sha256') != sha256(work / result_name)):
            raise ValueError('Repair COMPLETE/result bytes are not bound to the expected campaign and mode')
        result = read_json(work / result_name)
        if (result.get('repair_campaign') != REPAIR_CAMPAIGN or result.get('variant') != binding['variant']
                or result.get('status') != mode + '_completed'
                or result.get('checkpoint_sha256') != own_complete['checkpoint_sha256']
                or sha256(work / 'training_state/best_model.bin') != own_complete['checkpoint_sha256']):
            raise ValueError('Repair result/checkpoint identity binding differs')
        study, code = read_json(work / 'study.json'), read_json(work / 'code_identity.json')
        if (study.get('repair_campaign') != REPAIR_CAMPAIGN or study.get('variant') != binding['variant']
                or study.get('run_id') != identity['run_id'] or study.get('seed') != 2024
                or code.get('engine', {}).get('head') != REPAIR_ENGINE):
            raise ValueError('Repair study or recorded engine differs from the exact campaign')
        initial, integrity = read_json(work / 'initial_state_identity.json'), read_json(work / 'training_integrity.json')
        count = 84 if binding['variant'] == 'control' else 90
        if (initial.get('repair_campaign') != REPAIR_CAMPAIGN or initial.get('variant') != binding['variant']
                or any(initial.get(key) != result.get(key) for key in ('initial_base_state_sha256', 'initial_rng_sha256'))
                or integrity != result.get('training_integrity') or integrity.get('passed') is not True
                or integrity.get('all_gradient_coverage') is not True or integrity.get('all_optimizer_steps_match') is not True
                or integrity.get('all_parameter_tensors') != count or integrity.get('optimizer_state_count') != count):
            raise ValueError('Repair initialization or gradient/optimizer evidence is missing or inconsistent')
    profile_result = read_json(profile / 'profile_results.json')
    recovery = read_json(profile / 'recovery_probe/recovery_check.json')
    recorded_recovery = {key: value for key, value in profile_result['recovery_check'].items() if key != 'wall_seconds'}
    if (recovery != recorded_recovery or recovery.get('status') != 'passed'
            or recovery.get('strict_model_optimizer_scheduler_RNG_restore') is not True
            or recovery.get('all_gradient_coverage') is not True or recovery.get('optimizer_updates_executed') != 1
            or recovery.get('updates_before') != 104 or recovery.get('updates_after') != 105
            or profile_result.get('test_data_evaluated') is not False):
        raise ValueError('Repair profile actual-recovery evidence is missing or differs from its result')
    hashes = recovery['original_checkpoint_hashes_unchanged']
    if (sha256(profile / 'training_state/last.pt') != hashes['last']
            or sha256(profile / 'recovery_probe/source_last.pt') != hashes['last']
            or sha256(profile / 'training_state/best_model.bin') != hashes['best']
            or sha256(profile / 'recovery_probe/best_model.bin') != hashes['best']):
        raise ValueError('Repair profile/recovery checkpoint bytes no longer agree')


def pack(args):
    goal = args.goal_root.resolve()
    run = goal / 'runs' / args.run_id
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError('Preserve existing delivery; use a new explicit filename')
    complete = read_json(run / 'COMPLETE.json')
    if complete.get('mode') != 'formal' or complete.get('status') != 'formal_completed':
        raise ValueError('A smoke or incomplete run cannot be packaged as formal delivery')
    required = ['FIT_COMPLETE.json', 'TEST_STARTED.json', 'TEST_RESULT.json',
                'results.json', 'checkpoint_reload.json', 'parsed_config.json',
                'input_identities.json', 'training_state/best_model.bin',
                'training_state/last.pt']
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    identity = read_json(run / 'identity.json')
    if identity['run_id'] != args.run_id:
        raise ValueError('Requested run identity differs from completed artifacts')
    repair = repair_identity(identity, args.run_id, args.engine_commit)
    if repair:
        repair_evidence(run, identity, complete, repair)
    elif complete.get('repair_campaign') is not None or complete.get('variant') is not None:
        raise ValueError('Repair completion cannot be packaged under a legacy identity')
    best = run / 'training_state/best_model.bin'
    if sha256(best) != complete['checkpoint_sha256']:
        raise ValueError('Completed checkpoint bytes changed before packaging')

    inputs = read_json(run / 'input_identities.json')
    workspace = Path(inputs['runtime_workspace']).resolve()
    if workspace != (run / 'workspace').resolve():
        raise ValueError('Unexpected runtime input workspace')
    input_records = []
    for record in inputs['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe input identity path')
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing/nonregular bound input: {path}')
        observed = sha256(path)
        if observed != record['sha256'] or path.stat().st_size != record['size_bytes']:
            raise ValueError(f'Runtime input bytes no longer match lock: {path}')
        input_records.append({'path': relative.as_posix(), 'sha256': observed,
                              'expected_sha256': record['sha256'],
                              'size_bytes': path.stat().st_size,
                              'runtime_absolute_path': str(path)})
    if len(input_records) != 12:
        raise ValueError('The fixed PSE recipe must retain all twelve bound input identities')
    input_receipt = {
        'status': 'passed', 'run_id': args.run_id,
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'verification_method': 'sha256_of_runtime_workspace_bytes',
        'files': input_records,
        'scope': 'Current runtime bytes match preserved migration assets; original historical runtime hashes were not recorded',
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = {}

    def add_source(member, path):
        path = Path(path)
        if member in sources or not path.is_file() or path.is_symlink():
            raise ValueError(f'Duplicate or nonregular package member: {member}')
        sources[member] = path

    for path in run.glob('*.json'):
        add_source('run/' + path.name, path)
    directories = ['attempts', 'diagnostics', 'run_metadata', 'logs', 'training_state']
    if repair:
        directories.append('profile')
    for directory in directories:
        for path in (run / directory).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.csv', '.log', '.pt', '.bin', '.bak'}:
                add_source('run/' + path.relative_to(run).as_posix(), path)
    light_inputs = [record for record in input_records
                    if record['path'].endswith(('.sem_ids', '/id_mapping.json', '/all_item_seqs.json'))]
    if len(light_inputs) != 3:
        raise ValueError('Expected exactly SID, item mapping, and user sequences as light inputs')
    for record in light_inputs:
        add_source('run/workspace/' + record['path'], workspace / record['path'])
    # These are only this Goal's private delivery files, never account config.
    delivery = goal / 'delivery'
    delivery_files = delivery.rglob('*') if repair else delivery.iterdir()
    delivery_suffixes = {'.json', '.log', '.py', '.bundle'} | ({'.csv', '.pt', '.bin', '.bak'} if repair else set())
    for path in delivery_files:
        if path.is_file() and path.suffix in delivery_suffixes:
            add_source('platform_and_preparation/' + path.relative_to(delivery).as_posix(), path)

    with tempfile.TemporaryDirectory(prefix='stage07_delivery_', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        if temporary.resolve().parent != destination.parent:
            raise ValueError('Temporary sources must remain inside the delivery directory')
        source_identity = {}
        for label, directory, expected in [('root', goal / 'code', args.root_commit),
                                            ('engine', goal / 'engine', args.engine_commit)]:
            actual = git(directory, 'rev-parse', 'HEAD').decode().strip()
            if actual != expected:
                raise ValueError(f'{label} source no longer matches the frozen execution version')
            if git(directory, 'status', '--porcelain').strip():
                raise ValueError(f'{label} execution checkout contains unrecorded changes')
            archive = temporary / (label + '_' + expected[:12] + '.tar')
            subprocess.run(['git', '-C', str(directory), 'archive', '--format=tar',
                            '--output', str(archive), expected], check=True)
            add_source('source/' + archive.name, archive)
            source_identity[label] = {'commit': expected, 'tar_sha256': sha256(archive)}
        baseline_source = goal / 'code/reproducibility/code/DiffGRM_39e94e6153c9.tar'
        add_source('source/baseline_engine_39e94e6153c9.tar', baseline_source)
        difference = git(goal / 'engine', 'diff', '--binary',
                         '39e94e6153c97e59877ba02d4ef1d9a202ed5384', args.engine_commit)
        patch = temporary / 'engine_changes.patch'
        patch.write_bytes(difference)
        add_source('source/engine_changes.patch', patch)
        # A copy of this exact CPU collector binds the post-training archive.
        add_source('source/pack_item_local_delivery.py', Path(__file__).resolve())

        virtual = {'run/input_hash_receipt.json': input_receipt,
                   'source/source_identity.json': source_identity}
        duplicates = set(virtual).intersection(sources)
        if duplicates:
            raise ValueError(f'Generated receipts would duplicate existing members: {sorted(duplicates)}')
        manifest = {'format': 'stage07_delivery_v1', 'run_id': args.run_id,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'scope': 'Artifact bytes only; final GPU release/accounting is audited separately',
                    'files': []}
        if repair:
            manifest.update(repair)
        partial = destination.with_name(destination.name + f'.partial_{time.time_ns()}')
        with partial.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as archive:
                    for member, path in sorted(sources.items()):
                        digest = sha256(path)
                        size = path.stat().st_size
                        archive.add(path, arcname=member, recursive=False)
                        if sha256(path) != digest:
                            raise RuntimeError(f'Source changed while packaging: {path}')
                        manifest['files'].append({'member': member, 'sha256': digest, 'size_bytes': size})
                    for member, value in sorted(virtual.items()):
                        data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                        info = tarfile.TarInfo(member)
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
                        manifest['files'].append({'member': member, 'sha256': hashlib.sha256(data).hexdigest(),
                                                  'size_bytes': len(data)})
                    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                    info = tarfile.TarInfo('MANIFEST.json')
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        partial.replace(destination)
    receipt = {'status': 'PACKAGED_NOT_YET_LOCAL_VERIFIED', 'run_id': args.run_id,
               'archive': str(destination), 'sha256': sha256(destination),
               'size_bytes': destination.stat().st_size, 'members': len(manifest['files']) + 1,
               'runtime_input_files_verified': len(input_records), 'local_light_input_files': 3,
               'checkpoint_sha256': complete['checkpoint_sha256']}
    if repair:
        receipt.update(repair)
    sidecar = destination.with_name(destination.name + '.receipt.json')
    with sidecar.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--run-id', default='stage07_item_local_seed2024_20261020_0045')
    parser.add_argument('--root-commit', default='8b7b9afe17ef425efe9279524ecf7a1e97148853')
    parser.add_argument('--engine-commit', default='1dc2a0c5ad01e2a2878b9de400b91f3a7e809f25')
    parser.add_argument('--output', type=Path, required=True)
    pack(parser.parse_args())


if __name__ == '__main__':
    main()
