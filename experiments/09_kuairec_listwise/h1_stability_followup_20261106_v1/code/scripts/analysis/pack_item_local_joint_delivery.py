"""CPU byte delivery for the completed two-arm alternating joint campaign.

--mode pack --goal-root G --root-commit EXECUTION_COMMIT --output NEW_PACKAGE_DIR [--part-mib 1024]
--mode verify --package-dir PACKAGE_DIR --output NEW_EXTRACTION_DIR
--mode self-test

Weights/state, source snapshots and evidence are preserved byte-for-byte. The
13 input byte streams are verified at packing and identified, but data/cache
bytes are excluded. Relocated resume needs those inputs and path rebinding.
Byte verification proves neither research success nor GPU/resource release.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve()
HELPER = HERE.with_name('verify_item_local_archive.py')
if not HELPER.is_file():
    HELPER = next((parent / 'code/scripts/analysis/verify_item_local_archive.py' for parent in HERE.parents
                   if (parent / 'code/scripts/analysis/verify_item_local_archive.py').is_file()), HELPER)
sys.dont_write_bytecode = True
sys.path.insert(0, str(HELPER.parent))
from verify_item_local_archive import safe_name, archive_members, hash_file, hash_stream, parse_json, record_value, inside

CAMPAIGN = 'stage07_item_local_alternating_joint_seed2024_20261020_r1'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'
FORMAT = 'stage07_item_local_joint_delivery_v1'
EXTENSIONS = {'.json', '.csv', '.log', '.yaml', '.yml', '.txt', '.py', '.patch', '.pt', '.bin', '.npy', '.bak'}

def need(condition, message):
    if not condition: raise ValueError(message)

def read(path):
    return parse_json(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hash_file(Path(path))[0]

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')

def file_record(member, path):
    safe_name(member)
    need(path.is_file() and not any(parent.is_symlink() for parent in (path, *path.parents)), 'Nonregular/symlink source: ' + str(path))
    digest, size = hash_file(path)
    return {'member': member, 'sha256': digest, 'size_bytes': size}

def validate_run(run, variant, mode):
    name = 'smoke_results.json' if mode == 'smoke' else 'results.json'
    names = ('identity.json', 'study.json', 'recipe.json', 'input_identities.json', 'resolved_config.json', 'parsed_config.json',
             'code_identity.json', 'environment.json', 'initial_state_identity.json', 'phase_integrity.json', 'phase_integrity.csv',
             'training_integrity.json', 'FIT_COMPLETE.json', 'checkpoint_reload.json', 'evaluation_timings.json', 'epoch_timings.json',
             'COMPLETE.json', name, 'training_state/last.pt', 'training_state/best_model.bin', 'training_state/epoch_timings.json')
    for relative in names: file_record(relative, run / relative)
    complete, result, identity = read(run / 'COMPLETE.json'), read(run / name), read(run / 'identity.json')
    run_id = f'stage07_item_local_joint_{variant}_seed2024_20261020_r1' + ('_smoke' if mode == 'smoke' else '')
    status = 'smoke_passed_not_formal' if mode == 'smoke' else 'formal_completed'
    need(all(value['campaign_id'] == CAMPAIGN and value['variant'] == variant and value['run_id'] == run_id
             for value in (complete, result, identity)), 'Run/campaign/variant identity differs')
    best = run / 'training_state/best_model.bin'
    need(complete['mode'] == mode and complete['status'] == result['status'] == status and complete['result_file'] == name
         and complete['result_sha256'] == sha(run / name) and complete['checkpoint_sha256'] == result['checkpoint_sha256'] == sha(best), 'Incomplete result/checkpoint binding')
    need(identity['seed'] == 2024 and identity['engine_commit'] == ENGINE_COMMIT, 'Wrong seed/engine')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    state = torch.load(run / 'training_state/last.pt', map_location='cpu', weights_only=False)
    selected = torch.load(best, map_location='cpu', weights_only=True)
    need(state['format'] == 'diffgrm_joint_phase_boundary_v1' and state['progress']['training_complete'] is True
         and set(state['model']) == set(selected) and 'joint_alignment.hard_codes' in selected, 'Not a real terminal joint checkpoint')
    need(all(key in state for key in ('optimizer', 'scheduler', 'rng', 'scaler', 'joint_runtime', 'configuration_fingerprint'))
         and len(state['configuration_fingerprint']) == 64 and state['optimizer']['state'] and state['scheduler']
         and len(state['diagnostic_offsets']) == 5 and 'joint_progress' in state['progress'], 'Checkpoint lacks recovery state')
    for stored, offset in state['diagnostic_offsets'].items():
        path = Path(stored); need(path.resolve().is_relative_to(run.resolve()) and path.is_file() and path.stat().st_size >= offset, 'Committed diagnostic bytes are missing')
    need(all(key in state['rng'] for key in ('python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'loader_counters', 'loader_generators')), 'Incomplete RNG state')
    need(all(torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all()) for tensor in (*selected.values(), *state['model'].values())), 'Nonfinite/non-tensor checkpoint')
    need(state['best_checkpoint_sha256'] == hashlib.sha256(state['best_checkpoint_bytes']).hexdigest() == sha(best)
         and result['fit']['training_state_sha256'] == sha(run / 'training_state/last.pt'), 'Terminal state/best bytes differ')
    sid = hashlib.sha256(selected['joint_alignment.hard_codes'][1:].contiguous().numpy().tobytes()).hexdigest()
    reload = read(run / 'checkpoint_reload.json')
    need(sid == result['selected_SID_sha256'] == reload['matching_committed_SID_sha256'] and reload['strict_all_tensor_match'] is True
         and reload['checkpoint_sha256'] == sha(best), 'Selected checkpoint/SID reload evidence differs')
    if mode == 'formal':
        for suffix in ('_run_metadata.json', '_joint_semantic_ids.npy', '_joint_crossed_codebooks.npy'):
            need(any((run / 'run_metadata').rglob('*' + suffix)), 'Missing final joint metadata: ' + suffix)
        start, test = read(run / 'TEST_STARTED.json'), read(run / 'TEST_RESULT.json')
        need(result['test_data_evaluated'] is True and test['started_receipt_sha256'] == sha(run / 'TEST_STARTED.json')
             and test['checkpoint_sha256'] == start['checkpoint_sha256'] == sha(best)
             and test['selected_SID_sha256'] == sid and test['test_results'] == result['test_results'], 'Final test evidence missing/different')
        snapshots = test['diagnostic_snapshots']
        need(len(snapshots) == 5 and len({row['attribute'] for row in snapshots}) == 5, 'Missing once-only test diagnostic snapshots')
        for row in snapshots:
            snapshot = inside(run, safe_name(row['snapshot_relative_path']))
            original = inside(run, safe_name(row['original_relative_path']))
            need(snapshot.stat().st_size == row['size_bytes'] and sha(snapshot) == row['sha256'] == sha(original), 'Test diagnostic snapshot damaged')
    else:
        recovery = read(run / 'recovery_probe/recovery_check.json')
        need(result['test_data_evaluated'] is False and not (run / 'TEST_STARTED.json').exists() and recovery == result['recovery_check']
             and recovery['status'] == 'passed' and recovery['strict_model_optimizer_scheduler_RNG_joint_runtime_restore'] is True, 'Smoke recovery evidence missing')
        hashes = recovery['original_checkpoint_hashes_unchanged']
        need(hashes == {'last': sha(run / 'training_state/last.pt'), 'best': sha(best), 'source': sha(run / 'smoke_tokenizer_boundary.pt')}
             and sha(run / 'recovery_probe/source_last.pt') == hashes['source'], 'Smoke recovery source bytes differ')
        file_record('best_model.bin', run / 'recovery_probe/best_model.bin')
        source = torch.load(run / 'recovery_probe/source_last.pt', map_location='cpu', weights_only=False)
        need(sha(run / 'recovery_probe/best_model.bin') == source['best_checkpoint_sha256']
             == hashlib.sha256(source['best_checkpoint_bytes']).hexdigest(), 'Private recovery embedded best bytes differ')
    return result

def archive_files(sources, output, metadata, part_bytes):
    files = [file_record(member, path) for member, path in sorted(sources.items())]
    manifest = {'format': FORMAT, 'campaign_id': CAMPAIGN, 'files': files, **metadata}
    archive = output / 'joint_delivery.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', dereference=True) as tar:
            for record in files:
                path = sources[record['member']]; tar.add(path, arcname=record['member'], recursive=False)
                need(file_record(record['member'], path) == record, 'Source changed while packaging: ' + str(path))
            data = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
            info = tarfile.TarInfo('MANIFEST.json'); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    digest, size = hash_file(archive)
    parts = []
    if size > part_bytes:
        with archive.open('rb') as handle:
            while block := handle.read(part_bytes):
                part = output / f'joint_delivery.tar.gz.part{len(parts) + 1:04d}'
                with part.open('xb') as target: target.write(block)
                parts.append(file_record(part.name, part))
        archive.unlink()  # Only the collector's own redundant archive in its new output directory.
    else: parts.append(file_record(archive.name, archive))
    receipt = {'format': FORMAT, 'status': 'PACKAGED_BYTES_NOT_RESEARCH_AUDIT', 'campaign_id': CAMPAIGN,
               'archive_sha256': digest, 'archive_size_bytes': size, 'parts': parts, 'files': len(files),
               'manifest_sha256': hashlib.sha256(data).hexdigest(), 'goal_complete': False}
    write(output / 'PACKAGE.json', receipt)
    return receipt

def pack(goal, output, part_bytes, root_commit):
    need(not os.path.lexists(output), 'Choose a new package directory')
    goal, output = goal.resolve(), output.resolve()
    need(not any(output.is_relative_to(goal / name) for name in ('code', 'engine', 'runs', 'inputs', 'delivery')), 'Keep package outside source/input/run/delivery trees')
    need(re.fullmatch('[0-9a-f]{40}', root_commit or ''), 'Require full lowercase --root-commit of the training execution')
    api = runpy.run_path(str(goal / 'code/scripts/train/item_local_joint_gpu_job.py'))
    binding = api['source_binding'](goal, root_commit, ENGINE_COMMIT); rows = api['study_rows'](goal)
    receipts = [(path, read(path)) for path in (goal / 'delivery').rglob('receipt.json')]
    successful = [(path, value) for path, value in receipts if value.get('schema') == 'stage07_item_local_joint_job_phase_v1'
                  and value.get('campaign_id') == CAMPAIGN and value.get('status') == value.get('mode', '') + '_completed']
    smoke = [(path, value) for path, value in successful if value['mode'] == 'smoke' and value.get('source_binding') == binding]
    need(len(smoke) == 1, 'Need one successful same-source paired smoke wrapper receipt')
    smoke_gate = api['smoke_gate'](smoke[0][0], rows, binding)
    for row in rows:
        matches = [value for _, value in successful if value['mode'] == 'formal' and value.get('variant') == row['variant'] and value.get('source_binding') == binding]
        need(matches, 'Missing successful same-source formal wrapper receipt: ' + row['variant'])
        _, result, hashes = api['completion'](row, 'formal', binding)
        for value in matches:
            need(value['smoke_evidence'] == smoke_gate and len(value['children']) == 1, 'Formal wrapper smoke/child binding differs')
            child = value['children'][0]
            need(child['variant'] == row['variant'] and child['returncode'] == 0 and child['status'] in ('completed', 'skipped_complete')
                 and all(child[key] == digest for key, digest in hashes.items()), 'Formal child did not complete or bound bytes changed')
    sources, results, input_receipts = {}, {}, {}
    def add(path, member):
        need(member not in sources, 'Duplicate package member'); file_record(member, path); sources[member] = path
    def collect(work, prefix):
        for path in work.iterdir():
            if path.is_file() and path.suffix in EXTENSIONS: add(path, prefix + '/' + path.name)
        for name in ('attempts', 'diagnostics', 'run_metadata', 'logs', 'training_state', 'post_test_diagnostics', 'recovery_probe'):
            for path in (work / name).rglob('*'):
                if path.is_file() and path.suffix in EXTENSIONS: add(path, prefix + '/' + path.relative_to(work).as_posix())
    for variant in ('control', 'item_local'):
        run = goal / 'runs' / f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'
        results[variant] = validate_run(run, variant, 'formal')
        validate_run(run / 'smoke', variant, 'smoke')
        for work in (run, run / 'smoke'):
            relative = work.relative_to(goal).as_posix(); collect(work, relative)
            identity = read(work / 'input_identities.json'); rows = identity['artifacts']
            recipe = read(work / 'recipe.json'); lock = read(goal / 'code/experiments/05_crosspse_joint/reproducibility/assets.lock.json')['artifacts']
            need(len(rows) == 13 and {row['path'] for row in rows} == set(recipe['required_artifacts']), 'Expected thirteen exact input identities')
            for row in rows:
                path = inside(work / 'workspace', safe_name(row['path'])); observed = file_record(row['path'], path)
                need(observed['sha256'] == row['sha256'] == lock[row['path']]['sha256']
                     and observed['size_bytes'] == row['size_bytes'] == lock[row['path']]['size_bytes'], 'Runtime input bytes differ from identity/lock')
            input_receipts[relative] = {'status': 'verified_runtime_input_bytes', 'files': rows, 'data_bytes_included': False}
    for field in ('initial_base_state_sha256', 'initial_rng_sha256'):
        need(results['control'][field] == results['item_local'][field], 'Paired initial identity differs')
    for path in (goal / 'delivery').rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.csv', '.log', '.yaml', '.yml', '.txt', '.py', '.patch'}:
            add(path, path.relative_to(goal).as_posix())
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='source_snapshots_', dir=output) as temporary:
        temporary = Path(temporary); commits = {}
        for name, commit in (('code', root_commit), ('engine', ENGINE_COMMIT)):
            repository = goal / name
            actual = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip()
            need(actual == commit and not dirty, 'Source checkout is not the exact clean execution commit')
            snapshot = temporary / (name + '.tar')
            subprocess.run(['git', '-C', str(repository), 'archive', '--format=tar', '--output', str(snapshot), commit], check=True)
            add(snapshot, 'source/' + name + '.tar'); commits[name] = {'commit': commit, 'archive_sha256': sha(snapshot)}
        inputs = temporary / 'input_byte_verification.json'; write(inputs, input_receipts); add(inputs, 'input_byte_verification.json')
        add(HERE, 'delivery_tools/pack_item_local_joint_delivery.py'); add(HELPER, 'delivery_tools/verify_item_local_archive.py')
        result = archive_files(sources, output, {'source_identity': commits, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'tool_identity': {'collector_sha256': sha(HERE), 'byte_helper_sha256': sha(HELPER), 'training_source_binding': binding},
                    'scope': 'Artifacts/source bytes only; thirteen input streams remain external; no GPU-release or research-success claim'}, part_bytes)
    return result

def verify(package, output):
    need(not os.path.lexists(output), 'Extraction destination must be new')
    package, output = package.resolve(), output.resolve(); receipt = read(package / 'PACKAGE.json')
    need(receipt['format'] == FORMAT and not output.exists() and not output.is_symlink(), 'Wrong package or existing extraction destination')
    need(receipt['parts'] and len({row['member'].casefold() for row in receipt['parts']}) == len(receipt['parts']), 'Missing/duplicate volumes')
    with tempfile.TemporaryFile() as raw:
        for row in receipt['parts']:
            safe_name(row['member']); need('/' not in row['member'], 'Volume must be a direct package file')
            part = package / row['member']; need(file_record(row['member'], part) == record_value(row), 'Volume bytes/SHA differ')
            with part.open('rb') as handle: shutil.copyfileobj(handle, raw, 1024 * 1024)
        raw.seek(0); need(hash_stream(raw) == (receipt['archive_sha256'], receipt['archive_size_bytes']), 'Whole archive SHA/size differs')
        observed, names, manifest = {}, set(), None
        for member, stream in archive_members(raw):
            key = member.name.casefold(); need(key not in names and manifest is None, 'Duplicate/colliding member or data after manifest'); names.add(key)
            if member.name == 'MANIFEST.json':
                need(member.size <= 16 * 1024 * 1024, 'Manifest too large'); data = stream.read(); manifest = parse_json(data.decode('utf-8'))
                need(hashlib.sha256(data).hexdigest() == receipt['manifest_sha256'], 'Manifest bytes differ')
            else:
                digest, size = hash_stream(stream); need(size == member.size, 'Truncated tar member')
                observed[member.name] = {'member': member.name, 'sha256': digest, 'size_bytes': size}
        need(manifest and manifest['format'] == FORMAT and manifest['campaign_id'] == receipt['campaign_id'] == CAMPAIGN, 'Missing/wrong manifest identity')
        declared = [record_value(row) for row in manifest['files']]
        need(len(declared) == len(observed) == receipt['files'] and {row['member']: row for row in declared} == observed
             and 'BYTE_VERIFICATION.json' not in observed, 'Manifest does not cover exact archive bytes or reserves verification output')
        need(all('/'.join(name.split('/')[:index]) not in names for name in names for index in range(1, len(name.split('/')))), 'File/directory prefix collision')
        output.mkdir(parents=True, exist_ok=False)
        extracted = set()
        for member, stream in archive_members(raw):
            need(member.name not in extracted and (member.name in observed or member.name == 'MANIFEST.json'), 'Second-pass archive membership changed'); extracted.add(member.name)
            target = inside(output, member.name); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle: shutil.copyfileobj(stream, handle, 1024 * 1024)
            expected = observed.get(member.name, {'member': 'MANIFEST.json', 'sha256': receipt['manifest_sha256'], 'size_bytes': len(data)})
            need(file_record(member.name, target) == expected, 'Extracted bytes differ')
        need(extracted == set(observed) | {'MANIFEST.json'}, 'Extracted file set differs')
        raw.seek(0); need(hash_stream(raw) == (receipt['archive_sha256'], receipt['archive_size_bytes']), 'Archive changed while extracting')
        result = {'status': 'BYTE_RESTORATION_VERIFIED', 'archive_sha256': receipt['archive_sha256'], 'files': len(observed),
                  'research_audit_performed': False, 'goal_complete': False}
        write(output / 'BYTE_VERIFICATION.json', result); return result

def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); source = root / 'bytes.bin'; source.write_bytes(bytes(range(256)) * 37)
        package = root / 'package'; package.mkdir(); archive_files({'sample/bytes.bin': source}, package, {'scope': 'CPU byte roundtrip fixture only'}, 137)
        verify(package, root / 'restored'); need((root / 'restored/sample/bytes.bin').read_bytes() == source.read_bytes(), 'Roundtrip failed')
        try: validate_run(root, 'control', 'formal')
        except (FileNotFoundError, ValueError): pass
        else: raise AssertionError('Incomplete formal run accepted')
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\b'):
            try: safe_name(name)
            except ValueError: continue
            raise AssertionError('Unsafe name accepted')
    return {'status': 'CPU_SELF_TEST_PASSED', 'checks': ['split_volume_byte_roundtrip', 'incomplete_formal_rejected', 'unsafe_tar_names_rejected'], 'GPU_or_platform_used': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('pack', 'verify', 'self-test'), required=True)
    parser.add_argument('--goal-root', type=Path); parser.add_argument('--package-dir', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--root-commit', help='Full training execution commit; delivery tool hashes are recorded separately')
    parser.add_argument('--part-mib', type=int, default=1024)
    args = parser.parse_args(); need(args.part_mib > 0, 'Volume size must be positive')
    if args.mode == 'self-test': result = self_test()
    elif args.mode == 'pack': need(args.goal_root and args.output, 'pack requires --goal-root/--output'); result = pack(args.goal_root, args.output, args.part_mib * 1024 * 1024, args.root_commit)
    else: need(args.package_dir and args.output, 'verify requires --package-dir/--output'); result = verify(args.package_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
