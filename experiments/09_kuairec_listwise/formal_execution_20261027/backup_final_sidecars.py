"""Owned Stage09 operations/evaluations sidecars only; no network/model/GPU work.

Requires unchanged adjacent backup_completed_run.py. CPU pack is Linux-only.
Native terminal/release evidence must be collected externally; this helper does
not manufacture it, query the platform, or establish freshness of an old receipt.
Partial outputs remain on failure. No overwrite, recursive delivery scan, or tar
inside tar. Restore is into a new local sidecars/<scope> directory.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tarfile

BASE_SHA = '0800f6dd99b573b8780ebe8b1eea3ca64fe4713d527475e620a6b943fb65f403'
base_path = Path(__file__).with_name('backup_completed_run.py')
if base_path.is_symlink() or hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError('Adjacent audited backup helper missing or changed')
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('stage09_backup_base', base_path)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
need = b.require
SCHEMA = 'stage09_final_sidecars_v1'
ARMS = ('J0', 'J1', 'H0', 'H1')
ROOT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
SUP = 'delivery/four_arm_supervisor_r1/'
EVIDENCE = 'delivery/backup_evidence/'
TRAIN_JOB = 'nancy-stage09-four-arm-20261027-r1'
CONFIG_SHA = 'd8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb'
SUBSET = 'code/experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json'
SUBSET_SHA = 'c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af'
OPERATIONS = ('config/efficiency_r2.json', SUBSET,
    'delivery/FORMAL_WARMUP_BINDING.json', 'delivery/FORMAL_BRANCH_BINDING.json',
    'delivery/formal_warmup_precheck.py', 'delivery/formal_branch_acceptance.py',
    'delivery/formal_four_arm_job.py', 'delivery/backup_completed_run.py', 'delivery/backup_final_sidecars.py',
    'delivery/INPUT_TRANSFER_MANIFEST.json', 'delivery/SOURCE_LOCK.json',
    'logs/warmup_j_20261027_r1.log', 'logs/warmup_h_20261027_r1.log',
    'logs/four_arm_training_20261027_r1.log')
SUP_REQUIRED = ('started.json', 'supervisor.json', 'gpu_inventory.json', 'pair_J.json', 'pair_H.json') + tuple(
    name for arm in ARMS for name in ('ready_' + arm + '.json', 'train_' + arm + '.log'))
SUP_ALLOWED = set(SUP_REQUIRED) | {name + '.stderr' for name in SUP_REQUIRED}
VARIANTS = ('test_beam32', 'validation_subset_beam32', 'validation_subset_beam256')
EVAL_FILES = tuple('evaluations/' + arm + '_' + variant + '/' + name
    for arm in ARMS for variant in VARIANTS for name in ('summary.json', 'per_sample.jsonl.gz'))
EVAL_LOGS = tuple('logs/final_eval_' + arm.lower() + '_20261027_r1.log' for arm in ARMS)
EVAL_HELPERS = ('delivery/formal_final_acceptance.py', 'delivery/backup_final_sidecars.py',
                'delivery/backup_completed_run.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def index_name(scope):
    need(scope in ('operations', 'evaluations'), 'Unknown sidecar scope')
    return EVIDENCE + scope + '_terminal.json'


def output_names(scope):
    index_name(scope)
    return 'delivery/stage09_' + scope + '_r1.tar', 'delivery/STAGE09_' + scope.upper() + '_R1.json'


def evidence_path(name):
    b.parts(name)
    return bool(re.fullmatch(re.escape(EVIDENCE) + r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}\.json', name))


def allowed(name, scope):
    b.parts(name)
    if evidence_path(name) or name in ('runs/' + a + '/complete.json' for a in ARMS):
        return True
    if scope == 'operations':
        return name in OPERATIONS or (name.startswith(SUP) and name[len(SUP):] in SUP_ALLOWED)
    return name in EVAL_FILES + EVAL_LOGS + EVAL_HELPERS or name in (SUBSET, 'config/efficiency_r2.json') or name == SUP + 'supervisor.json'


def read_json(name, expected_sha=None):
    with os.fdopen(b.open_remote(name), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= 8 * 1024 * 1024, 'JSON evidence exceeds 8 MiB: ' + name)
        raw = stream.read(8 * 1024 * 1024 + 1)
        need(b.identity(before) == b.identity(os.fstat(stream.fileno())), 'JSON changed while reading')
    if expected_sha is not None:
        need(bool(re.fullmatch('[0-9a-f]{64}', expected_sha)) and hashlib.sha256(raw).hexdigest() == expected_sha,
             'Evidence SHA mismatch: ' + name)
    return json.loads(raw)


def check_native_job(status, events, name, audit):
    # These two native shapes are present in recorded warmup status receipts.
    need(status.get('success') is True and isinstance(status.get('data'), dict), 'Invalid native status receipt')
    data = status['data']
    rows = data.get('items', [data])
    selected = [row for row in rows if row.get('name') == name]
    need(len(selected) == 1 and selected[0].get('status') == 'SUCCEEDED', 'Job is not natively SUCCEEDED: ' + name)
    need(events.get('success') is True and isinstance(events.get('data'), dict), 'Invalid native events receipt')
    deleted = [row for row in events['data'].get('items', []) if row.get('reason') == 'SuccessfulDeletePod']
    need(deleted and all(row.get('time') for row in deleted), 'No native SuccessfulDeletePod release evidence')
    # Native event exports lack a job name; a human/outer verifier must bind the
    # scoped query to its exact job. This assertion alone is NOT platform proof.
    need(audit.get('job_name') == name and audit.get('all_allocated_pods_released') is True
         and audit.get('native_events_query_was_scoped_to_this_job') is True
         and isinstance(audit.get('checked_at_utc'), str) and audit.get('checked_at_utc')
         and isinstance(audit.get('auditor'), str) and audit.get('auditor'), 'Missing explicit external release audit')


def terminal_guard(scope, index_sha):
    name = index_name(scope)
    index = read_json(name, index_sha)
    need(index.get('schema') == 'stage09_sidecar_terminal_inputs_v1' and index.get('root') == ROOT
         and index.get('scope') == scope, 'Wrong terminal index identity')
    expected = [TRAIN_JOB] + (['nancy-stage09-final-eval-' + a.lower() + '-20261027-r1' for a in ARMS]
                             if scope == 'evaluations' else [])
    jobs = index.get('jobs', [])
    need(sorted(row.get('name', '') for row in jobs) == sorted(expected), 'Terminal job inventory differs')
    files = [name]
    for row in jobs:
        contents = {}
        for key in ('status_receipt', 'events_receipt'):
            ref = row[key]
            need(evidence_path(ref['path']) and ref['path'] != name, 'Native evidence must be a plain JSON in backup_evidence')
            contents[key] = read_json(ref['path'], ref['sha256'])
            files.append(ref['path'])
        check_native_job(contents['status_receipt'], contents['events_receipt'], row['name'], row['release_audit'])
    # Copies of actual submitted argv/config, SHA checks or final acceptance
    # receipts can be included, never arbitrary paths or large binary payloads.
    for ref in index.get('additional_json', []):
        need(evidence_path(ref['path']) and ref['path'] != name, 'Extra provenance must be a plain JSON in backup_evidence')
        read_json(ref['path'], ref['sha256'])
        files.append(ref['path'])
    need(len(files) == len(set(files)), 'Duplicate terminal evidence paths')
    return index, files


def training_guard():
    complete = {}
    for arm in ARMS:
        row = b.completion_guard(arm)
        need(row.get('r_updates') == 24255 and row.get('t_updates') == 400, 'Incomplete formal update counts: ' + arm)
        complete[arm] = row
    supervisor = read_json(SUP + 'supervisor.json')
    need(supervisor.get('schema') == 'stage09_four_arm_job_v1'
         and supervisor.get('status') == 'training_processes_completed'
         and not supervisor.get('cleanup_signals') and not supervisor.get('owned_groups_alive_after_cleanup')
         and all(supervisor.get('process_exits', {}).get(a) == 0 for a in ARMS), 'Formal wrapper did not exit cleanly')
    return complete


def evaluate_guard(index):
    ref = index['four_best_freeze']
    need(evidence_path(ref['path']), 'Freeze copy must be a plain JSON in backup_evidence')
    freeze = read_json(ref['path'], ref['sha256'])
    need(freeze.get('schema') == 'stage09_four_arm_final_selection_v1' and freeze.get('status') == 'passed'
         and freeze.get('root') == ROOT and freeze.get('test_used_for_selection') is False
         and set(freeze.get('four_best_sha256', {})) == set(ARMS), 'Invalid four-best freeze')
    subset_ids = {row['sample_id'] for row in read_json(SUBSET)['rows']}
    for arm in ARMS:
        row, _ = b.hash_remote('runs/' + arm + '/best.pt')
        need(row['sha256'] == freeze['four_best_sha256'][arm]
             and row['bytes'] == freeze['four_best_bytes'][arm], 'Frozen best changed: ' + arm)
        for variant in VARIANTS:
            prefix = 'evaluations/' + arm + '_' + variant + '/'
            summary = read_json(prefix + 'summary.json')
            full = variant == 'test_beam32'
            expected_count = 53194 if full else 1024
            need(summary.get('split') == ('test' if full else 'validation')
                 and summary.get('examples') == expected_count and summary.get('full_split') is full
                 and summary.get('beam') == (256 if variant.endswith('256') else 32)
                 and summary.get('architecture') == arm[0] and summary.get('seed') == 2024
                 and summary.get('inference_precision') == 'fp32'
                 and summary.get('diagnostic_prefix_limit') is None, 'Incomplete/wrong evaluation summary: ' + prefix)
            need(summary.get('subset') is None if full else
                 summary.get('subset', {}).get('manifest_sha256') == SUBSET_SHA,
                 'Evaluation subset differs: ' + prefix)
            count, ids = 0, set()
            with os.fdopen(b.open_remote(prefix + 'per_sample.jsonl.gz'), 'rb') as source:
                with gzip.GzipFile(fileobj=source, mode='rb') as stream:
                    while True:
                        line = stream.readline(b.CHUNK + 1)
                        if not line:
                            break
                        need(len(line) <= b.CHUNK and line.endswith(b'\n'), 'Oversized/partial prediction row')
                        row = json.loads(line)
                        need('sample_id' in row and row['sample_id'] not in ids, 'Duplicate/missing sample ID')
                        ids.add(row['sample_id'])
                        count += 1
                need(count == expected_count, 'Prediction count differs from completed summary: ' + prefix)
                need(full or ids == subset_ids, 'Validation predictions differ from fixed 1024 sample IDs')
    return ref['path']


def inventory(scope, index_sha):
    complete = training_guard()
    index, names = terminal_guard(scope, index_sha)
    names += ['runs/' + arm + '/complete.json' for arm in ARMS]
    if scope == 'operations':
        names += list(OPERATIONS)
        fd = b.open_remote(SUP.rstrip('/'), directory=True)
        try:
            with os.scandir(fd) as entries:
                actual = [entry.name for entry in entries]
        finally:
            os.close(fd)
        need(set(SUP_REQUIRED) <= set(actual) <= SUP_ALLOWED, 'Unexpected/missing supervisor file; inspect explicitly')
        names += [SUP + name for name in actual]
    else:
        names += list(EVAL_FILES + EVAL_LOGS + EVAL_HELPERS) + [SUBSET, 'config/efficiency_r2.json', SUP + 'supervisor.json']
        names += [evaluate_guard(index)]
    need(len(names) == len(set(names)) and all(allowed(n, scope) for n in names), 'Invalid sidecar inventory')
    for name, expected in (('config/efficiency_r2.json', CONFIG_SHA), (SUBSET, SUBSET_SHA),
            ('delivery/backup_completed_run.py', BASE_SHA),
            ('delivery/backup_final_sidecars.py', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())):
        need(b.hash_remote(name)[0]['sha256'] == expected, 'Locked runtime input changed: ' + name)
    return sorted(names), complete


def pack(scope, index_sha):
    need(os.name == 'posix' and hasattr(os, 'O_NOFOLLOW'), 'Pack is fixed-root Linux only')
    archive_name, manifest_name = output_names(scope)
    need(not b.remote_exists(archive_name) and not b.remote_exists(manifest_name), 'Refuse existing backup outputs')
    names, complete = inventory(scope, index_sha)
    records, snapshots = [], {}
    for name in names:
        row, snapshots[name] = b.hash_remote(name)
        records.append(row)
    with os.fdopen(b.open_remote(archive_name, create=True), 'wb') as output:
        with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for row in records:
                name = row['path']
                with os.fdopen(b.open_remote(name), 'rb') as source:
                    before = os.fstat(source.fileno())
                    need(b.identity(before) == snapshots[name], 'Source changed before packing')
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = row['bytes'], 0o600, int(before.st_mtime)
                    reader = b.HashingReader(source)
                    archive.addfile(info, reader)
                    need((reader.count, reader.sha.hexdigest()) == (row['bytes'], row['sha256'])
                         and b.identity(os.fstat(source.fileno())) == snapshots[name], 'Source changed during packing')
        output.flush(); os.fsync(output.fileno())
    need(inventory(scope, index_sha) == (names, complete), 'Readiness/inventory changed during packing')
    for row in records:
        actual, snapshot = b.hash_remote(row['path'])
        need(actual == row and snapshot == snapshots[row['path']], 'Source changed after packing')
    with os.fdopen(b.open_remote(archive_name), 'rb') as source:
        before = os.fstat(source.fileno())
        b.verify_tar(source, records)
        source.seek(0)
        size, sha = b.digest(source)
        need(b.identity(before) == b.identity(os.fstat(source.fileno())) and size == before.st_size, 'Archive changed')
    training_guard(); terminal_guard(scope, index_sha)
    for row in records:
        with os.fdopen(b.open_remote(row['path']), 'rb') as stream:
            need(b.identity(os.fstat(stream.fileno())) == snapshots[row['path']], 'Source changed before manifest')
    manifest = dict(schema=SCHEMA, status='PASS', scope=scope, root=ROOT, created_at_utc=utc(),
        terminal_index=dict(path=index_name(scope), sha256=index_sha),
        archive=dict(path=archive_name, bytes=size, sha256=sha), files=records, file_count=len(records),
        payload_bytes=sum(r['bytes'] for r in records), base_helper_sha256=BASE_SHA,
        sidecar_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        before_during_after_source_sha_equal=True, archive_payloads_independently_rehashed=True,
        network_or_gpu_used=False, checkpoint_loaded=False,
        limitation='Stable filesystem backup; native receipts were externally collected. Not live platform verification or local restore acceptance.')
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with os.fdopen(b.open_remote(manifest_name, create=True), 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    need(read_json(manifest_name) == manifest, 'Manifest readback differs')
    b.log('sidecars_backup_pass', scope=scope, archive=manifest['archive'], manifest=manifest_name,
          manifest_sha256=hashlib.sha256(raw).hexdigest(), files=len(records))


def validate_manifest(manifest, scope):
    need(manifest.get('schema') == SCHEMA and manifest.get('status') == 'PASS'
         and manifest.get('scope') == scope and manifest.get('root') == ROOT
         and manifest.get('base_helper_sha256') == BASE_SHA, 'Wrong sidecar manifest identity')
    archive = manifest['archive']
    need(archive['path'] == output_names(scope)[0] and type(archive['bytes']) is int and archive['bytes'] > 0
         and re.fullmatch('[0-9a-f]{64}', archive['sha256']), 'Invalid archive metadata')
    expected = {}
    for row in manifest['files']:
        name = row['path']
        need(allowed(name, scope) and name not in expected and type(row['bytes']) is int and row['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Unsafe/duplicate manifest member')
        need(all(not PureWindowsPath(p).is_reserved() and not p.endswith((' ', '.')) for p in b.parts(name)),
             'Nonportable member path')
        expected[name] = row
    required = set(OPERATIONS) | {SUP + n for n in SUP_REQUIRED} if scope == 'operations' else set(EVAL_FILES + EVAL_LOGS + EVAL_HELPERS) | {SUBSET, 'config/efficiency_r2.json', SUP + 'supervisor.json'}
    required |= {index_name(scope)} | {'runs/' + a + '/complete.json' for a in ARMS}
    need(required <= set(expected) and manifest['file_count'] == len(expected)
         and manifest['payload_bytes'] == sum(r['bytes'] for r in expected.values()), 'Missing required sidecars/totals')
    return expected


def restore(scope, archive_path, manifest_path, manifest_sha, restore_name):
    need(re.fullmatch('[0-9a-f]{64}', manifest_sha) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', restore_name),
         'Unsafe manifest SHA/restore name')
    script = Path(__file__).resolve()
    workspace = script.parents[3]
    need(script.parent in (workspace / 'outputs/stage09_delivery/goal_20261027_r1',
         workspace / 'experiments/09_kuairec_listwise/formal_execution_20261027') and (workspace / '.git').exists(),
         'Restore helper must be in an allowlisted workspace directory')
    target = b.reject_links(workspace / 'outputs/stage09_restore' / restore_name / 'sidecars' / scope)
    receipt = target.parent / ('RESTORE_' + scope + '.json')
    need(not os.path.lexists(target) and not os.path.lexists(receipt), 'Refuse existing restore target/receipt')
    with b.open_local(manifest_path) as stream:
        before = os.fstat(stream.fileno())
        need(b.digest(stream) == (before.st_size, manifest_sha), 'Manifest SHA mismatch')
        stream.seek(0); manifest = json.load(stream)
        need(b.identity(before) == b.identity(os.fstat(stream.fileno())), 'Manifest changed')
    expected = validate_manifest(manifest, scope)
    with b.open_local(archive_path) as source:
        before = os.fstat(source.fileno())
        need(b.digest(source) == (manifest['archive']['bytes'], manifest['archive']['sha256']), 'Archive SHA mismatch')
        source.seek(0); b.verify_tar(source, list(expected.values()))
        need(b.identity(before) == b.identity(os.fstat(source.fileno())), 'Archive changed before extraction')
        target.parent.mkdir(parents=True, exist_ok=True); target.mkdir(exist_ok=False)
        source.seek(0); seen = set()
        with tarfile.open(fileobj=source, mode='r:') as archive:
            for member in archive:
                b.validate_member(member, expected, seen)
                path = target.joinpath(*b.parts(member.name))
                need(path.resolve().is_relative_to(target.resolve()), 'Extraction escapes sidecar target')
                b.reject_links(path.parent); path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as payload, b.open_local(path, create=True) as output:
                    for block in iter(lambda: payload.read(b.CHUNK), b''):
                        output.write(block)
                    output.flush(); os.fsync(output.fileno())
                seen.add(member.name)
        need(seen == set(expected) and b.identity(before) == b.identity(os.fstat(source.fileno())), 'Archive changed')
    actual = set()
    for folder, dirs, files in os.walk(target, followlinks=False):
        for name in dirs:
            b.reject_links(Path(folder) / name)
        for name in files:
            path = b.reject_links(Path(folder) / name)
            relative = path.relative_to(target).as_posix()
            need(relative in expected, 'Unexpected restored file')
            with b.open_local(path) as stream:
                need(b.digest(stream) == (expected[relative]['bytes'], expected[relative]['sha256']), 'Restored SHA differs')
            actual.add(relative)
    need(actual == set(expected), 'Restored file set incomplete')
    result = dict(schema='stage09_sidecars_local_restore_v1', status='PASS', scope=scope, created_at_utc=utc(),
        target=str(target), manifest_sha256=manifest_sha, archive=manifest['archive'], file_count=len(expected),
        all_files_independently_rehashed=True, network_or_gpu_used=False, checkpoint_loaded=False)
    with b.open_local(receipt, create=True) as stream:
        stream.write((json.dumps(result, indent=2) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())
    with b.open_local(receipt) as stream:
        need(json.load(stream) == result, 'Restore receipt readback differs')
    b.log('sidecars_restore_pass', **result, receipt=str(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack'); p.add_argument('--terminal-index-sha256', required=True)
    r = commands.add_parser('restore')
    for sub in (p, r):
        sub.add_argument('--scope', choices=('operations', 'evaluations'), required=True)
    for name in ('archive', 'manifest', 'manifest-sha256', 'restore-name'):
        r.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.scope, args.terminal_index_sha256)
    else:
        restore(args.scope, args.archive, args.manifest, args.manifest_sha256, args.restore_name)


if __name__ == '__main__':
    main()
