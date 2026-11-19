"""Transfer one finished freeze35 branch; never a full backup or metric audit.

Standard library only. The explicitly supplied backup tool is SHA-pinned before
import; its existing safe byte-extraction implementation is reused unchanged.
No platform access, tensor loading, training, source mutation or retry exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
from pathlib import Path, PurePosixPath
import sys
import tarfile

BACKUP_SHA = 'f99e2293ac7dc528b53151ecd0dddbb582095ef564846021fa9af12ef2df811d'
FORMAT = 'stage09_freeze35_early_branch_transfer_v1'
SCOPE = 'early branch byte transfer, not full Goal backup/model restore/metric audit'
METADATA = ('FREEZE35_PARENTS.json', 'FREEZE35_REFERENCE_RESULTS.json', 'FREEZE35_EXECUTION_MANIFEST.json')


def helper(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BACKUP_SHA:
        raise ValueError('backup_delivery.py differs from the independently reviewed source')
    spec = importlib.util.spec_from_file_location('_freeze35_backup_transfer', path)
    module = importlib.util.module_from_spec(spec)
    original = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    module.no_links(path)
    return module


def selected_files(h, root, branch):
    """Inventory ONLY this branch, its finished job files, and three metadata."""
    paths = {}
    for prefix in ('runs/' + h.NAMES[branch], 'jobs/' + branch):
        directory = h.child(root, prefix)
        if not directory.is_dir():
            raise ValueError('Required branch directory missing: ' + prefix)
        paths.update({prefix + '/' + name: directory / name for name in h.inventory(directory)})
    paths.update({'code/' + name: h.child(root, 'code/' + name) for name in METADATA})
    known, records = {}, {}
    for name, path in sorted(paths.items()):
        h.register(name, known)
        records[name] = h.file_record(path)
    return paths, records


def validate(h, root, branch, execution_sha, *, source_root=None):
    """Structural/byte acceptance only; no tensor or recommendation metric audit."""
    directory = root / 'runs' / h.NAMES[branch]
    if (directory / '.running').exists() or (directory / '.running').is_symlink():
        raise ValueError('Branch is still running')
    read = h.read
    job = read(root / 'jobs' / branch / 'result.json')
    job_request = read(root / 'jobs' / branch / 'request.json')
    original_root = PurePosixPath(str(source_root or root).replace('\\', '/'))
    original_run = original_root / 'runs' / h.NAMES[branch]
    if (job.get('status') != 'predictions_complete_independent_audit_pending'
            or job.get('branch') != branch or job.get('automatic_retry') is not False
            or job.get('train_process_complete') is not True or job.get('test_process_complete') is not True
            or job.get('signal') is not None or PurePosixPath(job['output'].replace('\\', '/')) != original_run
            or job_request.get('branch') != branch or job_request.get('manifest_sha') != execution_sha):
        raise ValueError('Job is not the requested finished, non-retried branch')
    execution = read(root / 'code' / METADATA[2])
    if (h.sha(root / 'code' / METADATA[2]) != execution_sha
            or execution.get('format') != 'stage09_h_freeze35_execution_v1'
            or execution.get('package_mode') != 'committed' or execution.get('gpu_execution_allowed') is not True):
        raise ValueError('Caller-approved execution manifest mismatch')
    for name in METADATA[:2]:
        h.check(root / 'code' / name, execution['metadata'][name])
    parents, refs = read(root / 'code' / METADATA[0]), read(root / 'code' / METADATA[1])
    if parents['parents'][branch]['sha256'] != h.PARENTS[branch]:
        raise ValueError('Parent checkpoint identity differs')
    request, complete = read(directory / 'request.json'), read(directory / 'complete.json')
    selection, tests = read(directory / 'selection.json'), read(directory / 'test_predictions_complete.json')
    selection_sha = h.sha(directory / 'selection.json')
    if (request['branch'] != branch or request['parent_sha256'] != h.PARENTS[branch]
            or request['reference_manifest_sha256'] != h.sha(root / 'code' / METADATA[1])
            or complete['status'] != 'training_and_validation_complete' or complete['branch'] != branch
            or (complete['phase'], complete['r_updates'], complete['t_updates']) != (45, 24255, 360 if branch == 'B' else 200)
            or selection.get('branch') != branch or selection.get('fixed_endpoint') != 35
            or complete['selection_sha256'] != selection_sha or tests['selection_sha256'] != selection_sha
            or tests['branch'] != branch or tests.get('independently_audited') is not False):
        raise ValueError('Training/selection/test completion is incomplete or mixed')
    coverage = read(directory / 'CHECKPOINT_COVERAGE.json')
    if coverage.get('status') != 'passed' or h.sha(directory / 'CHECKPOINT_COVERAGE.json') != complete['checkpoint_coverage_sha256']:
        raise ValueError('Checkpoint coverage SHA/status differs')
    specs = h.node_specs(branch)
    for filename, clocks in specs.items():
        receipt = read(directory / 'checkpoint_receipts' / (filename + '.json'))
        h.check(directory / filename, receipt)
        if (receipt['path'] != filename or receipt['r_updates'] != clocks['R'] or receipt['t_updates'] != clocks['T']
                or receipt['progress'] != {k: clocks[k] for k in ('phase', 'update', 'boundary_prepared')}
                or set(receipt['fields']) != {'model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'binding'}
                or not all(value is True for value in receipt['fields'].values())):
            raise ValueError('Incomplete checkpoint receipt: ' + filename)
        if filename != 'R35_training_boundary.pt' and coverage['records'][filename] != receipt:
            raise ValueError('Coverage record does not bind actual checkpoint receipt')
    if set(coverage['records']) != set(specs) - {'R35_training_boundary.pt'} or h.sha(directory / 'last.pt') != complete['last_sha256']:
        raise ValueError('Missing/extra checkpoints or changed rolling last')

    def local(path):
        relative = PurePosixPath(path.replace('\\', '/')).relative_to(original_root).as_posix()
        if not relative.startswith('runs/' + h.NAMES[branch] + '/'):
            raise ValueError('New evaluation points outside this branch')
        return h.child(root, relative)

    chosen = selection['selected']
    expected = {35, chosen['epoch']} if branch in ('A', 'C') else {35}
    if len(tests['results']) != len(expected) or {row['epoch'] for row in tests['results']} != expected:
        raise ValueError('Missing or additional test endpoints')
    fixed = read(directory / 'validation_records/R35.json')
    for target in (fixed, chosen):
        if target['kind'] == 'new':
            checkpoint = local(target['checkpoint'])
            if (checkpoint.parent != directory or checkpoint.name not in specs
                    or h.sha(checkpoint) != target['checkpoint_sha256']
                    or specs[checkpoint.name]['R'] != target['epoch'] * 693):
                raise ValueError('Locked validation model is outside accepted immutable coverage')
        elif (branch != 'A' or target['checkpoint_sha256']
                != refs['validation']['A'][str(target['epoch'])]['checkpoint_sha256']):
            raise ValueError('Historical selected model differs from frozen validation reference')
    if fixed['epoch'] != 35 or fixed['kind'] != 'new':
        raise ValueError('Fixed R35 model is not the new R35 endpoint')
    for result in tests['results']:
        target = fixed if result['epoch'] == 35 else chosen
        if result['checkpoint_sha256'] != target['checkpoint_sha256']:
            raise ValueError('Test checkpoint differs from locked validation selection')
        if result.get('reused') is True:
            reference = refs['test'][branch][str(result['epoch'])]
            if any(result[k] != reference[k] for k in ('summary_sha256', 'predictions_sha256', 'checkpoint_sha256')):
                raise ValueError('Reused historical test differs from frozen reference')
        else:
            for key in ('summary', 'predictions'):
                if h.sha(local(result[key])) != result[key + '_sha256']:
                    raise ValueError('New test payload SHA mismatch: ' + key)
            eval_complete = read(directory / 'evaluations' / ('test_R' + str(result['epoch'])) / 'complete.json')
            if (any(eval_complete[k + '_sha256'] != result[k + '_sha256'] for k in ('summary', 'predictions'))
                    or eval_complete['binding']['branch'] != branch or eval_complete['binding']['split'] != 'test'
                    or eval_complete['binding']['epoch'] != result['epoch']):
                raise ValueError('Test completion receipt differs from payload/endpoint')
    return dict(immutable_checkpoints=len(specs), selection_sha256=selection_sha,
                checkpoint_coverage_sha256=complete['checkpoint_coverage_sha256'], test_epochs=sorted(expected))


def pack(h, args):
    root, archive = h.no_links(args.root), h.no_links(args.archive)
    receipt_path = archive.with_suffix(archive.suffix + '.receipt.json')
    if archive.exists() or receipt_path.exists() or archive.is_relative_to(root):
        raise FileExistsError('Archive/receipt must be NEW and outside source root')
    structural = validate(h, root, args.branch, args.execution_manifest_sha)
    paths, before = selected_files(h, root, args.branch)
    manifest = dict(format=FORMAT, scope=SCOPE, branch=args.branch, source_root=root.as_posix(), files=before,
        execution_manifest_sha256=args.execution_manifest_sha, structural_checks=structural,
        backup_tool_sha256=BACKUP_SHA, exporter_sha256=h.sha(__file__), full_goal_backup=False,
        model_restore=False, metric_audit=False, self_contained=False,
        external_dependencies='Source, data, PSE, parent states and historical references are NOT included')
    raw = h.encoded(manifest); manifest_sha = hashlib.sha256(raw).hexdigest()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo(h.MANIFEST); info.size = len(raw); info.mode = 0o644
        tar.addfile(info, io.BytesIO(raw))
        for name, path in sorted(paths.items()):
            h.check(path, before[name]); info = tarfile.TarInfo(name)
            info.size = before[name]['bytes']; info.mode = 0o644
            with path.open('rb') as stream: tar.addfile(info, stream)
    if selected_files(h, root, args.branch)[1] != before:
        raise ValueError('Branch/job changed during export; archive is NOT accepted')
    result = dict(format=FORMAT, status='branch_packed_not_metric_audited', scope=SCOPE, branch=args.branch,
        archive=str(archive), **h.file_record(archive), manifest_sha256=manifest_sha, payload_files=len(before),
        source_before_after_sha256_equal=True, full_goal_backup=False, model_restore=False, metric_audit=False)
    h.write_new(receipt_path, result)
    return result


def extract(h, args):
    manifest = h.extract_bytes(args.archive, args.destination, args.archive_sha, args.manifest_sha)
    branch = manifest['branch']; root = h.no_links(args.destination)
    if (manifest.get('format') != FORMAT or manifest.get('scope') != SCOPE or branch not in h.NAMES
            or manifest.get('backup_tool_sha256') != BACKUP_SHA
            or manifest.get('exporter_sha256') != h.sha(__file__)
            or any(manifest.get(k) is not False for k in ('full_goal_backup', 'model_restore', 'metric_audit', 'self_contained'))):
        raise ValueError('Not the reviewed single-branch byte-transfer format')
    paths, actual = selected_files(h, root, branch)
    if actual != manifest['files'] or len(h.inventory(root)) != len(actual) + 1:
        raise ValueError('Archive includes files outside exactly one branch plus three metadata')
    structural = validate(h, root, branch, manifest['execution_manifest_sha256'], source_root=manifest['source_root'])
    if structural != manifest['structural_checks']:
        raise ValueError('Extracted structural proof differs from pack')
    result = dict(format=FORMAT, status='branch_bytes_verified_not_metric_audited', scope=SCOPE,
        branch=branch, archive_sha256=args.archive_sha, manifest_sha256=args.manifest_sha,
        payload_files=len(actual), all_payload_and_disk_sha256_verified=True,
        full_goal_backup=False, model_restore=False, metric_audit=False, platform_release_verified=False,
        external_dependencies=manifest['external_dependencies'])
    h.write_new(root / 'EARLY_BRANCH_TRANSFER_VERIFICATION.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-tool', required=True, help='Exact f99e2293... reviewed backup_delivery.py')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--root', required=True); p.add_argument('--branch', required=True, choices=('A', 'B', 'C'))
    p.add_argument('--execution-manifest-sha', required=True); p.add_argument('--archive', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--archive', required=True); p.add_argument('--archive-sha', required=True)
    p.add_argument('--manifest-sha', required=True); p.add_argument('--destination', required=True)
    args = parser.parse_args(argv); h = helper(args.backup_tool)
    result = pack(h, args) if args.command == 'pack' else extract(h, args)
    sys.stdout.buffer.write(h.encoded(result))


if __name__ == '__main__':
    main()
