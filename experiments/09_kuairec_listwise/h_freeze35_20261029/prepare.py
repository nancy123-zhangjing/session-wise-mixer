"""Freeze an authenticated historical runtime plus new recovery35 modules.

prepare/inspect run locally (checkpoint inspection needs CPU torch); verify is
stdlib-only and can run after SHA-checked deployment. Inputs are never repaired.
All output directories must be new. Fixed input bytes/provenance produce the
same gzip/tar bytes: no clock, output path, owner, or gzip filename is embedded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile


WORKSPACE = Path(__file__).resolve().parents[3]
FORMAT = 'stage09_h_freeze35_execution_v1'
REMOTE_BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
GOAL = WORKSPACE / 'outputs/stage09_restore/goal_20261027_r1'
RECOVERY = WORKSPACE / 'outputs/stage09_restore/h1_recovery_20261028_r2'
TEST = WORKSPACE / 'outputs/stage09_restore/h1_recovery_test_20261029_r1'
BASE_CODE = RECOVERY / 'code'
EXECUTION_SHA = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
OLD_BINDING_SHA = '4b5fcc0f04ad3aefad1c2d00b9c6eeb6e3428b6939991ee3393bf4516f00f807'
ARCHIVES = {
    'H0': ('outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_H0_R1.json',
           '815790d79317aacf497cdffd5c6c71daf02a48f389127fe5d058afe4959effa4', GOAL, REMOTE_BASE + '/goal_20261027_r1'),
    'H1': ('outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_H1_R1.json',
           '768ffba5dcb22f6223b3c731cd7292e1b4da26b7b6375b5d5308478a01b2e81c', GOAL, REMOTE_BASE + '/goal_20261027_r1'),
    'warmup': ('outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_warmup_H_R1.json',
               '0f8db52a3b374e507695f0c6705ec69bcaffcbc17fa278c2d36c7a0cc9fdc351', GOAL, REMOTE_BASE + '/goal_20261027_r1'),
    'evaluations': ('outputs/stage09_delivery/goal_20261027_r1/STAGE09_EVALUATIONS_R1.json',
                    '2a0b453b572f8affcbd9437b2f6b8bd327c1780c88c9207cb5702c52d7009561',
                    GOAL / 'sidecars/evaluations', REMOTE_BASE + '/goal_20261027_r1'),
    'recovery': ('outputs/stage09_restore/h1_recovery_20261028_r2/MANIFEST.json',
                 'f6dee18b9931c9efe9798211034e8c7205c5a405829f877c781828f299efbaec', RECOVERY, REMOTE_BASE + '/h1_recovery_20261028_r2'),
    'test': ('outputs/stage09_restore/h1_recovery_test_20261029_r1/MANIFEST.json',
             'ac70b6d1659d125b2e5023a2ae451d7b173f2fb028c6e48eec01eccc4cd80fd8', TEST, REMOTE_BASE + '/h1_recovery_test_20261029_r1'),
}
PARENTS = {
    'A': dict(name='H1-Freeze20-R35', archive='recovery', relative='runs/H1-freeze/last.pt',
              sha256='1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9', arm='H1-freeze', R=18018, T=200, phase=33),
    'B': dict(name='H0-Freeze32-R35', archive='H0', relative='runs/H0/best.pt',
              sha256='8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e', arm='H0', R=22176, T=360, phase=41),
    'C': dict(name='H0-Freeze20-R35', archive='H0', relative='runs/H0/initial.pt',
              sha256='b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166', arm='H0', R=3465, T=0, phase=5),
}
METADATA = ('FREEZE35_PARENTS.json', 'FREEZE35_REFERENCE_RESULTS.json', 'FREEZE35_EXECUTION_MANIFEST.json')
READ_BYTES = 4 * 1024 * 1024


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b''):
            value.update(block)
    return value.hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    for parent in [*reversed(path.parents), path]:
        if not parent.exists() and not parent.is_symlink():
            continue
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked/reparse path forbidden: ' + str(parent))
    return path


def child(root, relative):
    p = PurePosixPath(relative)
    if (not relative or p.is_absolute() or p.as_posix() != relative or '\\' in relative or ':' in relative
            or any(part in ('', '.', '..') or part.endswith(('.', ' ')) for part in p.parts)):
        raise ValueError('Unsafe relative path: ' + relative)
    return no_links(Path(root) / Path(*p.parts))


def check(path, record):
    path = no_links(path)
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
        raise ValueError('Source size/SHA mismatch: ' + str(path))
    return path


def inventory(root):
    result, folded = {}, set()
    for directory, dirs, files in os.walk(no_links(root), followlinks=False):
        dirs[:] = sorted(name for name in dirs if name != '__pycache__')
        for name in dirs:
            no_links(Path(directory) / name)
        for name in sorted(files):
            if name.endswith(('.pyc', '.pyo')):
                continue
            path = no_links(Path(directory) / name)
            relative = path.relative_to(root).as_posix()
            child(root, relative)
            if not path.is_file() or relative.casefold() in folded:
                raise ValueError('Nonregular/case-colliding source: ' + relative)
            folded.add(relative.casefold())
            result[relative] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return dict(sorted(result.items()))


def source_catalogs():
    result = {}
    for name, (relative, expected, local, remote) in ARCHIVES.items():
        path = no_links(WORKSPACE / relative)
        if sha256(path) != expected:
            raise ValueError('Accepted archive manifest changed: ' + relative)
        value = read(path)
        rows = value['files']
        records = {row['path']: row for row in rows} if isinstance(rows, list) else rows
        result[name] = dict(local=local, remote=remote, records=records, manifest=relative, manifest_sha256=expected)
    return result


def referenced(catalogs, archive, relative):
    source = catalogs[archive]
    record = source['records'][relative]
    path = check(child(source['local'], relative), record)
    return dict(local=str(path), remote=source['remote'] + '/' + relative,
                bytes=record['bytes'], sha256=record['sha256'], source_manifest=source['manifest'],
                source_manifest_sha256=source['manifest_sha256'])


def verify_binding(binding, code):
    for group, prefix in [('code', ''), ('engine', 'external/DiffGRM/')]:
        for relative, expected in binding[group].items():
            path = child(code, prefix + relative)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Parent {group} source changed: {relative}')


def reference_results(catalogs, parents):
    result = dict(format='stage09_h_freeze35_reference_results_v1', validation={key: {} for key in PARENTS},
                  test={key: {} for key in PARENTS}, path_convention='summary/predictions/checkpoint are remote absolute paths; local holds CPU paths',
                  path_mappings=[dict(remote=catalogs[key]['remote'], local=str(catalogs[key]['local']))
                                 for key in ('H0', 'recovery', 'test')]
                  + [dict(remote=catalogs['evaluations']['remote'] + '/evaluations',
                          local=str(catalogs['evaluations']['local'] / 'evaluations'))])

    def add(split, arm, epoch, archive, directory, checkpoint=None):
        summary = referenced(catalogs, archive, directory + '/summary.json')
        predictions = referenced(catalogs, archive, directory + '/per_sample.jsonl.gz')
        meta = read(summary['local'])
        expected = dict(split=split, full_split=True, subset=None, architecture='H', seed=2024, beam=32,
                        examples=71657 if split == 'validation' else 53194, r_updates=693 * epoch)
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError('Wrong historical evaluation protocol: ' + directory)
        record = dict(summary=summary['remote'], predictions=predictions['remote'],
                      summary_sha256=summary['sha256'], predictions_sha256=predictions['sha256'],
                      summary_bytes=summary['bytes'], predictions_bytes=predictions['bytes'],
                      local=dict(summary=summary['local'], predictions=predictions['local']),
                      source_manifest=summary['source_manifest'], source_manifest_sha256=summary['source_manifest_sha256'],
                      checkpoint_available=checkpoint is not None, r_updates=meta['r_updates'], t_updates=meta['t_updates'])
        if checkpoint is not None:
            record.update(checkpoint=checkpoint['remote'], checkpoint_sha256=checkpoint['sha256'], checkpoint_bytes=checkpoint['bytes'])
            record['local']['checkpoint'] = checkpoint['local']
        else:
            record['checkpoint_note'] = 'Historical intermediate checkpoint was not retained; predictions are replay references only.'
        result[split][arm][str(epoch)] = record

    h1_r20 = referenced(catalogs, 'H1', 'runs/H1/best.pt')
    add('validation', 'A', 20, 'H1', 'runs/H1/validation_R20', h1_r20)
    for epoch in (23, 26):
        prefix = f'runs/H1-freeze/evaluations/R{epoch}'
        receipt_ref = referenced(catalogs, 'recovery', prefix + '/complete.json')
        receipt = read(receipt_ref['local'])
        if re.fullmatch(r'attempt[1-9][0-9]*', receipt['attempt']) is None:
            raise ValueError('Unsafe accepted recovery evaluation attempt')
        checkpoint = (referenced(catalogs, 'recovery', 'runs/H1-freeze/phase28_R_R23_T200.pt')
                      if epoch == 23 else parents['A'])
        add('validation', 'A', epoch, 'recovery', prefix + '/' + receipt['attempt'], checkpoint)
        record = result['validation']['A'][str(epoch)]
        if any(record[key] != receipt[key] for key in ('summary_sha256', 'predictions_sha256')):
            raise ValueError('Accepted recovery evaluation receipt differs from archive')
    add('validation', 'B', 32, 'H0', 'runs/H0/validation_R32', parents['B'])
    add('validation', 'C', 5, 'warmup', 'runs/warmup_H/validation_R05', parents['C'])
    for epoch in (8, 11, 14, 17, 20):
        add('validation', 'C', epoch, 'H0', f'runs/H0/validation_R{epoch:02d}')
    add('test', 'A', 20, 'evaluations', 'evaluations/H1_test_beam32_r2', h1_r20)
    add('test', 'A', 26, 'test', 'H1-freeze/evaluation', parents['A'])
    add('test', 'B', 32, 'evaluations', 'evaluations/H0_test_beam32', parents['B'])
    return result


def collect_evidence():
    catalogs = source_catalogs()
    execution = read(BASE_CODE / 'RECOVERY_EXECUTION_MANIFEST.json')
    if (sha256(BASE_CODE / 'RECOVERY_EXECUTION_MANIFEST.json') != EXECUTION_SHA
            or sha256(BASE_CODE / 'RECOVERY_PARENT_BINDING.json') != OLD_BINDING_SHA):
        raise ValueError('Historical execution provenance changed')
    expected_base = {name.removeprefix('code/'): dict(bytes=row['bytes'], sha256=row['sha256'])
                     for name, row in catalogs['recovery']['records'].items() if name.startswith('code/')}
    actual = inventory(BASE_CODE)
    if actual != expected_base or len(actual) != 3029:
        raise ValueError('Authenticated r2 runtime inventory differs from archived code')
    for name, record in execution['execution_files'].items():
        if actual.get(name) != record:
            raise ValueError('Original r2 execution file differs: ' + name)
    original = read(BASE_CODE / 'RECOVERY_PARENT_BINDING.json')
    if len(original['code']) != 20 or len(original['engine']) != 20:
        raise ValueError('Expected original 20 code plus 20 engine bindings')
    verify_binding(original, BASE_CODE)
    verify_binding(original, GOAL / 'code')
    import torch
    parents = {}
    for arm, expected in PARENTS.items():
        record = referenced(catalogs, expected['archive'], expected['relative'])
        if record['sha256'] != expected['sha256']:
            raise ValueError('Wrong parent checkpoint: ' + arm)
        saved = torch.load(record['local'], map_location='cpu', weights_only=False)
        inner = saved['inner'] if arm == 'A' else saved
        if (inner['arm'], inner['r_updates'], inner['t_updates'], inner['progress']['phase'], inner['progress']['update'], inner['precision']) != (
                expected['arm'], expected['R'], expected['T'], expected['phase'], 0, 'bf16'):
            raise ValueError('Wrong parent arm/progress/precision: ' + arm)
        if saved.get('preflight_only') or inner.get('preflight_only'):
            raise ValueError('Preflight parent cannot start a formal branch')
        binding = inner['binding']
        if len(binding['engine']) != 20 or len(binding['code']) != (25 if arm == 'A' else 20):
            raise ValueError('Unexpected parent source binding size: ' + arm)
        verify_binding(binding, BASE_CODE)
        parents[arm] = dict(record, name=expected['name'], arm=inner['arm'], r_updates=inner['r_updates'],
            t_updates=inner['t_updates'], phase=inner['progress']['phase'], update=0, precision=inner['precision'],
            checkpoint_format=saved['format'], binding=binding)
        del saved, inner
    for arm in ('B', 'C'):
        for key in ('data', 'asset_manifest', 'engine'):
            if parents[arm]['binding'][key] != parents['A']['binding'][key]:
                raise ValueError('Parents have incompatible data/assets/engine: ' + key)
        if parents[arm]['binding']['code'] != original['code']:
            raise ValueError('H0 source differs from original 20-file code binding')
    dataset = dict(local=str(WORKSPACE / 'outputs/kuairec_listwise_v1/run1'),
                   remote=original['config']['dataset'], **original['data'])
    check(Path(dataset['local']) / 'manifest.json', dict(bytes=(Path(dataset['local']) / 'manifest.json').stat().st_size, sha256=dataset['manifest_sha256']))
    for name, record in dataset['files'].items():
        check(child(dataset['local'], name), record)
    assets_path = GOAL / 'assets/pse_v1'
    asset_manifest = read(assets_path / 'manifest.json')
    if sha256(assets_path / 'manifest.json') != original['asset_manifest'] or asset_manifest.get('toy') is not False:
        raise ValueError('Content/PSE manifest differs from original binding')
    for name, record in asset_manifest['artifacts'].items():
        check(child(assets_path, name), record)
    assets = dict(local=str(assets_path), remote=original['config']['assets'], manifest_sha256=original['asset_manifest'],
                  files=asset_manifest['artifacts'], payloads_included=False)
    ancestor = referenced(catalogs, 'H1', 'runs/H1/best.pt')
    if ancestor['sha256'] != 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1':
        raise ValueError('Original H1 R20 ancestor differs')
    results = reference_results(catalogs, parents)
    references = dict(format=FORMAT, parents=parents, ancestors={'A': ancestor}, dataset=dataset, assets=assets,
        existing_runtime=dict(python='/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python3.10',
            engine_relative_to_code='external/DiffGRM', previously_used_image='ngc-pytorch:25.02-cuda12.8.0-py3',
            remote_paths_revalidated_this_run=False, model_and_data_payloads_included=False),
        historical_execution_manifest_sha256=EXECUTION_SHA, original_parent_binding_sha256=OLD_BINDING_SHA,
        original_code_files=20, original_engine_files=20,
        source_manifests={key: dict(path=value['manifest'], sha256=value['manifest_sha256']) for key, value in catalogs.items()})
    return expected_base, references, results


def new_modules():
    result = {}
    for path in sorted((WORKSPACE / 'scripts/listwise_diffgrm').glob('recovery35*.py')):
        relative = path.relative_to(WORKSPACE).as_posix()
        if re.fullmatch(r'scripts/listwise_diffgrm/recovery35(?:_[a-z0-9_]+)?\.py', relative) is None:
            raise ValueError('New extension outside recovery35 whitelist: ' + relative)
        no_links(path)
        if (BASE_CODE / relative).exists():
            raise ValueError('New extension would overwrite historical source: ' + relative)
        result[relative] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    required = {'scripts/listwise_diffgrm/recovery35.py', 'scripts/listwise_diffgrm/recovery35_run.py'}
    if not required <= result.keys():
        raise ValueError('Required recovery35 controller/runner are not yet present')
    return result


def git_provenance(paths):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(WORKSPACE), *args], encoding='utf-8').strip()
    status = git('status', '--porcelain=v1', '--untracked-files=all', '--', *paths)
    tracked = set(git('ls-files', '--', *paths).splitlines())
    return dict(commit=git('rev-parse', 'HEAD'), package_sources_uncommitted=bool(status) or not set(paths) <= tracked,
                package_source_status=status.splitlines(), other_workspace_changes_ignored=True)


def fresh_output(path):
    path = no_links(path if Path(path).is_absolute() else WORKSPACE / path)
    if (not path.is_relative_to(WORKSPACE) or path == WORKSPACE
            or any(path.is_relative_to(root) or root.is_relative_to(path) for root in (GOAL, RECOVERY, TEST))
            or any(part in ('.git', '.codex', '.agents') for part in path.relative_to(WORKSPACE).parts)):
        raise ValueError('Output must be an independent new workspace directory')
    if path.exists():
        raise FileExistsError('Use a new output directory; execution copies are immutable: ' + str(path))
    return path


def pack_archive(output, code_records, manifest_sha, preparer_sha):
    members = {'code/' + name: (output / 'code' / name, row['sha256']) for name, row in code_records.items()}
    for name in METADATA:
        path = output / 'code' / name
        members['code/' + name] = path, sha256(path)
    members['delivery/prepare.py'] = Path(__file__), preparer_sha
    archive = output / 'h_freeze35_code.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=6) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as stream:
            for name, (path, expected) in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open('rb') as source:
                    stream.addfile(info, source)
                if sha256(path) != expected:
                    raise ValueError('File changed during packing: ' + name)
    with tarfile.open(archive, 'r:gz') as stream:
        rows = stream.getmembers()
        if len(rows) != len(members) or {row.name for row in rows} != set(members):
            raise ValueError('Archive inventory differs')
        for row in rows:
            if not row.isfile():
                raise ValueError('Archive contains a nonregular member')
            digest = hashlib.sha256()
            with stream.extractfile(row) as source:
                for block in iter(lambda: source.read(READ_BYTES), b''):
                    digest.update(block)
            if digest.hexdigest() != members[row.name][1]:
                raise ValueError('Archive member differs: ' + row.name)
    return dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive),
                manifest_sha256=manifest_sha, verified_members=len(members), deterministic_gzip_mtime=0,
                model_data_prediction_payloads_included=False)


def prepare(output_root, *, draft=False):
    output = fresh_output(output_root)
    base, references, results = collect_evidence()
    preparer = Path(__file__).resolve().relative_to(WORKSPACE).as_posix()
    initial_provenance = git_provenance(sorted([*new_modules(), preparer]))
    if initial_provenance['package_sources_uncommitted'] and not draft:
        raise ValueError('Commit this task\'s source first, or use --draft for local CPU preparation only')
    output.mkdir(parents=True)
    code = output / 'code'
    for name, record in base.items():
        source = child(BASE_CODE, name)
        destination = child(code, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        check(destination, record)
    # Capture the few new modules after the large immutable copy, so unrelated
    # development during that copy does not create a mixed source snapshot.
    overlays = new_modules()
    source_paths = sorted([*overlays, preparer])
    provenance = git_provenance(source_paths)
    if provenance['package_sources_uncommitted'] and not draft:
        raise ValueError('New source became uncommitted during preparation')
    for name, record in overlays.items():
        destination = child(code, name)
        shutil.copyfile(child(WORKSPACE, name), destination)
        check(destination, record)
    execution = dict(sorted({**base, **overlays}.items()))
    write_new(code / METADATA[0], references)
    write_new(code / METADATA[1], results)
    preparer_sha = sha256(__file__)
    manifest = dict(format=FORMAT, package_mode='draft_cpu_only' if draft else 'committed',
        gpu_execution_allowed=not draft, workspace=provenance,
        historical_source=dict(root='outputs/stage09_restore/h1_recovery_20261028_r2/code',
            execution_manifest_sha256=EXECUTION_SHA, archive_manifest_sha256=ARCHIVES['recovery'][1],
            original_code_engine_files_verified=40, retained_files=len(base), files=base),
        overlays=overlays, extension_allowlist='scripts/listwise_diffgrm/recovery35(?:_[a-z0-9_]+)?.py',
        execution_files=execution, execution_file_count=len(execution),
        metadata={name: dict(bytes=(code / name).stat().st_size, sha256=sha256(code / name)) for name in METADATA[:2]},
        preparer=dict(path=preparer, sha256=preparer_sha), payloads_excluded=['checkpoints', 'dataset', 'content/PSE', 'predictions'])
    if provenance != git_provenance(source_paths) or overlays != new_modules():
        raise ValueError('New source/Git provenance changed during preparation; use a fresh destination')
    write_new(code / METADATA[2], manifest)
    manifest_sha = sha256(code / METADATA[2])
    verify(code, manifest_sha, allow_draft=draft)
    receipt = dict(format=FORMAT, status='draft_verified_cpu_only' if draft else 'verified_committed_package',
        archive=pack_archive(output, execution, manifest_sha, preparer_sha), workspace=provenance,
        parent_sha256={key: value['sha256'] for key, value in references['parents'].items()},
        preserved_historical_files=len(base), new_overlay_files=len(overlays), source_inputs_modified=False)
    write_new(output / 'PACKAGE_RECEIPT.json', receipt)
    return receipt


def verify(code_root, manifest_sha, *, references='none', verify_results=False, allow_draft=False):
    """Remote-compatible stdlib verification; never loads checkpoint tensors."""
    code = no_links(code_root)
    if sha256(code / METADATA[2]) != manifest_sha:
        raise ValueError('Execution manifest differs from the caller\'s locked SHA')
    manifest = read(code / METADATA[2])
    if (manifest['format'] != FORMAT or manifest['package_mode'] not in ('committed', 'draft_cpu_only')
            or (manifest['package_mode'] != 'committed' and not allow_draft)):
        raise ValueError('Unrecognized or CPU-only draft package')
    if manifest['gpu_execution_allowed'] != (manifest['package_mode'] == 'committed'):
        raise ValueError('Package execution permission is inconsistent')
    if manifest['package_mode'] == 'committed' and manifest['workspace']['package_sources_uncommitted']:
        raise ValueError('Committed package records uncommitted source')
    expected = dict(manifest['execution_files'])
    expected.update(manifest['metadata'])
    expected[METADATA[2]] = dict(bytes=(code / METADATA[2]).stat().st_size, sha256=manifest_sha)
    if inventory(code) != expected:
        raise ValueError('Execution source inventory/SHA differs from the manifest')
    base = manifest['historical_source']['files']
    if set(manifest['execution_files']) != set(base) | set(manifest['overlays']) or set(base) & set(manifest['overlays']):
        raise ValueError('Original source was overwritten or inventory is not base plus overlays')
    for name, row in base.items():
        if manifest['execution_files'][name] != row:
            raise ValueError('Historical source bytes changed: ' + name)
    for name, row in manifest['overlays'].items():
        if re.fullmatch(r'scripts/listwise_diffgrm/recovery35(?:_[a-z0-9_]+)?\.py', name) is None:
            raise ValueError('Unexpected new module outside recovery35 whitelist')
        if manifest['execution_files'][name] != row:
            raise ValueError('New extension SHA metadata is inconsistent')
    old = read(code / 'RECOVERY_PARENT_BINDING.json')
    if (sha256(code / 'RECOVERY_EXECUTION_MANIFEST.json') != EXECUTION_SHA
            or sha256(code / 'RECOVERY_PARENT_BINDING.json') != OLD_BINDING_SHA):
        raise ValueError('Historical provenance metadata changed')
    verify_binding(old, code)
    ref = read(code / METADATA[0])
    for parent in ref['parents'].values():
        verify_binding(parent['binding'], code)
    checked = dict(parents=0, ancestors=0, dataset=0, assets=0, result_files=0)
    if references != 'none':
        for parent in ref['parents'].values():
            check(parent[references], parent)
            checked['parents'] += 1
        for ancestor in ref['ancestors'].values():
            check(ancestor[references], ancestor)
            checked['ancestors'] += 1
        for group in ('dataset', 'assets'):
            source = ref[group]
            root = Path(source[references])
            if sha256(root / 'manifest.json') != source['manifest_sha256']:
                raise ValueError('Referenced ' + group + ' manifest differs')
            for name, row in source['files'].items():
                check(child(root, name), row)
                checked[group] += 1
        if verify_results:
            results = read(code / METADATA[1])
            seen = set()
            for split in ('validation', 'test'):
                for rows in results[split].values():
                    for row in rows.values():
                        for key in ('summary', 'predictions', 'checkpoint'):
                            if key not in row:
                                continue
                            path = row['local'][key] if references == 'local' else row[key]
                            if path not in seen:
                                check(path, dict(bytes=row[key + '_bytes'], sha256=row[key + '_sha256']))
                                checked['result_files'] += 1
                                seen.add(path)
    return dict(format=FORMAT, status='verified', package_mode=manifest['package_mode'],
                code_files=len(manifest['execution_files']), historical_files=len(base), overlays=len(manifest['overlays']),
                original_bound_sources_verified=40, parent_binding_sizes={key: dict(code=len(value['binding']['code']), engine=len(value['binding']['engine'])) for key, value in ref['parents'].items()},
                external_references_verified=checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'prepare'):
        command = commands.add_parser(name)
        command.add_argument('--output-root', required=True)
        if name == 'prepare':
            command.add_argument('--draft', action='store_true', help='CPU-only copy; cannot pass normal remote verify')
    command = commands.add_parser('verify')
    command.add_argument('--code-root', required=True)
    command.add_argument('--manifest-sha', required=True)
    command.add_argument('--references', choices=('none', 'local', 'remote'), default='none')
    command.add_argument('--verify-results', action='store_true')
    command.add_argument('--allow-draft', action='store_true', help='Only for local CPU verification')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.output_root, draft=args.draft)
    elif args.command == 'inspect':
        output = fresh_output(args.output_root)
        base, parents, results = collect_evidence()
        output.mkdir(parents=True)
        write_new(output / METADATA[0], parents)
        write_new(output / METADATA[1], results)
        result = dict(status='local_references_verified_not_runtime_package', historical_files=len(base),
                      parents={key: {field: value[field] for field in ('name', 'remote', 'sha256', 'r_updates', 't_updates', 'phase')} for key, value in parents['parents'].items()},
                      reference_counts={split: {key: len(value) for key, value in results[split].items()} for split in ('validation', 'test')})
        write_new(output / 'INSPECTION.json', result)
    else:
        result = verify(args.code_root, args.manifest_sha, references=args.references,
                        verify_results=args.verify_results, allow_draft=args.allow_draft)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
