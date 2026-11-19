"""Authenticate the R20 source, overlay new recovery code, and pack an isolated copy.

The restored archive and checkpoint are read-only inputs. Repeating this command
never repairs or deletes an altered original file in the execution directory:
it rejects the mismatch and only refreshes this task's new overlay files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


WORKSPACE = Path(__file__).resolve().parents[2]
PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
RESTORED = WORKSPACE / 'outputs/stage09_restore/goal_20261027_r1'
FORMAT = 'stage09_h1_recovery_execution_package_v1'
EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
                 '.venv', '.runtime', 'node_modules', 'data', 'datasets', 'weights',
                 'checkpoints', 'ckpt', 'outputs', 'tmp'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.pt', '.pth', '.ckpt', '.safetensors', '.parquet',
                     '.npy', '.npz', '.pkl', '.pickle', '.bin', '.arrow', '.feather',
                     '.tar', '.gz', '.tgz', '.zip', '.7z', '.rar'}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def contained_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(part in ('..', '.') for part in rel.parts):
        raise ValueError(f'Unsafe relative path: {relative}')
    candidate = (root / Path(*rel.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes its declared root: {relative}')
    return candidate


def is_link(path):
    # On Windows a junction may not report is_symlink(), but it is a reparse point.
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


def excluded(path):
    return path.name in EXCLUDED_DIRS or path.suffix.lower() in EXCLUDED_SUFFIXES


def inventory(root):
    root = root.resolve()
    files, omissions = {}, []
    for directory, directories, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(directories):
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if excluded(path):
                omissions.append(dict(path=rel, reason='excluded_directory'))
                directories.remove(name)
            elif is_link(path):
                raise ValueError(f'Refusing linked source/execution directory: {path}')
        for name in names:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if excluded(path):
                omissions.append(dict(path=rel, reason='excluded_binary_archive_or_cache', bytes=path.stat().st_size))
                continue
            if is_link(path):
                raise ValueError(f'Refusing linked source/execution file: {path}')
            files[rel] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return dict(sorted(files.items())), sorted(omissions, key=lambda row: row['path'])


def verify_bound_sources(binding, root):
    result = {}
    for category, prefix in (('code', ''), ('engine', 'external/DiffGRM/')):
        entries = binding[category]
        if len(entries) != 20:
            raise ValueError(f'The authenticated parent must bind 20 {category} files, got {len(entries)}')
        for relative, expected in entries.items():
            path = contained_path(root, prefix + relative)
            if not path.is_file() or is_link(path) or sha256(path) != expected:
                raise ValueError(f'Parent-bound {category} bytes do not match: {path}')
            result[prefix + relative] = expected
    return result


def overlay_files(source_root):
    additions, preserved = {}, []
    candidates = list((WORKSPACE / 'scripts/listwise_diffgrm').glob('recovery*.py'))
    candidates += list((WORKSPACE / 'tests').glob('test_listwise*recovery*.py'))
    for path in sorted(candidates):
        relative = path.relative_to(WORKSPACE).as_posix()
        if is_link(path):
            raise ValueError(f'New overlay must not be a link: {path}')
        if (source_root / relative).exists():
            # test_listwise_recovery.py is historical, despite matching the glob.
            preserved.append(relative)
            continue
        additions[relative] = dict(source=str(path), bytes=path.stat().st_size, sha256=sha256(path))
    required = {'scripts/listwise_diffgrm/recovery.py', 'scripts/listwise_diffgrm/recovery_run.py',
                'tests/test_listwise_h1_recovery.py'}
    if not required.issubset(additions):
        raise ValueError(f'Required new recovery overlays are absent: {sorted(required - additions.keys())}')
    return additions, preserved


def git_provenance(paths):
    def run(*arguments):
        return subprocess.run(['git', '-C', str(WORKSPACE), *arguments], check=True,
                              capture_output=True, text=True, encoding='utf-8').stdout.strip()
    commit = run('rev-parse', 'HEAD')
    status = run('status', '--porcelain=v1', '--untracked-files=normal')
    relevant_status = run('status', '--porcelain=v1', '--untracked-files=all', '--', *paths)
    return dict(commit=commit, dirty=bool(status), status_porcelain=status.splitlines(),
                package_sources_uncommitted=bool(relevant_status),
                package_source_status_porcelain=relevant_status.splitlines(),
                commit_alone_identifies_overlay_bytes=not bool(relevant_status))


def verify_original_copy(source_files, execution_root, overlays):
    actual, _ = inventory(execution_root)
    for relative, expected in source_files.items():
        if actual.get(relative) != expected:
            raise ValueError('Refusing to repair/remove an altered original execution file: ' + relative)
    unknown = sorted(set(actual) - set(source_files) - set(overlays))
    if unknown:
        raise ValueError('Execution directory contains files outside original source/new overlays: ' + ', '.join(unknown[:10]))


def pack_archive(output, execution_root, files, manifest_path, binding_path):
    destination = output / 'h1_recovery_code.tar.gz'
    temporary = output / 'h1_recovery_code.tar.gz.tmp'
    members = {f'code/{relative}': (execution_root / relative, row['sha256'])
               for relative, row in files.items()}
    members['code/RECOVERY_EXECUTION_MANIFEST.json'] = (manifest_path, sha256(manifest_path))
    members['code/RECOVERY_PARENT_BINDING.json'] = (binding_path, sha256(binding_path))

    def normalize(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        info.mode &= 0o777
        info.mtime = 0
        if not info.isfile():
            raise ValueError(f'Archive member is not a plain file: {info.name}')
        return info

    with tarfile.open(temporary, mode='w:gz', compresslevel=6) as archive:
        for name, (path, _) in sorted(members.items()):
            archive.add(path, arcname=name, recursive=False, filter=normalize)
    # Independently read each compressed member before replacing the prior package.
    with tarfile.open(temporary, mode='r:gz') as archive:
        rows = archive.getmembers()
        if len(rows) != len(members) or {row.name for row in rows} != set(members):
            raise ValueError('Tar member set differs from the execution manifest')
        for row in rows:
            if not row.isfile() or PurePosixPath(row.name).is_absolute() or '..' in PurePosixPath(row.name).parts:
                raise ValueError(f'Unsafe archive member: {row.name}')
            digest = hashlib.sha256()
            with archive.extractfile(row) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != members[row.name][1]:
                raise ValueError(f'Archive member SHA mismatch: {row.name}')
    temporary.replace(destination)
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha256(destination),
                verified_members=len(members), member_root='code/',
                execution_manifest_sha256=sha256(manifest_path), parent_binding_sha256=sha256(binding_path))


def prepare(output_root):
    output = Path(output_root)
    if not output.is_absolute():
        output = WORKSPACE / output
    output = output.resolve()
    source = (RESTORED / 'code').resolve()
    parent = (RESTORED / 'runs/H1/best.pt').resolve()
    if (output == WORKSPACE or not output.is_relative_to(WORKSPACE)
            or output.is_relative_to(RESTORED.resolve()) or RESTORED.resolve().is_relative_to(output)
            or any(part in {'.git', '.agents', '.codex'} for part in output.relative_to(WORKSPACE).parts)):
        raise ValueError('--output-root must be an independent writable directory within the workspace')
    if sha256(parent) != PARENT_SHA256:
        raise ValueError('Original R20 checkpoint SHA does not match the locked parent')
    import torch
    state = torch.load(parent, map_location='cpu', weights_only=False)
    if (state.get('arm'), state.get('r_updates'), state.get('t_updates')) != ('H1', 13860, 200):
        raise ValueError('Authenticated parent counters/arm are not H1 R20/T200')
    binding = state['binding']
    bound = verify_bound_sources(binding, source)
    del state
    source_files, omissions = inventory(source)
    if not set(bound).issubset(source_files):
        raise ValueError('Packaging exclusions removed a parent-bound source file')
    overlays, preserved = overlay_files(source)
    relevant_paths = [*overlays, 'scripts/analysis/prepare_h1_recovery.py']
    provenance = git_provenance(relevant_paths)
    execution = output / 'code'
    previous_manifest = output / 'EXECUTION_MANIFEST.json'
    if previous_manifest.exists() and json.loads(previous_manifest.read_text(encoding='utf-8')).get('format') != FORMAT:
        raise ValueError('Output contains a manifest belonging to another task')
    if execution.exists():
        if not execution.is_dir() or is_link(execution):
            raise ValueError('Execution target is not an independent regular directory')
        verify_original_copy(source_files, execution, overlays)
    else:
        output.mkdir(parents=True, exist_ok=True)

        def ignore(directory, names):
            return [name for name in names if excluded(Path(directory) / name)]

        shutil.copytree(source, execution, ignore=ignore, copy_function=shutil.copy2)
        verify_original_copy(source_files, execution, overlays)
    # All original bytes passed before any existing overlay is refreshed.
    for relative, record in overlays.items():
        target = contained_path(execution, relative)
        if target.exists() and is_link(target):
            raise ValueError(f'Refusing linked overlay target: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.prepare_tmp')
        shutil.copy2(record['source'], temporary)
        if sha256(temporary) != record['sha256']:
            raise ValueError(f'Overlay changed while it was being copied: {relative}')
        temporary.replace(target)
    verify_bound_sources(binding, execution)
    verify_original_copy(source_files, execution, overlays)
    execution_files, _ = inventory(execution)
    if set(execution_files) != set(source_files) | set(overlays):
        raise ValueError('Execution package has an unexpected file inventory')
    if provenance != git_provenance(relevant_paths):
        raise ValueError('Workspace Git identity/status changed during preparation; rerun for a coherent manifest')
    binding_path = output / 'PARENT_BINDING.json'
    # This file is exactly the checkpoint binding; remote verification needs
    # only json/hashlib and can map engine paths under external/DiffGRM/.
    write_json(binding_path, binding)
    manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(source), execution_root=str(execution),
        parent=dict(path=str(parent), bytes=parent.stat().st_size, sha256=PARENT_SHA256,
                    arm='H1', r_updates=13860, t_updates=200),
        workspace=provenance, parent_bound_source_counts=dict(code=20, engine=20),
        parent_bound_sources_verified=True, original_files_preserved=len(source_files),
        original_source_files=source_files, overlay_files=overlays,
        historical_matching_files_preserved=preserved, excluded_source_entries=omissions,
        exclusion_policy=dict(directories=sorted(EXCLUDED_DIRS), suffixes=sorted(EXCLUDED_SUFFIXES)),
        execution_files=execution_files, execution_file_count=len(execution_files),
        execution_bytes=sum(row['bytes'] for row in execution_files.values()),
        parent_binding_json_sha256=sha256(binding_path),
        manifest_scope='All execution code files; this manifest and parent-binding metadata are separately hashed in TAR_RECEIPT.json',
        preparer=dict(path='scripts/analysis/prepare_h1_recovery.py',
                      sha256=sha256(WORKSPACE / 'scripts/analysis/prepare_h1_recovery.py')))
    write_json(previous_manifest, manifest)
    archive = pack_archive(output, execution, execution_files, previous_manifest, binding_path)
    receipt = dict(format=FORMAT, status='verified', archive=archive,
        parent_sha256=PARENT_SHA256, workspace=provenance,
        execution_file_count=len(execution_files), original_bound_files_verified=40,
        overlay_file_count=len(overlays), source_inputs_modified=False)
    write_json(output / 'TAR_RECEIPT.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args(argv)
    result = prepare(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
