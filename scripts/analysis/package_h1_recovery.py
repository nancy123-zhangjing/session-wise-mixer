"""Package stopped H1 R26 runs without changing inputs or importing torch.

The caller must first confirm both platform jobs have stopped. This tool cannot
check platform state. A receipt certifies byte packaging, not local delivery,
checkpoint restoration, scientific acceptance, or resource release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import uuid


FORMAT = 'stage09_h1_recovery_backup_v1'
PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
EXECUTION_MANIFEST_SHA256 = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
SMALL_FILE_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 64 * 1024 * 1024
READ_BYTES = 4 * 1024 * 1024


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or not path.parts or path.is_absolute() or path.as_posix() != name
            or any(ord(character) < 32 for character in name)
            or any(part in ('', '.', '..') or ':' in part or '\\' in part for part in path.parts)):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def checked_stat(path):
    """Reject links (including Windows junctions) in every path component."""
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Linked path is not allowed: {part}')
    return info


def signature(info):
    # Python 3.12 on Windows reports different ctime meanings for lstat/fstat.
    # Both still agree on identity, size and nanosecond modification time.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns if os.name != 'nt' else None)


def regular_signature(path):
    info = checked_stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Not a regular file: {path}')
    return signature(info)


class StableReader:
    """Hash the exact bytes consumed, checking the opened file and path identity."""
    def __init__(self, path, expected):
        self.path, self.expected = Path(path), expected
        if regular_signature(path) != expected:
            raise ValueError(f'Input changed before read: {path}')
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        self.stream = os.fdopen(fd, 'rb')
        if signature(os.fstat(self.stream.fileno())) != expected:
            self.stream.close()
            raise ValueError(f'Input replaced while opening: {path}')
        self.digest, self.count = hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.count += len(value)
        return value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        opened = signature(os.fstat(self.stream.fileno()))
        self.stream.close()
        if exc_type is None and (opened != self.expected or regular_signature(self.path) != self.expected
                                 or self.count != self.expected[2]):
            raise ValueError(f'Input changed during read: {self.path}')


def read_small(path, expected):
    if expected[2] > SMALL_FILE_BYTES:
        raise ValueError(f'Metadata exceeds small-file limit: {path}')
    with StableReader(path, expected) as reader:
        data = reader.read()
    return data


def digest_file(path, expected):
    with StableReader(path, expected) as reader:
        while reader.read(READ_BYTES):
            pass
    return reader.digest.hexdigest()


def scan_sources(root, parent):
    """Only visit named input trees; never visit root/backup or other runs."""
    selected, omitted, observed = {}, {}, {}
    scopes = [('code', None), ('runs/H1-decay', None), ('runs/H1-freeze', None),
              ('formal_jobs', None), ('delivery', {'.json', '.py'}),
              ('preflight_r1', {'.json', '.jsonl', '.log'})]
    for scope, suffixes in scopes:
        base = root / scope
        if not base.exists() and not base.is_symlink() and scope == 'preflight_r1':
            continue
        if not stat.S_ISDIR(checked_stat(base).st_mode):
            raise ValueError(f'Required input directory missing: {base}')
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs.sort()
            files.sort()
            for child in dirs:
                if not stat.S_ISDIR(checked_stat(Path(directory) / child).st_mode):
                    raise ValueError(f'Non-directory in directory inventory: {child}')
            for filename in files:
                path = Path(directory) / filename
                member = path.relative_to(root).as_posix()
                safe_member(member)
                snapshot = regular_signature(path)
                observed[member] = snapshot
                if suffixes is None or (path.suffix.lower() in suffixes and snapshot[2] <= SMALL_FILE_BYTES):
                    selected[member] = (path, snapshot)
                else:
                    omitted[member] = dict(bytes=snapshot[2], reason='outside_small_metadata_allowlist')
    parent_signature = regular_signature(parent)
    selected['references/R20_best.pt'] = (parent, parent_signature)
    observed['references/R20_best.pt'] = parent_signature
    return selected, omitted, observed


def inspect_inputs(selected):
    def read(name):
        path, snapshot = selected[name]
        data = read_small(path, snapshot)
        return json.loads(data), hashlib.sha256(data).hexdigest()

    execution, execution_sha = read('code/RECOVERY_EXECUTION_MANIFEST.json')
    if (execution_sha != EXECUTION_MANIFEST_SHA256
            or execution['workspace']['package_sources_uncommitted'] is not False
            or execution['parent']['sha256'] != PARENT_SHA256):
        raise ValueError('Execution package provenance does not match this experiment')
    binding, binding_sha = read('code/RECOVERY_PARENT_BINDING.json')
    if binding_sha != execution['parent_binding_json_sha256']:
        raise ValueError('Parent binding SHA mismatch')
    parent_path, parent_snapshot = selected['references/R20_best.pt']
    if (digest_file(parent_path, parent_snapshot) != PARENT_SHA256
            or parent_snapshot[2] != execution['parent']['bytes']):
        raise ValueError('R20 parent bytes/SHA mismatch')
    endpoints = {}
    for policy, t_updates in [('decay', 280), ('freeze', 200)]:
        complete, complete_sha = read(f'runs/H1-{policy}/complete.json')
        expected = dict(policy=policy, r_updates=18018, t_updates=t_updates, phase=33, endpoint_epoch=26)
        if any(type(complete.get(key)) is not type(value) or complete[key] != value for key, value in expected.items()):
            raise ValueError(f'H1-{policy} completion is not the expected R26 endpoint')
        checkpoint, snapshot = selected[f'runs/H1-{policy}/last.pt']
        last_sha = digest_file(checkpoint, snapshot)
        if last_sha != complete['endpoint_checkpoint_sha256']:
            raise ValueError(f'H1-{policy} completion/checkpoint SHA mismatch')
        endpoints[policy] = dict(**expected, complete_sha256=complete_sha, last_sha256=last_sha)
    dependencies = dict(format=FORMAT, self_contained_dataset=False, payload_files_included=False,
        statement='Dataset and content PSE payloads are external dependencies already backed up by the caller. '
                  'This packager records original binding identities; it does not verify that local backup. '
                  'Independent extraction/SHA verification and CPU checkpoint restoration are still required.',
        dataset=dict(original_path=binding['config']['dataset'], binding=binding['data']),
        content_pse=dict(original_path=binding['config']['assets'], manifest_sha256=binding['asset_manifest']),
        parent_binding_member='code/RECOVERY_PARENT_BINDING.json', parent_binding_sha256=binding_sha)
    return execution, binding, endpoints, dependencies


def add_bytes(archive, name, data):
    safe_member(name)
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    archive.addfile(info, io.BytesIO(data))
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), source='generated_metadata')


def verify_code(records, execution, binding):
    expected_files = execution['execution_files']
    metadata = {'RECOVERY_EXECUTION_MANIFEST.json', 'RECOVERY_PARENT_BINDING.json'}
    actual_files = {name.removeprefix('code/') for name in records if name.startswith('code/')}
    if (type(execution.get('execution_file_count')) is not int
            or execution['execution_file_count'] != len(expected_files)
            or actual_files != set(expected_files) | metadata):
        raise ValueError('Execution code inventory differs from its manifest')
    for name, expected in expected_files.items():
        safe_member(name)
        actual = records['code/' + name]
        if actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
            raise ValueError(f'Execution source differs from its manifest: {name}')
    for group, prefix in [('code', 'code/'), ('engine', 'code/external/DiffGRM/')]:
        for name, expected in binding[group].items():
            safe_member(name)
            if records[prefix + name]['sha256'] != expected:
                raise ValueError(f'Original parent-bound source differs: {name}')


def archive_hashes(path, chunk_bytes):
    """Byte ranges permit single-file resumable downloads without duplicate parts."""
    chunks, overall, offset = [], hashlib.sha256(), 0
    snapshot = regular_signature(path)
    with StableReader(path, snapshot) as stream:
        while offset < snapshot[2]:
            length = min(chunk_bytes, snapshot[2] - offset)
            block_hash, remaining = hashlib.sha256(), length
            while remaining:
                block = stream.read(min(READ_BYTES, remaining))
                if not block:
                    raise ValueError('Unexpected EOF while hashing archive')
                overall.update(block)
                block_hash.update(block)
                remaining -= len(block)
            chunks.append(dict(index=len(chunks), offset=offset, bytes=length, sha256=block_hash.hexdigest()))
            offset += length
    return dict(bytes=offset, sha256=overall.hexdigest(), chunk_bytes=chunk_bytes, chunks=chunks)


def package(root, parent, *, name=None, chunk_bytes=CHUNK_BYTES):
    root, parent = Path(os.path.abspath(root)), Path(os.path.abspath(parent))
    if not stat.S_ISDIR(checked_stat(root).st_mode):
        raise ValueError('Root must be a regular directory')
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError('Chunk bytes must be a positive integer')
    name = name or ('package_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or name in ('.', '..'):
        raise ValueError('Backup name must be one safe directory component')
    selected, omitted, before = scan_sources(root, parent)
    execution, binding, endpoints, dependencies = inspect_inputs(selected)
    backup = root / 'backup'
    backup.mkdir(exist_ok=True)
    if not stat.S_ISDIR(checked_stat(backup).st_mode):
        raise ValueError('Backup parent must be a regular directory')
    output = backup / name
    output.mkdir(exist_ok=False)
    partial = output / 'archive.tar.gz.partial'
    records = {}
    with partial.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for member, (path, snapshot) in sorted(selected.items()):
                    info = tarfile.TarInfo(member)
                    info.size, info.mode, info.mtime = snapshot[2], stat.S_IMODE(path.stat().st_mode) & 0o777, snapshot[3] // 10**9
                    with StableReader(path, snapshot) as stream:
                        archive.addfile(info, stream)
                    records[member] = dict(bytes=snapshot[2], sha256=stream.digest.hexdigest(),
                                           source=str(path), source_mtime_ns=snapshot[3])
                verify_code(records, execution, binding)
                for policy, endpoint in endpoints.items():
                    if records[f'runs/H1-{policy}/last.pt']['sha256'] != endpoint['last_sha256']:
                        raise ValueError(f'H1-{policy} checkpoint changed after initial verification')
                if records['references/R20_best.pt']['sha256'] != PARENT_SHA256:
                    raise ValueError('Parent checkpoint changed after initial verification')
                records['references/DEPENDENCIES.json'] = add_bytes(archive, 'references/DEPENDENCIES.json', json_bytes(dependencies))
                if scan_sources(root, parent)[2] != before:
                    raise ValueError('Input inventory changed during packaging')
                manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
                    source_root=str(root), parent_source=str(parent), files=records, file_count=len(records),
                    payload_bytes=sum(row['bytes'] for row in records.values()), endpoints=endpoints,
                    execution_manifest_sha256=EXECUTION_MANIFEST_SHA256, omitted_files=omitted,
                    metadata_limit_bytes=SMALL_FILE_BYTES, self_contained_dataset=False,
                    manifest_scope='All regular archive files except MANIFEST.json itself; its SHA is in ARCHIVE_RECEIPT.json.',
                    platform_stopped_verified_here=False, source_inputs_modified=False,
                    local_delivery_verified=False, checkpoint_restore_verified=False)
                manifest_data = json_bytes(manifest)
                add_bytes(archive, 'MANIFEST.json', manifest_data)
        raw.flush()
        os.fsync(raw.fileno())
    hashes = archive_hashes(partial, chunk_bytes)
    if scan_sources(root, parent)[2] != before:
        raise ValueError('Input inventory changed before backup receipt')
    archive_path = output / 'archive.tar.gz'
    partial.rename(archive_path)
    with (output / 'MANIFEST.json').open('xb') as stream:
        stream.write(manifest_data)
    receipt = dict(format=FORMAT, status='PACKAGED_BYTES_NOT_LOCAL_DELIVERY_OR_RESTORE',
        created_utc=datetime.now(timezone.utc).isoformat(), archive=archive_path.name, **hashes,
        manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), source_root=str(root),
        input_inventory_stable=True, source_inputs_modified=False, platform_stopped_verified_here=False,
        local_delivery_verified=False, checkpoint_restore_verified=False,
        chunk_scope='Contiguous byte ranges of archive.tar.gz, including the final shorter range; no separate chunk files.')
    with (output / 'ARCHIVE_RECEIPT.json').open('xb') as stream:
        stream.write(json_bytes(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    return dict(output=str(output), **receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--name', help='New directory name under root/backup; never reuses an existing output')
    parser.add_argument('--chunk-mib', type=int, default=64)
    args = parser.parse_args(argv)
    result = package(args.root, args.parent, name=args.name, chunk_bytes=args.chunk_mib * 1024 * 1024)
    print(json.dumps({key: value for key, value in result.items() if key != 'chunks'}, indent=2))


if __name__ == '__main__':
    main()
