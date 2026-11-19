"""Back up only final generated Stage09 assets at the fixed owned Linux root.

Run with ``python -B backup_generated_assets.py`` on the CPU notebook. Sources
are opened read-only. No network, model loading, GPU access, overwrites, or
deletions occur. On failure an incomplete output is deliberately preserved;
only a completed PASS manifest constitutes a completed backup.
"""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import tarfile


ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
BGE_PREFIX = 'assets/bge_m3_cuda_v1'
PSE_PREFIX = 'assets/pse_v1'
BGE_FILES = ('dense_embeddings.npy', 'original_item_ids.npy',
             'untruncated_token_lengths.npy', 'metadata.json', 'ordered_text.jsonl')
ARCHIVE = 'delivery/stage09_generated_assets_r1.tar'
MANIFEST = 'delivery/STAGE09_GENERATED_ASSETS_R1.json'
CHUNK = 4 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def log(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields), ensure_ascii=False), flush=True)


def relative_parts(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Invalid relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and
            all(part not in ('', '.', '..') for part in path.parts),
            f'Unsafe relative path: {name}')
    return path.parts


def root_fd():
    """Walk from / using directory descriptors; refuse every symlink ancestor."""
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in ROOT.parts[1:]:
            new_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_fixed(name, *, create=False, directory=False):
    """No symlink or traversal even if a source entry changes during enumeration."""
    parts = relative_parts(name)
    fd = root_fd()
    try:
        for part in parts[:-1]:
            new_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        # NONBLOCK lets fstat reject a mistakenly named FIFO without waiting
        # for a writer; it has no effect on the intended ordinary files.
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= os.O_WRONLY | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY
        if directory:
            flags |= os.O_DIRECTORY
        result = os.open(parts[-1], flags, 0o600, dir_fd=fd)
    finally:
        os.close(fd)
    mode = os.fstat(result).st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        os.close(result)
        raise RuntimeError(f'Not an ordinary {"directory" if directory else "file"}: {name}')
    return result


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def enumerate_pse():
    found = []

    def visit(relative):
        fd = open_fixed(relative, directory=True)
        try:
            # Scandir accepts a descriptor on the required Linux platform.
            with os.scandir(fd) as entries:
                rows = sorted(entries, key=lambda entry: entry.name)
            for entry in rows:
                name = relative + '/' + entry.name
                relative_parts(name)
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), f'Refuse source symlink: {name}')
                if stat.S_ISDIR(mode):
                    visit(name)
                else:
                    require(stat.S_ISREG(mode), f'Refuse nonregular source: {name}')
                    found.append(name)
        finally:
            os.close(fd)

    visit(PSE_PREFIX)
    require(PSE_PREFIX + '/manifest.json' in found, 'PSE manifest is missing')
    return sorted(found)


def source_names():
    return sorted([BGE_PREFIX + '/' + name for name in BGE_FILES] + enumerate_pse())


def digest_stream(stream):
    value, count = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(CHUNK), b''):
        value.update(block)
        count += len(block)
    return count, value.hexdigest()


class HashingReader:
    def __init__(self, stream):
        self.stream, self.sha, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.sha.update(block)
        self.count += len(block)
        return block


def verify_archive(records):
    """Re-open the written tar and independently hash every member payload."""
    expected = {row['path']: row for row in records}
    seen = set()
    with os.fdopen(open_fixed(ARCHIVE), 'rb') as source:
        before = os.fstat(source.fileno())
        with tarfile.open(fileobj=source, mode='r:') as archive:
            for member in archive:
                relative_parts(member.name)
                require(member.name in expected and member.name not in seen,
                        f'Unexpected or repeated tar member: {member.name}')
                require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(),
                        f'Nonregular tar member: {member.name}')
                row = expected[member.name]
                require(member.size == row['bytes'], f'Tar size differs: {member.name}')
                with archive.extractfile(member) as stream:
                    require(digest_stream(stream) == (row['bytes'], row['sha256']),
                            f'Tar payload differs: {member.name}')
                seen.add(member.name)
        require(seen == set(expected), 'Tar member set is incomplete')
        source.seek(0)
        result = digest_stream(source)
        require(identity(before) == identity(os.fstat(source.fileno())),
                'Archive changed during verification')
        require(result[0] == before.st_size, 'Archive read length differs')
    return result


def main():
    require(os.name == 'posix' and hasattr(os, 'O_NOFOLLOW'),
            'This fixed-root helper is only for the owned Linux CPU notebook')
    # Refuse even an existing valid output. A failed attempt remains available
    # for inspection; this helper never removes, replaces, or appends to it.
    for name in (ARCHIVE, MANIFEST):
        path = ROOT / name
        require(not os.path.lexists(path), f'Refuse existing output: {name}')
    names = source_names()
    snapshots = {}
    for name in names:
        with os.fdopen(open_fixed(name), 'rb') as stream:
            snapshots[name] = identity(os.fstat(stream.fileno()))
    log('inventory_checked', source_files=len(names), source_bytes=sum(row[2] for row in snapshots.values()))
    records = []
    with os.fdopen(open_fixed(ARCHIVE, create=True), 'wb') as output:
        with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for name in names:
                with os.fdopen(open_fixed(name), 'rb') as source:
                    before = os.fstat(source.fileno())
                    require(identity(before) == snapshots[name], f'Source changed before packing: {name}')
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = before.st_size, 0o600, int(before.st_mtime)
                    reader = HashingReader(source)
                    archive.addfile(info, reader)
                    require(reader.count == before.st_size and
                            identity(before) == identity(os.fstat(source.fileno())),
                            f'Source changed while packing: {name}')
                    records.append(dict(path=name, bytes=reader.count, sha256=reader.sha.hexdigest()))
                    log('file_packed', path=name, bytes=reader.count)
        output.flush()
        os.fsync(output.fileno())
    require(source_names() == names, 'Source file set changed while packing')
    for name in names:
        with os.fdopen(open_fixed(name), 'rb') as source:
            require(identity(os.fstat(source.fileno())) == snapshots[name],
                    f'Source changed since packing: {name}')
    size, sha = verify_archive(records)
    manifest = dict(
        schema='stage09_generated_assets_backup_v1', status='PASS',
        created_at_utc=datetime.now(timezone.utc).isoformat(), remote_root=ROOT.as_posix(),
        archive=dict(path=ARCHIVE, bytes=size, sha256=sha),
        source_policy=dict(bge_final_files=list(BGE_FILES), pse_scope='all_regular_files_recursive',
                           sources_read_only=True, symlinks_allowed=False, overwrite_allowed=False,
                           deletes_performed=False, network_or_gpu_used=False),
        file_count=len(records), payload_bytes=sum(row['bytes'] for row in records),
        files=records, archive_payloads_independently_rehashed=True,
        limitation='A tar without a complete PASS manifest is an incomplete backup; preserve failed outputs.')
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with os.fdopen(open_fixed(MANIFEST, create=True), 'wb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    with os.fdopen(open_fixed(MANIFEST), 'rb') as source:
        require(source.read() == payload, 'Manifest readback differs')
    log('backup_pass', archive=ARCHIVE, archive_bytes=size, archive_sha256=sha,
        manifest=MANIFEST, manifest_sha256=hashlib.sha256(payload).hexdigest(), file_count=len(records))


if __name__ == '__main__':
    main()
