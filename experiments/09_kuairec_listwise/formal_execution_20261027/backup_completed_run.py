"""Pack one completed owned Stage09 run, or independently restore its local copy.

No model loading, network, GPU, deletion, source mutation or output overwrite.
Pack is fixed-root Linux only. Restore writes a new runs/<run> directory under
this workspace's outputs/stage09_restore/<restore-name>. Failed partial outputs
are retained; only a PASS manifest/receipt establishes completion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tarfile

REMOTE_ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
RUNS = ('warmup_J', 'warmup_H', 'J0', 'J1', 'H0', 'H1')
CHUNK = 4 * 1024 * 1024
SCHEMA = 'stage09_completed_run_backup_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def log(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields), ensure_ascii=False), flush=True)


def parts(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Invalid archive-relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and all(part not in ('', '.', '..') for part in path.parts), 'Unsafe relative path: ' + name)
    return path.parts


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(stream):
    sha, count = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(CHUNK), b''):
        sha.update(block)
        count += len(block)
    return count, sha.hexdigest()


def filenames(run):
    require(run in RUNS, 'Run is not in the explicit whitelist')
    return ('delivery/stage09_run_' + run + '_r1.tar',
            'delivery/STAGE09_RUN_' + run + '_R1.json')


def open_remote(name, *, create=False, directory=False):
    """Walk every fixed-root ancestor through no-follow directory descriptors."""
    relative = parts(name)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in REMOTE_ROOT.parts[1:] + relative[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = following
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= os.O_WRONLY | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY
        if directory:
            flags |= os.O_DIRECTORY
        result = os.open(relative[-1], flags, 0o600, dir_fd=fd)
    finally:
        os.close(fd)
    info = os.fstat(result)
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        os.close(result)
        raise RuntimeError('Not an ordinary file/directory: ' + name)
    return result


def remote_exists(name):
    relative = parts(name)
    fd = open_remote('/'.join(relative[:-1]), directory=True)
    try:
        try:
            os.stat(relative[-1], dir_fd=fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False
    finally:
        os.close(fd)


def completion_guard(run):
    require(not remote_exists('runs/' + run + '/.running'), 'Run still has a .running entry')
    with os.fdopen(open_remote('runs/' + run + '/complete.json'), 'rb') as stream:
        complete = json.load(stream)
    expected = 'warmup' if run.startswith('warmup_') else run
    require(isinstance(complete, dict) and complete.get('arm') == expected,
            'Completion receipt missing or belongs to another run')
    return complete


def remote_files(run):
    found = []
    def visit(relative):
        fd = open_remote(relative, directory=True)
        try:
            with os.scandir(fd) as entries:
                rows = sorted(entries, key=lambda entry: entry.name)
            for entry in rows:
                name = relative + '/' + entry.name
                parts(name)
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), 'Refuse source symlink: ' + name)
                if stat.S_ISDIR(mode):
                    visit(name)
                else:
                    require(stat.S_ISREG(mode), 'Refuse nonregular source: ' + name)
                    found.append(name)
        finally:
            os.close(fd)
    visit('runs/' + run)
    return sorted(found)


def hash_remote(name):
    with os.fdopen(open_remote(name), 'rb') as stream:
        before = os.fstat(stream.fileno())
        count, sha = digest(stream)
        require(identity(before) == identity(os.fstat(stream.fileno())) and count == before.st_size,
                'File changed while hashing: ' + name)
    return dict(path=name, bytes=count, sha256=sha), identity(before)


class HashingReader:
    def __init__(self, stream):
        self.stream, self.sha, self.count = stream, hashlib.sha256(), 0

    def read(self, size):
        require(0 <= size <= CHUNK, 'Tar reader requested an unbounded buffer')
        block = self.stream.read(size)
        self.sha.update(block)
        self.count += len(block)
        return block


def validate_member(member, expected, seen):
    parts(member.name)
    require(member.name in expected and member.name not in seen,
            'Unexpected or repeated archive member: ' + member.name)
    require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(),
            'Refuse links, directories, sparse or special tar members: ' + member.name)
    require(member.size == expected[member.name]['bytes'], 'Archive member size differs: ' + member.name)


def verify_tar(stream, records):
    expected, seen = {row['path']: row for row in records}, set()
    with tarfile.open(fileobj=stream, mode='r:') as archive:
        for member in archive:
            validate_member(member, expected, seen)
            with archive.extractfile(member) as payload:
                require(digest(payload) == (expected[member.name]['bytes'], expected[member.name]['sha256']),
                        'Archive payload SHA differs: ' + member.name)
            seen.add(member.name)
    require(seen == set(expected), 'Archive member set incomplete')


def pack(run):
    require(os.name == 'posix' and hasattr(os, 'O_NOFOLLOW'), 'Pack is fixed-root Linux only')
    archive_name, manifest_name = filenames(run)
    for name in (archive_name, manifest_name):
        require(not remote_exists(name), 'Refuse existing backup output: ' + name)
    complete = completion_guard(run)
    names = remote_files(run)
    require('runs/' + run + '/complete.json' in names, 'No completion marker in inventory')
    records, snapshots = [], {}
    for name in names:
        row, snapshot = hash_remote(name)
        records.append(row)
        snapshots[name] = snapshot
        log('source_prehashed', path=name, bytes=row['bytes'])
    completion_guard(run)
    require(remote_files(run) == names, 'Source inventory changed before packing')
    with os.fdopen(open_remote(archive_name, create=True), 'wb') as output:
        with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for row in records:
                name = row['path']
                with os.fdopen(open_remote(name), 'rb') as source:
                    before = os.fstat(source.fileno())
                    require(identity(before) == snapshots[name], 'Source changed before tar read: ' + name)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = row['bytes'], 0o600, int(before.st_mtime)
                    reader = HashingReader(source)
                    archive.addfile(info, reader)
                    require((reader.count, reader.sha.hexdigest()) == (row['bytes'], row['sha256'])
                            and identity(os.fstat(source.fileno())) == snapshots[name],
                            'Source changed during tar read: ' + name)
                log('file_packed', path=name, bytes=row['bytes'])
        output.flush()
        os.fsync(output.fileno())
    # Independently rehash all original payloads, not just their stat metadata.
    require(completion_guard(run) == complete and remote_files(run) == names,
            'Completion state or source file set changed during packing')
    for expected in records:
        actual, snapshot = hash_remote(expected['path'])
        require(actual == expected and snapshot == snapshots[expected['path']],
                'Source changed after packing: ' + expected['path'])
    with os.fdopen(open_remote(archive_name), 'rb') as stream:
        before = os.fstat(stream.fileno())
        verify_tar(stream, records)
        stream.seek(0)
        archive_bytes, archive_sha = digest(stream)
        require(identity(before) == identity(os.fstat(stream.fileno())) and archive_bytes == before.st_size,
                'Archive changed during independent verification')
    require(completion_guard(run) == complete and remote_files(run) == names,
            'Run restarted or inventory changed before publishing manifest')
    # Check metadata again after archive verification, in addition to post-pack SHA pass.
    for row in records:
        with os.fdopen(open_remote(row['path']), 'rb') as stream:
            require(identity(os.fstat(stream.fileno())) == snapshots[row['path']],
                    'Source changed before manifest publication')
    manifest = dict(schema=SCHEMA, status='PASS', run=run,
        created_at_utc=datetime.now(timezone.utc).isoformat(), remote_root=REMOTE_ROOT.as_posix(),
        source_prefix='runs/' + run, complete=complete,
        archive=dict(path=archive_name, bytes=archive_bytes, sha256=archive_sha),
        file_count=len(records), payload_bytes=sum(row['bytes'] for row in records), files=records,
        source_sha_before_during_after_equal=True, archive_payloads_independently_rehashed=True,
        policy=dict(source_read_only=True, symlinks_allowed=False, overwrite_allowed=False,
                    checkpoint_loaded=False, network_or_gpu_used=False, manifest_outside_tar=True),
        limitation='Filesystem backup only, not checkpoint semantics or local-delivery acceptance.')
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with os.fdopen(open_remote(manifest_name, create=True), 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with os.fdopen(open_remote(manifest_name), 'rb') as stream:
        require(stream.read() == payload, 'Manifest readback differs')
    log('backup_pass', run=run, archive=archive_name, archive_bytes=archive_bytes,
        archive_sha256=archive_sha, manifest=manifest_name,
        manifest_sha256=hashlib.sha256(payload).hexdigest(), files=len(records))


def reject_links(path):
    """Reject existing symlinks and Windows junction/reparse ancestors."""
    path = Path(os.path.abspath(path))
    for entry in reversed((path, *path.parents)):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
                'Refuse link/reparse path: ' + str(entry))
    return path


def open_local(path, *, create=False):
    path = reject_links(path)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL) if create else os.O_RDONLY
    flags |= getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError('Local path is not an ordinary file: ' + str(path))
    return os.fdopen(fd, 'wb' if create else 'rb')


def validate_manifest(manifest, run):
    archive_name, _ = filenames(run)
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'PASS'
            and manifest.get('run') == run and manifest.get('source_prefix') == 'runs/' + run,
            'Wrong backup manifest schema/status/run')
    require(manifest.get('remote_root') == '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1',
            'Manifest belongs to another Goal root')
    archive = manifest['archive']
    require(archive['path'] == archive_name and type(archive['bytes']) is int and archive['bytes'] > 0
            and bool(re.fullmatch('[0-9a-f]{64}', archive['sha256'])), 'Invalid archive metadata')
    expected = {}
    for row in manifest['files']:
        name = row['path']
        components = parts(name)
        require(components[:2] == ('runs', run) and len(components) > 2, 'Member is outside selected run')
        require(all(not PureWindowsPath(part).is_reserved() and not part.endswith((' ', '.')) for part in components),
                'Member is not safely portable to Windows')
        require(name not in expected and type(row['bytes']) is int and row['bytes'] >= 0
                and bool(re.fullmatch('[0-9a-f]{64}', row['sha256'])), 'Invalid/duplicate file record')
        expected[name] = row
    require('runs/' + run + '/complete.json' in expected and 'runs/' + run + '/.running' not in expected,
            'Manifest lacks completion or includes a live-run lock')
    require(manifest['file_count'] == len(expected)
            and manifest['payload_bytes'] == sum(row['bytes'] for row in expected.values()), 'Inventory totals differ')
    return expected


def restore(run, archive_path, manifest_path, manifest_sha, restore_name):
    require(bool(re.fullmatch('[0-9a-f]{64}', manifest_sha)), 'Supply SHA from the remote PASS receipt')
    require(bool(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', restore_name)), 'Unsafe restore directory name')
    script = Path(__file__).resolve()
    workspace = script.parents[3]
    allowed_helper_directories = (
        workspace / 'outputs/stage09_delivery/goal_20261027_r1',
        workspace / 'experiments/09_kuairec_listwise/formal_execution_20261027',
    )
    require(script.parent in allowed_helper_directories and (workspace / '.git').exists(),
            'Run local restore from an allowlisted helper directory in the original Git workspace')
    destination = reject_links(workspace / 'outputs/stage09_restore' / restore_name)
    target = destination / 'runs' / run
    receipt = destination / ('RESTORE_' + run + '.json')
    require(not os.path.lexists(target) and not os.path.lexists(receipt), 'Refuse existing restore target/receipt')
    with open_local(manifest_path) as stream:
        before = os.fstat(stream.fileno())
        count, sha = digest(stream)
        require(sha == manifest_sha and count == before.st_size, 'Manifest SHA/size mismatch')
        stream.seek(0)
        manifest = json.load(stream)
        require(identity(before) == identity(os.fstat(stream.fileno())), 'Manifest changed during read')
    expected = validate_manifest(manifest, run)
    with open_local(archive_path) as source:
        before = os.fstat(source.fileno())
        require(digest(source) == (manifest['archive']['bytes'], manifest['archive']['sha256']),
                'Downloaded archive SHA/size mismatch')
        source.seek(0)
        verify_tar(source, list(expected.values()))
        require(identity(before) == identity(os.fstat(source.fileno())), 'Archive changed during pre-verification')
        # No extraction occurs until both whole archive and every tar payload pass.
        reject_links(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        source.seek(0)
        seen = set()
        with tarfile.open(fileobj=source, mode='r:') as archive:
            for member in archive:
                validate_member(member, expected, seen)
                path = destination.joinpath(*parts(member.name))
                require(path.resolve().is_relative_to(target.resolve()), 'Extraction escapes fresh run directory')
                reject_links(path.parent)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as payload, open_local(path, create=True) as output:
                    sha, count = hashlib.sha256(), 0
                    for block in iter(lambda: payload.read(CHUNK), b''):
                        output.write(block)
                        sha.update(block)
                        count += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                require((count, sha.hexdigest()) == (expected[member.name]['bytes'], expected[member.name]['sha256']),
                        'Extracted payload differs: ' + member.name)
                seen.add(member.name)
        require(seen == set(expected) and identity(before) == identity(os.fstat(source.fileno())),
                'Archive/member set changed during extraction')
    actual = set()
    for folder, dirs, files in os.walk(target, followlinks=False):
        for name in dirs:
            reject_links(Path(folder) / name)
        for name in files:
            path = reject_links(Path(folder) / name)
            relative = path.relative_to(destination).as_posix()
            require(relative in expected, 'Unexpected restored file')
            with open_local(path) as stream:
                row = expected[relative]
                require(digest(stream) == (row['bytes'], row['sha256']), 'Independent restored-file SHA differs')
            actual.add(relative)
    require(actual == set(expected), 'Restored file set incomplete')
    result = dict(schema='stage09_completed_run_local_restore_v1', status='PASS', run=run,
        created_at_utc=datetime.now(timezone.utc).isoformat(), destination=str(target),
        source_archive=str(Path(archive_path).absolute()), source_manifest=str(Path(manifest_path).absolute()),
        manifest_sha256=manifest_sha, archive=manifest['archive'], file_count=len(expected),
        payload_bytes=manifest['payload_bytes'], all_files_independently_rehashed=True,
        checkpoint_loaded=False, network_or_gpu_used=False,
        limitation='Byte-for-byte restore only. Model/state semantic acceptance remains separate.')
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with open_local(receipt, create=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with open_local(receipt) as stream:
        require(stream.read() == payload, 'Restore receipt readback differs')
    log('local_restore_pass', run=run, destination=str(target), receipt=str(receipt),
        receipt_sha256=hashlib.sha256(payload).hexdigest(), files=len(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack', help='Linux CPU: archive one completed run at the fixed Goal root')
    packing.add_argument('--run', choices=RUNS, required=True)
    restoring = commands.add_parser('restore', help='Local CPU: verify and extract into a new run directory')
    restoring.add_argument('--run', choices=RUNS, required=True)
    restoring.add_argument('--archive', type=Path, required=True)
    restoring.add_argument('--manifest', type=Path, required=True)
    restoring.add_argument('--manifest-sha256', required=True)
    restoring.add_argument('--restore-name', required=True,
                           help='Directory name below outputs/stage09_restore; selected runs/<run> must not exist')
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.run)
    else:
        restore(args.run, args.archive, args.manifest, args.manifest_sha256, args.restore_name)


if __name__ == '__main__':
    main()
