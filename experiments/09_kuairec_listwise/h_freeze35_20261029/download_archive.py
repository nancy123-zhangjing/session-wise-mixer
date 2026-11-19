"""Resume an owned freeze35 tar after ALL other writers/scp have exited.

Prefix SHA -> authenticated 8 MiB blocks -> full SHA and embedded manifest SHA.
Only byte transfer is accepted here: extraction, metric audit and model restore
remain separate. No remote writes, workload creation or automatic prefix reset.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

REMOTE_ROOT = PurePosixPath('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
LOCAL_ROOT = Path(__file__).absolute().parents[3] / 'outputs' / 'stage09_restore'
REMOTE_PY = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
MARKER = 'FREEZE35_ARCHIVE_PACKET '
CHUNK = 8 * 1024 * 1024
EMPTY_SHA = hashlib.sha256(b'').hexdigest()
SCOPE = 'byte transfer only; extraction, full backup coverage, metrics and model restore NOT verified'


def require(test, message):
    if not test: raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def log(stage, **values):
    # No raw CLI stdout/stderr, packet data, source command, paths or tokens.
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **values)), flush=True)


def no_links(path):
    path = Path(os.path.abspath(path))
    for node in [*reversed(path.parents), path]:
        if not node.exists() and not node.is_symlink(): continue
        info = node.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse local path is forbidden')
    return path


def paths(remote, output):
    value = PurePosixPath(remote)
    require(value.as_posix() == remote and value.is_absolute() and '\\' not in remote,
            'Remote path must be canonical POSIX absolute')
    relative = value.relative_to(REMOTE_ROOT)
    require(relative.parts and relative.parts[0].startswith('h_freeze35_20261029')
            and value.suffix == '.tar' and all(re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', p)
            and p not in ('.', '..') for p in relative.parts), 'Remote archive is outside this Goal export scope')
    local = no_links(output); limit = no_links(LOCAL_ROOT)
    require(local.is_relative_to(limit) and local != limit and local.suffix == '.tar'
            and not any(ord(c) < 32 for c in str(local)), 'Local tar must remain under outputs/stage09_restore')
    if local.exists():
        info = local.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Local output must be one ordinary unlinked file')
    return value, local


def identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def local_state(path, stream):
    no_links(path)
    value = identity(path.stat())
    opened = identity(os.fstat(stream.fileno()))
    # Windows CRT fstat reports legacy ctime differently from stat-by-path.
    # Compare dev/inode/size/mtime; retain path ctime in each boundary snapshot.
    require(value[:4] == opened[:4] and (os.name == 'nt' or value == opened) and path.stat().st_nlink == 1,
            'Local file was replaced/linked by another writer')
    return value


def open_exclusive(path):
    """Windows denies any other write handle; POSIX flock is advisory only."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        create = ctypes.WinDLL('kernel32', use_last_error=True).CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        handle = create(str(path), 0xC0000000, 1, None, 3 if path.exists() else 1, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise OSError('Exclusive writer access denied; wait for scp/other writers to EXIT')
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            ctypes.WinDLL('kernel32').CloseHandle(handle)
            raise
        return os.fdopen(fd, 'r+b')
    import fcntl
    stream = path.open('r+b' if path.exists() else 'x+b')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except BaseException:
        stream.close(); raise


def digest_stream(stream):
    stream.seek(0); digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def manifest_digest(stream):
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode='r:') as tar:
        members = [m for m in tar if m.name == 'DELIVERY_MANIFEST.json']
        require(len(members) == 1 and members[0].isreg() and not members[0].sparse
                and members[0].size <= 32 * 1024 * 1024, 'Missing/unsafe/duplicate embedded manifest')
        return hashlib.sha256(tar.extractfile(members[0]).read()).hexdigest()


def request(args, source):
    command = [shutil.which('inspire') or 'inspire', '--account', args.account, 'notebook', 'ssh', args.notebook,
        '--workspace', 'CPU资源空间', '--command-timeout', '150', '--timeout', '60', '--', REMOTE_PY, '-B', '-c', source]
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            start_new_session=os.name != 'nt')
    try:
        stdout, _ = proc.communicate(timeout=225)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate(timeout=20)
        raise TimeoutError('Owned block reader timed out; private output withheld') from None
    require(proc.returncode == 0, 'CLI read failed; private output withheld')
    require(len(stdout) <= 12 * 1024 * 1024, 'CLI packet exceeds the explicit output bound')
    rows = [line[len(MARKER):] for line in stdout.decode('utf-8').splitlines() if line.startswith(MARKER)]
    require(len(rows) == 1, 'Exactly one complete hash-framed packet required')
    return decode(rows[0])


def remote_common(remote, total):
    return f'''from pathlib import Path
import base64,hashlib,json,stat,tarfile
p=Path({str(remote)!r})
assert str(p.resolve())==str(p) and all(not x.is_symlink() for x in [p,*p.parents])
before=p.stat()
assert stat.S_ISREG(before.st_mode) and before.st_size=={total}
def ident(s): return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
'''


def prefix_source(common, args, offset, prefix):
    return common + f'''whole=hashlib.sha256(); prefix=hashlib.sha256(); count=0
with p.open('rb') as stream:
 for block in iter(lambda:stream.read(4194304),b''):
  whole.update(block)
  if count < {offset}: prefix.update(block[:max(0,{offset}-count)])
  count+=len(block)
with tarfile.open(p,'r:') as tar:
 members=[m for m in tar if m.name=='DELIVERY_MANIFEST.json']
 assert len(members)==1 and members[0].isreg() and not members[0].sparse and members[0].size<=33554432
 manifest_sha=hashlib.sha256(tar.extractfile(members[0]).read()).hexdigest()
assert ident(before)==ident(p.stat())
assert whole.hexdigest()=={args.sha256!r} and prefix.hexdigest()=={prefix!r} and manifest_sha=={args.manifest_sha256!r}
print({MARKER!r}+json.dumps(dict(status='prefix_verified',offset={offset},bytes=count,sha256=whole.hexdigest(),prefix_sha256=prefix.hexdigest(),manifest_sha256=manifest_sha,source_stat=ident(before))))
'''


def block_source(common, args, offset, size, source_stat):
    return common + f'''assert ident(before)=={source_stat!r}
with p.open('rb') as stream:
 stream.seek({offset}); data=stream.read({size})
assert len(data)=={size} and ident(before)==ident(p.stat())
print({MARKER!r}+json.dumps(dict(status='block',offset={offset},bytes=len(data),archive_sha256={args.sha256!r},manifest_sha256={args.manifest_sha256!r},source_stat=ident(before),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())))
'''


def unpack_block(packet, args, offset, size, source_stat):
    require(set(packet) == {'status', 'offset', 'bytes', 'archive_sha256', 'manifest_sha256', 'source_stat', 'sha256', 'data'},
            'Unexpected packet fields')
    require(packet['status'] == 'block' and packet['offset'] == offset and packet['bytes'] == size
            and packet['archive_sha256'] == args.sha256 and packet['manifest_sha256'] == args.manifest_sha256
            and packet['source_stat'] == source_stat, 'Block offset/source identity mismatch')
    data = base64.b64decode(packet['data'], validate=True)
    require(len(data) == size and hashlib.sha256(data).hexdigest() == packet['sha256'], 'Block content SHA mismatch')
    return data


def transfer(args):
    require(args.other_writers_stopped, 'Explicit confirmation required: wait until scp and ALL other writers EXIT')
    for digest in (args.sha256, args.manifest_sha256):
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Expected SHA must be lowercase 64-hex')
    require(args.bytes > 0, 'Positive archive byte count required')
    remote, local = paths(args.remote, args.output)
    local.parent.mkdir(parents=True, exist_ok=True); no_links(local)
    lock = no_links(str(local) + '.download.lock')
    receipt = no_links(str(local) + '.DOWNLOAD_BYTES_VERIFIED.json')
    journal = no_links(str(local) + '.download.jsonl')
    require(not receipt.exists(), 'A completed receipt exists; inspect it instead of overwriting')
    binding = dict(remote=str(remote), local=str(local), bytes=args.bytes, sha256=args.sha256,
                   manifest_sha256=args.manifest_sha256, chunk_bytes=CHUNK)
    owner = dict(pid=os.getpid(), token=os.urandom(16).hex())
    with lock.open('xb') as stream: stream.write(encoded(owner))
    try:
        if journal.exists():
            require(journal.is_file() and journal.stat().st_nlink == 1, 'Unsafe download journal')
            with journal.open('rb') as stream: require(decode(stream.readline()) == dict(stage='binding', **binding), 'Resume journal identity differs')
        else:
            with journal.open('xb') as stream: stream.write(encoded(dict(stage='binding', **binding)))
        with open_exclusive(local) as stream:
            state = local_state(local, stream); offset = state[2]; initial = offset
            require(offset <= args.bytes, 'Existing prefix exceeds expected archive; NEVER auto-reset')
            prefix = digest_stream(stream)
            require(local_state(local, stream) == state, 'Concurrent writer during prefix hash')
            common = remote_common(remote, args.bytes)
            proof = request(args, prefix_source(common, args, offset, prefix))
            expected = dict(status='prefix_verified', offset=offset, bytes=args.bytes, sha256=args.sha256,
                            prefix_sha256=prefix, manifest_sha256=args.manifest_sha256)
            require({k: v for k, v in proof.items() if k != 'source_stat'} == expected
                    and isinstance(proof.get('source_stat'), list) and len(proof['source_stat']) == 5,
                    'Remote prefix/full-file/manifest authentication differs')
            source_stat = proof['source_stat']; started = time.monotonic()
            require(local_state(local, stream) == state, 'Another writer changed prefix during remote verification')
            log('prefix_verified', offset=offset, total=args.bytes)
            while offset < args.bytes:
                size = min(CHUNK, args.bytes - offset)
                require(local_state(local, stream) == state, 'Concurrent writer before block request')
                for attempt in range(1, 4):
                    try:
                        packet = request(args, block_source(common, args, offset, size, source_stat))
                        data = unpack_block(packet, args, offset, size, source_stat)
                        break
                    except Exception as error:
                        log('read_failed_no_append', offset=offset, attempt=attempt, error_type=type(error).__name__)
                        if attempt == 3: raise
                require(local_state(local, stream) == state, 'Concurrent writer while reading remote block')
                stream.seek(0, os.SEEK_END); require(stream.tell() == offset, 'Unexpected append offset')
                require(stream.write(data) == size, 'Short block write; preserve prefix and stop')
                stream.flush(); os.fsync(stream.fileno()); offset += size
                state = local_state(local, stream); require(state[2] == offset, 'Append length differs')
                with journal.open('ab') as audit:
                    audit.write(encoded(dict(stage='block_verified', offset=packet['offset'], bytes=size,
                                             sha256=packet['sha256'], end=offset)))
                    audit.flush(); os.fsync(audit.fileno())
                log('block_verified', downloaded=offset, total=args.bytes, elapsed_seconds=round(time.monotonic()-started, 2))
            require(digest_stream(stream) == args.sha256 and manifest_digest(stream) == args.manifest_sha256,
                    'Final archive/embedded manifest SHA mismatch; do NOT reset or claim success')
            require(local_state(local, stream) == state, 'Another writer changed final file')
            result = dict(format='freeze35_resumable_byte_transfer_v1', status='BYTES_VERIFIED_ONLY', scope=SCOPE,
                **binding, resumed_from_bytes=initial, prefix_sha256=prefix, source_stat=source_stat,
                full_archive_and_embedded_manifest_sha256_verified=True, block_sha256_verified=True,
                caller_confirmed_other_writers_stopped=True,
                concurrency='Windows denies other write handles; POSIX advisory flock plus per-block stat checks',
                gpu_used=0, extraction_verified=False, metric_audit=False, model_restore=False,
                finished_at_utc=datetime.now(timezone.utc).isoformat())
            with receipt.open('xb') as audit: audit.write(encoded(result))
        log('BYTES_VERIFIED_ONLY', bytes=args.bytes, resumed_from_bytes=initial)
        return result
    finally:
        # Only this invocation's NEW owned sidecar; never delete/reset the tar.
        if lock.exists() and decode(lock.read_bytes()) == owner: lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--bytes', required=True, type=int); parser.add_argument('--sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--other-writers-stopped', action='store_true', help='REQUIRED: wait for scp/other writers to EXIT first')
    parser.add_argument('--account', default='sii-shared'); parser.add_argument('--notebook', default='nancy-stage09-prep-20261027-r1')
    args = parser.parse_args(argv)
    env_no_proxy = [p for p in os.environ.get('NO_PROXY', '').split(',') if p]
    os.environ['NO_PROXY'] = ','.join(dict.fromkeys([*env_no_proxy, '.sii.edu.cn', '<institution-host>']))
    return transfer(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('interrupted_prefix_preserved'); sys.exit(130)
    except Exception as error:
        log('failed_prefix_preserved', error_type=type(error).__name__); sys.exit(1)
