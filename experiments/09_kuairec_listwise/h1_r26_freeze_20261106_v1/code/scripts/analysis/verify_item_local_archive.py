"""Verify and safely extract Stage07 delivery bytes using only local CPU/files.

PASS means archive, manifest, receipt and extracted bytes agree. It does not
validate a model, recommendation metrics, GPU release or completion of a Goal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import tarfile


CHUNK = 1024 * 1024
MANIFEST = 'MANIFEST.json'
STATE = '.stage07-local-extraction.json'
BEST = 'run/training_state/best_model.bin'
SCOPE = ('Byte correspondence only; does not validate model quality, metrics, '
         'GPU/resource release, or Goal completion.')


def hash_stream(handle):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: handle.read(CHUNK), b''):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def hash_file(path):
    with path.open('rb') as handle:
        return hash_stream(handle)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=unique_object)


def safe_name(name):
    """Accept canonical relative POSIX names that are also safe on Windows."""
    if (not isinstance(name, str) or not name or name.startswith('/')
            or '\\' in name or ':' in name or ntpath.splitdrive(name)[0]
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError(f'Unsafe archive member: {name!r}')
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
    reserved.update(f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10))
    for part in parts:
        if (part in ('', '.', '..') or part.endswith((' ', '.'))
                or any(char in part for char in '<>"|?*')
                or part.split('.')[0].upper() in reserved):
            raise ValueError(f'Unsafe archive member: {name!r}')
    if parts[0].casefold() == STATE.casefold():
        raise ValueError('Archive member conflicts with local extraction state')
    return name


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def digest_value(value, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{label} must be a lowercase SHA256')
    return value


def record_value(record):
    if not isinstance(record, dict):
        raise ValueError('Manifest file record must be an object')
    return {'member': safe_name(record['member']),
            'sha256': digest_value(record['sha256'], 'file sha256'),
            'size_bytes': integer(record['size_bytes'], 'file size_bytes')}


def validate_receipt(receipt):
    if not isinstance(receipt, dict) or not isinstance(receipt.get('run_id'), str):
        raise ValueError('Receipt must contain a run_id string')
    if not receipt['run_id']:
        raise ValueError('Receipt run_id cannot be empty')
    digest_value(receipt['sha256'], 'archive sha256')
    digest_value(receipt['checkpoint_sha256'], 'checkpoint_sha256')
    integer(receipt['size_bytes'], 'archive size_bytes')
    integer(receipt['members'], 'archive members')


def archive_members(raw):
    # gzip validates its trailer; ignore_zeros scans beyond tar padding so a
    # second appended tar cannot hide members after the final manifest.
    raw.seek(0)
    with gzip.GzipFile(fileobj=raw, mode='rb') as compressed:
        with tarfile.open(fileobj=compressed, mode='r|', ignore_zeros=True) as archive:
            for member in archive:
                if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        or member.sparse is not None):
                    raise ValueError(f'Only regular, nonsparse files are allowed: {member.name!r}')
                safe_name(member.name)
                integer(member.size, 'tar member size')
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f'Cannot read archive member: {member.name}')
                with stream:
                    yield member, stream
        # Drain any remaining compressed data to enforce gzip CRC/EOF checks.
        for _ in iter(lambda: compressed.read(CHUNK), b''):
            pass


def verify_archive(raw, receipt):
    observed, names = {}, set()
    manifest, manifest_record = None, None
    for member, stream in archive_members(raw):
        key = member.name.casefold()
        if key in names:
            raise ValueError(f'Duplicate/colliding archive member: {member.name}')
        names.add(key)
        if manifest is not None:
            raise ValueError('MANIFEST.json must be the last archive member')
        if member.name == MANIFEST:
            if member.size > 16 * CHUNK:
                raise ValueError('Manifest exceeds the 16 MiB verification limit')
            data = stream.read()
            manifest = parse_json(data.decode('utf-8-sig'))
            manifest_record = {'member': MANIFEST,
                               'sha256': hashlib.sha256(data).hexdigest(),
                               'size_bytes': len(data)}
            if len(data) != member.size:
                raise ValueError('Manifest size differs from tar header')
        else:
            digest, size = hash_stream(stream)
            if size != member.size:
                raise ValueError(f'Tar size mismatch: {member.name}')
            observed[member.name] = {'member': member.name, 'sha256': digest,
                                     'size_bytes': size}
    if not isinstance(manifest, dict) or manifest.get('format') != 'stage07_delivery_v1':
        raise ValueError('Missing or unsupported Stage07 manifest')
    if manifest.get('run_id') != receipt['run_id']:
        raise ValueError('Receipt and manifest run_id differ')
    if not isinstance(manifest.get('files'), list):
        raise ValueError('Manifest files must be a list')
    declared = {}
    for value in manifest['files']:
        record = record_value(value)
        name = record['member']
        if name == MANIFEST or name in declared:
            raise ValueError(f'Duplicate/reserved manifest member: {name}')
        declared[name] = record
    if declared != observed:
        raise ValueError('Manifest and archive file sets, SHA256 or sizes differ')
    if receipt['members'] != len(observed) + 1:
        raise ValueError('Receipt member count differs from archive')
    # Reject a file used as the parent of another member before extraction.
    for name in names:
        parts = name.split('/')
        if any('/'.join(parts[:i]) in names for i in range(1, len(parts))):
            raise ValueError(f'File/directory path collision: {name}')
    if BEST not in declared or declared[BEST]['sha256'] != receipt['checkpoint_sha256']:
        raise ValueError('Best checkpoint is absent or differs from receipt SHA256')
    return manifest, {**declared, MANIFEST: manifest_record}


def inside(root, name):
    target = root.joinpath(*name.split('/'))
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Extraction path escaped its new root: {name}')
    return target


def write_json_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def verify_and_extract(args, audit):
    destination = args.destination.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError('Destination already exists; choose a new directory')
    destination = destination.resolve()
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise ValueError('Destination needs an existing parent and must not exist')
    if args.output.resolve().is_relative_to(destination):
        raise ValueError('Keep the audit output outside the extraction directory')
    receipt = parse_json(args.receipt.read_text(encoding='utf-8-sig'))
    validate_receipt(receipt)
    with args.archive.open('rb') as raw:
        digest, size = hash_stream(raw)
        if (digest, size) != (receipt['sha256'], receipt['size_bytes']):
            raise ValueError('Archive SHA256 or size differs from external receipt')
        manifest, records = verify_archive(raw, receipt)
        audit.update(run_id=manifest['run_id'], archive_sha256=digest,
                     archive_size_bytes=size, archive_members=len(records),
                     checkpoint_sha256=receipt['checkpoint_sha256'])
        # Atomic mkdir refuses even a target created since the initial check.
        destination.mkdir(exist_ok=False)
        root = destination.resolve(strict=True)
        audit['extraction_root'] = str(root)
        audit['local_run_path'] = str(root / 'run')
        state = root / STATE
        partial = {'status': 'PARTIAL_DO_NOT_USE', 'scope': SCOPE,
                   'archive_sha256': digest, 'run_id': manifest['run_id']}
        write_json_new(state, partial)
        extracted = set()
        for member, stream in archive_members(raw):
            if member.name not in records or member.name in extracted:
                raise ValueError('Archive membership changed after validation')
            extracted.add(member.name)
            target = inside(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = inside(root, member.name)
            file_digest, file_size = hashlib.sha256(), 0
            with target.open('xb') as output:
                for block in iter(lambda: stream.read(CHUNK), b''):
                    output.write(block)
                    file_digest.update(block)
                    file_size += len(block)
            expected = records[member.name]
            if (file_digest.hexdigest(), file_size) != (expected['sha256'], expected['size_bytes']):
                raise ValueError(f'Member changed during extraction: {member.name}')
        if extracted != records.keys():
            raise ValueError('Archive lost members after validation')
        raw.seek(0)
        if hash_stream(raw) != (digest, size):
            raise ValueError('Archive bytes changed during extraction')
    for name, expected in records.items():
        if hash_file(inside(root, name)) != (expected['sha256'], expected['size_bytes']):
            raise ValueError(f'Extracted file SHA256 or size mismatch: {name}')
    if hash_file(inside(root, BEST))[0] != receipt['checkpoint_sha256']:
        raise ValueError('Extracted checkpoint SHA256 differs from receipt')
    audit['files_verified_after_extraction'] = len(records)
    audit['status'] = 'PASS'
    # Retain a distinct completion marker; failures above leave PARTIAL intact.
    with state.open('w', encoding='utf-8') as handle:
        json.dump({**partial, 'status': 'VERIFIED_BYTES_ONLY'}, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.destination.expanduser().resolve()):
        parser.error('Keep the audit output outside the extraction directory')
    if os.path.lexists(args.output) or os.path.lexists(args.output.resolve()):
        parser.error('Audit output already exists; choose a new filename')
    if not args.output.parent.is_dir():
        parser.error('Audit output parent directory must already exist')
    audit = {'status': 'FAIL', 'scope': SCOPE,
             'checked_utc': datetime.now(timezone.utc).isoformat(),
             'archive': str(args.archive.resolve()),
             'receipt': str(args.receipt.resolve())}
    try:
        verify_and_extract(args, audit)
    except Exception as error:
        audit['status'] = 'FAIL'
        audit['error'] = f'{type(error).__name__}: {error}'
    write_json_new(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
