"""Independent, stdlib-only local byte-delivery audit; never restores a model.

Run only after the caller has confirmed that all three downloads have completed.
Every output must be new. Failure preserves partial output for inspection; this
script never removes, overwrites, repairs, or merges an existing directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'stage09_h1_recovery_backup_v1'
EXPECTED_PARENT = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
EXPECTED_EXECUTION = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
CHUNK_BYTES = 64 * 1024 * 1024
READ_BYTES = 4 * 1024 * 1024


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def decode_json(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=no_duplicate_keys)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b''):
            result.update(block)
    return result.hexdigest()


def check_sha(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid lowercase SHA256 in receipt/manifest')
    return value


def nonnegative_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Expected a nonnegative integer')
    return value


def no_link_ancestors(path):
    """Do not accept POSIX links, Windows junctions, or other reparse points."""
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        if not part.exists() and not part.is_symlink():
            continue
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Linked/reparse path is forbidden: {part}')


def signature(path):
    no_link_ancestors(path)
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Not a regular input file: {path}')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def safe_member(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute() or path.as_posix() != name
            or any(ord(c) < 32 for c in name) or '\\' in name or ':' in name
            or any(part in ('', '.', '..') or part.endswith(('.', ' ')) for part in path.parts)):
        raise ValueError(f'Unsafe member path: {name!r}')
    name.encode('utf-8', errors='strict')
    for part in path.parts:
        if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0], flags=re.I):
            raise ValueError(f'Windows reserved member path: {name!r}')
    return path


def register_member(name, known):
    parts = safe_member(name).parts
    for length in range(1, len(parts) + 1):
        prefix = '/'.join(parts[:length])
        key, role = prefix.casefold(), 'file' if length == len(parts) else 'directory'
        if key in known:
            original, previous_role = known[key]
            if original != prefix or role != previous_role or role == 'file':
                raise ValueError(f'Duplicate/case/type collision: {name!r}')
        else:
            known[key] = prefix, role


def verify_archive_ranges(archive, receipt):
    total = nonnegative_integer(receipt['bytes'])
    if total == 0 or signature(archive)[2] != total or receipt['chunk_bytes'] != CHUNK_BYTES:
        raise ValueError('Archive size or required 64MiB block size differs')
    chunks = receipt['chunks']
    if len(chunks) != (total + CHUNK_BYTES - 1) // CHUNK_BYTES:
        raise ValueError('Block count does not cover the archive exactly')
    overall, offset = hashlib.sha256(), 0
    with archive.open('rb') as stream:
        for index, row in enumerate(chunks):
            expected_size = min(CHUNK_BYTES, total - offset)
            if (type(row.get('index')) is not int or row['index'] != index
                    or type(row.get('offset')) is not int or row['offset'] != offset
                    or type(row.get('bytes')) is not int or row['bytes'] != expected_size):
                raise ValueError('Block ranges are not contiguous and complete')
            block_digest, remaining = hashlib.sha256(), expected_size
            while remaining:
                data = stream.read(min(remaining, READ_BYTES))
                if not data:
                    raise ValueError('Truncated archive within a block')
                overall.update(data)
                block_digest.update(data)
                remaining -= len(data)
            if block_digest.hexdigest() != check_sha(row['sha256']):
                raise ValueError(f'Archive block SHA mismatch at index {index}')
            offset += expected_size
        if stream.read(1):
            raise ValueError('Archive has trailing bytes beyond the receipt')
    if overall.hexdigest() != check_sha(receipt['sha256']):
        raise ValueError('Whole archive SHA mismatch')
    return dict(bytes=total, sha256=overall.hexdigest(), block_bytes=CHUNK_BYTES,
                block_count=len(chunks), all_blocks_verified=True, contiguous_complete_ranges=True)


def checkpoint_inventory(destination, records):
    result = {}
    stages = [(25, 'T', 20), (26, 'R', 21), (27, 'R', 22), (28, 'R', 23),
              (29, 'T', 23), (30, 'R', 24), (31, 'R', 25), (32, 'R', 26)]
    for policy in ('decay', 'freeze'):
        names = ['initial.pt', 'last.pt']
        for phase, kind, epoch in stages:
            t = (240 if phase < 29 else 280) if policy == 'decay' else 200
            names.append(f'phase{phase:02d}_{kind}_R{epoch:02d}_T{t:03d}.pt')
        files = {}
        for name in names:
            member = f'runs/H1-{policy}/{name}'
            row = records.get(member)
            if row is None or row['bytes'] <= 0:
                raise ValueError(f'Required checkpoint missing/empty: {member}')
            files[name] = row
        complete_member = f'runs/H1-{policy}/complete.json'
        complete = decode_json((destination / complete_member).read_bytes())
        expected = dict(policy=policy, r_updates=18018, t_updates=280 if policy == 'decay' else 200,
                        phase=33, endpoint_epoch=26)
        if any(type(complete.get(k)) is not type(v) or complete[k] != v for k, v in expected.items()):
            raise ValueError(f'Invalid completion receipt: {policy}')
        if complete['endpoint_checkpoint_sha256'] != files['last.pt']['sha256']:
            raise ValueError(f'Final checkpoint SHA differs from completion receipt: {policy}')
        result[policy] = dict(expected_counts=expected, files=files,
            saved_stage_boundaries=8, freeze_skipped_T_boundaries_present=(policy == 'freeze'),
            checkpoint_content_loaded=False, note='Presence and bytes only; separate cold CPU restore is required.')
    if records['references/R20_best.pt']['sha256'] != EXPECTED_PARENT:
        raise ValueError('Archived R20 parent SHA mismatch')
    return result


def verify(delivery, destination, output):
    delivery, destination, output = (Path(os.path.abspath(p)) for p in (delivery, destination, output))
    restore_parent = (ROOT / 'outputs/stage09_restore').resolve()
    if not destination.resolve().is_relative_to(restore_parent) or destination.resolve() == restore_parent:
        raise ValueError('New extraction directory must be within workspace outputs/stage09_restore')
    if destination.exists() or destination.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError('Extraction directory and independent receipt must both be new')
    if output.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Independent audit receipt must be outside extracted input tree')
    for path in (delivery, destination.parent, output.parent):
        no_link_ancestors(path)
    paths = {name: delivery / name for name in ('archive.tar.gz', 'MANIFEST.json', 'ARCHIVE_RECEIPT.json')}
    initial = {name: signature(path) for name, path in paths.items()}
    receipt_bytes, manifest_bytes = paths['ARCHIVE_RECEIPT.json'].read_bytes(), paths['MANIFEST.json'].read_bytes()
    receipt, manifest = decode_json(receipt_bytes), decode_json(manifest_bytes)
    if (receipt.get('format') != FORMAT or manifest.get('format') != FORMAT
            or receipt.get('status') != 'PACKAGED_BYTES_NOT_LOCAL_DELIVERY_OR_RESTORE'
            or receipt.get('archive') != 'archive.tar.gz'
            or receipt.get('input_inventory_stable') is not True
            or receipt.get('source_inputs_modified') is not False):
        raise ValueError('Unexpected packager receipt or unstable source inventory')
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != check_sha(receipt['manifest_sha256']):
        raise ValueError('External MANIFEST SHA does not match receipt')
    if manifest['execution_manifest_sha256'] != EXPECTED_EXECUTION:
        raise ValueError('Manifest does not name the accepted execution code')
    expected_files, known = manifest['files'], {}
    if (type(manifest.get('file_count')) is not int or manifest['file_count'] != len(expected_files)
            or 'MANIFEST.json' in expected_files):
        raise ValueError('Manifest file count/self-reference differs')
    for name, row in expected_files.items():
        register_member(name, known)
        nonnegative_integer(row['bytes'])
        check_sha(row['sha256'])
    if manifest['payload_bytes'] != sum(row['bytes'] for row in expected_files.values()):
        raise ValueError('Manifest payload byte count differs')
    archive_proof = verify_archive_ranges(paths['archive.tar.gz'], receipt)
    print(json.dumps(dict(event='archive_ranges_verified', **archive_proof)), flush=True)
    no_link_ancestors(destination.parent)
    destination.mkdir(parents=True, exist_ok=False)
    target_root = destination.resolve(strict=True)
    known, observed, records = {}, set(), {}
    expected_names = set(expected_files) | {'MANIFEST.json'}
    with tarfile.open(paths['archive.tar.gz'], 'r|gz') as archive:
        for member in archive:
            name = member.name
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f'Non-regular archive member forbidden: {name!r}')
            register_member(name, known)
            if name not in expected_names:
                raise ValueError(f'Unmanifested member: {name!r}')
            expected = (dict(bytes=len(manifest_bytes), sha256=manifest_sha)
                        if name == 'MANIFEST.json' else expected_files[name])
            if member.size != expected['bytes']:
                raise ValueError(f'Member size differs: {name!r}')
            path = destination.joinpath(*PurePosixPath(name).parts)
            if not path.resolve().is_relative_to(target_root):
                raise ValueError(f'Extraction target escapes the new root: {name!r}')
            path.parent.mkdir(parents=True, exist_ok=True)
            no_link_ancestors(path.parent)
            source = archive.extractfile(member)
            digest_value, count = hashlib.sha256(), 0
            with source, path.open('xb') as target:
                for block in iter(lambda: source.read(READ_BYTES), b''):
                    digest_value.update(block)
                    count += len(block)
                    target.write(block)
            if count != expected['bytes'] or digest_value.hexdigest() != expected['sha256']:
                raise ValueError(f'Extracted file byte/SHA mismatch: {name!r}')
            records[name] = dict(bytes=count, sha256=digest_value.hexdigest())
            observed.add(name)
    if observed != expected_names:
        raise ValueError('Missing or extra archive members')
    if (destination / 'MANIFEST.json').read_bytes() != manifest_bytes:
        raise ValueError('Inside and outside MANIFEST are not byte-identical')
    # Read every saved file independently, rather than only hashing transfer buffers.
    disk_files = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    if disk_files != expected_names:
        raise ValueError('Extracted on-disk file set differs')
    for name, row in records.items():
        path = destination / name
        if signature(path)[2] != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Independent on-disk hash differs: {name!r}')
    boundaries = checkpoint_inventory(destination, records)
    if (any(signature(paths[name]) != initial[name] for name in paths)
            or paths['MANIFEST.json'].read_bytes() != manifest_bytes
            or paths['ARCHIVE_RECEIPT.json'].read_bytes() != receipt_bytes
            or digest(paths['archive.tar.gz']) != archive_proof['sha256']):
        raise ValueError('Downloaded inputs changed during delivery audit')
    result = dict(format='stage09_h1_recovery_local_delivery_audit_v1', status='passed',
        created_utc=datetime.now(timezone.utc).isoformat(), delivery=str(delivery), extraction=str(destination),
        archive=archive_proof, manifest_sha256=manifest_sha,
        archive_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        manifest_inside_outside_bytes_equal=True, regular_members_only=True,
        exact_manifest_member_set=True, no_path_or_case_collisions=True,
        verified_payload_files=len(expected_files), verified_regular_members=len(records),
        verified_payload_bytes=manifest['payload_bytes'], extracted_disk_hashes_all_verified=True,
        input_bytes_unchanged=True, checkpoint_inventory=boundaries,
        self_contained_dataset=False, local_delivery_verified=True,
        checkpoint_restore_verified=False, gpu_execution_verified=False,
        evidence_scope='Independent archive/block/member and saved-disk byte verification only. '
                       'External data/PSE identities, actual checkpoint restoration, scientific metrics '
                       'and platform resource release require separate audits.',
        verifier_sha256=digest(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    no_link_ancestors(output.parent)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    value = verify(args.delivery, args.destination, args.output)
    print(json.dumps({key: value[key] for key in ('status', 'extraction', 'verified_payload_files',
                                                'local_delivery_verified', 'checkpoint_restore_verified')}, indent=2))


if __name__ == '__main__':
    main()
