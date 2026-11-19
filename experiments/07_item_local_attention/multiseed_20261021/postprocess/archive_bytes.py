"""Mature Stage07 archive byte pack/verify functions, with new identity/name
and a final complete source-byte recheck before publishing the package receipt.
Original implementation: scripts/analysis/pack_item_local_joint_delivery.py.
The adjacent safe-path helper is copied unchanged from the original delivery.
"""
from __future__ import annotations
import gzip, hashlib, io, json, os, re, shutil, tarfile, tempfile
from pathlib import Path
from verify_item_local_archive import safe_name, archive_members, hash_file, hash_stream, parse_json, record_value, inside
CAMPAIGN = 'stage07_item_local_joint_multiseed_20261021_r1'
FORMAT = 'stage07_multiseed_full_backup_v1'
ORIGINAL_PACKER_SHA256 = '849a296fe6c8f9dc52103b5edae9bab86ac5e148a072425a8b3aa9e319e0efba'
ORIGINAL_SAFE_HELPER_SHA256 = '4abc8f674e75b8f59e33eedbe4d4f141d93d59998316828defb4265d53b5bc08'

def need(condition, message):
    if not condition: raise ValueError(message)

def read(path):
    return parse_json(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hash_file(Path(path))[0]

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')

def file_record(member, path):
    safe_name(member)
    need(path.is_file() and not any(parent.is_symlink() for parent in (path, *path.parents)), 'Nonregular/symlink source: ' + str(path))
    digest, size = hash_file(path)
    return {'member': member, 'sha256': digest, 'size_bytes': size}

def archive_files(sources, output, metadata, part_bytes):
    files = [file_record(member, path) for member, path in sorted(sources.items())]
    manifest = {'format': FORMAT, 'campaign_id': CAMPAIGN, 'files': files, **metadata}
    archive = output / 'multiseed_delivery.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', dereference=True) as tar:
            for record in files:
                path = sources[record['member']]; tar.add(path, arcname=record['member'], recursive=False)
                need(file_record(record['member'], path) == record, 'Source changed while packaging: ' + str(path))
            data = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
            info = tarfile.TarInfo('MANIFEST.json'); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    for record in files:
        need(file_record(record['member'], sources[record['member']]) == record,
             'Source changed after its archive member was written: ' + record['member'])
    digest, size = hash_file(archive)
    parts = []
    if size > part_bytes:
        with archive.open('rb') as handle:
            while block := handle.read(part_bytes):
                part = output / f'multiseed_delivery.tar.gz.part{len(parts) + 1:04d}'
                with part.open('xb') as target: target.write(block)
                parts.append(file_record(part.name, part))
        archive.unlink()  # Only the collector's own redundant archive in its new output directory.
    else: parts.append(file_record(archive.name, archive))
    receipt = {'format': FORMAT, 'status': 'PACKAGED_BYTES_NOT_RESEARCH_AUDIT', 'campaign_id': CAMPAIGN,
               'archive_sha256': digest, 'archive_size_bytes': size, 'parts': parts, 'files': len(files),
               'manifest_sha256': hashlib.sha256(data).hexdigest(), 'goal_complete': False}
    write(output / 'PACKAGE.json', receipt)
    return receipt

def verify(package, output):
    need(not os.path.lexists(output), 'Extraction destination must be new')
    package, output = package.resolve(), output.resolve(); receipt = read(package / 'PACKAGE.json')
    need(receipt['format'] == FORMAT and not output.exists() and not output.is_symlink(), 'Wrong package or existing extraction destination')
    need(receipt['parts'] and len({row['member'].casefold() for row in receipt['parts']}) == len(receipt['parts']), 'Missing/duplicate volumes')
    with tempfile.TemporaryFile() as raw:
        for row in receipt['parts']:
            safe_name(row['member']); need('/' not in row['member'], 'Volume must be a direct package file')
            part = package / row['member']; need(file_record(row['member'], part) == record_value(row), 'Volume bytes/SHA differ')
            with part.open('rb') as handle: shutil.copyfileobj(handle, raw, 1024 * 1024)
        raw.seek(0); need(hash_stream(raw) == (receipt['archive_sha256'], receipt['archive_size_bytes']), 'Whole archive SHA/size differs')
        observed, names, manifest = {}, set(), None
        for member, stream in archive_members(raw):
            key = member.name.casefold(); need(key not in names and manifest is None, 'Duplicate/colliding member or data after manifest'); names.add(key)
            if member.name == 'MANIFEST.json':
                need(member.size <= 16 * 1024 * 1024, 'Manifest too large'); data = stream.read(); manifest = parse_json(data.decode('utf-8'))
                need(hashlib.sha256(data).hexdigest() == receipt['manifest_sha256'], 'Manifest bytes differ')
            else:
                digest, size = hash_stream(stream); need(size == member.size, 'Truncated tar member')
                observed[member.name] = {'member': member.name, 'sha256': digest, 'size_bytes': size}
        need(manifest and manifest['format'] == FORMAT and manifest['campaign_id'] == receipt['campaign_id'] == CAMPAIGN, 'Missing/wrong manifest identity')
        declared = [record_value(row) for row in manifest['files']]
        need(len(declared) == len(observed) == receipt['files'] and {row['member']: row for row in declared} == observed
             and 'BYTE_VERIFICATION.json' not in observed, 'Manifest does not cover exact archive bytes or reserves verification output')
        need(all('/'.join(name.split('/')[:index]) not in names for name in names for index in range(1, len(name.split('/')))), 'File/directory prefix collision')
        output.mkdir(parents=True, exist_ok=False)
        extracted = set()
        for member, stream in archive_members(raw):
            need(member.name not in extracted and (member.name in observed or member.name == 'MANIFEST.json'), 'Second-pass archive membership changed'); extracted.add(member.name)
            target = inside(output, member.name); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle: shutil.copyfileobj(stream, handle, 1024 * 1024)
            expected = observed.get(member.name, {'member': 'MANIFEST.json', 'sha256': receipt['manifest_sha256'], 'size_bytes': len(data)})
            need(file_record(member.name, target) == expected, 'Extracted bytes differ')
        need(extracted == set(observed) | {'MANIFEST.json'}, 'Extracted file set differs')
        raw.seek(0); need(hash_stream(raw) == (receipt['archive_sha256'], receipt['archive_size_bytes']), 'Archive changed while extracting')
        result = {'status': 'BYTE_RESTORATION_VERIFIED', 'archive_sha256': receipt['archive_sha256'], 'files': len(observed),
                  'research_audit_performed': False, 'goal_complete': False}
        write(output / 'BYTE_VERIFICATION.json', result); return result
