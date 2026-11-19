"""Stdlib-only packaging and fresh-directory verification of completed test results.

Only the two evaluation outputs, flat text logs, and evaluate_test.py are allowed.
No model/data payload is visited. Existing outputs are never replaced. A failed
operation preserves partial output for inspection; it never cleans up inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


FORMAT = 'stage09_h1_recovery_test_results_v1'
PARENT_SHA = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
EXECUTION_SHA = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
ENDPOINTS = {'decay': '2ccf71e14ade6dbc58848a249f9396cd36d061f64abb65f56152e8afb84585c6',
             'freeze': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9'}
ARM_FILES = {'request.json', 'complete.json', 'evaluation/summary.json', 'evaluation/per_sample.jsonl.gz'}
BLOCK = 4 * 1024 * 1024


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        if not part.exists() and not part.is_symlink():
            continue
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symlink/reparse path forbidden: ' + str(part))
    return path


def safe_member(name):
    if not isinstance(name, str):
        raise ValueError('Member name must be text')
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name or '\\' in name or ':' in name
            or any(ord(c) < 32 for c in name)
            or any(p in ('', '.', '..') or p.endswith(('.', ' ')) for p in path.parts)):
        raise ValueError('Unsafe archive member: ' + repr(name))
    for part in path.parts:
        if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0], flags=re.I):
            raise ValueError('Windows-reserved archive member: ' + name)
    return path


def register(name, known):
    parts = safe_member(name).parts
    for length in range(1, len(parts) + 1):
        prefix = '/'.join(parts[:length])
        value = (prefix, 'file' if length == len(parts) else 'directory')
        previous = known.get(prefix.casefold())
        if previous is not None and (previous != value or value[1] == 'file'):
            raise ValueError('Duplicate/case/type collision: ' + name)
        known[prefix.casefold()] = value


def allowed(name):
    parts = safe_member(name).parts
    if name == 'delivery/evaluate_test.py':
        return True
    if parts[0] in ('H1-decay', 'H1-freeze'):
        return '/'.join(parts[1:]) in ARM_FILES
    return (len(parts) == 2 and parts[0] == 'logs'
            and Path(parts[1]).suffix.lower() in ('.log', '.txt', '.json', '.jsonl', '.out', '.err'))


def scan(root):
    files, known = {}, {}
    for scope in ('H1-decay', 'H1-freeze', 'logs'):
        base = no_links(root / scope)
        if not base.is_dir():
            raise ValueError('Required result directory missing: ' + scope)
        for directory, dirs, names in os.walk(base, followlinks=False):
            for name in dirs:
                no_links(Path(directory) / name)
            for name in names:
                path = no_links(Path(directory) / name)
                relative = path.relative_to(root).as_posix()
                if not path.is_file() or not allowed(relative):
                    raise ValueError('Unexpected/model/data result payload: ' + relative)
                register(relative, known)
                files[relative] = path
    launcher = no_links(root / 'delivery/evaluate_test.py')
    if not launcher.is_file():
        raise ValueError('Missing delivery/evaluate_test.py')
    register('delivery/evaluate_test.py', known)
    files['delivery/evaluate_test.py'] = launcher
    for policy in ENDPOINTS:
        required = {f'H1-{policy}/{name}' for name in ARM_FILES} | {f'logs/{policy}.log'}
        if not required <= files.keys():
            raise ValueError('Incomplete test result/log files: ' + policy)
    return files


def inspect_results(root, records):
    launcher_sha = records['delivery/evaluate_test.py']['sha256']
    summaries, completed = {}, {}
    for policy, checkpoint_sha in ENDPOINTS.items():
        prefix = f'H1-{policy}/'
        read = lambda name: decode((root / (prefix + name)).read_bytes())
        receipt, request, summary = read('complete.json'), read('request.json'), read('evaluation/summary.json')
        expected = dict(status='complete', policy=policy, examples=53194, checkpoint_sha256=checkpoint_sha,
            parent_sha256=PARENT_SHA, execution_manifest_sha256=EXECUTION_SHA, launcher_sha256=launcher_sha,
            training_performed=False, fixed_endpoint='R26')
        if any(type(receipt.get(key)) is not type(value) or receipt[key] != value for key, value in expected.items()):
            raise ValueError('Invalid completed R26 test receipt: ' + policy)
        for field, name in [('predictions_sha256', 'evaluation/per_sample.jsonl.gz'),
                            ('summary_sha256', 'evaluation/summary.json'), ('request_sha256', 'request.json')]:
            if receipt[field] != records[prefix + name]['sha256']:
                raise ValueError('Receipt/file SHA mismatch: ' + prefix + name)
        state = receipt.get('before_state_sha256')
        if (not isinstance(state, str) or re.fullmatch(r'[0-9a-f]{64}', state) is None
                or receipt.get('after_state_sha256') != state or request.get('before_state_sha256') != state):
            raise ValueError('Evaluation state changed or is not recorded: ' + policy)
        for key in ('policy', 'checkpoint_sha256', 'parent_sha256', 'execution_manifest_sha256',
                    'launcher_sha256', 'training_performed', 'fixed_endpoint'):
            if request.get(key) != receipt[key]:
                raise ValueError('Request/completion identity differs: ' + key)
        expected_summary = dict(split='test', examples=53194, full_split=True, subset=None,
            architecture='H', seed=2024, beam=32, r_updates=18018, t_updates=280 if policy == 'decay' else 200)
        if any(type(summary.get(key)) is not type(value) or summary[key] != value for key, value in expected_summary.items()):
            raise ValueError('Incomplete or wrong test evaluation: ' + policy)
        if (request.get('split') != 'test' or request.get('inference_precision') != 'fp32'
                or request.get('dataset_test_sha256') != summary['dataset_binding']['files']['test.parquet']['sha256']):
            raise ValueError('Test request/data identity differs: ' + policy)
        summaries[policy], completed[policy] = summary, receipt
    for key in ('dataset_binding', 'metric_protocol', 'catalog_size'):
        if summaries['decay'][key] != summaries['freeze'][key]:
            raise ValueError('Paired test protocols differ: ' + key)
    return completed


def read_archive(archive, manifest_sha, destination=None):
    """Check all headers before writing; use manual files, never extractall()."""
    with tarfile.open(archive, 'r:gz') as stream:
        members, known = {}, {}
        for member in stream.getmembers():
            register(member.name, known)
            if not member.isfile() or member.issparse():
                raise ValueError('Only regular, non-sparse members are allowed')
            if member.name != 'MANIFEST.json' and not allowed(member.name):
                raise ValueError('Unexpected archive payload: ' + member.name)
            members[member.name] = member
        if 'MANIFEST.json' not in members or members['MANIFEST.json'].size > 1024 * 1024:
            raise ValueError('Missing or excessive manifest')
        payload = stream.extractfile(members['MANIFEST.json']).read()
        if hashlib.sha256(payload).hexdigest() != manifest_sha:
            raise ValueError('Manifest SHA differs from independent receipt')
        manifest = decode(payload)
        records = manifest['files']
        if manifest.get('format') != FORMAT or set(members) != set(records) | {'MANIFEST.json'}:
            raise ValueError('Archive/manifest member inventory differs')
        for name, row in records.items():
            if (type(row['bytes']) is not int or row['bytes'] < 0 or row['bytes'] != members[name].size
                    or not isinstance(row['sha256'], str) or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None):
                raise ValueError('Invalid member size/SHA: ' + name)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
        for name, member in members.items():
            expected = dict(bytes=len(payload), sha256=manifest_sha) if name == 'MANIFEST.json' else records[name]
            digest, size = hashlib.sha256(), 0
            target = None
            if destination is not None:
                path = no_links(destination / name)
                path.parent.mkdir(parents=True, exist_ok=True)
                target = path.open('xb')
            try:
                with stream.extractfile(member) as source:
                    for block in iter(lambda: source.read(BLOCK), b''):
                        digest.update(block)
                        size += len(block)
                        if target is not None:
                            target.write(block)
            finally:
                if target is not None:
                    target.close()
            if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Member SHA/size mismatch: ' + name)
            if destination is not None and sha256(destination / name) != expected['sha256']:
                raise ValueError('Extracted disk SHA mismatch: ' + name)
    return manifest


def pack(root, archive, receipt_path):
    root, archive, receipt_path = map(no_links, (root, archive, receipt_path))
    if archive == receipt_path or archive.exists() or receipt_path.exists():
        raise FileExistsError('Archive and receipt must be distinct new files')
    for target in (archive, receipt_path):
        if any(target.is_relative_to(root / scope) for scope in ('H1-decay', 'H1-freeze', 'logs')) or target == root / 'delivery/evaluate_test.py':
            raise ValueError('Output cannot be within an included source scope')
    selected = scan(root)
    records = {name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for name, path in sorted(selected.items())}
    complete = inspect_results(root, records)
    manifest = dict(format=FORMAT, files=records, examples_per_arm=53194, parent_sha256=PARENT_SHA,
                    endpoints={key: value['checkpoint_sha256'] for key, value in complete.items()},
                    models_included=False, datasets_included=False)
    payload = json_bytes(manifest)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz', format=tarfile.PAX_FORMAT) as stream:
        for name, path in sorted(selected.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = records[name]['bytes'], 0o644, 0
            with path.open('rb') as source:
                stream.addfile(info, source)
        info = tarfile.TarInfo('MANIFEST.json')
        info.size, info.mode, info.mtime = len(payload), 0o644, 0
        stream.addfile(info, io.BytesIO(payload))
    read_archive(archive, manifest_sha)
    current = scan(root)
    if set(current) != set(selected) or any(path.stat().st_size != records[name]['bytes'] or sha256(path) != records[name]['sha256'] for name, path in current.items()):
        raise ValueError('Input files changed during packaging')
    receipt = dict(format=FORMAT, status='packed', archive_name=archive.name, bytes=archive.stat().st_size,
                   sha256=sha256(archive), manifest_sha256=manifest_sha, payload_files=len(records),
                   input_bytes_unchanged=True, members_read_back_verified=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('xb') as output:
        output.write(json_bytes(receipt))
    return receipt


def verify(archive, receipt_path, destination):
    archive, receipt_path, destination = map(no_links, (archive, receipt_path, destination))
    if destination.exists():
        raise FileExistsError('Extraction destination must be new')
    receipt = decode(receipt_path.read_bytes())
    if (receipt.get('format') != FORMAT or receipt.get('status') != 'packed'
            or archive.stat().st_size != receipt['bytes'] or sha256(archive) != receipt['sha256']):
        raise ValueError('Archive bytes/SHA differ from independent receipt')
    manifest = read_archive(archive, receipt['manifest_sha256'], destination)
    files = scan(destination)
    if set(files) != set(manifest['files']) or len(files) != receipt['payload_files']:
        raise ValueError('Extracted payload inventory differs')
    inspect_results(destination, manifest['files'])
    report = dict(format=FORMAT, status='verified', archive_sha256=receipt['sha256'],
                  manifest_sha256=receipt['manifest_sha256'], payload_files=len(files),
                  extracted_files=len(files) + 1, both_completed_test_receipts_verified=True,
                  all_member_and_disk_sha256_verified=True)
    with (destination / 'VERIFICATION.json').open('xb') as output:
        output.write(json_bytes(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('pack')
    create.add_argument('--root', required=True)
    create.add_argument('--archive', required=True)
    create.add_argument('--receipt', help='Default: <archive>.receipt.json')
    check = commands.add_parser('verify')
    check.add_argument('--archive', required=True)
    check.add_argument('--receipt', required=True)
    check.add_argument('--destination', required=True)
    args = parser.parse_args(argv)
    result = (pack(args.root, args.archive, args.receipt or args.archive + '.receipt.json')
              if args.command == 'pack' else verify(args.archive, args.receipt, args.destination))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
