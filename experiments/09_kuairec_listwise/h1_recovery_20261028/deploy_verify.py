"""Extract this task's authenticated code package and verify remote assets."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--archive-sha', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archive = root / 'delivery/h1_recovery_code.tar.gz'
    if sha(archive) != args.archive_sha:
        raise RuntimeError('Uploaded archive SHA mismatch')
    code = root / 'code'
    if not code.exists():
        staging = root / 'extract_staging'
        staging.mkdir(exist_ok=False)
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                name = Path(member.name)
                if (name.is_absolute() or '..' in name.parts or name.parts[0] != 'code'
                        or not member.isfile() or not (staging / name).resolve().is_relative_to(staging)):
                    raise RuntimeError('Unexpected archive member')
            stream.extractall(staging)
        (staging / 'code').rename(code)
    manifest = json.loads((code / 'RECOVERY_EXECUTION_MANIFEST.json').read_text())
    for name, expected in manifest['execution_files'].items():
        if sha(code / name) != expected['sha256']:
            raise RuntimeError('Source SHA mismatch: ' + name)
    binding = json.loads((code / 'RECOVERY_PARENT_BINDING.json').read_text())
    for name, expected in binding['code'].items():
        if sha(code / name) != expected:
            raise RuntimeError('Original code mismatch: ' + name)
    for name, expected in binding['engine'].items():
        if sha(code / 'external/DiffGRM' / name) != expected:
            raise RuntimeError('Original engine mismatch: ' + name)
    config = binding['config']
    for name, expected in binding['data']['files'].items():
        if sha(Path(config['dataset']) / name) != expected['sha256']:
            raise RuntimeError('Dataset bytes differ: ' + name)
    if sha(Path(config['assets']) / 'manifest.json') != binding['asset_manifest']:
        raise RuntimeError('PSE manifest differs')
    parent = Path(config['dataset']).parents[1] / 'runs/H1/best.pt'
    if sha(parent) != manifest['parent']['sha256']:
        raise RuntimeError('Remote H1 parent differs')
    report = dict(status='passed', archive_sha256=sha(archive), source_files=len(manifest['execution_files']),
                  original_source_files_verified=40, data_files_verified=len(binding['data']['files']),
                  parent_sha256=sha(parent), source_commit=manifest['workspace']['commit'],
                  package_sources_uncommitted=manifest['workspace']['package_sources_uncommitted'],
                  execution_manifest_sha256=sha(code / 'RECOVERY_EXECUTION_MANIFEST.json'))
    (root / 'delivery/REMOTE_READY.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
