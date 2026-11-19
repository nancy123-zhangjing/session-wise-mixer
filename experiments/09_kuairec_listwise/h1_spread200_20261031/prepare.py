"""Stage an authenticated historical code tree plus explicit spread200 overlay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CODE = ROOT / 'outputs/stage09_spread200_20261031/code'
BASE = ROOT / 'outputs/stage09_restore/h1_recovery_20261028_r2/code'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parent = json.loads((HERE / 'PARENT_BINDING.json').read_text(encoding='utf-8'))
    record = ROOT / 'outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_H1_R1.json'
    if digest(record) != '768ffba5dcb22f6223b3c731cd7292e1b4da26b7b6375b5d5308478a01b2e81c':
        raise ValueError('Historical run receipt changed')
    original = json.loads(record.read_text(encoding='utf-8'))
    entry = next(r for r in original['files'] if r['path'] == 'runs/H1/initial.pt')
    if parent['sha256'] != entry['sha256'] or digest(parent['local']) != entry['sha256']:
        raise ValueError('R5 parent not authenticated by original delivery receipt')
    binding = parent['binding']
    for role, base in (('code', BASE), ('engine', BASE / 'external/DiffGRM')):
        for name, expected in binding[role].items():
            if digest(base / name) != expected:
                raise ValueError('Frozen historical source mismatch: ' + str(base / name))
    if not CODE.exists():
        CODE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE, CODE, symlinks=False)
    if (CODE / 'RUNNING_FORMAL').exists():
        raise RuntimeError('Never change deployed/running source')
    # Only these explicitly named staged files may be replaced before deployment.
    overlay = ['spread200.py', 'spread200_run.py', 'spread200_preflight.py',
               'recovery35.py', 'recovery35_run.py', 'recovery35_preflight.py',
               'recovery35_artifacts.py', 'recovery35_trajectory.py']
    for name in overlay:
        shutil.copyfile(ROOT / 'scripts/listwise_diffgrm' / name, CODE / 'scripts/listwise_diffgrm' / name)
    for name in ('test_listwise_spread200.py', 'test_listwise_spread200_runner.py'):
        shutil.copyfile(ROOT / 'tests' / name, CODE / 'tests' / name)
    manifest = {}
    for path in sorted(CODE.rglob('*')):
        if path.is_symlink():
            raise ValueError('No symlinks in scientific execution package')
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'SPREAD200_EXECUTION_MANIFEST.json':
            manifest[path.relative_to(CODE).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    document = dict(format='spread200_execution_manifest_v1', parent_sha256=parent['sha256'],
        authenticated_parent_receipt_sha256=digest(record), files=manifest, explicit_overlay=overlay,
        frozen_scientific_sources_preserved=True)
    target = CODE / 'SPREAD200_EXECUTION_MANIFEST.json'
    write(target, document)
    archive = HERE / 'execution_code.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for name in sorted(list(manifest) + [target.name]):
                path = CODE / name
                info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    receipt = dict(status='prepared', code_root=str(CODE), manifest_sha256=digest(target),
        archive=str(archive), archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
        files=len(manifest), parent_sha256=parent['sha256'], source_receipt_sha256=digest(record))
    write(HERE / 'EXECUTION_PACKAGE.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
