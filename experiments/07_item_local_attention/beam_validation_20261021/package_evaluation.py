"""Package only the new validation diagnostics; omit copied inputs and caches."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    results = root / 'results'
    assert (results / 'COMPLETE.json').is_file(), 'Only package a completed grid'
    archive = root / 'validation_evidence.tar.gz'
    manifest = root / 'VALIDATION_EVIDENCE_MANIFEST.json'
    receipt = root / 'VALIDATION_EVIDENCE_RECEIPT.json'
    assert not any(p.exists() for p in (archive, manifest, receipt)), 'Refuse to overwrite evidence'
    files = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if any(part in ('workspace', 'shared', 'cache', '__pycache__') for part in rel.parts):
            continue
        if p.suffix not in ('.json', '.jsonl', '.log', '.py', '.md', '.csv'):
            continue
        files.append({'path': rel.as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    manifest.write_text(json.dumps({'scope':'New evaluation sources, diagnostics and per-user results; unchanged old weights and copied inputs excluded',
                                    'files': files}, indent=2)+'\n', encoding='utf-8')
    with tarfile.open(archive, 'w:gz') as tf:
        for row in files:
            tf.add(root / row['path'], arcname=row['path'], recursive=False)
        tf.add(manifest, arcname=manifest.name, recursive=False)
    result = {'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': sha(archive), 'files': len(files), 'manifest_sha256': sha(manifest)}
    receipt.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
