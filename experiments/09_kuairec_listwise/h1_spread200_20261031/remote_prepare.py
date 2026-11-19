"""Verify the uploaded archive, extract only into this new run, bind inputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'archive-sha', 'manifest-sha', 'parent-sha', 'subset-sha', 'self-sha'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    expected_parent = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
    if root.parent != expected_parent or root.name != 'h1_spread200_20261031_r1':
        raise ValueError('Only the explicitly owned new experiment directory is permitted')
    if sha(__file__) != a.self_sha or sha(root / 'execution_code.tar.gz') != a.archive_sha:
        raise ValueError('Uploaded preparer/archive changed')
    code = root / 'code'
    if code.exists():
        raise FileExistsError('Never overwrite an existing execution tree')
    with tarfile.open(root / 'execution_code.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile() or member.size > 50*1024*1024:
                raise ValueError('Unsafe code archive member')
        code.mkdir()
        archive.extractall(code)
    manifest = code / 'SPREAD200_EXECUTION_MANIFEST.json'
    if sha(manifest) != a.manifest_sha:
        raise ValueError('Execution manifest mismatch')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    for name, entry in data['files'].items():
        if sha(code / name) != entry['sha256']:
            raise ValueError('Execution payload mismatch: ' + name)
    inputs = dict(parent=str(expected_parent / 'goal_20261027_r1/runs/H1/initial.pt'),
        dataset=str(expected_parent / 'goal_20261027_r1/inputs/dataset'),
        assets=str(expected_parent / 'goal_20261027_r1/assets/pse_v1'),
        warmup_predictions=str(expected_parent / 'goal_20261027_r1/runs/warmup_H/validation_R05/per_sample.jsonl.gz'),
        reference_training=str(expected_parent / 'goal_20261027_r1/runs/H1/training.jsonl'),
        subset=str(root / 'validation_subset_8192.json'))
    if sha(inputs['parent']) != a.parent_sha or sha(inputs['subset']) != a.subset_sha:
        raise ValueError('Parent or original subset changed')
    for key in ('dataset', 'assets', 'warmup_predictions', 'reference_training'):
        if not Path(inputs[key]).exists():
            raise FileNotFoundError('Missing authorized external dependency: ' + key)
    (root / 'INPUTS.json').write_text(json.dumps(inputs, indent=2) + '\n', encoding='utf-8')
    receipt = dict(status='prepared', parent_sha256=a.parent_sha, manifest_sha256=a.manifest_sha,
                   subset_sha256=a.subset_sha, archive_sha256=a.archive_sha, files=len(data['files']), inputs=inputs)
    (root / 'REMOTE_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
