"""Lock a label-blind, hash-ranked validation subset before model evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = 'stage09_validation_hash_subset_v1'
DEFAULT_SALT = 'stage09_validation_monitor_v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validation_subset_manifest(dataset_dir, *, size=1024, salt=DEFAULT_SALT):
    """Read only validation sample IDs; never open the test split or labels.

    SHA256 of the complete validation file binds its bytes without inspecting
    any label column. Stable hash ranking is independent of model/seed/metrics.
    """
    import pyarrow.parquet as pq

    root = Path(dataset_dir)
    path = root / 'validation.parquet'
    ids = pq.read_table(path, columns=['sample_id'])['sample_id'].to_pylist()
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= len(ids):
        raise ValueError('Subset size must be a positive integer no larger than validation')
    if not isinstance(salt, str) or not salt or '\n' in salt:
        raise ValueError('Hash salt must be a nonempty single line')
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError('sample_id must remain exact nonempty text; coercion is forbidden')
    if len(set(ids)) != len(ids):
        raise ValueError('Validation sample IDs are not unique')
    selected = sorted((hashlib.sha256((salt + '\n' + value).encode('utf-8')).hexdigest(), value, index)
                      for index, value in enumerate(ids))[:size]
    rows = [dict(index=index, sample_id=value, selection_sha256=digest) for digest, value, index in selected]
    binding = {'validation.parquet': {'sha256': file_sha256(path), 'bytes': path.stat().st_size}}
    # These describe the frozen source release; no train/test parquet is read.
    for name in ('manifest.json', 'config.json', 'item_map.json'):
        other = root / name
        if other.is_file():
            binding[name] = {'sha256': file_sha256(other), 'bytes': other.stat().st_size}
    return dict(format=FORMAT, split='validation', purpose='fixed_best_beam32_256_sensitivity_only',
                selection='smallest_sha256_utf8_salt_newline_exact_sample_id_then_sample_id',
                salt=salt, population_count=len(ids), count=size, dataset_files=binding,
                selected_rows_sha256=canonical_sha256(rows), rows=rows,
                label_columns_read=[], test_split_opened=False,
                note='Not the first N rows; not a replacement for full-validation best selection.')


def write_validation_subset(dataset_dir, output, *, size=1024, salt=DEFAULT_SALT):
    manifest = validation_subset_manifest(dataset_dir, size=size, salt=salt)
    path = Path(output)
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError('Refusing to replace a different locked subset')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as output_file:
            output_file.write(text)
    return manifest


def load_subset_indices(manifest_path, dataset_dir, *, split='validation'):
    """Verify source hashes and recompute the selection, not only row IDs."""
    if split != 'validation':
        raise ValueError('The locked sensitivity subset is validation-only')
    path = Path(manifest_path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('format') != FORMAT or saved.get('split') != split:
        raise ValueError('Unsupported subset manifest or split')
    expected = validation_subset_manifest(dataset_dir, size=saved.get('count'), salt=saved.get('salt'))
    if saved != expected:
        raise ValueError('Subset source hash, row identity, or deterministic selection mismatch')
    metadata = {key: value for key, value in saved.items() if key != 'rows'}
    metadata['manifest_sha256'] = file_sha256(path)
    return [row['index'] for row in saved['rows']], metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'verify'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--salt', default=DEFAULT_SALT)
    args = parser.parse_args(argv)
    if args.command == 'create':
        result = write_validation_subset(args.dataset, args.manifest, size=args.size, salt=args.salt)
    else:
        _, result = load_subset_indices(args.manifest, args.dataset)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
