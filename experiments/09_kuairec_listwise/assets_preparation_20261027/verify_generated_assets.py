"""Independent CPU-only restore audit of the locked Stage09 generated assets.

Run with .runtime/stage08_restore_env/Scripts/python.exe -B and this script.
No project model modules, platform, FAISS training, or GPU are used. Existing
restore/report paths are refused; failed partial restores are preserved.
"""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import tarfile


ROOT = Path(__file__).absolute().parents[3]
DELIVERY = ROOT / 'outputs/stage09_delivery/goal_20261027_r1'
EVIDENCE = ROOT / 'experiments/09_kuairec_listwise/assets_preparation_20261027'
SOURCE_MANIFEST = EVIDENCE / 'GENERATED_ASSETS_BACKUP_MANIFEST.json'
ARCHIVE = DELIVERY / 'stage09_generated_assets_r1.tar'
DESTINATION = ROOT / 'outputs/stage09_restore/goal_20261027_r1'
REPORT = EVIDENCE / 'GENERATED_ASSETS_LOCAL_RESTORE.json'
MANIFEST_SHA = 'c865c3bb4021ab4baaf4bbb8b8267b4b790eb5f43a35d8fd7e97554ba8da868f'
ARCHIVE_SHA = '94804ee383c8f2cf103d831c7afaabc1da975c1a47441bce200fcab6c91ba37a'
ARCHIVE_BYTES, PAYLOAD_BYTES, N = 94105600, 94090157, 10728
BGE = 'assets/bge_m3_cuda_v1/'
PSE = 'assets/pse_v1/'
EXPECTED_NAMES = {
    BGE + name for name in ('dense_embeddings.npy', 'original_item_ids.npy',
        'untruncated_token_lengths.npy', 'metadata.json', 'ordered_text.jsonl')
} | {
    PSE + name for name in ('assets.pt', 'base_pse.faiss', 'base_pse_codebooks.npy',
        'coarse_centroid.npy', 'content_features.npy', 'initial_mixer.pt', 'items.json',
        'manifest.json', 'opq_residual_embeddings.npy', 'original_item_ids.npy',
        'pca_transform.npz', 'reconstruction_errors.npy', 'semantic_ids.npy')
}
CHUNK = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def safe_path(path, *, create_parents=False):
    """Check every ancestor, including Windows junction/reparse-point aliases."""
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT), f'Path escapes workspace: {path}')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            if create_parents and current != path:
                require(current.is_relative_to(ROOT), 'Cannot create outside workspace')
                current.mkdir()
            else:
                continue
        info = current.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not
                (getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
                f'Symlink/reparse point refused: {current}')
        if current != path:
            require(stat.S_ISDIR(info.st_mode), f'Parent is not a directory: {current}')
    require(path.resolve(strict=False) == path, f'Path resolves elsewhere: {path}')
    return path


def digest(path):
    path = safe_path(path)
    require(stat.S_ISREG(path.lstat().st_mode), f'Not an ordinary file: {path}')
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(CHUNK), b''):
            value.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns) and size == before.st_size,
            f'File changed while reading: {path}')
    return dict(bytes=size, sha256=value.hexdigest())


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


def restore():
    require(not os.path.lexists(safe_path(DESTINATION)), 'Restore directory already exists')
    require(not os.path.lexists(safe_path(REPORT)), 'Restore report already exists')
    require(digest(SOURCE_MANIFEST)['sha256'] == MANIFEST_SHA, 'Backup manifest SHA differs')
    manifest = read_json(SOURCE_MANIFEST)
    require(manifest['schema'] == 'stage09_generated_assets_backup_v1' and manifest['status'] == 'PASS',
            'Backup manifest is not PASS')
    require(manifest['file_count'] == 18 and manifest['payload_bytes'] == PAYLOAD_BYTES,
            'Unexpected backup count/payload size')
    require(manifest['archive'] == dict(path='delivery/stage09_generated_assets_r1.tar',
            bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA), 'Archive identity differs')
    entries = {row['path']: row for row in manifest['files']}
    require(len(manifest['files']) == len(entries) == 18 and set(entries) == EXPECTED_NAMES,
            'Manifest member set differs from the locked 18 files')
    require(sum(row['bytes'] for row in entries.values()) == PAYLOAD_BYTES, 'Payload size sum differs')
    require(digest(ARCHIVE) == dict(bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA), 'Downloaded tar SHA/size differs')
    verified = []
    with tarfile.open(safe_path(ARCHIVE), 'r:') as archive:
        members = archive.getmembers()
        require(len(members) == 18 and {member.name for member in members} == EXPECTED_NAMES,
                'Tar has missing, repeated, or extra members')
        for member in members:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and str(name) == member.name and '\\' not in member.name and
                    ':' not in member.name and all(part not in ('', '.', '..') for part in name.parts),
                    'Unsafe tar path')
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(),
                    f'Nonregular tar member: {member.name}')
            require(member.size == entries[member.name]['bytes'], f'Tar member size differs: {member.name}')
        safe_path(DESTINATION, create_parents=True).mkdir()
        for member in members:
            target = safe_path(DESTINATION / member.name, create_parents=True)
            # Manual extraction; xb refuses replacement and no archive metadata
            # or permissions are applied to the local filesystem.
            with archive.extractfile(member) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(CHUNK), b''):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            actual = digest(target)
            require(actual == {key: entries[member.name][key] for key in ('bytes', 'sha256')},
                    f'Restored file hash differs: {member.name}')
            verified.append(dict(path=member.name, **actual))
    return verified


def audit_numeric(files):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    inventory = {row['path']: row for row in files}
    summaries = {}

    def array(name, shape, dtype):
        value = np.load(safe_path(DESTINATION / name), allow_pickle=False)
        require(value.shape == shape and value.dtype == np.dtype(dtype), f'NPY shape/dtype differs: {name}')
        require(np.isfinite(value).all(), f'Nonfinite NPY: {name}')
        summaries[name] = dict(shape=list(shape), dtype=str(value.dtype), finite=True)
        return value

    dense = array(BGE + 'dense_embeddings.npy', (N, 1024), 'float32')
    dense_ids = array(BGE + 'original_item_ids.npy', (N,), 'int64')
    lengths = array(BGE + 'untruncated_token_lengths.npy', (N,), 'int64')
    values = {
        'centroids': array(PSE + 'base_pse_codebooks.npy', (4, 256, 64), 'float32'),
        'content_features': array(PSE + 'content_features.npy', (N, 256), 'float32'),
        'residual_features': array(PSE + 'opq_residual_embeddings.npy', (N, 256), 'float32'),
        'coarse_centroid': array(PSE + 'coarse_centroid.npy', (256,), 'float32'),
        'catalog_sids': array(PSE + 'semantic_ids.npy', (N, 4), 'int64'),
        'reconstruction_errors': array(PSE + 'reconstruction_errors.npy', (N,), 'float32'),
        'original_item_ids': array(PSE + 'original_item_ids.npy', (N,), 'int64'),
    }
    require(np.array_equal(dense_ids, values['original_item_ids']) and
            len(np.unique(dense_ids)) == N and np.all(dense_ids >= 0), 'Catalog IDs differ or are invalid')
    require(np.all(lengths > 0), 'Invalid token lengths')
    require(np.allclose(np.linalg.norm(dense, axis=1), 1, atol=1e-4, rtol=0), 'Dense vectors are not normalized')
    require(np.allclose(np.linalg.norm(values['content_features'], axis=1), 1, atol=1e-4, rtol=0),
            'PCA content vectors are not normalized')
    codes = values['catalog_sids']
    require(np.all((codes >= 0) & (codes < 256)), 'SID values are out of range')
    require(read_json(DESTINATION / (PSE + 'items.json')) == [str(int(value)) for value in dense_ids],
            'items.json differs from catalog order')
    pse = read_json(DESTINATION / (PSE + 'manifest.json'))
    metadata = read_json(DESTINATION / (BGE + 'metadata.json'))
    remote_audit = read_json(EVIDENCE / 'PSE_ASSET_AUDIT.json')
    require(pse['toy'] is False and (pse['n_items'], pse['n_digit'], pse['codebook_size'], pse['code_dim']) ==
            (N, 4, 256, 64), 'PSE asset is toy or has a different layout')
    require(pse['dataset_binding'] == metadata['dataset_binding'] == remote_audit['dataset_binding'] and
            pse['encoder'] == metadata == remote_audit['encoder'], 'Dataset/encoder binding differs')
    require(remote_audit['asset_manifest_sha256'] == inventory[PSE + 'manifest.json']['sha256'],
            'Remote audit refers to another PSE manifest')
    require(set(pse['artifacts']) == {name[len(PSE):] for name in EXPECTED_NAMES if name.startswith(PSE)} -
            {'manifest.json'}, 'PSE internal artifact set differs')
    for name, row in pse['artifacts'].items():
        require(row == {key: inventory[PSE + name][key] for key in ('bytes', 'sha256')},
                f'PSE internal hash differs: {name}')
    for key, name in [('embeddings_sha256', 'dense_embeddings.npy'),
                      ('original_item_ids_sha256', 'original_item_ids.npy'),
                      ('token_lengths_sha256', 'untruncated_token_lengths.npy'),
                      ('ordered_text_file_sha256', 'ordered_text.jsonl')]:
        require(metadata[key] == inventory[BGE + name]['sha256'], f'BGE metadata hash differs: {key}')
    with safe_path(DESTINATION / (BGE + 'ordered_text.jsonl')).open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    require(len(rows) == N, 'Ordered text row count differs')
    require(metadata['n_items'] == N and metadata['dimension'] == 1024 and
            metadata['normalize_embeddings'] is True, 'BGE metadata layout differs')
    bundle = torch.load(safe_path(DESTINATION / (PSE + 'assets.pt')), map_location='cpu', weights_only=True)
    for key, value in values.items():
        require(isinstance(bundle[key], torch.Tensor) and np.array_equal(bundle[key].numpy(), value),
                f'Torch/NPY differ: {key}')
    require(bundle['schema_version'] == pse['schema_version'], 'Torch schema differs')
    mixer = torch.load(safe_path(DESTINATION / (PSE + 'initial_mixer.pt')), map_location='cpu', weights_only=True)
    require(mixer['initial_effective_equals_E0'] is True and mixer['anchor'] == 'base_pse_codebooks.npy',
            'Mixer initial-state identity differs')
    state = mixer['model']
    require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() for value in state.values()),
            'Mixer state contains nonfinite/non-tensor values')
    require(np.array_equal(state['base_codebooks'].numpy(), values['centroids']), 'Mixer E0 differs')
    require(state['residual_scale'].item() == 1.0 and
            torch.count_nonzero(state['attention.out_proj.weight']).item() == 0 and
            torch.count_nonzero(state['attention.out_proj.bias']).item() == 0, 'Mixer is not zero-output initialization')
    with np.load(safe_path(DESTINATION / (PSE + 'pca_transform.npz')), allow_pickle=False) as pca:
        require(all(np.isfinite(pca[key]).all() for key in pca.files), 'PCA contains nonfinite values')
        require(pca['components'].shape == (256, 1024) and pca['mean'].shape == (1024,) and
                pca['explained_variance'].shape == (256,) and np.all(pca['explained_variance'] > 0) and
                int(pca['n_samples']) == N, 'PCA layout differs')
        retained_variance = float(pca['explained_variance_ratio'].sum())
    # Independent implementation: do not import the production encoder or its
    # diagnostic helper. Use FP32 nearest-centroid semantics, then recompute
    # selected reconstruction SSE in FP64 rather than reusing saved errors.
    residual = values['residual_features'].reshape(N, 4, 64)
    books = values['centroids']
    nearest = np.empty_like(codes)
    for start in range(0, N, 128):
        end = min(N, start + 128)
        for digit in range(4):
            delta = residual[start:end, digit, None, :] - books[None, digit, :, :]
            distances = np.sum(delta * delta, axis=2, dtype=np.float32)
            nearest[start:end, digit] = distances.argmin(axis=1)
    require(np.array_equal(nearest, codes), 'Independent nearest-centroid SID recomputation differs')
    reconstruction = books[np.arange(4)[None, :], codes]
    errors = np.square(residual.astype(np.float64) - reconstruction.astype(np.float64)).sum(axis=(1, 2))
    require(np.allclose(errors, values['reconstruction_errors'], rtol=1e-6, atol=1e-7),
            'Independent FP64 reconstruction error differs')
    _, counts = np.unique(codes, axis=0, return_counts=True)
    collision = dict(unique_sids=len(counts), collided_items=int(counts[counts > 1].sum()),
                     colliding_buckets=int((counts > 1).sum()), largest_bucket=int(counts.max()),
                     collision_rate=float(counts[counts > 1].sum() / N))
    for key, value in collision.items():
        require(value == remote_audit['collision'][key], f'Remote collision audit differs: {key}')
        if key in pse['collision']:
            require(value == pse['collision'][key], f'PSE collision summary differs: {key}')
    require(np.isclose(errors.mean(), remote_audit['residual_squared_error_sum']['mean'], rtol=1e-6, atol=1e-7),
            'Mean reconstruction error differs from remote audit')
    return dict(n_items=N, toy=False, codebook_shape=[4, 256, 64], arrays=summaries,
                torch_npy_exact_match=True, initial_mixer_E0_exact=True, data_encoder_binding_match=True,
                sid_nearest_centroid_exact=True, errors_independent_fp64_match=True,
                mean_residual_squared_error_sum=float(errors.mean()),
                max_saved_error_absolute_difference=float(np.max(np.abs(errors - values['reconstruction_errors']))),
                collision=collision, pca_retained_variance_fraction=retained_variance,
                versions=dict(numpy=np.__version__, torch=torch.__version__),
                limitations=['FAISS index bytes verified, not refit or reloaded; GPU execution not tested.',
                             'Dataset binding compared to the saved audit; source parquet files not rehashed here.',
                             'Collision and reconstruction checks are consistency checks, not recommendation-quality acceptance.'])


def main():
    files = restore()
    numeric = audit_numeric(files)
    report = dict(schema='stage09_generated_assets_local_restore_v1', status='PASS',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), destination=str(DESTINATION),
        archive=dict(path=str(ARCHIVE), bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA),
        source_manifest=dict(path=str(SOURCE_MANIFEST), sha256=MANIFEST_SHA),
        file_count=len(files), payload_bytes=sum(row['bytes'] for row in files), files=files,
        numeric_audit=numeric, platform_accessed=False, gpu_used=False, deletes_performed=False,
        script_sha256=digest(Path(__file__))['sha256'])
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with safe_path(REPORT).open('xb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    require(safe_path(REPORT).read_bytes() == payload, 'Report readback differs')
    print(json.dumps(dict(status='PASS', files=len(files), report=str(REPORT),
                         report_sha256=hashlib.sha256(payload).hexdigest()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
