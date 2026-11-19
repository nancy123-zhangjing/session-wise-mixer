"""Fit dataset-bound PCA/PSE from already-generated BGE-M3 dense vectors.

Run separately from training. This command never downloads an encoder. Use
``encode_text`` to prepare the input files explicitly. Full catalog fitting is
the declared transductive static-content assumption, not use of held-out labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .assets import (catalog_original_ids, dataset_binding, fit_original_pse,
                     normalized_pca, sha256, validate_embedding_metadata, write_assets)


def prepare_assets(dataset_dir: Path, embeddings_path: Path, item_ids_path: Path,
                   metadata_path: Path, output_dir: Path, *, seed: int = 2024,
                   omp_threads: int = 4):
    root = Path(dataset_dir)
    if Path(output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite existing asset directory: {output_dir}")
    binding = dataset_binding(root)
    original = catalog_original_ids(root)
    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    item_ids = np.load(item_ids_path, allow_pickle=False)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if values.ndim != 2 or len(values) != len(original) or not np.array_equal(item_ids, original):
        raise ValueError("Embedding row count/order does not match the dataset's original item IDs")
    validate_embedding_metadata(metadata, binding, values.shape[1])
    for key, path in (("embeddings_sha256", embeddings_path), ("original_item_ids_sha256", item_ids_path)):
        if metadata.get(key) != sha256(Path(path)):
            raise ValueError(f"Embedding metadata/file hash mismatch: {key}")
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
        raise ValueError("Expected finite, L2-normalized BGE-M3 dense vectors")
    # Check optional dependency before fitting PCA or creating any output.
    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("CPU faiss is required for asset fitting; no dependencies were installed") from exc
    content, pca = normalized_pca(values, dimensions=256, seed=seed)
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="listwise-pse-", dir=Path(output_dir).parent) as temp:
        index = Path(temp) / "base_pse.faiss"
        residual, books, codes, center = fit_original_pse(
            content, index, n_digit=4, codebook_size=256, omp_threads=omp_threads)
        assets = write_assets(output_dir, binding=binding, original_item_ids=original,
                              content_features=content, residual_features=residual,
                              centroids=books, coarse_centroid=center,
                              encoder_metadata=metadata, pca_transform=pca,
                              source_index=index, seed=seed)
        if not np.array_equal(assets.catalog_sids[1:].numpy(), codes):
            raise ValueError("Persisted asset differs from original FAISS code extraction")
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-item-ids", type=Path, required=True)
    parser.add_argument("--embedding-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--omp-threads", type=int, default=4)
    args = parser.parse_args()
    assets = prepare_assets(args.dataset, args.embeddings, args.embedding_item_ids,
                            args.embedding_metadata, args.output, seed=args.seed,
                            omp_threads=args.omp_threads)
    print(json.dumps({"assets": str(assets.root), "items": assets.n_items,
                      "collision": assets.manifest["collision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
