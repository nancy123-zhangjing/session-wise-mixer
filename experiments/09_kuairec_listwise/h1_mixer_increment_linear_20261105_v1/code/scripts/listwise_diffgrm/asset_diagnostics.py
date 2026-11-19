"""Read-only CPU diagnostics of a completed dataset-bound PSE asset bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .assets import encode_residuals, load_assets, sha256


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require finite nonempty values")
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99))}


def collision_statistics(codes: np.ndarray, codebook_size: int) -> dict:
    codes = np.asarray(codes, dtype=np.int64)
    _, counts = np.unique(codes, axis=0, return_counts=True)
    digits = []
    prefixes = []
    for digit in range(codes.shape[1]):
        usage = np.bincount(codes[:, digit], minlength=codebook_size)
        probabilities = usage[usage > 0] / len(codes)
        digits.append({"digit": digit, "used_codes": int((usage > 0).sum()),
                       "usage": usage.tolist(), "largest_code_count": int(usage.max()),
                       "normalized_entropy": float(-(probabilities * np.log(probabilities)).sum() / np.log(codebook_size))})
        _, prefix_counts = np.unique(codes[:, :digit + 1], axis=0, return_counts=True)
        prefixes.append({"length": digit + 1, "unique_prefixes": len(prefix_counts),
                         "largest_bucket": int(prefix_counts.max())})
    return {"items": len(codes), "unique_sids": len(counts),
            "collided_items": int(counts[counts > 1].sum()),
            "collision_rate": float(counts[counts > 1].sum() / len(codes)),
            "colliding_buckets": int((counts > 1).sum()), "largest_bucket": int(counts.max()),
            "per_digit": digits, "per_prefix": prefixes}


def audit_assets(asset_path: Path, dataset_dir: Path | None = None) -> dict:
    assets = load_assets(asset_path, dataset_dir)
    books = assets.centroids.numpy()
    residual = assets.residual_features[1:].numpy()
    stored_codes = assets.catalog_sids[1:].numpy()
    nearest, errors = encode_residuals(residual, books)
    if not np.array_equal(nearest, stored_codes):
        raise ValueError("Asset SID codes do not equal nearest centroids in residual space")
    if not np.allclose(errors, assets.reconstruction_errors[1:].numpy(), rtol=1e-6, atol=1e-7):
        raise ValueError("Saved reconstruction errors do not match independent residual recomputation")
    reconstructed = books[np.arange(books.shape[0])[None], stored_codes].reshape(assets.n_items, -1)
    center = np.load(assets.root / "coarse_centroid.npy", allow_pickle=False)
    original_opq = residual.reshape(assets.n_items, -1) + center
    reconstructed_opq = reconstructed + center
    denominators = np.linalg.norm(original_opq, axis=1) * np.linalg.norm(reconstructed_opq, axis=1)
    cosine = (original_opq * reconstructed_opq).sum(1) / np.maximum(denominators, 1e-12)
    collision = collision_statistics(stored_codes, books.shape[1])
    manifest_collision = assets.manifest["collision"]
    for name in ("unique_sids", "collided_items", "largest_bucket"):
        if collision[name] != manifest_collision[name]:
            raise ValueError(f"Collision summary mismatch: {name}")
    result = {
        "schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_manifest_sha256": sha256(assets.root / "manifest.json"),
        "dataset_binding": assets.manifest["dataset_binding"], "toy": assets.manifest["toy"],
        "n_items": assets.n_items, "codebook_shape": list(books.shape),
        "finite": True, "nearest_sid_exact_match": True, "error_recompute_matches": True,
        "residual_squared_error_sum": distribution(errors),
        "residual_squared_error_per_dimension": distribution(errors / reconstructed.shape[1]),
        "opq_reconstruction_cosine": distribution(cosine),
        "original_opq_norm": distribution(np.linalg.norm(original_opq, axis=1)),
        "content_norm": distribution(assets.content_features[1:].norm(dim=-1).numpy()),
        "coarse_centroid_norm": float(np.linalg.norm(center)),
        "collision": collision, "encoder": assets.manifest["encoder"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    for package in ("faiss-cpu", "scikit-learn", "sentence-transformers", "transformers", "pyarrow"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None
    pca_path = assets.root / "pca_transform.npz"
    if pca_path.exists():
        with np.load(pca_path, allow_pickle=False) as pca:
            variance = pca["explained_variance"]
            if not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError("PCA has nonpositive/nonfinite retained explained variance")
            result["pca"] = {"component_shape": list(pca["components"].shape),
                             "retained_explained_variance": distribution(variance)}
            if "explained_variance_ratio" in pca:
                retained = float(pca["explained_variance_ratio"].sum())
                result["pca"].update({
                    "retained_variance_fraction": retained,
                    "discarded_variance_fraction": max(1.0 - retained, 0.0),
                    "normalized_dense_inverse_reconstruction_squared_error_mean":
                        float(pca["input_reconstruction_squared_error_mean"]),
                    "interpretation": "Variance/reconstruction loss before post-PCA L2 normalization; not semantic relevance retention.",
                })
    if dataset_dir is not None:
        seen = set(json.loads((Path(dataset_dir) / "train_seen.json").read_text(encoding="utf-8"))["item_ids"])
        groups = {"behavior_train_seen": np.asarray([i in seen for i in range(1, assets.n_items + 1)]),
                  "behavior_train_unseen": np.asarray([i not in seen for i in range(1, assets.n_items + 1)])}
        result["behavior_groups"] = {
            name: {"items": int(mask.sum()), "residual_error": distribution(errors[mask]) if mask.any() else None}
            for name, mask in groups.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic output exists; choose a new evidence path")
    result = audit_assets(args.assets, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": result["n_items"], "toy": result["toy"],
                      "collision_rate": result["collision"]["collision_rate"],
                      "reconstruction_error": result["residual_squared_error_sum"]["mean"],
                      "report": str(args.output)}))


if __name__ == "__main__":
    main()
