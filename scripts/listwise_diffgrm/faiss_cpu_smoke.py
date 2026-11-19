"""Small synthetic CPU check of real FAISS and parity with the previous fitter.

This does not encode KuaiRec or train a recommender. Its output is explicitly a
synthetic diagnostic, never a replacement for the four groups' shared assets.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .assets import encode_residuals, fit_original_pse, normalized_pca, sha256
from .asset_diagnostics import collision_statistics, distribution


def original_fitter():
    """Load only the unchanged standalone function, avoiding unrelated imports."""
    path = Path(__file__).resolve().parents[1] / "tokenizers/train_cross_pse_tokenizer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "fit_original_pse")
    isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"np": np, "math": math, "Path": Path}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["fit_original_pse"], sha256(path)


def run_smoke(output: Path, *, items: int = 1024, threads: int = 4) -> dict:
    import faiss
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new synthetic smoke directory")
    if items < 256:
        raise ValueError("OPQ has 256 internal clusters and needs at least 256 vectors")
    output.mkdir(parents=True)
    started = time.perf_counter()
    dense = np.random.default_rng(2024).normal(size=(items, 32)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    content, pca = normalized_pca(dense, dimensions=16)
    residual, books, codes, center = fit_original_pse(content, output / "new.faiss", omp_threads=threads)
    old_fitter, old_source_hash = original_fitter()
    old = old_fitter(content, np.ones(items, dtype=bool), 4, 256, threads, output / "original.faiss")
    matches = {name: bool(np.array_equal(new_value, old_value))
               for name, new_value, old_value in zip(
                   ("residuals", "centroids", "codes", "coarse_center"),
                   (residual, books, codes, center), old)}
    if not all(matches.values()):
        raise AssertionError(f"New/original PSE fitter parity failed: {matches}")
    nearest, errors = encode_residuals(residual, books)
    if not np.array_equal(nearest, codes):
        raise AssertionError("Residual nearest-centroid check failed")
    index = faiss.read_index(str(output / "new.faiss"))
    transform = faiss.downcast_VectorTransform(index.chain.at(0))
    decoded_index = faiss.downcast_index(index.index)
    reconstructed = books[np.arange(4)[None], codes].reshape(items, -1) + center
    transformed = transform.apply_py(content)
    cosine = (transformed * reconstructed).sum(1) / (
        np.linalg.norm(transformed, axis=1) * np.linalg.norm(reconstructed, axis=1) + 1e-12)
    np.savez(output / "synthetic_arrays.npz", dense=dense, content=content, residual=residual,
             codebooks=books, sid=codes, coarse_center=center, **{f"pca_{k}": v for k, v in pca.items()})
    report = {
        "schema_version": 1, "synthetic": True, "actual_kuairec_assets": False,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": items, "dense_dimension": 32, "pca_dimension": 16, "codebook_shape": list(books.shape),
        "factory": "OPQ4,IVF1,PQ4x8", "faiss_distribution": importlib.metadata.version("faiss-cpu"),
        "faiss_module_version": faiss.__version__, "faiss_import_path": faiss.__file__,
        "numpy": np.__version__, "threads": threads,
        "opq_d_in": transform.d_in, "opq_d_out": transform.d_out,
        "metric_inner_product": decoded_index.metric_type == faiss.METRIC_INNER_PRODUCT,
        "by_residual": bool(decoded_index.by_residual),
        "old_fitter_source_sha256": old_source_hash, "exact_parity": matches,
        "nearest_sid_exact_match": True,
        "residual_error_sum": distribution(errors), "opq_reconstruction_cosine": distribution(cosine),
        "collision": collision_statistics(codes, 256),
        "faiss_index_sha256": sha256(output / "new.faiss"),
        "elapsed_seconds": time.perf_counter() - started,
        "small_sample_warning": "Synthetic 1024-item smoke is below FAISS's recommended 9984 samples; formal 10728-item catalog exceeds it.",
        "passed": True,
    }
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run_smoke(args.output, items=args.items, threads=args.threads)
    print(json.dumps({"passed": result["passed"], "items": result["items"],
                      "exact_parity": result["exact_parity"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
