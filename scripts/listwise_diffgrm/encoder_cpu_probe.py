"""Encode 32 real catalog texts on CPU and estimate a conservative full pass.

This diagnostic never starts the full encoding job. Its report can inform that
decision only after the execution code has been committed and locked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .assets import dataset_binding, sha256
from .encode_text import _make_encoder, catalog_texts, untruncated_token_lengths
from .model_manifest import MODEL_ID, verify_snapshot


def probe(dataset: Path, model_path: Path, manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new CPU probe output directory")
    binding = dataset_binding(dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID:
        raise ValueError("Unexpected encoder model")
    verified = verify_snapshot(model_path, manifest["files"])
    available_before = psutil.virtual_memory().available
    # Float32 568M model plus loading/intermediate memory: fail rather than swap.
    if available_before < 6 * 1024 ** 3:
        raise RuntimeError("CPU probe needs at least 6 GiB available memory before loading BGE-M3")
    torch.set_num_threads(4)
    original, texts, text_hash = catalog_texts(dataset)
    load_started = time.perf_counter()
    model = _make_encoder(manifest["revision"], "cpu", False, model_path)
    load_seconds = time.perf_counter() - load_started
    lengths = []
    for start in range(0, len(texts), 128):
        lengths.extend(untruncated_token_lengths(model, texts[start:start + 128]))
    lengths = np.asarray(lengths, dtype=np.int64)
    count = min(32, len(texts))
    sorted_positions = np.argsort(lengths, kind="stable")
    quantiles = (np.arange(count) + .5) / count
    indices = sorted_positions[np.minimum((quantiles * len(texts)).astype(int), len(texts) - 1)]
    selected = [texts[int(i)] for i in indices]
    started = time.perf_counter()
    vectors = model.encode(selected, batch_size=8, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False, prompt="")
    elapsed = time.perf_counter() - started
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (count, 1024) or not np.isfinite(vectors).all():
        raise ValueError("Real CPU probe returned invalid embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Real CPU probe returned non-normalized embeddings")
    full_seconds = load_seconds + elapsed / count * len(texts)
    conservative_seconds = load_seconds + 1.3 * elapsed / count * len(texts)
    memory = psutil.Process().memory_info()
    output.mkdir(parents=True)
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    np.save(output / "original_item_ids.npy", original[indices], allow_pickle=False)
    report = {
        "schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_binding": binding, "ordered_text_sha256": text_hash,
        "model_id": MODEL_ID, "revision": manifest["revision"], "verified_model_files": verified,
        "device": "cpu", "torch_threads": 4, "cuda_initialized": torch.cuda.is_initialized(),
        "batch_size": 8, "probe_items": count, "catalog_items": len(texts),
        "selection": "32 midpoint quantiles of actual untruncated catalog token lengths",
        "selected_internal_item_ids": (indices + 1).tolist(), "token_lengths": lengths[indices].tolist(),
        "catalog_length_quantiles": np.quantile(lengths, [.5, .95, .99]).tolist(),
        "catalog_truncated_items": int((lengths > 512).sum()), "max_length": 512,
        "load_seconds": load_seconds, "encode_seconds": elapsed,
        "items_per_second": count / elapsed, "estimated_full_minutes": full_seconds / 60,
        "conservative_full_minutes_30percent_margin": conservative_seconds / 60,
        "available_memory_before_bytes": available_before, "rss_after_bytes": memory.rss,
        "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
        "memory_available_after_bytes": psutil.virtual_memory().available,
        "cpu_90minute_gate_passed": conservative_seconds <= 90 * 60,
        "full_encoding_started": False, "needs_locked_execution_commit_before_full_encoding": True,
        "vectors_sha256": sha256(output / "vectors.npy"), "passed": True,
    }
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = probe(args.dataset, args.model_path, args.manifest, args.output)
    print(json.dumps({key: report[key] for key in ("passed", "encode_seconds", "items_per_second",
                     "conservative_full_minutes_30percent_margin", "cpu_90minute_gate_passed")}))


if __name__ == "__main__":
    main()
