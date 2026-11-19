"""Fail fast if a frozen DiffGRM RQ-VAE tokenizer artifact is inconsistent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--n-digit", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--min-utilization", type=float, default=0.25)
    parser.add_argument("--min-unique-ratio", type=float, default=0.5)
    args = parser.parse_args()

    root = args.artifact_dir.resolve()
    required = (
        "best.pt", "semantic_ids.json", "semantic_ids.npy",
        "reconstructed_embeddings.npy", "summary.json", "training.csv",
        "geometry_summary.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete RQ-VAE artifact; missing={missing}")

    items = json.loads(args.items.resolve().read_text(encoding="utf-8"))
    mapping = json.loads((root / "semantic_ids.json").read_text(encoding="utf-8"))
    codes = np.load(root / "semantic_ids.npy")
    reconstructed = np.load(root / "reconstructed_embeddings.npy", mmap_mode="r")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    if list(mapping) != items:
        raise ValueError("RQ-VAE semantic-ID item order/catalog mismatch")
    if codes.shape != (len(items), args.n_digit):
        raise ValueError(
            f"RQ-VAE code shape {codes.shape} != {(len(items), args.n_digit)}"
        )
    json_codes = np.asarray([mapping[item] for item in items], dtype=np.int64)
    if not np.array_equal(codes, json_codes):
        raise ValueError("semantic_ids.npy and semantic_ids.json disagree")
    if codes.min() < 0 or codes.max() >= args.codebook_size:
        raise ValueError("RQ-VAE codes fall outside the configured codebook")
    if reconstructed.shape[0] != len(items):
        raise ValueError("RQ-VAE reconstructed embedding row count mismatch")
    if int(summary.get("n_digit", -1)) != args.n_digit:
        raise ValueError("RQ-VAE summary n_digit mismatch")
    if int(summary.get("codebook_size", -1)) != args.codebook_size:
        raise ValueError("RQ-VAE summary codebook_size mismatch")
    actual_hash = sha256(root / "semantic_ids.json")
    if summary.get("semantic_ids_sha256") != actual_hash:
        raise ValueError("RQ-VAE semantic-ID SHA256 mismatch")

    unique_ratio = len({tuple(map(int, row)) for row in codes}) / len(codes)
    utilizations = [
        len(np.unique(codes[:, position])) / args.codebook_size
        for position in range(args.n_digit)
    ]
    if unique_ratio < args.min_unique_ratio:
        raise ValueError(
            f"Collapsed RQ-VAE: unique SID ratio {unique_ratio:.4f}"
        )
    if min(utilizations) < args.min_utilization:
        raise ValueError(
            f"Collapsed RQ-VAE: code utilization {utilizations}"
        )

    result = {
        "artifact_dir": str(root),
        "items": len(items),
        "shape": list(codes.shape),
        "semantic_ids_sha256": actual_hash,
        "unique_sid_ratio": unique_ratio,
        "per_position_utilization": utilizations,
        "status": "OK",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("DiffGRM frozen RQ-VAE artifact: OK")


if __name__ == "__main__":
    main()
