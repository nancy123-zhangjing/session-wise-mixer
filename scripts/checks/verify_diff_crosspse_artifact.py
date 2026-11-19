"""Verify that a frozen Cross-PSE artifact is complete and row-aligned."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--n-digit", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--reference-pse-semantic-ids", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir.resolve()
    required = [
        "semantic_ids.json",
        "semantic_ids.npy",
        "base_pse_semantic_ids.json",
        "base_pse_semantic_ids.npy",
        "base_pse_codebooks.npy",
        "base_pse_coarse_centroid.npy",
        "base_pse.faiss",
        "opq_embeddings.npy",
        "opq_residual_embeddings.npy",
        "reconstructed_opq_embeddings.npy",
        "attention_by_book.npy",
        "best.pt",
        "training.csv",
        "summary.json",
    ]
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Cross-PSE artifact: {missing}")

    items = json.loads(args.items.resolve().read_text(encoding="utf-8"))
    semantic_path = artifact / "semantic_ids.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    base_semantic = json.loads(
        (artifact / "base_pse_semantic_ids.json").read_text(encoding="utf-8")
    )
    if list(semantic) != items or list(base_semantic) != items:
        raise ValueError("Cross-PSE semantic-id rows are not aligned to item keys")

    codes = np.load(artifact / "semantic_ids.npy")
    base_codes = np.load(artifact / "base_pse_semantic_ids.npy")
    expected_shape = (len(items), args.n_digit)
    if codes.shape != expected_shape or base_codes.shape != expected_shape:
        raise ValueError(
            f"Bad SID shape: Cross-PSE={codes.shape}, base={base_codes.shape}, "
            f"expected={expected_shape}"
        )
    if codes.min() < 0 or codes.max() >= args.codebook_size:
        raise ValueError("Cross-PSE contains an out-of-range code")
    json_codes = np.asarray([semantic[item] for item in items], dtype=np.int64)
    if not np.array_equal(codes, json_codes):
        raise ValueError("semantic_ids.json and semantic_ids.npy disagree")
    reference_match = None
    if args.reference_pse_semantic_ids:
        reference = json.loads(
            args.reference_pse_semantic_ids.resolve().read_text(encoding="utf-8")
        )
        if set(reference) != set(items):
            raise ValueError("Reference PSE semantic ids do not contain the same items")
        reference_codes = np.asarray([reference[item] for item in items], dtype=np.int64)
        reference_match = float(np.mean(reference_codes == base_codes))
        if reference_match < 0.999999:
            raise ValueError(
                "Recovered base PSE does not match the existing PSE baseline "
                f"(digit match rate={reference_match:.8f})"
            )

    summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    if summary.get("tokenizer_type") not in ("cross_pse", "refined_pse"):
        raise ValueError("summary.json is not a PSE-refinement summary")
    if int(summary.get("n_digit", -1)) != args.n_digit:
        raise ValueError("summary.json n_digit mismatch")
    if int(summary.get("codebook_size", -1)) != args.codebook_size:
        raise ValueError("summary.json codebook_size mismatch")
    actual_hash = sha256(semantic_path)
    if summary.get("semantic_ids_sha256") != actual_hash:
        raise ValueError("semantic_ids.json SHA256 does not match summary.json")

    for name in ("opq_embeddings.npy", "reconstructed_opq_embeddings.npy"):
        array = np.load(artifact / name, mmap_mode="r")
        if array.shape[0] != len(items) or not np.isfinite(array).all():
            raise ValueError(f"Invalid array: {name} shape={array.shape}")

    print(f"{summary.get('tokenizer_type')} artifact: OK")
    print(f"items: {len(items)}")
    print(f"layout: {args.n_digit} x {args.codebook_size}")
    print(f"best step: {summary.get('best_step')}")
    print(f"residual scale: {summary.get('residual_scale')}")
    print(f"validation items: {summary.get('validation_items')}")
    print(
        "relative codebook delta RMS: "
        f"{summary.get('relative_codebook_delta_rms')}"
    )
    if reference_match is not None:
        print(f"existing PSE baseline digit match rate: {reference_match:.8f}")
    print(
        "item SID change rate from base: "
        f"{summary.get('item_sid_change_rate_from_base')}"
    )
    print(f"semantic_ids SHA256: {actual_hash}")


if __name__ == "__main__":
    main()
