"""Measure geometry preservation and code statistics for a frozen tokenizer."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reconstructed", required=True, type=Path)
    parser.add_argument("--semantic-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        help="Per-position codebook sizes; overrides --codebook-size.",
    )
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--neighbor-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def neighbor_recall(
    source: np.ndarray,
    reconstructed: np.ndarray,
    k: int,
    n_queries: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    query_ids = rng.choice(len(source), min(n_queries, len(source)), replace=False)
    source_index = faiss.IndexFlatIP(source.shape[1])
    reconstructed_index = faiss.IndexFlatIP(reconstructed.shape[1])
    source_index.add(source)
    reconstructed_index.add(reconstructed)
    _, source_neighbors = source_index.search(source[query_ids], k + 1)
    _, reconstructed_neighbors = reconstructed_index.search(reconstructed[query_ids], k + 1)
    scores = []
    for query_id, left, right in zip(query_ids, source_neighbors, reconstructed_neighbors):
        left_list = [int(value) for value in left if int(value) != int(query_id)][:k]
        right_list = [int(value) for value in right if int(value) != int(query_id)][:k]
        scores.append(len(set(left_list) & set(right_list)) / k)
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    source = np.ascontiguousarray(np.load(args.source).astype(np.float32, copy=False))
    reconstructed = np.ascontiguousarray(
        np.load(args.reconstructed).astype(np.float32, copy=False)
    )
    codes = np.load(args.semantic_ids)
    if source.shape != reconstructed.shape or len(source) != len(codes):
        raise ValueError(
            f"Shape mismatch: source={source.shape}, reconstructed={reconstructed.shape}, "
            f"codes={codes.shape}"
        )

    source /= np.linalg.norm(source, axis=1, keepdims=True) + 1e-12
    reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-12
    tuples = [tuple(map(int, row)) for row in codes]
    counts = Counter(tuples)
    codebook_sizes = args.codebook_sizes or [args.codebook_size] * codes.shape[1]
    if len(codebook_sizes) != codes.shape[1]:
        raise ValueError(
            f"Expected {codes.shape[1]} codebook sizes, got {codebook_sizes}"
        )
    per_position = []
    for position in range(codes.shape[1]):
        codebook_size = int(codebook_sizes[position])
        _, position_counts = np.unique(codes[:, position], return_counts=True)
        probabilities = position_counts / position_counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        per_position.append(
            {
                "position": position + 1,
                "codebook_size": codebook_size,
                "unique_codes": int(len(position_counts)),
                "unused_codes": int(codebook_size - len(position_counts)),
                "utilization_rate": float(len(position_counts) / codebook_size),
                "entropy_nats": entropy,
                "normalized_entropy": float(entropy / math.log(codebook_size)),
                "effective_codes_perplexity": float(math.exp(entropy)),
                "largest_code_share": float(position_counts.max() / position_counts.sum()),
            }
        )
    collided_items = sum(size for size in counts.values() if size > 1)
    result = {
        "source": str(args.source),
        "reconstructed": str(args.reconstructed),
        "semantic_ids": str(args.semantic_ids),
        "codebook_sizes": [int(size) for size in codebook_sizes],
        "items": len(source),
        "reconstruction_mse_sum": float(np.mean(np.sum((source - reconstructed) ** 2, axis=1))),
        "reconstruction_cosine": float(np.mean(np.sum(source * reconstructed, axis=1))),
        f"neighbor_recall_at_{args.neighbor_k}": neighbor_recall(
            source,
            reconstructed,
            args.neighbor_k,
            args.neighbor_queries,
            args.seed,
        ),
        "unique_semantic_ids": len(counts),
        "collided_items": collided_items,
        "collision_rate": collided_items / max(len(source), 1),
        "colliding_groups": sum(size > 1 for size in counts.values()),
        "largest_collision_group": max(counts.values(), default=0),
        "per_position": per_position,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
