"""Analyze semantic-SID collisions, full-ID uniqueness, and position entropy."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def collision_stats(rows: np.ndarray) -> dict[str, object]:
    ids = [tuple(map(int, row)) for row in rows]
    counts = Counter(ids)
    collided_items = sum(count for count in counts.values() if count > 1)
    group_histogram = Counter(count for count in counts.values() if count > 1)
    return {
        "items": len(ids),
        "unique_ids": len(counts),
        "collided_items": collided_items,
        "colliding_groups": sum(count > 1 for count in counts.values()),
        "collision_rate": collided_items / max(len(ids), 1),
        "largest_collision_group": max(counts.values(), default=0),
        "collision_group_size_histogram": {
            str(size): groups for size, groups in sorted(group_histogram.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        help="Optional per-position sizes, e.g. 128 256 512 256.",
    )
    parser.add_argument(
        "--semantic-digits",
        type=int,
        help="Number of leading semantic digits; remaining digit is the collision suffix.",
    )
    parser.add_argument(
        "--require-unique-full",
        action="store_true",
        help="Exit with an error when the complete SID mapping is not one-to-one.",
    )
    args = parser.parse_args()

    item_to_tokens = np.load(args.input)
    # Row zero is the padding pseudo-item in the selected upstream.
    full_rows = item_to_tokens[1:]
    total_digits = full_rows.shape[1]
    semantic_digits = args.semantic_digits or total_digits
    if not 1 <= semantic_digits <= total_digits:
        raise ValueError(
            f"semantic-digits must be in [1, {total_digits}], got {semantic_digits}"
        )
    semantic_rows = full_rows[:, :semantic_digits]

    codebook_sizes = args.codebook_sizes or [args.codebook_size] * total_digits
    if len(codebook_sizes) != total_digits:
        raise ValueError(
            f"Expected {total_digits} codebook sizes, got {codebook_sizes}"
        )

    per_position = []
    for position in range(total_digits):
        codebook_size = int(codebook_sizes[position])
        _, position_counts = np.unique(full_rows[:, position], return_counts=True)
        probabilities = position_counts / position_counts.sum()
        entropy_nats = float(-(probabilities * np.log(probabilities)).sum())
        unique_codes = int(position_counts.size)
        per_position.append(
            {
                "position": position + 1,
                "role": "semantic" if position < semantic_digits else "collision_suffix",
                "codebook_size": codebook_size,
                "unique_codes": unique_codes,
                "unused_codes": codebook_size - unique_codes,
                "utilization_rate": unique_codes / codebook_size,
                "entropy_nats": entropy_nats,
                "normalized_entropy": entropy_nats / math.log(codebook_size),
                "effective_codes_perplexity": math.exp(entropy_nats),
                "largest_code_share": float(position_counts.max() / position_counts.sum()),
            }
        )

    result = {
        "input": str(args.input),
        "semantic_digits": semantic_digits,
        "total_digits": total_digits,
        "codebook_sizes": [int(size) for size in codebook_sizes],
        "semantic_sid": collision_stats(semantic_rows),
        "full_sid": collision_stats(full_rows),
        "per_position": per_position,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_unique_full and result["full_sid"]["collided_items"]:
        raise SystemExit("Full SID uniqueness check failed")


if __name__ == "__main__":
    main()
