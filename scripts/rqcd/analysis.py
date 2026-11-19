"""Analyze new saved predictions only; never loads a model or reruns old statistics."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .common import output_lock, sha256, write_json

TEMPORAL_CUT = 1598745441188500


def read_predictions(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate prediction IDs")
    return rows


def clustered_interval(values, users, repeats=2000, seed=2024):
    """Resample users, retain all rows; estimate sample-weighted mean."""
    values = np.asarray(values, dtype=np.float64)
    _, inverse = np.unique(users, return_inverse=True)
    n = inverse.max()+1
    counts = np.bincount(inverse, minlength=n)
    sums = np.bincount(inverse, weights=values, minlength=n)
    rng, replicates = np.random.default_rng(seed), []
    for start in range(0, repeats, 64):
        draws = rng.integers(n, size=(min(64, repeats-start), n))
        replicates.extend((sums[draws].sum(1)/counts[draws].sum(1)).tolist())
    return {"estimate": float(values.mean()),
            "ci95": np.quantile(replicates, [.025, .975]).tolist(),
            "users": int(n), "samples": int(len(values)), "repeats": repeats, "seed": seed}


def analyze(rows, reference=None):
    users = np.asarray([r["user_id"] for r in rows])
    result = {"metrics": {}, "sample_mean_not_user_mean": True}
    for key in ("ndcg@5", "hr@5", "recall@5"):
        result["metrics"][key] = clustered_interval([r["metrics"][key] for r in rows], users)
    predicted = np.asarray([r["prediction"] for r in rows], dtype=np.int64)
    items, counts = np.unique(predicted[predicted > 0], return_counts=True)
    result["recommendation_distribution"] = {
        "distinct_items": len(items),
        "top10_position_share": float(np.sort(counts)[-10:].sum()/max(counts.sum(), 1)),
        "denominator": "non-PAD output positions, repeated across samples",
    }
    if reference is not None:
        index = {r["sample_id"]: r for r in reference}
        if set(index) != {r["sample_id"] for r in rows}:
            raise ValueError("Reference sample set differs; no partial pairing allowed")
        aligned = [index[r["sample_id"]] for r in rows]
        for current, old in zip(rows, aligned):
            if current["user_id"] != old["user_id"] or current["target"] != old["target"]:
                raise ValueError("Reference user/ordered-target identity mismatch")
        result["paired_reference"] = {}
        for key in ("ndcg@5", "hr@5", "recall@5"):
            difference = [r["metrics"][key]-o["metrics"][key] for r, o in zip(rows, aligned)]
            result["paired_reference"][key] = clustered_interval(difference, users)
        result["comparison_limit"] = "Historical beam32 and new single sample have different search budgets"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--reference", type=Path, help="Already saved, explicitly aligned prediction table")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = read_predictions(args.predictions)
    reference = read_predictions(args.reference) if args.reference else None
    result = analyze(rows, reference)
    result["source_sha256"] = sha256(args.predictions)
    result["reference_sha256"] = sha256(args.reference) if args.reference else None
    if args.split == "validation":
        result["temporal"] = {}
        for name, condition in (("before", lambda t: t < TEMPORAL_CUT), ("after", lambda t: t >= TEMPORAL_CUT)):
            part = [r for r in rows if condition(r["first_target_timestamp_us"])]
            if not part:
                raise ValueError("Missing temporal segment")
            chosen = {r["sample_id"] for r in part}
            old = [r for r in reference if r["sample_id"] in chosen] if reference else None
            result["temporal"][name] = analyze(part, old)
        result["temporal_cut"] = TEMPORAL_CUT
    with output_lock(args.output) as root:
        write_json(root/"ANALYSIS.json", result)


if __name__ == "__main__":
    main()
