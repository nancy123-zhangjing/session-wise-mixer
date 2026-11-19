#!/usr/bin/env python3
"""Audit retained Stage 05 Beauty inputs for a future-K list prediction proxy.

Uses only the Python standard library, reads local files, and never starts a GPU
or downloads data. By default prints JSON; --output writes the requested report.

Example, from any working directory:
    python audit_beauty_list_data.py --output BEAUTY_DATA_AUDIT_20261026.json

The default input is the retained, verified Stage 05 long-run backup. This audit
does not use Stage 08 outputs. Future-K blocks are NOT observed request sessions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BEAUTY_ROOT = REPO_ROOT / (
    "outputs/stage05_long/extract_r1/goal_20261020_long_mixer_v1/inputs/"
    "data/cache/AmazonReviews2014/Beauty"
)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def nearest_index_quantile(sorted_values: list[int], fraction: float) -> int:
    # Explicitly matches the original read-only audit: Python round is ties-even.
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def audit(beauty_root: Path) -> dict:
    sequence_path = beauty_root / "processed/all_item_seqs.json"
    reviews_path = beauty_root / "raw/reviews_Beauty_5.json.gz"
    with sequence_path.open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError("Expected a nonempty user-to-item-sequence JSON object")
    if any(not isinstance(seq, list) or not seq for seq in sequences.values()):
        raise ValueError("Every processed user must have a nonempty item list")

    lengths = sorted(map(len, sequences.values()))
    length_counts = Counter(lengths)
    fields = Counter()
    raw_by_user = defaultdict(list)
    timestamps = []
    with gzip.open(reviews_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for required in ("reviewerID", "asin", "unixReviewTime"):
                if required not in row:
                    raise ValueError(f"Missing {required!r} on raw line {line_number}")
            timestamp = int(row["unixReviewTime"])
            fields.update(row.keys())
            timestamps.append(timestamp)
            raw_by_user[row["reviewerID"]].append((row["asin"], timestamp))

    gaps = []
    reconstructed = {}
    for user, events in raw_by_user.items():
        # Same stable date sort as AmazonReviews2014._get_item_seqs().
        events.sort(key=lambda event: event[1])
        reconstructed[user] = [item for item, _ in events]
        gaps.extend(right[1] - left[1] for left, right in zip(events, events[1:]))
    same_users = set(reconstructed) == set(sequences)
    matched_users = sum(
        user in reconstructed and reconstructed[user] == seq
        for user, seq in sequences.items()
    )
    if not same_users or matched_users != len(sequences):
        raise ValueError("Processed sequences differ from the stable time-sorted raw reviews")

    proxy_rows = []
    for target_length in (2, 3, 5):
        k = target_length
        eligible_lengths = [n for n in lengths if n >= 2 * k + 2]
        train_counts = [max(n - 3 * k - 1, 0) for n in lengths]
        proxy_rows.append({
            "target_items_K": k,
            "minimum_sequence_length_for_eval": 2 * k + 2,
            "eligible_validation_users": len(eligible_lengths),
            "eligible_test_users": len(eligible_lengths),
            "retained_user_percent": 100 * len(eligible_lengths) / len(lengths),
            "minimum_sequence_length_for_own_training_example": 3 * k + 2,
            "users_with_own_training_example": sum(count > 0 for count in train_counts),
            "eligible_eval_users_without_own_training_example": (
                len(eligible_lengths) - sum(count > 0 for count in train_counts)
            ),
            "training_examples_all_prefixes": sum(train_counts),
            "validation_history_length_median_before_truncation": (
                statistics.median([n - 2 * k for n in eligible_lengths])
                if eligible_lengths else None
            ),
        })

    positive_gaps = [gap for gap in gaps if gap > 0]
    session_field_candidates = [
        key for key in fields
        if any(term in key.lower() for term in ("session", "request", "exposure", "realshow"))
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "mode": "CPU-only standard-library read-only input audit",
            "network_access": False,
            "gpu_started": False,
            "input_origin": "retained Stage 05 long-run backup; no Stage 08 dependency",
            "script": fingerprint(Path(__file__)),
        },
        "inputs": {
            "processed_sequences": fingerprint(sequence_path),
            "raw_reviews_gzip": fingerprint(reviews_path),
        },
        "raw_processed_consistency": {
            "same_user_set": same_users,
            "exactly_matching_user_sequences": matched_users,
            "rule": "stable ascending unixReviewTime sort; ties retain raw-file order",
        },
        "dataset": {
            "users": len(lengths),
            "items": len({item for seq in sequences.values() for item in seq}),
            "interactions": sum(lengths),
            "sequence_length": {
                "minimum": lengths[0],
                "median": statistics.median(lengths),
                "mean": statistics.mean(lengths),
                "p75": nearest_index_quantile(lengths, 0.75),
                "p90": nearest_index_quantile(lengths, 0.90),
                "p95": nearest_index_quantile(lengths, 0.95),
                "maximum": lengths[-1],
                "quantile_rule": "sorted_values[round((N - 1) * p)]; Python ties-even round",
                "histogram": {str(n): count for n, count in sorted(length_counts.items())},
            },
        },
        "raw_schema_and_time": {
            "records": len(timestamps),
            "field_record_counts": dict(sorted(fields.items())),
            "session_request_exposure_field_candidates": session_field_candidates,
            "timestamps_multiple_of_86400": sum(t % 86400 == 0 for t in timestamps),
            "unique_timestamps": len(set(timestamps)),
            "adjacent_user_event_pairs": len(gaps),
            "same_timestamp_adjacent_pairs": gaps.count(0),
            "median_adjacent_gap_days": statistics.median(gaps) / 86400 if gaps else None,
            "minimum_positive_adjacent_gap_days": min(positive_gaps) / 86400 if positive_gaps else None,
        },
        "future_K_proxy_protocol": {
            "status": "hypothetical counting only; no split files or training produced",
            "target_semantics": "K consecutive future review interactions; not a real request session",
            "indexing": "zero-based, half-open Python slices; n is full user sequence length",
            "test_target": "seq[n-K:n]",
            "validation_target": "seq[n-2*K:n-K]",
            "training_prefix": "seq[:n-2*K]",
            "validation_history": "seq[:n-2*K][-50:]",
            "test_history": "seq[:n-K][-50:] (includes validation events as observed past; no refit)",
            "eval_user_eligibility": "n >= 2*K+2, so validation has at least 2 history items",
            "training_example": "history=seq[max(0,h-50):h], target=seq[h:h+K]",
            "training_prefix_positions": "h ranges from 2 through n-3*K inclusive; stride 1",
            "training_count_per_user": "max(n-3*K-1, 0)",
            "history_limit": 50,
            "prefix_enumeration": "all eligible history endpoints; not capped at the first 50 endpoints",
            "own_training_example_required_for_eval": False,
            "duplicate_policy": "counts review events as stored; no item deduplication",
            "same_date_policy": "retains existing stable order; does not claim within-day temporal order",
            "rows": proxy_rows,
        },
        "interpretation_limits": [
            "Raw reviews contain no recorded recommendation request or exposure list.",
            "All observed timestamps have day granularity; sub-day sessionization is unsupported.",
            "Future-K blocks cannot establish effects on genuine OneRec-style same-request sessions.",
            "Eligibility changes the evaluated user cohort, especially for K=5.",
            "Some eligible evaluation users have no own training window under this proposed split.",
            "New held-out blocks can overlap old training labels: do not reuse old behavior-trained checkpoints without a leakage audit.",
            "This is an input audit and sample-count proposal, not a completed experiment or an approved dataset protocol.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beauty-root", type=Path, default=DEFAULT_BEAUTY_ROOT)
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print it")
    args = parser.parse_args()
    result = audit(args.beauty_root.resolve())
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        destination = args.output.resolve()
        protected = {
            Path(result["inputs"]["processed_sequences"]["path"]),
            Path(result["inputs"]["raw_reviews_gzip"]["path"]),
            Path(__file__).resolve(),
        }
        if destination in protected:
            raise ValueError("Refusing to overwrite an input or the audit script")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
