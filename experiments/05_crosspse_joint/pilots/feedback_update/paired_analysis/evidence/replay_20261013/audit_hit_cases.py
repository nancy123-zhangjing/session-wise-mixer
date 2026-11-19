"""Preserve a local, SID-level case audit; never modify or overwrite evidence.

Run with Python only. This script does not import torch/numpy, invoke a remote
service, alter any model, or claim strict item-level hits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


ARMS = ("frozen", "cross_recon", "cross_rec")
EXPECTED_COUNT = 22363
EXPECTED_TARGET_CASES = {
    3252: (2390, (9, 0, 8)),
    9897: (814, (3, 3, 0)),
    11236: (2965, (0, 0, 6)),
    15829: (1225, (3, 0, 3)),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def canonical_history(row):
    values = row["history_item_ids"]
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid ordered history item IDs")
    return [value for value in values if value != 0]


def classify(target, history, changed):
    if target in changed:
        return "target_changed"
    return "history_only" if any(item in changed for item in history) else "unchanged"


def first_hit(row):
    predictions, label = row["top10_sids"], row["target_sid"]
    if len(predictions) != 10:
        raise ValueError("Exactly 10 raw predictions are required")
    return next((index + 1 for index, value in enumerate(predictions) if value == label), 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    delivery = Path(__file__).resolve().parent
    parser.add_argument("--formal-dir", type=Path,
                        default=delivery / "formal_local" / "paired_formal_20261013_1123")
    parser.add_argument("--output", type=Path, default=delivery / "hit_case_audit.json")
    args = parser.parse_args(argv)
    root, output = args.formal_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Case audit output already exists; nothing will be overwritten")
    catalog_path = root / "catalog_ids.json"
    record_paths = {arm: root / arm / "records.jsonl.gz" for arm in ARMS}
    sources = {"catalog_ids.json": catalog_path,
               **{f"{arm}/records.jsonl.gz": path for arm, path in record_paths.items()},
               "paired_summary.json": root / "paired_summary.json",
               "COMPLETE.json": root / "COMPLETE.json"}
    receipts = {name: {"path": str(path), "sha256": sha256(path),
                       "size_bytes": path.stat().st_size} for name, path in sources.items()}
    complete = json.loads(sources["COMPLETE.json"].read_text(encoding="utf-8"))
    if complete.get("smoke") is not False:
        raise ValueError("Expected complete formal inference, not smoke output")
    if complete["paired_summary_sha256"] != receipts["paired_summary.json"]["sha256"]:
        raise ValueError("Formal COMPLETE checksum does not bind the paired summary")
    paired = json.loads(sources["paired_summary.json"].read_text(encoding="utf-8"))
    if paired["validation_count"] != EXPECTED_COUNT:
        raise ValueError("Paired summary is not the full validation set")
    manifest = json.loads(catalog_path.read_text(encoding="utf-8"))
    initial = manifest["initial"]
    books = {arm: manifest["arms"][arm] for arm in ARMS}
    if len(initial) != 12101 or any(len(codes) != len(initial) for codes in books.values()):
        raise ValueError("Incomplete official item catalogue")
    if books["frozen"] != initial:
        raise ValueError("Frozen arm differs from the shared initial SID catalogue")
    catalogues = {"initial": initial, **books}
    collision_counts = {name: Counter(tuple(code) for code in values)
                        for name, values in catalogues.items()}
    changed_by_arm = {arm: {index + 1 for index, (old, new) in enumerate(zip(initial, codes))
                           if old != new} for arm, codes in books.items()}
    changed = set().union(*changed_by_arm.values())
    rows = {arm: load_records(path) for arm, path in record_paths.items()}
    if any(len(records) != EXPECTED_COUNT for records in rows.values()):
        raise ValueError("Missing or partial arm prediction records")
    cases = []
    for index in range(EXPECTED_COUNT):
        base = rows["frozen"][index]
        target, history = base["target_item_id"], canonical_history(base)
        if type(target) is not int or not 1 <= target <= len(initial):
            raise ValueError("Invalid target item identity")
        for arm in ARMS:
            row = rows[arm][index]
            if (row["sample_index"] != index or row["target_item_id"] != target
                    or canonical_history(row) != history):
                raise ValueError("Arm sample order or identity differs")
            if row["target_sid"] != books[arm][target - 1]:
                raise ValueError("Target SID differs from the official arm catalogue")
            rank = first_hit(row)
            if row["first_hit_rank"] != rank:
                raise ValueError("Stored rank differs from the first raw SID hit")
            expected_hit = float(rank != 0)
            if row["metrics"]["hit_rate@10"] != expected_hit:
                raise ValueError("Stored HR@10 does not match the raw first SID hit")
            ndcg = row["metrics"]["ndcg@10"]
            expected_ndcg = 1 / math.log2(rank + 1) if rank else 0.0
            if (not math.isfinite(ndcg)
                    or not math.isclose(ndcg, expected_ndcg, rel_tol=0, abs_tol=1e-6)):
                raise ValueError("Invalid original float32 NDCG@10")
        rec_hit = rows["cross_rec"][index]["metrics"]["hit_rate@10"]
        recon_hit = rows["cross_recon"][index]["metrics"]["hit_rate@10"]
        if rec_hit == recon_hit:
            continue
        sids = {name: codes[target - 1] for name, codes in catalogues.items()}
        cases.append({
            "sample_index": index, "group": classify(target, history, changed),
            "target_item_id": target, "history_item_ids": history,
            "primary_outcome": "new_sid_hit" if rec_hit > recon_hit else "lost_sid_hit",
            "target_sid": sids,
            "target_sid_changed_from_initial": {arm: target in changed_by_arm[arm] for arm in ARMS},
            "history_contains_changed_item_by_arm": {
                arm: any(item in changed_by_arm[arm] for item in history) for arm in ARMS},
            "target_collision_size": {
                name: collision_counts[name][tuple(code)] for name, code in sids.items()},
            "first_hit_rank": {arm: rows[arm][index]["first_hit_rank"] for arm in ARMS},
            "ndcg": {arm: {"ndcg@5": rows[arm][index]["metrics"]["ndcg@5"],
                           "ndcg@10": rows[arm][index]["metrics"]["ndcg@10"]} for arm in ARMS},
            "delta_ndcg_at10_rec_minus_recon":
                rows["cross_rec"][index]["metrics"]["ndcg@10"]
                - rows["cross_recon"][index]["metrics"]["ndcg@10"],
        })
    gained = sum(case["primary_outcome"] == "new_sid_hit" for case in cases)
    lost = sum(case["primary_outcome"] == "lost_sid_hit" for case in cases)
    if (gained, lost, len(cases)) != (24, 31, 55):
        raise ValueError("Expected 24 new / 31 lost SID hits in this completed run")
    target_cases = [case for case in cases if case["group"] == "target_changed"]
    actual_targets = {case["sample_index"]: (case["target_item_id"],
                     tuple(case["first_hit_rank"][arm] for arm in ARMS)) for case in target_cases}
    if actual_targets != EXPECTED_TARGET_CASES:
        raise ValueError("Target-change disagreement cases differ from independently audited records")
    groups = {}
    for group in ("target_changed", "history_only", "unchanged"):
        selected = [case for case in cases if case["group"] == group]
        new = sum(case["primary_outcome"] == "new_sid_hit" for case in selected)
        gone = len(selected) - new
        groups[group] = {"new_hits_at10": new, "lost_hits_at10": gone,
                         "net_hits_at10": new - gone, "disagreement_cases": len(selected)}
        recorded = paired["groups"][group]["pairs"]["cross_rec_minus_cross_recon"]
        if any(groups[group][key] != recorded[key]
               for key in ("new_hits_at10", "lost_hits_at10", "net_hits_at10")):
            raise ValueError("Case accounting does not reproduce the group paired summary")
    if any(sha256(path) != receipts[name]["sha256"] for name, path in sources.items()):
        raise RuntimeError("Original evidence changed during the read-only case audit")
    result = {
        "audit_id": "stage05_feedback_hr10_disagreement_case_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "source_inputs": receipts,
        "evaluation_level": "single-seed validation SID-level; not strict item-level",
        "primary_pair": "cross_rec_minus_cross_recon",
        "validation_count": EXPECTED_COUNT,
        "summary": {"new_hits_at10": gained, "lost_hits_at10": lost,
                    "net_hits_at10": gained - lost, "disagreement_cases": len(cases),
                    "changed_union_items": len(changed),
                    "changed_items_by_arm": {arm: len(values) for arm, values in changed_by_arm.items()},
                    "groups": groups, "target_changed_disagreement_cases": len(target_cases)},
        "checks": {"all_sample_identities_and_official_labels": True,
                   "raw_first_hit_and_original_scores": True,
                   "24_new_31_lost": True, "four_target_cases": True,
                   "group_counts_match_paired_summary": True,
                   "source_bytes_unchanged_during_audit": True},
        "interpretation_limits": [
            "New/lost hits refer to the same validation item identity but each arm's official SID label.",
            "A collision-group SID hit cannot establish which item was recommended.",
            "A retained target SID can outperform another arm's changed SID; this is not an active recoding gain.",
            "Unchanged target/history SIDs are not a wholly untreated control because decoder trajectories and the global legal SID set differ.",
            "Cases with unchanged HR but improved/worse hit rank are not selected by this HR disagreement audit.",
        ],
        "target_changed_cases": target_cases, "all_hr10_disagreement_cases": cases,
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "output": str(output),
                      "sha256": sha256(output), "summary": result["summary"],
                      "target_changed_cases": target_cases}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
