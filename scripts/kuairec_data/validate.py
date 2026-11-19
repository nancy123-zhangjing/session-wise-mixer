"""Independent, streaming invariant checks and descriptive KuaiRec statistics.

This module never calls the sample constructor. Full source-event replay is a
separate audit: these checks certify every emitted row, not omitted rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")
GROUP_FIELDS = ("item_ids", "original_item_ids", "source_event_id", "timestamp_us",
                "play_duration", "video_duration", "watch_ratio", "mask")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def array(batch, name):
    column = batch.column(batch.schema.get_field_index(name))
    require(column.null_count == 0, f"null values in {name}")
    return column.to_numpy(zero_copy_only=False)


def group_arrays(batch, prefix):
    lengths = None
    result = {}
    for suffix in GROUP_FIELDS:
        name = prefix + "_" + suffix
        column = batch.column(batch.schema.get_field_index(name))
        require(column.null_count == 0, f"null list in {name}")
        current = pc.list_value_length(column).to_numpy(zero_copy_only=False)
        if lengths is None:
            lengths = current
        else:
            require(np.array_equal(lengths, current), f"inconsistent list lengths: {name}")
        flattened = pc.list_flatten(column)
        require(flattened.null_count == 0, f"null list element in {name}")
        result[suffix] = flattened.to_numpy(zero_copy_only=False)
    return lengths, result


def load_map(path):
    mapping = json.loads(path.read_text(encoding="utf-8"))
    require(mapping["pad_id"] == 0, f"PAD must be 0: {path.name}")
    entries = mapping["entries"]
    require([r["id"] for r in entries] == list(range(1, len(entries) + 1)),
            f"non-dense or nonordered IDs: {path.name}")
    originals = [r["original_id"] for r in entries]
    require(originals == sorted(set(originals)), f"unstable original-ID order: {path.name}")
    return np.asarray([-1] + originals, dtype=np.int64)


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1])
    return {"count": int(len(values)), "mean": float(values.mean()),
            **{name: float(value) for name, value in zip(
                ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"), quantiles)}}


def beijing_time(timestamp_us):
    """Render exact integer microseconds without a float-seconds roundtrip."""
    value = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(timestamp_us))
    return value.astimezone(timezone(timedelta(hours=8))).isoformat(timespec="microseconds")


class Feedback:
    def __init__(self, edges):
        self.edges = np.asarray(edges)
        self.counts = np.zeros(len(edges) - 1, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, values):
        self.counts += np.histogram(values, self.edges)[0]
        self.count += len(values)
        self.total += float(values.sum())
        low, high = float(values.min()), float(values.max())
        self.minimum = low if self.minimum is None else min(self.minimum, low)
        self.maximum = high if self.maximum is None else max(self.maximum, high)

    def report(self):
        return {"count": self.count, "mean": self.total / self.count if self.count else None,
                "min": self.minimum, "max": self.maximum,
                "histogram_left_closed_right_open": {
                    f"[{left:g},{right:g})": int(n)
                    for left, right, n in zip(self.edges[:-1], self.edges[1:], self.counts)}}


def validate(dataset: Path):
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    ingestion = json.loads((dataset / "ingestion.json").read_text(encoding="utf-8"))
    construction = json.loads((dataset / "construction.json").read_text(encoding="utf-8"))
    user_original = load_map(dataset / "user_map.json")
    item_original = load_map(dataset / "item_map.json")
    raw_count = ingestion["counts"]["raw_events"]
    cuts = ingestion["cuts_us"]
    require(cuts[0] < cuts[1], "collapsed time cuts")
    require(config["list_length"] == 5, "this validator requires the five-item protocol")
    event_target_owner = np.zeros(raw_count + 1, dtype=np.uint8)
    train_users = np.zeros(len(user_original), dtype=bool)
    train_items = np.zeros(len(item_original), dtype=bool)
    text_rows = [json.loads(line) for line in (dataset / "item_text.jsonl").read_text(
        encoding="utf-8").splitlines()]
    text_by_item = {row["video_id"]: row for row in text_rows}
    require(len(text_by_item) == len(text_rows), "duplicate static-text item IDs")
    missing_text = np.asarray([True] + [text_by_item.get(int(i), {}).get("text_missing", True)
                                       for i in item_original[1:]], dtype=bool)
    sample_keys = []
    reports = {}
    checks = Counter()
    for split_index, split in enumerate(SPLITS):
        seen_users = np.zeros(len(user_original), dtype=bool)
        history_items = np.zeros(len(item_original), dtype=bool)
        target_items = np.zeros(len(item_original), dtype=bool)
        hist_lengths, spans, total_play = [], [], []
        per_user = np.zeros(len(user_original), dtype=np.int64)
        ratio = Feedback([0, .1, .5, 1, 2, 5, 10, 100, float("inf")])
        plays = Feedback([0, 1000, 5000, 10000, 30000, 60000, 300000, float("inf")])
        extra = Counter()
        cold_target_items, cold_history_items = set(), set()
        for batch in pq.ParquetFile(dataset / f"{split}.parquet").iter_batches(batch_size=2048):
            n = batch.num_rows
            uid = array(batch, "user_id")
            raw_uid = array(batch, "original_user_id")
            require(np.all((uid > 0) & (uid < len(user_original))), "invalid user/PAD ID")
            require(np.array_equal(user_original[uid], raw_uid), "user map mismatch")
            require(all(s == split for s in batch.column("split").to_pylist()), "split column mismatch")
            hl, h = group_arrays(batch, "history")
            tl, t = group_arrays(batch, "target")
            require(np.all(tl == 5), "target length is not five")
            require(np.all((hl >= config["history_min"]) & (hl <= config["history_max"])),
                    "history length outside configured bounds")
            for prefix, lengths, group in (("history", hl, h), ("target", tl, t)):
                ids = group["item_ids"]
                require(np.all((ids > 0) & (ids < len(item_original))), f"{prefix}: item/PAD out of range")
                require(np.array_equal(item_original[ids], group["original_item_ids"]), f"{prefix}: item map mismatch")
                require(group["mask"].all(), f"{prefix}: stored unpadded mask must be true")
                require(np.all((group["source_event_id"] > 0) & (group["source_event_id"] <= raw_count)),
                        f"{prefix}: source event outside raw file")
                require(np.all(group["play_duration"] > 0), f"{prefix}: nonpositive play")
                require(np.all(group["video_duration"] > 0), f"{prefix}: nonpositive video duration")
                require(np.isfinite(group["watch_ratio"]).all() and (group["watch_ratio"] >= 0).all(),
                        f"{prefix}: invalid watch ratio")
                differences = np.diff(group["timestamp_us"])
                boundaries = np.cumsum(lengths)[:-1] - 1
                differences[boundaries] = 1
                require(np.all(differences > 0), f"{prefix}: non-increasing within-row time")
            target_ids = t["item_ids"].reshape(n, 5)
            target_times = t["timestamp_us"].reshape(n, 5)
            require(np.all(np.diff(np.sort(target_ids, axis=1), axis=1) != 0), "repeated target item")
            starts, ends = target_times[:, 0], target_times[:, -1]
            require(np.all(h["timestamp_us"] < np.repeat(starts, hl)), "current/future event in history")
            session_start, session_end = array(batch, "session_start_us"), array(batch, "session_end_us")
            require(np.all((session_start <= starts) & (ends <= session_end)), "targets outside session bounds")
            for cut in cuts:
                require(not np.any((session_start < cut) & (cut <= session_end)), "cross-cut session emitted")
            expected_split = np.searchsorted(np.asarray(cuts), session_start, side="right")
            require(np.all(expected_split == split_index), "incorrect time-split assignment")
            span = array(batch, "target_span_us")
            require(np.array_equal(span, ends - starts), "stored target span mismatch")
            require(np.all(span <= config["max_target_span_seconds"] * 1_000_000), "target span exceeds limit")
            total = array(batch, "target_total_play_ms")
            require(np.array_equal(total, t["play_duration"].reshape(n, 5).sum(axis=1)), "stored total playback mismatch")
            optional = batch.column("train_optional_ge60s")
            if split == "train":
                require(optional.null_count == 0, "missing training optional flag")
                require(np.array_equal(optional.to_numpy(zero_copy_only=False), total >= config["train_optional_total_play_ms"]),
                        "incorrect optional 60-second training flag")
                extra["optional_total_play_ge60s_samples"] += int((total >= config["train_optional_total_play_ms"]).sum())
            else:
                require(optional.null_count == n, "optional training label appears in holdout")
            offsets = array(batch, "block_offset")
            require(np.all((offsets >= 0) & (offsets % 5 == 0)), "invalid block offset")
            batch_keys = []
            for sid, sample_id, original_user, offset in zip(batch.column("session_id").to_pylist(),
                    batch.column("sample_id").to_pylist(), raw_uid, offsets):
                match = re.fullmatch(r"u(\d+):e(\d+)", sid)
                require(match is not None and int(match[1]) == original_user, "session ID does not match user")
                event_id = int(match[2])
                require(1 <= event_id <= raw_count and offset < raw_count, "session source/block offset out of range")
                require(sample_id == f"{sid}:b{offset // 5}", "sample ID inconsistent with block")
                batch_keys.append(event_id * (raw_count + 1) + int(offset))
            sample_keys.append(np.asarray(batch_keys, dtype=np.int64))
            event_ids = t["source_event_id"]
            require(len(np.unique(event_ids)) == len(event_ids), "overlapping target events within batch")
            require(not event_target_owner[event_ids].any(), "overlapping target events across samples/splits")
            event_target_owner[event_ids] = split_index + 1
            seen_users[uid] = True
            history_items[h["item_ids"]] = True
            target_items[t["item_ids"]] = True
            np.add.at(per_user, uid, 1)
            if split == "train":
                train_users[uid] = True
                train_items[h["item_ids"]] = True
                train_items[t["item_ids"]] = True
            else:
                extra["cold_user_samples"] += int((~train_users[uid]).sum())
                h_cold, t_cold = ~train_items[h["item_ids"]], ~train_items[t["item_ids"]]
                extra["cold_history_item_occurrences"] += int(h_cold.sum())
                extra["cold_target_item_occurrences"] += int(t_cold.sum())
                extra["samples_with_cold_target_item"] += int(t_cold.reshape(n, 5).any(axis=1).sum())
                history_starts = np.concatenate(([0], np.cumsum(hl)[:-1]))
                extra["samples_with_cold_history_item"] += int(np.logical_or.reduceat(h_cold, history_starts).sum())
                cold_history_items.update(h["item_ids"][h_cold].tolist())
                cold_target_items.update(t["item_ids"][t_cold].tolist())
            extra["target_missing_text_occurrences"] += int(missing_text[t["item_ids"]].sum())
            extra["history_missing_text_occurrences"] += int(missing_text[h["item_ids"]].sum())
            extra["samples_with_missing_target_text"] += int(missing_text[target_ids].any(axis=1).sum())
            extra["target_watch_ratio_gt1_occurrences"] += int((t["watch_ratio"] > 1).sum())
            hist_lengths.append(hl.astype(np.int8))
            spans.append(span)
            total_play.append(total)
            ratio.add(t["watch_ratio"])
            plays.add(t["play_duration"])
            checks[split + "_samples_checked"] += n
        count = checks[split + "_samples_checked"]
        require(count > 0, f"empty split: {split}")
        require(count == construction["counts"]["retained_" + split], "construction/sample count mismatch")
        reports[split] = {
            "samples": count, "users": int(seen_users.sum()), "history_items": int(history_items.sum()),
            "target_items": int(target_items.sum()), "history_or_target_items": int((history_items | target_items).sum()),
            "history_length": distribution(np.concatenate(hist_lengths)),
            "target_span_seconds": distribution(np.concatenate(spans) / 1_000_000),
            "target_total_play_ms": distribution(np.concatenate(total_play)),
            "target_play_duration_ms": plays.report(), "target_watch_ratio": ratio.report(),
            "samples_per_user_with_samples": distribution(per_user[per_user > 0]),
            "max_user_fraction_of_samples": float(per_user.max() / count),
            "cold_users": int((seen_users & ~train_users).sum()),
            "cold_history_items": len(cold_history_items), "cold_target_items": len(cold_target_items),
            "history_missing_text_items": int((history_items & missing_text).sum()),
            "target_missing_text_items": int((target_items & missing_text).sum()),
            **dict(extra),
        }
        print(json.dumps({"stage": "validate", "split": split, "samples": count}), flush=True)
    keys = np.concatenate(sample_keys)
    require(len(np.unique(keys)) == len(keys), "duplicate sample IDs")
    all_samples = sum(reports[s]["samples"] for s in SPLITS)
    for report in reports.values():
        report["sample_fraction"] = report["samples"] / all_samples
    train_seen_original = set(item_original[train_items].tolist())
    train_seen_original_users = set(user_original[train_users].tolist())
    train_time_items, train_time_users = set(), set()
    timeline_items, timeline_users = set(), set()
    timeline_counts = Counter()
    timeline_min, timeline_max = None, None
    quantile_below = np.zeros(len(cuts), dtype=np.int64)
    quantile_equal = np.zeros(len(cuts), dtype=np.int64)
    for path in sorted((dataset / "events").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,
                columns=["timestamp_us", "event_valid", "ambiguous", "video_id", "user_id",
                         "play_duration", "watch_ratio"]):
            timestamps = array(batch, "timestamp_us")
            usable = array(batch, "event_valid") & ~array(batch, "ambiguous")
            selected = (timestamps < cuts[0]) & usable
            original_items, original_users = array(batch, "video_id"), array(batch, "user_id")
            train_time_items.update(original_items[selected].tolist())
            train_time_users.update(original_users[selected].tolist())
            timeline_items.update(original_items[usable].tolist())
            timeline_users.update(original_users[usable].tolist())
            if not usable.any():
                continue
            valid_times = timestamps[usable]
            for i, cut in enumerate(cuts):
                quantile_below[i] += int((valid_times < cut).sum())
                quantile_equal[i] += int((valid_times == cut).sum())
            low, high = int(timestamps[usable].min()), int(timestamps[usable].max())
            timeline_min = low if timeline_min is None else min(timeline_min, low)
            timeline_max = high if timeline_max is None else max(timeline_max, high)
            play = array(batch, "play_duration")[usable]
            timeline_counts["clean_valid_unambiguous_events"] += int(usable.sum())
            timeline_counts["played_events"] += int((play > 0).sum())
            timeline_counts["zero_play_events"] += int((play == 0).sum())
            timeline_counts["negative_play_events"] += int((play < 0).sum())
            # Quarantined rows may legitimately have null feedback; filter them
            # before checking or summarizing the usable timeline.
            ratios = batch.column("watch_ratio").to_numpy(zero_copy_only=False)[usable]
            require(np.isfinite(ratios).all(), "invalid feedback in usable event timeline")
            timeline_counts["watch_ratio_gt1_events"] += int((ratios > 1).sum())
    require(timeline_min is not None, "empty clean event timeline")
    require(timeline_counts["clean_valid_unambiguous_events"] == ingestion["counts"]["clean_unambiguous_events"],
            "independent timeline count disagrees with ingestion")
    quantile_checks = []
    require(len(config["split_quantiles"]) == len(cuts), "quantile/cut count mismatch")
    for q, cut, below, equal in zip(config["split_quantiles"], cuts, quantile_below, quantile_equal):
        rank = math.ceil(q * timeline_counts["clean_valid_unambiguous_events"])
        passed = int(below) < rank <= int(below + equal)
        require(passed, f"cut {cut} is not the nearest-rank quantile {q}")
        quantile_checks.append({"quantile": q, "cut_us": cut, "rank_one_based": rank,
                                "events_below_cut": int(below), "events_equal_cut": int(equal),
                                "passed": passed})
    report = {"schema_version": 1, "cold_start_reference": "actual retained training history and target union",
              "catalog": {"users_excluding_pad": len(user_original) - 1,
                          "items_excluding_pad": len(item_original) - 1,
                          "items_missing_text": int(missing_text[1:].sum())},
              "timeline": {"scope": "deduplicated, valid, unambiguous observed events; includes zero-play records",
                           "min_timestamp_us": timeline_min, "max_timestamp_us": timeline_max,
                           "min_time_beijing": beijing_time(timeline_min),
                           "max_time_beijing": beijing_time(timeline_max),
                           "users": len(timeline_users), "items": len(timeline_items), **dict(timeline_counts)},
              "time_boundaries": {"cuts_us": cuts, "cuts_beijing": [beijing_time(c) for c in cuts],
                                  "quantile_method": config.get("quantile_method", "nearest_rank_ceil"),
                                  "quantile_rank_checks": quantile_checks,
                                  "train_interval": "timestamp_us < first cut"},
              "train_time_observed_items": len(train_time_items),
              "train_time_observed_but_not_in_training_samples_items": len(train_time_items - train_seen_original),
              "train_time_observed_but_not_in_training_samples_original_item_ids": sorted(train_time_items - train_seen_original),
              "train_time_observed_users": len(train_time_users),
              "train_time_observed_but_not_in_training_samples_users": len(train_time_users - train_seen_original_users),
              "train_time_observed_but_not_in_training_samples_original_user_ids": sorted(train_time_users - train_seen_original_users),
              "splits": reports}
    write_json(dataset / "train_seen.json", {"reference": report["cold_start_reference"],
               "item_ids": np.flatnonzero(train_items).tolist(), "user_ids": np.flatnonzero(train_users).tolist()})
    write_json(dataset / "split_statistics.json", report)
    validation = {"passed": True, "scope": "all emitted samples; source reconstruction is audited separately",
                  "samples_checked": all_samples, "target_events_checked": int((event_target_owner > 0).sum()),
                  "unique_sample_ids": int(len(keys)), "checks": dict(checks),
                  "quantile_rank_checks": quantile_checks,
                  "checks_description": ["mapped IDs and PAD", "history and target length/order/masks/feedback",
                     "strict history cutoff", "distinct five-item targets", "whole-session temporal split",
                     "target span and playback sums", "training-only optional flag", "block/sample identities",
                     "unique nonoverlapping target events across all samples and splits"]}
    write_json(dataset / "validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.dataset)
    except Exception as exc:
        if args.dataset.is_dir():
            write_json(args.dataset / "validation.json", {"passed": False,
                       "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
