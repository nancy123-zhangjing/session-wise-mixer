"""Prepare locked item-level prefix datasets using only the Python standard library.

Never downloads, imports a training framework, changes IDs, or overwrites a run.
Only two allowlisted regular files can be read from the migration archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path("configs/06_continuous_baseline/data_protocols.json")
INPUT_PREFIX = "data/cache/AmazonReviews2014/Beauty/processed/"
INPUT_NAMES = ("all_item_seqs.json", "id_mapping.json")
PROTOCOLS = ("full_train_prefix_v1", "legacy_prefix_v1")
SPLITS = ("train", "validation", "test")
SEMANTICS = {
    "schema_version": 1,
    "dataset": "AmazonReviews2014/Beauty",
    "history": {"max_length": 50, "pad_id": 0, "padding": "left_at_collation"},
    "split": {"method": "leave_one_out", "validation_offset": -2, "test_offset": -1},
    "supervision": "one_next_item_per_prefix",
    "protocols": {
        "full_train_prefix_v1": {"min_history": 1, "max_target_position": None},
        "legacy_prefix_v1": {"min_history": 2, "max_target_position": 50},
    },
    "evaluation": {"level": "item", "candidates": "full_catalog_excluding_pad", "filter_history": False},
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(raw):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique_keys)


def project_path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path):
    config = parse_json(Path(path).read_bytes())
    if set(config) != set(SEMANTICS) | {"asset_lock", "expected"}:
        raise ValueError("Unknown or missing protocol configuration fields")
    for key, expected in SEMANTICS.items():
        # JSON encoding also distinguishes bool from int, unlike Python equality.
        if json_bytes(config[key]) != json_bytes(expected):
            raise ValueError(f"Versioned protocol semantics changed: {key}")
    if not isinstance(config["asset_lock"], str) or not config["asset_lock"]:
        raise ValueError("An existing asset lock is required")
    expected = config["expected"]
    if not isinstance(expected, dict) or set(expected) != {
        "users", "items", "interactions", "train_targets", "validation", "test"
    }:
        raise ValueError("Invalid expected-statistics contract")
    if not isinstance(expected["train_targets"], dict) or set(expected["train_targets"]) != set(PROTOCOLS):
        raise ValueError("Expected counts must cover both protocols")
    counts = [value for key, value in expected.items() if key != "train_targets"]
    counts.extend(expected["train_targets"].values())
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Expected counts must be nonnegative integers")
    return config


def locked_assets(lock_path):
    lock = parse_json(Path(lock_path).read_bytes())
    result = {}
    for name in INPUT_NAMES:
        entry = lock["artifacts"][INPUT_PREFIX + name]
        if entry["path"] != INPUT_PREFIX + name:
            raise ValueError("Asset lock path mismatch")
        member = entry["archive_member"]
        if member != "experiment1/" + INPUT_PREFIX + name:
            raise ValueError("Unexpected archive member in input lock")
        if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= 64 * 1024 * 1024:
            raise ValueError("Invalid or unexpectedly large locked input")
        result[name] = entry
    return result


def check_input_bytes(raw, assets):
    if set(raw) != set(INPUT_NAMES):
        raise ValueError("Exactly the two locked input files are required")
    for name in INPUT_NAMES:
        if len(raw[name]) != assets[name]["size_bytes"] or sha256_bytes(raw[name]) != assets[name]["sha256"]:
            raise ValueError(f"Locked input identity mismatch: {name}")


def read_inputs(assets, *, input_dir=None, archive=None):
    if (input_dir is None) == (archive is None):
        raise ValueError("Choose exactly one input directory or archive")
    raw = {}
    if input_dir is not None:
        raw = {name: (Path(input_dir) / name).read_bytes() for name in INPUT_NAMES}
    else:
        wanted = {entry["archive_member"]: name for name, entry in assets.items()}
        with tarfile.open(archive, mode="r|gz") as bundle:
            for entry in bundle:
                if entry.name not in wanted:
                    continue
                name = wanted[entry.name]
                if name in raw or not entry.isfile() or entry.size != assets[name]["size_bytes"]:
                    raise ValueError(f"Invalid selected archive entry: {name}")
                # No extract/extractall: archive paths never become write paths.
                raw[name] = bundle.extractfile(entry).read()
                if len(raw) == len(INPUT_NAMES):
                    break
    check_input_bytes(raw, assets)
    return raw


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or set(mapping) != {"item2id", "id2item", "user2id", "id2user"}:
        raise ValueError("Unexpected mapping schema")
    for kind in ("item", "user"):
        forward, reverse = mapping[kind + "2id"], mapping["id2" + kind]
        if not isinstance(forward, dict) or not isinstance(reverse, list) or not reverse:
            raise ValueError(f"Invalid {kind} mapping")
        if any(not isinstance(key, str) or not key for key in forward):
            raise ValueError("Mapping keys must be nonempty strings")
        if any(type(index) is not int for index in forward.values()):
            raise ValueError("Mapping IDs must be integers, not booleans")
        if forward.get("[PAD]") != 0 or reverse[0] != "[PAD]":
            raise ValueError("PAD must retain original ID zero")
        if len(forward) != len(reverse) or set(forward.values()) != set(range(len(reverse))):
            raise ValueError("Mapping must remain contiguous and bijective")
        for key, index in forward.items():
            if reverse[index] != key:
                raise ValueError("Forward/reverse mapping mismatch")


def normalize_inputs(raw):
    sequences = parse_json(raw["all_item_seqs.json"])
    mapping = parse_json(raw["id_mapping.json"])
    validate_mapping(mapping)
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError("Nonempty original user sequences required")
    if set(sequences) != set(mapping["user2id"]) - {"[PAD]"}:
        raise ValueError("Sequence users differ from the original mapping")
    normalized = []
    for user, seq in sequences.items():
        if not isinstance(seq, list) or len(seq) < 3:
            raise ValueError("At least three interactions needed for nonempty validation history")
        if any(not isinstance(item, str) or item == "[PAD]" or item not in mapping["item2id"] for item in seq):
            raise ValueError("Unknown, PAD, or non-string item in original sequences")
        normalized.append((mapping["user2id"][user], tuple(mapping["item2id"][item] for item in seq)))
    normalized.sort(key=lambda pair: pair[0])
    return normalized, mapping


def dataset_statistics(sequences, mapping):
    lengths = sorted(len(seq) for _, seq in sequences)
    train_items = {item for _, seq in sequences for item in seq[:-2]}
    return {
        "users": len(sequences), "items": len(mapping["id2item"]) - 1,
        "interactions": sum(lengths), "min_sequence_length": lengths[0],
        "median_sequence_length": statistics.median(lengths),
        "p95_sequence_length_nearest_rank": lengths[(95 * len(lengths) + 99) // 100 - 1],
        "max_sequence_length": lengths[-1], "train_visible_items": len(train_items),
        "validation_targets_unseen_in_train_interactions": sum(seq[-2] not in train_items for _, seq in sequences),
        "test_targets_unseen_in_train_interactions": sum(seq[-1] not in train_items for _, seq in sequences),
        "train_targets": {
            PROTOCOLS[0]: sum(len(seq) - 3 for _, seq in sequences),
            PROTOCOLS[1]: sum(max(0, min(len(seq) - 3, 50) - 1) for _, seq in sequences),
        },
        "validation": len(sequences), "test": len(sequences),
    }


def check_statistics(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Expected statistic mismatch: {key}: {actual.get(key)!r} != {value!r}")


def iter_records(sequences, protocol, split):
    if protocol not in PROTOCOLS or split not in SPLITS:
        raise ValueError("Unknown protocol or split")
    for user_id, seq in sequences:
        if split == "train":
            positions = (range(1, len(seq) - 2) if protocol == PROTOCOLS[0]
                         else range(2, min(len(seq) - 3, 50) + 1))
        else:
            positions = (len(seq) - (2 if split == "validation" else 1),)
        for position in positions:
            start = max(0, position - 50)
            yield {"user_id": user_id, "target_position": position,
                   "history_start": start, "history": list(seq[start:position]),
                   "target": seq[position]}


def safe_new_output(root, output):
    root = Path(root).resolve()
    base = root / "outputs/stage06_continuous"
    literal = Path(output) if Path(output).is_absolute() else root / output
    if literal.is_symlink():
        raise FileExistsError(f"Refusing an existing output symlink: {literal}")
    output = project_path(root, output)
    # Also reject a redirected outputs tree, even if the final leaf is absent.
    if not base.resolve().is_relative_to(root) or not output.is_relative_to(base.resolve()) or output == base.resolve():
        raise ValueError("Output must be a new child of project outputs/stage06_continuous")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    return output


def git_state(root):
    result = {}
    for label, command in (("head", ["git", "rev-parse", "HEAD"]),
                           ("status", ["git", "status", "--short"])):
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        result[label] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def prepare(*, root, config_path, output, input_dir=None, archive=None):
    root = Path(root).resolve()
    output = safe_new_output(root, output)
    config_path = project_path(root, config_path)
    config = load_config(config_path)
    lock_path = project_path(root, config["asset_lock"])
    assets = locked_assets(lock_path)
    raw = read_inputs(assets, input_dir=input_dir, archive=archive)
    sequences, mapping = normalize_inputs(raw)
    stats = dataset_statistics(sequences, mapping)
    check_statistics(stats, config["expected"])
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "command": list(sys.orig_argv),
        "platform": platform.platform(), "git": git_state(root), "local_cpu_only": True,
        "gpu_used": False, "model_training": False, "remote_assets_verified": False,
        "source": {"kind": "archive_selected_members" if archive else "input_directory",
                   "path": str(Path(archive or input_dir).resolve()),
                   "whole_archive_digest_checked": False if archive else None},
        "asset_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "config_source": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "producer_sha256": sha256_file(__file__),
    }
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def register(relative, *, records=None):
        path = output / relative
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if records is not None:
            files[relative]["records"] = records

    def write(relative, content):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
        register(relative)

    for name in INPUT_NAMES:
        write("inputs/" + name, raw[name])
    write("resolved_protocol.json", json_bytes(config))
    write("catalog_item_ids.json", json_bytes(list(range(1, len(mapping["id2item"])))))
    write("statistics.json", json_bytes(stats))
    for protocol in PROTOCOLS:
        for split in SPLITS:
            relative = f"{protocol}/{split}.jsonl"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with path.open("xb") as handle:
                for record in iter_records(sequences, protocol, split):
                    handle.write(json_bytes(record))
                    count += 1
            expected = stats["train_targets"][protocol] if split == "train" else stats[split]
            if count != expected:
                raise RuntimeError("Generated row count differs from the input-derived count")
            register(relative, records=count)
    manifest = {"schema_version": 1, "files": files,
                "record_order": "original_user_id_then_target_position",
                "position_indexing": "zero_based_in_original_user_sequence",
                "history_representation": "unpadded_original_item_ids",
                "input_sha256": {name: sha256_bytes(raw[name]) for name in INPUT_NAMES}}
    for name, payload in (("manifest.json", manifest), ("provenance.json", provenance)):
        with (output / name).open("xb") as handle:
            handle.write(json_bytes(payload))
    with (output / "PREPARED.json").open("xb") as handle:
        handle.write(json_bytes({"status": "generated_requires_independent_verification",
                                 "manifest_sha256": sha256_file(output / "manifest.json"),
                                 "provenance_sha256": sha256_file(output / "provenance.json")}))
    return {"output": str(output), "statistics": stats,
            "manifest_sha256": sha256_file(output / "manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    result = prepare(root=args.project_root, config_path=args.config, output=args.output,
                     input_dir=args.input_dir, archive=args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
