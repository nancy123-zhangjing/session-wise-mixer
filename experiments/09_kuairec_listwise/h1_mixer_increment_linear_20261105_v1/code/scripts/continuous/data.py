"""Read only the already verified stage-06 JSONL protocols."""
import hashlib
import json
from pathlib import Path

import torch

from .protocol import atomic_json

DATA_MANIFEST_SHA256 = "6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa"
EXPECTED_ROWS = {"full_train_prefix_v1": {"train": 131413, "validation": 22363, "test": 22363},
                 "legacy_prefix_v1": {"train": 105668, "validation": 22363, "test": 22363}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(directory, protocol, split, n_items=12101, max_length=50):
    if protocol not in {"full_train_prefix_v1", "legacy_prefix_v1"}:
        raise ValueError("Unknown protocol")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    path = Path(directory) / protocol / (split + ".jsonl")
    histories, targets, users, positions = [], [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            history = row["history"]
            if not 1 <= len(history) <= max_length:
                raise ValueError("Empty or oversized history")
            if any(not 1 <= item <= n_items for item in history + [row["target"]]):
                raise ValueError("PAD or invalid item in raw data")
            histories.append([0] * (max_length - len(history)) + history)
            targets.append(row["target"])
            users.append(row["user_id"])
            positions.append(row["target_position"])
    if not histories:
        raise ValueError("Empty split")
    return {
        "histories": torch.tensor(histories, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "users": torch.tensor(users, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "source_path": str(path.resolve()), "sha256": sha256(path),
    }


def verify_prepared(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != DATA_MANIFEST_SHA256:
        raise ValueError("Dataset manifest differs from accepted stage-1 identity")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Manifest path escapes data directory")
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Dataset byte mismatch: {name}")
    return manifest


def prepare_cache(directory, output):
    """CPU-only deterministic tensor cache in a new directory, without rewriting data."""
    import numpy as np
    manifest = verify_prepared(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = {"schema_version": 1, "data_manifest_sha256": DATA_MANIFEST_SHA256,
             "source_directory": str(Path(directory).resolve()), "splits": {}}
    for protocol in EXPECTED_ROWS:
        for split, expected in EXPECTED_ROWS[protocol].items():
            data = load_split(directory, protocol, split)
            if len(data["targets"]) != expected:
                raise ValueError("Accepted dataset count mismatch")
            location = output / protocol / split
            location.mkdir(parents=True)
            files = {}
            for key in ("histories", "targets", "users", "positions"):
                path = location / (key + ".npy")
                with path.open("xb") as stream:
                    np.save(stream, data[key].numpy(), allow_pickle=False)
                files[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
            cache["splits"][protocol + "/" + split] = {
                "rows": expected, "source_sha256": data["sha256"], "files": files}
            if data["sha256"] != manifest["files"][protocol + "/" + split + ".jsonl"]["sha256"]:
                raise ValueError("Cache input binding mismatch")
    atomic_json(output / "cache.json", cache)
    atomic_json(output / "CACHE_COMPLETE.json", {"cache_sha256": sha256(output / "cache.json"),
                "data_manifest_sha256": DATA_MANIFEST_SHA256})
    return cache


def load_cached_split(directory, protocol, split):
    import numpy as np
    directory = Path(directory)
    receipt = json.loads((directory / "CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["data_manifest_sha256"] != DATA_MANIFEST_SHA256:
        raise ValueError("Cache dataset identity mismatch")
    if sha256(directory / "cache.json") != receipt["cache_sha256"]:
        raise ValueError("Cache manifest tampering")
    metadata = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
    entry = metadata["splits"][protocol + "/" + split]
    if entry["rows"] != EXPECTED_ROWS[protocol][split]:
        raise ValueError("Cache count mismatch")
    result = {"sha256": entry["source_sha256"]}
    for name, info in entry["files"].items():
        path = directory / info["path"]
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != info["sha256"]:
            raise ValueError("Cache file escape or byte mismatch")
        values = np.load(path, allow_pickle=False)
        if values.dtype != np.int64:
            raise ValueError("Unexpected cache dtype")
        result[name] = torch.from_numpy(values)
    if result["histories"].shape != (entry["rows"], 50):
        raise ValueError("Unexpected cache history shape")
    if any(result[key].shape != (entry["rows"],) for key in ("targets", "users", "positions")):
        raise ValueError("Unexpected cache target/identity shape")
    return result
