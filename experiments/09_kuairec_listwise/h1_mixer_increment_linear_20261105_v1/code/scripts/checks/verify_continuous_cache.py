"""Independent full-row NumPy cache verifier. Does not import producer/model code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data, cache):
    data, cache = Path(data), Path(cache)
    accepted = "6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa"
    if digest(data / "manifest.json") != accepted:
        raise ValueError("Data manifest is not accepted stage-1 identity")
    original = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    receipt = json.loads((cache / "CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["cache_sha256"] != digest(cache / "cache.json"):
        raise ValueError("Cache receipt mismatch")
    checks = []
    for protocol, train_rows in (("full_train_prefix_v1", 131413), ("legacy_prefix_v1", 105668)):
        for split, expected in (("train", train_rows), ("validation", 22363), ("test", 22363)):
            relative = protocol + "/" + split
            source = data / (relative + ".jsonl")
            source_hash = digest(source)
            if source_hash != original["files"][relative + ".jsonl"]["sha256"]:
                raise ValueError("Original JSONL bytes changed")
            entry = metadata["splits"][relative]
            if source_hash != entry["source_sha256"] or entry["rows"] != expected:
                raise ValueError("Cache input identity/count mismatch")
            arrays = {}
            for key, info in entry["files"].items():
                path = cache / info["path"]
                if not path.resolve().is_relative_to(cache.resolve()) or digest(path) != info["sha256"]:
                    raise ValueError("Cache file changed or escaped")
                arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
                if arrays[key].dtype != np.int64:
                    raise ValueError("Wrong integer dtype")
            if arrays["histories"].shape != (expected, 50):
                raise ValueError("History shape mismatch")
            if any(arrays[key].shape != (expected,) for key in ("targets", "users", "positions")):
                raise ValueError("Target/identity shape mismatch")
            count = 0
            with source.open(encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    padded = [0] * (50 - len(row["history"])) + row["history"]
                    if (not np.array_equal(arrays["histories"][index], padded)
                            or int(arrays["targets"][index]) != row["target"]
                            or int(arrays["users"][index]) != row["user_id"]
                            or int(arrays["positions"][index]) != row["target_position"]):
                        raise ValueError(f"Cache content differs at {relative} row {index}")
                    count += 1
            if count != expected:
                raise ValueError("Source count mismatch")
            checks.append({"split": relative, "rows": count, "source_sha256": source_hash})
    return {"status": "PASS", "data_manifest_sha256": accepted,
            "cache_sha256": digest(cache / "cache.json"), "checks": checks,
            "verified_rows": sum(c["rows"] for c in checks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = verify(args.data, args.cache)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))
