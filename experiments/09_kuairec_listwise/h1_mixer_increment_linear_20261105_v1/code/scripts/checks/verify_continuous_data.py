"""Read-only, row-level CPU verification of stage-06 item-prefix datasets.

Full-prefix expectations come from the original interaction positions. Legacy
expectations execute only the actual project's _sliding_train method via AST,
with a tiny Dataset.from_dict stand-in; no torch/datasets imports are needed.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = ROOT / "scripts/prepare/prepare_continuous_data.py"
spec = importlib.util.spec_from_file_location("stage06_prepare", PREPARE_PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def legacy_reference(sequences, source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    functions = [node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "AbstractDataset"
                 for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_sliding_train"]
    if len(functions) != 1:
        raise ValueError("Cannot locate the actual legacy split method")
    namespace = {"Dataset": SimpleNamespace(from_dict=lambda value: value)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_sliding_train"](SimpleNamespace(all_item_seqs=sequences), min_hist=2, max_hist=50)


def reference_inputs(output, assets):
    raw = {name: (output / "inputs" / name).read_bytes() for name in prepare.INPUT_NAMES}
    prepare.check_input_bytes(raw, assets)
    sequences = prepare.parse_json(raw["all_item_seqs.json"])
    mapping = prepare.parse_json(raw["id_mapping.json"])
    # Independently check the original bijection, rather than using the writer's
    # normalized sequences or trusting a generated catalog.
    for kind in ("item", "user"):
        ids = mapping[kind + "2id"]
        reverse = mapping["id2" + kind]
        if ids.get("[PAD]") != 0 or reverse[0] != "[PAD]" or len(ids) != len(reverse):
            raise ValueError("Invalid original PAD/mapping contract")
        if any(type(value) is not int for value in ids.values()):
            raise ValueError("Original IDs must be integers")
        if sorted(ids.values()) != list(range(len(ids))):
            raise ValueError("Original IDs are not a bijection")
        if any(reverse[value] != key for key, value in ids.items()):
            raise ValueError("Original reverse mapping differs")
    if set(sequences) != set(mapping["user2id"]) - {"[PAD]"}:
        raise ValueError("Original user set differs")
    if any(not isinstance(seq, list) or len(seq) < 3 or
           any(item == "[PAD]" or item not in mapping["item2id"] for item in seq)
           for seq in sequences.values()):
        raise ValueError("Invalid original interactions")
    return sequences, mapping


def full_reference(sequences, mapping, split):
    for user in sorted(sequences, key=mapping["user2id"].__getitem__):
        seq = sequences[user]
        # The target positions, not item-value disjointness, define leakage.
        positions = ([p for p in range(len(seq)) if 0 < p < len(seq) - 2] if split == "train"
                     else [len(seq) + (-2 if split == "validation" else -1)])
        for position in positions:
            history = seq[:position][-50:]
            yield {"user_id": mapping["user2id"][user], "target_position": position,
                   "history_start": position - len(history),
                   "history": [mapping["item2id"][item] for item in history],
                   "target": mapping["item2id"][seq[position]]}


def legacy_records(reference, mapping, split):
    data = reference["val" if split == "validation" else split]
    records = []
    for user, seq in zip(data["user"], data["item_seq"]):
        history = seq[:-1][-50:]
        records.append({"user_id": mapping["user2id"][user], "target_position": len(seq) - 1,
                        "history_start": len(seq) - 1 - len(history),
                        "history": [mapping["item2id"][item] for item in history],
                        "target": mapping["item2id"][seq[-1]]})
    return iter(sorted(records, key=lambda record: (record["user_id"], record["target_position"])))


def check_rows(path, expected):
    count = 0
    with path.open("rb") as handle:
        for count, row in enumerate(expected, start=1):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Missing row {count}: {path.name}")
            actual = prepare.parse_json(raw)
            if actual != row or prepare.json_bytes(actual) != prepare.json_bytes(row):
                raise ValueError(f"Row differs from original-position oracle: {path}: {count}")
            if raw != prepare.json_bytes(actual):
                raise ValueError(f"Noncanonical record encoding: {path}: {count}")
        if handle.readline():
            raise ValueError(f"Unexpected extra row: {path}")
    return count


def verify(output, *, root=ROOT, config_path=prepare.CONFIG, legacy_source=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    config = prepare.load_config(prepare.project_path(root, config_path))
    lock_path = prepare.project_path(root, config["asset_lock"])
    assets = prepare.locked_assets(lock_path)
    manifest = prepare.parse_json((output / "manifest.json").read_bytes())
    marker = prepare.parse_json((output / "PREPARED.json").read_bytes())
    if marker != {"status": "generated_requires_independent_verification",
                  "manifest_sha256": prepare.sha256_file(output / "manifest.json"),
                  "provenance_sha256": prepare.sha256_file(output / "provenance.json")}:
        raise ValueError("Preparation marker does not bind the current manifest/provenance")
    expected_files = {"inputs/" + name for name in prepare.INPUT_NAMES} | {
        "resolved_protocol.json", "catalog_item_ids.json", "statistics.json"
    } | {f"{protocol}/{split}.jsonl" for protocol in prepare.PROTOCOLS for split in prepare.SPLITS}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("Output inventory must cover exactly both protocols and their inputs")
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual_files != expected_files | {"manifest.json", "provenance.json", "PREPARED.json"}:
        raise ValueError("Unexpected or missing files in prepared output")
    for relative, entry in manifest["files"].items():
        path = output / relative
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("Prepared files must be contained regular files")
        if entry["sha256"] != prepare.sha256_file(path) or entry["size_bytes"] != path.stat().st_size:
            raise ValueError(f"Output file hash/size differs: {relative}")
    resolved_raw = (output / "resolved_protocol.json").read_bytes()
    if resolved_raw != prepare.json_bytes(config):
        raise ValueError("Prepared protocol differs from the current explicit contract")
    if manifest.get("schema_version") != 1 or manifest.get("record_order") != "original_user_id_then_target_position" or \
       manifest.get("position_indexing") != "zero_based_in_original_user_sequence" or \
       manifest.get("history_representation") != "unpadded_original_item_ids":
        raise ValueError("Unknown record schema/order")
    if manifest.get("input_sha256") != {name: entry["sha256"] for name, entry in assets.items()}:
        raise ValueError("Manifest does not bind the original inputs")
    provenance = prepare.parse_json((output / "provenance.json").read_bytes())
    if provenance.get("local_cpu_only") is not True or provenance.get("gpu_used") is not False or \
       provenance.get("model_training") is not False or provenance.get("remote_assets_verified") is not False:
        raise ValueError("Incorrect scope/provenance flags")
    if provenance["asset_lock"]["sha256"] != prepare.sha256_file(lock_path):
        raise ValueError("Input authority lock changed")
    sequences, mapping = reference_inputs(output, assets)
    catalog = prepare.parse_json((output / "catalog_item_ids.json").read_bytes())
    if prepare.json_bytes(catalog) != prepare.json_bytes(sorted(mapping["item2id"][item]
                                                             for item in mapping["item2id"] if item != "[PAD]")):
        raise ValueError("Catalog changed IDs or includes PAD")
    legacy_source = Path(legacy_source or root / "external/DiffGRM/genrec/dataset.py").resolve()
    legacy = legacy_reference(sequences, legacy_source)
    counts = {}
    for protocol in prepare.PROTOCOLS:
        counts[protocol] = {}
        for split in prepare.SPLITS:
            expected = (full_reference(sequences, mapping, split) if protocol == prepare.PROTOCOLS[0]
                        else legacy_records(legacy, mapping, split))
            relative = f"{protocol}/{split}.jsonl"
            count = check_rows(output / relative, expected)
            if count != manifest["files"][relative].get("records"):
                raise ValueError("Manifest row count differs from verified rows")
            counts[protocol][split] = count
    # Shared holdouts must be byte-identical across protocols, not merely equal in count.
    for split in ("validation", "test"):
        hashes = {manifest["files"][f"{protocol}/{split}.jsonl"]["sha256"] for protocol in prepare.PROTOCOLS}
        if len(hashes) != 1:
            raise ValueError("Protocols changed the held-out examples")
    lengths = sorted(map(len, sequences.values()))
    train_items = {item for seq in sequences.values() for item in seq[:-2]}
    stats = {
        "users": len(sequences), "items": len(catalog), "interactions": sum(lengths),
        "min_sequence_length": min(lengths), "median_sequence_length": statistics.median(lengths),
        "p95_sequence_length_nearest_rank": lengths[(len(lengths) * 95 + 99) // 100 - 1],
        "max_sequence_length": max(lengths), "train_visible_items": len(train_items),
        "validation_targets_unseen_in_train_interactions": sum(seq[-2] not in train_items for seq in sequences.values()),
        "test_targets_unseen_in_train_interactions": sum(seq[-1] not in train_items for seq in sequences.values()),
        "train_targets": {protocol: counts[protocol]["train"] for protocol in prepare.PROTOCOLS},
        "validation": counts[prepare.PROTOCOLS[0]]["validation"],
        "test": counts[prepare.PROTOCOLS[0]]["test"],
    }
    if prepare.json_bytes(stats) != prepare.json_bytes(prepare.parse_json((output / "statistics.json").read_bytes())):
        raise ValueError("Statistics do not match independently verified inputs/rows")
    prepare.check_statistics(stats, config["expected"])
    return {"status": "PASS", "scope": "local_cpu_data_only", "output": str(output),
            "counts": counts, "statistics": stats, "verified_rows": sum(sum(v.values()) for v in counts.values()),
            "manifest_sha256": prepare.sha256_file(output / "manifest.json"),
            "legacy_source_sha256": prepare.sha256_file(legacy_source),
            "verifier_sha256": prepare.sha256_file(__file__),
            "content_sha256": {name: entry["sha256"] for name, entry in manifest["files"].items()},
            "remote_assets_verified": False, "gpu_used": False}


def compare(left, right, **kwargs):
    if Path(left).resolve() == Path(right).resolve():
        raise ValueError("Determinism requires two different output directories")
    results = [verify(path, **kwargs) for path in (left, right)]
    if results[0]["content_sha256"] != results[1]["content_sha256"]:
        raise ValueError("Independent output content hashes differ")
    return {"status": "PASS", "independent_outputs": 2,
            "all_content_hashes_equal": True, "runs": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=prepare.CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--legacy-source", type=Path)
    args = parser.parse_args(argv)
    kwargs = {"root": args.project_root, "config_path": args.config, "legacy_source": args.legacy_source}
    output = prepare.project_path(args.project_root, args.output)
    result = (compare(output, prepare.project_path(args.project_root, args.peer), **kwargs)
              if args.peer else verify(output, **kwargs))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
