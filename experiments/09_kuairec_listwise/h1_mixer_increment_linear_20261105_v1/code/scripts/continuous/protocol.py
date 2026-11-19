"""Frozen campaign definition and filesystem primitives (standard library only)."""
import hashlib
import json
import os
import random
from pathlib import Path


SEARCH_SEED = 20261015
MODELS = ("sasrec_fullce", "esasrec")
COMMON = {
    "n_items": 12101, "max_length": 50, "ffn_multiplier": 4,
    "max_epochs": 300, "patience": 50, "adam_betas": [0.9, 0.98],
    "grad_clip": 1.0, "label_smoothing": 0.0, "eval_batch_size": 512,
    "precision": "bf16", "lr_schedule": "constant", "cpu_threads": 2,
    "evaluation": "full_catalog_item_id_tie_ascending_no_history_filter",
}
SPACE = {
    "dim": [64, 128, 256, 512], "layers": [1, 2, 4], "heads": [1, 2, 4],
    "dropout": [0.1, 0.2, 0.3, 0.5], "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4], "batch_size": [128, 256],
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generate_search():
    rng = random.Random(SEARCH_SEED)
    trials = []
    for model in MODELS:
        configurations = [
            dict(dim=64, layers=1, heads=1, dropout=0.2, lr=1e-3,
                 weight_decay=0.0, batch_size=128),
            dict(dim=256, layers=2, heads=4, dropout=0.2, lr=3e-4,
                 weight_decay=1e-5, batch_size=256),
        ]
        if model == "esasrec":
            for config in configurations:
                config["negatives"] = 256
        seen = {canonical_hash(c) for c in configurations}
        while len(configurations) < 32:
            candidate = {name: rng.choice(values) for name, values in SPACE.items()}
            if model == "esasrec":
                candidate["negatives"] = rng.choice([256, 1024])
            digest = canonical_hash(candidate)
            if digest not in seen:
                seen.add(digest)
                configurations.append(candidate)
        for index, sampled in enumerate(configurations):
            config = {**COMMON, "model": model, **sampled}
            trials.append({"trial_id": f"{model}_{index:02d}", "phase": "search",
                           "seed": 2024, "protocol": "full_train_prefix_v1",
                           "config": config, "config_sha256": canonical_hash(config)})
    return {
        "schema_version": 1, "search_seed": SEARCH_SEED,
        "budget": {"search": 64, "confirmation": 12, "formal": 10, "legacy": 3,
                   "valid_training_runs": 89, "max_infrastructure_retries": 2},
        "confirmation_seeds": [2025, 2026], "formal_seeds": list(range(3031, 3036)),
        "legacy_seeds": [3031, 3032, 3033], "selection_metric": "NDCG@10",
        "tie_breaks": ["HR@10_desc", "parameter_count_asc", "trial_id_asc"],
        "workers": 8, "gpu_count": 8, "nodes": 1,
        "space": {**SPACE, "esasrec_negatives": [256, 1024]}, "trials": trials,
    }
