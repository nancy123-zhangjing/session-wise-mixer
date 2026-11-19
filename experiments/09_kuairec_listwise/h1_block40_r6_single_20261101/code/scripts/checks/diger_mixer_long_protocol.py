"""Immutable inputs, exact phase budgets, and controls for the stage05 long study.

Standard library only. This command never allocates a GPU or loads model tensors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "stage05_shared_start_full_mixer_long_gumbel_v1"
CAMPAIGN = "goal_20261020_long_mixer_v1"
ARMS = ("no_mixer_frozen", "no_mixer_joint", "mixer_frozen", "mixer_joint")
DEFAULT_CONFIG = ROOT / "configs/05_crosspse_joint/long/mixer_long_seed2024.json"
ENGINE_FILES = (
    "genrec/models/DIFF_GRM/model.py", "genrec/models/DIFF_GRM/trainer.py",
    "genrec/models/DIFF_GRM/training_state.py", "genrec/models/DIFF_GRM/cross_pse.py",
    "genrec/models/DIFF_GRM/joint_cross_pse.py", "genrec/models/DIFF_GRM/tokenizer.py",
    "genrec/models/DIFF_GRM/collate.py", "genrec/models/DIFF_GRM/config.yaml",
    "genrec/default.yaml", "genrec/datasets/AmazonReviews2014/config.yaml",
    "genrec/datasets/AmazonReviews2014/dataset.py", "genrec/pipeline.py", "genrec/utils.py",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate(value):
    need(value["protocol_id"] == PROTOCOL and value["campaign_id"] == CAMPAIGN, "Unknown long study")
    need(value["arms"] == list(ARMS), "Exactly the four authorized ordered arms are required")
    expected = dict(seed=2024, recipe_id="cross_joint_v2", gpu_count=1, max_concurrent_gpu_jobs=4, gpu_type="H100",
                    max_allocated_gpu_seconds=43200, max_acceptance_gpu_seconds=1800,
                    decoder_warmup_epochs=5, cycles=10, decoder_epochs_per_cycle=3,
                    decoder_epochs=35, decoder_batches_per_epoch=1652,
                    tokenizer_batches_per_phase=96, gradient_accumulation_steps=16,
                    tokenizer_total_updates=60, decoder_total_updates=3640,
                    train_batch_size=64, eval_batch_size=8, decoder_lr=.001,
                    decoder_warmup_updates=500, decoder_weight_decay=0,
                    codebook_lr=1e-5, mixer_lr=1e-4, tokenizer_weight_decay=1e-4,
                    adam_betas=[.9, .999], adam_eps=1e-8, recommendation_weight=1,
                    reconstruction_weight=1, balance_weight=.01, anchor_weight=.01,
                    assignment_temperature=.1, balance_temperature=.1, softmax_temperature=1,
                    noise_start=1, noise_end=.2, noise_decay="cosine", sia_weight=0,
                    label_smoothing=.2, max_grad_norm=1, mixer_heads=4,
                    mixer_dropout=0, mixer_scope="full", mixer_residual_scale=1,
                    mixer_learn_residual_scale=False, mixer_zero_output=True)
    for key, actual in expected.items():
        if isinstance(actual, bool):
            need(type(value.get(key)) is bool, "Expected an explicit Boolean: " + key)
        need(value.get(key) == actual and (not isinstance(actual, (int, float))
             or isinstance(actual, bool) or not isinstance(value.get(key), bool)), "Fixed control differs: " + key)
    need(value["initial_decoder"] == "external/DiffGRM/saved/AmazonReviews2014_Aug-28-2026_19-01-02/pytorch_model.bin",
         "Must start before the historical five recommender epochs")
    need(value["evaluation"] == dict(level="sid", val_beam=32, test_beam=256,
         epochs=[5, 10, 15, 20, 25, 30, 35], primary_epoch=35,
         selection="0.8*ndcg@10+0.2*hit_rate@10", early_stopping=False), "Evaluation/selection differs")
    need(value["smoke"] == dict(decoder_warmup_epochs=1, cycles=1, decoder_epochs_per_cycle=1,
         train_batches=16, tokenizer_batches=16, eval_batches=2, train_batch_size=2, eval_batch_size=2), "Explicit smoke controls differ")
    for key in ("canonical_E0_file_sha256", "canonical_E0_array_sha256", "canonical_SID_array_sha256"):
        need(re.fullmatch("[0-9a-f]{64}", value[key]), "Invalid hash: " + key)
    for key in ("tokenizer_phase_seed", "decoder_seed_base", "data_seed_base", "gumbel_seed"):
        need(type(value[key]) is int and value[key] >= 0, "Invalid independent seed: " + key)
    return value


def phases(protocol, arm, smoke=False):
    validate(protocol)
    need(arm in ARMS, "Unknown arm")
    short = protocol["smoke"]
    warm = short["decoder_warmup_epochs"] if smoke else protocol["decoder_warmup_epochs"]
    cycles = short["cycles"] if smoke else protocol["cycles"]
    recovery = short["decoder_epochs_per_cycle"] if smoke else protocol["decoder_epochs_per_cycle"]
    tokenizer_batches = short["tokenizer_batches"] if smoke else protocol["tokenizer_batches_per_phase"]
    decoder_batches = short["train_batches"] if smoke else protocol["decoder_batches_per_epoch"]
    result, decoder, tokenizer = [], 0, 0
    def add(kind, index, batches):
        nonlocal decoder, tokenizer
        updates = math.ceil(batches / protocol["gradient_accumulation_steps"])
        decoder += kind == "recommender"
        tokenizer += updates if kind == "tokenizer" else 0
        result.append(dict(phase=kind, logical_index=index, batches=batches, optimizer_updates=updates,
                           decoder_epochs=decoder, tokenizer_updates=tokenizer))
    for _ in range(warm):
        add("recommender", decoder + 1, decoder_batches)
    for cycle in range(cycles):
        if arm.endswith("_joint"):
            add("tokenizer", cycle, tokenizer_batches)
        for _ in range(recovery):
            add("recommender", decoder + 1, decoder_batches)
    return result


def noise_scale(step, protocol):
    progress = min(max(step / protocol["tokenizer_total_updates"], 0), 1)
    return protocol["noise_end"] + (protocol["noise_start"] - protocol["noise_end"]) * .5 * (1 + math.cos(math.pi * progress))


def relative(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    need(not Path(name).is_absolute() and path.is_relative_to(root), "Input path escaped root: " + str(name))
    return path


def copy_bound(source, destination, digest):
    source, destination = Path(source), Path(destination)
    need(source.is_file() and sha(source) == digest, "Input missing or wrong hash: " + str(source))
    if destination.exists():
        need(destination.is_file() and not destination.is_symlink() and sha(destination) == digest,
             "Existing owned input differs: " + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".copying")
    shutil.copyfile(source, temporary)
    need(sha(temporary) == digest, "Input transfer checksum failed")
    temporary.replace(destination)


def prepare_inputs(protocol, project, package, goal):
    validate(protocol)
    project, package, goal = Path(project).resolve(), Path(package).resolve(), Path(goal).resolve()
    need(goal.name == CAMPAIGN, "Use the unique authorized campaign directory")
    if os.name != "nt":
        need(goal.parent == Path(protocol["remote_parent"]), "Remote output must be under the own global parent")
        need(project == Path(protocol["historical_project"]), "Read only the registered historical project")
    recipe_path = package / "reproducibility/recipes/cross_joint_v2.json"
    lock_path = package / "reproducibility/assets.lock.json"
    recipe, lock = read(recipe_path), read(lock_path)["artifacts"]
    need(recipe["initial_decoder"] == protocol["initial_decoder"] and not recipe["missing_from_archive"],
         "Wrong/missing pre-05 warm start")
    records = []
    need(not (goal / "inputs").is_symlink(), "Owned inputs must not be a symlink")
    for name in recipe["required_artifacts"]:
        digest = lock[name]["sha256"]
        destination = relative(goal / "inputs", name)
        need(destination.is_relative_to(goal), "Owned input destination escaped the campaign")
        copy_bound(relative(project, name), destination, digest)
        records.append(dict(path=name, sha256=digest, size_bytes=destination.stat().st_size))
    e0 = goal / "inputs/canonical/E0.npy"
    copy_bound(protocol["canonical_E0"], e0, protocol["canonical_E0_file_sha256"])
    records.append(dict(path="canonical/E0.npy", sha256=sha(e0), size_bytes=e0.stat().st_size))
    processed = goal / "inputs/data/cache/AmazonReviews2014/Beauty/processed"
    sequences, mapping = read(processed / "all_item_seqs.json"), read(processed / "id_mapping.json")
    lengths = [len(sequence) for sequence in sequences.values()]
    counts = dict(users=len(lengths), items=len(mapping["id2item"]) - 1, interactions=sum(lengths),
                  train=sum(max(0, min(length - 3, 50) - 2 + 1) for length in lengths),
                  val=sum(length > 1 for length in lengths), test=len(lengths))
    need(counts == dict(users=22363, items=12101, interactions=198502, train=105668, val=22363, test=22363),
         "Wrong Beauty data identity/counts")
    identity = dict(protocol_id=PROTOCOL, campaign_id=CAMPAIGN, protocol=protocol, records=records,
                    recipe_sha256=sha(recipe_path), assets_lock_sha256=sha(lock_path), counts=counts)
    destination = goal / "delivery/INPUTS.json"
    if destination.exists():
        need(read(destination) == identity, "Existing campaign input identity differs")
    else:
        write(destination, identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--package-root", type=Path, default=ROOT / "experiments/05_crosspse_joint")
    parser.add_argument("--goal-root", type=Path)
    args = parser.parse_args()
    protocol = validate(read(args.config))
    if args.plan_only:
        result = dict(protocol_id=PROTOCOL, campaign_id=CAMPAIGN, max_gpu_hours=12,
                      arms={arm: phases(protocol, arm)[-1] for arm in ARMS}, gpu_allocated=False)
    else:
        need(args.project_root is not None and args.goal_root is not None, "Input preparation needs explicit source and own goal roots")
        result = prepare_inputs(protocol, args.project_root, args.package_root, args.goal_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
