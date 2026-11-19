"""Shared-start, validation-only DIGER noise/update pilot.

This intentionally does NOT invoke Pipeline.run() or the historical fit loop.
It loads a complete shared model state, creates fresh optimizers for each arm,
updates the tokenizer at most once, and compares equal recommender budgets.
All new checkpoints and diagnostics stay underneath a new --output directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from types import MethodType
from typing import Any


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("all", "A", "B", "C"), default="all")
    parser.add_argument("--smoke", action="store_true", help="Truncated plumbing check; not a research result")
    parser.add_argument("--allow-cpu", action="store_true", help="Only permitted together with --smoke")
    return parser.parse_args(argv)


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def set_phase_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            raise TypeError(f"Non-tensor in model state: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def optimizer_receipt(optimizer):
    return None if optimizer is None else optimizer.state_dict()


def phase_rng_state(device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def rng_state_hash(device):
    """Small receipt for the streams which a diagnostic must not consume."""
    state = phase_rng_state(device)
    digest = hashlib.sha256(repr(state["python"]).encode("ascii"))
    numpy_state = state["numpy"]
    digest.update(repr((numpy_state[0], *numpy_state[2:])).encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    for name in ("torch_cpu", "torch_cuda"):
        value = state[name]
        if value is not None:
            digest.update(name.encode("ascii") + value.cpu().numpy().tobytes())
    return digest.hexdigest()


def feedback_loss_terms(output, config, expected_weight):
    """Verify the *actual* objective, not merely the arm's configuration label."""
    weight = float(config["joint_diger_recommendation_weight"])
    if weight != float(expected_weight) or weight not in (0.0, 1.0):
        raise RuntimeError("Recommendation feedback weight differs from the requested arm")
    if float(config.get("joint_tokenizer_sia_weight", 0.0)) != 0.0:
        raise RuntimeError("Feedback ablation requires SIA=0")
    auxiliary = (output.reconstruction_loss
                 + float(config["joint_balance_weight"]) * output.balance_loss
                 + float(config["joint_anchor_weight"]) * output.anchor_loss)
    weighted_recommendation = weight * output.recommendation_loss
    if not bool(torch.allclose(output.loss.detach(),
                              (auxiliary + weighted_recommendation).detach(),
                              rtol=1e-6, atol=1e-7)):
        raise RuntimeError("Actual tokenizer loss does not match auxiliary + weighted recommendation")
    return auxiliary, weighted_recommendation


def audit_feedback_gradient(model, output, auxiliary, weight):
    """Audit the contribution in total loss; raw direct gradients also exist at w=0."""
    parameters = [value for value in model.joint_alignment.parameters() if value.requires_grad]
    gradients = torch.autograd.grad(output.loss - auxiliary, parameters,
                                    retain_graph=True, allow_unused=True)
    values = [value.detach().float().square().sum() for value in gradients if value is not None]
    norm = float(torch.stack(values).sum().sqrt()) if values else 0.0
    raw_norm = float(output.diger_direct_gradient_norm)
    if not math.isfinite(norm) or not math.isfinite(raw_norm) or raw_norm <= 0:
        raise RuntimeError("Invalid recommendation feedback gradient audit")
    if weight == 0.0 and norm > 1e-8:
        raise RuntimeError("Reconstruction-only arm unexpectedly receives recommendation gradient")
    if weight == 1.0 and (norm <= 0 or not math.isclose(norm, raw_norm, rel_tol=1e-4, abs_tol=1e-7)):
        raise RuntimeError("Weighted recommendation gradient does not match the direct gradient")
    return {"recommendation_weight": weight, "raw_direct_gradient_norm": raw_norm,
            "weighted_direct_gradient_norm": norm, "measured_at_microbatch": 1}


def configure_arm(pipeline, settings):
    """Arm names are labels: budgets and loss weights determine execution."""
    weight = float(settings.get("recommendation_weight", 1.0))
    pipeline.config["joint_diger_recommendation_weight"] = weight
    pipeline.model.config["joint_diger_recommendation_weight"] = weight
    pipeline.model.joint_alignment.diger_noise_start = float(settings["noise_start"])
    pipeline.model.joint_alignment.diger_noise_end = float(settings["noise_end"])
    return int(settings["tokenizer_batches"]) > 0


def verify_controlled_settings(config, protocol):
    for key, expected in protocol.get("controlled_tokenizer_settings", {}).items():
        if key not in config or float(config[key]) != float(expected):
            raise ValueError(f"Resolved common control differs: {key}={config.get(key)!r}, expected {expected!r}")
    if protocol.get("controlled_tokenizer_settings"):
        for key, expected in (("train_batch_size", 64), ("eval_batch_size", 8),
                              ("gradient_accumulation_steps", 16)):
            if config.get(key) != expected:
                raise ValueError(f"Resolved shared execution control differs: {key}")


def compare_codes(initial, current):
    if initial.shape != current.shape or initial.ndim != 2 or not initial.size:
        raise ValueError("Expected matching nonempty SID arrays")
    changed = initial != current
    return {"changed_items": int(changed.any(-1).sum()),
            "item_change_rate": float(changed.any(-1).mean()),
            "changed_digits": int(changed.sum()), "digit_change_rate": float(changed.mean()),
            "per_position_change_rate": changed.mean(0).tolist()}


def validation_coverage(sequences, catalog_keys, changed_mask, max_history_len, min_seq_len=2):
    if len(set(catalog_keys)) != len(catalog_keys) or changed_mask.shape != (len(catalog_keys),):
        raise ValueError("Invalid catalogue order/coverage mask")
    changed_by_key = dict(zip(catalog_keys, changed_mask.tolist()))
    counts = {"examples": 0, "target_changed_examples": 0,
              "history_changed_examples": 0, "either_changed_examples": 0}
    for seq in sequences.values():
        if len(seq) < min_seq_len:
            continue
        target, history = seq[-2], seq[:-2][-max_history_len:]
        if any(key not in changed_by_key for key in [target, *history]):
            raise ValueError("Validation item missing from catalogue")
        target_changed = bool(changed_by_key[target])
        history_changed = any(changed_by_key[key] for key in history)
        counts["examples"] += 1
        counts["target_changed_examples"] += int(target_changed)
        counts["history_changed_examples"] += int(history_changed)
        counts["either_changed_examples"] += int(target_changed or history_changed)
    if not counts["examples"]:
        raise ValueError("No validation examples")
    for name in ("target", "history", "either"):
        counts[name + "_changed_rate"] = counts[name + "_changed_examples"] / counts["examples"]
    return counts


def absolute_engine_paths(config, project_root):
    """Keep historical relative configuration semantics despite an output cwd."""
    engine_root = project_root / "external" / "DiffGRM"
    result = dict(config)
    for key, value in list(result.items()):
        if isinstance(value, str) and (
            key.endswith("_path") or key.endswith("_dir")
            or key == "joint_recommender_checkpoint"
        ):
            candidate = Path(value).expanduser()
            result[key] = str(candidate.resolve() if candidate.is_absolute()
                              else (engine_root / candidate).resolve())
    return result


def finite_parameters(model):
    invalid = [name for name, param in model.named_parameters()
               if not bool(torch.isfinite(param.detach()).all())]
    if invalid:
        raise FloatingPointError(f"Non-finite parameters: {invalid[:8]}")


def gradient_sides(model):
    norms = {"recommender": 0.0, "tokenizer": 0.0}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError(f"Non-finite gradient: {name}")
            side = "tokenizer" if name.startswith("joint_alignment.") else "recommender"
            norms[side] += float(param.grad.detach().float().square().sum())
    return {name: math.sqrt(value) for name, value in norms.items()}


def restore_phase_mode(model, phase):
    """Undo upstream guided-scoring mode leakage before every micro-batch.

    set_joint_phase handles requires_grad/cache once at phase entry. Guided
    scoring temporarily calls model.eval()/train(), which does not preserve
    mixed Decoder/Tokenizer modes; never infer noise activation from schedule
    values alone. This changes only modes, not parameters or optimizer state.
    """
    if phase == "tokenizer":
        model.eval()
        model.joint_alignment.train()
    elif phase == "recommender":
        model.train()
        model.joint_alignment.eval()
    else:
        raise ValueError(f"Unknown training phase: {phase}")


class IsolatedGumbel:
    """Isolate only the assignment call's random consumption, not OCN forward.

    The original assignment implementation is reused verbatim. fork_rng saves
    and restores the surrounding torch random streams; an independent seed is
    used per call. The mixer is required to have dropout=0 for this isolation
    to represent Gumbel-only randomness rather than also isolating dropout.
    History and target remain independent calls, as in the existing protocol.
    """

    def __init__(self, alignment, device, seed):
        self.alignment, self.device, self.seed = alignment, device, int(seed)
        self.calls = 0
        self.training_enabled_calls = 0
        self.noise_applied_calls = 0
        self.original = alignment.diger_assignments

    def __enter__(self):
        controller = self

        def isolated(_alignment, item_ids, stochastic=True):
            if stochastic and not controller.alignment.training:
                raise RuntimeError("Tokenizer assignment is in eval mode; Gumbel would be silently skipped")
            training_enabled = bool(controller.alignment.training)
            noise_enabled = stochastic and training_enabled and float(
                controller.alignment.diger_schedule_state()["gumbel_noise_scale"]
            ) > 0
            devices = [controller.device.index or 0] if controller.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices, enabled=True):
                seed = controller.seed + controller.calls
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                result = controller.original(item_ids, stochastic=stochastic)
            controller.calls += 1
            controller.training_enabled_calls += int(training_enabled)
            controller.noise_applied_calls += int(noise_enabled)
            return result

        self.alignment.diger_assignments = MethodType(isolated, self.alignment)
        return self

    def __exit__(self, exc_type, exc, traceback):
        # Remove the instance override, restoring the unmodified class method.
        del self.alignment.diger_assignments


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_train_loader(pipeline, batch_size, shuffle_seed):
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    return torch.utils.data.DataLoader(
        pipeline.tokenized_datasets["train"], batch_size=batch_size, shuffle=True,
        collate_fn=pipeline.tokenizer.collate_fn["train"], generator=generator,
        num_workers=0,
    )


def autocast_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else nullcontext()


def train_phase(pipeline, optimizer, phase, total_batches, phase_index,
                protocol, device, batch_size, accumulator, gumbel_seed=None,
                feedback_weight=None, update_callback=None):
    model = pipeline.model
    model.set_joint_phase(phase)
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    # Seed by logical Decoder epoch, not wall-clock epoch or experiment arm.
    kind = "tokenizer" if phase == "tokenizer" else "decoder"
    phase_random_seed = phase_seed(int(protocol["tokenizer_phase_seed"] if phase == "tokenizer"
                                      else protocol["decoder_seed_base"]), kind, phase_index)
    shuffle_seed = phase_seed(int(protocol["data_seed_base"]), "data", phase_index)
    set_phase_seed(phase_random_seed)
    loader = make_train_loader(pipeline, batch_size, shuffle_seed)
    available = len(loader)
    limit = min(available, int(total_batches)) if total_batches else available
    if limit <= 0:
        raise ValueError("Training phase has no batches")
    if phase == "tokenizer" and limit != total_batches:
        raise ValueError("Tokenizer phase must execute its complete requested batch budget")
    means, counts, updates = {}, {}, 0
    feedback_audit = None
    data_digest = hashlib.sha256()
    started = time.perf_counter()
    isolated_context = IsolatedGumbel(model.joint_alignment, device, gumbel_seed) if phase == "tokenizer" else nullcontext()
    with isolated_context as isolated:
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= limit:
                break
            for key in ("history_item_ids", "target_item_id"):
                data_digest.update(raw_batch[key].contiguous().numpy().tobytes())
            batch = move_batch(raw_batch, device)
            restore_phase_mode(model, phase)
            with autocast_context(device, phase == "recommender"):
                output = model.forward_joint_tokenizer(batch, measure_direct_gradient=(batch_index == 0)) if phase == "tokenizer" else model(batch, return_loss=True)
                loss = output.loss
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"Non-finite {phase} loss at batch {batch_index + 1}")
            if phase == "tokenizer" and feedback_weight is not None:
                auxiliary, weighted_rec = feedback_loss_terms(output, model.config, feedback_weight)
                for name, value in (("auxiliary_loss", auxiliary), ("weighted_recommendation_loss", weighted_rec)):
                    scalar = float(value.detach())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"Non-finite {name}")
                    means[name] = means.get(name, 0.0) + scalar
                    counts[name] = counts.get(name, 0) + 1
                if batch_index == 0:
                    feedback_audit = audit_feedback_gradient(model, output, auxiliary, float(feedback_weight))
            # The final partial accumulation window is normalized by its true
            # number of micro-batches (1652 -> 103 full + 1 four-batch window).
            (loss / loss_divisor(batch_index, limit, accumulator)).backward()
            for name, attribute in (
                ("total_loss", "loss"), ("recommendation_loss", "recommendation_loss"),
                ("reconstruction_loss", "reconstruction_loss"), ("balance_loss", "balance_loss"),
                ("anchor_loss", "anchor_loss"), ("direct_gradient_norm", "diger_direct_gradient_norm"),
                ("assignment_entropy", "diger_assignment_entropy"), ("max_probability", "diger_max_probability"),
                ("softmax_temperature", "diger_softmax_temperature"), ("noise_scale", "diger_gumbel_noise_scale"),
            ):
                value = getattr(output, attribute, None)
                if value is not None:
                    value = float(value.detach()) if torch.is_tensor(value) else float(value)
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Non-finite {name}")
                    means[name] = means.get(name, 0.0) + value
                    counts[name] = counts.get(name, 0) + 1
            if (batch_index + 1) % accumulator == 0 or batch_index + 1 == limit:
                sides = gradient_sides(model)
                active = "tokenizer" if phase == "tokenizer" else "recommender"
                inactive = "recommender" if phase == "tokenizer" else "tokenizer"
                if sides[active] <= 0 or sides[inactive] != 0:
                    raise RuntimeError(f"Invalid gradient routing: {phase}, {sides}")
                clip = pipeline.config.get("max_grad_norm", 1.0)
                if clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError("Non-finite clipped gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if phase == "tokenizer":
                    model.joint_alignment.advance_diger_step()
                    if update_callback is not None:
                        update_callback(updates, batch_index + 1)
                print(f"[PILOT] {phase} index={phase_index} batch={batch_index + 1}/{limit} updates={updates}", flush=True)
        if phase == "tokenizer":
            expected_calls = 2 * limit  # one history and one target call per batch
            expected_noisy_calls = expected_calls if model.joint_alignment.diger_noise_start > 0 else 0
            if (isolated.calls != expected_calls or isolated.training_enabled_calls != expected_calls
                    or isolated.noise_applied_calls != expected_noisy_calls):
                raise RuntimeError("Tokenizer assignment/noise activation count differs from requested intervention")
    finite_parameters(model)
    return {
        "phase": phase, "logical_index": phase_index, "batches": limit,
        "optimizer_updates": updates, "phase_seed": phase_random_seed,
        "data_seed": shuffle_seed, "data_order_sha256": data_digest.hexdigest(),
        "gumbel_calls": isolated.calls if phase == "tokenizer" else 0,
        "assignment_training_enabled_calls": isolated.training_enabled_calls if phase == "tokenizer" else 0,
        "noise_applied_calls": isolated.noise_applied_calls if phase == "tokenizer" else 0,
        "loss_means": {key: value / counts[key] for key, value in means.items()},
        "feedback_gradient_audit": feedback_audit,
        "last_gradient_sides": sides, "elapsed_seconds": time.perf_counter() - started,
        "accumulation": accumulator, "final_window_batches": limit % accumulator or accumulator,
    }


def sid_catalog_stats(codes, sizes):
    """Fresh SID statistics; collisions count ALL items in nonsingleton groups."""
    def collision(values):
        _, counts = np.unique(values, axis=0, return_counts=True)
        collided = int(counts[counts > 1].sum())
        return {"unique_semantic_ids": int(len(counts)), "collided_items": collided,
                "collision_rate": collided / len(values),
                "collision_groups": int((counts > 1).sum()), "largest_group": int(counts.max())}
    return {"full_sid": collision(codes),
            "prefixes": {str(length): collision(codes[:, :length]) for length in range(1, codes.shape[1] + 1)},
            "positions": [{"position": index, "codebook_size": int(size),
                           "used_codes": int(len(np.unique(codes[:, index]))),
                           "utilization_rate": len(np.unique(codes[:, index])) / int(size)}
                          for index, size in enumerate(sizes)]}


def tokenizer_probe(pipeline, arm_dir, updates, batches, initial_codes, sequences, device):
    """Observe fresh nearest SIDs without refreshing deployment mappings or RNG."""
    alignment = pipeline.model.joint_alignment
    before_hash = tensor_state_hash(alignment.state_dict())
    before_buffers = tensor_state_hash(dict(pipeline.model.named_buffers()))
    before_modes = tuple((name, value.training) for name, value in pipeline.model.named_modules())
    before_rng = rng_state_hash(device)
    with torch.no_grad():
        books, _ = alignment.mixer.crossed_codebooks(return_attention=False)
        chunks, squared_error = [], 0.0
        for start in range(1, alignment.item_slots.shape[0], 2048):
            slots = alignment.item_slots[start:start + 2048].float()
            distances = alignment.mixer.squared_distances(slots, books.float())
            codes = distances.argmin(dim=-1)
            chunks.append(codes.cpu().numpy())
            restored = alignment.mixer.gather_codes(books, codes)
            squared_error += float((restored - slots).square().flatten(1).sum())
        fresh_codes = np.concatenate(chunks, axis=0)
        fresh_books = books.detach().cpu().numpy()
    if (before_hash != tensor_state_hash(alignment.state_dict())
            or before_buffers != tensor_state_hash(dict(pipeline.model.named_buffers()))
            or before_modes != tuple((name, value.training) for name, value in pipeline.model.named_modules())
            or before_rng != rng_state_hash(device)):
        raise RuntimeError("Tokenizer diagnostic changed model state, modes, or training RNG")
    changed = (fresh_codes != initial_codes).any(-1)
    target = arm_dir / "tokenizer_probes" / f"update{updates:04d}"
    target.mkdir(parents=True, exist_ok=False)
    np.save(target / "fresh_probe_codes.npy", fresh_codes)
    np.save(target / "crossed_codebooks.npy", fresh_books)
    stats = {"optimizer_updates": updates, "microbatches": batches,
             "diagnostic_only_not_resume_checkpoint": True,
             "no_mapping_refresh_no_rng_or_model_state_change": True,
             "tokenizer_tensor_state_sha256": before_hash,
             "semantic_ids_array_sha256": hashlib.sha256(fresh_codes.tobytes()).hexdigest(),
             "catalog_reconstruction_squared_l2_mean": squared_error / len(fresh_codes),
             "deterministic_sid_change": compare_codes(initial_codes, fresh_codes),
             "validation_coverage": validation_coverage(sequences, pipeline.tokenizer.id2item[1:], changed,
                                                        int(pipeline.config["max_hist_len"]), 2),
             "fresh_catalog_stats": sid_catalog_stats(fresh_codes, [fresh_books.shape[1]] * fresh_books.shape[0]),
             "files": {name: file_hash(target / name) for name in ("fresh_probe_codes.npy", "crossed_codebooks.npy")}}
    write_json(target / "summary.json", stats)
    print(f"[PROBE] update={updates} changed_items={stats['deterministic_sid_change']['changed_items']} (fresh SID; mapping untouched)", flush=True)
    return stats


def snapshot(pipeline, arm_dir, label, optimizers, markers, initial_codes, sequences, device):
    target = arm_dir / "snapshots" / label
    target.mkdir(parents=True, exist_ok=False)
    model = pipeline.model
    state = cpu_state(model)
    codes = model.joint_alignment.hard_codes[1:].detach().cpu().numpy().copy()
    changed = (codes != initial_codes).any(-1)
    with torch.no_grad():
        books, _ = model.joint_alignment.mixer.crossed_codebooks(return_attention=False)
    torch.save(state, target / "model.pt")
    torch.save({
        "optimizers": {name: optimizer_receipt(opt) for name, opt in optimizers.items()},
        "rng": phase_rng_state(device), "markers": markers,
        "fresh_optimizer_pilot_not_historical_exact_resume": True,
    }, target / "training_state.pt")
    np.save(target / "semantic_ids.npy", codes)
    np.save(target / "crossed_codebooks.npy", books.detach().cpu().numpy())
    stats = {
        "label": label, **markers, "model_tensor_state_sha256": tensor_state_hash(state),
        "semantic_ids_array_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        "deterministic_sid_change": compare_codes(initial_codes, codes),
        "validation_coverage": validation_coverage(sequences, pipeline.tokenizer.id2item[1:], changed,
                                                   int(pipeline.config["max_hist_len"]), 2),
        "collision_stats": pipeline.tokenizer.semantic_collision_stats,
        "schedule": model.joint_alignment.diger_schedule_state(),
        "files": {name: file_hash(target / name) for name in
                  ("model.pt", "training_state.pt", "semantic_ids.npy", "crossed_codebooks.npy")},
    }
    write_json(target / "summary.json", stats)
    return stats


def validate_snapshot(pipeline, loader, label, recorder_epoch, device):
    pipeline.model.set_joint_phase("recommender")
    pipeline.model.generate_w_decoding_graph = True
    started = time.perf_counter()
    with autocast_context(device, True):
        values = pipeline.trainer.evaluate(loader, split="val", epoch=recorder_epoch)
    metrics = {name: float(value) for name, value in values.items()}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Invalid validation metrics")
    print(f"[PILOT] {label} val={json.dumps(metrics, sort_keys=True)}", flush=True)
    return {"label": label, "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}


def configure_paths(base, output, arm=None):
    config = dict(base)
    current = output if arm is None else output / arm
    for key, suffix in (
        ("log_dir", "logs"), ("tensorboard_log_dir", "tensorboard"),
        ("ckpt_dir", "checkpoints"), ("diffusion_diagnostics_log_dir", "diagnostics"),
        ("run_metadata_dir", "metadata"),
    ):
        config[key] = str(current / suffix)
    return config


def shadow_dataset_cache(source_cache, runtime, category):
    """Copy mutable processed inputs; read raw archives through new links.

    Cache generation may therefore occur only below the new pilot directory.
    Linux links avoid copying large raw files; Windows falls back to copy2.
    Neither link creation nor copies change source assets or their permissions.
    """
    shadow_root = runtime / "data_cache"
    source = Path(source_cache) / "AmazonReviews2014" / category
    destination = shadow_root / "AmazonReviews2014" / category
    receipts = []
    for directory, names in (
        ("raw", (f"reviews_{category}_5.json.gz", f"meta_{category}.json.gz")),
        ("processed", ("id_mapping.json", "all_item_seqs.json", "metadata.sentence.json")),
    ):
        (destination / directory).mkdir(parents=True, exist_ok=False)
        for name in names:
            original = (source / directory / name).resolve()
            target = destination / directory / name
            if not original.is_file():
                raise FileNotFoundError(original)
            method = "copy"
            if directory == "raw" and os.name != "nt":
                target.symlink_to(original)
                method = "symlink_to_locked_source"
            else:
                shutil.copy2(original, target)
            receipts.append({"source": str(original), "destination": str(target), "method": method,
                             "sha256": file_hash(original)})
    return shadow_root, receipts


def run(args, protocol_helper=None):
    global torch, np, phase_seed, loss_divisor
    root = args.project_root.resolve()
    output_candidate = args.output.resolve() if args.output.is_absolute() else (root / args.output).resolve()
    # Hugging Face resolves this variable at import time, before Pipeline's
    # datasets imports; do not accidentally use the account's shared cache.
    os.environ["HF_DATASETS_CACHE"] = str(output_candidate / "runtime" / "hf_datasets_cache")
    import torch as torch_module
    import numpy as numpy_module
    torch, np = torch_module, numpy_module
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "external" / "DiffGRM"))
    # A temporary source delivery may execute outside --project-root. Load
    # the helper adjacent to the executed runner, not an older remote copy.
    helper = protocol_helper
    if helper is None:
        helper_path = Path(__file__).resolve().parents[1] / "checks" / "diger_noise_pilot_protocol.py"
        helper_spec = importlib.util.spec_from_file_location("_noise_pilot_protocol", helper_path)
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError(f"Cannot load executed protocol helper: {helper_path}")
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
    helper_path = Path(helper.__file__).resolve()
    validate_protocol, arm_settings = helper.validate_protocol, helper.arm_settings
    phase_seed, loss_divisor = helper.phase_seed, helper.loss_divisor
    verify_assets, source_receipt = helper.verify_assets, helper.source_receipt
    from genrec.pipeline import Pipeline
    from genrec.utils import get_config
    from genrec.models.DIFF_GRM.trainer import DIFF_GRMTrainer
    from datasets import disable_caching
    disable_caching()
    config_path = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    available_arms = list(helper.arm_names(protocol)) if hasattr(helper, "arm_names") else ["A", "B", "C"]
    selected_arms = available_arms if args.arm == "all" else [args.arm]
    if any(arm not in available_arms for arm in selected_arms):
        raise ValueError("Requested arm is outside the executed protocol")
    if args.allow_cpu and not args.smoke:
        raise ValueError("--allow-cpu is only permitted with --smoke")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This pilot requires one process and one GPU; do not launch with torchrun")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not (args.smoke and args.allow_cpu):
        raise RuntimeError("GPU required for the real pilot; CPU is allowed only for explicit smoke")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires BF16 GPU support")
    torch.set_num_threads(int(protocol.get("cpu_threads", 4)))
    torch.set_num_interop_threads(int(protocol.get("cpu_threads", 4)))
    # Keep nearest-neighbor logits in real FP32, including on H100.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    receipts = verify_assets(root, protocol)
    shared_checkpoint = (root / protocol["shared_checkpoint"]).resolve()
    shared_state = torch.load(shared_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(shared_state, dict) or not all(torch.is_tensor(value) for value in shared_state.values()):
        raise TypeError("Shared checkpoint must be a complete raw model state_dict")
    source_hash = tensor_state_hash(shared_state)
    if int(shared_state["joint_alignment.diger_optimizer_step"]) != int(protocol["expected_tokenizer_updates"]):
        raise ValueError("Shared checkpoint has unexpected tokenizer updates")
    if int(shared_state["joint_alignment.diger_total_optimizer_steps"]) != int(protocol["tokenizer_schedule_total_updates"]):
        raise ValueError("Shared checkpoint schedule denominator differs from historical 60 updates")
    output = args.output.resolve() if args.output.is_absolute() else (root / args.output).resolve()
    args.output = output
    output.mkdir(parents=True, exist_ok=False)
    args.output_owned = True
    write_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    runtime = output / "runtime"
    runtime.mkdir()
    os.environ["HF_DATASETS_CACHE"] = str(runtime / "hf_datasets_cache")
    config_layers = [str((root / name).resolve()) for name in protocol["config_layers"]]
    base = get_config("DIFF_GRM", "AmazonReviews2014", config_layers, {})
    base = absolute_engine_paths(base, root)
    base.update({
        "rand_seed": int(protocol["seed"]), "mixed_precision": "no",
        "joint_recommender_checkpoint": None, "joint_crosspse_enabled": True,
        "joint_gradient_mode": "diger_st", "joint_recommender_sia_weight": 0.0,
        "joint_tokenizer_sia_weight": 0.0, "joint_diger_recommendation_weight": 1.0,
        "joint_diger_assignment_temperature": float(protocol["assignment_temperature"]),
        "joint_diger_softmax_temperature_start": float(protocol["softmax_temperature"]),
        "joint_diger_softmax_temperature_end": float(protocol["softmax_temperature"]),
        "joint_diger_gumbel_noise_start": float(protocol["noise_start"]),
        "joint_diger_gumbel_noise_end": float(protocol["noise_end"]),
        "joint_diger_uncertainty_decay": "cosine", "force_regenerate_opq": False,
        "run_id": protocol["protocol_id"], "num_proc": 1,
    })
    verify_controlled_settings(base, protocol)
    if float(base.get("crosspse_dropout", 0.0)) != 0:
        raise ValueError("Gumbel isolation requires crosspse_dropout=0")
    if base.get("evaluation_level") != "sid" or base.get("eval_expand_sid_to_items", False):
        raise ValueError("The matched pilot must retain the historical SID-level evaluation")
    if args.smoke:
        base["train_batch_size"] = int(protocol.get("smoke", {}).get("train_batch_size", 2))
        base["eval_batch_size"] = int(protocol.get("smoke", {}).get("eval_batch_size", 2))
        base["max_eval_batches"] = int(protocol.get("smoke", {}).get("eval_batches", 2))
    else:
        base["max_eval_batches"] = None
    base = configure_paths(base, output / "runtime")
    shadow_root, shadow_receipts = shadow_dataset_cache(base["cache_dir"], runtime, base["category"])
    base["cache_dir"] = str(shadow_root)
    with working_directory(runtime):
        pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", checkpoint_path=str(shared_checkpoint), config_dict=base)
    if pipeline.accelerator.num_processes != 1:
        raise RuntimeError("Multiple Accelerator processes are unsupported")
    pipeline.model.to(device)
    if protocol.get("controlled_tokenizer_settings") and pipeline.model.joint_alignment.tokenizer_variant != "cross_pse":
        raise RuntimeError("All feedback arms must retain the same Cross-PSE interaction module")
    pipeline.model.load_state_dict(shared_state, strict=True)
    pipeline.model.sync_joint_tokenizer_mapping()
    initial_codes = shared_state["joint_alignment.hard_codes"][1:].numpy().copy()
    # Recompute on the actual execution backend before attributing any change
    # to learning; a precision/backend boundary change must not be hidden.
    pipeline.model.refresh_joint_codes()
    if not np.array_equal(pipeline.model.joint_alignment.hard_codes[1:].cpu().numpy(), initial_codes):
        raise RuntimeError("Execution-backend deterministic SID differs from shared checkpoint before learning")
    sequence_path = Path(base["cache_dir"]) / "AmazonReviews2014" / base["category"] / "processed" / "all_item_seqs.json"
    sequences = json.loads(sequence_path.read_text(encoding="utf-8"))
    val_loader = torch.utils.data.DataLoader(
        pipeline.tokenized_datasets["val"], batch_size=base["eval_batch_size"], shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn["val"],
        generator=torch.Generator().manual_seed(phase_seed(protocol["seed"], "evaluation", 0)), num_workers=0,
    )
    val_loader = pipeline.accelerator.prepare_data_loader(val_loader)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "smoke" if args.smoke else "pilot",
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "fresh_optimizer_pilot_not_historical_exact_resume": True,
        "shared_checkpoint": str(shared_checkpoint), "shared_model_tensor_sha256": source_hash,
        "config_sha256": file_hash(config_path), "assets": receipts,
        "source": source_receipt(root), "runner_sha256": file_hash(Path(__file__)),
        "executed_runner": str(Path(__file__).resolve()), "executed_helper": str(helper_path),
        "executed_helper_sha256": file_hash(helper_path), "executed_config": str(config_path),
        "executed_entrypoint": str(Path(getattr(args, "entrypoint", __file__)).resolve()),
        "executed_entrypoint_sha256": file_hash(Path(getattr(args, "entrypoint", __file__))),
        "tokenizer_variant": pipeline.model.joint_alignment.tokenizer_variant,
        "smoke_not_for_metric_comparison": bool(args.smoke),
        "tokenizer_forward_precision": "float32_tf32_disabled", "decoder_forward_precision": "bf16" if device.type == "cuda" else "float32",
        "gumbel_rng": "per-assignment-call isolated fork_rng; history and target remain independent",
        "phase_mode_guard": "restore mixed module modes per micro-batch; assert assignment training and noise activation counts",
        "mapping_cache": "isolated copied processed cache; newly derived mappings stay below pilot output",
        "shadow_cache": shadow_receipts,
        "raw_link_backup_note": "On Linux raw archives are symlinks to SHA-bound project assets, not standalone copies; retain source assets for restoration",
        "recovery_rng": "common seed by logical Decoder epoch and private DataLoader generator across all arms",
        "evaluation": "validation_only; no test selection; not strict item-level",
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "resolved_pipeline_config.json", {key: str(value) if key in ("device", "accelerator") else value for key, value in pipeline.config.items()})
    results = []
    try:
        for arm in selected_arms:
            arm_dir = output / arm
            arm_dir.mkdir()
            arm_runtime = arm_dir / "runtime"
            arm_runtime.mkdir()
            model = pipeline.model
            model.load_state_dict(shared_state, strict=True)
            model.sync_joint_tokenizer_mapping()
            if tensor_state_hash(model.state_dict()) != source_hash:
                raise RuntimeError(f"Arm {arm} does not start from the exact shared model/buffer state")
            settings = arm_settings(protocol, arm, smoke=args.smoke)
            update_tokenizer = configure_arm(pipeline, settings)
            pipeline.config.update(configure_paths(pipeline.config, output, arm))
            pipeline.config["run_id"] = protocol["protocol_id"] + "_" + arm
            if settings.get("audit_recommendation_feedback"):
                write_json(arm_dir / "resolved_pipeline_config.json", {
                    key: str(value) if key in ("device", "accelerator") else value
                    for key, value in pipeline.config.items()})
            with working_directory(arm_runtime):
                pipeline.trainer = DIFF_GRMTrainer(pipeline.config, model, pipeline.tokenizer)
            model.set_joint_phase("recommender")
            rec_optimizer = torch.optim.AdamW(
                list(model.joint_recommender_parameters()), lr=float(protocol["decoder_lr"]),
                weight_decay=float(protocol.get("decoder_weight_decay", 0.0)),
            )
            tok_groups = []
            if update_tokenizer:
                for group_name, parameters in model.joint_tokenizer_parameter_groups():
                    if parameters:
                        tok_groups.append({"params": parameters, "lr": float(base["joint_codebook_lr"] if group_name == "joint_codebook" else base["joint_mixer_lr"]),
                                           "weight_decay": float(base.get("joint_weight_decay", 1e-4)), "group_name": group_name})
            tok_optimizer = torch.optim.AdamW(tok_groups) if tok_groups else None
            if update_tokenizer and tok_optimizer is None:
                raise RuntimeError("Requested tokenizer update has no optimizer parameters")
            optimizers = {"recommender": rec_optimizer, "tokenizer": tok_optimizer}
            arm_result = {"arm": arm, "settings": settings, "training": [], "snapshots": [], "validation": [], "tokenizer_probes": []}
            write_json(arm_dir / "settings.json", settings)
            set_phase_seed(phase_seed(protocol["seed"], "evaluation", 0))
            markers = {"recommender_epochs": 0, "recommender_updates": 0, "tokenizer_updates": 0}
            arm_result["snapshots"].append(snapshot(pipeline, arm_dir, "initial", optimizers, markers, initial_codes, sequences, device))
            arm_result["validation"].append(validate_snapshot(pipeline, val_loader, "initial", 0, device))
            if tok_optimizer is not None:
                requested_probes = list(settings.get("tokenizer_probe_updates", []))

                def observe_update(updates, batches):
                    if updates in requested_probes:
                        arm_result["tokenizer_probes"].append(tokenizer_probe(
                            pipeline, arm_dir, updates, batches, initial_codes, sequences, device))

                phase_result = train_phase(pipeline, tok_optimizer, "tokenizer", settings["tokenizer_batches"], 0,
                                           protocol, device, base["train_batch_size"], protocol["gradient_accumulation_steps"],
                                           gumbel_seed=phase_seed(protocol["gumbel_seed"], "gumbel", 0),
                                           feedback_weight=(settings["recommendation_weight"] if settings.get("audit_recommendation_feedback") else None),
                                           update_callback=observe_update if requested_probes else None)
                arm_result["training"].append(phase_result)
                markers["tokenizer_updates"] = phase_result["optimizer_updates"]
                if [row["optimizer_updates"] for row in arm_result["tokenizer_probes"]] != requested_probes:
                    raise RuntimeError("Tokenizer diagnostic probe budget did not complete")
                model.refresh_joint_codes()
                post_tokenizer = snapshot(pipeline, arm_dir, "post_tokenizer", optimizers, markers.copy(), initial_codes, sequences, device)
                arm_result["snapshots"].append(post_tokenizer)
                if requested_probes and requested_probes[-1] == phase_result["optimizer_updates"]:
                    if arm_result["tokenizer_probes"][-1]["semantic_ids_array_sha256"] != post_tokenizer["semantic_ids_array_sha256"]:
                        raise RuntimeError("Final fresh probe SID differs from refreshed deployment SID")
                arm_result["validation"].append(validate_snapshot(pipeline, val_loader, "post_tokenizer", 1, device))
            else:
                arm_result["snapshots"].append(snapshot(pipeline, arm_dir, "post_tokenizer", optimizers, markers.copy(), initial_codes, sequences, device))
                arm_result["validation"].append({"label": "post_tokenizer", "metrics": dict(arm_result["validation"][0]["metrics"]),
                                                 "unchanged_control_reuses_initial_evaluation": True})
            frozen_tokenizer_hash = tensor_state_hash(model.joint_alignment.state_dict())
            for epoch in range(1, int(settings["decoder_epochs"]) + 1):
                phase_result = train_phase(pipeline, rec_optimizer, "recommender", settings["decoder_batch_limit"], epoch,
                                           protocol, device, base["train_batch_size"], protocol["gradient_accumulation_steps"])
                arm_result["training"].append(phase_result)
                markers["recommender_epochs"] = epoch
                markers["recommender_updates"] += phase_result["optimizer_updates"]
                if tensor_state_hash(model.joint_alignment.state_dict()) != frozen_tokenizer_hash:
                    raise RuntimeError(f"Arm {arm}: tokenizer weights/buffers changed during recovery")
                label = f"decoder_{epoch}"
                arm_result["snapshots"].append(snapshot(pipeline, arm_dir, label, optimizers, markers.copy(), initial_codes, sequences, device))
                arm_result["validation"].append(validate_snapshot(pipeline, val_loader, label, epoch + 1, device))
            arm_result["final_snapshot"] = f"snapshots/decoder_{settings['decoder_epochs']}"
            write_json(arm_dir / "metrics.json", arm_result)
            results.append(arm_result)
            del rec_optimizer, tok_optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Compare budgets/data order, not wall-clock labels. Stop on mismatch.
        if len(results) > 1:
            baseline = [row for row in results[0]["training"] if row["phase"] == "recommender"]
            for result in results[1:]:
                current = [row for row in result["training"] if row["phase"] == "recommender"]
                if len(current) != len(baseline):
                    raise RuntimeError(f"Unmatched Decoder epoch budget {result['arm']}")
                for a, b in zip(baseline, current):
                    for key in ("logical_index", "batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256"):
                        if a[key] != b[key]:
                            raise RuntimeError(f"Unmatched Decoder control {result['arm']}: {key}")
            tokenizer_rows = [row for result in results for row in result["training"] if row["phase"] == "tokenizer"]
            for row in tokenizer_rows[1:]:
                for key in ("batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256"):
                    if row[key] != tokenizer_rows[0][key]:
                        raise RuntimeError(f"Unmatched Tokenizer control: {key}")
        post_receipts = verify_assets(root, protocol)
        if post_receipts != receipts:
            raise RuntimeError("Bound source inputs changed during pilot execution")
        write_json(output / "post_input_verification.json", {"verified": True, "inputs": post_receipts})
        summary = {"protocol_id": protocol["protocol_id"], "mode": "smoke" if args.smoke else "pilot",
                   "status": "complete", "arms": results, "elapsed_seconds": time.perf_counter() - started,
                   "all_protocol_arms_complete": selected_arms == available_arms,
                   "no_test_evaluation": True, "no_historical_artifact_changes": True}
        write_json(output / "summary.json", summary)
        write_json(output / "COMPLETE.json", {"summary_sha256": file_hash(output / "summary.json"), "arms": selected_arms,
                                             "all_protocol_arms_complete": selected_arms == available_arms})
        print(f"[DONE] {output}", flush=True)
        return summary
    finally:
        pipeline.accelerator.end_training()


def main(argv=None):
    args = parse_args(argv)
    args.output_owned = False
    try:
        run(args)
    except Exception as exc:
        if args.output_owned and not (args.output.resolve() / "FAILED.json").exists():
            write_json(args.output.resolve() / "FAILED.json", {"error": type(exc).__name__, "message": str(exc), "complete": False})
        raise


if __name__ == "__main__":
    main()
