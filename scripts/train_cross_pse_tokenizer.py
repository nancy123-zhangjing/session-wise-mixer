"""Train controlled Refined-PSE or Cross-PSE tokenizers for DiffGRM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(DIFF_ROOT))

from genrec.models.DIFF_GRM.cross_pse import CrossPSECodebookMixer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--fit-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-artifact-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def load_config(paths: list[str]) -> dict:
    config: dict = {}
    for path in paths:
        config.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_report(codes: np.ndarray, codebook_size: int) -> dict:
    rows = [tuple(map(int, row)) for row in codes]
    counts = Counter(rows)
    per_prefix = []
    for length in range(1, codes.shape[1] + 1):
        prefix_counts = Counter(tuple(row[:length]) for row in rows)
        collided = sum(size for size in prefix_counts.values() if size > 1)
        per_prefix.append({
            "prefix_length": length,
            "items": len(rows),
            "unique_prefixes": len(prefix_counts),
            "collided_items": collided,
            "collision_rate": collided / max(len(rows), 1),
            "colliding_groups": sum(size > 1 for size in prefix_counts.values()),
            "largest_collision_group": max(prefix_counts.values(), default=0),
        })
    per_position = []
    for position in range(codes.shape[1]):
        _, code_counts = np.unique(codes[:, position], return_counts=True)
        probabilities = code_counts / max(code_counts.sum(), 1)
        entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
        per_position.append({
            "position": position + 1,
            "unique_codes": int(len(code_counts)),
            "unused_codes": int(codebook_size - len(code_counts)),
            "utilization_rate": float(len(code_counts) / codebook_size),
            "entropy_nats": entropy,
            "normalized_entropy": float(entropy / math.log(codebook_size)),
            "effective_codes_perplexity": float(math.exp(entropy)),
            "largest_code_share": float(code_counts.max() / code_counts.sum()),
        })
    collided = sum(size for size in counts.values() if size > 1)
    return {
        "items": len(rows),
        "unique_semantic_ids": len(counts),
        "collided_items": collided,
        "collision_rate": collided / max(len(rows), 1),
        "colliding_groups": sum(size > 1 for size in counts.values()),
        "largest_collision_group": max(counts.values(), default=0),
        "per_prefix": per_prefix,
        "per_position": per_position,
    }


def fit_original_pse(
    embeddings: np.ndarray,
    fit_mask: np.ndarray,
    n_digit: int,
    codebook_size: int,
    omp_threads: int,
    index_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the same CPU OPQ/PQ factory as the original DiffGRM tokenizer."""
    import faiss

    bits = int(round(math.log2(codebook_size)))
    if 2**bits != codebook_size:
        raise ValueError("Cross-PSE currently requires a power-of-two codebook")
    if embeddings.shape[1] % n_digit:
        raise ValueError("PCA dimension must be divisible by n_digit")
    faiss.omp_set_num_threads(omp_threads)
    index = faiss.index_factory(
        embeddings.shape[1],
        f"OPQ{n_digit},IVF1,PQ{n_digit}x{bits}",
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(np.ascontiguousarray(embeddings[fit_mask]))
    index.add(np.ascontiguousarray(embeddings))
    faiss.write_index(index, str(index_path))

    if not isinstance(index, faiss.IndexPreTransform):
        raise TypeError("Expected OPQ IndexPreTransform")
    transform = faiss.downcast_VectorTransform(index.chain.at(0))
    transformed = transform.apply_py(np.ascontiguousarray(embeddings)).astype(
        np.float32, copy=False
    )
    ivf = faiss.downcast_index(index.index)
    extracted_ivf = faiss.extract_index_ivf(ivf)
    pq = ivf.pq
    code_dim = embeddings.shape[1] // n_digit
    codebooks = faiss.vector_to_array(pq.centroids).reshape(
        n_digit, codebook_size, code_dim
    ).astype(np.float32, copy=False)
    # Match DiffGRM's original tokenizer exactly: read the encoded PQ bytes
    # from the IVF inverted list and align them by their stored item ids.
    # Calling pq.compute_codes(transformed) here would be wrong when IVF-PQ
    # encodes residuals relative to its coarse centroid.
    invlists = extracted_ivf.invlists
    if extracted_ivf.nlist != 1:
        raise ValueError("Cross-PSE v1 expects the original IVF1 layout")
    list_size = invlists.list_size(0)
    if list_size != len(embeddings):
        raise RuntimeError(
            f"IVF1 contains {list_size} vectors, expected {len(embeddings)}"
        )
    packed = faiss.rev_swig_ptr(
        invlists.get_codes(0), list_size * invlists.code_size
    ).reshape(list_size, invlists.code_size)
    stored_ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy()
    decoded = np.empty((list_size, n_digit), dtype=np.int64)
    for row_index, packed_code in enumerate(packed):
        reader = faiss.BitstringReader(
            faiss.swig_ptr(packed_code), invlists.code_size
        )
        decoded[row_index] = [reader.read(bits) for _ in range(n_digit)]
    base_codes = np.empty_like(decoded)
    base_codes[stored_ids] = decoded

    by_residual = bool(getattr(extracted_ivf, "by_residual", True))
    if by_residual:
        coarse_centroid = np.asarray(
            extracted_ivf.quantizer.reconstruct(0), dtype=np.float32
        )
        residual_embeddings = transformed - coarse_centroid[None, :]
    else:
        coarse_centroid = np.zeros(transformed.shape[1], dtype=np.float32)
        residual_embeddings = transformed

    # A strong invariant: nearest-centroid encoding in the recovered residual
    # space must reproduce the ids read from the original FAISS index.
    residual_slots = residual_embeddings.reshape(len(embeddings), n_digit, code_dim)
    nearest = np.empty_like(base_codes)
    for position in range(n_digit):
        distances = (
            residual_slots[:, position, None, :] - codebooks[position][None, :, :]
        )
        nearest[:, position] = np.square(distances).sum(axis=-1).argmin(axis=-1)
    match_rate = float(np.mean(nearest == base_codes))
    if match_rate < 0.999999:
        raise RuntimeError(
            "Recovered PSE residual space does not reproduce the original "
            f"FAISS codes (digit match rate={match_rate:.8f})"
        )
    return residual_embeddings, codebooks, base_codes, coarse_centroid


def initialize_or_load_pse(
    output_dir: Path,
    embeddings: np.ndarray,
    fit_mask: np.ndarray,
    config: dict,
    restart: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed_path = output_dir / "opq_residual_embeddings.npy"
    codebooks_path = output_dir / "base_pse_codebooks.npy"
    codes_path = output_dir / "base_pse_semantic_ids.npy"
    coarse_path = output_dir / "base_pse_coarse_centroid.npy"
    index_path = output_dir / "base_pse.faiss"
    paths = (transformed_path, codebooks_path, codes_path, coarse_path, index_path)
    if all(path.exists() for path in paths) and not restart:
        return (
            np.load(transformed_path).astype(np.float32, copy=False),
            np.load(codebooks_path).astype(np.float32, copy=False),
            np.load(codes_path),
            np.load(coarse_path).astype(np.float32, copy=False),
        )
    transformed, codebooks, codes, coarse_centroid = fit_original_pse(
        embeddings=embeddings,
        fit_mask=fit_mask,
        n_digit=int(config["n_digit"]),
        codebook_size=int(config["codebook_size"]),
        omp_threads=int(config.get("faiss_omp_num_threads", 16)),
        index_path=index_path,
    )
    np.save(transformed_path, transformed)
    np.save(codebooks_path, codebooks)
    np.save(codes_path, codes)
    np.save(coarse_path, coarse_centroid)
    return transformed, codebooks, codes, coarse_centroid


def scheduler_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def encode(
    model: CrossPSECodebookMixer,
    slots: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    crossed, _ = model.crossed_codebooks(return_attention=False)
    all_codes, all_reconstructed = [], []
    loss_sum = 0.0
    for start in range(0, len(slots), batch_size):
        batch = torch.from_numpy(slots[start : start + batch_size]).to(device)
        distances = model.squared_distances(batch, crossed)
        codes = distances.argmin(dim=-1)
        reconstructed = model.gather_codes(crossed, codes)
        per_item = reconstructed.flatten(1).sub(batch.flatten(1)).square().sum(dim=-1)
        loss_sum += float(per_item.sum())
        all_codes.append(codes.cpu().numpy())
        all_reconstructed.append(reconstructed.flatten(1).cpu().numpy())
    return (
        np.concatenate(all_codes),
        np.concatenate(all_reconstructed),
        loss_sum / max(len(slots), 1),
    )


@torch.no_grad()
def attention_by_book(model: CrossPSECodebookMixer) -> np.ndarray:
    model.eval()
    _, weights = model.crossed_codebooks(return_attention=True)
    if weights is None:
        return np.full((model.n_digit, model.n_digit), np.nan, dtype=np.float64)
    # [batch=1, heads, query, key]
    weights = weights[0]
    result = torch.zeros(
        model.n_digit, model.n_digit, dtype=torch.float64, device=weights.device
    )
    for query_book in range(model.n_digit):
        q0, q1 = query_book * model.codebook_size, (query_book + 1) * model.codebook_size
        for key_book in range(model.n_digit):
            k0, k1 = key_book * model.codebook_size, (key_book + 1) * model.codebook_size
            result[query_book, key_book] = weights[:, q0:q1, k0:k1].sum(-1).mean()
    return result.cpu().numpy()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("crosspse_seed", 2024))
    max_steps = int(config.get("crosspse_max_steps", 5000))
    eval_interval = int(config.get("crosspse_eval_interval", 100))
    if args.profile:
        max_steps = min(max_steps, int(config.get("crosspse_profile_steps", 50)))
        eval_interval = min(eval_interval, max(10, max_steps // 2))

    output_dir = args.output_dir.resolve()
    if args.profile:
        output_dir = output_dir.parent / f"{output_dir.name}_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    csv_path = output_dir / "training.csv"

    embeddings = np.load(args.embedding.resolve()).astype(np.float32, copy=False)
    items = json.loads(args.items.resolve().read_text(encoding="utf-8"))
    fit_mask = np.load(args.fit_mask.resolve()).astype(bool, copy=False)
    if len(items) != len(embeddings) or fit_mask.shape != (len(embeddings),):
        raise ValueError("Embedding, item list, and PSE fit mask are not row-aligned")
    if embeddings.shape[1] != int(config.get("sent_emb_pca", 256)):
        raise ValueError("Cross-PSE must consume the shared PCA embedding matrix")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    base_artifact_dir = (
        args.base_artifact_dir.resolve()
        if args.base_artifact_dir is not None else output_dir
    )
    residual_embeddings, base_codebooks, base_codes, coarse_centroid = (
        initialize_or_load_pse(
            base_artifact_dir,
            embeddings,
            fit_mask,
            config,
            args.restart and args.base_artifact_dir is None,
        )
    )
    if base_artifact_dir != output_dir:
        for name in (
            "opq_residual_embeddings.npy",
            "base_pse_codebooks.npy",
            "base_pse_semantic_ids.npy",
            "base_pse_coarse_centroid.npy",
            "base_pse.faiss",
        ):
            shutil.copy2(base_artifact_dir / name, output_dir / name)
    n_digit = int(config["n_digit"])
    code_dim = residual_embeddings.shape[1] // n_digit
    slots = residual_embeddings.reshape(len(residual_embeddings), n_digit, code_dim)
    fit_indices = np.flatnonzero(fit_mask)
    external_holdout_indices = np.flatnonzero(~fit_mask)
    validation_fraction = float(config.get("crosspse_validation_fraction", 0.0))
    split_seed = int(config.get("crosspse_split_seed", seed))
    selection_protocol = str(
        config.get("crosspse_selection_protocol", "heldout_reconstruction")
    )
    if selection_protocol == "fit_catalog_reconstruction":
        if validation_fraction != 0.0:
            raise ValueError(
                "fit_catalog_reconstruction requires "
                "crosspse_validation_fraction=0.0"
            )
        # PSE is a transductive catalog tokenizer. Fit the refinement and
        # select its checkpoint on exactly the same items used by the original
        # PSE; the small mask complement remains a diagnostic only.
        train_indices = fit_indices
        val_indices = fit_indices
        validation_source = "original_pse_fit_catalog_transductive_selection"
    elif selection_protocol == "heldout_reconstruction" and validation_fraction > 0.0:
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("crosspse_validation_fraction must be in (0, 0.5)")
        split_rng = np.random.default_rng(split_seed)
        shuffled = split_rng.permutation(fit_indices)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        val_indices = np.sort(shuffled[:validation_count])
        train_indices = np.sort(shuffled[validation_count:])
        validation_source = "internal_split_of_original_pse_fit_items"
    elif selection_protocol == "heldout_reconstruction":
        # Preserve the v1 protocol when the new option is absent.
        train_indices = fit_indices
        val_indices = external_holdout_indices
        validation_source = "original_pse_fit_mask_complement"
    else:
        raise ValueError(
            "crosspse_selection_protocol must be heldout_reconstruction or "
            "fit_catalog_reconstruction"
        )
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Cross-PSE train/validation split is empty")
    selection_equals_training_catalog = bool(
        np.array_equal(train_indices, val_indices)
        and np.array_equal(train_indices, fit_indices)
    )
    if (
        selection_protocol == "fit_catalog_reconstruction"
        and not selection_equals_training_catalog
    ):
        raise RuntimeError(
            "Catalog-selection invariant failed: training, checkpoint "
            "selection, and original PSE fit items must be identical"
        )
    print(
        "Cross-PSE split: "
        f"train={len(train_indices)}, validation={len(val_indices)}, "
        f"external_holdout={len(external_holdout_indices)}, "
        f"selection_protocol={selection_protocol}, "
        f"source={validation_source}, split_seed={split_seed}",
        flush=True,
    )

    model = CrossPSECodebookMixer(
        torch.from_numpy(base_codebooks),
        n_heads=int(config.get("crosspse_heads", 4)),
        dropout=float(config.get("crosspse_dropout", 0.0)),
        attention_scope=str(config.get("crosspse_attention_scope", "full")),
        residual_scale_init=float(config.get("crosspse_residual_scale_init", 0.0)),
        learn_residual_scale=bool(
            config.get("crosspse_learn_residual_scale", True)
        ),
        zero_init_attention_output=bool(
            config.get("crosspse_zero_init_attention_output", False)
        ),
        train_base_codebooks=bool(
            config.get("crosspse_train_base_codebooks", False)
        ),
        mixer_enabled=bool(config.get("crosspse_mixer_enabled", True)),
    ).to(device)
    mixer_lr = float(config.get("crosspse_lr", 0.001))
    codebook_lr = float(config.get("crosspse_codebook_lr", mixer_lr))
    weight_decay = float(config.get("crosspse_weight_decay", 0.0001))
    parameter_groups = []
    if isinstance(model.base_codebooks, torch.nn.Parameter):
        parameter_groups.append({
            "params": [model.base_codebooks],
            "lr": codebook_lr,
            "weight_decay": weight_decay,
            "group_name": "codebook",
        })
    mixer_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "base_codebooks" and parameter.requires_grad
    ]
    if mixer_parameters:
        parameter_groups.append({
            "params": mixer_parameters,
            "lr": mixer_lr,
            "weight_decay": weight_decay,
            "group_name": "mixer",
        })
    if not parameter_groups:
        raise ValueError("Cross-PSE has no trainable parameters")
    optimizer = torch.optim.AdamW(parameter_groups)
    warmup_steps = int(round(max_steps * float(config.get("crosspse_warmup_ratio", 0.05))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_factor(step, warmup_steps, max_steps),
    )
    precision = str(config.get("crosspse_mixed_precision", "bf16")).lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("crosspse_mixed_precision must be no, fp16, or bf16")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cross-PSE requested bf16 on a GPU without bf16 support")
    use_amp = precision != "no" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == "fp16" and device.type == "cuda")
    )
    batch_size = int(config.get("crosspse_batch_size", 1024))
    eval_batch_size = int(config.get("crosspse_eval_batch_size", 1024))
    balance_weight = float(config.get("crosspse_balance_weight", 0.01))
    anchor_weight = float(config.get("crosspse_anchor_weight", 0.1))
    balance_temperature = float(config.get("crosspse_balance_temperature", 0.1))
    patience = int(config.get("crosspse_patience", 10))
    min_delta = float(config.get("crosspse_min_delta", 1e-6))

    start_step, best_step, stale = 0, 0, 0
    best_val = float("inf")
    initial_validation_reconstruction = None
    initial_external_holdout_reconstruction = None
    rng = np.random.default_rng(seed)
    resuming = last_path.exists() and not args.restart
    if resuming:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng_state"]
        start_step = int(state["step"])
        best_step = int(state["best_step"])
        best_val = float(state["best_val"])
        initial_validation_reconstruction = state.get("initial_val")
        initial_external_holdout_reconstruction = state.get(
            "initial_external_holdout"
        )
        stale = int(state.get("stale", 0))
        print(f"Resuming Cross-PSE from step {start_step}", flush=True)

    if not csv_path.exists() or args.restart:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                "step", "train_total", "train_reconstruction", "train_balance",
                "train_anchor", "val_reconstruction", "residual_scale",
                "codebook_lr", "mixer_lr",
                "external_holdout_reconstruction", "elapsed_seconds",
            ])

    started = time.perf_counter()
    if not resuming:
        _, _, initial_validation_reconstruction = encode(
            model, slots[val_indices], eval_batch_size, device
        )
        if len(external_holdout_indices):
            _, _, initial_external_holdout_reconstruction = encode(
                model,
                slots[external_holdout_indices],
                eval_batch_size,
                device,
            )
        best_val = float(initial_validation_reconstruction)
        best_step = 0
        torch.save({
            "step": 0,
            "model": model.state_dict(),
            "config": config,
            "best_val": best_val,
        }, best_path)
        group_lrs = {
            group.get("group_name", f"group_{index}"): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                0, "", "", "", "", best_val,
                float(model.residual_scale.detach()),
                group_lrs.get("codebook", 0.0), group_lrs.get("mixer", 0.0),
                initial_external_holdout_reconstruction, 0.0,
            ])
        print(
            f"Selection step 0 (original PSE initialization): "
            f"reconstruction={best_val:.6f}, "
            f"external_holdout={initial_external_holdout_reconstruction}",
            flush=True,
        )
    rolling: list[tuple[float, float, float, float]] = []
    for step in range(start_step, max_steps):
        model.train()
        selected = rng.choice(
            train_indices, size=batch_size, replace=len(train_indices) < batch_size
        )
        batch = torch.from_numpy(slots[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            result = model.objective(
                batch, balance_weight, anchor_weight, balance_temperature
            )
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"Non-finite Cross-PSE loss at step {step + 1}")
        scaler.scale(result["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        rolling.append(tuple(float(result[key].detach()) for key in (
            "loss", "reconstruction", "balance", "anchor"
        )))
        rolling = rolling[-eval_interval:]
        current_step = step + 1
        current_lrs = {
            group.get("group_name", f"group_{index}"): lr
            for index, (group, lr) in enumerate(
                zip(optimizer.param_groups, scheduler.get_last_lr())
            )
        }
        if current_step == 1 or current_step % 100 == 0:
            print(
                f"Cross-PSE step {current_step}/{max_steps} "
                f"loss={np.mean([row[0] for row in rolling]):.6f} "
                f"alpha={float(model.residual_scale.detach()):.4f} "
                f"codebook_lr={current_lrs.get('codebook', 0.0):.2e} "
                f"mixer_lr={current_lrs.get('mixer', 0.0):.2e}",
                flush=True,
            )
        if current_step % eval_interval and current_step != max_steps:
            continue

        _, _, val_reconstruction = encode(
            model, slots[val_indices], eval_batch_size, device
        )
        external_holdout_reconstruction = None
        if len(external_holdout_indices):
            _, _, external_holdout_reconstruction = encode(
                model,
                slots[external_holdout_indices],
                eval_batch_size,
                device,
            )
        elapsed = time.perf_counter() - started
        means = [float(np.mean([row[i] for row in rolling])) for i in range(4)]
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                current_step, *means, val_reconstruction,
                float(model.residual_scale.detach()),
                current_lrs.get("codebook", 0.0),
                current_lrs.get("mixer", 0.0),
                external_holdout_reconstruction, elapsed,
            ])
        improved = val_reconstruction < best_val - min_delta
        if improved:
            best_val, best_step, stale = val_reconstruction, current_step, 0
            torch.save({
                "step": current_step,
                "model": model.state_dict(),
                "config": config,
                "best_val": best_val,
            }, best_path)
        else:
            stale += 1
        torch.save({
            "step": current_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": rng.bit_generator.state,
            "best_step": best_step,
            "best_val": best_val,
            "initial_val": initial_validation_reconstruction,
            "initial_external_holdout": initial_external_holdout_reconstruction,
            "stale": stale,
        }, last_path)
        print(
            f"Selection step {current_step}: reconstruction={val_reconstruction:.6f}, "
            f"external_holdout={external_holdout_reconstruction}, "
            f"best={best_val:.6f}@{best_step}, stale={stale}/{patience}",
            flush=True,
        )
        if not args.profile and patience > 0 and stale >= patience:
            print(f"Early stopping Cross-PSE at step {current_step}", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError("Cross-PSE did not produce a checkpoint")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    codes, reconstructed, catalog_reconstruction = encode(
        model, slots, eval_batch_size, device
    )
    semantic_ids = {item: [int(v) for v in row] for item, row in zip(items, codes)}
    base_semantic_ids = {
        item: [int(v) for v in row] for item, row in zip(items, base_codes)
    }
    semantic_path = output_dir / "semantic_ids.json"
    base_semantic_path = output_dir / "base_pse_semantic_ids.json"
    semantic_path.write_text(
        json.dumps(semantic_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    base_semantic_path.write_text(
        json.dumps(base_semantic_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output_dir / "semantic_ids.npy", codes)
    reconstructed_opq = reconstructed + coarse_centroid[None, :]
    np.save(output_dir / "reconstructed_opq_embeddings.npy", reconstructed_opq)
    book_attention = attention_by_book(model)
    np.save(output_dir / "attention_by_book.npy", book_attention)
    with torch.no_grad():
        final_codebooks, _ = model.crossed_codebooks(return_attention=False)
        delta = final_codebooks - model.initial_codebooks
        codebook_delta_rms = float(delta.square().mean().sqrt().cpu())
        base_codebook_rms = float(
            model.initial_codebooks.square().mean().sqrt().cpu()
        )
        relative_codebook_delta_rms = codebook_delta_rms / max(
            base_codebook_rms, 1e-12
        )
        base_parameter_delta_rms = float(
            model.base_codebooks.sub(model.initial_codebooks)
            .square().mean().sqrt().cpu()
        )
        mixer_delta_rms = float(
            final_codebooks.sub(model.base_codebooks).square().mean().sqrt().cpu()
        )

    original_opq = residual_embeddings + coarse_centroid[None, :]
    np.save(output_dir / "opq_embeddings.npy", original_opq)
    positions = np.arange(n_digit)[None, :]
    base_reconstructed = base_codebooks[positions, base_codes].reshape(
        len(base_codes), -1
    )
    base_reconstructed_opq = base_reconstructed + coarse_centroid[None, :]
    base_reconstruction_mse = float(
        np.square(base_reconstructed - residual_embeddings).sum(axis=1).mean()
    )
    base_cosine = np.sum(original_opq * base_reconstructed_opq, axis=1) / (
        np.linalg.norm(original_opq, axis=1)
        * np.linalg.norm(base_reconstructed_opq, axis=1)
        + 1e-12
    )
    cosine = np.sum(original_opq * reconstructed_opq, axis=1) / (
        np.linalg.norm(original_opq, axis=1)
        * np.linalg.norm(reconstructed_opq, axis=1)
        + 1e-12
    )
    external_holdout_reconstruction = None
    if len(external_holdout_indices):
        _, _, external_holdout_reconstruction = encode(
            model, slots[external_holdout_indices], eval_batch_size, device
        )
    initial_validation_reconstruction = float(
        initial_validation_reconstruction
        if initial_validation_reconstruction is not None
        else best_state["best_val"]
    )
    best_validation_reconstruction = float(best_state["best_val"])
    validation_improvement = (
        initial_validation_reconstruction - best_validation_reconstruction
    )
    attention_summary = book_attention.tolist() if model.mixer_enabled else None
    mean_cross_book_attention_mass = None
    if model.mixer_enabled:
        mean_cross_book_attention_mass = float(
            (book_attention.sum(axis=1) - np.diag(book_attention)).mean()
        )
    summary = {
        "tokenizer_type": "cross_pse" if model.mixer_enabled else "refined_pse",
        "run_id": config.get("run_id"),
        "reproduction_protocol": config.get("reproduction_protocol"),
        "seed": seed,
        "n_digit": n_digit,
        "codebook_size": int(config["codebook_size"]),
        "code_dim": code_dim,
        "base_codebooks_trainable": model.train_base_codebooks,
        "mixer_enabled": model.mixer_enabled,
        "attention_scope": model.attention_scope,
        "attention_heads": model.n_heads,
        "zero_init_attention_output": model.zero_init_attention_output,
        "learn_residual_scale": model.learn_residual_scale,
        "mixed_precision": precision,
        "max_steps": max_steps,
        "best_step": int(best_state["step"]),
        "initial_validation_reconstruction": initial_validation_reconstruction,
        "best_validation_reconstruction": best_validation_reconstruction,
        "validation_reconstruction_improvement": validation_improvement,
        "best_improves_initial": bool(
            int(best_state["step"]) > 0 and validation_improvement > min_delta
        ),
        "catalog_reconstruction_mse_sum": float(catalog_reconstruction),
        "catalog_reconstruction_cosine": float(np.mean(cosine)),
        "base_catalog_reconstruction_mse_sum": base_reconstruction_mse,
        "base_catalog_reconstruction_cosine": float(np.mean(base_cosine)),
        "digit_assignment_change_rate_from_base": float(np.mean(codes != base_codes)),
        "item_sid_change_rate_from_base": float(np.mean(np.any(codes != base_codes, axis=1))),
        "residual_scale": float(model.residual_scale.detach()),
        "codebook_delta_rms": codebook_delta_rms,
        "relative_codebook_delta_rms": relative_codebook_delta_rms,
        "base_parameter_delta_rms": base_parameter_delta_rms,
        "mixer_delta_rms": mixer_delta_rms,
        "codebook_lr": codebook_lr,
        "mixer_lr": mixer_lr if model.mixer_enabled else None,
        "train_items": int(len(train_indices)),
        "validation_items": int(len(val_indices)),
        "validation_fraction": validation_fraction,
        "validation_split_seed": split_seed,
        "validation_source": validation_source,
        "selection_protocol": selection_protocol,
        "selection_items": int(len(val_indices)),
        "selection_equals_training_catalog": selection_equals_training_catalog,
        "initial_selection_reconstruction": initial_validation_reconstruction,
        "best_selection_reconstruction": best_validation_reconstruction,
        "selection_reconstruction_improvement": validation_improvement,
        "external_holdout_items": int(len(external_holdout_indices)),
        "initial_external_holdout_reconstruction_mse_sum": (
            initial_external_holdout_reconstruction
        ),
        "external_holdout_reconstruction_mse_sum": external_holdout_reconstruction,
        "attention_by_query_book_to_key_book": attention_summary,
        "mean_cross_book_attention_mass": mean_cross_book_attention_mass,
        "base_pse": collision_report(base_codes, int(config["codebook_size"])),
        "base_pse_semantic_ids_sha256": sha256(base_semantic_path),
        "base_pse_codebooks_sha256": sha256(output_dir / "base_pse_codebooks.npy"),
        "base_pse_faiss_sha256": sha256(output_dir / "base_pse.faiss"),
        **collision_report(codes, int(config["codebook_size"])),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "embedding_sha256": sha256(args.embedding.resolve()),
        "items_sha256": sha256(args.items.resolve()),
        "fit_mask_sha256": sha256(args.fit_mask.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "semantic_ids_sha256": sha256(semantic_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Frozen {summary['tokenizer_type']} semantic IDs: {semantic_path}")


if __name__ == "__main__":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    main()
