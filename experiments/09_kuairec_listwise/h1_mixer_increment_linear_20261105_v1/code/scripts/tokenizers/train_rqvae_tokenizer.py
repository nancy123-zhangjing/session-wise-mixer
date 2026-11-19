"""Train a frozen RQ-VAE tokenizer on a shared item-embedding matrix.

This deliberately runs before AR training.  It reuses the upstream GenRec
RQ-VAE model, but not its separate Amazon data pipeline, so RQ-VAE and
RQ-KMeans consume exactly the same item rows and source embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENREC_ROOT = PROJECT_ROOT / "external" / "genrec"
sys.path.insert(0, str(GENREC_ROOT))

from genrec.models.rqvae import QuantizeForwardMode, RqVae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run only a few steps in a separate output directory to estimate runtime.",
    )
    return parser.parse_args()


def load_config(paths: list[str]) -> dict:
    config: dict = {}
    for path in paths:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config.update(loaded)
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_split(n_items: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("sid_fit_val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    n_val = max(1, int(round(n_items * val_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def rqvae_objective(model: RqVae, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
    quantized = model.get_semantic_ids(batch, gumbel_t=0.2)
    quantized_sum = quantized.embeddings.sum(dim=-1)
    reconstruction = F.normalize(model.decode(quantized_sum), p=2, dim=-1)
    reconstruction_per_item = ((reconstruction - batch) ** 2).sum(dim=-1)
    quantization_per_item = quantized.quantize_loss
    total = (reconstruction_per_item + quantization_per_item).mean()
    return (
        total,
        reconstruction_per_item.mean(),
        quantization_per_item.mean(),
        reconstruction,
        quantized.sem_ids,
        quantized.residuals,
    )


def mark_codebooks_initialized(model: RqVae) -> None:
    """The upstream flag is not in state_dict; restore it after checkpoint load."""
    for layer in model.layers:
        layer.kmeans_initted = True


@torch.no_grad()
def evaluate(
    model: RqVae,
    embeddings: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: list[float] = []
    reconstructions: list[float] = []
    quantizations: list[float] = []
    weights: list[int] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = torch.from_numpy(embeddings[selected]).to(device)
        total, reconstruction, quantization, *_ = rqvae_objective(model, batch)
        totals.append(float(total))
        reconstructions.append(float(reconstruction))
        quantizations.append(float(quantization))
        weights.append(len(selected))
    return {
        "total_loss": float(np.average(totals, weights=weights)),
        "reconstruction_loss": float(np.average(reconstructions, weights=weights)),
        "quantization_loss": float(np.average(quantizations, weights=weights)),
    }


def scheduler_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def collision_report(codes: np.ndarray, codebook_size: int) -> dict:
    tuples = [tuple(map(int, row)) for row in codes]
    counts = Counter(tuples)
    per_prefix = []
    for prefix_length in range(1, codes.shape[1] + 1):
        prefix_counts = Counter(
            tuple(map(int, row[:prefix_length])) for row in codes
        )
        prefix_collided_items = sum(
            size for size in prefix_counts.values() if size > 1
        )
        per_prefix.append({
            "prefix_length": prefix_length,
            "items": len(codes),
            "unique_prefixes": len(prefix_counts),
            "collided_items": prefix_collided_items,
            "collision_rate": prefix_collided_items / max(len(codes), 1),
            "colliding_groups": sum(size > 1 for size in prefix_counts.values()),
            "largest_collision_group": max(prefix_counts.values(), default=0),
        })
    per_position = []
    for position in range(codes.shape[1]):
        _, code_counts = np.unique(codes[:, position], return_counts=True)
        probabilities = code_counts / code_counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        per_position.append(
            {
                "position": position + 1,
                "unique_codes": int(len(code_counts)),
                "unused_codes": int(max(codebook_size - len(code_counts), 0)),
                "utilization_rate": float(len(code_counts) / codebook_size),
                "entropy_nats": entropy,
                "normalized_entropy": float(entropy / math.log(codebook_size)),
                "effective_codes_perplexity": float(math.exp(entropy)),
                "largest_code_share": float(code_counts.max() / code_counts.sum()),
            }
        )
    collided_items = sum(size for size in counts.values() if size > 1)
    return {
        "items": len(codes),
        "unique_semantic_ids": len(counts),
        "collided_items": collided_items,
        "collision_rate": collided_items / max(len(codes), 1),
        "colliding_groups": sum(size > 1 for size in counts.values()),
        "largest_collision_group": max(counts.values(), default=0),
        "per_prefix": per_prefix,
        "per_position": per_position,
    }


@torch.no_grad()
def code_health(
    model: RqVae,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_codes = []
    for start in range(0, len(embeddings), batch_size):
        batch = torch.from_numpy(embeddings[start : start + batch_size]).to(device)
        all_codes.append(model.get_semantic_ids(batch, gumbel_t=0.001).sem_ids.cpu().numpy())
    codes = np.concatenate(all_codes, axis=0)
    unique_sid_ratio = len({tuple(map(int, row)) for row in codes}) / len(codes)
    utilizations = []
    normalized_entropies = []
    for position in range(codes.shape[1]):
        _, counts = np.unique(codes[:, position], return_counts=True)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        utilizations.append(len(counts) / model.codebook_size)
        normalized_entropies.append(entropy / math.log(model.codebook_size))
    return {
        "unique_sid_ratio": float(unique_sid_ratio),
        "min_code_utilization": float(min(utilizations)),
        "min_normalized_entropy": float(min(normalized_entropies)),
    }


def sampled_neighbor_recall(
    source: np.ndarray,
    reconstructed: np.ndarray,
    k: int,
    n_queries: int,
    seed: int,
) -> float:
    import faiss

    rng = np.random.default_rng(seed)
    query_ids = rng.choice(len(source), size=min(n_queries, len(source)), replace=False)
    source = np.ascontiguousarray(source.astype(np.float32, copy=False))
    reconstructed = np.ascontiguousarray(reconstructed.astype(np.float32, copy=False))
    source_index = faiss.IndexFlatIP(source.shape[1])
    reconstructed_index = faiss.IndexFlatIP(reconstructed.shape[1])
    source_index.add(source)
    reconstructed_index.add(reconstructed)
    _, source_neighbors = source_index.search(source[query_ids], k + 1)
    _, reconstructed_neighbors = reconstructed_index.search(reconstructed[query_ids], k + 1)
    recalls = []
    for query_id, left, right in zip(query_ids, source_neighbors, reconstructed_neighbors):
        left_set = {int(value) for value in left if int(value) != int(query_id)}
        right_set = {int(value) for value in right if int(value) != int(query_id)}
        recalls.append(len(set(list(left_set)[:k]) & set(list(right_set)[:k])) / k)
    return float(np.mean(recalls))


@torch.no_grad()
def encode_catalog(
    model: RqVae,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    model.eval()
    all_codes = []
    all_reconstructions = []
    residual_sums = np.zeros(model.n_layers, dtype=np.float64)
    seen = 0
    for start in range(0, len(embeddings), batch_size):
        batch = torch.from_numpy(embeddings[start : start + batch_size]).to(device)
        quantized = model.get_semantic_ids(batch, gumbel_t=0.001)
        reconstructed = F.normalize(
            model.decode(quantized.embeddings.sum(dim=-1)), p=2, dim=-1
        )
        all_codes.append(quantized.sem_ids.cpu().numpy())
        all_reconstructions.append(reconstructed.cpu().numpy())
        residual_sums += quantized.residuals.norm(dim=1).sum(dim=0).cpu().numpy()
        seen += len(batch)
    return (
        np.concatenate(all_codes, axis=0),
        np.concatenate(all_reconstructions, axis=0),
        (residual_sums / max(seen, 1)).tolist(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("rqvae_seed", 2024))
    max_steps = int(args.max_steps or config.get("rqvae_max_steps", 60000))
    eval_interval = int(args.eval_interval or config.get("rqvae_eval_interval", 1000))
    if args.profile:
        max_steps = min(max_steps, int(config.get("rqvae_profile_steps", 50)))
        eval_interval = min(eval_interval, max(10, max_steps // 2))

    output_dir = args.output_dir.resolve()
    if args.profile:
        output_dir = output_dir.parent / f"{output_dir.name}_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    csv_path = output_dir / "training.csv"

    embeddings = np.load(args.embedding.resolve()).astype(np.float32, copy=False)
    items = json.loads(args.items.resolve().read_text(encoding="utf-8"))
    if len(items) != len(embeddings):
        raise ValueError("The shared item list and embedding matrix have different lengths")
    input_dim = int(config.get("rqvae_input_dim", embeddings.shape[1]))
    if embeddings.shape[1] != input_dim:
        raise ValueError(f"Embedding dim {embeddings.shape[1]} != rqvae_input_dim {input_dim}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_indices, val_indices = catalog_split(
        len(embeddings), float(config.get("sid_fit_val_fraction", 0.1)), seed
    )
    model = RqVae(
        input_dim=input_dim,
        embed_dim=int(config.get("rqvae_embed_dim", 32)),
        hidden_dims=list(config.get("rqvae_hidden_dims", [512, 256, 128, 64])),
        codebook_size=int(config.get("codebook_size", 256)),
        codebook_kmeans_init=True,
        codebook_normalize=bool(config.get("rqvae_codebook_normalize", False)),
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
        n_layers=int(config.get("semantic_n_digit", config.get("n_digit", 3))),
        commitment_weight=float(config.get("rqvae_commitment_weight", 0.25)),
        n_cat_features=0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("rqvae_lr", 0.001)),
        weight_decay=float(config.get("rqvae_weight_decay", 0.0001)),
    )
    warmup_steps = int(round(max_steps * float(config.get("rqvae_warmup_ratio", 0.01))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_factor(step, warmup_steps, max_steps),
    )
    precision = str(
        config.get(
            "rqvae_mixed_precision",
            "fp16" if config.get("rqvae_fp16", True) else "no",
        )
    ).lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "rqvae_mixed_precision must be one of: no, fp16, bf16"
        )
    use_amp = precision != "no" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("RQ-VAE requested bf16, but this GPU does not support it")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == "fp16" and device.type == "cuda")
    )
    batch_size = int(config.get("rqvae_batch_size", 1024))
    eval_batch_size = int(config.get("rqvae_eval_batch_size", 1024))
    rng = np.random.default_rng(seed)
    start_step = 0
    best_val = float("inf")
    best_step = 0

    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        mark_codebooks_initialized(model)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng_state"]
        start_step = int(state["step"])
        best_val = float(state["best_val"])
        best_step = int(state["best_step"])
        print(f"Resuming RQ-VAE from step {start_step}")

    if not csv_path.exists() or args.restart:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["step", "train_total", "train_reconstruction", "train_quantization",
                 "val_total", "val_reconstruction", "val_quantization",
                 "unique_sid_ratio", "min_code_utilization", "min_normalized_entropy",
                 "healthy", "lr", "elapsed_seconds"]
            )

    started = time.perf_counter()
    rolling_total: list[float] = []
    rolling_reconstruction: list[float] = []
    rolling_quantization: list[float] = []
    for step in range(start_step, max_steps):
        model.train()
        selected = rng.choice(train_indices, size=batch_size, replace=len(train_indices) < batch_size)
        batch = torch.from_numpy(embeddings[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            total, reconstruction, quantization, *_ = rqvae_objective(model, batch)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        rolling_total.append(float(total.detach()))
        rolling_reconstruction.append(float(reconstruction.detach()))
        rolling_quantization.append(float(quantization.detach()))
        rolling_total = rolling_total[-eval_interval:]
        rolling_reconstruction = rolling_reconstruction[-eval_interval:]
        rolling_quantization = rolling_quantization[-eval_interval:]

        current_step = step + 1
        if current_step % 100 == 0 or current_step == 1:
            print(
                f"RQ-VAE step {current_step}/{max_steps} "
                f"loss={np.mean(rolling_total):.5f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
        if current_step % eval_interval != 0 and current_step != max_steps:
            continue

        metrics = evaluate(model, embeddings, val_indices, eval_batch_size, device)
        health = code_health(model, embeddings, eval_batch_size, device)
        min_utilization = float(config.get("rqvae_min_code_utilization", 0.25))
        min_unique_ratio = float(config.get("rqvae_min_unique_sid_ratio", 0.5))
        healthy = (
            health["min_code_utilization"] >= min_utilization
            and health["unique_sid_ratio"] >= min_unique_ratio
        )
        elapsed = time.perf_counter() - started
        train_values = (
            float(np.mean(rolling_total)),
            float(np.mean(rolling_reconstruction)),
            float(np.mean(rolling_quantization)),
        )
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [current_step, *train_values, metrics["total_loss"],
                 metrics["reconstruction_loss"], metrics["quantization_loss"],
                 health["unique_sid_ratio"], health["min_code_utilization"],
                 health["min_normalized_entropy"], int(healthy),
                 scheduler.get_last_lr()[0], elapsed]
            )
        if (healthy or args.profile) and metrics["total_loss"] < best_val:
            best_val = metrics["total_loss"]
            best_step = current_step
            torch.save(
                {"step": current_step, "model": model.state_dict(), "config": config,
                 "best_val": best_val},
                best_path,
            )
        torch.save(
            {"step": current_step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "rng_state": rng.bit_generator.state, "best_val": best_val, "best_step": best_step},
            last_path,
        )
        print(
            f"Validation step {current_step}: total={metrics['total_loss']:.5f}, "
            f"rec={metrics['reconstruction_loss']:.5f}, "
            f"unique={health['unique_sid_ratio']:.3f}, "
            f"min_util={health['min_code_utilization']:.3f}, "
            f"healthy={healthy}, best={best_val:.5f}@{best_step}",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError(
            'RQ-VAE never produced a non-collapsed checkpoint. '
            'Do not start downstream AR; inspect training.csv and adjust the tokenizer.'
        )
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    mark_codebooks_initialized(model)
    codes, reconstructed, residual_norms = encode_catalog(
        model, embeddings, eval_batch_size, device
    )
    semantic_ids = {item: [int(code) for code in row] for item, row in zip(items, codes)}
    (output_dir / "semantic_ids.json").write_text(
        json.dumps(semantic_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output_dir / "semantic_ids.npy", codes)
    np.save(output_dir / "reconstructed_embeddings.npy", reconstructed)

    cosine = np.sum(embeddings * reconstructed, axis=1)
    summary = {
        "seed": seed,
        "n_digit": int(config.get("semantic_n_digit", config.get("n_digit", 3))),
        "codebook_size": int(config.get("codebook_size", 256)),
        "input_dim": input_dim,
        "embed_dim": int(config.get("rqvae_embed_dim", 32)),
        "mixed_precision": precision,
        "max_steps": max_steps,
        "best_step": best_step,
        "best_validation_total_loss": best_val,
        "train_items": int(len(train_indices)),
        "validation_items": int(len(val_indices)),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "reconstruction_mse_sum": float(np.mean(np.sum((embeddings - reconstructed) ** 2, axis=1))),
        "reconstruction_cosine": float(np.mean(cosine)),
        "neighbor_recall_at_10": sampled_neighbor_recall(
            embeddings,
            reconstructed,
            k=10,
            n_queries=int(config.get("tokenizer_neighbor_queries", 1000)),
            seed=seed,
        ),
        "mean_residual_norm_before_position": residual_norms,
        **collision_report(codes, int(config.get("codebook_size", 256))),
    }
    summary["embedding_sha256"] = file_sha256(args.embedding.resolve())
    summary["items_sha256"] = file_sha256(args.items.resolve())
    summary["best_checkpoint_sha256"] = file_sha256(best_path)
    summary["semantic_ids_sha256"] = file_sha256(
        output_dir / "semantic_ids.json"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Frozen RQ-VAE semantic IDs: {output_dir / 'semantic_ids.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    main()
