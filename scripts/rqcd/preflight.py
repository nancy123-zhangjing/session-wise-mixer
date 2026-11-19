"""Bounded synthetic H100 check/benchmark, never a real-data experiment.

Example (on an already allocated H100):
 python -m scripts.rqcd.preflight --device cuda:0 --require-h100 --output NEW_DIR
CPU checks are explicitly FP32 and cannot qualify CUDA/BF16 kernels.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import torch

from .common import (DEFAULT_CONFIG, load_config, configure_runtime, seed_all, derive_seed,
                     cpu_copy, output_lock, write_json, source_binding, attention_backend)
from .catalog import Catalog, generate
from .model import ContinuousListModel, Diffusion
from .training import Trainer
from .tokenizer import RQVAE


def synthetic_asset(cfg):
    # Small catalog for diagnostics, full network geometry. Do not let benchmark
    # fixtures contaminate formal assets or claim full-catalog throughput.
    c = cfg["rq"]
    g = torch.Generator().manual_seed(72)
    n = max(512, c["codes"]*2)
    books = torch.randn(c["levels"], c["codes"], c["latent_dim"], generator=g)*0.2
    z = torch.randn(n, c["latent_dim"], generator=g)
    res, codes = z.clone(), []
    for book in books:
        code = (res[:, None]-book).square().sum(-1).argmin(-1)
        codes.append(code)
        res = res-book[code]
    return {"z": z, "sid": torch.stack(codes, -1), "books": books,
            "mean": z.mean(0), "scale": z.std(), "errors": res.square().sum(-1),
            "original_ids": torch.arange(n)}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if args.require_h100:
        if device.type != "cuda" or not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(device):
            raise RuntimeError("--require-h100 refuses a CPU or different GPU")
    if not 1 <= args.updates <= 10:
        raise ValueError("Preflight bounded to 1..10 timed optimizer updates")
    if device.type == "cpu":
        cfg["training"]["precision"] = "fp32"
        cfg["runtime"]["data_device"] = "cpu"
        torch.set_num_threads(min(4, torch.get_num_threads()))
    runtime = configure_runtime(cfg, device)
    seed_all(cfg["seed"])
    with output_lock(args.output) as root:
        catalog = Catalog(synthetic_asset(cfg)).to(device)
        b = cfg["training"]["effective_batch"] if device.type == "cuda" else 8
        history = torch.arange(b*cfg["history_max"], device=device).reshape(b, -1) % (len(catalog.y)-1)+1
        history[:, :10] = 0
        targets = torch.arange(b*cfg["list_length"], device=device).reshape(b, -1) % (len(catalog.y)-1)+1
        trainer = Trainer(cfg, catalog, max(b, cfg["training"]["effective_batch"])*4, {"preflight": True})

        # FP32 equivalence of optimized views and target-by-target reference.
        check_cfg = copy.deepcopy(cfg)
        check_cfg["model"]["dropout"] = 0.0
        check_model = ContinuousListModel(check_cfg).to(device).eval()
        h, y = catalog.y[history[:3]], catalog.y[targets[:3]]
        tau = torch.full((3, 5), 100, device=device, dtype=torch.long)
        noisy = torch.randn_like(y)
        with attention_backend(cfg["runtime"]["sdpa_backend"]):
            optimized = check_model.training_prediction(h, history[:3].ne(0), y, noisy, tau)
            mem = check_model.encode_history(h, history[:3].ne(0))
            reference = torch.stack([check_model.predict(
                noisy[:, j], tau[:, j], y[:, :j], j, mem, history[:3].ne(0)) for j in range(5)], 1)
        torch.testing.assert_close(optimized, reference, atol=2e-5, rtol=2e-4)
        del check_model, optimized, reference, mem

        # Actual optimizer step with the locked BF16/fused settings on CUDA.
        trainer.update(history, targets)
        saved = trainer.state()
        expected = trainer.update(history, targets)
        expected_weights = cpu_copy(trainer.model.state_dict())
        trainer.restore(saved)
        actual = trainer.update(history, targets)
        torch.testing.assert_close(actual["loss"], expected["loss"], atol=0, rtol=0)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected_weights[key], atol=0, rtol=0)
        catalog.verify_frozen()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(args.updates):
            result = trainer.update(history, targets)
        synchronize(device)
        elapsed = time.perf_counter()-started
        training = {"updates": args.updates, "lists_per_update": b,
                    "seconds": elapsed, "lists_per_second": b*args.updates/elapsed,
                    "seconds_per_update": elapsed/args.updates,
                    "loss": float(result["loss"]),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        attention_kernels = None
        if device.type == "cuda":
            # Outside the timing interval: disclose what auto dispatch actually
            # selected instead of promising FlashAttention from the API name.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as prof:
                trainer.update(history, targets)
                synchronize(device)
            attention_kernels = [{"operator": event.key, "calls": event.count}
                                 for event in prof.key_averages()
                                 if any(word in event.key.lower() for word in ("attention", "adam", "gemm"))]
        # A short actual RQ gradient step checks FP64 Sinkhorn on the device.
        rq = RQVAE(cfg).to(device)
        content = torch.nn.functional.normalize(torch.randn(cfg["rq"]["batch_size"], cfg["rq"]["input_dim"],
                                                            device=device), dim=-1)
        rq.initialize(content, torch.Generator(device=device).manual_seed(derive_seed(2024, "preflight-rq")))
        rq.train()
        rq_loss = rq(content)["loss"]
        rq_loss.backward()
        torch.nn.utils.clip_grad_norm_(rq.parameters(), 1.0, error_if_nonfinite=True)
        del rq
        trainer.model.eval()
        synchronize(device)
        started = time.perf_counter()
        with attention_backend(cfg["runtime"]["sdpa_backend"]):
            prediction, diag = generate(trainer.model, trainer.diffusion, catalog, history[:2],
                                        ["synthetic-0", "synthetic-1"], "validation", cfg)
        synchronize(device)
        inference_seconds = time.perf_counter()-started
        if not diag["complete"].all():
            raise AssertionError("Incomplete synthetic generation")
        receipt = {
            "status": "passed", "synthetic_only": True, "runtime": runtime,
            "source": source_binding(), "config": cfg,
            "checks": {"batched_reference_fp32": True, "real_optimizer_update": True,
                       "exact_next_update_resume": True, "frozen_asset": True,
                       "rq_gradient_sinkhorn": True, "generation": True},
            "training": training, "generation_seconds_batch2": inference_seconds,
            "attention_and_optimizer_operators": attention_kernels,
            "predictions": prediction.cpu().tolist(),
            "limitations": [f"Synthetic catalog has {len(catalog.y)-1} items; not end-to-end real-data throughput",
                            "No formal experiment or resource allocation performed by this command"],
        }
        write_json(root/"PREFLIGHT.json", receipt)
        print({"output": str(root/"PREFLIGHT.json"), "runtime": runtime, "training": training}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--device", required=True)
    p.add_argument("--require-h100", action="store_true")
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
