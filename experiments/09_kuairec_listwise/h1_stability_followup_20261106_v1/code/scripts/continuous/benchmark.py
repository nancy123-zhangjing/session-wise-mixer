"""Exclusive-window GPU inference benchmark; never run alongside the queue."""
import statistics
import time

import numpy as np
import torch

from .trainer import autocast
from .metrics import stable_top_items


@torch.inference_mode()
def benchmark(model, validation, precision="bf16", warmup=50, repeats=200):
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("Formal latency benchmark requires GPU")
    model.eval()
    device = next(model.parameters()).device
    results = []
    for batch_size in (1, 128):
        inputs_cpu = validation["histories"][:batch_size].cpu()
        inputs = inputs_cpu.to(device)
        with autocast(device, precision):
            query = model(inputs)

        def forward():
            with autocast(device, precision):
                return model(inputs)

        def retrieve():
            with autocast(device, precision):
                return stable_top_items(model.score_catalog(query))

        def end_to_end():
            copied = inputs_cpu.to(device)
            with autocast(device, precision):
                return stable_top_items(model.score_catalog(model(copied))).cpu()

        for name, function in (("query_forward", forward), ("catalog_top20", retrieve),
                               ("end_to_end", end_to_end)):
            for _ in range(warmup):
                function()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                if name == "end_to_end":
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    function()
                    torch.cuda.synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000)
                else:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    function()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
            results.append({"batch_size": batch_size, "component": name,
                "warmup": warmup, "repeats": repeats, "milliseconds": samples,
                "p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
                "throughput_items_per_second": batch_size * 1000 / statistics.mean(samples),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    from .model import CUDA_ATTENTION_POLICY
    return {"gpu": torch.cuda.get_device_name(device), "precision": precision,
            "attention_precision_policy": CUDA_ATTENTION_POLICY,
            "parameters": sum(p.numel() for p in model.parameters()), "measurements": results,
            "scope": {"query_forward": "GPU input tensor to query; CUDA-event timing",
                      "catalog_top20": "resident query to full-catalog scores and top20; CUDA-event timing",
                      "end_to_end": "prepared pageable CPU tensor H2D, query, full-catalog top20, D2H; wall clock",
                      "excluded": "JSON parsing and CPU sequence preprocessing; model loading",
                      "tie_policy": "Exact score-desc, item-id-asc integer-key Top-K, identical to evaluation"}}
