"""Streaming true-item evaluation; no beam scores or test-driven selection."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import time

import torch

from scripts.listwise_diffgrm.metrics import ranking_metrics
from .catalog import generate, evaluation_noise
from .common import preserve_rng, attention_backend, write_json


@torch.inference_mode()
def evaluate(model, diffusion, catalog, data, cfg, output):
    output = Path(output)
    # Commit a receipt only after complete prediction output; a partial file is
    # never a valid resumable evaluation.
    if (output / "summary.json").exists():
        raise FileExistsError("Evaluation already complete; reuse its receipt explicitly")
    output.mkdir(parents=True, exist_ok=True)
    started, totals, diagnostics = time.perf_counter(), {}, {}
    device = catalog.y.device
    old_mode = model.training
    prediction_tmp = output / "predictions.jsonl.gz.partial"
    predictions = output / "predictions.jsonl.gz"
    try:
        model.eval()
        with preserve_rng(), attention_backend(cfg["runtime"]["sdpa_backend"]), gzip.open(
                prediction_tmp, "wt", encoding="utf-8") as stream:
            for start in range(0, len(data), cfg["evaluation"]["batch_size"]):
                indices = torch.arange(start, min(start+cfg["evaluation"]["batch_size"], len(data)))
                history, target = data.batch(indices, device)
                ids = data.sample_ids[start:start+len(indices)]
                generated, diag = generate(model, diffusion, catalog, history, ids, data.split, cfg)
                if not torch.isfinite(generated.float()).all():
                    raise RuntimeError("Invalid generated values")
                metrics = ranking_metrics(generated, target, ks=(cfg["list_length"],),
                                          catalog_item_ids=catalog.item_ids)
                m_cpu = {k: v.cpu().tolist() for k, v in metrics.items()}
                d_cpu = {k: v.cpu().tolist() for k, v in diag.items()}
                pred, truth = generated.cpu().tolist(), target.cpu().tolist()
                for k, values in m_cpu.items():
                    totals[k] = totals.get(k, 0.0)+sum(values)
                for k in ("raw_valid", "raw_available", "changed"):
                    diagnostics[k] = diagnostics.get(k, 0)+sum(sum(row) for row in d_cpu[k])
                for i, sample in enumerate(ids):
                    row = {"sample_id": sample, "user_id": data.users[start+i],
                           "first_target_timestamp_us": int(data.first_target_time[start+i]),
                           "prediction": pred[i], "target": truth[i],
                           "metrics": {k: v[i] for k, v in m_cpu.items()},
                           "diagnostics": {k: v[i] for k, v in d_cpu.items()}}
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_tmp.replace(predictions)
        result = {"count": len(data), "metrics": {k: v/len(data) for k, v in totals.items()},
                  "diagnostics": {k: v/(len(data)*cfg["list_length"]) for k, v in diagnostics.items()},
                  "seconds": time.perf_counter()-started, "candidate_lists": 1, "ranking": "generation_order"}
        write_json(output/"summary.json", result)
        return result
    finally:
        model.train(old_mode)


@torch.inference_mode()
def probe(model, diffusion, catalog, data, cfg, indices):
    """Teacher-forced subset MSE; cannot select a checkpoint."""
    old_mode = model.training
    model.eval()
    results = {}
    device, length, dim = catalog.y.device, cfg["list_length"], catalog.y.shape[-1]
    try:
        with preserve_rng(), attention_backend(cfg["runtime"]["sdpa_backend"]):
            for t in cfg["training"]["probe_times"]:
                total = torch.zeros(length, device=device, dtype=torch.float64)
                for chunk in indices.split(cfg["evaluation"]["batch_size"]):
                    history_ids, target_ids = data.batch(chunk, device)
                    ids = [data.sample_ids[i] for i in chunk.tolist()]
                    epsilon = torch.stack([
                        evaluation_noise(cfg, ids, data.split, j, dim, tag=f"probe:{t}")
                        for j in range(length)], 1).to(device)
                    times = torch.full((len(chunk), length), t, device=device, dtype=torch.long)
                    targets = catalog.y[target_ids]
                    predicted = model.training_prediction(
                        catalog.y[history_ids], history_ids.ne(0), targets,
                        diffusion.noise(targets, times, epsilon), times)
                    total += (predicted.float()-targets).square().mean(-1).sum(0).double()
                values = (total/len(indices)).cpu().tolist()
                results[str(t)] = {"mse": sum(values)/length, "position_mse": values}
        return {"count": len(indices), "teacher_forced": True, "selection_eligible": False, "times": results}
    finally:
        model.train(old_mode)
