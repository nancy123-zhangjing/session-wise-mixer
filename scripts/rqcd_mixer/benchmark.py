"""One explicitly invoked CUDA benchmark; never invoked by train or import."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from .catalog import Catalog, generate, evaluation_noise
from .common import (output_lock, write_json, atomic_save, read_state, tensor_digest, autocast,
                     attention_backend, seed_all, seed_for, sha256)
from .config import ARMS
from .data import PackedSplit
from .losses import recommendation_losses
from .model import Recommender, Diffusion
from .tokenizer import RQTokenizer
from .training import JointTrainer


def relative_error(actual, expected):
    return float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8))


def numerical_check(cfg, runtime, initial, history, target, device):
    """Small same-weight FP32/BF16 prediction/loss/gradient and cache comparison."""
    model = Recommender(cfg).to(device).eval()
    model.load_state_dict(initial["recommender"])
    catalog = Catalog(initial["benchmark_asset"], device)
    diffusion = Diffusion(cfg).to(device)
    count = min(4, len(history))
    history, target = history[:count], target[:count]
    times = torch.full((count, cfg["list_length"]), cfg["diffusion"]["steps"]//2, device=device, dtype=torch.long)
    eps = torch.from_numpy(evaluation_noise(cfg, [f"benchmark:{i}" for i in range(count)], "benchmark")).to(device)
    clean = catalog.x_pad[target]
    noisy = diffusion.noise(clean, times, eps)
    results = []
    for precision in ("fp32", runtime["precision"]):
        model.zero_grad(set_to_none=True)
        with attention_backend(runtime["attention_backend"]), autocast({**runtime, "precision": precision}, device):
            prediction = model.training_prediction(model.lookup(catalog.sid_pad[history]), history != 0,
                                                   model.lookup(catalog.sid_pad[target]), noisy, times)
        loss = recommendation_losses(prediction, clean, catalog.z_pad[target], catalog.sid_pad[target],
                                     catalog.books, catalog.stats, cfg, catalog=catalog, target_ids=target)["total"]
        loss.backward()
        gradients = torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])
        results.append((prediction.detach().float(), loss.detach(), gradients))
    errors = dict(zip(("prediction", "loss", "gradient"),
                      [relative_error(a, b) for a, b in zip(results[1], results[0])]))
    with torch.inference_mode(), attention_backend(runtime["attention_backend"]), autocast(runtime, device):
        memory, valid = model.encode_history(model.lookup(catalog.sid_pad[history]), history != 0)
        cache = model.start_cache(memory, valid)
        cached_errors = []
        for j in range(cfg["list_length"]):
            prefix = model.lookup(catalog.sid_pad[target[:, :j]])
            full = model.predict_uncached(noisy[:, j], times[:, j], prefix, memory, valid)
            cached = model.predict_cached(noisy[:, j], times[:, j], cache)
            cached_errors.append(relative_error(cached, full))
            if j < cfg["list_length"]-1:
                model.append_prefix(model.lookup(catalog.sid_pad[target[:, j]]), cache)
    limits = {"prediction": 0.05, "loss": 0.03, "gradient": 0.10, "cache": 0.05}
    errors["cache"] = max(cached_errors)
    return {"relative_l2_errors": errors, "limits": limits,
            "passed": all(errors[k] <= limits[k] for k in limits),
            "scope": "small fixed batch; eval/dropout off; not training bitwise equivalence"}


def resume_check(cfg, runtime, initial, data, device, output):
    """One real R update after a saved state, including CUDA dropout/fused AdamW."""
    def create():
        token = RQTokenizer(cfg, "causal").to(device)
        token.load_state_dict(initial["tokenizers"]["B2"])
        return JointTrainer(cfg, runtime, token, initial["benchmark_asset"], initial["recommender"],
                            {"benchmark_resume_only": True}, device)
    trainer = create()
    seed_all(seed_for(cfg["seed"], "benchmark_resume"))
    trainer.advance(data)
    atomic_save(output/"resume_probe.pt", trainer.state())
    first = trainer.advance(data)
    expected = trainer.state()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = create()
    resumed.restore(read_state(output/"resume_probe.pt"))
    second = resumed.advance(data)
    actual = resumed.state()
    same = (first == second and expected["r_updates"] == actual["r_updates"] and
            tensor_digest(expected["model"]) == tensor_digest(actual["model"]) and
            tensor_digest(expected["tokenizer"]) == tensor_digest(actual["tokenizer"]) and
            torch.equal(expected["r_stream"]["order"], actual["r_stream"]["order"]) and
            expected["r_stream"]["cursor"] == actual["r_stream"]["cursor"] and
            all(torch.equal(v, actual["generators"][k]) for k, v in expected["generators"].items()) and
            all(torch.equal(v, w) for v, w in zip(expected["rng"]["cuda"], actual["rng"]["cuda"])))
    for opt in ("optimizer_r", "optimizer_t"):
        for key, state in expected[opt]["state"].items():
            for name, value in state.items():
                other = actual[opt]["state"][key][name]
                same = same and (torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
    return {"passed": bool(same), "scope": "one next R update; weights, optimizer, stream and CUDA RNG",
            "checkpoint_sha256": sha256(output/"resume_probe.pt")}


def benchmark(args, cfg):
    from .run import _runtime, _prepared
    runtime, environment = _runtime(args, cfg, require_batches=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The execution benchmark requires CUDA; CPU unit tests are separate")
    if args.repeats < 1 or any(m < 1 or m > cfg["training"]["effective_batch"] for m in args.microbatches):
        raise ValueError("Invalid benchmark candidates/repetitions")
    prepared, meta, initial = _prepared(args, cfg)
    data = PackedSplit(args.dataset, "train", cfg, device=device if runtime["data_device"] == "cuda" else "cpu")
    start = time.perf_counter()
    history, target = data.batch(torch.arange(cfg["training"]["effective_batch"]), device)
    torch.cuda.synchronize(device)
    gather_seconds = time.perf_counter()-start
    candidates = sorted(set(args.microbatches))
    rows, generation_rows, profiles = [], [], []
    with output_lock(args.output) as root:
        numeric = numerical_check(cfg, runtime, initial, history, target, device)
        write_json(root/"numerical_check.json", numeric)
        for phase in ("R", "T"):
            for micro in candidates:
                for arm, mode in ARMS.items():
                    trainer = None
                    try:
                        run = {**runtime, "micro_r": micro, "micro_t": micro, "evaluation_batch": micro}
                        token = RQTokenizer(cfg, mode).to(device)
                        token.load_state_dict(initial["tokenizers"][arm])
                        seed_all(seed_for(cfg["seed"], "benchmark_dropout"))
                        trainer = JointTrainer(cfg, run, token, initial["benchmark_asset"], initial["recommender"],
                                               {"benchmark_only": True}, device)
                        trainer.update(history, target, phase)  # warm-up, disposable state
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start = time.perf_counter()
                        for _ in range(args.repeats):
                            trainer.update(history, target, phase)
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter()-start)/args.repeats
                        rows.append({"arm": arm, "phase": phase, "microbatch": micro, "ok": True,
                                     "effective_update_seconds": elapsed,
                                     "sessions_per_second": len(history)/elapsed,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
                        # One profile across the complete effective update,
                        # including input gather, distances, attention and backward.
                        if arm == "B2" and micro == candidates[0]:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA],
                                                        record_shapes=True) as prof:
                                h, y = data.batch(torch.arange(len(history)), device)
                                trainer.update(h, y, phase)
                            prof.export_chrome_trace(str(root/f"{phase}_trace.json"))
                            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
                            (root/f"{phase}_profile.txt").write_text(table, encoding="utf-8")
                            profiles.append(phase)
                        if phase == "T":
                            torch.cuda.synchronize(device)
                            start = time.perf_counter()
                            trainer.refresh_catalog()
                            torch.cuda.synchronize(device)
                            rows[-1]["catalog_refresh_seconds"] = time.perf_counter()-start
                    except (torch.OutOfMemoryError, RuntimeError) as error:
                        rows.append({"arm": arm, "phase": phase, "microbatch": micro, "ok": False,
                                     "error": str(error)})
                    finally:
                        del trainer
                        gc.collect()
                        torch.cuda.empty_cache()
        # Validation latency matters as much as training. Same complete
        # five-item, 20-step generator; no reduced inference budget.
        model = Recommender(cfg).to(device).eval()
        model.load_state_dict(initial["recommender"])
        diffusion, catalog = Diffusion(cfg).to(device), Catalog(initial["benchmark_asset"], device)
        for batch in candidates:
            try:
                noise = torch.from_numpy(evaluation_noise(cfg, [f"benchmark:{i}" for i in range(batch)], "benchmark")).to(device)
                with torch.inference_mode(), attention_backend(runtime["attention_backend"]):
                    generate(model, diffusion, catalog, history[:batch], noise, runtime)
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    prediction, _ = generate(model, diffusion, catalog, history[:batch], noise, runtime)
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-start
                generation_rows.append({"batch": batch, "ok": True, "seconds": elapsed,
                                        "lists_per_second": batch/elapsed,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                del prediction
            except (torch.OutOfMemoryError, RuntimeError) as error:
                generation_rows.append({"batch": batch, "ok": False, "error": str(error)})
                gc.collect()
                torch.cuda.empty_cache()
        chosen = {}
        for phase, key in (("R", "micro_r"), ("T", "micro_t")):
            eligible = {}
            for micro in candidates:
                matches = [r for r in rows if r["phase"] == phase and r["microbatch"] == micro]
                if len(matches) == len(ARMS) and all(r["ok"] for r in matches):
                    eligible[micro] = sum(r["effective_update_seconds"] for r in matches)
            chosen[key] = min(eligible, key=eligible.get) if eligible else None
        good_eval = [r for r in generation_rows if r["ok"]]
        chosen["evaluation_batch"] = max(good_eval, key=lambda r: r["lists_per_second"])["batch"] if good_eval else None
        ready = numeric["passed"] and all(v is not None for v in chosen.values())
        recovery = {"passed": False, "scope": "not attempted because preceding checks did not pass"}
        if ready:
            # Free generator workspaces before the single recovery probe.
            del model, diffusion, catalog
            gc.collect()
            torch.cuda.empty_cache()
            recovery = resume_check(cfg, {**runtime, **chosen}, initial, data, device, root)
            ready = recovery["passed"]
        result = {"benchmark_only": True, "formal_training_started": False, "environment": environment,
                  "prepared_sha256": sha256(prepared/"PREPARED.json"), "data_gather_seconds": gather_seconds,
                  "measurements": rows, "generation": generation_rows, "numerical": numeric,
                  "profiled_phases": profiles, "selected_common_execution": chosen, "recovery": recovery, "ready": ready,
                  "limitations": "initial shared latent assets; formal trained-state throughput may differ; no accuracy claim"}
        write_json(root/"BENCHMARK.json", result)
        if ready:
            write_json(root/"runtime.locked.json", {**runtime, **chosen})
    return result
