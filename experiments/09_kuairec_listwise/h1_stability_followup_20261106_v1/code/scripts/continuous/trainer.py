"""Single-device training, full-catalog evaluation and epoch-boundary resume."""
import contextlib
import csv
import io
import json
import os
import random
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import sha256
from .metrics import metrics_from_ranks, target_ranks, stable_top_items
from .model import ContinuousRecommender, ModelConfig, CUDA_ATTENTION_POLICY
from .protocol import atomic_json, canonical_hash
from .checkpoints import atomic_checkpoint, load_checkpoint, commit_epoch, repair_committed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def model_config(config):
    return ModelConfig(**{f.name: config[f.name] for f in fields(ModelConfig) if f.name in config})


def autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def on_device(data, device):
    return {**data, "histories": data["histories"].to(device),
            "targets": data["targets"].to(device)}


@torch.inference_mode()
def evaluate(model, data, batch_size=512, precision="bf16", predictions=False):
    previous = model.training
    model.eval()
    device = next(model.parameters()).device
    data = on_device(data, device)
    all_ranks, all_top, all_nll, nll = [], [], [], 0.0
    try:
        for start in range(0, len(data["targets"]), batch_size):
            stop = start + batch_size
            targets = data["targets"][start:stop]
            with autocast(device, precision):
                queries = model(data["histories"][start:stop])
                scores = model.score_catalog(queries).float()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite catalog scores")
            row_nll = F.cross_entropy(scores, targets - 1, reduction="none")
            nll += row_nll.sum().item()
            all_ranks.append(target_ranks(scores, targets).cpu())
            if predictions:
                all_nll.extend(row_nll.cpu().tolist())
                all_top.append(stable_top_items(scores).cpu())
        ranks = torch.cat(all_ranks)
        result = metrics_from_ranks(ranks)
        result["NLL"] = nll / len(ranks)
        if predictions:
            result["ranks"] = ranks.tolist()
            result["top_items"] = torch.cat(all_top).tolist()
            result["nlls"] = all_nll
        return result
    finally:
        model.train(previous)


def source_identity(root):
    root = Path(root)
    sources = sorted((root / "scripts/continuous").glob("*.py"))
    sources.extend(sorted((root / "scripts/train").glob("run_continuous*.py")))
    sources.extend(sorted((root / "scripts/checks").glob("*continuous*.py")))
    sources.extend(sorted((root / "tests").glob("test_continuous*.py")))
    for relative in (
                     "configs/06_continuous_baseline/search_manifest.json",
                     "experiments/06_continuous_baseline/upstream_sources.json"):
        if (root / relative).is_file():
            sources.append(root / relative)
    result = {"files": {p.relative_to(root).as_posix(): sha256(p) for p in sources}}
    try:
        result["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = None
    result["source_sha256"] = canonical_hash(result["files"])
    return result


def write_curves(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    os.replace(temporary, path)


def fit(config, seed, protocol, train_data, validation_data, output, *,
        device="cpu", precision="float32", resume=False, stop_after_epochs=None,
        max_steps=None, smoke=False, root=None):
    """No test data is loaded or accepted by training. Partial runs never COMPLETE.

    Resume is exact at the last saved epoch boundary, including optimizer/RNG.
    An interruption inside an epoch repeats only that unfinished epoch.
    """
    output, device = Path(output), torch.device(device)
    torch.set_num_threads(config.get("cpu_threads", 2))
    if (stop_after_epochs is not None or max_steps is not None) and not smoke:
        raise ValueError("Partial limits require explicit smoke=True")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requested but unsupported")
    source = source_identity(root) if root else {"source_sha256": "unit-test"}
    binding = {"config_sha256": canonical_hash(config), "seed": seed, "protocol": protocol,
               "train_sha256": train_data["sha256"], "validation_sha256": validation_data["sha256"],
               "source_sha256": source["source_sha256"], "precision": precision,
               "smoke": smoke}
    if resume:
        if (output / "COMPLETE.json").exists():
            raise FileExistsError("Refusing to resume a complete run")
        old = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if old["binding"] != binding:
            raise ValueError("Resume config/data/source binding mismatch")
        event = {"time": time.time(), "pid": os.getpid(), "device": str(device),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "gpu_uuid": os.environ.get("CONTINUOUS_GPU_UUID")}
        with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run.json", {"binding": binding, "config": config,
                    "source": source, "pid": os.getpid(), "device": str(device),
                    "attention_precision_policy": CUDA_ATTENTION_POLICY if device.type == "cuda" else "native",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "torch_version": torch.__version__, "numpy_version": np.__version__,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "gpu_uuid": os.environ.get("CONTINUOUS_GPU_UUID")})
    seed_everything(seed)
    model = ContinuousRecommender(model_config(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 betas=tuple(config.get("adam_betas", (0.9, 0.98))),
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    train_data, validation_data = on_device(train_data, device), on_device(validation_data, device)
    start_epoch, updates, bad_epochs, best_key, best_epoch, curves = 0, 0, 0, None, None, []
    elapsed_before, best_storage_file = 0.0, None
    if resume:
        repair_committed(output)
        state = load_checkpoint(output / "last.pt", "cpu")
        if state["binding"] != binding:
            raise ValueError("Resume state binding mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_random_state(state["random_state"])
        start_epoch, updates, bad_epochs = state["epoch"], state["updates"], state["bad_epochs"]
        best_key, best_epoch, curves = state["best_key"], state["best_epoch"], state["curves"]
        elapsed_before = state["elapsed_seconds"]
        best_storage_file = state["best_storage_file"]
        write_curves(output / "curves.csv", curves)
    clock = time.perf_counter()
    n_train = len(train_data["targets"])
    finished = start_epoch >= config["max_epochs"] or bad_epochs >= config["patience"]
    epochs = range(start_epoch + 1, config["max_epochs"] + 1) if not finished else ()
    for epoch in epochs:
        epoch_clock = time.perf_counter()
        model.train()
        order = torch.randperm(n_train, device=device)
        loss_sum, seen = 0.0, 0
        for start in range(0, n_train, config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                loss = model.loss(train_data["histories"][batch], train_data["targets"][batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("grad_clip", 1.0), error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            seen += len(batch)
            loss_sum += loss.detach().item() * len(batch)
            if max_steps is not None and updates >= max_steps:
                break
        train_seconds = time.perf_counter() - epoch_clock
        validation_clock = time.perf_counter()
        validation = evaluate(model, validation_data, config.get("eval_batch_size", 512), precision)
        validation_seconds = time.perf_counter() - validation_clock
        key = (validation["NDCG@10"], validation["HR@10"])
        improved = best_key is None or key > tuple(best_key)
        if improved:
            best_key, best_epoch, bad_epochs = key, epoch, 0
        else:
            bad_epochs += 1
        scheduler.step()
        row = {"epoch": epoch, "updates": updates, "train_samples": seen,
               "train_loss": loss_sum / seen, "grad_norm_last": float(grad_norm),
               "train_seconds": train_seconds, "validation_seconds": validation_seconds,
               **{f"validation_{k}": v for k, v in validation.items()},
               "elapsed_seconds": elapsed_before + time.perf_counter() - clock}
        curves.append(row)
        state = {"binding": binding, "config": config, "epoch": epoch, "updates": updates,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "random_state": random_state(),
                 "best_key": best_key, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                 "curves": curves, "elapsed_seconds": row["elapsed_seconds"],
                 "validation_metrics": validation, "best_storage_file": best_storage_file}
        checkpoint_clock = time.perf_counter()
        best_storage_file = commit_epoch(output, state, improved)
        write_curves(output / "curves.csv", curves)
        performance = {"epoch": epoch, "updates": updates, "train_seconds": train_seconds,
                       "validation_seconds": validation_seconds,
                       "checkpoint_seconds": time.perf_counter() - checkpoint_clock}
        with (output / "performance.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(performance) + "\n")
        print(json.dumps({"event": "epoch", **row}), flush=True)
        finished = epoch == config["max_epochs"] or bad_epochs >= config["patience"]
        if finished or (stop_after_epochs is not None and epoch >= stop_after_epochs):
            break
        if max_steps is not None and updates >= max_steps:
            break
    result = {"binding": binding, "epochs": len(curves), "updates": updates,
              "best_epoch": best_epoch, "best_validation": json.loads(
                  (output / "best_validation.json").read_text(encoding="utf-8"))["metrics"],
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "elapsed_seconds": elapsed_before + time.perf_counter() - clock,
              "best_sha256": sha256(output / "best.pt"),
              "last_sha256": sha256(output / "last.pt"),
              "train_samples_per_full_epoch": n_train,
              "validation_samples": len(validation_data["targets"]), "smoke": smoke}
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    result["status"] = "SMOKE_COMPLETE" if smoke else ("SUCCEEDED" if finished else "INTERRUPTED")
    marker = "SMOKE.json" if smoke else ("COMPLETE.json" if finished else "INTERRUPTED.json")
    atomic_json(output / marker, result)
    return result


def restore_model(checkpoint, device="cpu"):
    # Optimizer/RNG snapshots stay on CPU during inference and benchmarking.
    state = load_checkpoint(checkpoint, "cpu")
    model = ContinuousRecommender(model_config(state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state
