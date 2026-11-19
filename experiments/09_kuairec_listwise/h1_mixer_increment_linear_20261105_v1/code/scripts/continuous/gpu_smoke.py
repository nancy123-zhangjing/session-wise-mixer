"""Bounded GPU gate. Called only inside an explicitly allocated eight-H100 Job."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .data import sha256
from .protocol import COMMON, atomic_json, canonical_hash, generate_search
from .trainer import load_checkpoint, restore_model, source_identity


def check_training_gradients(model, histories, targets):
    """Fixed negatives/p=0 isolate AMP error from backend dropout RNG changes.

    This disposable model is also checked in training mode, since an inference
    cosine can remain excellent while the attention backward is incorrect.
    """
    import torch
    from .trainer import autocast
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    negatives = torch.randint(1, model.config.n_items + 1,
                              (len(targets), model.config.negatives), device=targets.device)
    device = next(model.parameters()).device
    reference, losses = {}, []
    for precision in ("float32", "bf16"):
        model.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = model.loss(histories, targets, negative_ids=negatives)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite calibration loss")
        loss.backward()
        losses.append(loss.detach().item())
        if precision == "float32":
            reference = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                         if p.grad is not None}
    square_error = square_reference = square_actual = dot = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        actual, expected = parameter.grad.detach().double(), reference[name].double()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise FloatingPointError("Nonfinite calibration gradient")
        square_error += (actual - expected).square().sum().item()
        square_reference += expected.square().sum().item()
        square_actual += actual.square().sum().item()
        dot += (actual * expected).sum().item()
    relative_l2 = (square_error / max(square_reference, 1e-30)) ** 0.5
    cosine = dot / max((square_reference * square_actual) ** 0.5, 1e-30)
    if relative_l2 >= 0.05 or cosine <= 0.999:
        raise FloatingPointError("BF16 training gradients disagree with FP32 reference")
    model.zero_grad(set_to_none=True)
    return {"model": model.config.model, "dim": model.config.dim, "heads": model.config.heads,
            "training_mode": True, "dropout_disabled": True, "fixed_negatives": True,
            "relative_gradient_l2": relative_l2, "gradient_cosine": cosine,
            "relative_l2_max": 0.05, "cosine_min": 0.999,
            "fp32_loss": losses[0], "bf16_loss": losses[1]}


def run_numerical_stress(cache, directory, config, gpu_uuid, max_seconds=360):
    """Regression for the original epoch-28 failure; no validation/selection.

    Same fixed search configuration and 2024 seed, 35 complete training epochs,
    no checkpoint writes. Separate from, and never counted among, the 89 runs.
    """
    import math
    import torch
    from .data import load_cached_split
    from .model import ContinuousRecommender, CUDA_ATTENTION_POLICY
    from .trainer import autocast, model_config, seed_everything
    torch.set_num_threads(2)
    seed_everything(2024)
    started = time.monotonic()
    data = load_cached_split(cache, "full_train_prefix_v1", "train")
    histories, targets = data["histories"].cuda(), data["targets"].cuda()
    model = ContinuousRecommender(model_config(config)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 betas=tuple(config["adam_betas"]), weight_decay=config["weight_decay"])
    rows = []
    for epoch in range(1, 36):
        model.train()
        order = torch.randperm(len(targets), device="cuda:0")
        maximum_norm = loss_sum = 0.0
        updates = seen = 0
        for start in range(0, len(targets), config["batch_size"]):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Numerical regression exceeded bounded smoke time")
            batch = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            with autocast(torch.device("cuda:0"), config["precision"]):
                loss = model.loss(histories[batch], targets[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite numerical regression loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                                  error_if_nonfinite=True)
            if norm.item() >= 1e6:
                raise FloatingPointError("Numerical regression gradient exploded")
            optimizer.step()
            maximum_norm = max(maximum_norm, norm.item())
            loss_sum += loss.detach().item() * len(batch)
            seen += len(batch)
            updates += 1
        rows.append({"epoch": epoch, "updates": updates, "samples": seen,
                     "train_loss": loss_sum / seen, "maximum_gradient_norm": maximum_norm})
        atomic_json(directory / "NUMERICAL_STRESS_PROGRESS.json", {"epochs": rows})
        print(json.dumps({"numerical_stress_epoch": epoch, "maximum_gradient_norm": maximum_norm}), flush=True)
    if any(r["updates"] != math.ceil(len(targets) / config["batch_size"]) for r in rows):
        raise ValueError("Incomplete numerical regression epoch")
    result = {"status": "PASS", "not_formal_training": True, "validation_or_test_used": False,
              "config": config, "seed": 2024, "gpu_uuid": gpu_uuid,
              "attention_precision_policy": CUDA_ATTENTION_POLICY, "epochs": rows,
              "max_seconds": max_seconds, "elapsed_seconds": time.monotonic() - started,
              "gradient_norm_sanity_limit": 1e6}
    atomic_json(directory / "NUMERICAL_STRESS.json", result)
    del model, optimizer, histories, targets
    torch.cuda.empty_cache()
    return result


def validate_cpu_gate(path, cache, source_sha256):
    proof = json.loads(Path(path).read_text(encoding="utf-8"))
    if (proof.get("status") != "PASS" or proof.get("skipped") != 0
            or proof.get("source_sha256") != source_sha256
            or proof.get("cache_manifest_sha256") != sha256(Path(cache) / "cache.json")):
        raise ValueError("CPU gate, source or verified cache identity differs")
    return proof


def gpu_inventory():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
         "--format=csv,noheader,nounits"], text=True)
    rows = [dict(zip(("index", "uuid", "name", "memory_mib", "driver_version"),
                     [value.strip() for value in line.split(",")]))
            for line in text.strip().splitlines()]
    if (len(rows) != 8 or len({r["uuid"] for r in rows}) != 8
            or any("H100" not in row["name"] for row in rows)):
        raise ValueError("GPU smoke requires exactly eight distinct H100 GPUs")
    return rows


def bounded_workers(root, directory, cache, specs, inventory, timeout=600):
    """At most one process per UUID; refill immediately and retain PID evidence."""
    pending, running, completed, dispatches = list(specs), {}, [], []
    started = time.monotonic()
    try:
        while pending or running:
            for gpu, record in list(running.items()):
                if record["process"].poll() is None:
                    continue
                record["log"].close()
                if record["process"].returncode:
                    raise RuntimeError(f"GPU smoke worker failed: {record['output']}")
                result = json.loads((record["output"] / "SMOKE.json").read_text(encoding="utf-8"))
                run = json.loads((record["output"] / "run.json").read_text(encoding="utf-8"))
                if (result["status"] != "SMOKE_COMPLETE" or result["validation_samples"] != 22363
                        or run["gpu_uuid"] != inventory[gpu]["uuid"]):
                    raise ValueError("Smoke validation/device binding failed")
                completed.append({"name": record["name"], "gpu_slot": gpu,
                    "gpu_uuid": inventory[gpu]["uuid"], "output": str(record["output"]),
                    "result": result})
                del running[gpu]
            for gpu in range(8):
                if gpu in running:
                    continue
                index = next((i for i, spec in enumerate(pending) if spec.get("gpu", gpu) == gpu), None)
                if index is None:
                    continue
                spec = pending.pop(index)
                task_path = directory / "tasks" / (spec["name"] + ".json")
                atomic_json(task_path, spec["task"])
                output = directory / spec["name"]
                command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"),
                           "train", "--task", str(task_path), "--cache", str(cache),
                           "--output", str(output), "--smoke", "--max-steps", str(spec["steps"])]
                if spec.get("resume"):
                    command.append("--resume")
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": inventory[gpu]["uuid"],
                    "CONTINUOUS_GPU_UUID": inventory[gpu]["uuid"], "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                log = (directory / (spec["name"] + ("_resume" if spec.get("resume") else "") + ".log")).open("xb")
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = {"process": process, "log": log, "output": output, "name": spec["name"]}
                dispatches.append({"name": spec["name"], "pid": process.pid, "gpu_slot": gpu,
                                   "gpu_uuid": inventory[gpu]["uuid"], "monotonic": time.monotonic(),
                                   "concurrent_worker_count": len(running)})
            if time.monotonic() - started > timeout:
                raise TimeoutError("Bounded smoke exceeded time budget")
            time.sleep(1)
        return completed, dispatches
    finally:
        for record in running.values():
            if record["process"].poll() is None:
                record["process"].terminate()
            try:
                record["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                record["process"].kill()
                record["process"].wait()
            record["log"].close()


def run_gpu_gate(root, directory, cache, cpu_gate):
    root, directory, cache = Path(root).resolve(), Path(directory).resolve(), Path(cache).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity(root)
    cpu = validate_cpu_gate(cpu_gate, cache, source["source_sha256"])
    inventory = gpu_inventory()
    search = generate_search()["trials"]
    anchors = [next(t for t in search if t["config"]["model"] == model)
               for model in ("sasrec_fullce", "esasrec")]
    first = [{"name": "base_" + task["config"]["model"], "gpu": index,
              "task": task, "steps": 20} for index, task in enumerate(anchors)]
    base, first_dispatch = bounded_workers(root, directory, cache, first, inventory)
    before_resume = {}
    for record in base:
        state = load_checkpoint(Path(record["output"]) / "last.pt")
        before_resume[record["name"]] = {"updates": state["updates"],
            "optimizer_steps": [float(v["step"]) for v in state["optimizer"]["state"].values()]}
    resumed, resume_dispatch = bounded_workers(root, directory, cache,
        [{**spec, "steps": 21, "resume": True} for spec in first], inventory)
    resume_checks = []
    for record in resumed:
        state = load_checkpoint(Path(record["output"]) / "last.pt")
        original = before_resume[record["name"]]
        steps = [float(v["step"]) for v in state["optimizer"]["state"].values()]
        if state["updates"] != original["updates"] + 1 or any(
                after != before + 1 for before, after in zip(original["optimizer_steps"], steps)):
            raise ValueError("Optimizer state was not actually resumed")
        resume_checks.append({"name": record["name"], "updates_before": original["updates"],
                              "updates_after": state["updates"], "optimizer_steps_restored": True})
    parallel = []
    for index in range(10):
        task = json.loads(json.dumps(anchors[0 if index % 8 < 4 else 1]))
        if index in (0, 4):
            task["config"].update(dim=512, layers=4, heads=4, batch_size=256, dropout=0.5)
            if task["config"]["model"] == "esasrec":
                task["config"]["negatives"] = 1024
            task["config_sha256"] = canonical_hash(task["config"])
        parallel.append({"name": f"parallel_{index:02d}", "task": task, "steps": 2})
    parallel_results, parallel_dispatch = bounded_workers(root, directory, cache, parallel, inventory)
    if {r["gpu_slot"] for r in parallel_results} != set(range(8)):
        raise ValueError("Not all eight GPU slots were tested")
    if max(r["concurrent_worker_count"] for r in parallel_dispatch) != 8 or len(parallel_dispatch) != 10:
        raise ValueError("Eight-worker refill gate incomplete")
    precision_checks = []
    import torch
    from .data import load_cached_split
    from .trainer import autocast
    from .metrics import stable_top_items
    calibration = load_cached_split(cache, "full_train_prefix_v1", "validation")
    histories = calibration["histories"][:128]
    gradient_checks = []
    for record in resumed:
        model, state = restore_model(Path(record["output"]) / "last.pt", "cuda:0")
        model.eval()
        inputs = histories.to("cuda:0")
        with torch.inference_mode():
            reference = model(inputs).float()
            with autocast(torch.device("cuda:0"), "bf16"):
                approximate = model(inputs).float()
                scores = model.score_catalog(approximate).float()
            cosine = torch.nn.functional.cosine_similarity(reference, approximate, dim=1)
            expected_top = scores.argsort(dim=1, descending=True, stable=True)[:, :20] + 1
            if not torch.equal(stable_top_items(scores), expected_top):
                raise ValueError("GPU deterministic Top-K differs from full stable sort")
        if not torch.isfinite(approximate).all() or cosine.min().item() < 0.98:
            raise ValueError("BF16 numerical check failed")
        precision_checks.append({"model": state["config"]["model"],
                                 "minimum_query_cosine_fp32_bf16": cosine.min().item(),
                                 "gpu_topk_matches_full_stable_sort": True})
        gradient_checks.append(check_training_gradients(model, inputs, calibration["targets"][:128].cuda()))
        del model
        torch.cuda.empty_cache()
    # Include the large per-head dimension which the original short gate missed.
    from .model import ContinuousRecommender
    from .trainer import model_config, seed_everything
    for model_name in ("sasrec_fullce", "esasrec"):
        seed_everything(61204)
        config = {**anchors[0]["config"], "model": model_name, "dim": 512, "heads": 2}
        model = ContinuousRecommender(model_config(config)).cuda()
        gradient_checks.append(check_training_gradients(model, inputs, calibration["targets"][:128].cuda()))
        del model
        torch.cuda.empty_cache()
    critical = next(t for t in search if t["trial_id"] == "esasrec_17")
    stress = run_numerical_stress(cache, directory, critical["config"], inventory[0]["uuid"])
    if source_identity(root)["source_sha256"] != source["source_sha256"]:
        raise ValueError("Source changed during GPU gate")
    proof = {"status": "PASS", "source_sha256": source["source_sha256"],
             "cache_manifest_sha256": cpu["cache_manifest_sha256"],
             "cpu_gate_sha256": sha256(cpu_gate), "inventory": inventory,
             "base": base, "resumed": resumed, "resume_checks": resume_checks,
             "parallel": parallel_results, "precision_checks": precision_checks,
             "gradient_checks": gradient_checks, "numerical_stress": stress,
             "dispatches": first_dispatch + resume_dispatch + parallel_dispatch,
             "smoke_not_formal_training": True}
    atomic_json(directory / "GPU_GATE.json", proof)
    return proof
