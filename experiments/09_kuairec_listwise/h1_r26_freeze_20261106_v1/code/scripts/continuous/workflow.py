"""One-node GPU lifecycle. CPU delivery is separate so GPUs release promptly."""
import json
import os
import math
import re
import time
from pathlib import Path

from .campaign import run_queue
from .data import sha256
from .delivery import benchmark_checkpoints, export_checkpoints, finalize_delivery, read_json
from .gpu_smoke import gpu_inventory, run_gpu_gate
from .protocol import MODELS, atomic_json, generate_search
from .resources import ResourceMonitor
from .trainer import source_identity


def reconcile_workflow(directory, evidence):
    """Preserve interrupted phase artifacts only after verified terminal Job evidence."""
    from .campaign import INFRASTRUCTURE_REASONS, reconcile_terminal_job
    directory = Path(directory).resolve()
    status = read_json(directory / "PIPELINE.json")
    if (evidence.get("terminal_status") not in {"FAILED", "CANCELLED", "SUCCEEDED"}
            or evidence.get("reason") not in INFRASTRUCTURE_REASONS
            or evidence.get("platform_job_name") != status["job_name"]
            or not evidence.get("platform_job_id") or not evidence.get("raw_platform_observations")
            or not evidence.get("observed_unix")):
        raise ValueError("Matching terminal platform Job observations required")
    ledger_path = directory / "infrastructure_retries.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    if any(entry["evidence"]["platform_job_id"] == evidence["platform_job_id"] for entry in ledger):
        return {"status": "ALREADY_RECONCILED"}
    candidates = [(directory / "smoke", "GPU_GATE.json")]
    candidates += [(directory / "initial_benchmarks" / m, "result.json") for m in MODELS]
    candidates += [(directory / "campaign/final_benchmarks" / m, "result.json") for m in MODELS]
    evaluation_root = directory / "campaign/evaluations"
    if evaluation_root.is_dir():
        candidates += [(p, "result.json") for p in evaluation_root.iterdir()
                       if p.is_dir() and ".interrupted-" not in p.name]
    moves = []
    for path, marker in candidates:
        if not path.exists():
            continue
        complete = False
        try:
            entry = read_json(path / marker)
            complete = bool(entry) and (entry.get("status") == "PASS" if marker == "GPU_GATE.json" else True)
        except (OSError, ValueError):
            pass
        if complete:
            continue
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("Interrupted phase escapes this workflow")
        key = path.relative_to(directory).as_posix()
        attempts = sum(key in entry["archived_paths"] for entry in ledger)
        if attempts >= 2:
            raise RuntimeError("Interrupted phase has already used two infrastructure retries")
        target = path.with_name(path.name + f".interrupted-{attempts+1}")
        if target.exists():
            raise FileExistsError("Interrupted archive destination exists")
        moves.append((path, target, key))
    queue = directory / "campaign/queue.json"
    if queue.is_file():
        reconcile_terminal_job(queue.parent, evidence)
    archived = []
    for source, target, key in moves:
        source.rename(target)
        log = source.with_suffix(".log")
        if log.is_file():
            log.rename(target / "interrupted_worker.log")
        archived.append(key)
    ledger.append({"evidence": evidence, "archived_paths": archived})
    atomic_json(ledger_path, ledger)
    return {"status": "RECONCILED_AFTER_TERMINAL_JOB", "archived_paths": archived}


def budget_projection(smoke, directory, rate):
    rates, maxima = {}, {}
    for record in smoke["base"] + [r for r in smoke["parallel"] if r["name"] in {"parallel_00", "parallel_04"}]:
        path = Path(record["output"]) / "performance.jsonl"
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        config = read_json(Path(record["output"]) / "run.json")["config"]
        destination = rates if record["name"].startswith("base_") else maxima
        destination[config["model"]] = {
            "config": config,
            "train_seconds_per_update": first["train_seconds"] / first["updates"],
            "validation_seconds": first["validation_seconds"],
            "checkpoint_seconds": first["checkpoint_seconds"]}
    epochs = {m: [] for m in MODELS}
    for trial in generate_search()["trials"]:
        config = trial["config"]
        low, high = rates[config["model"]], maxima[config["model"]]
        work = config["batch_size"] * config["layers"] * config["dim"]**2
        low_work = 128 * 64**2
        high_work = 256 * 4 * 512**2
        fraction = max(0.0, min(1.0, (work-low_work)/(high_work-low_work)))
        interpolate = lambda key: low[key] + fraction * (max(low[key], high[key])-low[key])
        seconds = (math.ceil(131413/config["batch_size"]) * interpolate("train_seconds_per_update")
                   + interpolate("validation_seconds") + interpolate("checkpoint_seconds"))
        epochs[config["model"]].append(seconds)
    # The extra selected-config runs are still unknown. Show a range using
    # each family's fastest/slowest measured shape extrapolation and 51..300 epochs.
    lower = sum(sum(values) * 51 for values in epochs.values())
    upper = sum(sum(values) * 300 for values in epochs.values())
    for model, extra in (("sasrec_fullce", 14), ("esasrec", 11)):
        lower += min(epochs[model]) * 51 * extra
        upper += max(epochs[model]) * 300 * extra
    maximum_trial = max(max(values) for values in epochs.values()) * 300
    hours = [lower/8/3600, (upper/8 + maximum_trial)/3600]
    result = {"observed_base_rates": rates, "training_run_budget": 89, "max_epochs_per_run": 300,
              "maximum_concurrent_workers": 8, "points_per_node_hour": rate,
              "observed_maximum_shape_rates": maxima,
              "projected_node_hours_range": hours, "projected_gpu_hours_range": [8*h for h in hours],
              "projected_points_range": [rate*h for h in hours],
              "interpretation": "Sizing estimate from base/max-shape smoke, linear compute-work interpolation, 51..300 epochs, and an upper tail allowance. Not a runtime or convergence guarantee; excludes platform scheduling and final evaluation overhead."}
    atomic_json(directory / "budget_projection.json", result)


def run_gpu_workflow(root, directory, cache, cpu_gate, job_name, points_per_node_hour=8.0):
    root, directory, cache = Path(root).resolve(), Path(directory).resolve(), Path(cache).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name) or not math.isfinite(points_per_node_hour) or points_per_node_hour <= 0:
        raise ValueError("Valid platform job name and positive quota rate required")
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    stream = (directory / "pipeline.lock").open("a+")
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = source_identity(root)
    marker = directory / "GPU_WORK_COMPLETE.json"
    if marker.is_file():
        result = read_json(marker)
        if result["source_sha256"] != source["source_sha256"]:
            raise ValueError("Completed GPU pipeline has a different source identity")
        stream.close()
        return result
    inventory = gpu_inventory()
    status_path = directory / "PIPELINE.json"
    if status_path.exists():
        previous = read_json(status_path)
        if previous["source_sha256"] != source["source_sha256"]:
            raise ValueError("Cannot mix pipeline source versions")
    monitor_root = directory / "resource_attempts"
    monitor_root.mkdir(exist_ok=True)
    monitor = ResourceMonitor(monitor_root / job_name, points_per_node_hour=points_per_node_hour)
    status = {"job_name": job_name, "source_sha256": source["source_sha256"],
              "started_unix": time.time(), "pid": os.getpid(), "cache": str(cache)}

    def phase(name):
        status["phase"], status["updated_unix"] = name, time.time()
        atomic_json(status_path, status)
        with (directory / "pipeline_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status) + "\n")
        print(json.dumps({"event": "phase", "phase": name, "time": time.time()}), flush=True)
        if source_identity(root)["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("Pipeline execution code changed")

    monitor.start()
    try:
        phase("gpu_smoke")
        gate_path = directory / "smoke/GPU_GATE.json"
        if gate_path.is_file():
            smoke = read_json(gate_path)
            if smoke["source_sha256"] != source["source_sha256"] or smoke["status"] != "PASS":
                raise ValueError("Previous GPU gate is stale")
        else:
            smoke = run_gpu_gate(root, directory / "smoke", cache, cpu_gate)
        phase("initial_exclusive_benchmark")
        base_checkpoints = {model: next(Path(r["output"]) / "best.pt" for r in smoke["resumed"]
            if read_json(Path(r["output"]) / "run.json")["config"]["model"] == model) for model in MODELS}
        benchmark_checkpoints(root, directory / "initial_benchmarks", cache, base_checkpoints, inventory[0]["uuid"])
        budget_projection(smoke, directory, points_per_node_hour)
        phase("training_queue")
        campaign = directory / "campaign"
        state = run_queue(root, campaign, cache, gate_path)
        phase("final_test")
        from .delivery import run_evaluations
        evaluations = run_evaluations(root, campaign, cache, inventory)
        phase("final_exclusive_benchmark")
        # Seed fixed before test: no best-test-seed selection for timing/export.
        checkpoints = {model: Path(next(t["output"] for t in state["tasks"]
            if t["phase"] == "formal" and t["seed"] == 3031 and t["config"]["model"] == model)) / "best.pt"
            for model in MODELS}
        benchmark_checkpoints(root, campaign / "final_benchmarks", cache, checkpoints, inventory[0]["uuid"])
        phase("gpu_work_complete")
        result = {"status": "GPU_WORK_COMPLETE", "source_sha256": source["source_sha256"],
                  "training_runs": 89, "test_evaluations": len(evaluations), "job_name": job_name,
                  "selection_sha256": sha256(campaign / "FROZEN_SELECTION.json"),
                  "remaining_cpu_work": ["vector export", "independent full-run and metric audit", "curves and report", "platform billing and release verification"]}
        atomic_json(marker, result)
        return result
    except BaseException as error:
        status.update(phase="STOPPED_FOR_AUDIT", error_type=type(error).__name__, error=str(error))
        atomic_json(status_path, status)
        raise
    finally:
        monitor.stop()
        stream.close()


def finalize_on_cpu(root, directory, data):
    root, directory = Path(root), Path(directory)
    complete = read_json(directory / "GPU_WORK_COMPLETE.json")
    if complete["source_sha256"] != source_identity(root)["source_sha256"]:
        raise ValueError("CPU finalization must use the same frozen implementation")
    campaign = directory / "campaign"
    exports = export_checkpoints(root, campaign)
    atomic_json(campaign / "export_index.json", exports)
    proof = finalize_delivery(root, campaign, data)
    atomic_json(directory / "CPU_DELIVERY_PASS.json", proof)
    return proof
