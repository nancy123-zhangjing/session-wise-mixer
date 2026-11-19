"""Deterministic dependency graph and durable single-node worker queue.

Only complete, hash-bound, full-dataset runs can unlock dependent experiments.
This module does not submit platform jobs or allocate any additional node.
"""
import json
import math
import os
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .data import EXPECTED_ROWS, sha256
from .protocol import MODELS, atomic_json, canonical_hash, generate_search
from .trainer import source_identity


def candidate_key(task):
    result = task["result"]
    metrics = result["best_validation"]
    return (-metrics["NDCG@10"], -metrics["HR@10"], result["parameter_count"], task["trial_id"])


def task_record(base, phase, seed, protocol="full_train_prefix_v1"):
    identity = f"{phase}__{base['trial_id']}__seed{seed}"
    return {"task_id": identity, "trial_id": base["trial_id"], "phase": phase,
            "seed": seed, "protocol": protocol, "config": base["config"],
            "config_sha256": base["config_sha256"], "status": "pending", "attempt": 0}


def initialize_state(source_sha256):
    manifest = generate_search()
    return {"schema_version": 1, "source_sha256": source_sha256,
            "manifest_sha256": canonical_hash(manifest), "status": "PREPARED",
            "tasks": [task_record(t, "search", 2024) for t in manifest["trials"]],
            "top3": {}, "selected_configs": {}, "workers": {}, "events": []}


def expand_dependencies(state):
    """Completion order changes dispatch timing, never the candidate population."""
    created = []
    for model in MODELS:
        searches = [t for t in state["tasks"] if t["phase"] == "search" and t["config"]["model"] == model]
        if model not in state["top3"] and len(searches) == 32 and all(t["status"] == "succeeded" for t in searches):
            top3 = sorted(searches, key=candidate_key)[:3]
            state["top3"][model] = [t["trial_id"] for t in top3]
            for base in top3:
                for seed in (2025, 2026):
                    created.append(task_record(base, "confirmation", seed))
        confirmations = [t for t in state["tasks"] + created
                         if t["phase"] == "confirmation" and t["config"]["model"] == model]
        if model in state["top3"] and model not in state["selected_configs"] and len(confirmations) == 6:
            if all(t["status"] == "succeeded" for t in confirmations):
                candidates = []
                for trial_id in state["top3"][model]:
                    group = [t for t in searches + confirmations if t["trial_id"] == trial_id]
                    if len(group) != 3 or {t["seed"] for t in group} != {2024, 2025, 2026}:
                        raise ValueError("Incomplete confirmation seeds")
                    average = {metric: statistics.mean(t["result"]["best_validation"][metric] for t in group)
                               for metric in ("NDCG@10", "HR@10")}
                    candidates.append({"base": group[0], "metrics": average,
                                       "parameters": group[0]["result"]["parameter_count"]})
                selected = min(candidates, key=lambda c: (-c["metrics"]["NDCG@10"],
                               -c["metrics"]["HR@10"], c["parameters"], c["base"]["trial_id"]))
                base = selected["base"]
                state["selected_configs"][model] = {"trial_id": base["trial_id"],
                    "config": base["config"], "config_sha256": base["config_sha256"],
                    "confirmation_mean": selected["metrics"],
                    "candidates": [{"trial_id": c["base"]["trial_id"], "metrics": c["metrics"]}
                                   for c in candidates]}
                created.extend(task_record(base, "formal", seed) for seed in range(3031, 3036))
                if model == "sasrec_fullce":
                    created.extend(task_record(base, "legacy", seed, "legacy_prefix_v1")
                                   for seed in (3031, 3032, 3033))
    ids = {t["task_id"] for t in state["tasks"]}
    for task in created:
        if task["task_id"] in ids:
            raise ValueError("Duplicate dependent task")
        ids.add(task["task_id"])
    state["tasks"].extend(created)
    if len(state["tasks"]) > 89:
        raise ValueError("Training budget exceeded")
    return created


def next_task(state, gpu):
    pending = [t for t in state["tasks"] if t["status"] == "pending"]
    if not pending:
        return None
    preferred = "sasrec_fullce" if gpu < 4 else "esasrec"
    family = [t for t in pending if t["config"]["model"] == preferred]
    return (family or pending)[0]


def verify_result(task, output, source_sha256):
    output = Path(output)
    result = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    expected = {"config_sha256": task["config_sha256"], "seed": task["seed"],
                "protocol": task["protocol"], "source_sha256": source_sha256,
                "precision": task["config"]["precision"], "smoke": False}
    if result["status"] != "SUCCEEDED" or result["smoke"]:
        raise ValueError("Not a valid formal training result")
    if any(result["binding"][key] != value for key, value in expected.items()):
        raise ValueError("Run/source/config binding mismatch")
    n_train = EXPECTED_ROWS[task["protocol"]]["train"]
    if result["validation_samples"] != 22363 or result["best_validation"]["samples"] != 22363:
        raise ValueError("Partial validation is not a valid trial")
    if result["train_samples_per_full_epoch"] != n_train or not 1 <= result["epochs"] <= 300:
        raise ValueError("Incomplete training data or invalid epoch budget")
    if result["updates"] != result["epochs"] * math.ceil(n_train / task["config"]["batch_size"]):
        raise ValueError("Incomplete epoch/update count")
    for metric in ("NDCG@10", "HR@10", "NLL"):
        if not math.isfinite(result["best_validation"][metric]):
            raise ValueError("Nonfinite selection metric")
    for name in ("best", "last"):
        if sha256(output / (name + ".pt")) != result[name + "_sha256"]:
            raise ValueError("Run checkpoint tampering")
    return result


def process_identity(pid):
    """Linux identity prevents treating a reused PID as the original worker."""
    try:
        start = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return {"pid": pid, "start_ticks": start, "command": command, "host": socket.gethostname()}
    except (OSError, IndexError):
        return None


INFRASTRUCTURE_REASONS = {"preempted", "node_lost", "resource_reclaimed", "platform_evicted"}


def reconcile_terminal_job(directory, evidence):
    """Apply only after the agent has verified terminal platform status/events.

    Evidence retains raw platform observations and an explicit classification;
    a stale PID or local lock file alone is deliberately insufficient.
    """
    directory = Path(directory)
    state = json.loads((directory / "queue.json").read_text(encoding="utf-8"))
    if (evidence.get("terminal_status") not in {"FAILED", "CANCELLED", "SUCCEEDED"}
            or evidence.get("reason") not in INFRASTRUCTURE_REASONS
            or not evidence.get("platform_job_name") or not evidence.get("platform_job_id")
            or not evidence.get("observed_unix") or not evidence.get("raw_platform_observations")):
        raise ValueError("Confirmed terminal platform evidence and infrastructure reason required")
    for worker in state["workers"].values():
        if process_identity(worker["identity"]["pid"]) == worker["identity"]:
            raise RuntimeError("Original worker remains live; never duplicate it")
    if any(t["attempt"] >= 2 for t in state["tasks"] if t["status"] in {"running", "interrupted", "failed"}):
        raise RuntimeError("Two infrastructure retries already used; do not partially reconcile")
    changed = []
    outputs = {w["task_id"]: w["output"] for w in state["workers"].values()}
    for task in state["tasks"]:
        if task["status"] not in {"running", "interrupted", "failed"}:
            continue
        old = Path(task.get("output") or outputs[task["task_id"]])
        if (old / "COMPLETE.json").is_file():
            task["result"] = verify_result(task, old, state["source_sha256"])
            task["status"], task["output"] = "succeeded", str(old)
            continue
        if task["attempt"] >= 2:
            raise RuntimeError("Two infrastructure retries already used")
        task["attempt"] += 1
        new = old.parent / f"attempt_{task['attempt']}"
        if new.exists():
            raise FileExistsError("Retry destination exists; reconcile without overwriting")
        if (old / "CURRENT.json").is_file():
            from .checkpoints import repair_committed
            committed = repair_committed(old)
            if committed["binding"]["config_sha256"] != task["config_sha256"]:
                raise ValueError("Retry checkpoint config changed")
            new.mkdir()
            for name in ("run.json", "best.pt", "best.receipt.json", "last.pt", "last.receipt.json",
                         "best_validation.json", "curves.csv", "performance.jsonl", "resume_events.jsonl", "CURRENT.json"):
                path = old / name
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, new / name)
            pointer = json.loads((old / "CURRENT.json").read_text(encoding="utf-8"))
            for relative in {pointer[kind]["file"] for kind in ("last", "best")}:
                source = old / relative
                (new / relative).parent.mkdir(exist_ok=True)
                shutil.copyfile(source, new / relative)
                shutil.copyfile(source.with_suffix(".receipt.json"), (new / relative).with_suffix(".receipt.json"))
        task["status"] = "pending"
        task["resume_from_attempt"] = str(old)
        task["output"] = str(new)
        changed.append(task["task_id"])
    state["workers"] = {}
    state["status"] = "RECONCILED_AFTER_TERMINAL_JOB"
    state["events"].append({"type": "infrastructure_reconcile", "evidence": evidence,
                            "tasks": changed, "time": time.time()})
    atomic_json(directory / "queue.json", state)
    return state


def stop_verified_workers(state, handles):
    """Stop both newly launched and adopted workers, but only matching identities."""
    targets = []
    for worker in state["workers"].values():
        identity = worker["identity"]
        if process_identity(identity["pid"]) == identity:
            try:
                os.killpg(identity["pid"], signal.SIGTERM)
                targets.append(identity)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while targets and time.monotonic() < deadline:
        targets = [i for i in targets if process_identity(i["pid"]) == i]
        if targets:
            time.sleep(0.2)
    for identity in targets:
        if process_identity(identity["pid"]) == identity:
            try:
                os.killpg(identity["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in handles.values():
        process.wait(timeout=10)


def freeze_validation_selection(state):
    if len(state["tasks"]) != 89 or any(t["status"] != "succeeded" for t in state["tasks"]):
        raise ValueError("All 89 verified training runs must precede final test unlock")
    models = {}
    for model in MODELS:
        formal = [t for t in state["tasks"] if t["phase"] == "formal" and t["config"]["model"] == model]
        if len(formal) != 5 or {t["seed"] for t in formal} != set(range(3031, 3036)):
            raise ValueError("Missing formal seed")
        average = {metric: statistics.mean(t["result"]["best_validation"][metric] for t in formal)
                   for metric in ("NDCG@10", "HR@10")}
        models[model] = {**state["selected_configs"][model], "formal_validation_mean": average,
                         "parameter_count": formal[0]["result"]["parameter_count"],
                         "formal_tasks": [t["task_id"] for t in formal]}
    choice = min(models, key=lambda m: (-models[m]["formal_validation_mean"]["NDCG@10"],
                 -models[m]["formal_validation_mean"]["HR@10"], models[m]["parameter_count"], m))
    return {"validation_only": True, "test_unlocked": True, "selected_model": choice,
            "models": models, "source_sha256": state["source_sha256"],
            "training_results_sha256": canonical_hash([
                {"task_id": t["task_id"], "result": t["result"]} for t in state["tasks"]])}


def run_queue(root, directory, cache, gpu_gate, poll_seconds=10):
    root, directory, cache = Path(root).resolve(), Path(directory).resolve(), Path(cache).resolve()
    if os.name != "posix":
        raise ValueError("GPU queue runs only inside the Linux single-node Job")
    import fcntl
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    lock_stream = (directory / "scheduler.lock").open("a+")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if torch.cuda.device_count() != 8 or any("H100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError("Exactly eight visible H100 GPUs required")
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()
    gpu_inventory = [dict(zip(("index", "uuid", "name", "memory_mib"),
                              [value.strip() for value in row.split(",")])) for row in gpu_rows]
    if (len(gpu_inventory) != 8 or len({g["uuid"] for g in gpu_inventory}) != 8
            or any("H100" not in g["name"] for g in gpu_inventory)):
        raise ValueError("Ambiguous single-node H100 UUID inventory")
    identity = source_identity(root)
    gate = json.loads(Path(gpu_gate).read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("source_sha256") != identity["source_sha256"]:
        raise ValueError("Missing or stale GPU smoke gate")
    if gate.get("cache_manifest_sha256") != sha256(cache / "cache.json"):
        raise ValueError("Campaign cache differs from the GPU-validated cache")
    state_path = directory / "queue.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["source_sha256"] != identity["source_sha256"]:
            raise ValueError("Execution source changed during campaign")
        for worker in state["workers"].values():
            if worker["identity"] != process_identity(worker["identity"]["pid"]):
                # Do not infer that another platform Job is terminal from a stale
                # local PID alone. The agent must reconcile platform status first.
                raise RuntimeError("Worker handle disappeared; platform reconciliation required")
    else:
        state = initialize_state(identity["source_sha256"])
    state["status"] = "RUNNING"
    state["gpu_inventory"] = gpu_inventory
    state["owner"] = process_identity(os.getpid())
    handles, log_streams = {}, {}
    atomic_json(state_path, state)
    try:
        while True:
            if source_identity(root)["source_sha256"] != state["source_sha256"]:
                raise RuntimeError("Code changed: refusing to dispatch further tasks")
            if sha256(cache / "cache.json") != gate["cache_manifest_sha256"]:
                raise RuntimeError("Cache identity changed during campaign")
            for gpu_text, worker in list(state["workers"].items()):
                gpu = int(gpu_text)
                task = next(t for t in state["tasks"] if t["task_id"] == worker["task_id"])
                process = handles.get(gpu)
                live = process_identity(worker["identity"]["pid"])
                if live == worker["identity"] and (process is None or process.poll() is None):
                    continue
                output = Path(worker["output"])
                if process is not None and process.poll() == 0:
                    task["result"] = verify_result(task, output, state["source_sha256"])
                elif (output / "COMPLETE.json").is_file():
                    task["result"] = verify_result(task, output, state["source_sha256"])
                else:
                    task["status"] = "failed"
                    state["status"] = "FAILED"
                    task["exit_code"] = process.returncode if process else None
                    atomic_json(state_path, state)
                    raise RuntimeError(f"Worker failed: {task['task_id']}; inspect log before retry")
                task["status"], task["output"] = "succeeded", str(output)
                task["finished_at"] = time.time()
                del state["workers"][gpu_text]
                if gpu in log_streams:
                    log_streams.pop(gpu).close()
                handles.pop(gpu, None)
            expand_dependencies(state)
            if len(state["tasks"]) == 89 and all(t["status"] == "succeeded" for t in state["tasks"]):
                selection = freeze_validation_selection(state)
                path = directory / "FROZEN_SELECTION.json"
                if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
                    raise ValueError("Cannot replace a frozen model selection")
                atomic_json(path, selection)
                state["status"] = "TRAINING_COMPLETE"
                atomic_json(state_path, state)
                return state
            for gpu in range(8):
                if str(gpu) in state["workers"]:
                    continue
                task = next_task(state, gpu)
                if task is None:
                    continue
                task_path = directory / "tasks" / (task["task_id"] + ".json")
                atomic_json(task_path, {k: task[k] for k in ("task_id", "trial_id", "phase", "seed",
                    "protocol", "config", "config_sha256")})
                output = directory / "runs" / task["task_id"] / f"attempt_{task['attempt']}"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"),
                           "train", "--task", str(task_path), "--cache", str(cache),
                           "--output", str(output), "--device", "cuda:0"]
                if output.is_dir():
                    command.append("--resume")
                gpu_uuid = gpu_inventory[gpu]["uuid"]
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid,
                               "CONTINUOUS_GPU_UUID": gpu_uuid,
                               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                               "PYTHONUNBUFFERED": "1"}
                log = (output.parent / f"attempt_{task['attempt']}.log").open("ab")
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                worker_identity = process_identity(process.pid)
                if worker_identity is None:
                    raise RuntimeError("Cannot verify launched worker identity")
                handles[gpu], log_streams[gpu] = process, log
                state["workers"][str(gpu)] = {"task_id": task["task_id"], "identity": worker_identity,
                                             "output": str(output), "started_at": time.time(),
                                             "gpu_uuid": gpu_uuid}
                task["status"], task["gpu"] = "running", gpu
            atomic_json(state_path, state)
            if not state["workers"]:
                raise RuntimeError("No runnable dependency path; campaign incomplete")
            time.sleep(poll_seconds)
    except BaseException:
        state["status"] = "STOPPED_FOR_AUDIT"
        stop_verified_workers(state, handles)
        for worker in state["workers"].values():
            task = next(t for t in state["tasks"] if t["task_id"] == worker["task_id"])
            task["output"] = worker["output"]
            if task["status"] == "running":
                task["status"] = "interrupted"
        atomic_json(state_path, state)
        raise
    finally:
        for stream in log_streams.values():
            stream.close()
        lock_stream.close()
