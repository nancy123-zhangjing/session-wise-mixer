"""Versioned process supervisor around the unchanged b240 stage-06 training tree.

Training, checkpoint bindings, selection, evaluation and gates are imported from
the frozen tree. This file has its own deployment hash and never changes that
tree. It replaces only the process-management loop that mistook an empty
/proc command line for a dead child.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TRAINING_SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def process_token(pid):
    """Only stable identity fields; argv is observational and may be empty."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": pid, "start_ticks": fields[19], "host": socket.gethostname(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def same_process(token):
    return token is not None and process_token(token["pid"]) == token


def child_finished(process):
    # The parent's child handle is authoritative. An argv read cannot turn an
    # existing, unexited child into a failed experiment.
    return process.poll() is not None


def worker_running(process, token):
    if child_finished(process):
        return False
    if same_process(token):
        return True
    # A child may exit between poll() and /proc sampling. Reap it before
    # classifying a stable-identity discrepancy as an error.
    if child_finished(process):
        return False
    raise RuntimeError("Stable worker identity lost while child remains unexited")


def stop_children(records, grace_seconds=5, kill_seconds=5):
    """Bounded cleanup; failures are returned and cannot hide the root error."""
    errors = []
    for record in records.values():
        process, token = record["process"], record["token"]
        if process.poll() is None and same_process(token):
            try:
                if os.getpgid(process.pid) != process.pid:
                    raise RuntimeError("Owned worker is no longer its session leader")
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append({"pid": process.pid, "stage": "terminate", "error": repr(error)})
    deadline = time.monotonic() + grace_seconds
    while any(r["process"].poll() is None for r in records.values()) and time.monotonic() < deadline:
        time.sleep(0.05)
    for record in records.values():
        process, token = record["process"], record["token"]
        if process.poll() is None and same_process(token):
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append({"pid": process.pid, "stage": "kill", "error": repr(error)})
    for record in records.values():
        try:
            record["process"].wait(timeout=kill_seconds)
        except Exception as error:
            errors.append({"pid": record["process"].pid, "stage": "reap", "error": repr(error)})
        record["log"].close()
    return errors


def bind_training_root(root):
    sys.path.insert(0, str(root))
    from scripts.continuous.trainer import source_identity
    identity = source_identity(root)
    if identity["source_sha256"] != TRAINING_SOURCE:
        raise ValueError("Only the unchanged precision-v2 training implementation is supported")
    return identity


def run_queue(root, directory, cache, gate_path, deployment_manifest, poll_seconds=2):
    import fcntl
    from scripts.continuous.campaign import expand_dependencies, freeze_validation_selection, next_task, verify_result
    from scripts.continuous.data import sha256
    from scripts.continuous.gpu_smoke import gpu_inventory
    from scripts.continuous.protocol import atomic_json
    root, directory, cache = Path(root), Path(directory), Path(cache)
    manifest_sha = sha256(deployment_manifest)
    supervisor_sha = sha256(__file__)
    deployment = json.loads(Path(deployment_manifest).read_text())
    if deployment["supervisor_sha256"] != supervisor_sha or deployment["training_source_sha256"] != TRAINING_SOURCE:
        raise ValueError("Supervisor deployment does not match its sealed manifest")
    inventory = gpu_inventory()
    gate = json.loads(Path(gate_path).read_text())
    if (gate["status"] != "PASS" or gate["source_sha256"] != TRAINING_SOURCE
            or gate["cache_manifest_sha256"] != sha256(cache / "cache.json")):
        raise ValueError("Missing matching GPU gate")
    path = directory / "queue.json"
    state = json.loads(path.read_text())
    if state["workers"] or state["source_sha256"] != TRAINING_SOURCE:
        raise ValueError("Prepared recovery state required; never adopt an unverified old Job")
    with (directory / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.update(status="RUNNING", gpu_inventory=inventory,
                     owner=process_token(os.getpid()), supervisor_sha256=supervisor_sha,
                     deployment_manifest_sha256=manifest_sha)
        records = {}
        atomic_json(path, state)
        try:
            while True:
                bind_training_root(root)
                if sha256(__file__) != supervisor_sha or sha256(deployment_manifest) != manifest_sha:
                    raise RuntimeError("Supervisor deployment changed during execution")
                if sha256(cache / "cache.json") != gate["cache_manifest_sha256"]:
                    raise RuntimeError("Cache changed")
                for gpu, record in list(records.items()):
                    process, task = record["process"], record["task"]
                    if worker_running(process, record["token"]):
                        continue
                    if process.returncode != 0:
                        task.update(status="failed", output=str(record["output"]), exit_code=process.returncode)
                        raise RuntimeError(f"Worker exited {process.returncode}: {task['task_id']}; inspect its log")
                    task["result"] = verify_result(task, record["output"], TRAINING_SOURCE)
                    task.update(status="succeeded", output=str(record["output"]), finished_at=time.time())
                    record["log"].close()
                    del records[gpu]
                    del state["workers"][str(gpu)]
                expand_dependencies(state)
                if len(state["tasks"]) == 89 and all(t["status"] == "succeeded" for t in state["tasks"]):
                    atomic_json(directory / "FROZEN_SELECTION.json", freeze_validation_selection(state))
                    state["status"] = "TRAINING_COMPLETE"
                    atomic_json(path, state)
                    return state
                for gpu in range(8):
                    if gpu in records:
                        continue
                    task = next_task(state, gpu)
                    if task is None:
                        continue
                    task_path = directory / "tasks" / (task["task_id"] + ".json")
                    atomic_json(task_path, {k: task[k] for k in ("task_id", "trial_id", "phase", "seed", "protocol", "config", "config_sha256")})
                    output = directory / "runs" / task["task_id"] / f"attempt_{task['attempt']}"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"), "train",
                               "--task", str(task_path), "--cache", str(cache), "--output", str(output), "--device", "cuda:0"]
                    if output.is_dir():
                        command.append("--resume")
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": inventory[gpu]["uuid"],
                                   "CONTINUOUS_GPU_UUID": inventory[gpu]["uuid"], "OMP_NUM_THREADS": "2",
                                   "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                    log = (output.parent / f"attempt_{task['attempt']}.log").open("ab")
                    process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    token = process_token(process.pid)
                    record = {"process": process, "token": token, "log": log, "task": task, "output": output}
                    records[gpu] = record
                    if token is None:
                        raise RuntimeError("Launched worker has no stable identity; inspect log")
                    state["workers"][str(gpu)] = {"task_id": task["task_id"], "stable_identity": token,
                        "expected_argv": command, "output": str(output), "started_at": time.time(),
                        "gpu_uuid": inventory[gpu]["uuid"]}
                    task.update(status="running", gpu=gpu, output=str(output))
                atomic_json(path, state)
                if not records:
                    raise RuntimeError("No runnable dependency path")
                time.sleep(poll_seconds)
        except BaseException as error:
            state.update(status="STOPPED_FOR_AUDIT", root_error={"type": type(error).__name__, "message": str(error)})
            # Save the triggering error before cleanup, even if an OS operation fails.
            atomic_json(path, state)
            state["cleanup_errors"] = stop_children(records)
            for record in records.values():
                if record["task"]["status"] == "running":
                    record["task"].update(status="interrupted", output=str(record["output"]))
            atomic_json(path, state)
            raise


def run_workflow(args):
    root, directory, cache = Path(args.root).resolve(), Path(args.output).resolve(), Path(args.cache).resolve()
    bind_training_root(root)
    from scripts.continuous.data import sha256
    from scripts.continuous.delivery import benchmark_checkpoints, run_evaluations
    from scripts.continuous.gpu_smoke import gpu_inventory, run_gpu_gate
    from scripts.continuous.protocol import MODELS, atomic_json
    from scripts.continuous.resources import ResourceMonitor
    import fcntl
    with (directory / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = ResourceMonitor(directory / "resource_attempts" / args.job_name)
        status = {"job_name": args.job_name, "source_sha256": TRAINING_SOURCE,
                  "supervisor_sha256": sha256(__file__), "started_unix": time.time(), "pid": os.getpid(), "cache": str(cache)}
        def phase(name):
            status.update(phase=name, updated_unix=time.time())
            atomic_json(directory / "PIPELINE.json", status)
            with (directory / "pipeline_events.jsonl").open("a") as stream:
                stream.write(json.dumps(status) + "\n")
            print(json.dumps({"event": "phase", **status}), flush=True)
        monitor.start()
        try:
            phase("gpu_smoke")
            gate_path = directory / "smoke/GPU_GATE.json"
            if not gate_path.exists():
                run_gpu_gate(root, directory / "smoke", cache, args.cpu_gate)
            phase("training_queue")
            state = run_queue(root, directory / "campaign", cache, gate_path, args.deployment_manifest)
            phase("final_test")
            inventory = gpu_inventory()
            evaluations = run_evaluations(root, directory / "campaign", cache, inventory)
            phase("final_exclusive_benchmark")
            checkpoints = {model: Path(next(t["output"] for t in state["tasks"] if t["phase"] == "formal"
                and t["seed"] == 3031 and t["config"]["model"] == model)) / "best.pt" for model in MODELS}
            benchmark_checkpoints(root, directory / "campaign/final_benchmarks", cache, checkpoints, inventory[0]["uuid"])
            phase("gpu_work_complete")
            result = {"status": "GPU_WORK_COMPLETE", "source_sha256": TRAINING_SOURCE,
                "supervisor_sha256": sha256(__file__), "training_runs": 89, "test_evaluations": len(evaluations),
                "job_name": args.job_name, "selection_sha256": sha256(directory / "campaign/FROZEN_SELECTION.json")}
            atomic_json(directory / "GPU_WORK_COMPLETE.json", result)
            return result
        except BaseException as error:
            status.update(phase="STOPPED_FOR_AUDIT", error_type=type(error).__name__, error=str(error))
            atomic_json(directory / "PIPELINE.json", status)
            raise
        finally:
            monitor.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "cache", "cpu-gate", "job-name", "deployment-manifest"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Execution requires the explicitly allocated Linux GPU Job")
    print(json.dumps(run_workflow(args)), flush=True)


if __name__ == "__main__":
    main()
