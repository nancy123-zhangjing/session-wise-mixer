"""Bounded validation trace workers, one independent process per visible H100.

This wrapper never submits jobs, trains, evaluates the test split, or retries.
The CPU preflight and private input copies must already exist. A persistent
exclusive ownership receipt prevents a second allocation from using this output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

CELLS = ("control_beam32", "control_beam256", "item_local_beam32", "item_local_beam256")
# Long cells first also balances the two-GPU queue.
QUEUE = ("control_beam256", "item_local_beam256", "control_beam32", "item_local_beam32")
EXPECTED_CHECKPOINTS = {
    "control": "55960f52906b0c4972a90b0ad06c078ef12fceac10fa4400b113e41d9a5c1f5f",
    "item_local": "f093c22e4e0cb354870b467e41f4ebab96d90047daf9d0e3d33991e3108b348b",
}
EXPECTED_SIDS = {
    "control": "a5483ad7d06a0076e035e5746f5c5d85ecc92eac511fa05e87ab91c257c1d0c1",
    "item_local": "075bf705f54deaba794324193de95e6d7f0ae3f985166b39652f90d15859b6e2",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_record(row, parent=None):
    path = Path(row["path"]).resolve()
    if parent is not None:
        need(path.is_relative_to(parent.resolve()), "Receipt refers outside its cell output")
    need(path.is_file() and path.stat().st_size == row["size_bytes"]
         and sha(path) == row["sha256"], f"Artifact bytes changed: {path}")


def under(parent, relative):
    parent = Path(parent).resolve()
    target = (parent / relative).resolve()
    need(target.is_relative_to(parent), "Input escaped its locked workspace")
    return target


def verify_frozen(plan, output):
    """Independently recheck sources, selected checkpoints and copied inputs."""
    sources = plan["sources"]
    for kind in ("root", "engine"):
        source = Path(sources[kind]["path"])
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        need(commit == sources[kind]["commit"] and not dirty, f"Frozen {kind} source changed")
    for relative, digest in sources["engine_sources"].items():
        need(sha(under(sources["engine"]["path"], relative)) == digest, "Frozen engine source bytes changed")
    # Include any extra observer/reference records added by the trace evaluator.
    def walk(value):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"}.issubset(value) and Path(value["path"]).is_absolute():
                verify_record(value)
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(plan)
    for variant, arm in plan["arms"].items():
        need(arm["checkpoint"]["sha256"] == EXPECTED_CHECKPOINTS[variant]
             and arm["selected_SID_sha256"] == EXPECTED_SIDS[variant], "Selected model or SID identity differs")
        need(len(arm["inputs"]) == 13, "Expected all thirteen locked inputs")
        for row in arm["inputs"]:
            for parent in (Path(arm["run"]) / "workspace", output / "shared/workspace"):
                path = under(parent, row["path"])
                need(path.is_file() and path.stat().st_size == row["size_bytes"]
                     and sha(path) == row["sha256"], "Original/private input bytes differ")


def verify_cell(output, name, plan, preflight):
    cell = output / name
    complete_path = cell / "CELL_COMPLETE.json"
    result = read(complete_path)
    need(result["status"] == "cell_completed" and result["cell"] == name
         and result["grid_identity_sha256"] == plan["identity_sha256"], "Cell completion identity differs")
    need({"metrics", "per_user", "trace_parity"}.issubset(result["files"]), "Missing trace evidence")
    for row in result["files"].values():
        verify_record(row, cell)
    metrics = read(result["files"]["metrics"]["path"])
    variant, beam = name.rsplit("_beam", 1)
    need(metrics["cell"] == name and metrics["split"] == "val" and metrics["count"] == 22363
         and metrics["beam_act"] == int(beam) and metrics["beam_max"] == int(beam)
         and metrics["return_count"] == 10 and metrics["partial_diagnostic"] is False,
         "Cell geometry or full validation coverage differs")
    need(metrics["model_state_unchanged"] is True and metrics["selected_SID_unchanged"] is True
         and metrics["gradients_created"] is False and metrics["test_data_evaluated"] is False,
         "Cell changed state, produced gradients or evaluated test data")
    need(metrics["checkpoint_sha256"] == EXPECTED_CHECKPOINTS[variant]
         and metrics["selected_SID_sha256"] == EXPECTED_SIDS[variant]
         and metrics["model_state_before_sha256"] == metrics["model_state_after_sha256"]
         == preflight["best_states"][variant]["model_state_sha256"], "Cell model bytes differ from CPU preflight")
    need(metrics["validation_user_history_sha256"] == plan["validation_user_history_sha256"], "Validation histories differ")
    need(metrics["trace_parity"]["status"] == "matched"
         and metrics["original_predictions_reproduction"]["status"] == "matched"
         and metrics["original_predictions_reproduction"]["users"] == 22363,
         "Instrumentation parity or original predictions did not reproduce")
    parity = read(result["files"]["trace_parity"]["path"])
    need(parity["status"] == "matched" and parity["users"] > 0, "Missing original-versus-observer smoke parity")
    if beam == "32":
        need(metrics["original_val32_reproduction"]["status"] == "matched", "Original val32 metrics did not reproduce")
    # Recompute the user/target order from the hashed records, not just metadata.
    order = hashlib.sha256()
    count = 0
    with Path(result["files"]["per_user"]["path"]).open(encoding="utf-8") as handle, \
            Path(plan["references"][name]["per_user"]["path"]).open(encoding="utf-8") as reference:
        for count, (line, original_line) in enumerate(zip_longest(handle, reference), 1):
            need(line is not None and original_line is not None, "Original/current per-user lengths differ")
            row = json.loads(line)
            original = json.loads(original_line)
            need(all(key in row and row[key] == value for key, value in original.items()),
                 "Actual per-user prediction/metric differs from the prior frozen grid")
            need(row["index"] == count - 1 and len(row["predicted_SIDs"]) == 10, "Per-user coverage/Top10 shape differs")
            order.update(json.dumps([row["index"], row["user"], row["target_item_id"]], separators=(",", ":")).encode() + b"\n")
    need(count == 22363 and order.hexdigest() == metrics["user_target_order_sha256"], "Actual user order/count differs")
    return {"receipt": record(complete_path), "metrics": metrics}


def stop_groups(running):
    """Terminate every worker session, including orphaned dataset workers."""
    states = [row for row in running.values() if not row.get("group_cleaned", False)]
    processes = [row["process"] for row in states]
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    finish = time.monotonic() + 8
    for process in processes:
        try:
            process.wait(timeout=max(0.01, finish - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A parent may have exited while its descendants still hold the process group.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    for row in states:
        row["group_cleaned"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--max-seconds", type=int, default=1700)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    if not 60 < args.max_seconds <= 1700:
        parser.error("Process cap must be 61..1700 seconds inside the 1800-second platform cap")
    need(os.name == "posix", "GPU orchestration requires POSIX process-group cleanup")
    goal, output = args.goal_root.resolve(), args.output.resolve()
    script, observer = args.script.resolve(), args.observer.resolve()
    need(all(path != goal and not path.is_relative_to(goal) for path in (output, script, observer)),
         "Trace outputs and scripts must be outside the immutable training campaign")
    need(output.is_dir(), "CPU preflight must create the output first")
    need(not (output / "COMPLETE.json").exists()
         and not any((output / cell / "CELL_COMPLETE.json").exists() for cell in CELLS),
         "Existing completion evidence must not be overwritten or automatically resumed")
    need(sha(script) == args.script_sha256 and sha(observer) == args.observer_sha256, "Evaluator/observer SHA256 mismatch")
    plan_path, preflight_path = output / "GRID_PLAN.json", output / "GRID_PREFLIGHT.json"
    plan, preflight = read(plan_path), read(preflight_path)
    need(plan["split"] == "val" and plan["validation_count"] == 22363
         and plan["max_eval_batches"] is None and tuple(plan["cells"]) == CELLS,
         "Only the complete four-cell validation trace is authorized")
    need(Path(plan["goal_root"]).resolve() == goal and plan["sources"]["tool"]["sha256"] == args.script_sha256,
         "Preflight refers to a different campaign/evaluator")
    need(Path(plan["sources"]["tool"]["path"]).resolve() == script
         and Path(plan["sources"]["observer"]["path"]).resolve() == observer
         and plan["sources"]["observer"]["sha256"] == args.observer_sha256,
         "Evaluator/observer paths do not match CPU preflight")
    need(preflight["status"] == "CPU_PREFLIGHT_PASSED_NO_INFERENCE"
         and preflight["grid_identity_sha256"] == plan["identity_sha256"]
         and preflight["cuda_initialized"] is False and preflight["test_data_evaluated"] is False,
         "A matching CPU preflight is required")
    started, monotonic_started = time.time(), time.monotonic()
    deadline = started + args.max_seconds - 30
    receipt_path = output / "GPU_PROCESS_RECEIPT.json"
    # This is deliberately persistent after failure: no accidental duplicate run.
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    frozen = {str(path): sha(path) for path in (script, observer, Path(__file__).resolve(), plan_path, preflight_path)}
    receipt = {"status": "starting", "started_unix": started,
               "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
               "allocated_gpu_count": args.gpus, "max_process_seconds": args.max_seconds,
               "platform_cap_seconds": 1800, "deadline_unix": deadline,
               "max_platform_gpu_hours": args.gpus * 0.5,
               "training_authorized": False, "test_evaluation_authorized": False,
               "grid_identity_sha256": plan["identity_sha256"], "frozen_script_and_plan_sha256": frozen,
               "launch_mode": "independent_single_gpu_workers_no_DDP", "workers": {},
               "time_scope": "Process time; actual GPU allocation must be reconciled with platform lifecycle"}
    write(receipt_path, receipt)
    running, all_workers, completed = {}, {}, {}
    old_handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f"Wrapper received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupted)
        sys.path.insert(0, str(goal / "code/scripts/train"))
        from item_local_gpu_job import single_process_environment
        environment, removed = single_process_environment(os.environ)
        environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                           WANDB_MODE="offline", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        receipt["normalized_single_process_variables"] = removed
        verify_frozen(plan, output)
        import torch
        need(torch.cuda.is_available() and torch.cuda.device_count() == args.gpus,
             "Visible GPU allocation must exactly match --gpus; do not waste extra allocated cards")
        devices = [torch.cuda.get_device_name(index) for index in range(args.gpus)]
        need(all("H100" in name.upper() for name in devices), "This measured protocol requires H100 GPUs")
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        gpu_tokens = visible.split(",") if visible is not None else [str(index) for index in range(args.gpus)]
        need(len(gpu_tokens) == args.gpus and all(token.strip() for token in gpu_tokens), "Unrecognized visible GPU mapping")
        receipt.update(status="running", devices=devices)
        write(receipt_path, receipt)
        pending, available = list(QUEUE), list(range(args.gpus))
        while pending or running:
            need(time.time() < deadline and time.monotonic() - monotonic_started < args.max_seconds - 30,
                 "Bounded trace evaluation reached its deadline")
            while pending and available:
                name, slot = pending.pop(0), available.pop(0)
                cell = output / name
                cell.mkdir(exist_ok=True)
                log_path = cell / "GPU_CONSOLE.log"
                command = [sys.executable, "-u", "-X", "utf8", "-B", str(script), "--goal-root", str(goal),
                           "--output", str(output), "--mode", "evaluate", "--cell", name,
                           "--deadline-unix", str(deadline)]
                worker_env = dict(environment, CUDA_VISIBLE_DEVICES=gpu_tokens[slot].strip())
                log = log_path.open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=str(output), env=worker_env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                state = {"process": process, "log": log, "gpu_slot": slot}
                running[name] = all_workers[name] = state
                receipt["workers"][name] = {"status": "running", "pid": process.pid, "gpu_slot": slot,
                                            "cuda_visible_devices": gpu_tokens[slot].strip(), "argv": command,
                                            "started_unix": time.time(), "console": str(log_path)}
                write(receipt_path, receipt)
                print(json.dumps({"status": "cell_started", "cell": name, "gpu_slot": slot}), flush=True)
            for name, state in list(running.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["log"].close()
                receipt["workers"][name].update(returncode=code, finished_unix=time.time())
                need(code == 0, f"Cell {name} exited with code {code}; stopping all other workers")
                stop_groups({name: state})
                completed[name] = verify_cell(output, name, plan, preflight)
                receipt["workers"][name]["status"] = "verified_cell_completed"
                write(receipt_path, receipt)
                available.append(state["gpu_slot"])
                del running[name]
                print(json.dumps({"status": "cell_verified", "cell": name, "completed": len(completed)}), flush=True)
            if running:
                time.sleep(0.5)
        need(set(completed) == set(CELLS), "Not all four trace cells completed")
        need(len({row["metrics"]["user_target_order_sha256"] for row in completed.values()}) == 1,
             "Cells evaluated different user/target order")
        verify_frozen(plan, output)
        need(all(sha(path) == digest for path, digest in frozen.items()), "Scripts/plan/preflight changed during evaluation")
        need(time.monotonic() - monotonic_started < args.max_seconds, "Post-run verification exceeded the process cap")
        complete_path = output / "COMPLETE.json"
        need(not complete_path.exists(), "Refusing to overwrite completion evidence")
        write(complete_path, {"schema": "stage07_validation_beam_trace_v1", "status": "validation_beam_trace_completed",
              "grid_identity_sha256": plan["identity_sha256"], "cells": {name: completed[name] for name in CELLS},
              "original_artifacts_unchanged": True, "scripts_and_plan_unchanged": True,
              "same_user_target_order": True, "all_trace_parity_matched": True,
              "all_original_predictions_reproduced": True, "test_data_evaluated": False, "training_performed": False,
              "scope": "Validation-only post-hoc tracing of frozen A/B best25 and saved SIDs; no decoder decision changes"})
        receipt.update(status="process_completed_pending_independent_result_review", complete_sha256=sha(complete_path))
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        if "deadline" in str(error).lower() or "cap" in str(error).lower():
            receipt["timed_out"] = True
        raise
    finally:
        # Ignore repeated termination signals while cleaning every owned process group.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            stop_groups(all_workers)
        finally:
            for state in all_workers.values():
                state["log"].close()
            receipt.update(finished_unix=time.time(), elapsed_process_seconds=time.monotonic() - monotonic_started)
            receipt["process_allocation_gpu_hours"] = args.gpus * receipt["elapsed_process_seconds"] / 3600
            write(receipt_path, receipt)
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
