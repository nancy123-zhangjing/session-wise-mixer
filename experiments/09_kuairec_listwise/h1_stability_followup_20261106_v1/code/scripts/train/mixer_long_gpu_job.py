"""One-H100, bounded-allocation stage05 Job entry; never creates platform jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/checks"))
import diger_mixer_long_protocol as protocol_lib

read, write, need, sha = protocol_lib.read, protocol_lib.write, protocol_lib.need, protocol_lib.sha


def budget(ledger, process_cap, platform_cap, mode, *, attempt=None, arm="all"):
    need(ledger["campaign_id"] == protocol_lib.CAMPAIGN and ledger["platform_prior_allocation_verified"] is True,
         "Need a platform-verified prior allocation ledger for this campaign")
    prior, acceptance = ledger["allocated_gpu_seconds"], ledger["acceptance_allocated_gpu_seconds"]
    need(all(not isinstance(value, bool) and math.isfinite(value) and value >= 0
             for value in (prior, acceptance, process_cap, platform_cap)), "Invalid finite budget")
    need(process_cap > 60 and platform_cap >= process_cap + 150, "Reserve 150 seconds for platform/container overhead")
    reservations = ledger.get("reservations", [])
    if reservations:
        need(len(reservations) <= 4 and len({row["attempt"] for row in reservations}) == len(reservations),
             "At most four distinct single-GPU reservations are authorized")
        need(all(isinstance(row["platform_cap_seconds"], (int, float))
                 and not isinstance(row["platform_cap_seconds"], bool)
                 and math.isfinite(row["platform_cap_seconds"]) and row["platform_cap_seconds"] > 150
                 for row in reservations), "Invalid GPU reservation caps")
        own = [row for row in reservations if row["attempt"] == attempt]
        need(len(own) == 1 and own[0]["mode"] == mode and own[0]["arm"] == arm
             and own[0]["platform_cap_seconds"] == platform_cap, "Own immutable reservation differs")
        if len(reservations) > 1:
            need(all(row["mode"] == "train_arm" and row["arm"] in protocol_lib.ARMS for row in reservations)
                 and len({row["arm"] for row in reservations}) == len(reservations),
                 "Only distinct training arms may overlap; selection/test waits for all training")
        reserved = sum(row["platform_cap_seconds"] for row in reservations)
    else:
        need(mode in ("smoke", "formal"), "Parallel-capable jobs require explicit allocation reservations")
        reserved = platform_cap
    need(prior + reserved <= 43200, "Cumulative allocation cap exceeds 12 GPU hours")
    need(all(row["released"] is True for row in ledger["prior_jobs"]), "A previous own Job has not been verified released")
    if mode == "smoke":
        need(acceptance + platform_cap <= 1800, "Cumulative H100 acceptance allocation exceeds 0.5 hours")


def source_binding(goal, code_commit, engine_commit):
    need(ROOT.resolve() == (goal / "code").resolve(), "Execute the wrapper from its own frozen campaign checkout")
    for directory, expected in ((goal / "code", code_commit), (goal / "engine", engine_commit)):
        need(len(expected) == 40, "Expected a complete frozen commit SHA")
        head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
        need(head == expected and not dirty, "Wrong or dirty execution checkout: " + str(directory))
    return dict(code_commit=code_commit, engine_commit=engine_commit,
                wrapper_sha256=sha(Path(__file__)), runner_sha256=sha(ROOT / "scripts/train/run_diger_mixer_long.py"),
                protocol_sha256=sha(ROOT / "configs/05_crosspse_joint/long/mixer_long_seed2024.json"))


def formal_ready(goal, binding, ledger_path, process_cap, mode="formal", arm="all"):
    ready_path = goal / "delivery/FORMAL_READINESS.json"
    ready = read(ready_path)
    need(ready["campaign_id"] == protocol_lib.CAMPAIGN and ready["cpu_gate_passed"] is True
         and ready["h100_gate_passed"] is True and ready["platform_policy_verified"] is True,
         "Missing CPU/H100/platform-policy readiness evidence")
    need(ready["binding"] == binding and ready["allocation_ledger_sha256"] == sha(ledger_path), "Readiness binding/ledger differs")
    need(ready["smoke_pass_sha256"] == sha(goal / "smoke/SMOKE_PASS.json")
         and ready["profile_sha256"] == sha(goal / "smoke/PROFILE.json"), "H100 gate/profile receipt changed")
    estimate = (ready["estimated_remaining_seconds_with_margin"] if mode == "formal"
                else ready["estimated_seconds_by_task"][mode + ":" + arm])
    need(estimate + 120 <= process_cap, "Declared process cap cannot contain measured remaining work")
    need(read(goal / "smoke/SMOKE_PASS.json")["status"] == "H100_SMOKE_PASS_NOT_FORMAL", "CPU smoke cannot authorize formal GPU training")
    return sha(ready_path)


def sampler(path, stop):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,index,utilization_gpu,memory_used_mib,memory_total_mib\n")
        while not stop.is_set():
            value = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                                    "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=15)
            if value.returncode == 0:
                handle.write(value.stdout)
                handle.flush()
            stop.wait(10)


def terminate_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=20)


def child(command, log_path, deadline):
    need(time.time() < deadline - 30, "No child process budget remains")
    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.time() > deadline:
                    terminate_owned(process)
                    raise TimeoutError("Owned process deadline reached; retain committed recovery state")
                time.sleep(5)
            need(process.returncode == 0, "Child failed; see " + str(log_path))
        finally:
            terminate_owned(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal", "train_arm", "evaluate"), required=True)
    parser.add_argument("--arm", choices=("all", *protocol_lib.ARMS), default="all")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--process-seconds", type=float, required=True)
    parser.add_argument("--platform-cap-seconds", type=float, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-engine-commit", required=True)
    args = parser.parse_args()
    goal = args.goal_root.resolve()
    need(goal.name == protocol_lib.CAMPAIGN and goal.parent == Path(read(protocol_lib.DEFAULT_CONFIG)["remote_parent"]),
         "Wrong own global campaign directory")
    need(args.attempt and all(letter.isalnum() or letter in "_-" for letter in args.attempt), "Invalid attempt name")
    ledger = read(args.ledger)
    need((args.mode == "train_arm") == (args.arm in protocol_lib.ARMS), "Only train_arm may select one independent arm")
    budget(ledger, args.process_seconds, args.platform_cap_seconds, args.mode, attempt=args.attempt, arm=args.arm)
    binding = source_binding(goal, args.expected_code_commit, args.expected_engine_commit)
    ready_sha = formal_ready(goal, binding, args.ledger, args.process_seconds, args.mode, args.arm) if args.mode != "smoke" else None
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        os.environ.pop(key, None)
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    attempt = goal / "delivery" / args.attempt
    attempt.mkdir(exist_ok=False)
    start = time.time()
    deadline = start + args.process_seconds
    receipt = dict(campaign_id=protocol_lib.CAMPAIGN, mode=args.mode, attempt=args.attempt,
                   started_utc=datetime.now(timezone.utc).isoformat(), process_deadline_unix=deadline,
                   declared_platform_cap_seconds=args.platform_cap_seconds, binding=binding,
                   prior_allocation_ledger_sha256=sha(args.ledger), formal_readiness_sha256=ready_sha,
                   process_time_is_not_platform_allocation=True)
    write(attempt / "STARTED.json", receipt)
    stop = threading.Event()
    monitor = threading.Thread(target=sampler, args=(attempt / "gpu_samples.csv", stop), daemon=True)
    monitor.start()
    common = [sys.executable, "-B", str(ROOT / "scripts/train/run_diger_mixer_long.py"),
              "--goal-root", str(goal), "--engine-root", str(goal / "engine"),
              "--expected-code-commit", args.expected_code_commit, "--expected-engine-commit", args.expected_engine_commit,
              "--deadline-unix", str(deadline)]
    try:
        if args.mode == "smoke":
            child(common + ["--mode", "smoke", "--profile"], attempt / "smoke.log", deadline)
            need(read(goal / "smoke/SMOKE_PASS.json")["status"] == "H100_SMOKE_PASS_NOT_FORMAL", "No actual H100 smoke completion")
        elif args.mode == "train_arm":
            child(common + ["--mode", "train", "--arm", args.arm, "--resume"], attempt / "train.log", deadline)
            need(read(goal / "formal" / args.arm / "TRAIN_COMPLETE.json")["status"] == "trained_not_tested",
                 "Independent arm did not complete its full training budget")
        elif args.mode == "evaluate":
            child(common + ["--mode", "evaluate"], attempt / "evaluate.log", deadline)
            need(read(goal / "formal/COMPUTE_COMPLETE.json")["status"] == "four_arm_training_and_testing_complete_not_delivery",
                 "All four selected tests must complete after all four training arms")
        else:
            child(common + ["--mode", "train", "--resume"], attempt / "train.log", deadline)
            child(common + ["--mode", "evaluate"], attempt / "evaluate.log", deadline)
            need(read(goal / "formal/COMPUTE_COMPLETE.json")["status"] == "four_arm_training_and_testing_complete_not_delivery",
                 "Four-arm compute completion is missing")
        write(attempt / "PROCESS_COMPLETE.json", dict(receipt, ended_utc=datetime.now(timezone.utc).isoformat(),
              process_seconds=time.time() - start, platform_release_verified=False))
        print("[STAGE05_JOB_PROCESS_COMPLETE] " + args.mode, flush=True)
    except Exception as exception:
        write(attempt / "PROCESS_FAILED.json", dict(receipt, ended_utc=datetime.now(timezone.utc).isoformat(),
              process_seconds=time.time() - start, error=type(exception).__name__, message=str(exception),
              platform_release_verified=False))
        raise
    finally:
        stop.set()
        monitor.join(timeout=20)


if __name__ == "__main__":
    main()
