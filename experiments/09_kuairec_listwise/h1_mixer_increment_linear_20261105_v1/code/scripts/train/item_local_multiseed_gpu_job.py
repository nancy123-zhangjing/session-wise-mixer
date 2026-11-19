"""One bounded H100 Job for a new Stage07 seed, with no automatic submission.

Smoke runs the seed's A/B pair sequentially. Formal runs one seed/arm to the
fixed training boundary, then evaluates that selected best on val beam32/256.
The 14 GPU-hour cap is this campaign's execution safety cap, not old spending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
CAMPAIGN = "stage07_item_local_joint_multiseed_20261021_r1"
SEEDS = (2025, 2026)
VARIANTS = ("control", "item_local")
CONFIG = "configs/07_item_local_attention/multiseed_20261021.json"
ADAPTER = "scripts/train/run_item_local_joint_multiseed.py"
WRAPPER = "scripts/train/item_local_multiseed_gpu_job.py"
EVALUATOR = "scripts/analysis/evaluate_item_local_multiseed.py"
ENGINE_COMMIT = "1c879252f6395d6e64462511823c3929b9c42c00"
ENGINE_FILES = (
    "genrec/models/DIFF_GRM/model.py", "genrec/models/DIFF_GRM/trainer.py",
    "genrec/models/DIFF_GRM/training_state.py", "genrec/models/DIFF_GRM/joint_cross_pse.py",
    "genrec/models/DIFF_GRM/cross_pse.py", "genrec/models/DIFF_GRM/item_local_attention.py",
    "genrec/models/DIFF_GRM/tokenizer.py", "genrec/models/DIFF_GRM/collate.py",
    "genrec/models/DIFF_GRM/config.yaml", "genrec/default.yaml",
    "genrec/datasets/AmazonReviews2014/config.yaml", "genrec/utils.py", "genrec/pipeline.py",
    "genrec/dataset.py", "genrec/models/DIFF_GRM/beam.py", "genrec/models/DIFF_GRM/evaluator.py",
)
SOURCE_FILES = (ADAPTER, WRAPPER, EVALUATOR, CONFIG, "scripts/train/run_item_local_joint.py",
                "scripts/train/item_local_gpu_job.py", "scripts/train/ocn_repair_gpu_job.py",
                "scripts/repro/stage05_repro.py")
CPU_TESTS = (
    "tests/test_item_local_joint_multiseed.py", "tests/test_item_local_multiseed_evaluation.py",
    "tests/test_item_local_multiseed_gpu_job.py", "tests/test_item_local_alternating_joint.py",
    "tests/test_item_local_joint_training_state.py", "tests/test_ocn_autocast_gradient.py",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def verify_record(row, root=None):
    path = Path(row["path"]).resolve()
    need(root is None or path.is_relative_to(Path(root).resolve()), "Evidence escaped its expected directory")
    need(path.is_file() and path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"],
         f"Evidence bytes differ: {path}")
    return path


def under(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    need(path.is_relative_to(root), "Relative path escaped campaign root")
    return path


def run_id(seed, variant):
    need(seed in SEEDS and variant in VARIANTS, "Unknown seed/arm")
    return f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"


def seed_campaign(seed):
    return f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1"


def run_path(goal, seed, variant):
    return goal / "seeds" / f"seed{seed}" / "runs" / run_id(seed, variant)


def job_key(mode, seed, variant, attempt):
    return f"{mode}_seed{seed}" + (f"_{variant}" if mode == "formal" else "") + f"_attempt{attempt}"


def source_gate(goal, path):
    lock = load(path)
    need(lock["schema"] == "stage07_multiseed_deployment_v1" and lock["campaign_id"] == CAMPAIGN,
         "Wrong deployment lock")
    need(lock["engine_commit"] == ENGINE_COMMIT, "The frozen training engine must be unchanged")
    for name in ("code", "engine"):
        expected = lock[name + "_commit"]
        need(re.fullmatch("[0-9a-f]{40}", expected) is not None, "Expected full frozen Git commit")
        head = subprocess.check_output(["git", "-C", str(goal / name), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(goal / name), "status", "--porcelain"], text=True).strip()
        need(head == expected and not dirty, f"Frozen {name} checkout changed")
    records = lock["files"]
    names = [row["path"] for row in records]
    required = {"code/" + name for name in SOURCE_FILES} | {"engine/" + name for name in ENGINE_FILES}
    need(len(names) == len(set(names)) and required <= set(names), "Deployment omitted required sources/configuration")
    for row in records:
        target = under(goal, row["path"])
        need(target.is_file() and target.stat().st_size == row["size_bytes"] and sha(target) == row["sha256"],
             f"Frozen source bytes differ: {row['path']}")
    need(Path(__file__).resolve() == goal / "code" / WRAPPER, "Run this wrapper from the frozen campaign checkout")
    return lock


def cpu_gate(goal, path, deployment_path):
    gate = load(path)
    need(gate["schema"] == "stage07_multiseed_cpu_gate_v1" and gate["campaign_id"] == CAMPAIGN
         and gate["status"] == "passed" and gate["tests_passed"] is True and gate["cuda_initialized"] is False
         and gate["deployment_lock_sha256"] == sha(deployment_path)
         and gate["config_sha256"] == sha(goal / "code" / CONFIG), "CPU preparation gate differs")
    tests = gate["tests"]
    need(len(tests) == len(CPU_TESTS) and {row["test_file"] for row in tests} == set(CPU_TESTS),
         "CPU gate must bind all six required test files")
    fields = ("status", "test_file", "source_sha256", "tests_run", "skipped", "failures", "errors",
              "cuda_initialized_before", "cuda_initialized_after", "cuda_available")
    for row in tests:
        result = load(verify_record(row["receipt"]))
        verify_record(row["log"])
        need(all(result[field] == row[field] for field in fields), "CPU test receipt summary differs")
        need(row["status"] == "passed" and type(row["tests_run"]) is int and row["tests_run"] > 0
             and row["skipped"] == row["failures"] == row["errors"] == 0
             and row["cuda_initialized_before"] is False and row["cuda_initialized_after"] is False
             and row["cuda_available"] is False and row["source_sha256"] == sha(goal / "code" / row["test_file"]),
             "Required CPU test was skipped, failed or used CUDA")
    rows = gate["prepared_runs"]
    need(len(rows) == 4 and {(r["seed"], r["variant"]) for r in rows} == {(s, v) for s in SEEDS for v in VARIANTS},
         "CPU gate must contain all four unique seed/arm preparations")
    for row in rows:
        work = run_path(goal, row["seed"], row["variant"])
        need(Path(row["output"]).resolve() == work and row["run_id"] == run_id(row["seed"], row["variant"]),
             "CPU prepared output identity differs")
        need({"identity", "resolved_config", "input_identities", "code_identity"} <= set(row["files"]),
             "Missing CPU preparation records")
        for item in row["files"].values():
            verify_record(item, work)
        identity, config, inputs = [load(row["files"][key]["path"]) for key in ("identity", "resolved_config", "input_identities")]
        need(identity["seed"] == row["seed"] and identity["variant"] == row["variant"]
             and identity["campaign_id"] == seed_campaign(row["seed"])
             and identity["parent_campaign_id"] == CAMPAIGN and identity["engine_commit"] == ENGINE_COMMIT,
             "CPU run identity differs")
        need(config["rand_seed"] == row["seed"] and config["patience"] == 0 and config["epochs"] == 45
             and config["vectorized_beam_search"]["val"] == {"beam_act": 32, "beam_max": 32}
             and config["item_local_attention"]["enabled"] == (row["variant"] == "item_local"),
             "CPU resolved training policy differs")
        need(len(inputs["artifacts"]) == 13 and len({r["path"] for r in inputs["artifacts"]}) == 13,
             "CPU inputs must include thirteen distinct locked artifacts")
        for item in inputs["artifacts"]:
            target = under(work / "workspace", item["path"])
            need(target.stat().st_size == item["size_bytes"] and sha(target) == item["sha256"], "CPU private input differs")
    return gate


def budget_gate(path, mode, seed, variant, attempt, maximum, platform, claims):
    budget = load(path)
    need(budget["schema"] == "stage07_multiseed_budget_v1" and budget["campaign_id"] == CAMPAIGN
         and budget["status"] == "passed" and budget["execution_cap_gpu_hours"] == 14
         and budget["budget_scope"] == "execution_safety_cap_not_user_total_budget"
         and budget["automatic_retry"] is False, "Wrong new-campaign execution budget")
    rows, spent = budget["allocations"], budget.get("spent_allocations", [])
    keys = [r["key"] for r in rows + spent]
    need(len(keys) == len(set(keys)), "Duplicate budget reservation")
    total = 0.0
    for row in rows:
        need(row["seed"] in SEEDS and row["mode"] in ("smoke", "formal") and row["gpu_count"] == 1
             and type(row["attempt"]) is int and row["attempt"] >= 1, "Invalid allocation identity")
        need(row["variant"] in VARIANTS if row["mode"] == "formal" else row["variant"] is None, "Wrong allocation arm")
        need(row["key"] == job_key(row["mode"], row["seed"], row["variant"], row["attempt"]), "Wrong reservation key")
        process_cap, allocation_cap = row["max_process_seconds"], row["platform_max_seconds"]
        need(all(type(n) in (int, float) and math.isfinite(n) for n in (process_cap, allocation_cap))
             and process_cap > 90 and allocation_cap >= process_cap + 150, "Allocation/process margins differ")
        limit = 900 if row["mode"] == "smoke" else 10620
        need(allocation_cap <= limit and process_cap <= (750 if row["mode"] == "smoke" else 10440), "Job exceeds its full-work cap")
        total += allocation_cap
    for row in spent:
        need(row["gpu_count"] == 1 and math.isfinite(row["allocated_seconds"]) and row["allocated_seconds"] >= 0,
             "Invalid consumed allocation")
        release = load(verify_record(row["release_evidence"]))
        need(release["status"] == "passed" and release["campaign_id"] == CAMPAIGN
             and release["job_terminal"] is True and release["gpu_released"] is True
             and release["allocated_seconds"] == row["allocated_seconds"], "Consumed budget lacks release/accounting evidence")
        total += row["allocated_seconds"]
    need(total <= 14 * 3600, "Reserved and consumed new allocations exceed the 14 GPU-hour execution cap")
    need(sum(r["platform_max_seconds"] for r in rows if r["mode"] == "smoke")
         + sum(r["allocated_seconds"] for r in spent if r["key"].startswith("smoke_")) <= 1800,
         "Total smoke allocation exceeds 0.5 GPU hour")
    if claims.exists():
        need({p.stem for p in claims.glob("*.json")} <= set(keys), "Budget omitted an already claimed job")
    key = job_key(mode, seed, variant, attempt)
    selected = [r for r in rows if r["key"] == key]
    need(len(selected) == 1 and selected[0]["max_process_seconds"] == maximum
         and selected[0]["platform_max_seconds"] == platform, "This exact Job has no matching budget reservation")
    if attempt == 1:
        initial_keys = {job_key("smoke", s, None, 1) for s in SEEDS} | {job_key("formal", s, v, 1) for s in SEEDS for v in VARIANTS}
        need(initial_keys <= set(keys), "Initial budget must reserve both paired smokes and all four formal runs")
    return {"reserved_and_spent_gpu_seconds": total, "selected": selected[0], "budget_receipt": record(path)}


def training_result(goal, seed, variant, mode):
    work = run_path(goal, seed, variant)
    if mode == "smoke":
        work /= "smoke"
    name = "smoke_results.json" if mode == "smoke" else "results.json"
    result, complete, identity = [load(work / n) for n in (name, "COMPLETE.json", "identity.json")]
    expected_status = "smoke_passed_not_formal" if mode == "smoke" else "formal_training_completed_validation_pending"
    actual_id = run_id(seed, variant) + ("_smoke" if mode == "smoke" else "")
    need(result["status"] == complete["status"] == expected_status and complete["mode"] == mode,
         "Training did not complete the expected mode")
    for value in (result, complete, identity):
        need(value["campaign_id"] == seed_campaign(seed) and value["variant"] == variant and value["run_id"] == actual_id,
             "Training completion identity differs")
    need(identity["seed"] == seed and identity["parent_campaign_id"] == CAMPAIGN
         and identity["study_sha256"] == sha(goal / "code" / CONFIG) and identity["engine_commit"] == ENGINE_COMMIT,
         "Training used a different seed/study/engine")
    adapter = identity["multiseed_adapter"]
    need(Path(adapter["path"]).resolve() == goal / "code" / ADAPTER and adapter["sha256"] == sha(goal / "code" / ADAPTER)
         and adapter["base_runner_sha256"] == sha(goal / "code/scripts/train/run_item_local_joint.py"),
         "Training adapter/base binding differs")
    need(complete["result_file"] == name and complete["result_sha256"] == sha(work / name)
         and complete["checkpoint_sha256"] == result["checkpoint_sha256"] == sha(work / "training_state/best_model.bin"),
         "Training results/best bytes differ")
    need(result["test_data_evaluated"] is False and not any((work / n).exists() for n in ("TEST_STARTED.json", "TEST_RESULT.json")),
         "Test evaluation is forbidden in this campaign")
    timings = load(work / "evaluation_timings.json")
    expected_evaluations = {0, 1, 3} if mode == "smoke" else {0, 5, 12, 19, 25, 32, 39, 45}
    need(all(row["split"] == "val" for row in timings) and {row["epoch"] for row in timings} == expected_evaluations,
         "Unexpected split or missing selection evaluations")
    reload, initial, config = [load(work / n) for n in ("checkpoint_reload.json", "initial_state_identity.json", "parsed_config.json")]
    need(reload["strict_all_tensor_match"] is True and reload["mapping_synchronized_without_requantization"] is True
         and reload["checkpoint_sha256"] == result["checkpoint_sha256"]
         and reload["matching_committed_SID_sha256"] == result["selected_SID_sha256"], "Selected best/SID was not strictly reloaded")
    need(config["rand_seed"] == seed and config["patience"] == 0 and config["vectorized_beam_search"]["val"] == {"beam_act": 32, "beam_max": 32},
         "Training seed/stopping/selection protocol differs")
    need(all(initial[key] == result[key] for key in ("initial_base_state_sha256", "initial_rng_sha256")), "Initial state receipt differs")
    fit, integrity = result["fit"], load(work / "training_integrity.json")
    need(fit["training_state_sha256"] == sha(work / "training_state/last.pt"), "Committed complete training state bytes differ")
    need(integrity == result["training_integrity"] and integrity["status"] == "passed"
         and integrity["global_codebook_mixer_enabled"] is False and integrity["local_block_parameter_side"] == "recommender",
         "Alternating training integrity failed")
    r_updates, t_updates, n_phases, refreshes = (4, 2, 3, 1) if mode == "smoke" else (3640, 60, 45, 10)
    progress, opt = fit["joint_progress"], integrity["optimizer"]
    need(fit["completed_epochs"] == n_phases and fit["training_stop_reason"] == "max_epochs"
         and fit["cumulative_recommender_optimizer_updates"] == r_updates
         and progress["tokenizer_optimizer_updates"] == t_updates and progress["sid_refresh_count"] == refreshes
         and progress["tokenizer_epochs_completed"] == refreshes, "Fixed training quantity was not completed")
    need(opt["status"] == "passed" and opt["all_active_optimizer_steps_match"] is True
         and opt["recommender_optimizer_updates"] == r_updates and opt["tokenizer_optimizer_updates"] == t_updates,
         "Adam clock integrity differs")
    covered = set(opt["covered_parameter_names"])
    need(len(covered) == len(opt["covered_parameter_names"]) == opt["optimizer_state_count"], "Duplicate/missing optimizer states")
    tokenizer = {"joint_alignment.mixer.base_codebooks"}
    recommender = covered - tokenizer
    need(tokenizer <= covered and recommender and not any(n.startswith(("joint_alignment.", "joint_sequence_adapter.")) for n in recommender),
         "Unexpected optimizer parameter ownership")
    local = {n for n in recommender if n.startswith("item_local_attention.")}
    need(bool(local) == (variant == "item_local"), "Local attention ownership differs")
    phases = load(work / "phase_integrity.json")
    need(len(phases) == n_phases and sha(work / "phase_integrity.json") == integrity["phase_records_sha256"], "Phase history binding differs")
    expected_phases = ["recommender", "tokenizer", "recommender"] if mode == "smoke" else ["recommender"] * 5 + ["tokenizer", "recommender", "recommender", "recommender"] * 10
    need([r["phase"] for r in phases] == expected_phases and [r["epoch"] for r in phases] == list(range(1, n_phases + 1)),
         "Alternating schedule differs")
    for phase in phases:
        active = tokenizer if phase["phase"] == "tokenizer" else recommender
        expected_updates = 2 if mode == "smoke" else (6 if phase["phase"] == "tokenizer" else 104)
        changed, calls = set(phase["changed_parameters"]), phase["active_gradient_hook_calls"]
        need(phase["optimizer_updates"] == expected_updates and phase["frozen_parameters_unchanged"] is True
             and phase["all_active_gradients_finite"] is True and phase["active_parameter_tensors"] == len(active)
             and set(calls) == active and all(type(n) is int and n > 0 for n in calls.values())
             and changed and changed <= active, "Missing gradients/updates or frozen-side mutation")
        if phase["phase"] == "tokenizer":
            need(math.isfinite(phase["direct_recommendation_gradient_norm"]) and phase["direct_recommendation_gradient_norm"] > 0
                 and phase["SID_integrity"]["nearest_assignments_verified"] is True, "Tokenizer lacks recommendation gradient/SID refresh")
        elif local:
            need(bool(changed & local), "Local attention was not updated")
        need(phase["SID_integrity"]["catalogue_mapping_matches"] is True, "SID lookup is stale")
    if mode == "smoke":
        recovery = result["recovery_check"]
        need(recovery == load(work / "recovery_probe/recovery_check.json") and recovery["status"] == "passed"
             and recovery["strict_model_optimizer_scheduler_RNG_joint_runtime_restore"] is True
             and recovery["optimizer_updates_executed"] == 1 and recovery["tokenizer_parameters_and_SID_unchanged"] is True,
             "Smoke recovery probe did not pass")
    else:
        need(result["seed"] == seed and result["parent_campaign_id"] == CAMPAIGN
             and result["initial_SID_sha256"] == initial["initial_SID_sha256"]
             and result["training_budget_policy"] == "fixed_45" and result["final_validation_pending"] == [32, 256],
             "Formal identity/evaluation policy differs")
    inputs = load(work / "input_identities.json")["artifacts"]
    need(len(inputs) == 13, "Training input set differs")
    return {"complete": record(work / "COMPLETE.json"), "result": record(work / name),
            "checkpoint": record(work / "training_state/best_model.bin"),
            "selected_SID_sha256": result["selected_SID_sha256"], "fit": fit,
            **{key: initial[key] for key in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256")},
            "input_artifacts": inputs, "phase_checks_passed": True}


def smoke_gate(goal, path, seed, deployment):
    receipt = load(path)
    need(receipt["schema"] == "stage07_multiseed_gpu_job_v1" and receipt["campaign_id"] == CAMPAIGN
         and receipt["mode"] == "smoke" and receipt["seed"] == seed and receipt["status"] == "smoke_completed"
         and receipt["source_lock"]["sha256"] == sha(deployment) and receipt["owned_process_groups_alive_after_cleanup"] == [],
         "Formal requires a completed and cleaned-up same-seed paired smoke")
    summaries = {variant: training_result(goal, seed, variant, "smoke") for variant in VARIANTS}
    need(receipt["paired_smoke"] == summaries, "Paired smoke artifacts changed after completion")
    for key in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256", "input_artifacts"):
        need(summaries["control"][key] == summaries["item_local"][key], "Same-seed paired smoke initial state differs: " + key)
    return summaries


def evaluation_result(run, seed, variant, training):
    path = run / "validation/EVALUATION_COMPLETE.json"
    value = load(path)
    need(value["status"] == "selected_validation_completed" and value["seed"] == seed and value["variant"] == variant
         and value["count_per_cell"] == 22363 and value["model_state_unchanged"] is True
         and value["selected_SID_unchanged"] is True and value["training_gradients_created"] is False
         and value["test_data_evaluated"] is False and value["training_performed"] is False,
         "Final validation pair is incomplete or changed scope")
    need(value["checkpoint_sha256"] == training["checkpoint"]["sha256"]
         and value["selected_SID_sha256"] == training["selected_SID_sha256"], "Final evaluations use another best/SID")
    need(set(value["cells"]) == {"beam32", "beam256"}, "Both final beam settings are required")
    orders = []
    for name, entry in value["cells"].items():
        cell_receipt = load(verify_record(entry["receipt"], run / "validation"))
        need(cell_receipt["status"] == "cell_completed" and cell_receipt["identity_sha256"] == value["identity_sha256"]
             and cell_receipt["beam"] == int(name[4:]), "Final cell identity differs")
        verify_record(cell_receipt["started_receipt"], run / "validation")
        for row in cell_receipt["files"].values():
            verify_record(row, run / "validation")
        metrics = load(cell_receipt["files"]["metrics"]["path"])
        need(entry["files"] == cell_receipt["files"] and entry["metrics"] == metrics,
             "Final pair summary differs from its actual cell evidence")
        need(metrics["count"] == 22363 and metrics["split"] == "val" and metrics["beam_act"] == int(name[4:])
             and metrics["beam_max"] == int(name[4:]) and metrics["seed"] == seed and metrics["variant"] == variant
             and metrics["model_state_unchanged"] is True and metrics["selected_SID_unchanged"] is True
             and metrics["training_gradients_created"] is False and metrics["test_data_evaluated"] is False,
             "Final evaluation state/count/protocol differs")
        need(metrics["checkpoint_sha256"] == training["checkpoint"]["sha256"]
             and metrics["selected_SID_sha256"] == training["selected_SID_sha256"]
             and metrics["best_epoch"] == value["best_epoch"] == training["fit"]["best_epoch"]
             and metrics["model_state_before_sha256"] == metrics["model_state_after_sha256"], "Final cell best/SID differs")
        if name == "beam32":
            need(metrics["original_val32_reproduction"]["status"] == "matched", "Selected validation score did not reproduce")
        orders.append(metrics["user_target_order_sha256"])
    need(len(set(orders)) == 1, "Final beams evaluated different user/target order")
    need(value["original_val32_reproduction"]["status"] == "matched", "Final pair lacks original best-score reproduction")
    return {"receipt": record(path), "cells": value["cells"]}


def release_gate(path, owner, seed, variant, attempt):
    release = load(path)
    old = load(owner)
    previous = verify_record(release["previous_receipt"])
    value = load(previous)
    need(release["campaign_id"] == CAMPAIGN and release["status"] == "passed"
         and release["seed"] == seed and release["variant"] == variant
         and release["job_terminal"] is True and release["gpu_released"] is True
         and release["previous_attempt"] == old["attempt"] < attempt
         and release["previous_receipt"]["sha256"] == sha(old["receipt"])
         and value["seed"] == seed and value["variant"] == variant and value["status"] == "failed_or_incomplete"
         and value["owned_process_groups_alive_after_cleanup"] == [], "Prior run was not safely terminated/released")
    return record(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--deployment-lock", type=Path, required=True)
    parser.add_argument("--cpu-gate", type=Path, required=True)
    parser.add_argument("--budget-reservation", type=Path, required=True)
    parser.add_argument("--smoke-gate", type=Path)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--platform-max-seconds", type=int)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release-evidence", type=Path)
    args = parser.parse_args(argv)
    need(os.name == "posix", "GPU wrapper requires POSIX process-group cleanup")
    need((args.mode == "formal") == (args.variant is not None), "Smoke runs both arms; formal selects exactly one")
    need(args.attempt >= 1 and ((args.attempt == 1 and not args.resume) or
         (args.mode == "formal" and args.attempt > 1 and args.resume and args.release_evidence)), "No automatic retry; recovery requires an explicit new formal attempt")
    maximum = args.max_seconds if args.max_seconds is not None else (750 if args.mode == "smoke" else 10440)
    platform = args.platform_max_seconds if args.platform_max_seconds is not None else (900 if args.mode == "smoke" else 10620)
    goal = args.goal_root.resolve()
    need(goal.is_dir(), "CPU-prepared campaign root is missing")
    key = job_key(args.mode, args.seed, args.variant, args.attempt)
    seed_root = goal / "seeds" / f"seed{args.seed}"
    claims = goal / "delivery/claims"
    budget = budget_gate(args.budget_reservation, args.mode, args.seed, args.variant, args.attempt, maximum, platform, claims)
    claims.mkdir(parents=True, exist_ok=True)
    phase = seed_root / "delivery" / key
    phase.mkdir(parents=True, exist_ok=False)
    receipt_path = phase / "receipt.json"
    with (claims / (key + ".json")).open("x", encoding="utf-8") as handle:
        json.dump({"key": key, "campaign_id": CAMPAIGN, "receipt": str(receipt_path), "reservation": record(args.budget_reservation)}, handle)
    started, monotonic_start = time.time(), time.monotonic()
    deadline = started + maximum - 90
    receipt = {"schema": "stage07_multiseed_gpu_job_v1", "campaign_id": CAMPAIGN,
               "mode": args.mode, "seed": args.seed, "variant": args.variant, "attempt": args.attempt,
               "status": "starting", "pid": os.getpid(), "started_utc": utc(), "started_unix": started,
               "deadline_unix": deadline, "allocated_gpu_count": 1, "max_process_seconds": maximum,
               "platform_max_seconds": platform, "budget": budget, "source_lock": record(args.deployment_lock),
               "cpu_gate": record(args.cpu_gate), "children": [], "test_evaluation_authorized": False,
               "automatic_retry": False, "gpu_release_verified": False,
               "release_scope": "Owned process groups are checked here; actual GPU release requires platform Pod-deletion evidence"}
    write(receipt_path, receipt)
    owner_root = run_path(goal, args.seed, args.variant) if args.mode == "formal" else seed_root
    owner = owner_root / "GPU_OWNER.json"
    active = owner_root / ".active_gpu_wrapper"
    active_acquired = False
    processes, sampler, previous_signals = [], None, {}
    stopped = threading.Event()

    def signal_exit(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    def cleanup():
        live = [row for row in processes if not row.get("cleaned", False)]
        for row in live:
            try: os.killpg(row["process"].pid, signal.SIGTERM)
            except ProcessLookupError: pass
        until = time.monotonic() + 8
        for row in live:
            try: row["process"].wait(timeout=max(.01, until - time.monotonic()))
            except subprocess.TimeoutExpired: pass
        for row in live:
            try: os.killpg(row["process"].pid, signal.SIGKILL)
            except ProcessLookupError: pass
            row["process"].wait(timeout=5)
            row["log"].close()
            row["cleaned"] = True

    def child(label, command, environment):
        need(time.time() < deadline, "No execution time remains for the next step")
        log_path = phase / (label + ".log")
        log = log_path.open("x", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=str(phase), env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close(); raise
        state = {"process": process, "log": log}
        processes.append(state)
        entry = {"label": label, "argv": command, "pid": process.pid, "console": str(log_path),
                 "started_unix": time.time(), "status": "running"}
        receipt["children"].append(entry); write(receipt_path, receipt)
        print(json.dumps({"status": "child_started", "key": key, "step": label}), flush=True)
        while process.poll() is None:
            need(time.time() < deadline + 30 and time.monotonic() - monotonic_start < maximum - 60,
                 "Whole Job execution deadline reached")
            time.sleep(.5)
        entry.update(returncode=process.returncode, ended_unix=time.time(), status="completed" if process.returncode == 0 else "failed")
        cleanup(); write(receipt_path, receipt)
        need(process.returncode == 0, f"Child {label} failed; inspect {log_path}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_signals[signum] = signal.signal(signum, signal_exit)
        source_gate(goal, args.deployment_lock)
        cpu_gate(goal, args.cpu_gate, args.deployment_lock)
        owner_root.mkdir(parents=True, exist_ok=True)
        active.mkdir(exist_ok=False); active_acquired = True
        if owner.exists():
            need(args.resume and args.release_evidence is not None, "This run already had a GPU owner; no duplicate training")
            receipt["previous_release"] = release_gate(args.release_evidence, owner, args.seed, args.variant, args.attempt)
        else:
            need(not args.resume, "No previous owned run exists to recover")
        write(owner, {"campaign_id": CAMPAIGN, "key": key, "seed": args.seed, "variant": args.variant,
                      "attempt": args.attempt, "receipt": str(receipt_path), "pid": os.getpid()})
        sys.path.insert(0, str(goal / "code/scripts/train"))
        from item_local_gpu_job import single_process_environment
        from ocn_repair_gpu_job import gpu_sampler
        environment, removed = single_process_environment(os.environ)
        environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
        receipt["normalized_single_process_variables"] = removed
        import torch
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and "H100" in torch.cuda.get_device_name(0).upper() and torch.cuda.is_bf16_supported(), "Each Job requires exactly one BF16-capable H100")
        summaries = None
        if args.mode == "formal":
            need(args.smoke_gate is not None, "Formal requires the same-seed paired smoke gate")
            summaries = smoke_gate(goal, args.smoke_gate, args.seed, args.deployment_lock)
            receipt["smoke_gate"] = record(args.smoke_gate)
        receipt["gpu_sampling"] = {}
        sampler = threading.Thread(target=gpu_sampler, args=(phase / "gpu_samples.csv", environment, stopped, receipt["gpu_sampling"]), daemon=True)
        sampler.start()
        variants = VARIANTS if args.mode == "smoke" else (args.variant,)
        completed = {}
        for variant in variants:
            run = run_path(goal, args.seed, variant)
            work = run / "smoke" if args.mode == "smoke" else run
            if (work / "COMPLETE.json").exists():
                need(args.resume, "This training run already completed; no duplicate execution")
                completed[variant] = training_result(goal, args.seed, variant, args.mode)
                for field in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256", "input_artifacts"):
                    need(completed[variant][field] == summaries[variant][field], "Completed training differs from accepted smoke")
                receipt["training_reused_after_verified_interruption"] = True
                continue
            if args.resume:
                need((work / "training_state/last.pt").is_file(), "Recovery requires this exact run's committed last.pt")
            else:
                need(not (work / "STARTED.json").exists() and not (work / "training_state/last.pt").exists(), "A previous run needs explicit own-last recovery")
            reserve = 60 if args.mode == "smoke" else 1200
            need(time.time() < deadline - reserve, "Complete training/evaluation no longer fits the reserved execution window")
            command = [sys.executable, "-u", "-B", str(goal / "code" / ADAPTER), "--seed", str(args.seed),
                       "--mode", args.mode, "--config", str(goal / "code" / CONFIG), "--variant", variant,
                       "--output", str(run), "--project-root", args.project_root, "--engine-root", str(goal / "engine"),
                       "--expected-engine-commit", ENGINE_COMMIT, "--deadline-unix", str(deadline),
                       "--training-deadline-reserve-seconds", str(reserve)]
            if args.resume:
                command += ["--resume", str(work / "training_state/last.pt")]
            child("train_" + variant, command, environment)
            completed[variant] = training_result(goal, args.seed, variant, args.mode)
            if summaries is not None:
                for field in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256", "input_artifacts"):
                    need(completed[variant][field] == summaries[variant][field], "Formal initialization differs from accepted same-seed smoke: " + field)
        if args.mode == "smoke":
            for field in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256", "input_artifacts"):
                need(completed["control"][field] == completed["item_local"][field], "Paired smoke initialization differs: " + field)
            receipt["paired_smoke"] = completed
        else:
            run = run_path(goal, args.seed, args.variant)
            receipt["training"] = completed[args.variant]
            command = [sys.executable, "-u", "-B", str(goal / "code" / EVALUATOR), "--run", str(run),
                       "--source-root", str(goal / "code"), "--engine-root", str(goal / "engine"),
                       "--output", str(run / "validation"), "--mode", "evaluate", "--deadline-unix", str(deadline)]
            child("validation_beam32_256", command, environment)
            receipt["evaluation"] = evaluation_result(run, args.seed, args.variant, completed[args.variant])
        source_gate(goal, args.deployment_lock)
        need(sha(args.deployment_lock) == receipt["source_lock"]["sha256"]
             and sha(args.cpu_gate) == receipt["cpu_gate"]["sha256"], "Deployment/CPU evidence changed during Job")
        receipt["status"] = args.mode + "_completed"
    except BaseException as error:
        receipt.update(status="failed_or_incomplete", error={"type": type(error).__name__, "message": str(error)})
    finally:
        for signum in previous_signals:
            signal.signal(signum, signal.SIG_IGN)
        try:
            cleanup()
        except BaseException as error:
            receipt.update(status="failed_or_incomplete", cleanup_error=str(error))
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=6)
        alive = []
        for row in processes:
            try: os.killpg(row["process"].pid, 0); alive.append(row["process"].pid)
            except ProcessLookupError: pass
        receipt["owned_process_groups_alive_after_cleanup"] = alive
        if alive or (sampler is not None and sampler.is_alive()):
            receipt["status"] = "failed_or_incomplete"
        if active_acquired and not alive:
            try:
                active.rmdir()
            except OSError as error:
                receipt.update(status="failed_or_incomplete", owner_cleanup_error=str(error))
        receipt.update(ended_utc=utc(), ended_unix=time.time(), process_wall_seconds=time.monotonic() - monotonic_start)
        write(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "key": key, "receipt": str(receipt_path)}), flush=True)
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
    return 0 if receipt["status"] in ("smoke_completed", "formal_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
