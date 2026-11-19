"""Offline CPU consistency audit of the completed alternating-joint pair.

Requires the original goal/{code,engine,runs,inputs,delivery} layout. Imports
only the hash-bound NEW joint runner/wrapper helpers; never runs a model,
Trainer, recovery, CUDA operation, platform request or historical fixed-SID
audit. PASS describes experimental artifact consistency, not GPU release.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
CAMPAIGN = "stage07_item_local_alternating_joint_seed2024_20261020_r1"
ARMS = ("control", "item_local")
METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10", "weighted_score")
SID_PRECISION_MODES = ("strict_fp32_cpu", "tf32_nearest_even_cpu", "tf32_nearest_ties_away_cpu")


def need(value, message):
    if not value:
        raise ValueError(message)


def round_tf32_operand(value, mode):
    """Round fresh CPU FP32 dot operands to ten fraction bits, not norm inputs."""
    import torch
    need(mode in ("nearest_even", "nearest_ties_away"), "Unknown TF32 rounding mode")
    need(value.device.type == "cpu" and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "TF32 emulation requires finite CPU FP32 operands")
    bits = value.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    remainder = bits & 0x1FFF
    tie = remainder == 0x1000
    increment = (remainder > 0x1000) | (tie & (((bits >> 13) & 1) != 0)) if mode == "nearest_even" else remainder >= 0x1000
    rounded = ((bits & 0xFFFFE000) + increment.to(torch.int64) * 0x2000).to(torch.int32).view(torch.float32)
    need(bool(torch.isfinite(rounded).all()), "TF32 operand rounding overflowed")
    return rounded


def audit_sid_precision(slots, books, codes, chunk_size=2048):
    """Require one exact full-catalogue mode; preserve former FP64 failures."""
    import torch
    need(slots.device.type == books.device.type == codes.device.type == "cpu"
         and slots.dtype == books.dtype == torch.float32 and codes.dtype == torch.int64,
         "SID precision audit requires CPU FP32 features/codebooks and int64 codes")
    need(slots.ndim == books.ndim == 3 and codes.shape == slots.shape[:2] and slots.shape[0] > 0
         and slots.shape[1] == books.shape[0] and slots.shape[2] == books.shape[2] and books.shape[1] >= 2
         and bool(((codes >= 0) & (codes < books.shape[1])).all())
         and bool(torch.isfinite(slots).all() and torch.isfinite(books).all()), "Malformed SID precision inputs")
    summaries = {mode: {"argmin_mismatches": 0, "minimum_top2_margin": None, "maximum_saved_distance_gap": 0.0}
                 for mode in SID_PRECISION_MODES}
    high = {"argmin_mismatches": 0, "old_tolerance_failures": 0, "maximum_saved_distance_gap": 0.0,
            "maximum_old_tolerance_ratio": 0.0, "old_tolerance_formula": "1e-6*(1+abs(saved_distance_fp64))",
            "examples": [], "example_limit": 16}
    quantized_books = {mode: round_tf32_operand(books, mode) for mode in ("nearest_even", "nearest_ties_away")}
    c64 = books.double()
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        for start in range(0, len(slots), chunk_size):
            x, saved = slots[start:start + chunk_size], codes[start:start + chunk_size]
            norm32 = x.square().sum(-1, keepdim=True) + books.square().sum(-1).unsqueeze(0)
            distances = {"strict_fp32_cpu": (norm32 - 2.0 * torch.einsum("bmd,mkd->bmk", x, books)).clamp_min(0)}
            for mode, quantized in quantized_books.items():
                distances["tf32_" + mode + "_cpu"] = (norm32 - 2.0 * torch.einsum(
                    "bmd,mkd->bmk", round_tf32_operand(x, mode), quantized)).clamp_min(0)
            for mode, distance in distances.items():
                need(bool(torch.isfinite(distance).all()), "Nonfinite SID distance arithmetic")
                minimum, argmin = distance.min(-1)
                chosen = distance.gather(-1, saved.unsqueeze(-1)).squeeze(-1)
                margin = distance.topk(2, dim=-1, largest=False).values.diff(dim=-1).squeeze(-1)
                item = summaries[mode]
                item["argmin_mismatches"] += int((argmin != saved).sum())
                item["maximum_saved_distance_gap"] = max(item["maximum_saved_distance_gap"], float((chosen - minimum).max()))
                current_margin = float(margin.min())
                item["minimum_top2_margin"] = current_margin if item["minimum_top2_margin"] is None else min(item["minimum_top2_margin"], current_margin)
            x64 = x.double()
            # Preserve the original auditor's unclamped FP64 formula and threshold.
            d64 = x64.square().sum(-1, keepdim=True) + c64.square().sum(-1).unsqueeze(0) - 2 * torch.einsum("bmd,mkd->bmk", x64, c64)
            need(bool(torch.isfinite(d64).all()), "Nonfinite FP64 SID distance arithmetic")
            minimum64, argmin64 = d64.min(-1)
            chosen64 = d64.gather(-1, saved.unsqueeze(-1)).squeeze(-1)
            gap64, tolerance = chosen64 - minimum64, 1e-6 * (1 + chosen64.abs())
            high["argmin_mismatches"] += int((argmin64 != saved).sum())
            high["old_tolerance_failures"] += int((gap64 > tolerance).sum())
            high["maximum_saved_distance_gap"] = max(high["maximum_saved_distance_gap"], float(gap64.max()))
            high["maximum_old_tolerance_ratio"] = max(high["maximum_old_tolerance_ratio"], float((gap64 / tolerance).max()))
            for row, digit in (argmin64 != saved).nonzero().tolist():
                if len(high["examples"]) < high["example_limit"]:
                    high["examples"].append({"catalogue_row_zero_based": start + row, "digit_zero_based": digit,
                        "saved_code": int(saved[row, digit]), "fp64_argmin": int(argmin64[row, digit]),
                        "saved_distance_fp64": float(chosen64[row, digit]), "minimum_distance_fp64": float(minimum64[row, digit]),
                        "gap_fp64": float(gap64[row, digit]), "old_tolerance": float(tolerance[row, digit])})
    matching = [mode for mode in SID_PRECISION_MODES if summaries[mode]["argmin_mismatches"] == 0]
    need(matching, "No single exact full-catalogue SID arithmetic mode matches: " + json.dumps(summaries, sort_keys=True))
    high["examples_truncated"] = high["argmin_mismatches"] > len(high["examples"])
    return {"code_positions": codes.numel(), "matching_modes": matching, "modes": summaries, "strict_fp64_reference": high,
            "arithmetic": "Norms use original FP32 inputs; TF32 variants round only dot operands to 10 fraction bits, accumulate on CPU in FP32, then clamp_min(0). Exact first argmin required.",
            "scope": "CPU arithmetic consistency, not historical CUDA flags or a bitwise CUDA replay; no distance tolerance grants acceptance."}


def common_sid_precision_modes(reports):
    need(set(reports) == {arm + "/" + kind for arm in ARMS for kind in ("best", "last")}, "Need exactly four named checkpoint precision reports")
    for report in reports.values():
        need(report["matching_modes"] and set(report["matching_modes"]) <= set(SID_PRECISION_MODES)
             and set(report["matching_modes"]) == {mode for mode in SID_PRECISION_MODES if report["modes"][mode]["argmin_mismatches"] == 0}, "Malformed per-state precision acceptance")
    common = [mode for mode in SID_PRECISION_MODES if all(mode in report["matching_modes"] for report in reports.values())]
    need(common, "Four checkpoints have no common exact SID arithmetic mode; per-state/per-position mode mixing is forbidden")
    return common


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    need(not isinstance(value, bool), "Boolean is not a numeric observation")
    value = float(value)
    need(math.isfinite(value), "Nonfinite numeric observation")
    return value


def integer(value):
    result = number(value)
    need(result.is_integer(), "Nonintegral count")
    return int(result)


def close(left, right):
    return math.isclose(number(left), number(right), rel_tol=1e-7, abs_tol=1e-9)


def metrics(value):
    value = {key: number(item) for key, item in value.items()}
    for k in (5, 10):
        if f"recall@{k}" in value:
            need(f"hit_rate@{k}" not in value or close(value[f"hit_rate@{k}"], value[f"recall@{k}"]), "HR/recall alias mismatch")
            value.setdefault(f"hit_rate@{k}", value[f"recall@{k}"])
    need(all(name in value and 0 <= value[name] <= 1 for name in METRICS), "Missing/out-of-range SID metric")
    need(abs(value["weighted_score"] - .8 * value["ndcg@10"] - .2 * value["hit_rate@10"]) <= 1e-7,
         "Weighted score formula mismatch")
    return {name: value[name] for name in METRICS}


def phase(epoch_index):
    if epoch_index < 5:
        return "recommender", -1
    index = (epoch_index - 5) // 4
    return ("tokenizer" if epoch_index >= 5 and index < 10 and (epoch_index - 5) % 4 == 0 else "recommender", index)


def selection(values, count):
    """Recompute the original initial-validation / patience rule, not a new rule."""
    need(1 <= count <= 45, "Logical epoch count outside the original 45-epoch protocol")
    expected = [0]
    rec_epochs, tokenizer_epochs = 0, 0
    for epoch in range(1, count + 1):
        if phase(epoch - 1)[0] == "recommender":
            rec_epochs += 1
            if rec_epochs % 5 == 0:
                expected.append(epoch)
        else:
            tokenizer_epochs += 1
    need(sorted(values) == expected, "Validation epochs differ from initial plus every five recommender epochs")
    best_epoch, best, patience, stopped = 0, number(values[0]), 0, None
    for epoch in expected[1:]:
        improved = number(values[epoch]) > best
        if improved:
            best_epoch, best = epoch, number(values[epoch])
        updates = sum(phase(index)[0] == "recommender" for index in range(epoch)) * 104
        active = updates >= 500
        patience = 0 if improved or not active else patience + 1
        if active and patience >= 5:
            stopped = epoch
            break
    need(count == (stopped if stopped is not None else 45), "Incomplete/truncated training is not a completed original protocol")
    return dict(completed_epochs=count, best_epoch=best_epoch, best_val_score=best,
                stop_reason="early_stopping" if stopped is not None else "max_epochs",
                eval_count=len(expected), no_improve_count=patience,
                decoder_epochs_completed=rec_epochs, tokenizer_epochs_completed=tokenizer_epochs,
                recommender_updates=rec_epochs * 104, tokenizer_updates=tokenizer_epochs * 6)


def difference(control, local):
    a, b = metrics(control), metrics(local)
    changes = {key: {"control": a[key], "item_local": b[key], "absolute_delta": b[key] - a[key],
                     "relative_delta": (b[key] / a[key] - 1) if a[key] else None} for key in METRICS}
    signal = a["ndcg@10"] > 0 and b["ndcg@10"] >= 1.03 * a["ndcg@10"] and b["hit_rate@10"] >= a["hit_rate@10"]
    return {"metrics": changes, "exploratory_positive_signal": signal,
            "signal_rule": "relative NDCG@10 >=3% and HR@10 does not decrease; same-seed SID-level pair, no significance claim"}


def mapped(run, original, *, file=True):
    value = str(original).replace("\\", "/")
    anchor = "/" + run.name + "/"
    need(value.count(anchor) == 1, "Artifact path does not identify this formal run")
    relative = PurePosixPath(value.split(anchor, 1)[1])
    need(not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] not in ("smoke", "profile"), "Unsafe/nonformal artifact path")
    result = (run / str(relative)).resolve()
    need(result.is_relative_to(run.resolve()) and (result.is_file() if file else result.is_dir()), "Missing/mislocated formal artifact")
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def metric_groups(path):
    groups = {}
    for row in rows(path):
        key = row["split"], integer(row["epoch"])
        group = groups.setdefault(key, {})
        need(row["metric"] not in group, "Duplicated committed metric CSV observation")
        group[row["metric"]] = number(row["value"])
    return {key: metrics(value) for key, value in groups.items()}


def normalize_config(value, identifier):
    ignored = {"device", "run_local_time", "resume_training_state", "training_deadline_unix_seconds",
               "training_deadline_reserve_seconds", "item_local_attention", "variant"}
    if isinstance(value, dict):
        return {key: normalize_config(item, identifier) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [normalize_config(item, identifier) for item in value]
    return value.replace(identifier, "<RUN>") if isinstance(value, str) else value


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def smoke_receipt_binding(goal, formal_receipt, binding, smoke, wrapper):
    evidence = formal_receipt["smoke_evidence"]
    original = str(evidence["path"]).replace("\\", "/")
    need(original.count("/delivery/") == 1, "Smoke evidence is outside the owned delivery tree")
    relative = PurePosixPath(original.split("/delivery/", 1)[1])
    need(not relative.is_absolute() and ".." not in relative.parts, "Unsafe smoke evidence path")
    path = goal / "delivery" / str(relative)
    receipt = load(path)
    need(receipt["schema"] == "stage07_item_local_joint_job_phase_v1"
         and sha(path) == evidence["sha256"] and evidence["paired_smoke_checks"] == smoke
         and receipt["source_binding"] == binding and receipt["campaign_id"] == CAMPAIGN
         and receipt["mode"] == "smoke" and receipt["status"] == "smoke_completed"
         and receipt["owned_process_groups_alive_after_cleanup"] == []
         and receipt["stdout_reader_alive_after_cleanup"] is False and receipt["paired_smoke_checks"] == smoke,
         "Formal used an unbound/unsuccessful same-source paired smoke")
    children = receipt["children"]
    need([child["variant"] for child in children] == list(ARMS), "Paired smoke children differ")
    for child, row in zip(children, wrapper.study_rows(goal)):
        _, _, hashes = wrapper.completion(row, "smoke", binding)
        need(child["status"] == "completed" and child["returncode"] == 0 and child["resume"] is False
             and child["run_id"] == row["run_id"] and child["study_sha256"] == row["study_sha256"]
             and str(child["output"]).replace("\\", "/").endswith("/runs/" + row["run_id"])
             and all(child[key] == value for key, value in hashes.items()), "Actual paired smoke child binding differs")
    need(number(children[0]["ended_unix"]) <= number(children[1]["started_unix"]), "Smoke children overlapped")


def audit_weights(run, config, payload, result, expected, runner, recipe):
    import numpy as np
    import torch
    best = torch.load(run / "training_state/best_model.bin", map_location="cpu", weights_only=False)
    last, runtime = payload["model"], payload["joint_runtime"]
    workspace = run / "workspace"
    offline = torch.load(workspace / recipe["initial_decoder"], map_location="cpu", weights_only=False)
    token = torch.load(workspace / recipe["initial_tokenizer_dir"] / "best.pt", map_location="cpu", weights_only=False)
    token = token.get("model", token)
    local_shapes = {"norm.weight": (256,), "norm.bias": (256,), "attention.in_proj_weight": (768, 256),
                    "attention.in_proj_bias": (768,), "attention.out_proj.weight": (256, 256), "attention.out_proj.bias": (256,)}
    adapters = {f"joint_sequence_adapter.{layer}.{suffix}": ((256, 256) if suffix == "weight" else (256,))
                for layer in (0, 2) for suffix in ("weight", "bias")}
    local = {"item_local_attention." + key: shape for key, shape in local_shapes.items()} if result["variant"] == "item_local" else {}
    buffers = {"joint_alignment.item_slots", "joint_alignment.hard_codes", "joint_alignment.diger_optimizer_step", "joint_alignment.diger_total_optimizer_steps"}
    shapes = {key: tuple(value.shape) for key, value in offline.items()}
    shapes.update({"joint_alignment.mixer." + key: tuple(value.shape) for key, value in token.items()})
    shapes.update(adapters); shapes.update(local)
    catalogue = load(workspace / "data/cache/AmazonReviews2014/Beauty/processed/id_mapping.json")["id2item"]
    item_ids = {item: index for index, item in enumerate(catalogue)}
    item_order = load(workspace / "data/diff_formal_embeddings/beauty_item_keys.json")
    raw_slots = np.load(workspace / recipe["initial_tokenizer_dir"] / "opq_residual_embeddings.npy", allow_pickle=False)
    slots_reference = torch.zeros(12102, 4, 64, dtype=torch.float32)
    slots_reference[[item_ids[item] for item in item_order]] = torch.from_numpy(raw_slots.reshape(12101, 4, 64))
    need(set(best) == set(last) == set(shapes) | buffers, "Checkpoint architecture differs from locked warm start plus joint buffers/local block")
    sid_precision = {}
    for state_kind, state in (("best", best), ("last", last)):
        for key, value in state.items():
            need(torch.is_tensor(value) and value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Nonfinite/non-CPU checkpoint tensor")
            if key in shapes:
                need(tuple(value.shape) == shapes[key], "Checkpoint shape mismatch: " + key)
        for key, value in token.items():
            if key != "base_codebooks":
                need(torch.equal(state["joint_alignment.mixer." + key], value), "Disabled global Mixer changed")
        for key in adapters:
            need(torch.equal(best[key], last[key]), "Disabled SIA adapter changed during training")
        codes, books, slots = (state["joint_alignment." + key] for key in ("hard_codes", "mixer.base_codebooks", "item_slots"))
        need(tuple(codes.shape) == (12102, 4) and codes.dtype == torch.int64 and bool((codes[0] == -1).all())
             and bool(((codes[1:] >= 0) & (codes[1:] < 256)).all()), "Invalid committed SID catalogue")
        need(tuple(books.shape) == (4, 256, 64) and tuple(slots.shape) == (12102, 4, 64), "Invalid quantizer geometry")
        need(torch.equal(slots, slots_reference), "Committed feature slots differ from the locked item features/catalogue order")
        sid_precision[state_kind] = audit_sid_precision(slots[1:], books, codes[1:])
    need(tensor_sha(best["joint_alignment.hard_codes"][1:]) == result["selected_SID_sha256"], "Selected best checkpoint SID mismatch")
    need(tensor_sha(last["joint_alignment.hard_codes"][1:]) == runtime["hard_codes_sha256"], "Terminal runtime SID mismatch")
    need(int(last["joint_alignment.diger_optimizer_step"]) == expected["tokenizer_updates"]
         and int(last["joint_alignment.diger_total_optimizer_steps"]) == 60, "Terminal DIGER schedule steps differ")
    need(runtime["version"] == 1 and runtime["phase"] == "recommender"
         and runtime["diger_schedule"]["optimizer_step"] == expected["tokenizer_updates"]
         and runtime["diger_schedule"]["total_optimizer_steps"] == 60, "Committed joint runtime/phase/DIGER clock differs")
    selected_tok = sum(phase(index)[0] == "tokenizer" for index in range(expected["best_epoch"])) * 6
    need(int(best["joint_alignment.diger_optimizer_step"]) == selected_tok, "Best checkpoint has the wrong tokenizer clock")
    base = torch.from_numpy(np.load(workspace / recipe["initial_tokenizer_dir"] / "base_pse_codebooks.npy", allow_pickle=False))
    need(torch.equal(runtime["initial_codebooks"], base), "Nonpersistent anchor was not preserved")
    need(torch.equal(runtime["recommender_codebook_cache"], last["joint_alignment.mixer.base_codebooks"]), "Frozen recommender codebook cache is stale")
    layout = runtime["optimizer_groups"]
    rec_names, adapter_names = set(offline) | set(local), set(adapters)
    need([row["group_name"] for row in layout] == ["recommender", "joint_codebook"]
         and layout[0]["parameters"] == list(offline) + list(adapters) + list(local)
         and layout[1]["parameters"] == ["joint_alignment.mixer.base_codebooks"], "Actual optimizer parameter-name manifest differs")
    all_parameters = rec_names | adapter_names | {"joint_alignment.mixer." + key for key in token if key != "residual_scale"}
    need(set(runtime["requires_grad"]) == all_parameters
         and all(value is (name in rec_names | adapter_names) for name, value in runtime["requires_grad"].items()), "Terminal requires-grad layout differs from recommender boundary")
    proxy = SimpleNamespace(named_parameters=lambda: [(name, last[name]) for group in layout for name in group["parameters"]])
    checked = runner.optimizer_integrity(payload, proxy)
    need(checked == result["training_integrity"]["optimizer"], "Adam integrity receipt disagrees with actual checkpoint")
    need(payload["scheduler"]["last_epoch"] == expected["recommender_updates"] and payload["scaler"] is None, "Scheduler/BF16 scaler state mismatch")
    rng = payload["rng"]
    need(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda_all", "loader_counters", "loader_generators"}
         and isinstance(rng["torch_cuda_all"], list) and len(rng["torch_cuda_all"]) == 1, "Incomplete one-GPU RNG checkpoint")
    for value in [rng["torch_cpu"], *rng["torch_cuda_all"], *rng["loader_generators"].values()]:
        need(value.device.type == "cpu" and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0, "Malformed RNG state bytes")
    return best, last, rec_names, checked, sid_precision


def audit_arm(row, binding, runner, wrapper, state_helper, recipe, formal, layer_hashes, smoke):
    import torch
    run, result, hashes = wrapper.completion(row, "formal", binding)
    doc = lambda name: load(run / name)
    config, initial, integrity = doc("parsed_config.json"), doc("initial_state_identity.json"), doc("training_integrity.json")
    need(doc("recipe.json") == recipe and doc("study.json") == runner.validate_study(load(Path(row["config"]))), "Recipe/study changed")
    runtime_keys = {"run_id", "run_local_time", "joint_recommender_checkpoint", "joint_crosspse_checkpoint",
                    "resume_training_state", "training_deadline_unix_seconds", "training_deadline_reserve_seconds",
                    "training_state_identity", "item_local_attention"}
    for key, value in formal.items():
        if not key.endswith(("_path", "_dir")) and key not in runtime_keys:
            need(config.get(key) == value, "Formal setting changed: " + key)
    tokenizer_dir = "workspace/" + recipe["initial_tokenizer_dir"]
    for key, relative in {"joint_recommender_checkpoint": "workspace/" + recipe["initial_decoder"],
                          "joint_crosspse_checkpoint": tokenizer_dir + "/best.pt",
                          "joint_crosspse_slots_path": tokenizer_dir + "/opq_residual_embeddings.npy",
                          "precomputed_semantic_ids_path": tokenizer_dir + "/semantic_ids.json",
                          "crosspse_summary_path": tokenizer_dir + "/summary.json",
                          "joint_crosspse_items_path": "workspace/data/diff_formal_embeddings/beauty_item_keys.json"}.items():
        need(mapped(run, config[key]) == (run / relative).resolve(), "Wrong isolated warm-start input path: " + key)
    need(mapped(run, config["cache_dir"], file=False) == (run / "workspace/data/cache").resolve(), "Dataset cache escaped the isolated workspace")
    need(config["item_local_attention"] == {"enabled": row["variant"] == "item_local", "num_heads": 4, "dropout": 0.0}
         and config["max_eval_batches"] in (None, 0) and config["max_train_batches_per_epoch"] in (None, 0)
         and config.get("use_ddp") is False, "Wrong local variant/truncation/distributed mode")
    wrapper.formal_matches_smoke(row, result, {"paired_smoke_checks": smoke})
    need(initial["resume_requested"] is False and initial["campaign_id"] == CAMPAIGN and initial["variant"] == row["variant"], "Initial warm start is not fresh")
    lock_path = Path(row["config"]).parents[2] / "experiments/05_crosspse_joint/reproducibility/assets.lock.json"
    lock, inputs = load(lock_path)["artifacts"], doc("input_identities.json")
    need(inputs["assets_lock_sha256"] == sha(lock_path) and len(inputs["artifacts"]) == 13
         and {item["path"] for item in inputs["artifacts"]} == set(recipe["required_artifacts"]), "Missing/changed thirteen input identities")
    for item in inputs["artifacts"]:
        need(item["sha256"] == lock[item["path"]]["sha256"], "Input lock mismatch")
        for directory in (run / "workspace", run.parent.parent / "inputs", run / "smoke/workspace"):
            path = runner.stage05.under(directory, item["path"])
            need(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "Actual locked input bytes differ")
    import numpy as np
    catalogue = load(run / "workspace/data/cache/AmazonReviews2014/Beauty/processed/id_mapping.json")["id2item"][1:]
    sid_json = load(run / "workspace" / recipe["initial_tokenizer_dir"] / "semantic_ids.json")
    need(hashlib.sha256(np.asarray([sid_json[item] for item in catalogue], dtype=np.int64).tobytes()).hexdigest()
         == initial["initial_SID_sha256"], "Initial SID digest does not match the actual locked catalogue")
    executions = sorted((run / "attempts").glob("*/STARTED.json"))
    need(executions and doc("STARTED.json") == load(executions[0]), "Missing initial real execution binding")
    for index, start_path in enumerate(executions):
        start, attempt = load(start_path), start_path.parent
        need(start["mode"] == "formal" and start["campaign_id"] == CAMPAIGN and start["variant"] == row["variant"]
             and start["resume"] is bool(index), "Unexpected fresh restart or wrong actual attempt")
        code, actual = load(attempt / "code_identity.json"), load(attempt / "parsed_config.json")
        need(code["engine"]["head"] == binding["engine_commit"] and code["engine"]["tracked_dirty"] is False
             and code["engine_sources"] == binding["engine_files"] and code["runner_sha256"] == binding["runner_sha256"]
             and code["stage05_helper_sha256"] == binding["stage05_helper_sha256"]
             and [item["sha256"] for item in code["configuration_layers"]] == layer_hashes, "Actual attempt source mismatch")
        need(normalize_config(actual, row["run_id"]) == normalize_config(config, row["run_id"]), "Recipe drift across attempts")
        environment = load(attempt / "environment.json")
        need(environment["visible_gpus"] == 1 and "H100" in environment["gpu"].upper(), "Actual attempt did not record one H100")
        for field in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256"):
            need(load(attempt / "initial_state_identity.json")[field] == initial[field], "Warm-start/RNG changed between attempts")
    payload = torch.load(run / "training_state/last.pt", map_location="cpu", weights_only=False)
    need(payload["format"] == "diffgrm_joint_phase_boundary_v1" and payload["progress"]["training_complete"] is True, "Formal joint fit is unfinished")
    p, jp = payload["progress"], payload["progress"]["joint_progress"]
    need(payload["configuration_fingerprint"] == state_helper.configuration_fingerprint(config), "Actual committed configuration fingerprint differs")
    curves = rows(mapped(run, p["output_paths"]["training_metrics_csv"]))
    groups = metric_groups(mapped(run, p["output_paths"]["evaluation_metrics_csv"]))
    expected = selection({epoch: value["weighted_score"] for (split, epoch), value in groups.items() if split == "val"}, len(curves))
    need(set(groups) == {("val", epoch) for epoch in [0] + [i for i in range(1, len(curves) + 1) if phase(i - 1)[0] == "recommender"
                           and sum(phase(j)[0] == "recommender" for j in range(i)) % 5 == 0]} | {("test", 0)}, "Unexpected evaluation CSV groups")
    for key in ("best_epoch", "best_val_score", "eval_count", "no_improve_count", "decoder_epochs_completed", "stop_reason"):
        need(close(p[key], expected[key]) if key == "best_val_score" else p[key] == expected[key], "Terminal selection/progress mismatch: " + key)
    need(p["next_epoch"] == expected["completed_epochs"] and p["cumulative_optimizer_updates"] == expected["recommender_updates"], "Terminal epoch/rec clock mismatch")
    need(jp["tokenizer_epochs_completed"] == jp["sid_refresh_count"] == expected["tokenizer_epochs_completed"]
         and jp["tokenizer_optimizer_updates"] == expected["tokenizer_updates"] and jp["initial_validation_complete"] is True
         and jp["last_committed_phase"] == jp["boundary_phase"] == phase(len(curves) - 1)[0]
         and (jp["next_phase"], jp["next_phase_index"]) == phase(len(curves)), "Committed alternating/DIGER/refresh progress mismatch")
    need(integrity == result["training_integrity"] and integrity["status"] == "passed"
         and integrity["global_codebook_mixer_enabled"] is False and integrity["local_block_parameter_side"] == "recommender", "Invalid training integrity")
    best, last, rec_names, adam, sid_precision = audit_weights(run, config, payload, result, expected, runner, recipe)
    phases, loops, phase_csv = doc("phase_integrity.json"), doc("epoch_timings.json"), rows(run / "phase_integrity.csv")
    need(len(phases) == len(loops) == len(curves) == len(phase_csv) and sha(run / "phase_integrity.json") == result["phase_integrity_sha256"] == integrity["phase_records_sha256"], "Invalid phase inventory")
    need(loops == p["epoch_timing_records"], "Terminal timing records differ from the published phase curves")
    cumulative, sid = 0, initial["initial_SID_sha256"]
    total_rec_updates = sum(phase(index)[0] == "recommender" for index in range(45)) * 104
    def learning_rate(step):
        factor = max(step, 1) / 500 if step < 500 else .5 * (1 + math.cos(math.pi * min(max((step - 500) / (total_rec_updates - 500), 0), 1)))
        return config["lr"] * factor
    for epoch, (event, curve, loop) in enumerate(zip(phases, curves, loops), 1):
        kind = phase(epoch - 1)[0]; active = {"joint_alignment.mixer.base_codebooks"} if kind == "tokenizer" else rec_names
        updates, batches = (6, 96) if kind == "tokenizer" else (104, 1652)
        before_updates = cumulative
        cumulative += updates if kind == "recommender" else 0
        need(event["epoch"] == integer(curve["epoch"]) == loop["epoch"] == epoch and event["phase"] == kind
             and event["optimizer_updates"] == integer(curve["optimizer_updates"]) == updates
             and integer(curve["cumulative_optimizer_updates"]) == cumulative and loop["batches"] == batches
             and number(loop["seconds"]) > 0 and number(curve["train_loss"]) >= 0, "Incomplete/wrong phase training budget")
        need(close(curve["lr_start"], learning_rate(before_updates)) and close(curve["lr_end"], learning_rate(cumulative)), "Alternating recommender LR schedule differs")
        changed, calls = set(event["changed_parameters"]), event["active_gradient_hook_calls"]
        need(all(phase_csv[epoch - 1][column] == ("" if event[column] is None else str(event[column])) for column in runner.PHASE_COLUMNS), "Phase CSV/JSON mismatch")
        need(changed and len(changed) == event["changed_parameter_tensors"] == len(event["changed_parameters"])
             and changed <= active and set(calls) == active and all(integer(value) > 0 for value in calls.values())
             and event["active_parameter_tensors"] == len(active) and event["frozen_parameters_unchanged"] is True
             and event["frozen_parameter_tensors"] == len(payload["joint_runtime"]["requires_grad"]) - len(active)
             and event["all_active_gradients_finite"] is True, "Invalid phase freezing/actual gradient/update evidence")
        need(event["SID_integrity"]["hard_codes_sha256"] == event["sid_sha256"]
             and event["SID_integrity"]["catalogue_mapping_matches"] is True, "Phase SID mapping mismatch")
        if kind == "tokenizer":
            need(number(event["direct_recommendation_gradient_norm"]) > 0 and event["SID_integrity"]["nearest_assignments_verified"] is True, "No direct recommendation gradient / refreshed SID proof")
        else:
            need(event["sid_sha256"] == sid, "Recommender phase changed frozen SID")
            need(row["variant"] == "control" or any(name.startswith("item_local_attention.") for name in changed), "Local attention never updated in recommender phase")
        sid = event["sid_sha256"]
    need(sid == payload["joint_runtime"]["hard_codes_sha256"], "Terminal SID differs from final committed phase")
    need(result["selected_SID_sha256"] == (initial["initial_SID_sha256"] if expected["best_epoch"] == 0 else phases[expected["best_epoch"] - 1]["sid_sha256"]), "Best weights/SID belong to different phases")
    fit = result["fit"]
    need(fit["completed_epochs"] == expected["completed_epochs"] and fit["best_epoch"] == expected["best_epoch"]
         and close(fit["best_val_score"], expected["best_val_score"]) and fit["joint_progress"] == jp, "Result fit receipt disagrees with actual committed state")
    need(fit["training_state_sha256"] == sha(run / "training_state/last.pt"), "Result fit receipt does not bind the actual terminal checkpoint bytes")
    root_fit = doc("FIT_COMPLETE.json")
    need({key: value for key, value in root_fit.items() if key != "training_state_sha256"}
         == {key: value for key, value in fit.items() if key != "training_state_sha256"}, "Root fit receipt differs beyond legal terminal serialization")
    need(hashlib.sha256(payload["best_checkpoint_bytes"]).hexdigest() == payload["best_checkpoint_sha256"] == hashes["checkpoint_sha256"], "Embedded best bytes mismatch")
    test = doc("TEST_RESULT.json"); runner.validate_test_snapshots(test, run)
    for snapshot in test["diagnostic_snapshots"]:
        need(sha(runner.stage05.under(run, snapshot["original_relative_path"])) == snapshot["sha256"], "Once-only test CSV tail was lost on recovery")
    observed = groups[("test", 0)]
    for item in (result["test_results"], test["test_results"]):
        need(all(close(metrics(item)[key], observed[key]) for key in METRICS), "CSV/JSON test results disagree")
    position = [item for item in rows(mapped(run, p["output_paths"]["position_metrics_csv"])) if item["split"] == "test"]
    need(len(position) == 4 and {integer(item["position"]) for item in position} == set(range(4))
         and all(integer(item["supervised_tokens"]) == integer(item["available_views"]) == 22363 for item in position), "Final test was truncated or repeated")
    timings = doc("evaluation_timings.json"); tests = [item for item in timings if item["split"] == "test"]
    need(len(tests) == 1 and timings[-1] == tests[0] and tests[0]["epoch"] is None, "Final test was absent, repeated or premature")
    need(all(item["attempt"] in {path.parent.name for path in executions} and number(item["seconds"]) > 0 for item in timings), "Timing refers to unknown execution")
    val_calls = [item for item in timings if item["split"] != "test"]
    expected_vals = {epoch for split, epoch in groups if split == "val"}
    need(all(item["split"] == "val" and item["epoch"] in expected_vals for item in val_calls)
         and {item["epoch"] for item in val_calls} == expected_vals, "Unplanned/missing validation timing")
    for epoch in expected_vals:
        repeats = [item["attempt"] for item in val_calls if item["epoch"] == epoch]
        need(len(repeats) == len(set(repeats)), "Validation repeated within a single execution attempt")
    ready = load(run / "attempts" / tests[0]["attempt"] / "FINAL_EVALUATION_READY.json")
    start, complete = doc("TEST_STARTED.json"), doc("COMPLETE.json")
    stamps = [datetime.fromisoformat(value) for value in (ready["utc"], start["utc"], test["completed_utc"], complete["completed_utc"])]
    need(ready["training_terminal"] is True and stamps == sorted(stamps), "Terminal-to-test chronology differs")
    metadata = load(mapped(run, p["output_paths"]["run_metadata_path"]))
    need(metadata["joint_crosspse_enabled"] is True and metadata["crosspse_mixer_enabled"] is False
         and metadata["evaluation_level"] == "sid" and metadata["best_epoch"] == expected["best_epoch"]
         and close(metadata["best_val_score"], expected["best_val_score"])
         and all(close(metrics(metadata["test_results"])[key], observed[key]) for key in METRICS), "Final metadata differs from actual joint selection/test")
    need(np.array_equal(np.load(mapped(run, metadata["joint_final_semantic_ids_path"]), allow_pickle=False), best["joint_alignment.hard_codes"][1:].numpy()), "Exported selected SID differs from best checkpoint")
    successful = []
    for receipt_path in (run.parent.parent / "delivery").glob(f"formal_{row['variant']}_attempt*/receipt.json"):
        receipt = load(receipt_path)
        if receipt.get("status") == "formal_completed":
            need(receipt["schema"] == "stage07_item_local_joint_job_phase_v1"
                 and receipt["source_binding"] == binding and len(receipt["children"]) == 1
                 and receipt["campaign_id"] == CAMPAIGN and receipt["mode"] == "formal" and receipt["variant"] == row["variant"], "Formal Job wrapper source/child binding differs")
            child = receipt["children"][0]
            need(child["status"] in ("completed", "skipped_complete") and child["returncode"] == 0
                 and child["run_id"] == row["run_id"] and child["study_sha256"] == row["study_sha256"]
                 and str(child["output"]).replace("\\", "/").endswith("/runs/" + row["run_id"]),
                 "Successful formal wrapper contradicts child status/exit code")
            smoke_receipt_binding(run.parent.parent, receipt, binding, smoke, wrapper)
            if child["variant"] == row["variant"] and all(child[key] == value for key, value in hashes.items()):
                successful.append(sha(receipt_path))
    need(successful, "No successful formal wrapper receipt binds these exact final result bytes")
    need(result["historical_comparison_role"] == "Stage03/05 results are background only", "Historical metrics substituted for the paired control")
    return {"variant": row["variant"], "metrics": observed, "selection": expected, "initial": initial,
            "config": normalize_config(config, row["run_id"]), "checkpoint_hashes": hashes,
            "actual_Adam_state_count": adam["optimizer_state_count"], "phase_count": len(phases), "formal_wrapper_receipts": successful,
            "SID_precision": sid_precision}


def audit(args):
    goal = args.goal_root.resolve()
    for arm in ARMS:
        need((goal / "runs" / f"stage07_item_local_joint_{arm}_seed2024_20261020_r1" / "COMPLETE.json").is_file(), "Both true formal runs must be complete; smoke/preparation is not acceptance")
    need(re.fullmatch("[0-9a-f]{40}", args.root_commit), "A full frozen root commit is required")
    for relative in ("scripts/train/item_local_joint_gpu_job.py", "scripts/train/run_item_local_joint.py"):
        blob = subprocess.check_output(["git", "-C", str(goal / "code"), "show", args.root_commit + ":" + relative])
        need(hashlib.sha256(blob).hexdigest() == sha(goal / "code" / relative), "Frozen helper bytes differ before import")
    wrapper = module(goal / "code/scripts/train/item_local_joint_gpu_job.py", "_joint_pair_frozen_wrapper")
    binding = wrapper.source_binding(goal, args.root_commit, args.engine_commit)
    runner = module(goal / "code/scripts/train/run_item_local_joint.py", "_joint_pair_frozen_runner")
    rows_for_arms = wrapper.study_rows(goal)
    smoke = wrapper.paired_smoke(rows_for_arms, binding)
    package = goal / "code/experiments/05_crosspse_joint"
    recipe = runner.stage05.load_recipe(package, "refined_joint")
    formal, _, layers = runner.merge_recipe(goal / "engine", package, recipe)
    # CPU-only loading: make CUDA invisible before importing torch; no CUDA API is called.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    state_helper = module(goal / "engine/genrec/models/DIFF_GRM/training_state.py", "_joint_pair_frozen_state_helper")
    arms = {row["variant"]: audit_arm(row, binding, runner, wrapper, state_helper, recipe, formal, [item["sha256"] for item in layers], smoke)
            for row in rows_for_arms}
    need(arms["control"]["config"] == arms["item_local"]["config"], "Pair differs beyond the authorized local-attention switch")
    for name in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256"):
        need(arms["control"]["initial"][name] == arms["item_local"]["initial"][name], "Pair does not share the same warm start / RNG / SID")
    precision_reports = {arm + "/" + kind: arms[arm]["SID_precision"][kind] for arm in ARMS for kind in ("best", "last")}
    common_precision = common_sid_precision_modes(precision_reports)
    for value in arms.values():
        value.pop("config")
    return {"status": "JOINT_PAIR_ARTIFACT_AUDIT_PASS", "campaign_id": CAMPAIGN, "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_binding": binding, "auditor_sha256": sha(__file__), "variants": arms, "comparison": difference(arms["control"]["metrics"], arms["item_local"]["metrics"]),
            "SID_precision_acceptance": {"common_exact_matching_modes": common_precision,
                "state_matching_modes": {name: report["matching_modes"] for name, report in precision_reports.items()},
                "rule": "One common arithmetic mode must match the exact first argmin at every catalogue position of all four best/last checkpoints; no per-position or per-state mixing.",
                "scope": "CPU emulation consistency only; not proof of historical CUDA precision flags or bitwise CUDA replay."},
            "goal_complete": False, "scope": "Offline CPU artifact consistency; no platform allocation, GPU release or settlement acceptance",
            "limitations": ["Initial shared state/RNG aggregation is a SHA-bound runner attestation, not independent reconstruction of every initial byte.",
                            "Per-phase freezing/direct-CE gradients are SHA-bound runtime attestations cross-checked against curves and actual Adam clocks.",
                            "SID acceptance requires a common exact CPU FP32 or explicitly rounded TF32 operand mode; original FP64 tolerance failures/gaps remain reported and grant no acceptance.",
                            "Matching CPU arithmetic does not prove which CUDA precision flags were active historically; saved weights/SID bytes are unchanged.",
                            "Single seed and SID-level metrics; Stage03/05 historical scores are background only."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--root-commit", required=True)
    parser.add_argument("--engine-commit", default="1c879252f6395d6e64462511823c3929b9c42c00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "Preserve an existing audit receipt; choose a new output path")
    try:
        result = audit(args)
        code = 0
    except Exception as error:
        result = {"status": "JOINT_PAIR_ARTIFACT_AUDIT_FAILED", "campaign_id": CAMPAIGN,
                  "error_type": type(error).__name__, "error": str(error), "goal_complete": False}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.output), "sha256": sha(args.output)}, ensure_ascii=False))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
