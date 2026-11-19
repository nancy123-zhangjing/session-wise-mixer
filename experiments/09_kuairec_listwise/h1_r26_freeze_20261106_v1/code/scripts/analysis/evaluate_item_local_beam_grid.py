"""Read-only Stage07 best-checkpoint validation grid; never trains or tests.

preflight verifies frozen sources, all thirteen input bytes and both CPU-loaded
best states. evaluate runs isolated control/item_local x beam32/256 workers.
Successful cells are SHA-bound and reusable; interrupted attempts stay partial.
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
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_COMMIT = "842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1"
ENGINE_COMMIT = "1c879252f6395d6e64462511823c3929b9c42c00"
CAMPAIGN = "stage07_item_local_alternating_joint_seed2024_20261020_r1"
VARIANTS = ("control", "item_local")
CELLS = tuple(f"{variant}_beam{beam}" for variant in VARIANTS for beam in (32, 256))
RUN_PREFIX = "stage07_item_local_joint_"
SCHEMA = "stage07_validation_beam_grid_v1"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def under(root, relative):
    path = (Path(root) / relative).resolve()
    need(path.is_relative_to(Path(root).resolve()), f"Unsafe input path: {relative}")
    return path


def runner_module(source):
    spec = importlib.util.spec_from_file_location("frozen_stage07_joint_runner", source / "scripts/train/run_item_local_joint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_identity(source, engine, runner):
    identities = {}
    for name, directory, expected in (("root", source, ROOT_COMMIT), ("engine", engine, ENGINE_COMMIT)):
        need(git(directory, "rev-parse", "HEAD") == expected, f"Frozen {name} commit differs")
        need(not git(directory, "status", "--porcelain", "--untracked-files=no"), f"Frozen {name} tracked files are dirty")
        identities[name] = {"path": str(directory), "commit": expected}
    files = list(runner.ENGINE_FILES) + ["genrec/dataset.py", "genrec/models/DIFF_GRM/beam.py", "genrec/models/DIFF_GRM/evaluator.py"]
    identities["engine_sources"] = {name: sha(engine / name) for name in files}
    identities["runner"] = record(source / "scripts/train/run_item_local_joint.py")
    identities["tool"] = record(Path(__file__).resolve())
    return identities


def validation_rows(sequences):
    return [(user, sequence[:-1]) for user, sequence in sequences.items() if len(sequence) > 1]


def original_validation(run):
    candidates = list((run / "diagnostics").rglob("*_evaluation_metrics.csv"))
    need(len(candidates) == 1, "Expected one original formal evaluation CSV")
    with candidates[0].open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val" and float(row["epoch"]) == 25]
    values = {row["metric"]: float(row["value"]) for row in rows}
    keys = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10", "weighted_score")
    need(len(values) == len(rows) and all(key in values and math.isfinite(values[key]) for key in keys), "Original best25 validation metrics are missing/duplicated")
    return record(candidates[0]), {key: values[key] for key in keys}


def prepare(args):
    goal = Path(args.goal_root).resolve()
    source = Path(args.source_root or goal / "code").resolve()
    engine = Path(args.engine_root or goal / "engine").resolve()
    output = Path(args.output).resolve()
    need(not output.is_relative_to(goal), "Output must be outside the immutable training goal")
    for parent in (Path(args.output).absolute(), *Path(args.output).absolute().parents):
        need(not parent.is_symlink(), "Output may not traverse symlinks")
    runner = runner_module(source)
    sources = source_identity(source, engine, runner)
    arms = {}
    for variant in VARIANTS:
        run = goal / "runs" / runner.run_id(variant)
        files = {name: record(run / name) for name in (
            "identity.json", "input_identities.json", "resolved_config.json", "results.json",
            "COMPLETE.json", "checkpoint_reload.json", "FIT_COMPLETE.json")}
        files["formal_evaluation_csv"], original_metrics = original_validation(run)
        identity, inputs, config, result, complete, reload = [read(run / name) for name in (
            "identity.json", "input_identities.json", "resolved_config.json", "results.json", "COMPLETE.json", "checkpoint_reload.json")]
        best = run / "training_state/best_model.bin"
        need(identity["campaign_id"] == result["campaign_id"] == complete["campaign_id"] == CAMPAIGN
             and identity["variant"] == result["variant"] == complete["variant"] == variant
             and identity["seed"] == config["rand_seed"] == 2024, "Campaign/variant/seed differs")
        need(identity["engine_commit"] == ENGINE_COMMIT and identity["engine_source_sha256"] ==
             {name: sources["engine_sources"][name] for name in runner.ENGINE_FILES}, "Original engine binding differs")
        need(complete["mode"] == "formal" and complete["status"] == "formal_completed"
             and complete["run_id"] == result["run_id"] == runner.run_id(variant)
             and complete["result_file"] == "results.json" and complete["result_sha256"] == files["results.json"]["sha256"], "Formal completion/result binding failed")
        digest = sha(best)
        need(digest == complete["checkpoint_sha256"] == result["checkpoint_sha256"] == reload["checkpoint_sha256"]
             and reload["strict_all_tensor_match"] is True and reload["mapping_synchronized_without_requantization"] is True
             and result["fit"]["best_epoch"] == reload["best_epoch"] == 25
             and result["selected_SID_sha256"] == reload["matching_committed_SID_sha256"], "Selected best/SID binding failed")
        need(config["beam_search_modes"] == ["confidence"] and config["topk"] == [5, 10]
             and config["eval_batch_size"] == 8 and config["max_history_len"] == 50
             and config["eval_expand_sid_to_items"] is False and config["dedup_strategy"] == "simple"
             and config["item_local_attention"]["enabled"] == (variant == "item_local")
             and config["mixed_precision"] == "bf16" and config["joint_crosspse_enabled"] is True,
             "Frozen evaluation configuration differs")
        need(len(inputs["artifacts"]) == 13 and len({row["path"] for row in inputs["artifacts"]}) == 13, "Expected thirteen distinct locked inputs")
        for row in inputs["artifacts"]:
            path = under(run / "workspace", row["path"])
            need(path.is_file() and path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"], f"Input bytes differ: {path}")
        arms[variant] = {"run": str(run), "files": files, "checkpoint": record(best), "selected_SID_sha256": result["selected_SID_sha256"],
                         "original_best25_validation": original_metrics,
                         "inputs": inputs["artifacts"], "initial_base_state_sha256": result["initial_base_state_sha256"],
                         "initial_rng_sha256": result["initial_rng_sha256"]}
    need(arms["control"]["inputs"] == arms["item_local"]["inputs"], "Arm input identities differ")
    for key in ("initial_base_state_sha256", "initial_rng_sha256"):
        need(arms["control"][key] == arms["item_local"][key], f"Original paired initialization differs: {key}")
    sequences_path = under(Path(arms["control"]["run"]) / "workspace", "data/cache/AmazonReviews2014/Beauty/processed/all_item_seqs.json")
    rows = validation_rows(read(sequences_path))
    need(len(rows) == 22363, "Expected full validation catalogue of 22363 users")
    plan = {"schema": SCHEMA, "sources": sources, "goal_root": str(goal), "arms": arms,
            "max_eval_batches": args.max_eval_batches, "validation_user_history_sha256": json_sha(rows),
            "validation_count": len(rows), "split": "val", "cells": list(CELLS),
            "protocol": {"batch_size": 8, "max_history_len": 50, "topk": [5, 10], "mode": "confidence",
                         "warmup_batches_per_cell": 1, "private_input_copy_bytes": sum(row["size_bytes"] for row in arms["control"]["inputs"]),
                         "candidate_diagnostics": "returned top10 only; repeat-fill inferred from simple dedup; no internal candidate instrumentation",
                         "test_examples_constructed": False, "test_data_evaluated": False}}
    plan["identity_sha256"] = json_sha(plan)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "GRID_PLAN.json"
    if plan_path.exists():
        need(read(plan_path) == plan, "Existing output belongs to another grid/source/checkpoint/config")
    else:
        need(not list(output.iterdir()), "New output must be empty")
        write(plan_path, plan)
    return plan, runner, output


def verify_old_bytes(plan):
    for arm in plan["arms"].values():
        for item in [*arm["files"].values(), arm["checkpoint"]]:
            need(record(item["path"]) == item, f"Original evidence changed: {item['path']}")
        for row in arm["inputs"]:
            path = under(Path(arm["run"]) / "workspace", row["path"])
            need(path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"], "Original input bytes changed")


def cpu_states(plan):
    import torch
    runner = runner_module(Path(plan["sources"]["root"]["path"]))
    need(not torch.cuda.is_initialized(), "Preflight must not initialize CUDA")
    records = {}
    for variant, arm in plan["arms"].items():
        state = torch.load(arm["checkpoint"]["path"], map_location="cpu", weights_only=False)
        need(isinstance(state, dict) and state, "Best checkpoint is not a state mapping")
        for name, value in state.items():
            need(torch.is_tensor(value) and torch.isfinite(value).all().item(), f"Nonfinite/non-tensor best state: {name}")
        codes = state["joint_alignment.hard_codes"]
        need(tuple(codes.shape) == (12102, 4) and codes.dtype == torch.int64
             and ((codes[1:] >= 0) & (codes[1:] < 256)).all().item(), "Saved SID geometry differs")
        digest = hashlib.sha256(codes[1:].contiguous().numpy().tobytes()).hexdigest()
        need(digest == arm["selected_SID_sha256"], "Saved best SID bytes differ from the formal selected SID")
        records[variant] = {"checkpoint_sha256": sha(arm["checkpoint"]["path"]), "selected_SID_sha256": digest,
                            "model_state_sha256": runner.json_sha(runner.nested_summary(state)),
                            "finite_state_tensors": len(state)}
        del state
    return records


def private_inputs(plan, output):
    root = output / "shared/workspace"
    for row in plan["arms"]["control"]["inputs"]:
        target = under(root, row["path"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = under(Path(plan["arms"]["control"]["run"]) / "workspace", row["path"])
            temporary = target.with_suffix(target.suffix + ".copying")
            shutil.copyfile(source, temporary)
            need(sha(temporary) == row["sha256"], "Private input transfer differs")
            os.replace(temporary, target)
        need(target.is_file() and target.stat().st_size == row["size_bytes"] and sha(target) == row["sha256"], "Private input copy differs")
    return root


def remap(value, old, new):
    if isinstance(value, dict):
        return {key: remap(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def top10_diagnostics(predictions, label, legal):
    tuples = [tuple(row) for row in predictions]
    target = tuple(label)
    rank = tuples.index(target) + 1 if target in tuples else None
    unique = set(tuples)
    unique_legal = sum(legal(row) for row in unique)
    return {"target_rank": rank, "top10_unique_count": len(unique), "top10_unique_legal_count": unique_legal,
            "inferred_repeat_fill_slots": len(tuples) - len(unique), "no_legal_top10_fallback": unique_legal == 0}


def cell_complete(output, name, plan):
    path = output / name / "CELL_COMPLETE.json"
    if not path.exists():
        return None
    result = read(path)
    need(result["status"] == "cell_completed" and result["grid_identity_sha256"] == plan["identity_sha256"]
         and result["cell"] == name, "Existing cell completion identity differs")
    for row in result["files"].values():
        need(record(row["path"]) == row, "Completed cell bytes changed")
    metrics = read(result["files"]["metrics"]["path"])
    need(metrics["count"] == min(22363, (plan["max_eval_batches"] or 2796) * 8)
         and metrics["model_state_unchanged"] is True and metrics["selected_SID_unchanged"] is True,
         "Completed cell is incomplete or changed model state")
    return result


def worker(args):
    output = Path(args.output).resolve()
    plan = read(output / "GRID_PLAN.json")
    need(plan["sources"]["tool"]["sha256"] == sha(__file__), "Worker tool changed")
    need(time.time() < args.deadline_unix, "Evaluation deadline expired")
    variant, beam_text = args.cell.rsplit("_beam", 1)
    beam = int(beam_text)
    arm = plan["arms"][variant]
    cell = output / args.cell
    attempt = cell / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "runtime").mkdir()
    os.chdir(attempt / "runtime")
    os.environ.update(HF_HOME=str(output / "shared/huggingface"), HF_DATASETS_CACHE=str(output / "shared/huggingface/datasets"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      TORCHDYNAMO_DISABLE="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "ACCELERATE_USE_CPU"):
        os.environ.pop(key, None)
    sys.path.insert(0, plan["sources"]["engine"]["path"])
    runner = runner_module(Path(plan["sources"]["root"]["path"]))
    import torch
    from torch.utils.data import DataLoader
    from datasets import Dataset
    import genrec.utils as utils
    from genrec.dataset import AbstractDataset
    from genrec.pipeline import Pipeline
    from genrec.models.DIFF_GRM.evaluator import DIFF_GRMEvaluator
    import importlib
    trainer_module = importlib.import_module("genrec.models.DIFF_GRM.trainer")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and "H100" in torch.cuda.get_device_name(0).upper() and torch.cuda.is_bf16_supported(), "Evaluation requires one BF16-capable H100")
    def forbidden(*unused, **keywords):
        raise RuntimeError("Training/refresh/test evaluation is forbidden in this validation-only tool")
    trainer_module.DIFF_GRMTrainer.fit = forbidden
    trainer_module.DIFF_GRMTrainer.evaluate = forbidden
    torch.optim.Optimizer.step = forbidden
    torch.optim.AdamW.step = forbidden
    config = remap(read(Path(arm["run"]) / "resolved_config.json"), str(Path(arm["run"]) / "workspace"), str(output / "shared/workspace"))
    config.update(run_id=f"beam_grid_{args.cell}_{attempt.name}", resume_training_state=False,
                  training_deadline_unix_seconds=None, training_deadline_reserve_seconds=0, max_eval_batches=None)
    for key in ("log_dir", "tensorboard_log_dir", "ckpt_dir", "run_metadata_dir", "diffusion_diagnostics_log_dir", "position_loss_log_dir", "training_state_dir"):
        config[key] = str(attempt / key)
    config["vectorized_beam_search"]["val"] = {"beam_act": beam, "beam_max": beam}
    config["current_split"] = "val"
    users, histories = [], []
    def validation_only_split(dataset):
        rows = validation_rows(dataset.all_item_seqs)
        need(len(rows) == 22363 and json_sha(rows) == plan["validation_user_history_sha256"], "Actual validation order/history differs")
        users.extend(user for user, _ in rows)
        histories.extend(sequence for _, sequence in rows)
        return {"val": Dataset.from_dict({"user": users, "item_seq": histories})}
    AbstractDataset.split = validation_only_split
    utils.get_local_time = lambda: config["run_id"]
    started_build = time.perf_counter()
    pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", config_dict=config)
    model = pipeline.model
    need(set(pipeline.tokenized_datasets) == {"val"} and len(pipeline.tokenized_datasets["val"]) == 22363
         and pipeline.accelerator.num_processes == 1 and not pipeline.config["use_ddp"]
         and pipeline.accelerator.mixed_precision == "bf16" and model.n_digit == 4
         and model.item_local_attention_enabled == (variant == "item_local"), "Actual pipeline/evaluation geometry differs")
    state = torch.load(arm["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    del state
    model.sync_joint_tokenizer_mapping()
    runner.sid_integrity(model)
    need(model.joint_codes_sha256() == arm["selected_SID_sha256"], "Strictly loaded selected SID differs")
    model.refresh_joint_codes = forbidden
    model.joint_alignment.refresh_hard_codes = forbidden
    model.requires_grad_(False)
    model.eval()
    # Match fit's Accelerator.prepare wrapping of forward, with no extra outer
    # autocast around generate/forward_decoder_only (the historical precision path).
    generator = torch.Generator().manual_seed(2024)
    loader = DataLoader(pipeline.tokenized_datasets["val"], batch_size=8, shuffle=False,
                        collate_fn=pipeline.tokenizer.collate_fn["val"], generator=generator)
    model, loader = pipeline.accelerator.prepare(model, loader)
    raw = pipeline.accelerator.unwrap_model(model)
    before = runner.json_sha(runner.nested_summary(raw.state_dict()))
    need(before == read(output / "GRID_PREFLIGHT.json")["best_states"][variant]["model_state_sha256"], "Strictly loaded model tensor bytes differ from the CPU-read original checkpoint")
    def predict(batch):
        need(time.time() < args.deadline_unix, "Evaluation deadline reached")
        need(not any(module.training for module in raw.modules()) and not torch.is_grad_enabled(), "Training mode/grad enabled during evaluation")
        prepared = raw.prepare_joint_batch(batch, training=False)
        hidden = model(prepared, return_loss=False).hidden_states
        prepared["encoder_hidden"] = hidden
        return raw.generate(prepared, n_return_sequences=10, mode="confidence"), prepared["labels"]
    write(attempt / "parsed_config.json", {key: str(value) if key == "device" else value for key, value in pipeline.config.items() if key != "accelerator"})
    write(attempt / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
          "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
          "float32_matmul_precision": torch.get_float32_matmul_precision(), "build_seconds": time.perf_counter() - started_build,
          "precision_scope": "prepared encoder forward; custom decoder outside extra outer autocast", "shared_cache": str(output / "shared")})
    with torch.no_grad():
        torch.cuda.synchronize()
        warm_started = time.perf_counter()
        warm_predictions, _ = predict(next(iter(loader)))
        del warm_predictions
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - warm_started
        evaluator = DIFF_GRMEvaluator(pipeline.config, pipeline.tokenizer)
        values, count, order = {}, 0, hashlib.sha256()
        unique_total = legal_unique_total = repeat_total = no_legal = 0
        per_user = attempt / "per_user.jsonl"
        torch.cuda.synchronize()
        started = time.perf_counter()
        with per_user.open("x", encoding="utf-8") as handle:
            for batch_index, batch in enumerate(loader):
                if plan["max_eval_batches"] and batch_index >= plan["max_eval_batches"]:
                    break
                predictions, labels = predict(batch)
                metrics = evaluator.calculate_metrics(predictions, labels)
                for key, value in metrics.items():
                    values.setdefault(key, []).extend(value.detach().cpu().tolist())
                pred_rows, label_rows = predictions.detach().cpu().tolist(), labels.detach().cpu().tolist()
                need(len(pred_rows) == len(label_rows) and all(len(row) == 10 for row in pred_rows), "Returned Top10 geometry differs")
                for offset, (pred, label) in enumerate(zip(pred_rows, label_rows)):
                    index = count + offset
                    diagnostic = top10_diagnostics(pred, label, lambda row: pipeline.tokenizer.codebooks_to_item_id(list(row)) is not None)
                    target_id = int(batch["target_item_id"][offset].item())
                    row = {"index": index, "user": users[index], "target_item_id": target_id, "target_SID": label,
                           "predicted_SIDs": pred, **diagnostic,
                           "metrics": {key: float(value[offset].item()) for key, value in metrics.items() if value.numel() == len(pred_rows)}}
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    order.update(json.dumps([index, users[index], target_id], separators=(",", ":")).encode() + b"\n")
                    unique_total += diagnostic["top10_unique_count"]
                    legal_unique_total += diagnostic["top10_unique_legal_count"]
                    repeat_total += diagnostic["inferred_repeat_fill_slots"]
                    no_legal += int(diagnostic["no_legal_top10_fallback"])
                count += len(pred_rows)
                if (batch_index + 1) % 100 == 0:
                    print(json.dumps({"cell": args.cell, "batches": batch_index + 1, "users": count}), flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    import numpy as np
    expected = min(22363, (plan["max_eval_batches"] or len(loader)) * 8)
    need(count == expected, "Evaluation is truncated")
    after = runner.json_sha(runner.nested_summary(raw.state_dict()))
    need(before == after and raw.joint_codes_sha256() == arm["selected_SID_sha256"]
         and not any(parameter.grad is not None for parameter in raw.parameters()), "Evaluation changed model/SID or produced gradients")
    aggregate = {key: float(np.mean(items)) for key, items in values.items()}
    need(all(math.isfinite(value) for value in aggregate.values()), "Nonfinite aggregate metrics")
    reproduction = None
    if beam == 32 and plan["max_eval_batches"] is None:
        differences = {key: aggregate[key] - expected for key, expected in arm["original_best25_validation"].items()}
        reproduction = {"status": "matched" if all(abs(delta) <= 1e-9 for delta in differences.values()) else "mismatch",
                        "absolute_tolerance": 1e-9, "metric_differences": differences,
                        "scope": "Same selected best25 and val32 protocol; 1e-9 absolute aggregate allowance, not a performance significance threshold"}
    metrics_path = attempt / "metrics.json"
    write(metrics_path, {"cell": args.cell, "variant": variant, "beam_act": beam, "beam_max": beam, "return_count": 10,
          "split": "val", "count": count, "metrics": aggregate, "seconds": seconds, "warmup_seconds_excluded": warm_seconds,
          "timing_scope": "synchronized full validation loop including metric calculation and per-user JSONL writes; excludes model/data construction, warmup and final state hashing",
          "mean_top10_unique_count": unique_total / count, "mean_top10_unique_legal_count": legal_unique_total / count,
          "inferred_repeat_fill_slots": repeat_total, "no_legal_top10_fallback_users": no_legal,
          "user_target_order_sha256": order.hexdigest(), "validation_user_history_sha256": plan["validation_user_history_sha256"],
          "checkpoint_sha256": sha(arm["checkpoint"]["path"]), "selected_SID_sha256": raw.joint_codes_sha256(),
          "original_val32_reproduction": reproduction,
          "model_state_before_sha256": before, "model_state_after_sha256": after,
          "model_state_unchanged": before == after, "selected_SID_unchanged": True, "gradients_created": False,
          "test_data_evaluated": False, "partial_diagnostic": plan["max_eval_batches"] is not None})
    need(reproduction is None or reproduction["status"] == "matched", "Original val32 metrics did not reproduce; retain diagnostics and stop the grid")
    need(sha(arm["checkpoint"]["path"]) == arm["checkpoint"]["sha256"], "Source best checkpoint changed")
    write(cell / "CELL_COMPLETE.json", {"status": "cell_completed", "cell": args.cell, "grid_identity_sha256": plan["identity_sha256"],
          "files": {"metrics": record(metrics_path), "per_user": record(per_user), "parsed_config": record(attempt / "parsed_config.json"),
                    "environment": record(attempt / "environment.json")}})
    pipeline.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("preflight", "evaluate"), required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--engine-root")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--max-eval-batches", type=int, help="Partial diagnostic only; never writes full COMPLETE")
    parser.add_argument("--cell", choices=CELLS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    need(args.max_eval_batches is None or args.max_eval_batches > 0, "max-eval-batches must be positive")
    if args.cell:
        need(args.mode == "evaluate" and args.deadline_unix is not None, "Internal worker requires bounded evaluate mode")
        worker(args)
        return 0
    if args.mode == "preflight":
        os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", ACCELERATE_USE_CPU="true")
    else:
        need(args.deadline_unix is not None and math.isfinite(args.deadline_unix) and time.time() < args.deadline_unix, "evaluate requires a future --deadline-unix")
    plan, runner, output = prepare(args)
    with runner.execution_guard(output):
        if args.mode == "preflight":
            states = cpu_states(plan)
            private_inputs(plan, output)
            verify_old_bytes(plan)
            write(output / "GRID_PREFLIGHT.json", {"status": "CPU_PREFLIGHT_PASSED_NO_INFERENCE", "grid_identity_sha256": plan["identity_sha256"],
                  "best_states": states, "input_count": 13, "input_bytes": plan["protocol"]["private_input_copy_bytes"],
                  "cuda_initialized": False, "test_data_evaluated": False, "scope": "CPU source/input/checkpoint/SID binding; does not prove GPU inference"})
            print("CPU_PREFLIGHT_PASSED_NO_INFERENCE", flush=True)
            return 0
        need((output / "GRID_PREFLIGHT.json").is_file()
             and read(output / "GRID_PREFLIGHT.json")["grid_identity_sha256"] == plan["identity_sha256"], "Run CPU preflight on these exact bytes first")
        private_inputs(plan, output)
        completed = {}
        for name in CELLS:
            existing = cell_complete(output, name, plan)
            if existing is None:
                cell = output / name
                cell.mkdir(exist_ok=True)
                log = cell / (datetime.now(timezone.utc).strftime("worker_%Y%m%dT%H%M%S_%fZ.log"))
                command = [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--goal-root", args.goal_root,
                           "--output", str(output), "--mode", "evaluate", "--cell", name, "--deadline-unix", str(args.deadline_unix)]
                # Inherit the bounded outer wrapper's process group: no new session.
                with log.open("x", encoding="utf-8") as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                                   timeout=max(1, args.deadline_unix - time.time()))
                existing = cell_complete(output, name, plan)
            need(existing is not None, "Cell returned without complete evidence")
            completed[name] = {"receipt": record(output / name / "CELL_COMPLETE.json"),
                               "metrics": read(existing["files"]["metrics"]["path"])}
        need(len({row["metrics"]["user_target_order_sha256"] for row in completed.values()}) == 1, "Grid cells evaluated different user/target order")
        verify_old_bytes(plan)
        need(source_identity(Path(plan["sources"]["root"]["path"]), Path(plan["sources"]["engine"]["path"]), runner) == plan["sources"], "Frozen source/tool bytes changed during the grid")
        name = "DIAGNOSTIC_COMPLETE.json" if args.max_eval_batches else "COMPLETE.json"
        write(output / name, {"schema": SCHEMA, "status": "partial_validation_diagnostic_completed" if args.max_eval_batches else "validation_beam_grid_completed",
              "grid_identity_sha256": plan["identity_sha256"], "cells": completed, "original_artifacts_unchanged": True,
              "test_data_evaluated": False, "training_performed": False,
              "scope": "Fixed single-training-seed best25/SID validation-only beam interaction diagnosis; no test retuning or significance claim"})
        print(name, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BEAM_GRID_FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
