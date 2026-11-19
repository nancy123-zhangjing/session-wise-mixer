"""Read-only selected-checkpoint validation for Stage07 seeds 2025/2026.

Each complete training run is evaluated at beam32 and beam256, on validation
only. Selection remains the training run's original val32 weighted score.
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


SCHEMA = "stage07_multiseed_selected_validation_v1"
PARENT_CAMPAIGN = "stage07_item_local_joint_multiseed_20261021_r1"
ENGINE_COMMIT = "1c879252f6395d6e64462511823c3929b9c42c00"
N_USERS = 22363
BEAMS = (32, 256)
SELECTED_METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10", "weighted_score")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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


def under(root, relative):
    path = (Path(root) / relative).resolve()
    need(path.is_relative_to(Path(root).resolve()), f"Unsafe relative path: {relative}")
    return path


def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def load_helper(source):
    path = Path(source) / "scripts/train/run_item_local_joint.py"
    spec = importlib.util.spec_from_file_location("multiseed_evaluation_frozen_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_rows(sequences):
    # Last interaction is test; validation ends at the penultimate interaction.
    return [(user, sequence[:-1]) for user, sequence in sequences.items() if len(sequence) > 1]


def audit_training(identity, config, result, fit, phase_rows):
    seed, variant = identity["seed"], identity["variant"]
    need(seed in (2025, 2026) and variant in ("control", "item_local"), "Unexpected multiseed arm")
    campaign = f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1"
    run_id = f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"
    need(identity["campaign_id"] == result["campaign_id"] == campaign
         and identity["parent_campaign_id"] == PARENT_CAMPAIGN
         and identity["run_id"] == result["run_id"] == config["run_id"] == run_id
         and result["variant"] == variant and config["rand_seed"] == seed,
         "Run/seed/campaign identities disagree")
    need(result["status"] == "formal_training_completed_validation_pending"
         and result["test_data_evaluated"] is False and result["final_validation_pending"] == [32, 256],
         "Expected completed training with validation pending and no test")
    need(result["fit"] == fit and fit["completed_epochs"] == 45
         and fit["training_stop_reason"] == "max_epochs"
         and fit["cumulative_recommender_optimizer_updates"] == 3640,
         "Training did not complete the fixed recommender budget")
    joint = fit["joint_progress"]
    need(joint["tokenizer_epochs_completed"] == 10 and joint["tokenizer_optimizer_updates"] == 60
         and joint["sid_refresh_count"] == 10, "Training did not complete the fixed tokenizer budget")
    need(len(phase_rows) == 45 and [r["epoch"] for r in phase_rows] == list(range(1, 46)), "Phase history is incomplete or duplicated")
    expected_phases = ["recommender"] * 5 + [phase for _ in range(10) for phase in ("tokenizer", "recommender", "recommender", "recommender")]
    need([r["phase"] for r in phase_rows] == expected_phases, "Alternating phase schedule differs")
    for row in phase_rows:
        need(row["optimizer_updates"] == (104 if row["phase"] == "recommender" else 6)
             and row["frozen_parameters_unchanged"] and row["all_active_gradients_finite"],
             "Phase update/freeze/gradient audit failed")
    need(config["epochs"] == 45 and config["patience"] == 0 and config["eval_start_epoch"] == 5
         and config["eval_interval"] == 5 and config["val_metric"] == "weighted_score"
         and config["vectorized_beam_search"]["val"] == {"beam_act": 32, "beam_max": 32}, "Selection or stopping rule differs")
    need(config["beam_search_modes"] == ["confidence"] and config["topk"] == [5, 10]
         and config["eval_batch_size"] == 8 and config["max_history_len"] == 50
         and config["mixed_precision"] == "bf16" and config["joint_crosspse_enabled"]
         and config["crosspse_mixer_enabled"] is False and config["crosspse_train_base_codebooks"] is True
         and config["joint_gradient_mode"] == "diger_st"
         and config["eval_expand_sid_to_items"] is False and config["dedup_strategy"] == "simple"
         and config["item_local_attention"]["enabled"] == (variant == "item_local"), "Evaluation configuration differs")
    need(type(fit["best_epoch"]) is int and 0 <= fit["best_epoch"] <= 45 and math.isfinite(fit["best_val_score"]), "Invalid selected checkpoint")
    return {"seed": seed, "variant": variant, "campaign_id": campaign, "run_id": run_id,
            "logical_epochs": 45, "recommender_epochs": 35, "tokenizer_phases": 10,
            "recommender_updates": 3640, "tokenizer_updates": 60, "best_epoch": fit["best_epoch"]}


def selected_validation(path, best_epoch, best_score):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val" and float(row["epoch"]) == best_epoch]
    metrics = {row["metric"]: float(row["value"]) for row in rows}
    need(len(metrics) == len(rows) and all(key in metrics and math.isfinite(metrics[key]) for key in SELECTED_METRICS), "Selected validation row is missing, duplicate or nonfinite")
    need(abs(metrics["weighted_score"] - best_score) <= 1e-12, "Selected score differs from fit receipt")
    return {key: metrics[key] for key in SELECTED_METRICS}


def source_identity(source, engine, helper):
    need(git(engine, "rev-parse", "HEAD") == ENGINE_COMMIT and not git(engine, "status", "--porcelain", "--untracked-files=no"), "Frozen engine changed")
    names = set(helper.ENGINE_FILES) | {"genrec/dataset.py", "genrec/models/DIFF_GRM/beam.py", "genrec/models/DIFF_GRM/evaluator.py"}
    root_names = ("scripts/train/run_item_local_joint.py", "scripts/train/run_item_local_joint_multiseed.py", "scripts/repro/stage05_repro.py")
    return {"source_root": str(source), "source_commit": git(source, "rev-parse", "HEAD"),
            "engine_root": str(engine), "engine_commit": ENGINE_COMMIT,
            "engine_files": {name: record(engine / name) for name in sorted(names)},
            "root_files": {name: record(source / name) for name in root_names}, "evaluator": record(__file__)}


def prepare(args):
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    source, engine = Path(args.source_root).resolve(), Path(args.engine_root).resolve()
    need(output == run / "validation", "Evaluation output must be the run's dedicated validation directory")
    for path in (Path(args.output).absolute(), *Path(args.output).absolute().parents):
        need(not path.is_symlink(), "Evaluation output may not traverse symlinks")
    need(not (run / "TEST_STARTED.json").exists() and not (run / "TEST_RESULT.json").exists(), "Unexpected test artifacts")
    helper = load_helper(source)
    sources = source_identity(source, engine, helper)
    names = ("identity.json", "resolved_config.json", "results.json", "COMPLETE.json", "FIT_COMPLETE.json",
             "checkpoint_reload.json", "input_identities.json", "code_identity.json", "phase_integrity.json", "training_integrity.json")
    original_files = {name: record(run / name) for name in names}
    identity, config, result, complete, fit, reload, inputs, code, phases, integrity = [read(run / name) for name in names]
    audit = audit_training(identity, config, result, fit, phases)
    need(run.name == audit["run_id"] and complete["mode"] == "formal"
         and complete["status"] == result["status"] and complete["run_id"] == audit["run_id"]
         and complete["campaign_id"] == audit["campaign_id"] and complete["variant"] == audit["variant"]
         and complete["result_file"] == "results.json" and complete["result_sha256"] == original_files["results.json"]["sha256"], "Training completion binding differs")
    need(identity["engine_commit"] == ENGINE_COMMIT and identity["engine_source_sha256"] == {name: sources["engine_files"][name]["sha256"] for name in helper.ENGINE_FILES}, "Training engine binding differs")
    adapter = sources["root_files"]["scripts/train/run_item_local_joint_multiseed.py"]
    need(identity["multiseed_adapter"]["sha256"] == adapter["sha256"]
         and code["multiseed_adapter"]["sha256"] == adapter["sha256"]
         and code["runner_sha256"] == sources["root_files"]["scripts/train/run_item_local_joint.py"]["sha256"]
         and code["stage05_helper_sha256"] == sources["root_files"]["scripts/repro/stage05_repro.py"]["sha256"], "Training helper/adapter binding differs")
    need(integrity["status"] == "passed" and integrity["phase_records_sha256"] == original_files["phase_integrity.json"]["sha256"]
         and result["phase_integrity_sha256"] == original_files["phase_integrity.json"]["sha256"], "Training integrity binding differs")
    checkpoint = record(run / "training_state/best_model.bin")
    need(checkpoint["sha256"] == result["checkpoint_sha256"] == complete["checkpoint_sha256"] == reload["checkpoint_sha256"]
         and reload["strict_all_tensor_match"] and reload["mapping_synchronized_without_requantization"]
         and reload["best_epoch"] == audit["best_epoch"]
         and reload["matching_committed_SID_sha256"] == result["selected_SID_sha256"], "Best checkpoint or saved SID binding differs")
    need(len(inputs["artifacts"]) == 13 and len({r["path"] for r in inputs["artifacts"]}) == 13, "Expected thirteen locked inputs")
    for row in inputs["artifacts"]:
        item = record(under(run / "workspace", row["path"]))
        need(item["sha256"] == row["sha256"] and item["size_bytes"] == row["size_bytes"], "Original input changed")
    csvs = list((run / "diagnostics").rglob("*_evaluation_metrics.csv"))
    need(len(csvs) == 1, "Expected one original evaluation CSV")
    original_files["selected_validation_csv"] = record(csvs[0])
    expected = selected_validation(csvs[0], audit["best_epoch"], fit["best_val_score"])
    sequence_path = under(run / "workspace", "data/cache/AmazonReviews2014/Beauty/processed/all_item_seqs.json")
    rows = validation_rows(read(sequence_path))
    need(len(rows) == N_USERS, "Validation cardinality differs")
    plan = {"schema": SCHEMA, "run": str(run), "audit": audit, "sources": sources, "original_files": original_files,
            "inputs": inputs["artifacts"], "checkpoint": checkpoint, "selected_SID_sha256": result["selected_SID_sha256"],
            "selected_validation": expected, "validation_user_history_sha256": json_sha(rows), "validation_count": N_USERS,
            "split": "val", "beams": list(BEAMS), "batch_size": 8, "topk": [5, 10], "reselection_allowed": False,
            "initial_base_state_sha256": result["initial_base_state_sha256"], "initial_rng_sha256": result["initial_rng_sha256"]}
    plan["identity_sha256"] = json_sha(plan)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "EVALUATION_PLAN.json"
    if plan_path.exists():
        need(read(plan_path) == plan, "Existing evaluation belongs to different source/state/protocol")
    else:
        need(not list(output.iterdir()), "New evaluation directory must be empty")
        write(plan_path, plan)
    return plan, helper, output


def verify_sources(plan):
    records = [*plan["original_files"].values(), plan["checkpoint"], *plan["sources"]["root_files"].values(),
               *plan["sources"]["engine_files"].values(), plan["sources"]["evaluator"]]
    for item in records:
        need(record(item["path"]) == item, f"Bound source/state changed: {item['path']}")
    for row in plan["inputs"]:
        item = record(under(Path(plan["run"]) / "workspace", row["path"]))
        need(item["sha256"] == row["sha256"] and item["size_bytes"] == row["size_bytes"], "Bound input changed")


def verify_private_inputs(plan, output):
    for row in plan["inputs"]:
        item = record(under(Path(output) / "shared/workspace", row["path"]))
        need(item["sha256"] == row["sha256"] and item["size_bytes"] == row["size_bytes"], "Private evaluation input changed")


def cpu_preflight(plan, helper, output):
    import torch
    need(not torch.cuda.is_initialized(), "CPU preflight must not initialize CUDA")
    state = torch.load(plan["checkpoint"]["path"], map_location="cpu", weights_only=False)
    need(isinstance(state, dict) and state, "Selected checkpoint is not a state mapping")
    need(all(torch.is_tensor(t) and bool(torch.isfinite(t).all()) for t in state.values()), "Selected state has nonfinite/non-tensor entries")
    codes = state["joint_alignment.hard_codes"]
    need(tuple(codes.shape) == (12102, 4) and codes.dtype == torch.int64 and bool(((codes[1:] >= 0) & (codes[1:] < 256)).all()), "Saved SID geometry differs")
    sid_sha = hashlib.sha256(codes[1:].contiguous().numpy().tobytes()).hexdigest()
    need(sid_sha == plan["selected_SID_sha256"], "CPU-read saved SID differs")
    model_sha = helper.json_sha(helper.nested_summary(state))
    del state
    workspace = output / "shared/workspace"
    for row in plan["inputs"]:
        destination = under(workspace, row["path"])
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".copying")
            shutil.copyfile(under(Path(plan["run"]) / "workspace", row["path"]), temporary)
            need(sha(temporary) == row["sha256"], "Private input copy changed")
            os.replace(temporary, destination)
        need(destination.stat().st_size == row["size_bytes"] and sha(destination) == row["sha256"], "Private input differs")
    verify_sources(plan)
    result = {"status": "CPU_PREFLIGHT_PASSED_NO_INFERENCE", "identity_sha256": plan["identity_sha256"],
              "model_state_sha256": model_sha, "selected_SID_sha256": sid_sha,
              "cuda_initialized": False, "test_data_evaluated": False}
    write(output / "EVALUATION_PREFLIGHT.json", result)
    return result


def remap(value, old, new):
    if isinstance(value, dict):
        return {key: remap(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def top10_diagnostics(predictions, label, legal):
    tuples, target = [tuple(row) for row in predictions], tuple(label)
    rank = tuples.index(target) + 1 if target in tuples else None
    unique = set(tuples)
    unique_legal = sum(legal(row) for row in unique)
    return {"target_rank": rank, "top10_unique_count": len(unique), "top10_unique_legal_count": unique_legal,
            "inferred_repeat_fill_slots": len(tuples) - len(unique), "no_legal_top10_fallback": unique_legal == 0}


def worker(args):
    output = Path(args.output).resolve()
    plan = read(output / "EVALUATION_PLAN.json")
    preflight = read(output / "EVALUATION_PREFLIGHT.json")
    need(plan["sources"]["evaluator"] == record(__file__) and preflight["identity_sha256"] == plan["identity_sha256"], "Worker/preflight identity differs")
    verify_private_inputs(plan, output)
    need(time.time() < args.deadline_unix, "Evaluation deadline expired")
    cell = output / f"beam{args.beam}"
    cell.mkdir(exist_ok=True)
    started_path = cell / "CELL_STARTED.json"
    with started_path.open("x", encoding="utf-8") as handle:
        json.dump({"identity_sha256": plan["identity_sha256"], "beam": args.beam, "started_unix": time.time(),
                   "checkpoint_sha256": plan["checkpoint"]["sha256"], "selected_SID_sha256": plan["selected_SID_sha256"]}, handle)
    attempt = cell / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    (attempt / "runtime").mkdir(parents=True, exist_ok=False)
    os.chdir(attempt / "runtime")
    os.environ.update(HF_HOME=str(output / "shared/huggingface"), HF_DATASETS_CACHE=str(output / "shared/huggingface/datasets"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
                      TORCHDYNAMO_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "ACCELERATE_USE_CPU"):
        os.environ.pop(key, None)
    sys.path.insert(0, plan["sources"]["engine_root"])
    helper = load_helper(plan["sources"]["source_root"])
    import torch
    import importlib
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from genrec.dataset import AbstractDataset
    from genrec.pipeline import Pipeline
    from genrec.models.DIFF_GRM.evaluator import DIFF_GRMEvaluator
    import genrec.utils as utils
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0).upper()
         and torch.cuda.is_bf16_supported(), "Evaluation requires exactly one visible BF16 H100")
    def forbidden(*unused, **keywords):
        raise RuntimeError("Training, SID refresh and test evaluation are forbidden in selected validation")
    trainer = importlib.import_module("genrec.models.DIFF_GRM.trainer")
    trainer.DIFF_GRMTrainer.fit = forbidden
    trainer.DIFF_GRMTrainer.evaluate = forbidden
    torch.optim.Optimizer.step = forbidden
    torch.optim.AdamW.step = forbidden
    run = Path(plan["run"])
    config = remap(read(run / "resolved_config.json"), str(run / "workspace"), str(output / "shared/workspace"))
    config.update(run_id=f"{plan['audit']['run_id']}_val_beam{args.beam}_{attempt.name}", resume_training_state=False,
                  training_deadline_unix_seconds=None, training_deadline_reserve_seconds=0, max_eval_batches=None)
    for key in ("log_dir", "tensorboard_log_dir", "ckpt_dir", "run_metadata_dir", "diffusion_diagnostics_log_dir", "position_loss_log_dir", "training_state_dir"):
        config[key] = str(attempt / key)
    config["vectorized_beam_search"]["val"] = {"beam_act": args.beam, "beam_max": args.beam}
    config["current_split"] = "val"
    users = []
    def val_only_split(dataset):
        rows = validation_rows(dataset.all_item_seqs)
        need(len(rows) == N_USERS and json_sha(rows) == plan["validation_user_history_sha256"], "Validation user/history order differs")
        users.extend(user for user, _ in rows)
        return {"val": Dataset.from_dict({"user": users, "item_seq": [history for _, history in rows]})}
    AbstractDataset.split = val_only_split
    utils.get_local_time = lambda: config["run_id"]
    pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", config_dict=config)
    model = pipeline.model
    need(set(pipeline.tokenized_datasets) == {"val"} and len(pipeline.tokenized_datasets["val"]) == N_USERS
         and pipeline.accelerator.num_processes == 1 and not pipeline.config["use_ddp"]
         and pipeline.accelerator.mixed_precision == "bf16" and model.n_digit == 4
         and model.item_local_attention_enabled == (plan["audit"]["variant"] == "item_local"), "Validation pipeline differs")
    state = torch.load(plan["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    del state
    model.sync_joint_tokenizer_mapping()
    helper.sid_integrity(model)
    need(model.joint_codes_sha256() == plan["selected_SID_sha256"], "Loaded saved SID differs")
    model.refresh_joint_codes = forbidden
    model.joint_alignment.refresh_hard_codes = forbidden
    model.requires_grad_(False)
    model.eval()
    loader = DataLoader(pipeline.tokenized_datasets["val"], batch_size=8, shuffle=False,
                        collate_fn=pipeline.tokenizer.collate_fn["val"], generator=torch.Generator().manual_seed(plan["audit"]["seed"]))
    model, loader = pipeline.accelerator.prepare(model, loader)
    raw = pipeline.accelerator.unwrap_model(model)
    before = helper.json_sha(helper.nested_summary(raw.state_dict()))
    need(before == preflight["model_state_sha256"], "Strictly loaded state bytes differ")
    def predict(batch):
        need(time.time() < args.deadline_unix, "Evaluation deadline reached")
        need(not torch.is_grad_enabled() and not any(module.training for module in raw.modules()), "Evaluation training mode/grad violation")
        prepared = raw.prepare_joint_batch(batch, training=False)
        prepared["encoder_hidden"] = model(prepared, return_loss=False).hidden_states
        # Match historical Accelerator-wrapped encoder; no outer decoder autocast.
        return raw.generate(prepared, n_return_sequences=10, mode="confidence"), prepared["labels"]
    write(attempt / "parsed_config.json", {k: str(v) if k == "device" else v for k, v in pipeline.config.items() if k != "accelerator"})
    write(attempt / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
          "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
          "float32_matmul_precision": torch.get_float32_matmul_precision(),
          "precision_scope": "prepared encoder forward; custom decoder outside extra outer autocast"})
    with torch.no_grad():
        warm = time.perf_counter()
        predict(next(iter(loader)))
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - warm
        evaluator = DIFF_GRMEvaluator(pipeline.config, pipeline.tokenizer)
        values, count, order = {}, 0, hashlib.sha256()
        per_user = attempt / "per_user.jsonl"
        started = time.perf_counter()
        with per_user.open("x", encoding="utf-8") as handle:
            for batch_index, batch in enumerate(loader):
                predictions, labels = predict(batch)
                metrics = evaluator.calculate_metrics(predictions, labels)
                for key, value in metrics.items():
                    values.setdefault(key, []).extend(value.detach().cpu().tolist())
                pred_rows, label_rows = predictions.detach().cpu().tolist(), labels.detach().cpu().tolist()
                need(len(pred_rows) == len(label_rows) and all(len(row) == 10 for row in pred_rows), "Returned Top10 geometry differs")
                for offset, (pred, label) in enumerate(zip(pred_rows, label_rows)):
                    index, item_id = count + offset, int(batch["target_item_id"][offset].item())
                    diagnostic = top10_diagnostics(pred, label, lambda row: pipeline.tokenizer.codebooks_to_item_id(list(row)) is not None)
                    row = {"index": index, "user": users[index], "target_item_id": item_id, "target_SID": label,
                           "predicted_SIDs": pred, **diagnostic,
                           "metrics": {key: float(value[offset].item()) for key, value in metrics.items() if value.numel() == len(pred_rows)}}
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    order.update(json.dumps([index, users[index], item_id], separators=(",", ":")).encode() + b"\n")
                count += len(pred_rows)
                if (batch_index + 1) % 100 == 0:
                    print(json.dumps({"beam": args.beam, "users": count, "batches": batch_index + 1}), flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    import numpy as np
    need(count == N_USERS, "Validation was truncated")
    after = helper.json_sha(helper.nested_summary(raw.state_dict()))
    need(before == after and raw.joint_codes_sha256() == plan["selected_SID_sha256"]
         and not any(parameter.grad is not None for parameter in raw.parameters()), "Validation changed model/SID or created gradients")
    aggregate = {key: float(np.mean(items)) for key, items in values.items()}
    need(all(math.isfinite(value) for value in aggregate.values()), "Nonfinite validation metrics")
    reproduction = None
    if args.beam == 32:
        differences = {key: aggregate[key] - value for key, value in plan["selected_validation"].items()}
        reproduction = {"status": "matched" if all(abs(delta) <= 1e-9 for delta in differences.values()) else "mismatch",
                        "best_epoch": plan["audit"]["best_epoch"], "absolute_tolerance": 1e-9, "metric_differences": differences}
    metrics_path = attempt / "metrics.json"
    write(metrics_path, {"schema": SCHEMA, "seed": plan["audit"]["seed"], "variant": plan["audit"]["variant"],
          "beam_act": args.beam, "beam_max": args.beam, "split": "val", "count": count, "metrics": aggregate,
          "checkpoint_sha256": plan["checkpoint"]["sha256"], "selected_SID_sha256": plan["selected_SID_sha256"],
          "best_epoch": plan["audit"]["best_epoch"], "user_target_order_sha256": order.hexdigest(),
          "validation_user_history_sha256": plan["validation_user_history_sha256"], "original_val32_reproduction": reproduction,
          "model_state_before_sha256": before, "model_state_after_sha256": after, "model_state_unchanged": True,
          "selected_SID_unchanged": True, "training_gradients_created": False, "test_data_evaluated": False,
          "seconds": seconds, "warmup_seconds_excluded": warm_seconds,
          "timing_scope": "synchronized full validation loop including metrics and per-user writes, not serving latency"})
    need(reproduction is None or reproduction["status"] == "matched", "Selected val32 score did not reproduce")
    verify_sources(plan)
    verify_private_inputs(plan, output)
    write(cell / "CELL_COMPLETE.json", {"status": "cell_completed", "identity_sha256": plan["identity_sha256"],
          "beam": args.beam, "started_receipt": record(started_path),
          "files": {"metrics": record(metrics_path), "per_user": record(per_user), "parsed_config": record(attempt / "parsed_config.json"),
                    "environment": record(attempt / "environment.json")}})
    pipeline.accelerator.end_training()


def completed_cell(output, beam, plan):
    path = output / f"beam{beam}" / "CELL_COMPLETE.json"
    if not path.exists():
        return None
    receipt = read(path)
    need(receipt["status"] == "cell_completed" and receipt["identity_sha256"] == plan["identity_sha256"] and receipt["beam"] == beam, "Cell completion identity differs")
    for item in [receipt["started_receipt"], *receipt["files"].values()]:
        need(record(item["path"]) == item, "Completed cell bytes changed")
    metrics = read(receipt["files"]["metrics"]["path"])
    need(metrics["count"] == N_USERS and metrics["split"] == "val" and metrics["beam_act"] == beam
         and metrics["checkpoint_sha256"] == plan["checkpoint"]["sha256"] and metrics["selected_SID_sha256"] == plan["selected_SID_sha256"]
         and metrics["model_state_unchanged"] and metrics["selected_SID_unchanged"]
         and not metrics["training_gradients_created"] and not metrics["test_data_evaluated"], "Cell acceptance failed")
    need(beam != 32 or metrics["original_val32_reproduction"]["status"] == "matched", "Selected val32 reproduction missing")
    return {"receipt": record(path), "files": receipt["files"], "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("preflight", "evaluate"), required=True)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--beam", type=int, choices=BEAMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "evaluate":
        need(args.deadline_unix is not None and math.isfinite(args.deadline_unix) and time.time() < args.deadline_unix, "Evaluation requires a future bounded deadline")
    if args.beam is not None:
        need(args.mode == "evaluate", "Internal worker is evaluation only")
        worker(args)
        return
    if args.mode == "preflight":
        os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", ACCELERATE_USE_CPU="true")
    plan, helper, output = prepare(args)
    with helper.execution_guard(output):
        preflight_path = output / "EVALUATION_PREFLIGHT.json"
        if not preflight_path.exists():
            cpu_preflight(plan, helper, output)
        else:
            need(read(preflight_path)["identity_sha256"] == plan["identity_sha256"], "Existing preflight identity differs")
        if args.mode == "preflight":
            print("CPU_PREFLIGHT_PASSED_NO_INFERENCE", flush=True)
            return
        cells = {}
        for beam in BEAMS:
            accepted = completed_cell(output, beam, plan)
            if accepted is None:
                cell = output / f"beam{beam}"
                cell.mkdir(exist_ok=True)
                need(not (cell / "CELL_STARTED.json").exists(), "Cell already started without completion; inspect before any explicit retry")
                command = [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--run", args.run,
                           "--source-root", args.source_root, "--engine-root", args.engine_root, "--output", str(output),
                           "--mode", "evaluate", "--beam", str(beam), "--deadline-unix", str(args.deadline_unix)]
                with (cell / "worker.log").open("x", encoding="utf-8") as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=max(1, args.deadline_unix - time.time()))
                accepted = completed_cell(output, beam, plan)
            need(accepted is not None, "Worker returned without complete evidence")
            cells[f"beam{beam}"] = accepted
        need(len({row["metrics"]["user_target_order_sha256"] for row in cells.values()}) == 1, "Beams evaluated different user/target order")
        verify_sources(plan)
        write(output / "EVALUATION_COMPLETE.json", {"schema": SCHEMA, "status": "selected_validation_completed", "identity_sha256": plan["identity_sha256"],
              "seed": plan["audit"]["seed"], "variant": plan["audit"]["variant"], "cells": cells, "count_per_cell": N_USERS,
              "checkpoint_sha256": plan["checkpoint"]["sha256"], "selected_SID_sha256": plan["selected_SID_sha256"],
              "best_epoch": plan["audit"]["best_epoch"], "model_state_unchanged": True, "selected_SID_unchanged": True,
              "training_gradients_created": False, "test_data_evaluated": False, "training_performed": False,
              "original_val32_reproduction": cells["beam32"]["metrics"]["original_val32_reproduction"],
              "scope": "Both fixed beams use the same validation-selected best and saved SID; no reselection or test evaluation"})
        print("EVALUATION_COMPLETE.json", flush=True)


if __name__ == "__main__":
    main()
