"""One frozen-PSE item-local-attention run; CPU preparation never starts CUDA.

The historical recipe is resolved by the existing replay helpers. Smoke uses
only validation targets, and formal testing is guarded by a persistent marker.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/repro"))
import project_replay as replay
import project_repro as repro


REPAIR_CAMPAIGN = "stage07_ocn_repair_seed2024_20261020_r1"
REPAIR_ENGINE = "8a48a300daa6621d6c705966ff73fa837ef8f12e"


def study_controls(study):
    """Keep the old pilot contract separate from the new repaired A/B pair."""
    campaign = study.get("repair_campaign")
    repair = campaign is not None
    if repair:
        if campaign != REPAIR_CAMPAIGN or study.get("variant") not in ("control", "item_local"):
            raise ValueError("Unknown repaired campaign or variant")
        expected = f"stage07_ocn_repair_{study['variant']}_seed2024_20261020_r1"
        if study["run_id"] != expected:
            raise ValueError("Repaired campaign requires its exact separate run_id")
        enabled, cap = study["variant"] == "item_local", 28.0
    else:
        if study["run_id"].startswith("stage07_ocn_repair_") or study.get("variant") is not None:
            raise ValueError("Repair variants cannot use the historical pilot identity")
        enabled, cap = True, 8.0
    if study.get("item_local_attention") != {"enabled": enabled, "num_heads": 4, "dropout": 0.0}:
        raise ValueError("Unexpected item-local-attention controls for this study")
    hours = float(study.get("max_gpu_hours", cap))
    if not math.isfinite(hours) or not 0 < hours <= cap:
        raise ValueError(f"Study GPU allocation cap must be positive and <= {cap} cumulative hours")
    return repair


def nested_summary(value):
    """Stable byte identities, independent of pickle storage/object IDs."""
    import numpy as np
    import torch
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": hashlib.sha256(data).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): nested_summary(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [nested_summary(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def summary_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_same_nested(actual, expected, name):
    import numpy as np
    import torch
    if torch.is_tensor(expected):
        equal = torch.is_tensor(actual) and actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, np.ndarray):
        equal = isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"Recovery {name} keys differ")
        for key in expected:
            require_same_nested(actual[key], expected[key], f"{name}.{key}")
        return
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError(f"Recovery {name} sequence differs")
        for index, item in enumerate(expected):
            require_same_nested(actual[index], item, f"{name}.{index}")
        return
    else:
        equal = actual == expected
    if not equal:
        raise RuntimeError(f"Recovery {name} differs")


def gradient_hooks(model):
    """Accumulate finite flags on-device; synchronize once at audit time."""
    import torch
    counts, finite_flags, handles = {}, {}, []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise RuntimeError(f"Unexpected frozen repaired-model parameter: {name}")
        counts[name] = 0

        def observe(gradient, key=name):
            counts[key] += 1
            finite = torch.isfinite(gradient.detach()).all()
            if key in finite_flags:
                finite_flags[key].logical_and_(finite)
            else:
                finite_flags[key] = finite

        handles.append(parameter.register_hook(observe))
    return counts, finite_flags, handles


def gradient_receipt(counts, finite_flags):
    missing = [name for name, count in counts.items() if count == 0]
    nonfinite = [name for name, flag in finite_flags.items() if not bool(flag.item())]
    if missing or nonfinite:
        raise RuntimeError(f"Incomplete/nonfinite parameter gradients: missing={missing}, nonfinite={nonfinite}")
    return {"passed": True, "all_gradient_coverage": True, "all_parameter_tensors": len(counts),
            "gradient_hook_calls": counts, "all_observed_gradients_finite": True}


def optimizer_receipt(payload, model, expected_updates):
    import torch
    parameters = list(model.named_parameters())
    optimizer = payload["optimizer"]
    groups = optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
        raise RuntimeError("Optimizer parameter-group layout differs from the repaired model")
    identifiers = groups[0]["params"]
    if len(set(identifiers)) != len(parameters) or set(optimizer["state"]) != set(identifiers):
        raise RuntimeError("Optimizer state does not cover every expected parameter")
    for identifier, (name, parameter) in zip(identifiers, parameters):
        entry = optimizer["state"][identifier]
        step = float(entry["step"].item() if torch.is_tensor(entry["step"]) else entry["step"])
        if step != int(expected_updates):
            raise RuntimeError(f"Optimizer update count differs for {name}: {step}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = entry[key]
            if tuple(value.shape) != tuple(parameter.shape) or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Invalid optimizer moments for {name}")
    return {"optimizer_state_count": len(optimizer["state"]), "all_parameter_tensors": len(parameters),
            "cumulative_optimizer_updates": int(expected_updates), "all_optimizer_steps_match": True}


def profile_recovery_check(pipeline, trainer, work):
    """Restore a private copy of a terminal profile state, then train one update."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler
    from genrec.models.DIFF_GRM.training_state import capture_rng, configuration_fingerprint, restore_boundary
    accelerator, config = pipeline.accelerator, pipeline.config
    last, best = work / "training_state/last.pt", work / "training_state/best_model.bin"
    original_hashes = {"last": repro.sha(last), "best": repro.sha(best)}
    payload = torch.load(last, map_location="cpu", weights_only=False)
    before_updates = int(payload["progress"]["cumulative_optimizer_updates"])
    optimizer_receipt(payload, accelerator.unwrap_model(trainer.model), before_updates)
    directory = work / "recovery_probe"
    directory.mkdir(exist_ok=False)
    shutil.copyfile(last, directory / "source_last.pt")
    copied_best = directory / "best_model.bin"
    copied_best.write_bytes(payload["best_checkpoint_bytes"])
    if repro.sha(copied_best) != payload["best_checkpoint_sha256"]:
        raise RuntimeError("Profile recovery embedded best checkpoint is damaged")
    payload["best_checkpoint"] = str(copied_best)
    # A fresh model avoids accidentally reusing a wrapped forward closure.
    model_type = type(accelerator.unwrap_model(trainer.model))
    model = model_type(config, pipeline.raw_dataset, pipeline.tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loader = DataLoader(pipeline.tokenized_datasets["train"], batch_size=config["train_batch_size"],
                        shuffle=True, collate_fn=pipeline.tokenizer.collate_fn["train"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=trainer.effective_warmup_steps,
                              num_training_steps=trainer.total_optimizer_steps)
    fingerprint = configuration_fingerprint(config)
    if fingerprint != payload["configuration_fingerprint"]:
        raise RuntimeError("Profile recovery configuration fingerprint differs")
    # Keep expected values immutable if optimizer.load_state_dict shares tensors.
    expected = copy.deepcopy(payload)
    restore_boundary(payload, model=model, optimizer=optimizer, scheduler=scheduler,
                     accelerator=accelerator, loader=loader, config_fingerprint=fingerprint)
    require_same_nested(accelerator.unwrap_model(model).state_dict(), expected["model"], "model")
    require_same_nested(optimizer.state_dict(), expected["optimizer"], "optimizer")
    require_same_nested(scheduler.state_dict(), expected["scheduler"], "scheduler")
    require_same_nested(capture_rng(loader), expected["rng"], "RNG")
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    counts, flags, handles = gradient_hooks(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_batches, updates = 0, 0
    try:
        for batch in loader:
            trainer._check_training_deadline()
            with accelerator.accumulate(model):
                loss = model(batch, return_loss=True).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite loss in recovered profile step")
                accelerator.backward(loss)
                if accelerator.sync_gradients and config["max_grad_norm"] is not None:
                    norm = accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError("Nonfinite gradient norm in recovered profile step")
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
            micro_batches += 1
            if updates == 1:
                break
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    coverage = gradient_receipt(counts, flags)
    if updates != 1 or micro_batches != int(config["gradient_accumulation_steps"]):
        raise RuntimeError("Recovery probe did not execute exactly one complete accumulated update")
    optimizer_receipt({"optimizer": optimizer.state_dict()}, accelerator.unwrap_model(model), before_updates + 1)
    if int(scheduler.state_dict()["last_epoch"]) != int(expected["scheduler"]["last_epoch"]) + 1:
        raise RuntimeError("Recovered scheduler update is not continuous")
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(before[name], parameter.detach().cpu())]
    if len(changed) != len(before):
        raise RuntimeError("Recovered optimizer update did not change every parameter tensor")
    if original_hashes != {"last": repro.sha(last), "best": repro.sha(best)}:
        raise RuntimeError("Recovery probe changed the original profile last/best checkpoint")
    result = {"status": "passed", "updates_before": before_updates, "updates_after": before_updates + 1,
              "optimizer_updates_executed": updates, "micro_batches": micro_batches,
              "strict_model_optimizer_scheduler_RNG_restore": True,
              "all_gradient_coverage": coverage["all_gradient_coverage"], "gradient_integrity": coverage,
              "actual_changed_parameter_tensors": len(changed), "original_checkpoint_hashes_unchanged": original_hashes,
              "terminal_profile_state": bool(expected["progress"]["training_complete"]),
              "manual_one_update_probe": True, "trainer_fit_resume_not_exercised": True, "cuda_bitwise_claim": False}
    write_json(directory / "recovery_check.json", result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                   default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def record_attempt(work, attempt, name, value):
    """Keep the first receipt intact and record every later invocation."""
    write_json(attempt / name, value)
    original = work / name
    if not original.exists():
        write_json(original, value)


def git_record(path):
    def command(*args):
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return {"path": str(path), "head": command("rev-parse", "HEAD"),
            "status": command("status", "--porcelain")}


def copy_bound(source, destination, digest):
    if not source.is_file() or repro.sha(source) != digest:
        raise ValueError(f"Input missing or different from asset lock: {source}")
    if destination.exists():
        if not destination.is_file() or repro.sha(destination) != digest:
            raise ValueError(f"Preserving differing existing input: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, destination.open("xb") as out:
        shutil.copyfileobj(inp, out)
    if repro.sha(destination) != digest:
        raise ValueError(f"Input copy failed hash verification: {destination}")


def baseline_row(pack):
    with (pack / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["id"] == "pse_seed2024"]
    if len(rows) != 1:
        raise ValueError("Expected the single locked Stage03 PSE result")
    return rows[0]


def prepare(args):
    import yaml
    study = read_json(args.config)
    run_id = study["run_id"]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must use letters, digits, underscores or hyphens")
    if study.get("baseline_id", "pse_seed2024") != "pse_seed2024":
        raise ValueError("Only the authorized Stage03 PSE baseline is supported")
    if int(study.get("seed", 2024)) != 2024:
        raise ValueError("The authorized exploration uses only seed2024")
    repair = study_controls(study)
    attention = study["item_local_attention"]
    if args.mode == "profile" and not repair:
        raise ValueError("Profile mode is reserved for the repaired campaign")
    if args.mode == "profile" and args.resume:
        raise ValueError("Profile is a fresh isolated acceptance run, not a training-state resume")
    output = Path(args.output).resolve()
    project = Path(args.project_root).resolve()
    engine = Path(args.engine_root).resolve()
    if output in (project, engine, ROOT):
        raise ValueError("Output must be a separate experiment directory")
    if not (engine / "genrec/pipeline.py").is_file():
        raise ValueError("Selected engine is not a DiffGRM checkout")
    if repair:
        engine_identity = git_record(engine)
        if engine_identity["head"] != REPAIR_ENGINE or engine_identity["status"] != "":
            raise ValueError("The repaired campaign requires the clean frozen 8a48a30 engine")
    work = output / args.mode if args.mode in ("smoke", "profile") else output
    actual_run_id = run_id + "_" + args.mode if args.mode in ("smoke", "profile") else run_id
    if repair and isinstance(args.resume, str) and Path(args.resume).resolve() != (work / "training_state/last.pt").resolve():
        raise ValueError("Repair recovery may only use this new run's own last.pt; historical/profile states are forbidden")
    if (work / "COMPLETE.json").is_file():
        prior = read_json(work / "identity.json")
        if prior.get("run_id") != actual_run_id or prior.get("study_sha256") != repro.sha(args.config):
            raise ValueError("Completed directory belongs to a different run/study")
        return {"already_complete": read_json(work / "COMPLETE.json"), "work": work}
    pack = ROOT / "experiments/03_diffgrm_tokenizer_comparison"
    recipe = read_json(pack / "reproducibility/recipes/pse_seed2024.json")
    metadata = read_json(pack / recipe["metadata"])
    code = [r for r in read_json(pack / "reproducibility/code_versions.json")["repositories"]
            if r["name"] == "DiffGRM"]
    if len(code) != 1:
        raise ValueError("Expected one pinned engine source snapshot")
    snapshot = repro.under(ROOT, code[0]["snapshot"])
    if repro.sha(snapshot) != code[0]["sha256"]:
        raise ValueError("Pinned baseline source archive changed")
    formal, sources = replay.load_formal_config(recipe, pack, snapshot, yaml)
    config_audit = replay.audit_formal_config(formal, recipe, metadata)
    lock_path = pack / "reproducibility/assets.lock.json"
    lock = read_json(lock_path)["artifacts"]
    names = list(dict.fromkeys(recipe["required_artifacts"] + recipe["tokenizer_artifacts"]))
    inputs = output.parent / "inputs"
    workspace = work / "workspace"
    records = []
    for name in names:
        digest = lock[name]["sha256"]
        immutable = repro.under(inputs, name)
        copy_bound(repro.under(project, name), immutable, digest)
        # Each mode gets its own writable dataset/tokenization cache. Only the
        # immutable input copy is shared; generated caches never cross modes.
        copy_bound(immutable, repro.under(workspace, name), digest)
        records.append({"path": name, "sha256": digest,
                        "size_bytes": immutable.stat().st_size})
    data_audit = replay.dataset_audit(workspace, formal, recipe)
    if replay.expected_runtime_sid(formal, recipe) != recipe["initial_sid"]:
        raise ValueError("Unexpected tokenizer SID naming change")
    resolved = replay.configure_paths(formal, workspace, work, recipe, actual_run_id)
    source_files = [engine / "genrec/models/DIFF_GRM" / name for name in
                    ("model.py", "item_local_attention.py", "trainer.py", "training_state.py")]
    source_hashes = {str(path.relative_to(engine)): repro.sha(path) for path in source_files}
    resolved.update(
        item_local_attention=attention, run_local_time=actual_run_id,
        training_state_dir=str(work / "training_state"),
        resume_training_state=str(Path(args.resume).resolve()) if isinstance(args.resume, str) else bool(args.resume),
        training_deadline_unix_seconds=args.deadline_unix,
        training_deadline_reserve_seconds=args.training_deadline_reserve_seconds,
        training_state_identity={"engine_files": source_hashes,
                                 "inputs": {r["path"]: r["sha256"] for r in records}},
    )
    if args.mode == "smoke":
        resolved.update(epochs=1, max_train_batches_per_epoch=32, max_eval_batches=8,
                        eval_start_epoch=1, eval_interval=1, patience=1)
    elif args.mode == "profile":
        # Retain all examples and the formal warmup. The first 104 updates
        # match the formal schedule despite this isolated one-epoch profile.
        resolved.update(epochs=1, max_train_batches_per_epoch=None, max_eval_batches=None,
                        eval_start_epoch=1, eval_interval=1, patience=1)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    # A run directory cannot silently change identity between prepare/resume.
    identity = {"run_id": actual_run_id, "study_sha256": repro.sha(args.config),
                "sid_sha256": lock[recipe["initial_sid"]]["sha256"],
                "baseline_run_id": recipe["run_id"]}
    if repair:
        identity.update(repair_campaign=REPAIR_CAMPAIGN, variant=study["variant"], repaired_engine_commit=REPAIR_ENGINE)
    identity_path = work / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Existing output directory belongs to a different run")
    attempt = work / "attempts" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{os.getpid()}_{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    record_attempt(work, attempt, "identity.json", identity)
    record_attempt(work, attempt, "study.json", study)
    record_attempt(work, attempt, "baseline_resolved_config.json", formal)
    record_attempt(work, attempt, "resolved_config.json", resolved)
    record_attempt(work, attempt, "input_identities.json", {
        "immutable_root": str(inputs), "runtime_workspace": str(workspace),
        "asset_lock_sha256": repro.sha(lock_path), "files": records,
        "sid_path": str(repro.under(workspace, recipe["initial_sid"])),
        "provenance": "Migration assets; original runtime input hash was not recorded",
    })
    record_attempt(work, attempt, "protocol_audit.json", {
        "configuration": config_audit, "dataset": data_audit,
        "sources": sources, "baseline_code_snapshot": code[0],
        "recipe_sha256": repro.sha(pack / "reproducibility/recipes/pse_seed2024.json"),
        "mode_overrides": {key: {"historical": formal.get(key), "resolved": value}
                           for key, value in resolved.items() if value != formal.get(key)},
    })
    record_attempt(work, attempt, "code_identity.json", {
        "project": git_record(ROOT), "engine": git_record(engine),
        "files": source_hashes,
        "runner_sha256": repro.sha(Path(__file__)),
        "historical_runtime_commit": None,
    })
    return {"work": work, "workspace": workspace, "records": records,
            "config": resolved, "recipe": recipe, "baseline": baseline_row(pack),
            "engine": engine, "attempt": attempt, "repair": repair, "study": study}


def finite_metrics(results):
    required = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10")
    normalized = dict(results)
    for k in (5, 10):
        if f"hit_rate@{k}" not in normalized and f"recall@{k}" in normalized:
            normalized[f"hit_rate@{k}"] = normalized[f"recall@{k}"]
    for key in required:
        if key not in normalized or not math.isfinite(float(normalized[key])):
            raise ValueError(f"Missing/nonfinite metric: {key}")
    return {key: float(value) for key, value in normalized.items()}


def run(args, prepared):
    import torch
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke/profile/formal require exactly one visible CUDA GPU")
    if "H100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("The authorized GPU allocation is one H100")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires BF16 support")
    work, config, engine = prepared["work"], prepared["config"], prepared["engine"]
    attempt = prepared["attempt"]
    repair = prepared.get("repair", False)
    initial_reserve = 0.0 if (work / "FIT_COMPLETE.json").exists() else args.training_deadline_reserve_seconds
    if args.deadline_unix and time.time() >= args.deadline_unix - initial_reserve:
        raise RuntimeError("Insufficient remaining allocation before the training reserve")
    if (work / "TEST_STARTED.json").exists() and not (work / "TEST_RESULT.json").exists():
        raise RuntimeError("Final test was already started without a result receipt; refusing a duplicate test")
    if (work / "STARTED.json").exists() and not args.resume:
        raise RuntimeError("Run was already started; inspect its state and explicitly use --resume")
    runtime = work / "runtime"
    runtime.mkdir(exist_ok=True)
    os.chdir(runtime)
    os.environ.update(
        HF_HOME=str(work / "huggingface"), HF_DATASETS_CACHE=str(work / "huggingface/datasets"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        TORCHDYNAMO_DISABLE="1", WANDB_MODE="disabled", PYTHONNOUSERSITE="1",
    )
    sys.path.insert(0, str(engine))
    import genrec.utils as utils
    utils.get_local_time = lambda: config["run_id"]
    from genrec.pipeline import Pipeline
    pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", config_dict=config)
    trainer = pipeline.trainer
    timing_path = work / "evaluation_timings.json"
    evaluation_timings = read_json(timing_path) if timing_path.exists() else []
    if not isinstance(evaluation_timings, list):
        raise ValueError("Existing evaluation timings are not an appendable list")
    timing_start = len(evaluation_timings)
    evaluate_original = trainer.evaluate

    def evaluate_timed(*positional, **keywords):
        torch.cuda.synchronize()
        beginning = time.perf_counter()
        answer = evaluate_original(*positional, **keywords)
        torch.cuda.synchronize()
        evaluation_timings.append({
            "split": keywords.get("split", "test"), "epoch": keywords.get("epoch"),
            "seconds": time.perf_counter() - beginning, "attempt": attempt.name,
        })
        # Existing entries retain their exact values; only append new calls.
        write_json(timing_path, evaluation_timings)
        write_json(attempt / "evaluation_timings.json", evaluation_timings[timing_start:])
        return answer

    trainer.evaluate = evaluate_timed
    actual = {key: value for key, value in pipeline.config.items() if key != "accelerator"}
    record_attempt(work, attempt, "parsed_config.json", actual)
    if pipeline.config["run_local_time"] != config["run_id"]:
        raise RuntimeError("Stable run identity was not applied before trainer construction")
    if Path(pipeline.tokenizer.semantic_ids_path).resolve() != Path(read_json(work / "input_identities.json")["sid_path"]).resolve():
        raise RuntimeError("Tokenizer consumed a different SID path")
    replay.verify_copies(prepared["workspace"], prepared["records"])
    record_attempt(work, attempt, "environment.json", {
        "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
    })
    module = pipeline.model.item_local_attention
    initial = {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()} if module is not None else {}
    initial_all = {name: parameter.detach().cpu().clone() for name, parameter in pipeline.model.named_parameters()} if repair else {}
    record_attempt(work, attempt, "model_identity.json", {
        "total_parameters": sum(p.numel() for p in pipeline.model.parameters()),
        "added_parameters": sum(p.numel() for p in module.parameters()) if module is not None else 0,
        "item_local_attention_enabled": module is not None,
        "location": "history SID embeddings, before concatenation and item_mlp" if module is not None else "control: original item_mlp path",
        "input_output": "[B,L,4,256] -> [B*L,4,256] -> [B,L,4,256]" if module is not None else "[B,L,4,256] -> concatenate -> item_mlp",
    })
    loaders = {}
    for split in ("train", "val"):
        loaders[split] = DataLoader(
            pipeline.tokenized_datasets[split],
            batch_size=pipeline.config["train_batch_size" if split == "train" else "eval_batch_size"],
            shuffle=split == "train", collate_fn=pipeline.tokenizer.collate_fn[split],
        )
    initial_identity = None
    if repair:
        from genrec.models.DIFF_GRM.training_state import capture_rng
        base_summary = nested_summary({name: value for name, value in pipeline.model.state_dict().items()
                                       if not name.startswith("item_local_attention.")})
        # Move before installing hooks so their Parameter objects are the ones
        # actually consumed by the Accelerator-prepared training loop.
        pipeline.model.to(pipeline.accelerator.device)
        rng_summary = nested_summary(capture_rng(loaders["train"]))
        initial_identity = {"initial_base_state_sha256": summary_sha(base_summary),
                            "initial_rng_sha256": summary_sha(rng_summary),
                            "base_tensors_excluding_item_local_attention": base_summary,
                            "initial_rng": rng_summary,
                            "scope": "fresh model and raw training loader immediately before fit; global RNG supplies a loader without its own generator",
                            "resume_requested": bool(args.resume), "repair_campaign": REPAIR_CAMPAIGN,
                            "variant": prepared["study"]["variant"]}
        record_attempt(work, attempt, "initial_state_identity.json", initial_identity)
    started = time.perf_counter()
    record_attempt(work, attempt, "STARTED.json", {"mode": args.mode, "utc": utc_now(), "pid": os.getpid(),
                                                  "resume": args.resume, "deadline_unix": args.deadline_unix})
    try:
        fit_receipt = work / "FIT_COMPLETE.json"
        if fit_receipt.exists():
            fit = read_json(fit_receipt)
            if not trainer._resume_payload or not trainer._resume_payload["progress"]["training_complete"]:
                raise RuntimeError("FIT_COMPLETE requires the matching terminal recovery state")
        # Terminal recovery restores optimizer/scheduler/precision wrapping,
        # then skips every training epoch; it does not repeat a finished fit.
        tracking = gradient_hooks(pipeline.model) if repair else None
        try:
            best_epoch, best_score = trainer.fit(loaders["train"], loaders["val"])
        finally:
            if tracking is not None:
                for handle in tracking[2]:
                    handle.remove()
        if getattr(trainer, "training_incomplete", False):
            raise RuntimeError("Trainer stopped incomplete; final testing is forbidden")
        if best_epoch is None or best_epoch <= 0 or not math.isfinite(float(best_score)):
            raise RuntimeError("Training did not produce a validation-selected checkpoint")
        training_integrity = None
        if repair:
            # A terminal fit resumed for publication executes no backwards;
            # retain its already-recorded coverage instead of inventing calls.
            if trainer._resume_payload and trainer._resume_payload["progress"]["training_complete"]:
                prior_integrity = work / "training_integrity.json"
                if not prior_integrity.is_file():
                    raise RuntimeError("Terminal recovery lacks prior real gradient-coverage evidence")
                training_integrity = read_json(prior_integrity)
                if not training_integrity.get("all_gradient_coverage"):
                    raise RuntimeError("Terminal recovery has incomplete prior gradient coverage")
            else:
                training_integrity = gradient_receipt(tracking[0], tracking[1])
            terminal = torch.load(work / "training_state/last.pt", map_location="cpu", weights_only=False)
            if not terminal["progress"]["training_complete"]:
                raise RuntimeError("Repaired fit did not persist a complete terminal training state")
            training_integrity.update(optimizer_receipt(terminal, pipeline.model, terminal["progress"]["cumulative_optimizer_updates"]))
            write_json(work / "training_integrity.json", training_integrity)
        if fit_receipt.exists():
            if int(best_epoch) != fit["best_epoch"] or float(best_score) != fit["best_val_score"]:
                raise RuntimeError("Recovered terminal fit disagrees with its receipt")
        else:
            fit = {"best_epoch": int(best_epoch), "best_val_score": float(best_score),
                   "fit_elapsed_seconds": trainer.fit_elapsed_seconds,
                   "fit_validation_seconds": sum(row["seconds"] for row in evaluation_timings[timing_start:]),
                   "fit_validation_timing_attempt": attempt.name,
                   "completed_utc": utc_now()}
            write_json(fit_receipt, fit)
        checkpoint = Path(trainer.saved_model_ckpt).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise RuntimeError("Checkpoint contains nonfinite model tensors")
        model = pipeline.accelerator.unwrap_model(trainer.model)
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(tensor.detach().cpu(), state[name]) for name, tensor in model.state_dict().items()):
            raise RuntimeError("Strict checkpoint reload does not match saved weights")
        model.to(pipeline.accelerator.device)
        trainer.model = model
        changed = [name for name, parameter in model.item_local_attention.named_parameters()
                   if not torch.equal(initial[name], parameter.detach().cpu())] if model.item_local_attention is not None else []
        if model.item_local_attention is not None and not changed:
            raise RuntimeError("New attention parameters never changed")
        changed_all = [name for name, parameter in model.named_parameters()
                       if not torch.equal(initial_all[name], parameter.detach().cpu())] if repair else []
        if repair and len(changed_all) != len(initial_all):
            raise RuntimeError("Not every repaired-model parameter tensor changed from its fresh initialization")
        record_attempt(work, attempt, "checkpoint_reload.json", {
            "path": str(checkpoint), "sha256": repro.sha(checkpoint),
            "strict_all_tensor_match": True, "changed_local_parameters": changed,
            "changed_parameter_tensors": len(changed_all) if repair else None,
        })
        # Fit's reserve is for evaluation. Once training is complete, consume
        # that reserved interval while preserving the actual allocation cap.
        pipeline.config["training_deadline_reserve_seconds"] = 0.0
        record_attempt(work, attempt, "runtime_phase.json", {
            "phase": "smoke_profile" if args.mode == "smoke" else "profile_validation" if args.mode == "profile" else "final_test",
            "training_complete": True, "deadline_unix": args.deadline_unix,
            "training_reserve_seconds": args.training_deadline_reserve_seconds,
            "evaluation_reserve_seconds": 0.0, "utc": utc_now(),
        })
        if args.mode == "profile":
            training_steps = len(loaders["train"])
            with Path(trainer.training_metrics_csv).open(encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
            loops = trainer.epoch_timing_records
            if (training_steps != 1652 or len(history) != 1 or len(loops) != 1
                    or int(loops[0]["epoch"]) != 1 or int(loops[0]["batches"]) != training_steps
                    or int(history[0]["optimizer_updates"]) != 104
                    or not math.isfinite(float(history[0]["train_loss"]))):
                raise RuntimeError("Profile must finish exactly one full 1652-batch/104-update training epoch")
            train_seconds = float(loops[0]["seconds"])
            val_calls = [row for row in evaluation_timings[timing_start:] if row["split"] == "val" and row["epoch"] == 1]
            if len(val_calls) != 1 or not math.isfinite(train_seconds) or train_seconds <= 0:
                raise RuntimeError("Profile lacks exact full-loop/full-validation timing evidence")
            full_val_seconds = float(val_calls[0]["seconds"])
            if not math.isfinite(full_val_seconds) or full_val_seconds <= 0:
                raise RuntimeError("Profile full validation duration is invalid")
            with Path(trainer.evaluation_metrics_csv).open(encoding="utf-8", newline="") as handle:
                val_rows = [row for row in csv.DictReader(handle) if row["split"] == "val" and int(row["epoch"]) == 1]
            if len({row["metric"] for row in val_rows}) != len(val_rows):
                raise RuntimeError("Profile validation contains duplicate metric events")
            validation = finite_metrics({row["metric"]: float(row["value"]) for row in val_rows})
            with Path(trainer.position_metrics_csv).open(encoding="utf-8", newline="") as handle:
                position_rows = [row for row in csv.DictReader(handle) if row["split"] == "val" and int(row["epoch"]) == 1]
            if (len(position_rows) != 4 or {int(row["position"]) for row in position_rows} != set(range(4))
                    or any(int(row["supervised_tokens"]) != 22363 or int(row["available_views"]) != 22363 for row in position_rows)):
                raise RuntimeError("Profile validation did not cover all 22363 targets at every SID position")
            recovery_started = time.perf_counter()
            recovery_check = profile_recovery_check(pipeline, trainer, work)
            recovery_check["wall_seconds"] = time.perf_counter() - recovery_started
            val_loader = pipeline.accelerator.prepare(loaders["val"])
            old_limit = pipeline.config["max_eval_batches"]
            old_benchmark = pipeline.config["vectorized_beam_search"].get("benchmark")
            pipeline.config["max_eval_batches"] = 2
            pipeline.config["vectorized_beam_search"]["benchmark"] = copy.deepcopy(pipeline.config["vectorized_beam_search"]["test"])
            try:
                trainer._check_training_deadline()
                torch.cuda.synchronize()
                beam_started = time.perf_counter()
                trainer.evaluate(val_loader, split="benchmark", epoch=1)
                torch.cuda.synchronize()
                beam_seconds = time.perf_counter() - beam_started
            finally:
                pipeline.config["max_eval_batches"] = old_limit
                if old_benchmark is None:
                    pipeline.config["vectorized_beam_search"].pop("benchmark", None)
                else:
                    pipeline.config["vectorized_beam_search"]["benchmark"] = old_benchmark
            if not math.isfinite(beam_seconds) or beam_seconds <= 0:
                raise RuntimeError("Profile beam256 validation benchmark duration is invalid")
            if any(row["split"] == "test" for row in evaluation_timings) or (work / "TEST_STARTED.json").exists():
                raise RuntimeError("Profile must never evaluate the final test data")
            result = {"status": "profile_completed", **fit, "repair_campaign": REPAIR_CAMPAIGN,
                      "variant": prepared["study"]["variant"], "validation_results": validation,
                      "training_loop_seconds": train_seconds, "formal_train_batches_per_epoch": training_steps,
                      "formal_eval_batches": math.ceil(len(pipeline.tokenized_datasets["val"]) / pipeline.config["eval_batch_size"]),
                      "full_val_seconds": full_val_seconds, "beam256_val_2_batches_seconds": beam_seconds,
                      "initial_base_state_sha256": initial_identity["initial_base_state_sha256"],
                      "initial_rng_sha256": initial_identity["initial_rng_sha256"],
                      "all_gradient_coverage": training_integrity["all_gradient_coverage"],
                      "optimizer_state_count": training_integrity["optimizer_state_count"],
                      "all_parameter_tensors": training_integrity["all_parameter_tensors"],
                      "training_integrity": training_integrity, "actual_changed_parameter_tensors": len(changed_all),
                      "recovery_check": recovery_check, "test_data_evaluated": False,
                      "checkpoint_sha256": repro.sha(checkpoint), "process_wall_seconds": time.perf_counter() - started,
                      "limitations": "One full training epoch and validation only; isolated manual recovery/update probe, not CUDA bitwise replay or formal recommendation evidence"}
            write_json(work / "profile_results.json", result)
        elif args.mode == "smoke":
            with Path(trainer.training_metrics_csv).open(encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
            if len(history) != 1 or int(history[0]["optimizer_updates"]) != 2 or not math.isfinite(float(history[0]["train_loss"])):
                raise RuntimeError("Smoke must finish 32 batches/two updates with finite loss")
            val_loader = pipeline.accelerator.prepare(loaders["val"])
            torch.cuda.synchronize()
            val_started = time.perf_counter()
            validation = finite_metrics(trainer.evaluate(val_loader, split="val", epoch=1))
            torch.cuda.synchronize()
            val_seconds = time.perf_counter() - val_started
            # Same held-out validation rows, with the historical TEST beam.
            # This separate split name prevents test metadata/result writes.
            old_limit = pipeline.config["max_eval_batches"]
            pipeline.config["max_eval_batches"] = 2
            pipeline.config["vectorized_beam_search"]["benchmark"] = copy.deepcopy(
                pipeline.config["vectorized_beam_search"]["test"])
            torch.cuda.synchronize()
            profile_started = time.perf_counter()
            trainer.evaluate(val_loader, split="benchmark", epoch=1)
            torch.cuda.synchronize()
            profile_seconds = time.perf_counter() - profile_started
            pipeline.config["max_eval_batches"] = old_limit
            training_steps = math.ceil(len(loaders["train"]))
            loop_records = [row for row in trainer.epoch_timing_records
                            if int(row["epoch"]) == 1 and int(row["batches"]) == 32]
            if len(loop_records) != 1:
                raise RuntimeError("Missing the smoke's exact 32-batch training-loop timing")
            train_seconds = float(loop_records[0]["seconds"])
            if not math.isfinite(train_seconds) or train_seconds <= 0:
                raise RuntimeError("Smoke training-loop timing must be finite and positive")
            result = {"status": "smoke_passed_not_formal", **fit,
                      "train_batches": 32, "optimizer_updates": 2, "validation_results": validation,
                      "validation_8_batches_seconds": val_seconds,
                      "test_beam256_on_validation_2_batches_seconds": profile_seconds,
                      "training_seconds_per_batch_estimate": train_seconds / 32,
                      "training_loop_seconds": train_seconds,
                      "training_timing_basis": "CUDA-synchronized batch loop; excludes checkpoint writes and validation",
                      "formal_train_batches_per_epoch": training_steps,
                      "formal_eval_batches": math.ceil(len(pipeline.tokenized_datasets["val"]) / pipeline.config["eval_batch_size"]),
                      "test_data_evaluated": False, "checkpoint_sha256": repro.sha(checkpoint),
                      "process_wall_seconds": time.perf_counter() - started}
            write_json(work / "smoke_results.json", result)
        else:
            if args.deadline_unix and time.time() >= args.deadline_unix:
                raise RuntimeError("Allocation deadline reached before final evaluation")
            result_receipt = work / "TEST_RESULT.json"
            if result_receipt.exists():
                test_results = finite_metrics(read_json(result_receipt)["test_results"])
            else:
                test_loader = DataLoader(
                    pipeline.tokenized_datasets["test"], batch_size=pipeline.config["eval_batch_size"],
                    shuffle=False, collate_fn=pipeline.tokenizer.collate_fn["test"],
                )
                test_loader = pipeline.accelerator.prepare(test_loader)
                with (work / "TEST_STARTED.json").open("x", encoding="utf-8") as handle:
                    json.dump({"utc": utc_now(), "checkpoint_sha256": repro.sha(checkpoint)}, handle)
                model.generate_w_decoding_graph = True
                test_results = finite_metrics(trainer.evaluate(test_loader, split="test"))
                write_json(result_receipt, {"test_results": test_results, "completed_utc": utc_now()})
            trainer._write_run_metadata(test_results)
            baseline = prepared["baseline"]
            comparison = {key: {"baseline": float(baseline[key]), "new": test_results[key],
                                "absolute_delta": test_results[key] - float(baseline[key]),
                                "relative_change": test_results[key] / float(baseline[key]) - 1.0}
                          for key in ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10")}
            signal = comparison["ndcg@10"]["relative_change"] >= .03 and comparison["hit_rate@10"]["absolute_delta"] >= 0
            result = {"status": "formal_completed", **fit, "test_results": test_results,
                      "baseline_comparison": comparison, "exploratory_positive_signal": signal,
                      "signal_rule": "relative NDCG@10 >=3% and HR@10 does not decrease",
                      "evaluation_level": "sid", "checkpoint_sha256": repro.sha(checkpoint),
                      "process_wall_seconds": time.perf_counter() - started,
                      "limitations": "Single seed versus historical baseline; original runtime commit/full config/RNG were not recorded; not statistical significance or strict item-level evidence"}
            if repair:
                result.update(repair_campaign=REPAIR_CAMPAIGN, variant=prepared["study"]["variant"],
                              initial_base_state_sha256=initial_identity["initial_base_state_sha256"],
                              initial_rng_sha256=initial_identity["initial_rng_sha256"],
                              training_integrity=training_integrity,
                              all_gradient_coverage=training_integrity["all_gradient_coverage"],
                              optimizer_state_count=training_integrity["optimizer_state_count"],
                              all_parameter_tensors=training_integrity["all_parameter_tensors"],
                              actual_changed_parameter_tensors=len(changed_all),
                              baseline_comparison_role="historical_background_only",
                              primary_comparison="Repaired control versus repaired item_local, compared by the campaign controller",
                              limitations="Single-seed repaired A/B campaign; old Stage03 metrics are historical background only; SID-level evaluation, not statistical significance")
                result["historical_background_positive_signal"] = result.pop("exploratory_positive_signal")
                result["historical_background_signal_rule"] = result.pop("signal_rule")
            write_json(work / "results.json", result)
        replay.verify_copies(prepared["workspace"], prepared["records"])
        complete = {"mode": args.mode, "status": result["status"],
                    "completed_utc": utc_now(), "checkpoint_sha256": repro.sha(checkpoint)}
        if repair:
            result_name = "profile_results.json" if args.mode == "profile" else "smoke_results.json" if args.mode == "smoke" else "results.json"
            complete.update(repair_campaign=REPAIR_CAMPAIGN, variant=prepared["study"]["variant"],
                            result_file=result_name, result_sha256=repro.sha(work / result_name))
        write_json(work / "COMPLETE.json", complete)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        write_json(work / "INCOMPLETE.json", {
            "utc": utc_now(), "error": str(error), "error_type": type(error).__name__,
            "training_incomplete": bool(getattr(trainer, "training_incomplete", False)),
            "training_stop_reason": getattr(trainer, "training_stop_reason", None),
            "resume_state": str(work / "training_state/last.pt"),
            "process_wall_seconds": time.perf_counter() - started,
        })
        raise
    finally:
        trainer.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "profile", "formal"), required=True)
    parser.add_argument("--project-root", required=True, help="Read-only historical asset project")
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True, help="Formal run directory; smoke/profile use separate smoke/ or profile/ children")
    parser.add_argument("--config", default=str(ROOT / "configs/07_item_local_attention/item_local_seed2024.json"))
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume last.pt, or an explicit state path")
    parser.add_argument("--deadline-unix", type=float, default=None)
    parser.add_argument("--training-deadline-reserve-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.training_deadline_reserve_seconds < 0:
        parser.error("Deadline reserve cannot be negative")
    prepared = prepare(args)
    if prepared.get("already_complete"):
        print(json.dumps(prepared["already_complete"], ensure_ascii=False, indent=2))
        return
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared_cpu_only", "output": str(prepared["work"])}, indent=2))
        return
    run(args, prepared)


if __name__ == "__main__":
    main()
