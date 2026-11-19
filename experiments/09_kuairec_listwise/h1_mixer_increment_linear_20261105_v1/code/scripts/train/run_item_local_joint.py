"""Hash-bound Refined-PSE / DiffGRM alternating joint-training experiment.

The optional item-local block belongs to the recommender, after history SID
embeddings and before item_mlp.  DIGER-ST carries recommendation gradients
through the frozen recommender into the tokenizer during tokenizer phases.
CPU preparation never imports torch or initializes CUDA.  The earlier fixed-
SID runner is deliberately neither imported nor used as a fallback.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import MethodType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/repro"))
import stage05_repro as stage05

CAMPAIGN = "stage07_item_local_alternating_joint_seed2024_20261020_r1"
VARIANTS = ("control", "item_local")
ENGINE_FILES = (
    "genrec/models/DIFF_GRM/model.py", "genrec/models/DIFF_GRM/trainer.py",
    "genrec/models/DIFF_GRM/training_state.py", "genrec/models/DIFF_GRM/joint_cross_pse.py",
    "genrec/models/DIFF_GRM/cross_pse.py", "genrec/models/DIFF_GRM/item_local_attention.py",
    "genrec/models/DIFF_GRM/tokenizer.py", "genrec/models/DIFF_GRM/collate.py",
    "genrec/models/DIFF_GRM/config.yaml", "genrec/default.yaml",
    "genrec/datasets/AmazonReviews2014/config.yaml", "genrec/utils.py", "genrec/pipeline.py",
)
PHASE_COLUMNS = ("epoch", "phase", "optimizer_updates", "direct_recommendation_gradient_norm",
                 "active_parameter_tensors", "changed_parameter_tensors", "frozen_parameter_tensors",
                 "frozen_parameters_unchanged", "all_active_gradients_finite", "sid_sha256")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def first_json(path, value):
    """Preserve first metadata; execution attempts have their own full copies."""
    path = Path(path)
    if not path.exists():
        write_json(path, value)


@contextmanager
def execution_guard(work):
    """The operating system releases this advisory lock on process death."""
    with (work / "EXECUTION.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sha(path):
    return stage05.sha(path)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def run_id(variant):
    need(variant in VARIANTS, "Unknown joint-training variant")
    return f"stage07_item_local_joint_{variant}_seed2024_20261020_r1"


def validate_study(study):
    allowed = {"schema_version", "campaign_id", "protocol_id", "seed", "recipe_id", "variants",
               "max_gpu_hours", "prior_allocated_gpu_seconds", "gpu_type", "gpu_count",
               "initialization", "local_attention_location", "training", "global_codebook_mixer_enabled",
               "tokenizer_codebooks_trainable", "recommendation_gradient_bridge", "evaluation_level",
               "historical_results_role", "signal_rule"}
    need(not set(study) - allowed, f"Unknown campaign settings: {sorted(set(study) - allowed)}")
    need(study.get("campaign_id") == CAMPAIGN and study.get("protocol_id") == CAMPAIGN,
         "A separate, exact alternating-joint campaign identity is required")
    need(study.get("seed") == 2024 and study.get("recipe_id") == "refined_joint",
         "Only the locked seed2024 Refined joint warm start is supported")
    need(study.get("gpu_type") == "H100" and study.get("gpu_count") == 1,
         "This campaign requires one H100")
    need(study.get("max_gpu_hours") == 28.0, "Keep the authorized cumulative 28 GPU-hour cap")
    prior = study.get("prior_allocated_gpu_seconds")
    need(isinstance(prior, (int, float)) and not isinstance(prior, bool)
         and math.isfinite(prior) and 11637 <= prior < 28 * 3600,
         "Prior allocation must include all 11637 seconds already used")
    expected = {name: {"run_id": run_id(name), "item_local_attention": {
        "enabled": name == "item_local", "num_heads": 4, "dropout": 0.0}} for name in VARIANTS}
    need(json_sha(study.get("variants")) == json_sha(expected), "Both joint variants and their sole local-attention difference must be exact")
    for name, value in {"global_codebook_mixer_enabled": False, "tokenizer_codebooks_trainable": True,
                        "recommendation_gradient_bridge": "diger_st", "evaluation_level": "sid",
                        "historical_results_role": "background only"}.items():
        need(name not in study or study[name] == value, f"Campaign declaration contradicts the recipe: {name}")
    return study


def check_output(output, variant):
    raw = Path(output).expanduser()
    for parent in (raw, *raw.parents):
        need(not parent.is_symlink(), f"Symlink output/ancestor is unsupported: {parent}")
    output = raw.resolve()
    need(output.name == run_id(variant) and output.parent.name == "runs",
         "Use goal/runs/<the exact variant run_id>")
    return output


def git_identity(directory, *, expected=None, require_clean=False):
    directory = Path(directory)
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    head = result.stdout.strip()
    dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=normal"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if expected is not None:
        need(bool(re.fullmatch(r"[0-9a-f]{40}", expected)), "Expected engine commit must be a full 40-character SHA")
        need(head == expected, "Engine HEAD differs from --expected-engine-commit")
    if require_clean:
        need(not dirty, "GPU execution requires an engine with no tracked modifications")
    return {"head": head, "tracked_dirty": bool(dirty)}


def copy_bound(source, destination, digest):
    source, destination = Path(source), Path(destination)
    need(source.is_file() and not source.is_symlink(), f"Missing/nonregular locked input: {source}")
    need(sha(source) == digest, f"Locked input hash differs: {source}")
    if destination.exists():
        need(destination.is_file() and not destination.is_symlink() and sha(destination) == digest,
             f"Preserved destination input differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        need(sha(temporary) == digest, f"Input copy failed byte verification: {destination}")
        if destination.exists():
            need(sha(destination) == digest, f"Concurrent input copy differs: {destination}")
        else:
            os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_recipe(engine, package, recipe):
    """The same layer order/scalar conversion as stage05_repro.run_replay."""
    import yaml
    config = {"model": "DIFF_GRM", "dataset": "AmazonReviews2014"}
    layers = [engine / "genrec/default.yaml", engine / "genrec/datasets/AmazonReviews2014/config.yaml",
              engine / "genrec/models/DIFF_GRM/config.yaml"]
    layers += [stage05.under(package, name) for name in recipe["config_layers"]]
    for layer in layers:
        value = yaml.safe_load(layer.read_text(encoding="utf-8")) or {}
        need(isinstance(value, dict), f"Config layer is not a mapping: {layer}")
        config.update(value)
    for key, value in list(config.items()):
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                config[key] = value.lower() == "true"
            else:
                try:
                    config[key] = json.loads(value)
                except (ValueError, TypeError):
                    try:
                        config[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
    for key, value in (("joint_frozen_tokenizer_control", False), ("crosspse_mixer_enabled", True),
                       ("crosspse_train_base_codebooks", True)):
        config.setdefault(key, value)
    required = {"epochs": 45, "joint_recommender_warmup_epochs": 5, "joint_cycle_epochs": 4,
                "joint_tokenizer_epochs_per_cycle": 1, "joint_max_tokenizer_phases": 10,
                "joint_tokenizer_batches_per_phase": 96, "joint_crosspse_enabled": True,
                "joint_gradient_mode": "diger_st", "joint_frozen_tokenizer_control": False,
                "crosspse_mixer_enabled": False, "crosspse_train_base_codebooks": True,
                "joint_recommender_sia_weight": 0.0, "joint_tokenizer_sia_weight": 0.0,
                "joint_diger_recommendation_weight": 1.0, "n_digit": 4, "codebook_size": 256,
                "mixed_precision": "bf16", "train_batch_size": 64, "gradient_accumulation_steps": 16,
                "label_smoothing": 0.2, "evaluation_level": "sid", "eval_expand_sid_to_items": False}
    need(all(config.get(key) == value for key, value in required.items()), "Locked Refined joint recipe changed")
    audit = stage05.audit_formal_config(config, read_json(package / recipe["metadata"]))
    return config, audit, [{"path": str(path), "sha256": sha(path)} for path in layers]


def dataset_audit(workspace, sid_path):
    processed = workspace / "data/cache/AmazonReviews2014/Beauty/processed"
    mappings, sequences = read_json(processed / "id_mapping.json"), read_json(processed / "all_item_seqs.json")
    sids = read_json(sid_path)
    keys = mappings["id2item"][1:]
    need(set(keys) == set(sids) and len(keys) == len(sids) == 12101, "Locked SID and item catalogue differ")
    for digits in sids.values():
        need(len(digits) == 4 and all(type(digit) is int and 0 <= digit < 256 for digit in digits),
             "Invalid Refined SID geometry")
    lengths = [len(values) for values in sequences.values()]
    counts = {"real_users": len(lengths), "real_items": len(keys), "interactions": sum(lengths),
              "train_examples": sum(max(0, min(length - 3, 50) - 2 + 1) for length in lengths),
              "val_examples": sum(length > 1 for length in lengths), "test_examples": len(lengths)}
    need(counts == dict(real_users=22363, real_items=12101, interactions=198502,
                        train_examples=105668, val_examples=22363, test_examples=22363),
         "Dataset split/counts differ from the locked Beauty protocol")
    return {"status": "passed", "counts": counts, "basis": "SHA256-bound catalogue and exact sliding-window bounds"}


def verify_copies(prepared):
    for row in prepared["records"]:
        need(sha(stage05.under(prepared["workspace"], row["path"])) == row["sha256"],
             f"Workspace source input changed: {row['path']}")


def prepare(args):
    study_path = Path(args.config).resolve()
    study = validate_study(read_json(study_path))
    output = check_output(args.output, args.variant)
    work = output / "smoke" if args.mode == "smoke" else output
    need(not args.resume or args.mode == "formal", "Only formal may explicitly resume its own committed last state")
    own_last = work / "training_state/last.pt"
    if isinstance(args.resume, str):
        need(Path(args.resume).resolve() == own_last.resolve(), "Resume may use only this run's own training_state/last.pt")
    if args.resume:
        need(own_last.is_file(), "Explicit resume has no committed own last.pt")
    engine, project = Path(args.engine_root).resolve(), Path(args.project_root).resolve()
    gpu = args.mode != "prepare"
    need(not gpu or args.expected_engine_commit is not None, "GPU modes require --expected-engine-commit")
    engine_git = git_identity(engine, expected=args.expected_engine_commit, require_clean=gpu)
    sources = {name: sha(engine / name) for name in ENGINE_FILES}
    package = ROOT / "experiments/05_crosspse_joint"
    recipe = stage05.load_recipe(package, "refined_joint")
    formal, config_audit, layers = merge_recipe(engine, package, recipe)
    lock_path = package / "reproducibility/assets.lock.json"
    lock = read_json(lock_path)["artifacts"]
    need(len(recipe["required_artifacts"]) == 13 and not recipe["missing_from_archive"], "Expected all 13 Refined warm-start inputs")
    identity = {"campaign_id": CAMPAIGN, "protocol_id": CAMPAIGN, "recipe_id": "refined_joint",
                "variant": args.variant, "seed": 2024, "formal_run_id": run_id(args.variant),
                "run_id": run_id(args.variant) + ("_smoke" if args.mode == "smoke" else ""),
                "study_sha256": sha(study_path), "engine_commit": engine_git["head"],
                "engine_source_sha256": sources}
    if (work / "identity.json").exists():
        need(read_json(work / "identity.json") == identity, "Existing run identity/source/config differs")
    elif work.exists():
        allowed = {"smoke"} if args.mode != "smoke" else set()
        need(not set(child.name for child in work.iterdir()) - allowed, "Refusing an unrelated/nonempty run directory")
    if (work / "COMPLETE.json").exists():
        complete = read_json(work / "COMPLETE.json")
        expected_mode = "smoke" if args.mode == "smoke" else "formal"
        result_name = "smoke_results.json" if expected_mode == "smoke" else "results.json"
        need(complete.get("mode") == expected_mode and complete.get("campaign_id") == CAMPAIGN
             and complete.get("variant") == args.variant and complete.get("run_id") == identity["run_id"]
             and complete.get("result_file") == result_name, "Existing completion belongs to another mode/campaign/run")
        result_path = work / result_name
        need(complete.get("result_sha256") == sha(result_path), "Existing completion/result SHA is damaged")
        need(complete.get("checkpoint_sha256") == sha(work / "training_state/best_model.bin"), "Existing completed best checkpoint is damaged")
        return {"already_complete": complete, "work": work}
    if gpu and (work / "STARTED.json").exists():
        need(bool(args.resume), "Run already started; inspect evidence and explicitly resume instead of duplicate training")
    if (work / "TEST_STARTED.json").exists():
        need((work / "TEST_RESULT.json").exists(), "Test already started without a result; duplicate final evaluation is forbidden")
    work.mkdir(parents=True, exist_ok=True)
    inputs, workspace = output.parent.parent / "inputs", work / "workspace"
    need(not inputs.is_symlink() and not workspace.is_symlink(), "Owned input/workspace roots must not be symlinks")
    records = []
    for name in recipe["required_artifacts"]:
        digest = lock[name]["sha256"]
        immutable = stage05.under(inputs, name)
        copy_bound(stage05.under(project, name), immutable, digest)
        copy_bound(immutable, stage05.under(workspace, name), digest)
        records.append({"path": name, "sha256": digest, "size_bytes": immutable.stat().st_size})
    artifact = stage05.under(workspace, recipe["initial_tokenizer_dir"])
    config = copy.deepcopy(formal)
    config.update(run_id=identity["run_id"], rand_seed=2024, run_local_time=identity["run_id"],
                  joint_recommender_checkpoint=str(stage05.under(workspace, recipe["initial_decoder"])),
                  joint_crosspse_artifact_dir=str(artifact), joint_crosspse_checkpoint=str(artifact / "best.pt"),
                  joint_crosspse_slots_path=str(artifact / "opq_residual_embeddings.npy"),
                  joint_crosspse_items_path=str(workspace / "data/diff_formal_embeddings/beauty_item_keys.json"),
                  precomputed_semantic_ids_path=str(artifact / "semantic_ids.json"),
                  crosspse_summary_path=str(artifact / "summary.json"),
                  item_local_attention=study["variants"][args.variant]["item_local_attention"],
                  cache_dir=str(workspace / "data/cache"), training_state_dir=str(work / "training_state"),
                  resume_training_state=str(own_last) if args.resume else False,
                  training_deadline_unix_seconds=args.deadline_unix,
                  training_deadline_reserve_seconds=args.training_deadline_reserve_seconds,
                  training_state_identity={"campaign_id": CAMPAIGN, "variant": args.variant,
                                           "inputs": {row["path"]: row["sha256"] for row in records}, "sources": sources})
    for key, subdir in (("log_dir", "logs"), ("tensorboard_log_dir", "tensorboard"), ("ckpt_dir", "checkpoints"),
                        ("run_metadata_dir", "run_metadata"), ("diffusion_diagnostics_log_dir", "diagnostics")):
        config[key] = str(work / subdir)
    if args.mode == "smoke":
        config.update(epochs=3, max_train_batches_per_epoch=32, max_eval_batches=8,
                      joint_recommender_warmup_epochs=1, joint_cycle_epochs=2,
                      joint_tokenizer_epochs_per_cycle=1, joint_max_tokenizer_phases=1,
                      joint_tokenizer_batches_per_phase=32, warmup_steps=1,
                      eval_start_epoch=1, eval_interval=1, patience=0)
    input_identity = {"artifacts": records, "assets_lock_sha256": sha(lock_path),
                      "initial_decoder": recipe["initial_decoder"], "initial_tokenizer_dir": recipe["initial_tokenizer_dir"]}
    counts = dataset_audit(workspace, artifact / "semantic_ids.json")
    attempt = work / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    values = {"identity.json": identity, "study.json": study, "recipe.json": recipe,
              "input_identities.json": input_identity, "resolved_config.json": config,
              "formal_configuration_audit.json": config_audit, "dataset_audit.json": counts,
              "code_identity.json": {"engine": engine_git, "engine_sources": sources,
                                     "runner_sha256": sha(Path(__file__)), "stage05_helper_sha256": sha(Path(stage05.__file__)),
                                     "configuration_layers": layers}}
    for name, value in values.items():
        write_json(attempt / name, value)
        first_json(work / name, value)
    write_json(attempt / "PREPARED.json", {"utc": utc_now(), "mode": args.mode, "cuda_initialized": False})
    return {"work": work, "attempt": attempt, "config": config, "identity": identity,
            "study": study, "recipe": recipe, "workspace": workspace, "engine": engine, "records": records}


def nested_summary(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): nested_summary(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (tuple, list)):
        return [nested_summary(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def require_same(actual, expected, label):
    need(nested_summary(actual) == nested_summary(expected), f"Recovered {label} differs")


def expected_parameters(model, phase):
    if phase == "tokenizer":
        return {"joint_alignment.mixer.base_codebooks"}
    return {name for name, _ in model.named_parameters()
            if not name.startswith(("joint_alignment.", "joint_sequence_adapter."))}


def sid_integrity(model, *, nearest=False):
    import torch
    bridge = model.joint_alignment
    codes = bridge.hard_codes.detach().cpu().tolist()
    for item_id, item in enumerate(model.tokenizer.id2item[1:], 1):
        expected = tuple(codes[item_id][digit] + model.tokenizer.sid_offset + digit * model.codebook_size
                         for digit in range(model.n_digit))
        need(model.tokenizer.item2tokens[item] == expected, "Tokenizer reverse catalogue is stale after SID refresh")
    if nearest:
        with torch.no_grad():
            books, _ = bridge.mixer.crossed_codebooks(return_attention=False)
            for start in range(1, bridge.item_slots.shape[0], 2048):
                slots = bridge.item_slots[start:start + 2048].float()
                current = bridge.mixer.squared_distances(slots, books.float()).argmin(-1)
                need(torch.equal(current, bridge.hard_codes[start:start + len(slots)]),
                     "Refreshed hard codes differ from current deterministic nearest assignments")
    return {"hard_codes_sha256": model.joint_codes_sha256(), "catalogue_mapping_matches": True,
            "nearest_assignments_verified": bool(nearest), "SID_change_required": False}


def optimizer_integrity(payload, model):
    """Check both Adam clocks by parameter identity, exempting disabled SIA."""
    import torch
    state, runtime = payload["optimizer"], payload["joint_runtime"]
    layout = runtime["optimizer_groups"]
    need([group["group_name"] for group in layout] == ["recommender", "joint_codebook"],
         "No global Mixer group is allowed in the Refined joint experiment")
    need(len(layout) == len(state["param_groups"]), "Joint optimizer group count differs")
    progress = payload["progress"]
    expected_steps = {"recommender": int(progress["cumulative_optimizer_updates"]),
                      "joint_codebook": int(progress["joint_progress"]["tokenizer_optimizer_updates"])}
    parameters, covered, exempt = dict(model.named_parameters()), [], []
    all_ids = []
    for definition, group in zip(layout, state["param_groups"]):
        names, identifiers = definition["parameters"], group["params"]
        need(group.get("group_name") == definition["group_name"] and len(names) == len(identifiers),
             "Joint optimizer named group layout differs")
        all_ids.extend(identifiers)
        for name, identifier in zip(names, identifiers):
            entry = state["state"].get(identifier)
            if name.startswith("joint_sequence_adapter."):
                need(entry is None, "Disabled SIA adapter unexpectedly received optimizer state")
                exempt.append(name)
                continue
            updates = expected_steps[definition["group_name"]]
            if not updates:
                need(entry is None, "Inactive optimizer clock has state before any updates")
                continue
            need(entry is not None and name in parameters, f"Missing active optimizer state: {name}")
            step = entry["step"].item() if torch.is_tensor(entry["step"]) else entry["step"]
            need(float(step) == updates, f"Wrong alternating optimizer clock for {name}: {step} vs {updates}")
            for moment in ("exp_avg", "exp_avg_sq"):
                need(tuple(entry[moment].shape) == tuple(parameters[name].shape)
                     and bool(torch.isfinite(entry[moment]).all()), f"Invalid Adam moments: {name}")
            covered.append(name)
    need(len(set(all_ids)) == len(all_ids) and set(state["state"]).issubset(set(all_ids)),
         "Duplicate or foreign optimizer-state identities")
    return {"status": "passed", "recommender_optimizer_updates": expected_steps["recommender"],
            "tokenizer_optimizer_updates": expected_steps["joint_codebook"],
            "covered_parameter_names": covered, "legitimate_inactive_SIA_parameters": exempt,
            "optimizer_state_count": len(state["state"]), "all_active_optimizer_steps_match": True}


class PhaseAudit:
    """Observe actual forwards/backwards/steps, without changing their losses."""
    def __init__(self, model, trainer, work, resume_epoch=0):
        import torch
        self.torch, self.model, self.trainer, self.work = torch, model, trainer, work
        self.records_path = work / "phase_integrity.json"
        self.records = read_json(self.records_path) if self.records_path.exists() else []
        tails = [row for row in self.records if row["epoch"] > resume_epoch]
        if tails:
            write_json(work / f"phase_integrity.interrupted_tail_{time.time_ns()}.json", tails)
            self.records = [row for row in self.records if row["epoch"] <= resume_epoch]
        self.current, self.before, self.calls, self.finite, self.direct = None, None, {}, {}, None
        self.handles = []
        allowed = expected_parameters(model, "recommender") | expected_parameters(model, "tokenizer")
        for name, parameter in model.named_parameters():
            if name not in allowed:
                continue
            need(parameter.requires_grad, f"Expected initially trainable parameter: {name}")
            def observe(gradient, key=name):
                if self.before is not None:
                    self.calls[key] = self.calls.get(key, 0) + 1
                    finite = torch.isfinite(gradient.detach()).all()
                    if key in self.finite:
                        self.finite[key].logical_and_(finite)
                    else:
                        self.finite[key] = finite
            self.handles.append(parameter.register_hook(observe))
        self.original_phase = model.set_joint_phase
        self.original_forward = model.forward
        self.original_tokenizer = model.forward_joint_tokenizer
        self.original_record = trainer._record_joint_metrics
        owner = self
        def set_phase(_model, phase):
            owner.original_phase(phase)
            owner.current = phase
        def forward(_model, batch, return_loss=True):
            if return_loss:
                owner.begin()
            return owner.original_forward(batch, return_loss=return_loss)
        def tokenizer(_model, batch, measure_direct_gradient=False):
            owner.begin()
            output = owner.original_tokenizer(batch, measure_direct_gradient=measure_direct_gradient)
            if owner.direct is None:
                codebook = _model.joint_alignment.mixer.base_codebooks
                gradient = torch.autograd.grad(output.recommendation_loss, codebook, retain_graph=True)[0]
                owner.direct = float(gradient.detach().float().norm())
                need(math.isfinite(owner.direct) and owner.direct > 0, "Recommendation-only gradient did not reach base codebooks")
            return output
        def record(_trainer, epoch, phase, means, grad_norm, updates, refresh, gradient_sides):
            owner.finish(epoch, phase, updates)
            return owner.original_record(epoch, phase, means, grad_norm, updates, refresh, gradient_sides)
        model.set_joint_phase = MethodType(set_phase, model)
        model.forward = MethodType(forward, model)
        model.forward_joint_tokenizer = MethodType(tokenizer, model)
        trainer._record_joint_metrics = MethodType(record, trainer)

    def begin(self):
        if self.before is not None:
            return
        need(self.current in ("tokenizer", "recommender"), "Training forward has no explicit joint phase")
        active = expected_parameters(self.model, self.current)
        for name, parameter in self.model.named_parameters():
            if name in active:
                need(parameter.requires_grad, f"Expected active parameter is frozen: {name}")
            elif self.current == "tokenizer" or name.startswith("joint_alignment."):
                need(not parameter.requires_grad, f"Inactive side was not frozen: {name}")
        self.before = {name: parameter.detach().cpu().clone() for name, parameter in self.model.named_parameters()}
        self.before_sid = self.model.joint_codes_sha256()
        self.calls, self.finite, self.direct = {}, {}, None

    def finish(self, epoch, phase, updates):
        need(self.before is not None and self.current == phase, "Missing actual phase forwards")
        active = expected_parameters(self.model, phase)
        parameters = dict(self.model.named_parameters())
        changed = {name for name, before in self.before.items()
                   if not self.torch.equal(before, parameters[name].detach().cpu())}
        frozen_changed = changed - active
        missing = sorted(name for name in active if self.calls.get(name, 0) == 0)
        nonfinite = sorted(name for name, flag in self.finite.items() if not bool(flag.item()))
        unexpected = sorted(name for name, count in self.calls.items() if count and name not in active)
        need(not frozen_changed and not missing and not nonfinite and not unexpected,
             f"Phase integrity failed: frozen_changed={sorted(frozen_changed)}, missing={missing}, nonfinite={nonfinite}, unexpected_gradients={unexpected}")
        need(bool(changed & active) and int(updates) > 0, "Actual optimizer updates did not change the active side")
        if phase == "tokenizer":
            need(self.direct is not None and self.direct > 0, "Tokenizer phase lacks recommendation-only direct-gradient evidence")
        else:
            local_names = {name for name in active if name.startswith("item_local_attention.")}
            need(not local_names or bool(changed & local_names), "Recommender updates did not change the local block")
            need(self.model.joint_codes_sha256() == self.before_sid, "Recommender phase changed the frozen hard SID snapshot")
        row = {"epoch": int(epoch), "phase": phase, "optimizer_updates": int(updates),
               "direct_recommendation_gradient_norm": self.direct, "active_parameter_tensors": len(active),
               "changed_parameter_tensors": len(changed), "changed_parameters": sorted(changed),
               "frozen_parameter_tensors": len(parameters) - len(active), "frozen_parameters_unchanged": True,
               "all_active_gradients_finite": True, "active_gradient_hook_calls": self.calls,
               "sid_sha256": self.model.joint_codes_sha256(),
               "SID_integrity": sid_integrity(self.model, nearest=phase == "tokenizer"),
               "legal_inactive_parameters": sorted(name for name in parameters if name.startswith("joint_sequence_adapter."))}
        self.records.append(row)
        write_json(self.records_path, self.records)
        with (self.work / "phase_integrity.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PHASE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        self.before = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.set_joint_phase = self.original_phase
        self.model.forward = self.original_forward
        self.model.forward_joint_tokenizer = self.original_tokenizer
        self.trainer._record_joint_metrics = self.original_record


def finite_metrics(results):
    normalized = {str(key): float(value) for key, value in results.items()}
    for k in (5, 10):
        recall, hit = f"recall@{k}", f"hit_rate@{k}"
        if hit not in normalized and recall in normalized:
            normalized[hit] = normalized[recall]
        if recall in normalized and hit in normalized:
            need(abs(normalized[hit] - normalized[recall]) < 1e-12, "HR/recall aliases disagree")
    need(all(math.isfinite(value) for value in normalized.values()), "Evaluation returned nonfinite metrics")
    for name in ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10"):
        need(name in normalized and 0 <= normalized[name] <= 1, f"Missing/out-of-range metric {name}")
    if "weighted_score" in normalized:
        need(abs(normalized["weighted_score"] - (.8 * normalized["ndcg@10"] + .2 * normalized["hit_rate@10"])) < 1e-7,
             "Weighted validation score disagrees with the retained protocol")
    return normalized


def joint_optimizer(model, config):
    import torch
    groups = [{"params": list(model.joint_recommender_parameters()), "lr": float(config["lr"]),
               "weight_decay": float(config["weight_decay"]), "group_name": "recommender"}]
    lrs = {"joint_codebook": config["joint_codebook_lr"], "joint_mixer": config["joint_mixer_lr"]}
    for name, parameters in model.joint_tokenizer_parameter_groups():
        if parameters:
            groups.append({"params": parameters, "lr": float(lrs[name]),
                           "weight_decay": float(config["joint_weight_decay"]), "group_name": name})
    return torch.optim.AdamW(groups)


def joint_scheduler(optimizer, warmup_steps, total_steps):
    import torch
    def factor(step):
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [factor if group["group_name"] == "recommender"
                                                       else lambda _step: 1.0 for group in optimizer.param_groups])


def smoke_recovery(pipeline, trainer, work, source):
    """Restore a private real tokenizer boundary, then perform one rec update."""
    import torch
    from torch.utils.data import DataLoader
    from genrec.models.DIFF_GRM.training_state import capture_rng, configuration_fingerprint, restore_boundary, capture_joint_runtime
    last, best = work / "training_state/last.pt", work / "training_state/best_model.bin"
    hashes = {"last": sha(last), "best": sha(best), "source": sha(source)}
    payload = torch.load(source, map_location="cpu", weights_only=False)
    need(payload["format"] == "diffgrm_joint_phase_boundary_v1"
         and payload["progress"]["joint_progress"]["last_committed_phase"] == "tokenizer",
         "Smoke recovery requires the actual committed tokenizer boundary")
    optimizer_integrity(payload, pipeline.accelerator.unwrap_model(trainer.model))
    directory = work / "recovery_probe"
    directory.mkdir(exist_ok=False)
    shutil.copyfile(source, directory / "source_last.pt")
    copied_best = directory / "best_model.bin"
    copied_best.write_bytes(payload["best_checkpoint_bytes"])
    need(sha(copied_best) == payload["best_checkpoint_sha256"], "Embedded recovery best checkpoint is damaged")
    payload["best_checkpoint"] = str(copied_best)
    config, accelerator = pipeline.config, pipeline.accelerator
    model = type(accelerator.unwrap_model(trainer.model))(config, pipeline.raw_dataset, pipeline.tokenizer)
    optimizer = joint_optimizer(model, config)
    loader = DataLoader(pipeline.tokenized_datasets["train"], batch_size=config["train_batch_size"],
                        shuffle=True, collate_fn=pipeline.tokenizer.collate_fn["train"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    scheduler = joint_scheduler(optimizer, trainer.effective_warmup_steps, trainer.total_optimizer_steps)
    expected = copy.deepcopy(payload)
    restore_boundary(payload, model=model, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator,
                     loader=loader, config_fingerprint=configuration_fingerprint(config))
    raw = accelerator.unwrap_model(model)
    require_same(raw.state_dict(), expected["model"], "model/committed SID buffers")
    require_same(optimizer.state_dict(), expected["optimizer"], "optimizer")
    require_same(scheduler.state_dict(), expected["scheduler"], "scheduler")
    require_same(capture_rng(loader), expected["rng"], "RNG/loader")
    require_same(capture_joint_runtime(raw, optimizer, expected["progress"]), expected["joint_runtime"], "joint runtime/anchor/cache/phase")
    before = {name: parameter.detach().cpu().clone() for name, parameter in raw.named_parameters()}
    before_codes = raw.joint_codes_sha256()
    raw.set_joint_phase("recommender")
    optimizer.zero_grad(set_to_none=True)
    batches, updates = 0, 0
    for batch in loader:
        trainer._check_training_deadline()
        raw.train()
        raw.joint_alignment.eval()
        with accelerator.accumulate(model):
            loss = model(batch, return_loss=True).recommendation_loss
            need(bool(torch.isfinite(loss)), "Recovered recommendation loss is nonfinite")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                norm = accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                need(bool(torch.isfinite(norm)), "Recovered gradients are nonfinite")
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)
        batches += 1
        if updates:
            break
    changed = {name for name, parameter in raw.named_parameters() if not torch.equal(before[name], parameter.detach().cpu())}
    need(updates == 1 and batches == config["gradient_accumulation_steps"], "Recovery did not perform one complete accumulated update")
    need(changed and not any(name.startswith("joint_alignment.") for name in changed), "Recovery changed tokenizer or no recommender parameters")
    if raw.item_local_attention_enabled:
        need(any(name.startswith("item_local_attention.") for name in changed), "Recovered update did not change local attention")
    need(raw.joint_codes_sha256() == before_codes, "Recommender recovery changed committed SID")
    need(scheduler.state_dict()["last_epoch"] == expected["scheduler"]["last_epoch"] + 1, "Recovery scheduler did not advance exactly once")
    after_progress = copy.deepcopy(expected["progress"])
    after_progress["cumulative_optimizer_updates"] += 1
    optimizer_after = optimizer_integrity({"optimizer": optimizer.state_dict(), "joint_runtime": expected["joint_runtime"],
                                          "progress": after_progress}, raw)
    need(hashes == {"last": sha(last), "best": sha(best), "source": sha(source)}, "Private recovery changed original checkpoints")
    result = {"status": "passed", "strict_model_optimizer_scheduler_RNG_joint_runtime_restore": True,
              "source_last_sha256": hashes["source"], "source_phase": "tokenizer",
              "optimizer_updates_executed": updates, "micro_batches": batches,
              "actual_changed_recommender_parameters": sorted(changed), "tokenizer_parameters_and_SID_unchanged": True,
              "optimizer_clock_check": optimizer_after,
              "original_checkpoint_hashes_unchanged": hashes, "manual_one_update_probe": True,
              "trainer_fit_resume_not_exercised": True, "cuda_bitwise_claim": False}
    write_json(directory / "recovery_check.json", result)
    # Restore the original model's derived catalogue table after the private model used the shared tokenizer object.
    accelerator.unwrap_model(trainer.model).sync_joint_tokenizer_mapping()
    return result


def snapshot_test_diagnostics(trainer, work):
    """Preserve once-only test CSV tails before any later recovery truncation."""
    directory = work / "post_test_diagnostics"
    directory.mkdir(exist_ok=False)
    records = []
    for attribute in ("position_metrics_csv", "order_metrics_csv", "training_metrics_csv",
                      "evaluation_metrics_csv", "joint_metrics_csv"):
        source = Path(getattr(trainer, attribute)).resolve()
        need(source.is_relative_to(work.resolve()), "Test diagnostic path escaped the owned run")
        destination = directory / source.name
        shutil.copyfile(source, destination)
        records.append({"attribute": attribute, "original_relative_path": source.relative_to(work.resolve()).as_posix(),
                        "snapshot_relative_path": destination.relative_to(work.resolve()).as_posix(),
                        "sha256": sha(source), "size_bytes": source.stat().st_size})
    return records


def restore_test_diagnostics(receipt, trainer, work):
    validate_test_snapshots(receipt, work)
    records = receipt.get("diagnostic_snapshots")
    need(isinstance(records, list) and len(records) == 5, "Once-only test receipt lacks all five diagnostic snapshots")
    attributes = {"position_metrics_csv", "order_metrics_csv", "training_metrics_csv", "evaluation_metrics_csv", "joint_metrics_csv"}
    need({row["attribute"] for row in records} == attributes, "Test diagnostic snapshot attributes differ")
    for row in records:
        source = stage05.under(work, row["snapshot_relative_path"])
        destination = stage05.under(work, row["original_relative_path"])
        need(destination.resolve() == Path(getattr(trainer, row["attribute"])).resolve(), "Test diagnostic destination changed")
        need(source.stat().st_size == row["size_bytes"] and sha(source) == row["sha256"], "Preserved test diagnostic bytes are damaged")
        shutil.copyfile(source, destination)
        need(sha(destination) == row["sha256"], "Failed restoring the once-only test diagnostic tail")


def validate_test_snapshots(receipt, work):
    start_path = work / "TEST_STARTED.json"
    need(start_path.is_file() and receipt.get("started_receipt_sha256") == sha(start_path),
         "Once-only test result is not bound to its start marker")
    records = receipt.get("diagnostic_snapshots")
    expected = {"position_metrics_csv", "order_metrics_csv", "training_metrics_csv", "evaluation_metrics_csv", "joint_metrics_csv"}
    need(isinstance(records, list) and len(records) == 5 and {row["attribute"] for row in records} == expected,
         "Once-only test receipt lacks its complete diagnostic snapshots")
    for row in records:
        source = stage05.under(work, row["snapshot_relative_path"])
        need(source.is_file() and source.stat().st_size == row["size_bytes"] and sha(source) == row["sha256"],
             "Once-only test diagnostic snapshot is missing or damaged")


def run(args, prepared):
    import torch
    from torch.utils.data import DataLoader
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "GPU modes require exactly one visible GPU")
    need("H100" in torch.cuda.get_device_name(0).upper() and torch.cuda.is_bf16_supported(), "One BF16-capable H100 is required")
    work, attempt, config, engine = (prepared[key] for key in ("work", "attempt", "config", "engine"))
    reserve = 0 if (work / "FIT_COMPLETE.json").exists() else args.training_deadline_reserve_seconds
    need(args.deadline_unix is not None and time.time() < args.deadline_unix - reserve,
         "GPU execution requires a future bounded deadline with training reserve")
    need(args.deadline_unix - time.time() <= 28 * 3600 - prepared["study"]["prior_allocated_gpu_seconds"],
         "Requested process deadline exceeds the cumulative campaign allocation remaining")
    if (work / "TEST_RESULT.json").exists():
        # Validate the immutable copy before Trainer can roll uncommitted CSV tails back.
        validate_test_snapshots(read_json(work / "TEST_RESULT.json"), work)
    runtime = work / "runtime"
    runtime.mkdir(exist_ok=True)
    os.chdir(runtime)
    os.environ.update(HF_HOME=str(work / "workspace/huggingface"), HF_DATASETS_CACHE=str(work / "workspace/huggingface/datasets"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      TORCHDYNAMO_DISABLE="1", WANDB_MODE="disabled", PYTHONNOUSERSITE="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(engine))
    import genrec.utils as utils
    utils.get_local_time = lambda: config["run_id"]
    from genrec.pipeline import Pipeline
    trainer_module = importlib.import_module("genrec.models.DIFF_GRM.trainer")
    from genrec.models.DIFF_GRM.training_state import capture_rng
    pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", config_dict=config)
    trainer, model = pipeline.trainer, pipeline.model
    need(pipeline.accelerator.num_processes == 1 and pipeline.config["joint_crosspse_enabled"], "Joint single-process pipeline was not created")
    need(not model.joint_alignment.mixer.mixer_enabled and model.joint_alignment.gradient_mode == "diger_st", "Tokenizer/global mixer settings differ")
    need(model.item_local_attention_enabled == (args.variant == "item_local"), "Wrong local-attention variant")
    need(Path(pipeline.tokenizer.semantic_ids_path).resolve() == Path(config["precomputed_semantic_ids_path"]).resolve(), "Consumed SID artifact differs")
    verify_copies(prepared)
    loaders = {split: DataLoader(pipeline.tokenized_datasets[split],
                                batch_size=config["train_batch_size" if split == "train" else "eval_batch_size"],
                                shuffle=split == "train", collate_fn=pipeline.tokenizer.collate_fn[split]) for split in ("train", "val")}
    actual = {key: str(value) if key == "device" else value for key, value in pipeline.config.items() if key != "accelerator"}
    write_json(attempt / "parsed_config.json", actual)
    first_json(work / "parsed_config.json", actual)
    environment = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "visible_gpus": 1,
                   "packages": sorted(f"{entry.metadata['Name']}=={entry.version}" for entry in importlib.metadata.distributions())}
    write_json(attempt / "environment.json", environment)
    first_json(work / "environment.json", environment)
    model.to(pipeline.accelerator.device)
    initial = {"campaign_id": CAMPAIGN, "variant": args.variant, "resume_requested": bool(args.resume),
               "initial_base_state_sha256": json_sha(nested_summary({name: value for name, value in model.state_dict().items()
                                                                    if not name.startswith("item_local_attention.")})),
               "initial_rng_sha256": json_sha(nested_summary(capture_rng(loaders["train"]))),
               "initial_SID_sha256": model.joint_codes_sha256(),
               "scope": "fresh shared warm-start state before fit; excludes only optional recommender-side local parameters"}
    write_json(attempt / "initial_state_identity.json", initial)
    first_json(work / "initial_state_identity.json", initial)
    if args.resume:
        saved_initial = read_json(work / "initial_state_identity.json")
        need(all(initial[key] == saved_initial[key] for key in ("initial_base_state_sha256", "initial_rng_sha256", "initial_SID_sha256")),
             "Reconstructed shared warm start changed before resuming own last")
    started = time.perf_counter()
    started_receipt = {"mode": args.mode, "utc": utc_now(), "pid": os.getpid(), "resume": bool(args.resume),
                       "deadline_unix": args.deadline_unix, "campaign_id": CAMPAIGN, "variant": args.variant}
    write_json(attempt / "STARTED.json", started_receipt)
    first_json(work / "STARTED.json", started_receipt)
    timings_path = work / "evaluation_timings.json"
    timings = read_json(timings_path) if timings_path.exists() else []
    original_evaluate = trainer.evaluate
    def evaluate_timed(*positional, **keywords):
        need(time.time() < args.deadline_unix, "Allocation deadline reached before evaluation")
        torch.cuda.synchronize()
        beginning = time.perf_counter()
        result = original_evaluate(*positional, **keywords)
        torch.cuda.synchronize()
        timings.append({"split": keywords.get("split", "test"), "epoch": keywords.get("epoch"),
                        "seconds": time.perf_counter() - beginning, "attempt": attempt.name})
        write_json(timings_path, timings)
        return result
    trainer.evaluate = evaluate_timed
    resume_epoch = int(trainer._resume_payload["progress"]["next_epoch"]) if trainer._resume_payload else 0
    audit = PhaseAudit(model, trainer, work, resume_epoch)
    original_save = trainer_module.save_boundary
    recovery_source = work / "smoke_tokenizer_boundary.pt"
    def save_observed(path, **keywords):
        payload = original_save(path, **keywords)
        if args.mode == "smoke" and payload["progress"]["joint_progress"]["last_committed_phase"] == "tokenizer":
            need(not recovery_source.exists(), "Smoke tokenizer recovery source would be duplicated")
            shutil.copyfile(path, recovery_source)
        return payload
    trainer_module.save_boundary = save_observed
    try:
        best_epoch, best_score = trainer.fit(loaders["train"], loaders["val"])
        need(not trainer.training_incomplete, "Training ended incomplete; final testing is forbidden")
        need(best_epoch is not None and math.isfinite(float(best_score)), "No validation-selected checkpoint")
        terminal_path = work / "training_state/last.pt"
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        need(terminal["format"] == "diffgrm_joint_phase_boundary_v1" and terminal["progress"]["training_complete"],
             "No complete joint training boundary was persisted")
        joint_progress = terminal["progress"]["joint_progress"]
        need(joint_progress["tokenizer_optimizer_updates"] > 0 and joint_progress["sid_refresh_count"] > 0,
             "Completed joint run did not actually update tokenizer and refresh SID")
        need(int(terminal["model"]["joint_alignment.diger_optimizer_step"]) == joint_progress["tokenizer_optimizer_updates"],
             "Committed tokenizer step and DIGER schedule disagree")
        training_integrity = {"status": "passed", "phase_records_sha256": sha(work / "phase_integrity.json"),
                              "optimizer": optimizer_integrity(terminal, pipeline.accelerator.unwrap_model(trainer.model)),
                              "global_codebook_mixer_enabled": False,
                              "local_block_parameter_side": "recommender"}
        write_json(work / "training_integrity.json", training_integrity)
        fit = {"best_epoch": int(best_epoch), "best_val_score": float(best_score), "completed_epochs": trainer.completed_epochs,
               "training_stop_reason": trainer.training_stop_reason, "joint_progress": joint_progress,
               "cumulative_recommender_optimizer_updates": terminal["progress"]["cumulative_optimizer_updates"],
               "training_state_sha256": sha(terminal_path)}
        if (work / "FIT_COMPLETE.json").exists():
            previous_fit = read_json(work / "FIT_COMPLETE.json")
            need({key: value for key, value in previous_fit.items() if key != "training_state_sha256"}
                 == {key: value for key, value in fit.items() if key != "training_state_sha256"},
                 "Existing completed fit changed beyond a terminal-state serialization")
        else:
            write_json(work / "FIT_COMPLETE.json", fit)
        write_json(attempt / "FIT_COMPLETE.json", fit)
        write_json(work / "epoch_timings.json", trainer.epoch_timing_records)
        checkpoint = Path(trainer.saved_model_ckpt)
        need(sha(checkpoint) == terminal["best_checkpoint_sha256"], "Selected best differs from committed embedded checkpoint")
        raw = pipeline.accelerator.unwrap_model(trainer.model)
        raw.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        raw.sync_joint_tokenizer_mapping()
        selected_sid = raw.joint_codes_sha256()
        sid_integrity(raw)
        write_json(work / "checkpoint_reload.json", {"strict_all_tensor_match": True, "checkpoint_sha256": sha(checkpoint),
                                                     "best_epoch": int(best_epoch), "matching_committed_SID_sha256": selected_sid,
                                                     "mapping_synchronized_without_requantization": True})
        # Training is terminal. The reserved time is now available for recovery
        # acceptance / final evaluation; all remaining work keeps the hard deadline.
        pipeline.config["training_deadline_reserve_seconds"] = 0.0
        trainer.config["training_deadline_reserve_seconds"] = 0.0
        write_json(attempt / "FINAL_EVALUATION_READY.json", {"utc": utc_now(), "training_terminal": True,
                                                            "deadline_unix": args.deadline_unix,
                                                            "training_reserve_released_seconds": args.training_deadline_reserve_seconds})
        common = {"campaign_id": CAMPAIGN, "protocol_id": CAMPAIGN, "variant": args.variant,
                  "run_id": config["run_id"], "evaluation_level": "sid", "checkpoint_sha256": sha(checkpoint),
                  "selected_SID_sha256": selected_sid, "initial_base_state_sha256": initial["initial_base_state_sha256"],
                  "initial_rng_sha256": initial["initial_rng_sha256"], "fit": fit,
                  "training_integrity": training_integrity,
                  "primary_comparison": "Current joint control versus current joint item_local; campaign layer compares the two",
                  "historical_comparison_role": "Stage03/05 results are background only",
                  "limitations": "Single seed; SID-level evaluation; no significance claim; alternating updates, not simultaneous optimizer updates"}
        if args.mode == "smoke":
            rows = audit.records
            need([row["phase"] for row in rows] == ["recommender", "tokenizer", "recommender"]
                 and all(row["optimizer_updates"] == 2 for row in rows), "Smoke did not execute rec/tok/rec with two updates each")
            need(joint_progress["tokenizer_optimizer_updates"] == 2 and joint_progress["sid_refresh_count"] == 1,
                 "Smoke tokenizer step/refresh counts differ")
            recovery = smoke_recovery(pipeline, trainer, work, recovery_source)
            need(all(row["split"] != "test" for row in timings) and not (work / "TEST_STARTED.json").exists(),
                 "Smoke must never evaluate test targets")
            result_name = "smoke_results.json"
            result = {**common, "status": "smoke_passed_not_formal", "phase_integrity": rows,
                      "recovery_check": recovery, "test_data_evaluated": False,
                      "validation_max_batches_per_call": 8, "process_wall_seconds": time.perf_counter() - started}
        else:
            result_name = "results.json"
            test_path = work / "TEST_RESULT.json"
            if test_path.exists():
                receipt = read_json(test_path)
                start_path = work / "TEST_STARTED.json"
                need(start_path.is_file() and receipt.get("started_receipt_sha256") == sha(start_path),
                     "Existing test result is not bound to its actual test-start marker")
                need(receipt["checkpoint_sha256"] == sha(checkpoint) and receipt["selected_SID_sha256"] == selected_sid,
                     "Existing test result belongs to different selected weights/SID")
                restore_test_diagnostics(receipt, trainer, work)
                test_results = finite_metrics(receipt["test_results"])
            else:
                need(not (work / "TEST_STARTED.json").exists(), "Final test was already started; refusing duplicate evaluation")
                need(time.time() < args.deadline_unix, "Allocation deadline reached before final test")
                with (work / "TEST_STARTED.json").open("x", encoding="utf-8") as handle:
                    json.dump({"utc": utc_now(), "checkpoint_sha256": sha(checkpoint), "selected_SID_sha256": selected_sid}, handle)
                test_loader = DataLoader(pipeline.tokenized_datasets["test"], batch_size=config["eval_batch_size"],
                                         shuffle=False, collate_fn=pipeline.tokenizer.collate_fn["test"])
                test_loader = pipeline.accelerator.prepare(test_loader)
                raw.generate_w_decoding_graph = True
                test_results = finite_metrics(trainer.evaluate(test_loader, split="test"))
                diagnostic_snapshots = snapshot_test_diagnostics(trainer, work)
                write_json(test_path, {"test_results": test_results, "checkpoint_sha256": sha(checkpoint),
                                       "selected_SID_sha256": selected_sid,
                                       "diagnostic_snapshots": diagnostic_snapshots,
                                       "started_receipt_sha256": sha(work / "TEST_STARTED.json"), "completed_utc": utc_now()})
            trainer._write_run_metadata(test_results)
            result = {**common, "status": "formal_completed", "test_results": test_results,
                      "test_data_evaluated": True, "phase_integrity_sha256": sha(work / "phase_integrity.json"),
                      "process_wall_seconds": time.perf_counter() - started}
        verify_copies(prepared)
        write_json(work / result_name, result)
        write_json(work / "COMPLETE.json", {"mode": args.mode, "status": result["status"], "campaign_id": CAMPAIGN,
                                            "variant": args.variant, "run_id": config["run_id"], "completed_utc": utc_now(),
                                            "result_file": result_name, "result_sha256": sha(work / result_name),
                                            "checkpoint_sha256": sha(checkpoint)})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        write_json(attempt / "INCOMPLETE.json", {"utc": utc_now(), "error_type": type(error).__name__, "error": str(error),
                                                "resume_state": str(work / "training_state/last.pt"),
                                                "training_incomplete": bool(trainer.training_incomplete),
                                                "training_stop_reason": trainer.training_stop_reason})
        raise
    finally:
        trainer_module.save_boundary = original_save
        audit.close()
        trainer.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "formal"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", required=True, help="Read-only historical assets")
    parser.add_argument("--engine-root", required=True, help="Independent fixed DiffGRM engine clone")
    parser.add_argument("--expected-engine-commit", help="Full frozen engine SHA; required in both GPU modes")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Explicit own last.pt recovery, formal only")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--training-deadline-reserve-seconds", type=float, default=1200.0)
    args = parser.parse_args()
    if not math.isfinite(args.training_deadline_reserve_seconds) or args.training_deadline_reserve_seconds < 0:
        parser.error("Training deadline reserve must be finite and nonnegative")
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        parser.error("Deadline must be finite")
    prepared = prepare(args)
    if prepared.get("already_complete"):
        print(json.dumps(prepared["already_complete"], indent=2))
    elif args.mode == "prepare":
        print(json.dumps({"status": "prepared_cpu_only", "work": str(prepared["work"]), "cuda_initialized": False}, indent=2))
    else:
        with execution_guard(prepared["work"]):
            run(args, prepared)


if __name__ == "__main__":
    main()
