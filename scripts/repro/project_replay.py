"""Opt-in isolated decoder replays for stages 01-04; never refit a tokenizer.

The controller uses immutable historical SID files and a pinned source archive.
It does not claim that current replay sources are the historical runtime commit,
or that passing a smoke check reproduces historical recommendation metrics.
The pinned AR trainer has no batch limits, so AR two-batch smoke is refused.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import uuid

import project_repro as p


def snapshot_text(snapshot, relative_path):
    """Read one regular source file, never extract a user-controlled tar path."""
    name = p.relative(relative_path)
    with tarfile.open(snapshot, "r:") as archive:
        member = archive.getmember(name)
        if not member.isfile():
            raise ValueError(f"Expected a regular source file: {name}")
        with archive.extractfile(member) as source:
            return source.read().decode("utf-8-sig")


def convert_scalars(config):
    """Match the pinned get_config top-level conversion, without model imports."""
    result = dict(config)
    for key, value in result.items():
        if not isinstance(value, str):
            continue
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
            continue
        try:
            converted = json.loads(value)
        except (ValueError, TypeError):
            try:
                converted = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
        if converted is None or isinstance(converted, (str, int, float, bool, list, dict, tuple)):
            result[key] = converted
    return result


def smoke_support(model, trainer_source):
    if model == "AR_GRM":
        return {
            "supported": False,
            "reason": "Pinned AR trainer has no train/eval batch limits. A full epoch is not a two-batch smoke; no model code is patched by this organizer.",
        }
    tokens = ("max_train_batches_per_epoch", "max_eval_batches", "batch_index >= epoch_steps", "batch_index >= eval_limit")
    supported = all(token in trainer_source for token in tokens)
    return {"supported": supported, "reason": "Pinned DiffGRM trainer exposes checked batch limits" if supported else "Pinned trainer batch-limit implementation not recognized; refuse an unbounded smoke"}


def load_formal_config(recipe, pack, snapshot, yaml):
    model = recipe["model"]
    config = {"model": model, "dataset": "AmazonReviews2014"}
    sources = []
    for name in ("genrec/default.yaml", "genrec/datasets/AmazonReviews2014/config.yaml", f"genrec/models/{model}/config.yaml"):
        values = yaml.safe_load(snapshot_text(snapshot, name)) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Configuration is not a mapping: {name}")
        config.update(values)  # Shallow, ordered merge, exactly like get_config.
        sources.append({"source": "pinned_source_archive", "path": name})
    for name in recipe["config_layers"]:
        path = p.under(pack, name)
        values = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Configuration is not a mapping: {name}")
        config.update(values)
        sources.append({"source": "evidence", "path": name, "sha256": p.sha(path)})
    config = convert_scalars(config)
    seed = int(recipe["seed"])
    config["rand_seed"] = seed
    if model == "AR_GRM":
        config["sid_fit_seed"] = seed
        if config.get("sid_quantizer") == "rq_kmeans":
            config["rq_kmeans_seed"] = seed
        elif config.get("sid_quantizer") == "rqvae":
            config["rqvae_seed"] = seed
    # Diff tokenizers are frozen from their original fit, not resampled for a
    # decoder seed. This catalogue currently has only Diff seed 2024.
    if config.get("joint_crosspse_enabled", False):
        raise ValueError("Stages 01-04 are offline-tokenizer experiments; joint training is not allowed")
    for key in ("joint_recommender_checkpoint", "checkpoint", "checkpoint_path"):
        if config.get(key):
            raise ValueError(f"Random decoder initialization conflicts with {key}")
    for key in ("force_regenerate_opq", "force_regenerate_codes", "force_regenerate_pca"):
        if config.get(key, False):
            raise ValueError(f"Historical SID replay must not enable {key}")
        config[key] = False
    for key in ("max_train_batches_per_epoch", "max_eval_batches"):
        if config.get(key) not in (None, 0):
            raise ValueError(f"Formal configuration unexpectedly truncates {key}")
    config["joint_crosspse_enabled"] = False
    config.setdefault("early_stop_after_warmup", False)
    config.setdefault("diffusion_loss_reduction", "masked_token")
    config.setdefault("numerical_guard", False)
    return config, sources


def audit_formal_config(config, recipe, metadata):
    """Only check historical settings actually recorded; do not fill gaps."""
    if metadata.get("run_id") != recipe["run_id"] or int(metadata.get("seed", -1)) != int(recipe["seed"]):
        raise ValueError("Historical metadata identity differs from the selected recipe")
    expected = {}
    keys = (
        "sid_quantizer", "codebook_size", "masking_strategy", "guided_select",
        "guided_refresh_each_step", "diffusion_loss_reduction", "train_batch_size",
        "gradient_accumulation_steps", "warmup_steps", "lr", "mixed_precision",
        "max_grad_norm", "numerical_guard", "numerical_guard_parameter_interval",
        "eval_start_epoch", "eval_interval", "patience", "early_stop_after_warmup",
        "evaluation_level", "crosspse_mixer_enabled", "crosspse_train_base_codebooks",
    )
    for key in keys:
        if metadata.get(key) is not None:
            expected[key] = metadata[key]
    if recipe["model"] == "DIFF_GRM":
        expected["n_digit"] = int(metadata["n_digit"])
        if config.get("evaluation_level", "sid") != recipe["evaluation_level"]:
            raise ValueError("DiffGRM evaluation level differs from recipe")
    else:
        if recipe["evaluation_level"] != "item" or not config.get("strict_item_eval", False):
            raise ValueError("Historical AR replay requires strict item-level evaluation")
        semantic = int(metadata["semantic_n_digit"])
        if int(config.get("semantic_n_digit", config["n_digit"])) != semantic:
            raise ValueError("Semantic digit count differs from historical AR metadata")
        total = semantic + int(bool(config.get("use_collision_suffix", False)))
        if total != int(metadata["total_n_digit"]):
            raise ValueError("Collision suffix / total AR digit count differs from history")
        if "semantic_codebook_sizes" in metadata:
            expected["semantic_codebook_sizes"] = metadata["semantic_codebook_sizes"]
    differences = {key: {"recorded": value, "resolved": config.get(key)} for key, value in expected.items() if config.get(key) != value}
    if differences:
        raise ValueError("Resolved formal configuration differs from recorded fields: " + json.dumps(differences))
    if config.get("run_id") != recipe["run_id"]:
        raise ValueError("Archive config layers resolve to a different historical run_id")
    if config.get("val_metric") != recipe["selection_metric"]:
        raise ValueError("Selection metric differs from the locked recipe")
    if list(config.get("semantic_codebook_sizes", [config["codebook_size"]] * config["n_digit"])) != recipe["semantic_codebook_sizes"]:
        raise ValueError("Semantic codebook layout differs from recipe")
    return {"status": "passed", "checked_metadata_fields": expected,
            "selection_metric": config["val_metric"],
            "limitation": "Historical runtime commit and complete resolved config were not recorded; checks cover available fields only"}


def dataset_audit(workspace, config, recipe):
    """Recompute catalogue/split counts from the exact copied catalogue."""
    processed = p.under(workspace, p.DATA + "processed")
    mappings = p.read_json(processed / "id_mapping.json")
    sequences = p.read_json(processed / "all_item_seqs.json")
    sid = p.read_json(p.under(workspace, recipe["initial_sid"]))
    item_keys = mappings["id2item"][1:]  # Source dataset reserves index 0.
    if set(sid) != set(item_keys) or len(sid) != len(item_keys):
        raise ValueError("Locked SID catalogue does not match locked item mapping")
    sizes = recipe["semantic_codebook_sizes"]
    for item, digits in sid.items():
        if len(digits) != len(sizes) or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < size for x, size in zip(digits, sizes)):
            raise ValueError(f"Invalid SID layout or out-of-range digit for item {item}")
    if config.get("split") not in ("leave_one_out", "last_out") or not config.get("train_sliding"):
        raise ValueError("Replay currently supports the recorded leave-one-out sliding protocol only")
    min_hist, max_hist = int(config["min_hist_len"]), int(config["max_hist_len"])
    lengths = [len(values) for values in sequences.values()]
    counts = {
        "real_users": len(sequences), "real_items": len(item_keys),
        "interactions": sum(lengths), "train_examples": sum(max(0, min(length - 3, max_hist) - min_hist + 1) for length in lengths),
        "val_examples": sum(length > 1 for length in lengths), "test_examples": len(lengths),
        "logged_users_with_padding": len(mappings["id2user"]), "logged_items_with_padding": len(mappings["id2item"]),
    }
    expected = dict(real_users=22363, real_items=12101, interactions=198502, train_examples=105668, val_examples=22363, test_examples=22363,
                    logged_users_with_padding=22364, logged_items_with_padding=12102)
    if counts != expected:
        raise ValueError("Dataset counts do not match the recorded Beauty protocol: " + json.dumps(counts))
    return {"status": "passed", "counts": counts,
            "basis": "SHA256-bound catalogue; split counts recomputed using pinned sliding-window bounds, not inferred from training metrics"}


def configure_paths(config, workspace, work, recipe, run_id):
    values = dict(config)
    values.update(run_id=run_id, cache_dir=str(workspace / "data/cache"), joint_crosspse_enabled=False)
    for key, directory in (("log_dir", "logs"), ("tensorboard_log_dir", "tensorboard"), ("ckpt_dir", "checkpoints"),
                           ("run_metadata_dir", "run_metadata"), ("position_loss_log_dir", "positions"),
                           ("diffusion_diagnostics_log_dir", "diagnostics")):
        values[key] = str(work / directory)
    if recipe["model"] == "AR_GRM":
        values["sent_emb_path"] = str(p.under(workspace, "data/formal_embeddings/beauty_sentence_t5_xl.npy"))
    elif values.get("sent_emb_path"):
        values["sent_emb_path"] = str(p.under(workspace, p.safe.project_path(values["sent_emb_path"])))
    if recipe.get("artifact_dir"):
        artifact = p.under(workspace, recipe["artifact_dir"])
        if values.get("sid_quantizer") == "rq_kmeans":
            values["tokenizer_artifact_dir"] = str(artifact)
            if not (artifact / "reconstructed_embeddings.npy").is_file():
                raise ValueError("RQ-KMeans reconstruction artifact missing: refusing implicit SID regeneration")
        if values.get("sid_quantizer") in ("rqvae", "cross_pse"):
            values["precomputed_semantic_ids_path"] = str(p.under(workspace, recipe["initial_sid"]))
        for key in ("rqvae_summary_path", "crosspse_summary_path"):
            if values.get(key):
                values[key] = str(artifact / "summary.json")
                if not (artifact / "summary.json").is_file():
                    raise ValueError(f"Required {key} was not copied into the isolated workspace")
    elif values.get("precomputed_semantic_ids_path") or values.get("tokenizer_artifact_dir"):
        raise ValueError("Unbound tokenizer path in recipe")
    return values


def apply_smoke(config):
    values = dict(config)
    values.update(epochs=1, warmup_steps=0, warmup_ratio=0.0, gradient_accumulation_steps=1,
                  train_batch_size=4, eval_batch_size=2, eval_start_epoch=1, eval_interval=1, patience=1,
                  early_stop_after_warmup=False, max_train_batches_per_epoch=2, max_eval_batches=2)
    beam = dict(values["vectorized_beam_search"])
    beam.update(beam_act=16, beam_max=16, val={"beam_act": 16, "beam_max": 16}, test={"beam_act": 16, "beam_max": 16})
    values["vectorized_beam_search"] = beam
    return values


def expected_runtime_sid(config, recipe):
    """Cache naming used by the pinned AR and DiffGRM tokenizer implementations."""
    kind = config["sid_quantizer"]
    if kind in ("rqvae", "cross_pse"):
        return recipe["initial_sid"]
    sizes = recipe["semantic_codebook_sizes"]
    depth = len(sizes)
    model_name = str(config["sent_emb_model"]).split("/")[-1]
    if kind == "opq_pq":
        if len(set(sizes)) != 1 or sizes[0] & (sizes[0] - 1):
            raise ValueError("PSE requires equal power-of-two codebooks")
        bits = sizes[0].bit_length() - 1
        tag = (f"OPQ{depth}," if not config.get("disable_opq", False) else "") + f"IVF1,PQ{depth}x{bits}"
    elif kind == "rq_kmeans":
        if recipe["model"] == "AR_GRM":
            tag = "RQKMEANS_" + "x".join(map(str, sizes))
            suffix = "inputpca" + str(config["sent_emb_pca"]) if config.get("rq_kmeans_use_pca") else "inputraw" + str(config["sent_emb_dim"])
        else:
            if len(set(sizes)) != 1 or sizes[0] & (sizes[0] - 1):
                raise ValueError("Pinned DiffGRM requires uniform power-of-two codebooks")
            tag = f"RQKMEANS{depth}x{sizes[0].bit_length() - 1}"
            suffix = "PCA" if config.get("rq_kmeans_use_pca") else "RAW"
        tag += f"_seed{config.get('rq_kmeans_seed', 1234)}_it{config.get('rq_kmeans_niters', 20)}_{suffix}"
    else:
        raise ValueError(f"Unsupported historical tokenizer: {kind}")
    return p.DATA + f"processed/{model_name}_pca{config['sent_emb_pca']}_{tag}.sem_ids"


def bind_runtime_sid(workspace, config, recipe, records, tokenizer_source):
    # The later capacity-ablation implementation renamed even uniform AR RQK
    # caches. Preserve both historical and runtime filenames, and bind by bytes.
    if recipe["model"] == "AR_GRM" and config["sid_quantizer"] == "rq_kmeans":
        if "self.index_factory = f'RQKMEANS_{layout}'" not in tokenizer_source:
            raise ValueError("AR cache naming implementation changed; review SID alias logic before execution")
    runtime = expected_runtime_sid(config, recipe)
    initial = recipe["initial_sid"]
    if runtime == initial:
        return runtime, None
    original = p.under(workspace, initial)
    alias = p.under(workspace, runtime)
    if alias.exists():
        raise ValueError(f"Unexpected existing runtime SID alias: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    with original.open("rb") as inp, alias.open("xb") as out:
        shutil.copyfileobj(inp, out)
    digest = p.sha(original)
    if p.sha(alias) != digest:
        raise ValueError("Runtime SID alias copy changed bytes")
    record = {"path": runtime, "sha256": digest, "byte_identical_alias_of": initial,
              "reason": "Pinned replay tokenizer cache naming differs from historical archive naming"}
    records.append(record)
    return runtime, record


def verify_copies(workspace, records):
    mismatches = [record["path"] for record in records if not p.under(workspace, record["path"]).is_file()
                  or p.sha(p.under(workspace, record["path"])) != record["sha256"]]
    if mismatches:
        raise ValueError("Bound inputs were modified inside replay: " + ", ".join(mismatches))


def post_run_audit(work, workspace, recipe, config, records, counts, mode, runtime_sid):
    verify_copies(workspace, records)
    files = list((work / "run_metadata").rglob("*_run_metadata.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one metadata file from the isolated replay")
    metadata = p.read_json(files[0])
    if metadata.get("run_id") != config["run_id"] or int(metadata.get("seed", -1)) != recipe["seed"]:
        raise ValueError("Replay metadata run identity mismatch")
    bound_sid = p.under(workspace, runtime_sid)
    if Path(metadata.get("semantic_ids_path", "")).resolve() != bound_sid.resolve():
        raise ValueError("Trainer consumed a different SID path than the bound input")
    if metadata.get("semantic_ids_sha256") and metadata["semantic_ids_sha256"] != p.sha(bound_sid):
        raise ValueError("Trainer SID hash differs from the locked historical SID")
    position_files = list((work / ("positions" if recipe["model"] == "AR_GRM" else "diagnostics")).rglob("*_position*.csv"))
    if len(position_files) != 1:
        raise ValueError("Expected one per-position diagnostic CSV")
    positions = p.rows(position_files[0])
    observed_splits = {row.get("split") for row in positions}
    if not {"train", "val", "test"}.issubset(observed_splits):
        raise ValueError("Missing train/val/test position diagnostics")
    for row in positions:
        if "nll" not in row or not math.isfinite(float(row["nll"])):
            raise ValueError("Missing or non-finite per-position NLL")
        for key in ("loss", "normalized_nll", "accuracy"):
            if row.get(key) not in (None, "") and not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite position metric: {key}")
    results = metadata.get("test_results") or {}
    for key in ("hit_rate@10", "ndcg@10"):
        if key not in results or not math.isfinite(float(results[key])):
            raise ValueError(f"Missing or non-finite test metric: {key}")
    console = (work / "console.log").read_text(encoding="utf-8")
    for label, key in (("users", "logged_users_with_padding"), ("items", "logged_items_with_padding"), ("interactions", "interactions")):
        matches = re.findall(r"Number of " + label + r":\s*(\d+)", console)
        if not matches or any(int(value) != counts[key] for value in matches):
            raise ValueError(f"Runtime dataset count was not confirmed for {label}")
    training_check = None
    if recipe["model"] == "DIFF_GRM":
        histories = list((work / "diagnostics").rglob("*_training_metrics.csv"))
        if len(histories) != 1:
            raise ValueError("Expected one DiffGRM training diagnostic CSV")
        history = p.rows(histories[0])
        if not history or any(not math.isfinite(float(row["train_loss"])) for row in history):
            raise ValueError("Missing or non-finite DiffGRM training loss")
        if mode == "smoke" and (len(history) != 1 or int(history[0]["optimizer_updates"]) != 2):
            raise ValueError("Smoke did not perform exactly one epoch / two optimizer updates")
        training_check = {"epochs": len(history), "finite_training_loss": True}
    else:
        losses = re.findall(r"Train Loss:\s*([^;\s]+)", console)
        if not losses or any(not math.isfinite(float(value)) for value in losses):
            raise ValueError("Missing or non-finite AR training loss")
        training_check = {"epochs_logged": len(losses), "finite_training_loss": True}
    return {"status": "passed", "unchanged_bound_inputs": len(records), "consumed_sid": runtime_sid,
            "historical_sid": recipe["initial_sid"],
            "dataset_counts_confirmed_in_log": counts, "position_rows_checked": len(positions), "training": training_check,
            "test_results": results, "mode": mode,
            "scope": "Input identity, dataset counts, and finite diagnostic outputs only; historical metric equality was not asserted"}


def run_replay(args):
    stage, mode = str(args.stage), str(args.mode)
    if stage not in p.STAGES or mode not in ("smoke", "formal"):
        raise ValueError("Replay supports only stages 01-04 and smoke/formal modes")
    p.verify(stage)
    recipe = p.load_recipe(stage, args.experiment)
    if recipe.get("decoder_initialization", {}).get("type") != "random_initialization_with_seed":
        raise ValueError("Recipe is not a random decoder initialization experiment")
    pack = p.package(stage)
    versions = p.read_json(pack / "reproducibility/code_versions.json")["repositories"]
    selected = [record for record in versions if record["name"] == "DiffGRM"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one pinned DiffGRM source snapshot")
    code = selected[0]
    snapshot = p.under(p.ROOT, code["snapshot"])
    trainer_source = snapshot_text(snapshot, f"genrec/models/{recipe['model']}/trainer.py")
    support = smoke_support(recipe["model"], trainer_source)
    print(json.dumps({"stage": stage, "experiment": recipe["id"], "mode": mode,
                      "decoder_initialization": recipe["decoder_initialization"], "fixed_sid": recipe["initial_sid"],
                      "config_layers": recipe["config_layers"], "code_commit_for_replay": code["commit"],
                      "historical_runtime_commit": None, "smoke": support,
                      "uses_latest_checkpoint_lookup": False, "tokenizer_retraining": False}, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Plan only; no training, dependency imports, input restoration or filesystem writes performed.")
        return
    if mode == "smoke" and not support["supported"]:
        raise ValueError(support["reason"])
    if os.name == "nt":
        raise ValueError("Execute replays in the Linux GPU environment; Windows supports plan/check/analyze only")
    p.check_inputs(stage, recipe, args.project_root)
    try:
        import yaml
        import torch
    except ImportError as error:
        raise ValueError("Activate grm-exp with PyYAML and CUDA-enabled PyTorch before --execute") from error
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no output directory or training was started")
    config, sources = load_formal_config(recipe, pack, snapshot, yaml)
    if config.get("mixed_precision") == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This replay requires BF16 support")
    config_audit = audit_formal_config(config, recipe, p.read_json(p.under(pack, recipe["metadata"])))
    root = Path(args.project_root).resolve()
    if not root.is_dir():
        raise ValueError("Input project root does not exist")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"stage{stage}_replay_{recipe['id']}_{mode}_{timestamp}_{uuid.uuid4().hex[:8]}"
    work = p.under(root, "outputs/project_replays/" + run_id)
    work.mkdir(parents=True, exist_ok=False)
    workspace = work / "workspace"
    engine = workspace / "external/DiffGRM"
    p.safe.unpack_code(snapshot, engine)
    lock = p.read_json(pack / "reproducibility/assets.lock.json")["artifacts"]
    copied = []
    for name in recipe["required_artifacts"]:
        source, destination = p.under(root, name), p.under(workspace, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as inp, destination.open("xb") as out:
            shutil.copyfileobj(inp, out)
        if p.sha(destination) != lock[name]["sha256"]:
            raise ValueError(f"Input changed while copying: {name}; isolated workspace retained for inspection")
        copied.append({"path": name, "sha256": lock[name]["sha256"]})
    data_check = dataset_audit(workspace, config, recipe)
    tokenizer_source = snapshot_text(snapshot, f"genrec/models/{recipe['model']}/tokenizer.py")
    runtime_sid, sid_alias = bind_runtime_sid(workspace, config, recipe, copied, tokenizer_source)
    config = configure_paths(config, workspace, work, recipe, run_id)
    formal_config = dict(config)
    if mode == "smoke":
        config = apply_smoke(config)
    overrides = {key: {"formal": formal_config.get(key), "smoke": value} for key, value in config.items() if value != formal_config.get(key)}
    p.write_json(work / "configuration_audit.json", config_audit)
    p.write_json(work / "dataset_audit.json", data_check)
    p.write_json(work / "input_copies.json", copied)
    config_path = work / "resolved_config.json"
    p.write_json(config_path, config)  # JSON is valid YAML for the pinned loader.
    environment = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "packages": sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())}
    p.write_json(work / "environment.json", environment)
    command = [sys.executable, "-u", "main.py", "--model", recipe["model"], "--dataset", "AmazonReviews2014", "--config", str(config_path)]
    launch = {"argv": command, "cwd": str(engine), "recipe": recipe, "config_layers": sources, "code": code,
              "config_sha256": p.sha(config_path), "smoke_overrides": overrides,
              "runtime_sid": runtime_sid, "sid_cache_alias": sid_alias,
              "status": "prepared", "checkpoint_argument": None,
              "scope": "Frozen historical SID, randomly initialized decoder; no tokenizer retraining"}
    p.write_json(work / "launch.json", launch)
    env = os.environ.copy()
    env.update(HF_HOME=str(work / "huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_CACHE=str(work / "huggingface/datasets"), TOKENIZERS_PARALLELISM="false",
               TORCHDYNAMO_DISABLE="1", WANDB_MODE="disabled", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    # Avoid importing an unrelated genrec checkout through inherited PYTHONPATH.
    env.pop("PYTHONPATH", None)
    returncode = None
    try:
        with (work / "console.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=engine, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
            launch.update(status="started", pid=child.pid)
            p.write_json(work / "launch.json", launch)
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if returncode:
            raise ValueError(f"Replay failed with exit {returncode}; preserved log: {work / 'console.log'}")
        audit = post_run_audit(work, workspace, recipe, config, copied, data_check["counts"], mode, runtime_sid)
        p.write_json(work / "replay_check.json", audit)
        p.write_json(work / "completion.json", {"exit_code": returncode, "status": "completed_and_checked",
                                              "completed_utc": datetime.now(timezone.utc).isoformat(),
                                              "historical_metrics_reproduced": "not_asserted"})
    except BaseException as error:
        p.write_json(work / "completion.json", {"exit_code": returncode, "status": "failed_or_interrupted",
                                              "error": str(error), "completed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    print(f"Replay completed and diagnostics checked: {work}")
