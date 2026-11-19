"""Offline CPU audit of the one authorized Stage07 formal delivery.

Usage: python -B scripts/analysis/audit_item_local_delivery.py \
    --run-dir <downloaded formal run> --output <new audit.json>

PyTorch (CPU is sufficient) and PyYAML are required for a passing audit. No
model is constructed, CUDA called, network accessed, or delivered file changed.
The receipt is an artifact audit, not Goal completion, GPU-release proof, or a
full training-resume acceptance. A negative exploratory signal is a valid result.

By default three small runtime inputs must be delivered under workspace/ and
all twelve inputs require an independent input_hash_receipt.json. Alternatively
--input-root verifies all twelve input byte streams locally. The remote receipt
schema is documented in audit_inputs(); it is evidence, not a digital signature.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "stage07_item_local_seed2024_20261020_0045"
BASELINE_RUN_ID = "diff_formal_pse_4x256_repo_repro_v6_bf16"
PROJECT_COMMIT = "8b7b9afe17ef425efe9279524ecf7a1e97148853"
PREPARE_COMMIT = "64abf0f06eb4ba4d9759091f74d397c91061726b"
ENGINE_COMMIT = "1dc2a0c5ad01e2a2878b9de400b91f3a7e809f25"
ENGINE_BASE = "39e94e6153c97e59877ba02d4ef1d9a202ed5384"
ENGINE_FILES = tuple("genrec/models/DIFF_GRM/" + name for name in
                     ("model.py", "item_local_attention.py", "trainer.py", "training_state.py"))
METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10")
PATH_KEYS = ("cache_dir", "log_dir", "tensorboard_log_dir", "ckpt_dir", "run_metadata_dir",
             "position_loss_log_dir", "diffusion_diagnostics_log_dir", "sent_emb_path")
NEW_KEYS = ("item_local_attention", "run_local_time", "training_state_dir", "resume_training_state",
            "training_deadline_unix_seconds", "training_deadline_reserve_seconds", "training_state_identity")
RUNTIME_KEYS = {"accelerator", "device", "use_ddp", "run_local_time", "current_split",
                "resume_training_state", "training_deadline_unix_seconds", "training_deadline_reserve_seconds"}
ATTEMPT_RUNTIME_KEYS = {"resume_training_state", "training_deadline_unix_seconds",
                        "training_deadline_reserve_seconds"}


class AuditFailure(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def finite(value):
    need(not isinstance(value, bool), "A boolean is not a numeric measurement")
    value = float(value)
    need(math.isfinite(value), "Nonfinite numeric value")
    return value


def integer(value):
    result = int(value)
    need(not isinstance(value, bool) and finite(value) == result, f"Expected integer, got {value!r}")
    return result


def equal_number(left, right):
    return math.isclose(finite(left), finite(right), rel_tol=1e-10, abs_tol=1e-12)


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    need(result.tzinfo is not None, "Timestamp lacks timezone")
    return result


def normpath(value):
    return str(value).replace("\\", "/").rstrip("/")


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_text(text):
    def invalid(value):
        raise AuditFailure(f"Nonstandard JSON numeric constant: {value}")
    return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=invalid)


def read_json(path):
    return json_text(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path_or_text, *, text=False):
    source = io.StringIO(path_or_text) if text else Path(path_or_text).open(encoding="utf-8-sig", newline="")
    with source as handle:
        reader = csv.DictReader(handle)
        need(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Missing/duplicate CSV header")
        rows = list(reader)
        need(all(None not in row and None not in row.values() for row in rows), "Malformed/truncated CSV row")
        return rows


def git_bytes(checkout, commit, name):
    result = subprocess.run(["git", "-C", str(checkout), "show", f"{commit}:{name}"],
                            capture_output=True, check=False)
    need(result.returncode == 0, f"Cannot read pinned offline source {commit}:{name}")
    return result.stdout


def normalized_metrics(values, *, selection=False):
    need(isinstance(values, dict), "Metrics must be an object")
    result = {key: finite(value) for key, value in values.items()}
    for k in (5, 10):
        canonical, alias = f"hit_rate@{k}", f"recall@{k}"
        if canonical not in result and alias in result:
            result[canonical] = result[alias]
        if canonical in result and alias in result:
            need(equal_number(result[canonical], result[alias]), f"Contradictory HR/recall alias at {k}")
    if "duplicate@10" not in result and "dup@10" in result:
        result["duplicate@10"] = result["dup@10"]
    if "dup@10" in result and "duplicate@10" in result:
        need(equal_number(result["dup@10"], result["duplicate@10"]), "Contradictory duplicate aliases")
    for key in METRICS:
        need(key in result and 0 <= result[key] <= 1, f"Missing/out-of-range metric {key}")
    need(result["ndcg@5"] <= result["hit_rate@5"] + 1e-12, "NDCG@5 exceeds HR@5")
    need(result["ndcg@10"] <= result["hit_rate@10"] + 1e-12, "NDCG@10 exceeds HR@10")
    need(result["hit_rate@5"] <= result["hit_rate@10"] + 1e-12, "HR decreases from k=5 to k=10")
    need(result["ndcg@5"] <= result["ndcg@10"] + 1e-12, "NDCG decreases from k=5 to k=10")
    score = .8 * result["ndcg@10"] + .2 * result["hit_rate@10"]
    if selection or "weighted_score" in result:
        # The pinned evaluator weights each sample in torch.float32 before
        # averaging. Weighting the two reported means in Python double differs
        # by rounding; keep this tolerance local to the formula reconstruction.
        need("weighted_score" in result and math.isclose(result["weighted_score"], score, rel_tol=0, abs_tol=1e-8), "weighted_score formula mismatch")
    return result


def expected_model_shapes(config):
    """Independent shape manifest of the pinned, nonjoint architecture.

    This model has no persistent buffers or duplicated parameter aliases. The
    manifest therefore permits parameter counting without importing the engine.
    An unfamiliar architecture fails rather than guessing which tensors count.
    """
    d, h = integer(config["n_embd"]), integer(config["n_inner"])
    m, codes = integer(config["n_digit"]), integer(config["codebook_size"])
    shapes = {"embedding.weight": (3 + m * codes, d), "item_mlp.0.weight": (d, m * d),
              "item_mlp.0.bias": (d,), "item_mlp.2.weight": (d, d), "item_mlp.2.bias": (d,),
              "mask_emb_table.weight": (m, d), "pos_emb_enc.weight": (integer(config["max_history_len"]), d)}

    def norm(prefix):
        shapes[prefix + ".weight"] = (d,)
        shapes[prefix + ".bias"] = (d,)

    def attention(prefix):
        shapes[prefix + ".qkv.weight"] = (3 * d, d)
        shapes[prefix + ".proj.weight"] = (d, d)
        shapes[prefix + ".proj.bias"] = (d,)

    def mlp(prefix):
        shapes[prefix + ".c_fc.weight"] = (h, d)
        shapes[prefix + ".c_fc.bias"] = (h,)
        shapes[prefix + ".c_proj.weight"] = (d, h)
        shapes[prefix + ".c_proj.bias"] = (d,)

    for index in range(integer(config["encoder_n_layer"])):
        prefix = f"encoder_blocks.{index}"
        norm(prefix + ".ln_1")
        attention(prefix + ".attn")
        norm(prefix + ".ln_2")
        mlp(prefix + ".mlp")
    for index in range(integer(config["decoder_n_layer"])):
        prefix = f"decoder_blocks.{index}"
        norm(prefix + ".ln_1")
        attention(prefix + ".self_attn")
        norm(prefix + ".ln_2")
        attention(prefix + ".cross_attn")
        norm(prefix + ".ln_3")
        mlp(prefix + ".mlp")
    norm("ln_f")
    norm("item_local_attention.norm")
    shapes.update({"item_local_attention.attention.in_proj_weight": (3 * d, d),
                   "item_local_attention.attention.in_proj_bias": (3 * d,),
                   "item_local_attention.attention.out_proj.weight": (d, d),
                   "item_local_attention.attention.out_proj.bias": (d,)})
    return shapes


class Audit:
    def __init__(self, args):
        self.args = args
        self.run = args.run_dir.resolve()
        self.pack = ROOT / "experiments/03_diffgrm_tokenizer_comparison"
        self.checks = []
        self.files = {}
        self.data = {}
        self.details = {}

    def record_file(self, path):
        path = Path(path).resolve()
        need(path.is_file(), f"Missing required file: {path}")
        if str(path) not in self.files:
            self.files[str(path)] = {"size_bytes": path.stat().st_size, "sha256": sha(path)}
        return path

    def local(self, name):
        path = (self.run / name).resolve()
        need(path.is_relative_to(self.run), f"Artifact path escapes run directory: {name}")
        return self.record_file(path)

    def document(self, name):
        if name not in self.data:
            self.data[name] = read_json(self.local(name))
        return self.data[name]

    def execution_attempts(self):
        """STARTED binds actual execution; root receipts retain first writes."""
        if not hasattr(self, "executions"):
            executions = []
            for path in sorted((self.run / "attempts").glob("*/STARTED.json")):
                prefix = path.parent.relative_to(self.run).as_posix() + "/"
                started = self.document(prefix + "STARTED.json")
                need(started["mode"] == "formal", "Nonformal execution in formal attempt directory")
                stamp(started["utc"])
                executions.append({"name": path.parent.name, "prefix": prefix, "started": started})
            need(executions, "No STARTED-bound formal execution attempts were delivered")
            executions.sort(key=lambda row: stamp(row["started"]["utc"]))
            self.executions = executions
        return self.executions

    def source_identity(self, code, project_commit, context):
        need(code["project"]["head"] == project_commit and code["engine"]["head"] == ENGINE_COMMIT,
             f"{context}: source commits differ from the authorized source")
        need(code["files"] == self.engine_hashes and code["runner_sha256"] == self.runner_hash,
             f"{context}: source file hashes differ from pinned commit bytes")
        need(code["engine"].get("status") == "", f"{context}: engine checkout was not recorded clean")
        need(code.get("historical_runtime_commit") is None, f"{context}: historical source gap was silently filled")

    def section(self, name, function):
        try:
            detail = function()
            self.checks.append({"check": name, "status": "passed", "details": detail})
        except Exception as error:
            self.checks.append({"check": name, "status": "failed", "error_type": type(error).__name__, "error": str(error)})

    def mapped(self, remote):
        """Map a recorded absolute runtime path by its run-id directory anchor."""
        value = normpath(remote)
        anchor = "/" + RUN_ID + "/"
        need(value.count(anchor) == 1, f"Recorded artifact path is not anchored to formal run: {remote}")
        relative = value.split(anchor, 1)[1]
        need(not relative.startswith("smoke/"), "Smoke path cannot identify a formal artifact")
        need(".." not in PurePosixPath(relative).parts, "Unsafe artifact path")
        return self.local(relative)

    def unique(self, directory, pattern):
        candidates = sorted((self.run / directory).rglob(pattern))
        need(len(candidates) == 1, f"Expected exactly one {directory}/{pattern}, found {len(candidates)}")
        return self.record_file(candidates[0])

    def audit_inventory(self):
        need(self.run.is_dir(), "--run-dir is not a directory")
        required = ("identity.json", "study.json", "baseline_resolved_config.json", "resolved_config.json",
                    "parsed_config.json", "input_identities.json", "protocol_audit.json", "code_identity.json",
                    "model_identity.json", "environment.json", "STARTED.json", "FIT_COMPLETE.json",
                    "checkpoint_reload.json", "runtime_phase.json", "TEST_STARTED.json", "TEST_RESULT.json",
                    "results.json", "COMPLETE.json", "evaluation_timings.json", "training_state/epoch_timings.json")
        missing = []
        for name in required:
            try:
                self.document(name)
            except Exception as error:
                missing.append({"file": name, "error": str(error)})
        need(not missing, f"Missing or invalid mandatory artifacts: {json.dumps(missing)}")
        need(not (self.run / "INCOMPLETE.json").exists(), "INCOMPLETE.json remains in formal root; reconcile this failed attempt before accepting delivery")
        need(not (self.run / "smoke_results.json").exists(), "Smoke results in selected run root")
        self.details["metadata_path"] = self.unique("run_metadata", "*_run_metadata.json")
        self.data["metadata"] = read_json(self.details["metadata_path"])
        for key, suffix in (("train_csv", "training_metrics"), ("eval_csv", "evaluation_metrics"),
                            ("position_csv", "position_metrics"), ("order_csv", "order_metrics")):
            self.details[key] = self.unique("diagnostics", f"*_{suffix}.csv")
        return {"mandatory_json_files": len(required), "last_pt_delivered": (self.run / "training_state/last.pt").is_file()}

    def audit_references(self):
        recipe_path = self.record_file(self.pack / "reproducibility/recipes/pse_seed2024.json")
        self.recipe = read_json(recipe_path)
        need(self.recipe["run_id"] == BASELINE_RUN_ID and self.recipe["seed"] == 2024, "Locked historical recipe identity changed")
        versions = read_json(self.record_file(self.pack / "reproducibility/code_versions.json"))
        engines = [row for row in versions["repositories"] if row["name"] == "DiffGRM"]
        need(len(engines) == 1 and engines[0]["commit"] == ENGINE_BASE, "Pinned reconstruction engine differs")
        self.base_source = engines[0]
        snapshot = self.record_file(ROOT / self.base_source["snapshot"])
        need(sha(snapshot) == self.base_source["sha256"], "Pinned baseline source archive SHA mismatch")
        self.lock_path = self.record_file(self.pack / "reproducibility/assets.lock.json")
        self.lock = read_json(self.lock_path)["artifacts"]
        self.historical = read_json(self.record_file(self.pack / self.recipe["metadata"]))
        rows = read_csv(self.record_file(self.pack / "results.csv"))
        selected = [row for row in rows if row.get("id") == "pse_seed2024"]
        need(len(selected) == 1, "Expected one Stage03 PSE baseline result")
        self.baseline = normalized_metrics({key: selected[0][key] for key in METRICS})
        for key in METRICS:
            need(equal_number(self.baseline[key], self.historical["test_results"][key]), f"Historical baseline sources disagree: {key}")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT / "scripts/repro"))
        import yaml
        import project_replay as replay
        self.replay = replay
        self.formal, self.formal_sources = replay.load_formal_config(self.recipe, self.pack, snapshot, yaml)
        self.config_audit = replay.audit_formal_config(self.formal, self.recipe, self.historical)
        self.engine_hashes = {name: digest_bytes(git_bytes(ROOT / "external/DiffGRM", ENGINE_COMMIT, name)) for name in ENGINE_FILES}
        self.study_bytes = git_bytes(ROOT, PROJECT_COMMIT, "configs/07_item_local_attention/item_local_seed2024.json")
        self.expected_study = json_text(self.study_bytes.decode("utf-8-sig"))
        self.runner_hash = digest_bytes(git_bytes(ROOT, PROJECT_COMMIT, "scripts/train/run_item_local_attention.py"))
        need(self.expected_study["run_id"] == RUN_ID, "Pinned study has a different run_id")
        return {"baseline_run_id": BASELINE_RUN_ID, "config_basis": self.recipe["config_basis"],
                "reconstructed_from": self.formal_sources, "historical_full_resolved_config_available": False}

    def audit_identity(self):
        identity, study = self.document("identity.json"), self.document("study.json")
        need(identity["run_id"] == RUN_ID and "smoke" not in identity["run_id"].lower(), "Selected run is not the authorized formal run")
        need(study == self.expected_study and study["seed"] == 2024, "Study differs from pinned seed2024 study")
        need(identity["study_sha256"] == digest_bytes(self.study_bytes), "Study source-byte hash differs from executed pinned study")
        need(identity["baseline_run_id"] == BASELINE_RUN_ID, "Wrong historical baseline identity")
        need(identity["sid_sha256"] == self.lock[self.recipe["initial_sid"]]["sha256"], "Identity SID SHA differs from asset lock")
        need(self.document("STARTED.json")["mode"] == "formal", "STARTED is not formal")
        complete, results = self.document("COMPLETE.json"), self.document("results.json")
        need(complete["mode"] == "formal" and complete["status"] == results["status"] == "formal_completed", "Formal terminal statuses missing")
        # record_attempt preserves each root file's first write independently.
        # code_identity may therefore come from the earlier CPU-only prepare,
        # while root STARTED/parsed_config first appear during actual training.
        code = self.document("code_identity.json")
        prepare_commit = code["project"]["head"]
        need(prepare_commit in {PREPARE_COMMIT, PROJECT_COMMIT}, "Unknown initial preparation source")
        self.source_identity(code, prepare_commit, "First-write preparation receipt")
        prepare_origins = []
        for path in sorted((self.run / "attempts").glob("*/code_identity.json")):
            name = path.relative_to(self.run).as_posix()
            if self.document(name) == code:
                prepare_origins.append(path.parent.name)
        need(prepare_origins, "First-write source receipt is not preserved in an attempt")
        if prepare_commit == PREPARE_COMMIT:
            need(any(not (self.run / "attempts" / name / "STARTED.json").exists()
                     for name in prepare_origins), "Initial 64abf0f source lacks a CPU-only preparation attempt")
        executions = self.execution_attempts()
        for execution in executions:
            prefix, name = execution["prefix"], execution["name"]
            self.source_identity(self.document(prefix + "code_identity.json"), PROJECT_COMMIT,
                                 "Formal execution " + name)
            need(self.document(prefix + "identity.json") == identity and self.document(prefix + "study.json") == study,
                 f"Formal execution {name}: run/study identity differs")
            environment = self.document(prefix + "environment.json")
            need(integer(environment["visible_gpus"]) == 1 and "H100" in environment["gpu"].upper(),
                 f"Formal execution {name}: did not record exactly one H100")
            need(stamp(execution["started"]["utc"]) <= stamp(complete["completed_utc"]),
                 "Execution starts after formal completion")
        first = executions[0]["prefix"]
        need(self.document("STARTED.json") == executions[0]["started"], "Root STARTED is not the first execution receipt")
        for name in ("parsed_config.json", "environment.json", "model_identity.json"):
            need(self.document(name) == self.document(first + name), f"Root {name} is not its preserved first execution receipt")
        need(self.recipe.get("historical_runtime_commit") is None, "Historical source gap was silently filled")
        protocol = self.document("protocol_audit.json")
        need(protocol["recipe_sha256"] == sha(self.pack / "reproducibility/recipes/pse_seed2024.json"), "Recipe SHA mismatch")
        need(protocol["sources"] == self.formal_sources and protocol["baseline_code_snapshot"] == self.base_source, "Baseline reconstruction provenance differs")
        need(protocol["configuration"] == self.config_audit, "Reported baseline metadata audit differs from independent reconstruction")
        return {"run_id": RUN_ID, "seed": 2024, "engine_commit": ENGINE_COMMIT,
                "project_commit": PROJECT_COMMIT, "historical_runtime_commit": None,
                "execution_attempts_checked": [row["name"] for row in executions],
                "first_write_project_commit": prepare_commit, "first_write_source_origin_attempts": prepare_origins,
                "metadata_scope": "Root source receipt preserves CPU preparation; every STARTED-bound formal attempt independently requires the frozen execution commits and file hashes"}

    def audit_configuration(self):
        baseline = self.document("baseline_resolved_config.json")
        need(baseline == self.formal, "baseline_resolved_config does not equal reconstruction of pinned Stage03 recipe")
        resolved, parsed = self.document("resolved_config.json"), self.document("parsed_config.json")
        allowed = set(PATH_KEYS) | set(NEW_KEYS) | {"run_id", "joint_crosspse_enabled"}
        differences = {key: {"baseline": baseline.get(key), "actual": resolved.get(key)}
                       for key in set(baseline) | set(resolved) if key not in allowed and baseline.get(key) != resolved.get(key)}
        need(not differences, f"Unapproved recipe changes: {json.dumps(differences)}")
        need(resolved["run_id"] == resolved["run_local_time"] == RUN_ID, "Resolved run identity differs")
        need(resolved["item_local_attention"] == {"enabled": True, "num_heads": 4, "dropout": 0.0}, "Attention configuration differs")
        need(resolved["rand_seed"] == 2024 and resolved["evaluation_level"] == "sid", "Seed/evaluation level differs")
        need(resolved.get("max_train_batches_per_epoch") in (None, 0) and resolved.get("max_eval_batches") in (None, 0), "Formal run has a batch truncation")
        need(not resolved.get("joint_crosspse_enabled") and not resolved.get("mask_schedule", {}).get("enabled"), "Unsupported joint/mask schedule")
        need(resolved["share_decoder_output_embedding"] is True and resolved["norm_type"] == "layernorm", "Unknown model parameter manifest")
        need(not resolved.get("checkpoint") and not resolved.get("checkpoint_path") and not resolved.get("joint_recommender_checkpoint"), "Unexpected pretrained recommender initialization")
        mismatches = {key: {"resolved": value, "parsed": parsed.get(key)} for key, value in resolved.items()
                      if key not in ATTEMPT_RUNTIME_KEYS and parsed.get(key) != value}
        need(not mismatches, f"Runtime parsed config differs: {json.dumps(mismatches)}")
        need(parsed.get("use_ddp", False) is False, "Formal run enabled DDP")
        inputs = self.document("input_identities.json")
        remote_root = normpath(inputs["runtime_workspace"])
        need(remote_root.endswith("/" + RUN_ID + "/workspace"), "Runtime workspace is not formal-run scoped")
        need(normpath(resolved["cache_dir"]) == remote_root + "/data/cache", "Dataset cache path binding differs")
        remote_run = remote_root.removesuffix("/workspace")
        expected_paths = {"log_dir": "logs", "tensorboard_log_dir": "tensorboard", "ckpt_dir": "checkpoints",
                          "run_metadata_dir": "run_metadata", "position_loss_log_dir": "positions",
                          "diffusion_diagnostics_log_dir": "diagnostics", "training_state_dir": "training_state"}
        for key, tail in expected_paths.items():
            need(normpath(resolved[key]) == remote_run + "/" + tail, f"Output path binding differs: {key}")
        if resolved.get("sent_emb_path"):
            need(normpath(resolved["sent_emb_path"]).startswith(remote_root + "/"), "Embedding path not bound to runtime workspace")
        need(resolved["training_state_identity"]["engine_files"] == self.engine_hashes, "Training-state source identity differs")
        expected_overrides = {key: {"historical": baseline.get(key), "resolved": value}
                              for key, value in resolved.items() if value != baseline.get(key)}
        need(self.document("protocol_audit.json")["mode_overrides"] == expected_overrides, "Protocol override receipt omits or changes a runtime override")
        executions = self.execution_attempts()
        for execution in executions:
            prefix, name = execution["prefix"], execution["name"]
            actual_resolved = self.document(prefix + "resolved_config.json")
            actual_parsed = self.document(prefix + "parsed_config.json")
            need(self.document(prefix + "baseline_resolved_config.json") == baseline,
                 f"Formal execution {name}: baseline reconstruction differs")
            recipe_fields = lambda value: {key: item for key, item in value.items() if key not in ATTEMPT_RUNTIME_KEYS}
            need(recipe_fields(actual_resolved) == recipe_fields(resolved),
                 f"Formal execution {name}: nonruntime resolved configuration changed")
            need(all(actual_parsed.get(key) == value for key, value in actual_resolved.items()),
                 f"Formal execution {name}: parsed config differs from its own resolved config")
            need(recipe_fields(actual_parsed) == recipe_fields(parsed),
                 f"Formal execution {name}: nonruntime parsed configuration changed")
            need(actual_parsed.get("use_ddp", False) is False, f"Formal execution {name}: enabled DDP")
            need(actual_resolved["resume_training_state"] == execution["started"]["resume"]
                 and actual_resolved["training_deadline_unix_seconds"] == execution["started"]["deadline_unix"],
                 f"Formal execution {name}: STARTED/runtime configuration binding differs")
            need(self.document(prefix + "input_identities.json") == inputs,
                 f"Formal execution {name}: input identities changed")
            actual_protocol = self.document(prefix + "protocol_audit.json")
            expected_protocol = dict(self.document("protocol_audit.json"))
            expected_protocol["mode_overrides"] = {key: {"historical": baseline.get(key), "resolved": value}
                                                    for key, value in actual_resolved.items() if value != baseline.get(key)}
            need(actual_protocol == expected_protocol, f"Formal execution {name}: protocol provenance/overrides differ")
        # The final checkpoint's effective recipe is checked against the latest
        # actual execution, without rewriting the preserved first root receipts.
        self.config = self.document(executions[-1]["prefix"] + "parsed_config.json")
        return {"all_reconstructed_recipe_fields_checked": True, "semantic_changes": ["item_local_attention"],
                "runtime_added_keys": sorted(set(parsed) - set(resolved)), "historical_original_full_config_gap_retained": True,
                "execution_attempts_checked": [row["name"] for row in executions],
                "cross_attempt_changes_allowed_only": sorted(ATTEMPT_RUNTIME_KEYS),
                "effective_config_attempt": executions[-1]["name"],
                "root_resolved_and_parsed_scope": "Independent first writes; CPU-prepare deadlines may differ from first execution, while all recipe fields remain fixed"}

    def audit_inputs(self):
        """Remote receipt: status=passed, run_id, verified_utc,
        verification_method=sha256_of_runtime_workspace_bytes,
        files=[{path, sha256, size_bytes, expected_sha256, runtime_absolute_path}].
        --input-root instead checks all bytes below that explicitly supplied root.
        """
        inputs = self.document("input_identities.json")
        names = list(dict.fromkeys(self.recipe["required_artifacts"] + self.recipe["tokenizer_artifacts"]))
        records = inputs["files"]
        need(len(records) == len(names) and {row["path"] for row in records} == set(names), "Input identity set is missing, duplicated, or unexpected")
        need(inputs["asset_lock_sha256"] == sha(self.lock_path), "Input asset-lock SHA differs")
        records = {row["path"]: row for row in records}
        expected_hashes = {name: self.lock[name]["sha256"] for name in names}
        for name, row in records.items():
            need(row["sha256"] == expected_hashes[name] and integer(row["size_bytes"]) == self.lock[name]["size_bytes"], f"Input identity differs from lock: {name}")
        need(self.config["training_state_identity"]["inputs"] == expected_hashes, "Training-state input identity is incomplete")
        sid = self.recipe["initial_sid"]
        need(normpath(inputs["sid_path"]) == normpath(inputs["runtime_workspace"]) + "/" + sid, "Runtime SID path binding differs")
        small = [sid, "data/cache/AmazonReviews2014/Beauty/processed/id_mapping.json",
                 "data/cache/AmazonReviews2014/Beauty/processed/all_item_seqs.json"]
        locally_checked = []
        for name in small:
            path = self.local("workspace/" + name)
            need(sha(path) == expected_hashes[name] and path.stat().st_size == records[name]["size_bytes"], f"Delivered small input bytes differ: {name}")
            locally_checked.append(name)
        dataset = self.replay.dataset_audit(self.run / "workspace", self.formal, self.recipe)
        need(dataset == self.document("protocol_audit.json")["dataset"], "Dataset counts/SID catalogue differ from protocol audit")
        if self.args.input_root is not None:
            input_root = self.args.input_root.resolve()
            for name in names:
                path = (input_root / name).resolve()
                need(path.is_relative_to(input_root), "Unsafe optional input path")
                self.record_file(path)
                need(sha(path) == expected_hashes[name] and path.stat().st_size == records[name]["size_bytes"], f"Optional full input bytes differ: {name}")
            scope = "all_twelve_inputs_locally_byte_verified"
        else:
            receipt_path = self.args.input_hash_receipt or self.run / "input_hash_receipt.json"
            receipt = read_json(self.record_file(receipt_path))
            need(receipt.get("status") == "passed" and receipt["run_id"] == RUN_ID, "Remote input-byte receipt is not passed/has wrong identity")
            stamp(receipt["verified_utc"])
            need(receipt["verification_method"] == "sha256_of_runtime_workspace_bytes", "Unknown remote input verification method")
            remote_rows = receipt["files"]
            need(len(remote_rows) == len(names) and {row["path"] for row in remote_rows} == set(names), "Remote byte receipt does not cover exactly twelve inputs")
            for row in remote_rows:
                name = row["path"]
                need(row["sha256"] == row["expected_sha256"] == expected_hashes[name], f"Remote actual input hash differs: {name}")
                need(integer(row["size_bytes"]) == records[name]["size_bytes"], f"Remote actual input size differs: {name}")
                need(normpath(row["runtime_absolute_path"]) == normpath(inputs["runtime_workspace"]) + "/" + name, f"Remote byte receipt checked a different path: {name}")
            scope = "identity_only_remote_bytes_verified_by_receipt"
        return {"identity_files": len(names), "local_byte_verified_inputs": locally_checked,
                "remaining_input_scope": scope, "dataset": dataset,
                "historical_hash_scope": "Migration asset hashes; original historical runtime hashes unavailable"}

    def audit_curves(self):
        rows = read_csv(self.details["train_csv"])
        need(rows, "Training curve is empty")
        count = len(rows)
        epochs = [integer(row["epoch"]) for row in rows]
        need(epochs == list(range(1, count + 1)) and count <= integer(self.config["epochs"]), "Training epochs are duplicate, gapped, reordered, or exceed maximum")
        batches = math.ceil(105668 / integer(self.config["train_batch_size"]))
        updates = math.ceil(batches / integer(self.config["gradient_accumulation_steps"]))
        warmup, total = integer(self.config["warmup_steps"]), updates * integer(self.config["epochs"])

        def learning_rate(step):
            factor = step / max(1, warmup) if step < warmup else max(0, .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
            return finite(self.config["lr"]) * factor

        for epoch, row in enumerate(rows, 1):
            need(finite(row["train_loss"]) >= 0, f"Invalid train loss at epoch {epoch}")
            need(integer(row["optimizer_updates"]) == updates and integer(row["cumulative_optimizer_updates"]) == epoch * updates, f"Truncated update count at epoch {epoch}")
            need(equal_number(row["lr_start"], learning_rate((epoch - 1) * updates)) and equal_number(row["lr_end"], learning_rate(epoch * updates)), f"Cosine/warmup LR curve differs at epoch {epoch}")
        timings = self.document("training_state/epoch_timings.json")
        need(isinstance(timings, list) and len(timings) == count, "Epoch timing curve is incomplete")
        for epoch, timing in enumerate(timings, 1):
            need(integer(timing["epoch"]) == epoch and integer(timing["batches"]) == batches and finite(timing["seconds"]) > 0, f"Incomplete/nonfinite epoch-loop timing at {epoch}")
        evaluations = read_csv(self.details["eval_csv"])
        groups = {}
        for row in evaluations:
            key = (row["split"], integer(row["epoch"]))
            need(key[0] in ("val", "test"), f"Unexpected formal evaluation split {key[0]}")
            group = groups.setdefault(key, {})
            need(row["metric"] not in group, f"Duplicate evaluation metric/event at {key}: {row['metric']}")
            group[row["metric"]] = finite(row["value"])
        val_epochs = [epoch for epoch in epochs if epoch >= integer(self.config["eval_start_epoch"]) and epoch % integer(self.config["eval_interval"]) == 0]
        need(val_epochs and set(groups) == {("val", epoch) for epoch in val_epochs} | {("test", 0)}, "Validation frequency/once-only final test differs from historical protocol")
        scores = {epoch: normalized_metrics(groups[("val", epoch)], selection=True)["weighted_score"] for epoch in val_epochs}
        best, best_epoch, patience_count, stop_epoch = -math.inf, None, 0, None
        for epoch in val_epochs:
            improved = scores[epoch] > best + finite(self.config.get("min_delta", 0))
            if improved:
                best, best_epoch = scores[epoch], epoch
            active = not self.config["early_stop_after_warmup"] or epoch * updates >= warmup
            patience_count = 0 if not active or improved else patience_count + 1
            if active and patience_count >= integer(self.config["patience"]):
                stop_epoch = epoch
                break
        expected_end = stop_epoch if stop_epoch is not None else integer(self.config["epochs"])
        need(count == expected_end, f"Curve ends at {count}; complete original stopping rule requires {expected_end} (no truncated fit accepted)")
        maximum_epoch = max(scores, key=lambda epoch: scores[epoch])
        need(best_epoch == maximum_epoch and equal_number(best, max(scores.values())), "Selected best is not first maximum weighted_score")
        for name in ("FIT_COMPLETE.json", "results.json"):
            receipt = self.document(name)
            need(integer(receipt["best_epoch"]) == best_epoch and equal_number(receipt["best_val_score"], best), f"Best-validation receipt differs: {name}")
            need(finite(receipt["fit_elapsed_seconds"]) > 0, f"Missing fit elapsed seconds: {name}")
        self.curve = {"epochs": count, "updates_per_epoch": updates, "cumulative_updates": count * updates,
                      "best_epoch": best_epoch, "best_val_score": best, "eval_count": len(val_epochs),
                      "no_improve_count": patience_count, "stop_reason": "early_stopping" if stop_epoch else "max_epochs"}
        self.training_rows, self.groups = rows, groups
        self.epoch_timings, self.val_epochs = timings, val_epochs
        self.test_metrics = normalized_metrics(groups[("test", 0)], selection=True)
        return {**self.curve, "batches_per_epoch": batches, "weighted_score_formula": ".8*NDCG@10 + .2*HR@10",
                "weighted_formula_absolute_tolerance": 1e-8,
                "weighted_formula_tolerance_reason": "Evaluator weights per-sample float32 values before averaging; best selection uses exact CSV scores",
                "all_training_rows_finite": True, "original_frequency_and_early_stop_recomputed": True}

    def audit_diagnostic_curves(self):
        events = [("train", epoch) for epoch in range(1, self.curve["epochs"] + 1)]
        events += [("val", epoch) for epoch in self.val_epochs] + [("test", 0)]
        positions = read_csv(self.details["position_csv"])
        expected_positions = {(split, epoch, "ocn_masked_views" if split == "train" else "fully_masked", slot)
                              for split, epoch in events for slot in range(4)}
        observed = set()
        position_columns = {"epoch", "split", "context", "position", "loss", "nll", "normalized_nll", "accuracy",
                            "supervised_tokens", "available_views", "mask_rate"}
        for row in positions:
            need(set(row) == position_columns, "Unknown/incomplete position diagnostic schema")
            key = row["split"], integer(row["epoch"]), row["context"], integer(row["position"])
            need(key in expected_positions and key not in observed, f"Unexpected/duplicate position event: {key}")
            observed.add(key)
            for field in ("loss", "nll", "normalized_nll"):
                need(finite(row[field]) >= 0, f"Negative/nonfinite position {field}")
            for field in ("accuracy", "mask_rate"):
                need(0 <= finite(row[field]) <= 1, f"Invalid position {field}")
            tokens, views = integer(row["supervised_tokens"]), integer(row["available_views"])
            expected_views = 105668 * 4 if row["split"] == "train" else 22363
            need(0 < tokens <= views == expected_views, f"Position counts show truncated data at {key}")
            if row["split"] != "train":
                need(tokens == views, f"Fully masked evaluation did not include every target at {key}")
            need(equal_number(row["mask_rate"], tokens / views), "Position mask rate/count mismatch")
            need(equal_number(row["normalized_nll"], finite(row["nll"]) / math.log(256)), "Position normalized NLL formula mismatch")
        need(observed == expected_positions, "Position diagnostic curve is incomplete")
        order = read_csv(self.details["order_csv"])
        expected_order = {(split, epoch, "ocn_noising_hard_to_easy" if split == "train" else "cpd_greedy_decode_easy_to_hard", rank, slot)
                          for split, epoch in events for rank in range(1, 5) for slot in range(4)}
        observed, counts_by_rank = set(), {}
        order_columns = {"epoch", "split", "order_type", "rank", "position", "count", "frequency", "mean_confidence", "accuracy", "nll"}
        for row in order:
            need(set(row) == order_columns, "Unknown/incomplete order diagnostic schema")
            key = row["split"], integer(row["epoch"]), row["order_type"], integer(row["rank"]), integer(row["position"])
            need(key in expected_order and key not in observed, f"Unexpected/duplicate decode-order event: {key}")
            observed.add(key)
            count = integer(row["count"])
            total = 105668 if row["split"] == "train" else 22363
            need(0 <= count <= total and equal_number(row["frequency"], count / total), "Decode-order frequency/count mismatch or truncated data")
            rank_key = key[:4]
            counts_by_rank[rank_key] = counts_by_rank.get(rank_key, 0) + count
            measurements = ("mean_confidence", "accuracy", "nll")
            if row["split"] == "train" or count == 0:
                need(all(row[field] == "" for field in measurements), "Unexpected measurements in count-only/empty order cell")
            else:
                need(0 <= finite(row["mean_confidence"]) <= 1 and 0 <= finite(row["accuracy"]) <= 1 and finite(row["nll"]) >= 0, "Invalid/nonfinite evaluation order measurements")
        need(observed == expected_order, "Decode-order diagnostic curve is incomplete")
        for key, count in counts_by_rank.items():
            need(count == (105668 if key[0] == "train" else 22363), "Decode-order rank did not see every example")
        return {"position_rows": len(positions), "order_rows": len(order), "events": len(events),
                "four_position_and_sixteen_order_cells_per_event": True, "all_nonempty_measurements_finite": True,
                "full_train_and_heldout_example_counts_verified": True}

    def audit_test_and_results(self):
        timing = self.document("evaluation_timings.json")
        need(isinstance(timing, list) and len(timing) >= len(self.val_epochs) + 1, "Evaluation timing calls are missing")
        test_calls = [row for row in timing if row["split"] == "test"]
        need(len(test_calls) == 1 and timing[-1] == test_calls[0] and test_calls[0]["epoch"] is None, "Timing shows missing, duplicate, or premature final test")
        val_calls = timing[:-1]
        need(all(row["split"] == "val" and row["epoch"] in self.val_epochs for row in val_calls), "Timing contains an unplanned split/validation epoch")
        need([row["epoch"] for row in val_calls] == sorted(row["epoch"] for row in val_calls), "Validation timing epochs are reordered")
        need(set(row["epoch"] for row in val_calls) == set(self.val_epochs), "Validation timings do not cover all committed epochs")
        repeated = []
        for epoch in self.val_epochs:
            calls = [row for row in val_calls if row["epoch"] == epoch]
            # An interruption after evaluation and before boundary commit keeps
            # the old timing, while recovery rolls back CSVs. Keep that history
            # visible, and accept a repeated epoch only in distinct attempts.
            need(len({row["attempt"] for row in calls}) == len(calls), f"Repeated validation within the same attempt at {epoch}")
            if len(calls) > 1:
                repeated.append({"epoch": epoch, "attempts": [row["attempt"] for row in calls],
                                 "committed_timing_basis": "last completed call in append order"})
        for row in timing:
            need(finite(row["seconds"]) > 0 and row.get("attempt"), "Evaluation timing lacks positive duration/attempt")
            need(row["attempt"] in {item["name"] for item in self.execution_attempts()},
                 "Evaluation timing references an unverified execution attempt")
        fit = self.document("FIT_COMPLETE.json")
        fit_attempt = fit["fit_validation_timing_attempt"]
        need(fit_attempt in {item["name"] for item in self.execution_attempts()},
             "FIT receipt references an unverified execution attempt")
        fit_val_seconds = sum(finite(row["seconds"]) for row in timing if row["split"] == "val" and row["attempt"] == fit_attempt)
        need(equal_number(fit["fit_validation_seconds"], fit_val_seconds), "FIT validation time differs from exact recorded attempt calls")
        started, test_started = self.document("STARTED.json"), self.document("TEST_STARTED.json")
        test_result, complete = self.document("TEST_RESULT.json"), self.document("COMPLETE.json")
        times = [stamp(started["utc"]), stamp(fit["completed_utc"]), stamp(test_started["utc"]),
                 stamp(test_result["completed_utc"]), stamp(complete["completed_utc"])]
        need(times == sorted(times), "STARTED/FIT_COMPLETE/TEST_STARTED/TEST_RESULT/COMPLETE chronology is invalid")
        results = self.document("results.json")
        metadata = self.data["metadata"]
        for name, source in (("TEST_RESULT", test_result), ("results", results), ("run_metadata", metadata)):
            values = normalized_metrics(source["test_results"], selection=True)
            need(values == self.test_metrics, f"Test results differ from single CSV evaluation: {name}")
        for key in ("best_epoch", "best_val_score", "fit_elapsed_seconds", "fit_validation_seconds", "fit_validation_timing_attempt", "completed_utc"):
            need(results[key] == fit[key], f"Results omit/change FIT receipt field {key}")
        need(results["evaluation_level"] == "sid", "Results are not SID-level")
        comparison = {}
        for key in METRICS:
            old, new = self.baseline[key], self.test_metrics[key]
            need(old != 0, f"Undefined baseline relative delta at {key}")
            comparison[key] = {"baseline": old, "new": new, "absolute_delta": new - old, "relative_change": new / old - 1}
            recorded = results["baseline_comparison"][key]
            need(all(equal_number(recorded[field], value) for field, value in comparison[key].items()), f"Incorrect baseline comparison: {key}")
        signal = comparison["ndcg@10"]["relative_change"] >= .03 and comparison["hit_rate@10"]["absolute_delta"] >= 0
        need(results["exploratory_positive_signal"] is signal, "Exploratory signal was not recomputed correctly")
        need(all(equal_number(self.document("study.json")["baseline_test"][key], self.baseline[key]) for key in METRICS), "Study baseline differs from Stage03 result")
        phase = self.document("runtime_phase.json")
        need(phase["phase"] == "final_test" and phase["training_complete"] is True and finite(phase["evaluation_reserve_seconds"]) == 0, "Runtime final-test phase receipt is invalid")
        need(times[1] <= stamp(phase["utc"]) <= times[2], "Final-test phase occurs outside fit-to-test transition")
        self.comparison = comparison
        return {"test_calls_in_timing": 1, "test_metric_groups_in_csv": 1,
                "repeated_validation_attempts_retained": repeated,
                "chronology_utc": [value.isoformat() for value in times], "baseline_comparison": comparison,
                "exploratory_positive_signal": signal, "signal_is_not_a_delivery_pass_requirement": True}

    def audit_metadata(self):
        metadata, config = self.data["metadata"], self.config
        need(metadata["run_id"] == RUN_ID and integer(metadata["seed"]) == 2024 and metadata["model"] == "DIFF_GRM", "Metadata run/seed/model differs")
        keys = ("sid_quantizer", "n_digit", "codebook_size", "masking_strategy", "guided_select", "guided_refresh_each_step",
                "diffusion_loss_reduction", "train_batch_size", "gradient_accumulation_steps", "lr", "mixed_precision",
                "max_grad_norm", "numerical_guard", "numerical_guard_parameter_interval", "eval_start_epoch", "eval_interval",
                "patience", "early_stop_after_warmup", "evaluation_level")
        need(all(metadata.get(key) == config.get(key) for key in keys), "Runtime metadata/config control fields differ")
        for key in ("best_epoch", "best_val_score", "updates_per_epoch"):
            need(equal_number(metadata[key], self.curve[key]), f"Runtime metadata/curve differs: {key}")
        need(integer(metadata["total_optimizer_steps"]) == integer(config["epochs"]) * self.curve["updates_per_epoch"], "Metadata scheduler total differs")
        need(integer(metadata["warmup_steps"]) == integer(config["warmup_steps"]), "Metadata warmup differs")
        need(normpath(metadata["semantic_ids_path"]) == normpath(self.document("input_identities.json")["sid_path"]), "Metadata consumed a different SID path")
        need(metadata["semantic_ids_sha256"] == self.document("identity.json")["sid_sha256"], "Metadata consumed a different SID hash")
        for field, local in (("training_metrics_csv", "train_csv"), ("evaluation_metrics_csv", "eval_csv"),
                             ("position_metrics_csv", "position_csv"), ("order_metrics_csv", "order_csv")):
            need(self.mapped(metadata[field]) == self.details[local], f"Metadata points to a different artifact: {field}")
        need(self.mapped(metadata["checkpoint"]) == self.local("training_state/best_model.bin"), "Metadata best checkpoint path differs")
        need(not metadata.get("joint_crosspse_enabled", False), "Runtime metadata enabled joint training")
        return {"checked_config_fields": list(keys), "SID_consumption_path_and_sha_bound": True}

    def load_torch(self):
        if not hasattr(self, "torch"):
            import torch
            self.torch = torch
        return self.torch

    def tensor_state(self, state, shapes):
        torch = self.load_torch()
        need(isinstance(state, dict) and set(state) == set(shapes), f"Checkpoint key layout differs; missing={sorted(set(shapes)-set(state))}, extra={sorted(set(state)-set(shapes))}")
        for name, shape in shapes.items():
            value = state[name]
            need(torch.is_tensor(value) and value.device.type == "cpu" and tuple(value.shape) == shape, f"Checkpoint tensor shape/device differs: {name}")
            need(value.is_floating_point() and bool(torch.isfinite(value).all()), f"Checkpoint tensor is nonfinite/nonfloating: {name}")

    def audit_best_checkpoint(self):
        path = self.local("training_state/best_model.bin")
        digest = sha(path)
        reload = self.document("checkpoint_reload.json")
        for name in ("TEST_STARTED.json", "results.json", "COMPLETE.json"):
            need(self.document(name)["checkpoint_sha256"] == digest, f"Best checkpoint SHA binding differs: {name}")
        need(reload["sha256"] == digest and self.mapped(reload["path"]) == path, "Reloaded best checkpoint path/SHA differs")
        need(reload["strict_all_tensor_match"] is True, "Runner did not attest strict all-tensor checkpoint reload")
        shapes = expected_model_shapes(self.config)
        changed = reload["changed_local_parameters"]
        local_names = {key.removeprefix("item_local_attention.") for key in shapes if key.startswith("item_local_attention.")}
        need(isinstance(changed, list) and changed and len(changed) == len(set(changed)) and set(changed) <= local_names, "Missing/invalid attention parameter-change evidence")
        torch = self.load_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.tensor_state(state, shapes)
        count = sum(math.prod(shape) for shape in shapes.values())
        added = sum(math.prod(shape) for key, shape in shapes.items() if key.startswith("item_local_attention."))
        identity = self.document("model_identity.json")
        need(integer(identity["total_parameters"]) == count and integer(identity["added_parameters"]) == added, "Model parameter counts differ from independently checked checkpoint shapes")
        need(added == 263680, "Unexpected local-attention parameter count")
        need(bool(torch.count_nonzero(state["item_local_attention.attention.out_proj.weight"])) or bool(torch.count_nonzero(state["item_local_attention.attention.out_proj.bias"])), "Zero-initialized attention output still wholly zero")
        self.best_sha, self.model_shapes = digest, shapes
        self.best_state = state
        return {"checkpoint_sha256": digest, "parameter_count": count, "added_parameters": added,
                "state_keys": len(shapes), "all_tensors_finite_on_cpu": True, "local_attention_keys": sorted(local_names),
                "strict_model_reload_scope": "Runner attestation cross-bound by SHA; CPU audit validates exact pinned keys/shapes, not execution",
                "fit_binding_scope": "FIT receipt has no SHA field; curve-selected epoch/score + reload + TEST_STARTED + results + COMPLETE bind the best checkpoint indirectly"}

    def audit_terminal_state(self):
        path = self.run / "training_state/last.pt"
        if not path.is_file():
            self.checks.append({"check": "terminal_last_pt_contents", "status": "not_audited", "reason": "last.pt was not delivered; no full-resume acceptance claimed"})
            return {"delivered": False, "resume_acceptance": "not_audited"}
        self.record_file(path)
        torch = self.load_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        need(payload["format"] == "diffgrm_epoch_boundary_v1", "Unknown last.pt schema")
        progress = payload["progress"]
        need(progress["training_complete"] is True, "last.pt is not a terminal completed training state")
        for key, expected in (("next_epoch", self.curve["epochs"]), ("best_epoch", self.curve["best_epoch"]),
                              ("best_val_score", self.curve["best_val_score"]), ("eval_count", self.curve["eval_count"]),
                              ("no_improve_count", self.curve["no_improve_count"]),
                              ("cumulative_optimizer_updates", self.curve["cumulative_updates"]),
                              ("decoder_epochs_completed", self.curve["epochs"])):
            need(equal_number(progress[key], expected), f"Terminal last.pt progress/curve differs: {key}")
        need(progress["stop_reason"] == self.curve["stop_reason"], "Terminal stopping reason differs from recomputed protocol")
        need(progress["epoch_timing_records"] == self.epoch_timings, "Terminal epoch timing history differs")
        need(payload["best_checkpoint_sha256"] == self.best_sha and digest_bytes(payload["best_checkpoint_bytes"]) == self.best_sha, "Terminal embedded best-checkpoint SHA differs")
        need(self.mapped(payload["best_checkpoint"]) == self.local("training_state/best_model.bin"), "Terminal best-checkpoint path differs")
        self.tensor_state(payload["model"], self.model_shapes)
        config_values = {key: value for key, value in self.config.items() if key not in RUNTIME_KEYS}
        fingerprint = digest_bytes(json.dumps(config_values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        need(payload["configuration_fingerprint"] == fingerprint, "Terminal configuration fingerprint differs from runtime parsed config")
        need(0 < finite(progress["fit_elapsed_seconds"]) <= finite(self.document("FIT_COMPLETE.json")["fit_elapsed_seconds"]), "Terminal fit time differs from FIT_COMPLETE")
        optimizer, scheduler = payload["optimizer"], payload["scheduler"]
        need(isinstance(optimizer["state"], dict) and len(optimizer["param_groups"]) == 1, "Unexpected optimizer state layout")
        group = optimizer["param_groups"][0]
        need(len(group["params"]) == len(self.model_shapes) and len(set(group["params"])) == len(group["params"]), "Optimizer parameter coverage differs")
        need(set(optimizer["state"]) == set(group["params"]), "Optimizer moments missing for parameters")
        # The shape manifest preserves the pinned Module registration order;
        # AdamW receives model.parameters() in precisely that order.
        for parameter, (name, expected_shape) in zip(group["params"], self.model_shapes.items()):
            state = optimizer["state"][parameter]
            need(equal_number(state["step"].item() if torch.is_tensor(state["step"]) else state["step"], self.curve["cumulative_updates"]), f"Optimizer update count differs: parameter {parameter}")
            need(all(torch.is_tensor(state[key]) and bool(torch.isfinite(state[key]).all()) for key in ("exp_avg", "exp_avg_sq")), "Optimizer moment tensors are missing/nonfinite")
            need(state["exp_avg"].shape == state["exp_avg_sq"].shape, "Optimizer moment pair shapes differ")
            need(tuple(state["exp_avg"].shape) == expected_shape, f"Optimizer moments are attached to wrong parameter shape: {name}")
        need(integer(scheduler["last_epoch"]) == self.curve["cumulative_updates"], "Scheduler last_epoch differs from cumulative updates")
        need(integer(scheduler["_step_count"]) == self.curve["cumulative_updates"] + 1, "Scheduler step count differs")
        need(len(scheduler["_last_lr"]) == 1 and equal_number(scheduler["_last_lr"][0], self.training_rows[-1]["lr_end"]), "Scheduler learning rate differs from final CSV row")
        need(len(scheduler["base_lrs"]) == 1 and equal_number(scheduler["base_lrs"][0], self.config["lr"]), "Scheduler base LR differs from configured LR")
        need(equal_number(group["lr"], scheduler["_last_lr"][0]) and equal_number(group["weight_decay"], self.config["weight_decay"]), "Optimizer LR/weight decay differs")
        need(payload["scaler"] is None and self.config["mixed_precision"] == "bf16", "Unexpected BF16 scaler layout")
        rng = payload["rng"]
        need({"python", "numpy", "torch_cpu", "torch_cuda_all", "loader_counters", "loader_generators"} <= set(rng), "RNG recovery payload incomplete")
        need(torch.is_tensor(rng["torch_cpu"]) and isinstance(rng["torch_cuda_all"], list) and len(rng["torch_cuda_all"]) == 1, "RNG layout does not record one GPU")
        if self.curve["best_epoch"] == self.curve["epochs"]:
            need(all(torch.equal(value, self.best_state[name]) for name, value in payload["model"].items()), "Terminal model differs from best despite best epoch being final epoch")
        need(stamp(self.document("FIT_COMPLETE.json")["completed_utc"]).timestamp() >= finite(payload["created_unix_seconds"]), "Terminal state was written after FIT_COMPLETE")
        paths, offsets = progress["output_paths"], payload["diagnostic_offsets"]
        required_paths = {"position_metrics_csv", "order_metrics_csv", "training_metrics_csv", "evaluation_metrics_csv",
                          "joint_metrics_csv", "run_metadata_path", "saved_model_ckpt"}
        need(set(paths) == required_paths, "Terminal output-path manifest is incomplete or unknown")
        need(set(offsets) == {remote for name, remote in paths.items() if name.endswith("_csv")}, "Terminal diagnostic-offset manifest is incomplete or unknown")
        expected_paths = {"position_metrics_csv": self.details["position_csv"], "order_metrics_csv": self.details["order_csv"],
                          "training_metrics_csv": self.details["train_csv"], "evaluation_metrics_csv": self.details["eval_csv"],
                          "joint_metrics_csv": self.unique("diagnostics", "*_joint_metrics.csv"),
                          "run_metadata_path": self.details["metadata_path"], "saved_model_ckpt": self.local("training_state/best_model.bin")}
        for name, remote in paths.items():
            path = self.mapped(remote)
            need(path == expected_paths[name], f"Terminal output path points to a different artifact: {name}")
            if not name.endswith("_csv"):
                continue
            need(remote in offsets, f"Terminal diagnostic offset missing: {name}")
            offset = integer(offsets[remote])
            data = path.read_bytes()
            need(0 <= offset <= len(data), f"Terminal diagnostic offset exceeds delivered bytes: {name}")
            need(not offset or data[:offset].endswith(b"\n"), f"Terminal offset cuts a CSV row: {name}")
            prefix = read_csv(data[:offset].decode("utf-8-sig"), text=True)
            if name == "training_metrics_csv":
                need(offset == len(data) and prefix == self.training_rows, "Terminal training CSV offset/history differs")
            elif name == "evaluation_metrics_csv":
                need(all(row["split"] == "val" for row in prefix), "Test already present at FIT terminal boundary")
                need(len(prefix) == sum(len(values) for (split, _), values in self.groups.items() if split == "val"), "Terminal validation CSV prefix is incomplete")
            elif len(data) > offset:
                header = data.splitlines(keepends=True)[0]
                tail = read_csv((header + data[offset:]).decode("utf-8-sig"), text=True)
                need(tail and all(row.get("split") == "test" for row in tail), f"Non-test diagnostic data appended after terminal fit: {name}")
        return {"delivered": True, "training_complete": True, "progress_matches_curve": True,
                "optimizer_scheduler_and_diagnostic_offsets_checked": True,
                "resume_acceptance": "contents_only_not_full_resume_test", "last_pt_sha256": sha(self.run / "training_state/last.pt")}

    def audit_unchanged_files(self):
        for name, recorded in self.files.items():
            path = Path(name)
            need(path.is_file() and path.stat().st_size == recorded["size_bytes"] and sha(path) == recorded["sha256"], f"Audited file changed during read-only audit: {path}")
        return {"file_count": len(self.files), "all_initial_hashes_rechecked": True}

    def receipt(self):
        failed = [row["check"] for row in self.checks if row["status"] == "failed"]
        return {"schema": "stage07_item_local_delivery_audit_v1", "status": "PASS" if not failed else "FAIL",
                "artifact_audit_only": True, "goal_complete": False, "run_id": RUN_ID,
                "run_dir": str(self.run), "audited_utc": datetime.now(timezone.utc).isoformat(),
                "failed_checks": failed, "checks": self.checks, "audited_files": self.files,
                "limitations": [
                    "Historical original runtime commit, complete resolved config, initialization/RNG and original input-byte hash receipts were not recorded.",
                    "This is a single-seed SID-level historical comparison, not statistical significance, strict paired initialization, or item-level evaluation.",
                    "A remote input hash receipt attests byte checks at packaging; it is not a digital signature or local full-input restoration.",
                    "The CPU checkpoint audit does not execute model inference or a full optimizer/RNG resume; strict reload execution remains the runner's SHA-bound evidence.",
                    "Platform job terminal state, GPU release, actual allocation duration and eight-GPU-hour budget are outside this script's proof scope.",
                    "A delivery PASS alone never certifies overall Goal completion."
                ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Downloaded formal run directory, not its smoke/ child")
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON outside the run directory; never overwritten")
    parser.add_argument("--input-root", type=Path, help="Optional root containing all twelve relative input paths for local byte checks")
    parser.add_argument("--input-hash-receipt", type=Path, help="Independent packaging byte-hash receipt; default: RUN/input_hash_receipt.json")
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(run):
        parser.error("--output must be a new file outside --run-dir; delivered artifacts are read-only")
    audit = Audit(args)
    for name, function in (("mandatory_artifacts", audit.audit_inventory), ("trusted_offline_references", audit.audit_references),
                           ("formal_seed_and_source_identity", audit.audit_identity), ("fixed_recipe_and_runtime_config", audit.audit_configuration),
                           ("input_and_SID_identities", audit.audit_inputs), ("complete_training_and_best_validation", audit.audit_curves),
                           ("complete_finite_position_and_order_curves", audit.audit_diagnostic_curves),
                           ("one_final_test_and_recomputed_results", audit.audit_test_and_results), ("runtime_metadata_crosscheck", audit.audit_metadata),
                           ("best_checkpoint_cpu_contents_and_SHA", audit.audit_best_checkpoint), ("terminal_state_content_audit", audit.audit_terminal_state),
                           ("audit_snapshot_unchanged", audit.audit_unchanged_files)):
        audit.section(name, function)
    receipt = audit.receipt()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "run_id": RUN_ID, "failed_checks": receipt["failed_checks"],
                      "output": str(output), "goal_complete": False}, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
