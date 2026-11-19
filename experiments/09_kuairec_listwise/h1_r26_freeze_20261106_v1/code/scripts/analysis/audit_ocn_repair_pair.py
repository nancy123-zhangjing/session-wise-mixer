"""Read-only CPU audit of the completed, repaired Stage07 A/B pair.

python -B scripts/analysis/audit_ocn_repair_pair.py --goal-root <original-goal> \
    --root-commit 2fc97b62693628f6507ad2ca60472f6063a84250 --output <new.json>

This mode reads the original runs/<run_id> trees, including their profile/
children and all twelve workspace inputs. It does not support flattened bundles.
--reference-root defaults to GOAL/code and --engine-root to GOAL/engine. The
postprocessing script may live elsewhere. CPU PyTorch and PyYAML are required.
No model, Trainer, CUDA operation, network request, restore or truncation runs.
A PASS covers artifact consistency, never GPU release or overall Goal success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import random
import sys
from types import SimpleNamespace


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
HELPER = HERE.with_name("audit_item_local_delivery.py")
if not HELPER.is_file():
    # Permit deploying only this postprocessor outside the frozen code checkout.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--goal-root", type=Path)
    bootstrap.add_argument("--reference-root", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()
    reference = bootstrap_args.reference_root or (bootstrap_args.goal_root / "code" if bootstrap_args.goal_root else None)
    if reference is not None:
        HELPER = reference / "scripts/analysis/audit_item_local_delivery.py"
spec = importlib.util.spec_from_file_location("ocn_repair_reusable_audit", HELPER)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
need, sha, read_json = legacy.need, legacy.sha, legacy.read_json
finite, integer, equal_number = legacy.finite, legacy.integer, legacy.equal_number
normpath, stamp, read_csv = legacy.normpath, legacy.stamp, legacy.read_csv
normalized_metrics, git_bytes = legacy.normalized_metrics, legacy.git_bytes
CAMPAIGN = "stage07_ocn_repair_seed2024_20261020_r1"
EXECUTION_ROOT = "2fc97b62693628f6507ad2ca60472f6063a84250"
ENGINE = "8a48a300daa6621d6c705966ff73fa837ef8f12e"
RUNS = {arm: f"stage07_ocn_repair_{arm}_seed2024_20261020_r1" for arm in ("control", "item_local")}
METRICS = legacy.METRICS


def summary_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value):
    need(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "Missing/noncanonical SHA256")
    return value


def shapes_for(config, arm):
    shapes = legacy.expected_model_shapes(config)
    if arm == "control":
        shapes = {name: shape for name, shape in shapes.items() if not name.startswith("item_local_attention.")}
    need(len(shapes) == (84 if arm == "control" else 90), "Unexpected repaired model architecture")
    return shapes


def tensor_digest(value, torch):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def gradient_integrity(value, shapes, *, updates=None):
    for field in ("passed", "all_gradient_coverage", "all_observed_gradients_finite"):
        need(value.get(field) is True, f"Gradient integrity is not passed: {field}")
    need(integer(value["all_parameter_tensors"]) == len(shapes), "Gradient parameter count differs")
    calls = value["gradient_hook_calls"]
    need(set(calls) == set(shapes), "Gradient hooks do not cover the exact parameter names")
    need(all(integer(count) > 0 for count in calls.values()), "A parameter never received a gradient")
    if updates is not None:
        need(value.get("all_optimizer_steps_match") is True, "Optimizer step integrity failed")
        need(integer(value["optimizer_state_count"]) == len(shapes), "Adam state coverage differs")
        need(integer(value["cumulative_optimizer_updates"]) == updates, "Adam receipt step differs")


def rng_contents(value, torch):
    import numpy as np
    need(set(value) == {"python", "numpy", "torch_cpu", "torch_cuda_all", "loader_counters", "loader_generators"},
         "Checkpoint RNG components incomplete/unknown")
    # Private CPU generators validate actual recoverability without changing RNG.
    random.Random().setstate(value["python"])
    np.random.RandomState().set_state(value["numpy"])
    torch.Generator(device="cpu").set_state(value["torch_cpu"])
    need(isinstance(value["torch_cuda_all"], list) and len(value["torch_cuda_all"]) == 1, "Checkpoint RNG did not record one GPU")
    for state in [value["torch_cpu"], *value["torch_cuda_all"]]:
        need(torch.is_tensor(state) and state.device.type == "cpu" and state.dtype == torch.uint8
             and state.ndim == 1 and state.numel() > 0, "Checkpoint Torch RNG bytes malformed")
    need(isinstance(value["loader_counters"], dict) and isinstance(value["loader_generators"], dict), "Checkpoint loader RNG state missing")
    for state in value["loader_generators"].values():
        torch.Generator(device="cpu").set_state(state)


class RepairAudit(legacy.Audit):
    """Reuse only checks whose implementation has no historical run-id assumption."""

    def __init__(self, args, arm):
        self.arm, self.run_id = arm, RUNS[arm]
        run = args.goal_root / "runs" / self.run_id
        super().__init__(SimpleNamespace(run_dir=run, input_root=run / "workspace", input_hash_receipt=None))
        self.options = args
        self.reference = args.reference_root
        self.pack = self.reference / "experiments/03_diffgrm_tokenizer_comparison"
        self.profile = self.run / "profile"

    def mapped(self, remote):
        value, anchor = normpath(remote), "/" + self.run_id + "/"
        need(value.count(anchor) == 1, f"Artifact path does not belong to {self.run_id}: {remote}")
        relative = value.split(anchor, 1)[1]
        need(not relative.startswith(("smoke/", "profile/")) and ".." not in PurePosixPath(relative).parts,
             "A formal artifact points to a diagnostic child/unsafe path")
        return self.local(relative)

    def audit_inventory(self):
        # Preserve old failure evidence. A completed later attempt must follow it.
        incomplete = self.run / "INCOMPLETE.json"
        if incomplete.exists():
            failure = self.document("INCOMPLETE.json")
            need(stamp(failure["utc"]) < stamp(self.document("COMPLETE.json")["completed_utc"]),
                 "INCOMPLETE is not older than completion")
            need(any(stamp(row["started"]["utc"]) > stamp(failure["utc"]) for row in self.execution_attempts()),
                 "No later execution reconciles the preserved failure")
        required = ("identity.json", "study.json", "baseline_resolved_config.json", "resolved_config.json",
                    "parsed_config.json", "input_identities.json", "protocol_audit.json", "code_identity.json",
                    "model_identity.json", "environment.json", "STARTED.json", "FIT_COMPLETE.json",
                    "checkpoint_reload.json", "runtime_phase.json", "TEST_STARTED.json", "TEST_RESULT.json",
                    "results.json", "COMPLETE.json", "evaluation_timings.json", "training_state/epoch_timings.json",
                    "initial_state_identity.json", "training_integrity.json")
        for name in required:
            self.document(name)
        self.local("training_state/last.pt")
        self.local("training_state/best_model.bin")
        need(not (self.run / "smoke_results.json").exists() and not (self.run / "profile_results.json").exists(),
             "Diagnostic results are present in formal root")
        self.details["metadata_path"] = self.unique("run_metadata", "*_run_metadata.json")
        self.data["metadata"] = read_json(self.details["metadata_path"])
        for key, suffix in (("train_csv", "training_metrics"), ("eval_csv", "evaluation_metrics"),
                            ("position_csv", "position_metrics"), ("order_csv", "order_metrics")):
            self.details[key] = self.unique("diagnostics", f"*_{suffix}.csv")
        return {"mandatory_json_files": len(required), "last_pt_required": True,
                "older_failure_preserved_and_reconciled": incomplete.exists()}

    def audit_references(self):
        recipe_path = self.record_file(self.pack / "reproducibility/recipes/pse_seed2024.json")
        self.recipe = read_json(recipe_path)
        need(self.recipe["run_id"] == legacy.BASELINE_RUN_ID and self.recipe["seed"] == 2024, "Historical recipe identity changed")
        versions = read_json(self.record_file(self.pack / "reproducibility/code_versions.json"))
        engines = [row for row in versions["repositories"] if row["name"] == "DiffGRM"]
        need(len(engines) == 1 and engines[0]["commit"] == legacy.ENGINE_BASE, "Historical reconstruction source differs")
        self.base_source = engines[0]
        snapshot = self.record_file(self.reference / self.base_source["snapshot"])
        need(sha(snapshot) == self.base_source["sha256"], "Historical snapshot SHA differs")
        self.lock_path = self.record_file(self.pack / "reproducibility/assets.lock.json")
        self.lock = read_json(self.lock_path)["artifacts"]
        self.historical = read_json(self.record_file(self.pack / self.recipe["metadata"]))
        rows = read_csv(self.record_file(self.pack / "results.csv"))
        selected = [row for row in rows if row.get("id") == "pse_seed2024"]
        need(len(selected) == 1, "Historical PSE reference missing/duplicated")
        self.baseline = normalized_metrics({key: selected[0][key] for key in METRICS})
        need(all(equal_number(self.baseline[key], self.historical["test_results"][key]) for key in METRICS),
             "Historical reference sources disagree")
        sys.path.insert(0, str(self.reference / "scripts/repro"))
        import yaml
        import project_replay as replay
        self.replay = replay
        self.formal, self.formal_sources = replay.load_formal_config(self.recipe, self.pack, snapshot, yaml)
        self.config_audit = replay.audit_formal_config(self.formal, self.recipe, self.historical)
        self.engine_hashes = {name: hashlib.sha256(git_bytes(self.options.engine_root, ENGINE, name)).hexdigest()
                              for name in legacy.ENGINE_FILES}
        study_name = f"configs/07_item_local_attention/ocn_repair_{self.arm}_seed2024.json"
        self.study_bytes = git_bytes(self.reference, self.options.root_commit, study_name)
        self.expected_study = legacy.json_text(self.study_bytes.decode("utf-8-sig"))
        self.runner_hash = hashlib.sha256(git_bytes(self.reference, self.options.root_commit,
                                                   "scripts/train/run_item_local_attention.py")).hexdigest()
        # Imported reusable checks must also be the reviewed execution-version bytes.
        helper = HELPER
        self.record_file(helper)
        need(sha(helper) == hashlib.sha256(git_bytes(self.reference, self.options.root_commit,
                                                    "scripts/analysis/audit_item_local_delivery.py")).hexdigest(),
             "Reusable audit helper differs from the pinned reviewed version")
        campaign = legacy.json_text(git_bytes(self.reference, self.options.root_commit,
                    "configs/07_item_local_attention/ocn_repair_campaign_seed2024.json").decode("utf-8-sig"))
        need(campaign["campaign_id"] == CAMPAIGN and campaign["engine_commit"] == ENGINE,
             "Pinned campaign/source identity differs")
        need(campaign["budget"]["cumulative_allocated_gpu_hours_limit"] == 28
             and campaign["protocol"]["allow_legacy_epoch84_resume"] is False
             and campaign["protocol"]["from_scratch"] is True, "Pinned repaired campaign controls differ")
        return {"root_commit": self.options.root_commit, "engine_commit": ENGINE,
                "historical_full_resolved_config_available": False, "old03_role": "background_only"}

    def source_identity(self, code, context):
        need(code["project"]["head"] == self.options.root_commit and code["engine"]["head"] == ENGINE,
             f"{context}: actual execution commits differ")
        need(code["files"] == self.engine_hashes and code["runner_sha256"] == self.runner_hash,
             f"{context}: source bytes differ from pinned commits")
        need(code["engine"].get("status") == "", f"{context}: engine checkout was dirty")
        need(code.get("historical_runtime_commit") is None, "Historical original-source gap was silently filled")

    def audit_identity(self):
        identity, study = self.document("identity.json"), self.document("study.json")
        need(identity["run_id"] == self.run_id and study == self.expected_study, "Formal run/study identity differs")
        need(study["repair_campaign"] == identity["repair_campaign"] == CAMPAIGN
             and study["variant"] == identity["variant"] == self.arm and study["seed"] == 2024,
             "Wrong repaired campaign, arm or seed")
        need(study["max_gpu_hours"] == 28 and study["item_local_attention"] ==
             {"enabled": self.arm == "item_local", "num_heads": 4, "dropout": 0.0}, "Wrong repaired study controls")
        need(identity["repaired_engine_commit"] == ENGINE and identity["study_sha256"] == hashlib.sha256(self.study_bytes).hexdigest(),
             "Study/engine SHA identity differs")
        need(identity["baseline_run_id"] == legacy.BASELINE_RUN_ID
             and identity["sid_sha256"] == self.lock[self.recipe["initial_sid"]]["sha256"], "SID/reference identity differs")
        complete, result = self.document("COMPLETE.json"), self.document("results.json")
        need(complete["mode"] == "formal" and complete["status"] == result["status"] == "formal_completed", "Formal completion is missing")
        for value in (complete, result):
            need(value["repair_campaign"] == CAMPAIGN and value["variant"] == self.arm, "Result campaign/arm differs")
        need(complete["result_file"] == "results.json" and complete["result_sha256"] == sha(self.local("results.json")),
             "COMPLETE does not bind actual formal result bytes")
        self.source_identity(self.document("code_identity.json"), "First preparation")
        executions = self.execution_attempts()
        need(executions[0]["started"]["resume"] is False, "Repaired formal run did not start from scratch")
        need(self.document("STARTED.json") == executions[0]["started"], "Root STARTED is not first actual execution")
        for execution in executions:
            prefix = execution["prefix"]
            self.source_identity(self.document(prefix + "code_identity.json"), execution["name"])
            need(self.document(prefix + "identity.json") == identity and self.document(prefix + "study.json") == study,
                 "Execution identity/study changed")
            environment = self.document(prefix + "environment.json")
            need(integer(environment["visible_gpus"]) == 1 and "H100" in environment["gpu"].upper(), "Execution did not record one H100")
            resume = execution["started"]["resume"]
            need(isinstance(resume, bool) or normpath(resume) == normpath(self.document("input_identities.json")["runtime_workspace"]).removesuffix("/workspace") + "/training_state/last.pt",
                 "Execution resumed an external, historical or profile state")
            need(stamp(execution["started"]["utc"]) <= stamp(complete["completed_utc"]), "Execution starts after completion")
        first = executions[0]["prefix"]
        for name in ("parsed_config.json", "environment.json", "model_identity.json", "initial_state_identity.json"):
            need(self.document(name) == self.document(first + name), f"Root {name} differs from first execution")
        protocol = self.document("protocol_audit.json")
        need(protocol["recipe_sha256"] == sha(self.pack / "reproducibility/recipes/pse_seed2024.json")
             and protocol["sources"] == self.formal_sources and protocol["baseline_code_snapshot"] == self.base_source
             and protocol["configuration"] == self.config_audit, "Reconstructed recipe provenance differs")
        return {"run_id": self.run_id, "seed": 2024, "variant": self.arm,
                "execution_attempts_checked": [row["name"] for row in executions], "fresh_first_execution": True}

    def audit_configuration(self):
        baseline = self.document("baseline_resolved_config.json")
        need(baseline == self.formal, "Baseline recipe reconstruction differs")
        resolved, parsed = self.document("resolved_config.json"), self.document("parsed_config.json")
        allowed = set(legacy.PATH_KEYS) | set(legacy.NEW_KEYS) | {"run_id", "joint_crosspse_enabled"}
        differences = {key: {"baseline": baseline.get(key), "actual": resolved.get(key)} for key in set(baseline) | set(resolved)
                       if key not in allowed and baseline.get(key) != resolved.get(key)}
        need(not differences, f"Unapproved formal recipe changes: {differences}")
        need(resolved["run_id"] == resolved["run_local_time"] == self.run_id, "Resolved formal run identity differs")
        need(resolved["item_local_attention"] == self.expected_study["item_local_attention"], "Attention control differs")
        need(resolved["rand_seed"] == 2024 and resolved["evaluation_level"] == "sid", "Wrong seed/evaluation level")
        need(resolved.get("max_train_batches_per_epoch") in (None, 0) and resolved.get("max_eval_batches") in (None, 0), "Truncated formal batches")
        need(not resolved.get("joint_crosspse_enabled") and not resolved.get("mask_schedule", {}).get("enabled"), "Unexpected joint/mask schedule")
        need(resolved["share_decoder_output_embedding"] is True and resolved["norm_type"] == "layernorm", "Unsupported model layout")
        need(not any(resolved.get(key) for key in ("checkpoint", "checkpoint_path", "joint_recommender_checkpoint")), "Pretrained recommender initialization")
        need(all(parsed.get(key) == value for key, value in resolved.items() if key not in legacy.ATTEMPT_RUNTIME_KEYS), "Parsed config differs from resolved config")
        inputs = self.document("input_identities.json")
        remote_root = normpath(inputs["runtime_workspace"])
        need(remote_root.endswith("/" + self.run_id + "/workspace"), "Workspace is not formal-run scoped")
        need(normpath(resolved["cache_dir"]) == remote_root + "/data/cache", "Cache workspace binding differs")
        paths = {"log_dir": "logs", "tensorboard_log_dir": "tensorboard", "ckpt_dir": "checkpoints",
                 "run_metadata_dir": "run_metadata", "position_loss_log_dir": "positions",
                 "diffusion_diagnostics_log_dir": "diagnostics", "training_state_dir": "training_state"}
        for key, tail in paths.items():
            need(normpath(resolved[key]) == remote_root.removesuffix("/workspace") + "/" + tail, f"Output path differs: {key}")
        if resolved.get("sent_emb_path"):
            need(normpath(resolved["sent_emb_path"]).startswith(remote_root + "/"), "Embedding input path escapes workspace")
        need(resolved["training_state_identity"]["engine_files"] == self.engine_hashes, "State source binding differs")
        protocol = self.document("protocol_audit.json")
        need(protocol["mode_overrides"] == {key: {"historical": baseline.get(key), "resolved": value}
                                           for key, value in resolved.items() if value != baseline.get(key)}, "Root override receipt differs")
        stable = lambda value: {key: item for key, item in value.items() if key not in legacy.ATTEMPT_RUNTIME_KEYS}
        for execution in self.execution_attempts():
            prefix = execution["prefix"]
            actual, actual_parsed = self.document(prefix + "resolved_config.json"), self.document(prefix + "parsed_config.json")
            need(self.document(prefix + "baseline_resolved_config.json") == baseline and stable(actual) == stable(resolved)
                 and stable(actual_parsed) == stable(parsed), "Recipe changed between formal attempts")
            need(all(actual_parsed.get(key) == value for key, value in actual.items()) and actual_parsed.get("use_ddp", False) is False,
                 "Actual parsed config differs or enabled DDP")
            need(actual["resume_training_state"] == execution["started"]["resume"]
                 and actual["training_deadline_unix_seconds"] == execution["started"]["deadline_unix"], "STARTED/config binding differs")
            need(self.document(prefix + "input_identities.json") == inputs, "Input identities changed across attempts")
            expected_protocol = dict(protocol)
            expected_protocol["mode_overrides"] = {key: {"historical": baseline.get(key), "resolved": value}
                                                   for key, value in actual.items() if value != baseline.get(key)}
            need(self.document(prefix + "protocol_audit.json") == expected_protocol, "Attempt protocol reconstruction differs")
        self.config = self.document(self.execution_attempts()[-1]["prefix"] + "parsed_config.json")
        return {"complete_config_checked": True, "historical_full_config_gap_retained": True,
                "formal_epochs": integer(self.config["epochs"]), "attention_enabled": self.arm == "item_local"}

    def audit_test_and_results(self):
        timing = self.document("evaluation_timings.json")
        need(isinstance(timing, list) and len(timing) >= len(self.val_epochs) + 1, "Missing evaluation timings")
        tests = [row for row in timing if row["split"] == "test"]
        need(len(tests) == 1 and timing[-1] == tests[0] and tests[0]["epoch"] is None, "Final test was missing, repeated or premature")
        vals = timing[:-1]
        need(all(row["split"] == "val" and row["epoch"] in self.val_epochs for row in vals)
             and [row["epoch"] for row in vals] == sorted(row["epoch"] for row in vals)
             and set(row["epoch"] for row in vals) == set(self.val_epochs), "Validation timing schedule differs")
        for epoch in self.val_epochs:
            calls = [row for row in vals if row["epoch"] == epoch]
            need(len({row["attempt"] for row in calls}) == len(calls), "Validation repeated within one attempt")
        names = {row["name"] for row in self.execution_attempts()}
        need(all(finite(row["seconds"]) > 0 and row["attempt"] in names for row in timing), "Invalid timing duration/execution binding")
        fit, result = self.document("FIT_COMPLETE.json"), self.document("results.json")
        need(fit["fit_validation_timing_attempt"] in names, "FIT validation references unknown attempt")
        need(equal_number(fit["fit_validation_seconds"], sum(finite(row["seconds"]) for row in vals
             if row["attempt"] == fit["fit_validation_timing_attempt"])), "FIT validation time differs")
        times = [stamp(self.document("STARTED.json")["utc"]), stamp(fit["completed_utc"]),
                 stamp(self.document("TEST_STARTED.json")["utc"]), stamp(self.document("TEST_RESULT.json")["completed_utc"]),
                 stamp(self.document("COMPLETE.json")["completed_utc"])]
        need(times == sorted(times), "STARTED/FIT/TEST/COMPLETE chronology differs")
        for name, value in (("TEST_RESULT", self.document("TEST_RESULT.json")), ("results", result), ("metadata", self.data["metadata"])):
            need(normalized_metrics(value["test_results"], selection=True) == self.test_metrics, f"{name} differs from once-only test CSV")
        for key in ("best_epoch", "best_val_score", "fit_elapsed_seconds", "fit_validation_seconds", "fit_validation_timing_attempt", "completed_utc"):
            need(result[key] == fit[key], f"Result changed FIT field: {key}")
        need(result["evaluation_level"] == "sid" and result["baseline_comparison_role"] == "historical_background_only", "Comparison scope differs")
        # Validate existing historical numbers without treating them as the A/B comparator.
        background = {}
        for key in METRICS:
            old, new = self.baseline[key], self.test_metrics[key]
            need(old != 0, f"Undefined recorded historical ratio: {key}")
            background[key] = {"baseline": old, "new": new, "absolute_delta": new - old, "relative_change": new / old - 1}
            need(all(equal_number(result["baseline_comparison"][key][field], value) for field, value in background[key].items()),
                 f"Incorrect historical background arithmetic: {key}")
        signal = background["ndcg@10"]["relative_change"] >= .03 and background["hit_rate@10"]["absolute_delta"] >= 0
        need(result["historical_background_positive_signal"] is signal and "exploratory_positive_signal" not in result,
             "Historical background signal is mislabeled or incorrect")
        phase = self.document("runtime_phase.json")
        need(phase["phase"] == "final_test" and phase["training_complete"] is True and finite(phase["evaluation_reserve_seconds"]) == 0
             and times[1] <= stamp(phase["utc"]) <= times[2], "Final-test phase/guard binding differs")
        return {"test_timing_calls": 1, "test_CSV_groups": 1, "chronology_checked": True, "old03_role": "background_only"}

    def audit_metadata(self):
        metadata, config = self.data["metadata"], self.config
        need(metadata["run_id"] == self.run_id and integer(metadata["seed"]) == 2024 and metadata["model"] == "DIFF_GRM", "Metadata identity differs")
        keys = ("sid_quantizer", "n_digit", "codebook_size", "masking_strategy", "guided_select", "guided_refresh_each_step",
                "diffusion_loss_reduction", "train_batch_size", "gradient_accumulation_steps", "lr", "mixed_precision",
                "max_grad_norm", "numerical_guard", "numerical_guard_parameter_interval", "eval_start_epoch", "eval_interval",
                "patience", "early_stop_after_warmup", "evaluation_level")
        need(all(metadata.get(key) == config.get(key) for key in keys), "Metadata config controls differ")
        for key in ("best_epoch", "best_val_score", "updates_per_epoch"):
            need(equal_number(metadata[key], self.curve[key]), f"Metadata curve mismatch: {key}")
        need(integer(metadata["total_optimizer_steps"]) == integer(config["epochs"]) * self.curve["updates_per_epoch"]
             and integer(metadata["warmup_steps"]) == integer(config["warmup_steps"]), "Metadata scheduler differs")
        need(normpath(metadata["semantic_ids_path"]) == normpath(self.document("input_identities.json")["sid_path"])
             and metadata["semantic_ids_sha256"] == self.document("identity.json")["sid_sha256"], "Metadata SID consumption differs")
        for field, local in (("training_metrics_csv", "train_csv"), ("evaluation_metrics_csv", "eval_csv"),
                             ("position_metrics_csv", "position_csv"), ("order_metrics_csv", "order_csv")):
            need(self.mapped(metadata[field]) == self.details[local], "Metadata output binding differs")
        need(self.mapped(metadata["checkpoint"]) == self.local("training_state/best_model.bin")
             and not metadata.get("joint_crosspse_enabled", False), "Metadata model/checkpoint differs")
        return {"config_fields_checked": list(keys), "SID_bytes_and_consumption_bound": True}

    def audit_best_checkpoint(self):
        path, reload = self.local("training_state/best_model.bin"), self.document("checkpoint_reload.json")
        digest_value = sha(path)
        for name in ("TEST_STARTED.json", "results.json", "COMPLETE.json"):
            need(self.document(name)["checkpoint_sha256"] == digest_value, f"Checkpoint SHA differs: {name}")
        need(reload["sha256"] == digest_value and self.mapped(reload["path"]) == path
             and reload["strict_all_tensor_match"] is True, "Strict reload is not bound to actual checkpoint bytes")
        shapes = shapes_for(self.config, self.arm)
        torch = self.load_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.tensor_state(state, shapes)
        count = sum(math.prod(shape) for shape in shapes.values())
        added = sum(math.prod(shape) for name, shape in shapes.items() if name.startswith("item_local_attention."))
        model = self.document("model_identity.json")
        need(integer(model["total_parameters"]) == count and integer(model["added_parameters"]) == added
             and model["item_local_attention_enabled"] is (self.arm == "item_local"), "Model parameter manifest differs")
        changed_local = reload["changed_local_parameters"]
        if self.arm == "control":
            need(added == 0 and changed_local == [], "Control contains local-attention changes")
        else:
            local_names = {name.removeprefix("item_local_attention.") for name in shapes if name.startswith("item_local_attention.")}
            need(added == 263680 and isinstance(changed_local, list) and changed_local
                 and len(set(changed_local)) == len(changed_local) and set(changed_local) <= local_names, "Invalid local-attention change evidence")
            need(bool(torch.count_nonzero(state["item_local_attention.attention.out_proj.weight"]))
                 or bool(torch.count_nonzero(state["item_local_attention.attention.out_proj.bias"])), "Zero-initialized local output never changed")
        need(integer(reload["changed_parameter_tensors"]) == len(shapes), "Runner did not prove all parameter tensors changed")
        self.best_sha, self.best_state, self.model_shapes = digest_value, state, shapes
        return {"checkpoint_sha256": digest_value, "parameter_tensors": len(shapes), "parameter_count": count,
                "added_parameters": added, "CPU_keys_shapes_finite_checked": True,
                "reload_scope": "SHA-bound frozen runner attestation; CPU does not execute model reload"}

    def initial_identity(self, value, shapes, *, fresh):
        need(value["repair_campaign"] == CAMPAIGN and value["variant"] == self.arm, "Initial identity campaign/arm differs")
        need(type(value["resume_requested"]) is bool and (not fresh or value["resume_requested"] is False), "Fresh initial RNG was captured for a resume")
        base, rng = value["base_tensors_excluding_item_local_attention"], value["initial_rng"]
        base_shapes = {name: shape for name, shape in shapes.items() if not name.startswith("item_local_attention.")}
        need(set(base) == set(base_shapes) and len(base) == 84, "Initial base tensor manifest incomplete")
        for name, row in base.items():
            need(set(row) == {"dtype", "shape", "sha256"} and row["dtype"] == "torch.float32"
                 and row["shape"] == list(base_shapes[name]), f"Initial tensor schema differs: {name}")
            digest(row["sha256"])
        need(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda_all", "loader_counters", "loader_generators"}, "Initial RNG components incomplete/unknown")
        need(isinstance(rng["python"], list) and len(rng["python"]) == 3
             and isinstance(rng["numpy"], list) and len(rng["numpy"]) == 5, "Initial Python/NumPy RNG missing")
        need(rng["numpy"][1]["shape"] == [624] and rng["numpy"][1]["dtype"] == "uint32", "Initial NumPy RNG state layout differs")
        digest(rng["numpy"][1]["sha256"])
        need(isinstance(rng["torch_cuda_all"], list) and len(rng["torch_cuda_all"]) == 1, "Initial RNG did not record exactly one CUDA device")
        for row in [rng["torch_cpu"], *rng["torch_cuda_all"]]:
            need(row["dtype"] == "torch.uint8" and len(row["shape"]) == 1 and integer(row["shape"][0]) > 0, "Torch RNG summary malformed")
            digest(row["sha256"])
        need(isinstance(rng["loader_counters"], dict) and isinstance(rng["loader_generators"], dict), "Loader RNG summary missing")
        need(digest(value["initial_base_state_sha256"]) == summary_sha(base)
             and digest(value["initial_rng_sha256"]) == summary_sha(rng), "Initial aggregate hashes differ from complete summaries")
        return value

    def audit_training_integrity(self):
        shapes, result = self.model_shapes, self.document("results.json")
        integrity = self.document("training_integrity.json")
        gradient_integrity(integrity, shapes, updates=self.curve["cumulative_updates"])
        need(result["training_integrity"] == integrity and result["all_gradient_coverage"] is True
             and integer(result["optimizer_state_count"]) == integer(result["all_parameter_tensors"]) == len(shapes)
             and integer(result["actual_changed_parameter_tensors"]) == len(shapes), "Result integrity/counts differ")
        initial = self.initial_identity(self.document("initial_state_identity.json"), shapes, fresh=True)
        for execution in self.execution_attempts():
            value = self.initial_identity(self.document(execution["prefix"] + "initial_state_identity.json"), shapes,
                                          fresh=execution["started"]["resume"] is False)
            need(value["resume_requested"] is bool(execution["started"]["resume"]), "Initial identity/STARTED resume binding differs")
            need(value["base_tensors_excluding_item_local_attention"] == initial["base_tensors_excluding_item_local_attention"],
                 "Freshly constructed model initializer changed between attempts")
        for field in ("initial_base_state_sha256", "initial_rng_sha256"):
            need(result[field] == initial[field], f"Formal results differ from fresh initial identity: {field}")
        base = initial["base_tensors_excluding_item_local_attention"]
        unchanged = [name for name, row in base.items() if tensor_digest(self.best_state[name], self.load_torch()) == row["sha256"]]
        need(not unchanged, f"Base parameter tensors did not actually change: {unchanged}")
        self.initial = initial
        return {"exact_gradient_names": len(shapes), "all_Adam_states_independently_checked_in_terminal_section": True,
                "base_parameter_tensors_independently_changed": len(base),
                "all_parameter_change_attestation": len(shapes),
                "local_change_scope": "Six local initial tensor hashes were not stored; all-90 change is the SHA-bound runner attestation, with zero-output change independently checked" if self.arm == "item_local" else "All 84 initial hashes independently compared"}

    def audit_terminal_state(self):
        self.local("training_state/last.pt")  # Required, unlike the historical audit's optional last.pt.
        result = super().audit_terminal_state()
        payload = self.load_torch().load(self.run / "training_state/last.pt", map_location="cpu", weights_only=False)
        rng_contents(payload["rng"], self.load_torch())
        result["private_CPU_RNG_state_validation"] = True
        return result

    def audit_profile(self):
        def document(name):
            return self.document("profile/" + name)
        def path(name):
            return self.local("profile/" + name)
        result, complete = document("profile_results.json"), document("COMPLETE.json")
        identity, started = document("identity.json"), document("STARTED.json")
        need(identity["run_id"] == self.run_id + "_profile" and identity["repair_campaign"] == CAMPAIGN
             and identity["variant"] == self.arm and identity["study_sha256"] == hashlib.sha256(self.study_bytes).hexdigest(), "Profile identity differs")
        need(started["mode"] == "profile" and started["resume"] is False, "Profile was not an independent fresh run")
        need(not (self.profile / "INCOMPLETE.json").exists(), "Profile retains an unreconciled failure")
        self.source_identity(document("code_identity.json"), "Profile")
        executions = sorted((self.profile / "attempts").glob("*/STARTED.json"))
        need(len(executions) == 1, "Profile must have exactly one actual execution attempt")
        prefix = executions[0].parent.relative_to(self.profile).as_posix() + "/"
        need(document(prefix + "STARTED.json") == started and document(prefix + "identity.json") == identity,
             "Profile STARTED is not bound to its actual execution attempt")
        self.source_identity(document(prefix + "code_identity.json"), "Profile actual execution")
        environment = document(prefix + "environment.json")
        need(integer(environment["visible_gpus"]) == 1 and "H100" in environment["gpu"].upper(), "Profile did not record one H100")
        need(document("study.json") == self.expected_study, "Profile study differs")
        need(complete["mode"] == "profile" and complete["status"] == result["status"] == "profile_completed"
             and complete["repair_campaign"] == result["repair_campaign"] == CAMPAIGN
             and complete["variant"] == result["variant"] == self.arm
             and complete["result_file"] == "profile_results.json" and complete["result_sha256"] == sha(path("profile_results.json")), "Profile completion/result SHA binding differs")
        need(result["test_data_evaluated"] is False and not (self.profile / "TEST_STARTED.json").exists()
             and not (self.profile / "TEST_RESULT.json").exists(), "Profile evaluated final test")
        config, resolved = document("parsed_config.json"), document("resolved_config.json")
        need(all(config.get(key) == value for key, value in resolved.items()), "Profile parsed config differs")
        profile_overrides = {"epochs": 1, "eval_start_epoch": 1, "eval_interval": 1, "patience": 1}
        skip = set(legacy.PATH_KEYS) | set(legacy.NEW_KEYS) | set(legacy.RUNTIME_KEYS) | {"run_id"} | set(profile_overrides)
        need({key: value for key, value in config.items() if key not in skip}
             == {key: value for key, value in self.config.items() if key not in skip}, "Profile and formal recipe differ")
        need(all(config[key] == value for key, value in profile_overrides.items())
             and config["max_train_batches_per_epoch"] in (None, 0) and config["max_eval_batches"] in (None, 0), "Profile is truncated or has different overrides")
        need(config["item_local_attention"] == self.config["item_local_attention"]
             and config["training_state_identity"] == self.config["training_state_identity"], "Profile model/input/source identity differs")
        shapes = shapes_for(config, self.arm)
        integrity = document("training_integrity.json")
        gradient_integrity(integrity, shapes, updates=104)
        need(result["training_integrity"] == integrity and result["all_gradient_coverage"] is True
             and integer(result["optimizer_state_count"]) == integer(result["all_parameter_tensors"]) == len(shapes)
             and integer(result["actual_changed_parameter_tensors"]) == len(shapes), "Profile gradient/Adam/change coverage differs")
        initial = self.initial_identity(document("initial_state_identity.json"), shapes, fresh=True)
        need(document(prefix + "initial_state_identity.json") == initial and document(prefix + "parsed_config.json") == config,
             "Profile initialization/config differs from execution attempt")
        for field in ("initial_base_state_sha256", "initial_rng_sha256"):
            need(result[field] == initial[field] == self.initial[field], f"Profile/formal initial identity differs: {field}")
        train = read_csv(self.unique("profile/diagnostics", "*_training_metrics.csv"))
        loops = document("training_state/epoch_timings.json")
        need(len(train) == len(loops) == 1 and integer(train[0]["epoch"]) == integer(loops[0]["epoch"]) == 1
             and integer(loops[0]["batches"]) == result["formal_train_batches_per_epoch"] == 1652
             and integer(train[0]["optimizer_updates"]) == integer(train[0]["cumulative_optimizer_updates"]) == 104
             and finite(train[0]["train_loss"]) >= 0 and equal_number(loops[0]["seconds"], result["training_loop_seconds"]), "Profile training was not a full epoch")
        need(equal_number(train[0]["lr_start"], 0) and equal_number(train[0]["lr_end"], finite(config["lr"]) * 104 / integer(config["warmup_steps"])),
             "Profile learning-rate warmup differs from formal")
        need(result["formal_eval_batches"] == 2796 and all(finite(result[key]) > 0 for key in
             ("training_loop_seconds", "full_val_seconds", "beam256_val_2_batches_seconds")), "Profile measured counts/durations invalid")
        evaluations = read_csv(self.unique("profile/diagnostics", "*_evaluation_metrics.csv"))
        groups = {}
        for row in evaluations:
            key = (row["split"], integer(row["epoch"]))
            need(key in {("val", 1), ("benchmark", 1)}, "Profile contains unexpected/test evaluation")
            values = groups.setdefault(key, {})
            need(row["metric"] not in values, "Profile duplicate metric")
            values[row["metric"]] = finite(row["value"])
        need(set(groups) == {("val", 1), ("benchmark", 1)}
             and normalized_metrics(groups[("val", 1)], selection=True) == normalized_metrics(result["validation_results"], selection=True), "Profile full-val/benchmark CSV incomplete")
        normalized_metrics(groups[("benchmark", 1)], selection=True)
        timings = document("evaluation_timings.json")
        need([(row["split"], row["epoch"]) for row in timings] == [("val", 1), ("benchmark", 1)]
             and all(finite(row["seconds"]) > 0 for row in timings)
             and equal_number(timings[0]["seconds"], result["full_val_seconds"]), "Profile evaluation calls differ")
        positions = read_csv(self.unique("profile/diagnostics", "*_position_metrics.csv"))
        for split, count in (("val", 22363), ("benchmark", 16)):
            rows = [row for row in positions if row["split"] == split]
            need(len(rows) == 4 and {integer(row["position"]) for row in rows} == set(range(4))
                 and all(integer(row["supervised_tokens"]) == integer(row["available_views"]) == count for row in rows),
                 f"Profile {split} does not prove expected target count")
        payload = self.load_torch().load(path("training_state/last.pt"), map_location="cpu", weights_only=False)
        need(payload["format"] == "diffgrm_epoch_boundary_v1" and payload["progress"]["training_complete"] is True
             and integer(payload["progress"]["next_epoch"]) == 1 and integer(payload["progress"]["cumulative_optimizer_updates"]) == 104,
             "Profile terminal training state invalid")
        progress = payload["progress"]
        val_score = normalized_metrics(result["validation_results"], selection=True)["weighted_score"]
        need(integer(progress["best_epoch"]) == integer(result["best_epoch"]) == 1 and integer(progress["eval_count"]) == 1
             and equal_number(progress["best_val_score"], val_score) and equal_number(result["best_val_score"], val_score),
             "Profile unique validation-selected epoch/score differs")
        self.tensor_state(payload["model"], shapes)
        self.check_optimizer(payload, shapes, 104)
        rng_contents(payload["rng"], self.load_torch())
        fingerprint = summary_sha({key: value for key, value in config.items() if key not in legacy.RUNTIME_KEYS})
        need(payload["configuration_fingerprint"] == fingerprint, "Profile terminal config fingerprint differs")
        best = path("training_state/best_model.bin")
        best_sha = sha(best)
        need(payload["best_checkpoint_sha256"] == hashlib.sha256(payload["best_checkpoint_bytes"]).hexdigest()
             == best_sha == result["checkpoint_sha256"] == complete["checkpoint_sha256"], "Profile best bytes/SHA binding differs")
        best_state = self.load_torch().load(best, map_location="cpu", weights_only=True)
        self.tensor_state(best_state, shapes)
        need(all(self.load_torch().equal(value, best_state[name]) for name, value in payload["model"].items()),
             "Profile terminal epoch1 model differs from its sole validation-selected checkpoint")
        need(payload["scaler"] is None and equal_number(payload["scheduler"]["_last_lr"][0], train[0]["lr_end"])
             and equal_number(payload["optimizer"]["param_groups"][0]["lr"], train[0]["lr_end"]), "Profile BF16 optimizer/scheduler state differs")
        reload = document("checkpoint_reload.json")
        need(reload["sha256"] == best_sha and reload["strict_all_tensor_match"] is True
             and reload["changed_parameter_tensors"] == len(shapes), "Profile strict reload/change evidence incomplete")
        recovery = result["recovery_check"]
        probe = document("recovery_probe/recovery_check.json")
        need({key: value for key, value in recovery.items() if key != "wall_seconds"} == probe, "Recovery probe receipt differs from profile result")
        need(recovery["status"] == "passed" and recovery["strict_model_optimizer_scheduler_RNG_restore"] is True
             and recovery["all_gradient_coverage"] is True and recovery["terminal_profile_state"] is True
             and recovery["manual_one_update_probe"] is True and recovery["trainer_fit_resume_not_exercised"] is True
             and recovery["cuda_bitwise_claim"] is False and recovery["updates_before"] == 104
             and recovery["updates_after"] == 105 and recovery["optimizer_updates_executed"] == 1
             and recovery["micro_batches"] == 16 and recovery["actual_changed_parameter_tensors"] == len(shapes), "Actual recovery/update evidence failed")
        gradient_integrity(recovery["gradient_integrity"], shapes)
        need(recovery["original_checkpoint_hashes_unchanged"] == {"last": sha(path("training_state/last.pt")), "best": best_sha},
             "Recovery probe changed the original profile checkpoint bytes")
        need(sha(path("recovery_probe/source_last.pt")) == sha(path("training_state/last.pt"))
             and sha(path("recovery_probe/best_model.bin")) == best_sha, "Recovery probe did not retain exact source checkpoint bytes")
        need(stamp(complete["completed_utc"]) <= stamp(self.document("STARTED.json")["utc"]), "Formal started before its profile completed")
        self.profile_result = result
        return {"full_train_batches": 1652, "full_val_users": 22363, "beam256_val_batches": 2,
                "test_evaluated": False, "manual_restore_and_update_attested": True,
                "CPU_replayed_restore": False, "CUDA_bitwise_claim": False}

    def check_optimizer(self, payload, shapes, updates):
        torch = self.load_torch()
        groups, states = payload["optimizer"]["param_groups"], payload["optimizer"]["state"]
        need(len(groups) == 1 and len(groups[0]["params"]) == len(shapes)
             and len(set(groups[0]["params"])) == len(shapes) and set(states) == set(groups[0]["params"]), "Adam state parameter coverage incomplete")
        for index, (name, shape) in zip(groups[0]["params"], shapes.items()):
            state = states[index]
            step = state["step"].item() if torch.is_tensor(state["step"]) else state["step"]
            need(integer(step) == updates, f"Adam step mismatch: {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                need(torch.is_tensor(state[key]) and tuple(state[key].shape) == shape
                     and bool(torch.isfinite(state[key]).all()), f"Adam moment mismatch: {name}/{key}")
        need(integer(payload["scheduler"]["last_epoch"]) == updates
             and integer(payload["scheduler"]["_step_count"]) == updates + 1, "Scheduler step differs")

    def receipt(self):
        failed = [row["check"] for row in self.checks if row["status"] == "failed"]
        return {"status": "FAIL" if failed else "PASS", "run_id": self.run_id, "variant": self.arm,
                "failed_checks": failed, "checks": self.checks, "audited_files": self.files}


def audit_launch_receipts(audits, args):
    """Bind the successful profile gate to the actual frozen wrapper receipts.

    Resource utilisation/release/accounting are deliberately separate evidence.
    """
    owner = audits["control"]
    binding = {"code_commit": args.root_commit, "engine_commit": ENGINE,
               "runner_sha256": owner.runner_hash,
               "engine_files": {PurePosixPath(name).name: value for name, value in owner.engine_hashes.items()}}
    for key, name in (("wrapper_sha256", "ocn_repair_gpu_job.py"), ("launcher_helper_sha256", "item_local_gpu_job.py")):
        binding[key] = hashlib.sha256(git_bytes(args.reference_root, args.root_commit, "scripts/train/" + name)).hexdigest()

    def completed(mode):
        rows = []
        for path in sorted((args.goal_root / "delivery").glob(mode + "_attempt*/receipt.json")):
            value = read_json(owner.record_file(path))
            need(value.get("campaign_id") == CAMPAIGN and value.get("mode") == mode, "Wrapper receipt campaign/mode differs")
            if value.get("status") == mode + "_completed":
                need(value.get("schema") == "stage07_ocn_repair_job_phase_v1"
                     and value["source_binding"] == binding, "Successful wrapper receipt source binding differs")
                children = value["children"]
                need(len(children) == 2 and {child["arm"] for child in children} == set(RUNS), "Wrapper did not bind exactly both repaired arms")
                for child in children:
                    audit = audits[child["arm"]]
                    need(child["run_id"] == audit.run_id and normpath(child["output"]) == normpath(audit.run)
                         and child["study_sha256"] == hashlib.sha256(audit.study_bytes).hexdigest()
                         and child["status"] in ("completed", "skipped_complete") and child["returncode"] == 0,
                         "Wrapper child identity/exit status differs")
                    suffix = "profile/" if mode == "profile" else ""
                    if child["status"] == "completed":
                        need(child["complete_sha256"] == sha(audit.local(suffix + "COMPLETE.json")), "Wrapper child COMPLETE bytes changed")
                    if mode == "profile":
                        need(child["status"] == "completed" and child["resume"] is False
                             and child["profile_results_sha256"] == sha(audit.local("profile/profile_results.json")),
                             "Wrapper did not bind actual fresh profile result bytes")
                rows.append((path, value))
        need(rows, f"No successful {mode} wrapper receipt was delivered")
        return rows

    profiles = completed("profile")
    need(len(profiles) == 1, "Expected exactly one completed two-arm profile phase")
    path, profile = profiles[0]
    checks = profile["paired_profile_checks"]
    need(checks["status"] == "passed" and checks["common_initial_base_and_rng_verified"] is True
         and set(checks["arms"]) == set(RUNS), "Wrapper paired profile checks missing")
    for arm, audit in audits.items():
        value, recorded = audit.profile_result, checks["arms"][arm]
        need(recorded["profile_results_sha256"] == sha(audit.local("profile/profile_results.json"))
             and recorded["initial_base_state_sha256"] == value["initial_base_state_sha256"]
             and recorded["initial_rng_sha256"] == value["initial_rng_sha256"]
             and recorded["all_parameter_tensors"] == len(audit.model_shapes)
             and recorded["all_gradient_and_optimizer_checks_passed"] is True
             and recorded["manual_recovery_update_passed"] is True,
             "Wrapper profile summary differs from actual per-arm evidence")
        need(equal_number(recorded["train_epoch_seconds"], value["training_loop_seconds"])
             and equal_number(recorded["full_val_seconds"], value["full_val_seconds"])
             and equal_number(recorded["full_test_seconds_estimate"], finite(value["beam256_val_2_batches_seconds"]) * 2796 / 2),
             "Wrapper profile timing summary differs")
    formals = completed("formal")
    for _, value in formals:
        evidence = value["profile_evidence"]
        need(normpath(evidence["path"]) == normpath(path.resolve()) and evidence["sha256"] == sha(path)
             and evidence["paired_profile_checks"] == checks, "Formal wrapper did not reference the accepted SHA-bound profile gate")
        need(stamp(value["started_utc"]) >= stamp(profile["started_utc"]), "Formal wrapper precedes profile")
    return {"status": "passed", "profile_receipt_sha256": sha(path),
            "formal_receipt_sha256": [sha(path) for path, _ in formals],
            "source_binding": binding, "resource_accounting_scope": "separate_platform_evidence_required"}


def compare_pair(audits):
    control, local = audits["control"], audits["item_local"]
    need(not control.receipt()["failed_checks"] and not local.receipt()["failed_checks"], "Both individual formal deliveries must pass before comparison")
    for field in ("base_tensors_excluding_item_local_attention", "initial_rng", "initial_base_state_sha256", "initial_rng_sha256"):
        need(control.initial[field] == local.initial[field], f"Paired complete initial identity differs: {field}")
    clean = lambda audit: {key: value for key, value in audit.config.items()
                          if key not in set(legacy.PATH_KEYS) | set(legacy.NEW_KEYS) | set(legacy.RUNTIME_KEYS) | {"run_id"}}
    need(clean(control) == clean(local), "A/B parsed training recipe differs beyond allowed attention/paths/runtime fields")
    need(control.document("input_identities.json")["files"] == local.document("input_identities.json")["files"], "A/B input identities differ")
    comparison = {}
    for key in METRICS:
        old, new = control.test_metrics[key], local.test_metrics[key]
        comparison[key] = {"control": old, "item_local": new, "absolute_delta": new - old,
                           "relative_change": new / old - 1 if old else None,
                           "relative_change_status": "defined" if old else "undefined_zero_control"}
    ndcg, hr = comparison["ndcg@10"], comparison["hit_rate@10"]
    need(ndcg["relative_change"] is not None, "NDCG@10 signal ratio is undefined at zero control")
    return {"status": "passed", "comparator": "same-campaign repaired control", "metrics": comparison,
            "exploratory_positive_signal": ndcg["relative_change"] >= .03 and hr["absolute_delta"] >= 0,
            "signal_rule": "item_local/control NDCG@10 relative gain >=3%, HR@10 does not decrease",
            "signal_is_not_a_delivery_pass_requirement": True, "evaluation_level": "sid", "seed": 2024,
            "initial_base_state_sha256": control.initial["initial_base_state_sha256"],
            "initial_rng_sha256": control.initial["initial_rng_sha256"],
            "result_sha256": {arm: sha(audit.local("results.json")) for arm, audit in audits.items()},
            "old03_role": "historical_background_only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal-root", type=Path, required=True, help="Original Goal tree containing runs/<exact-run-id>/{workspace,profile,...}")
    parser.add_argument("--root-commit", required=True, help="Explicit frozen execution commit; must be the authorized 2fc97b6 full SHA")
    parser.add_argument("--reference-root", type=Path, help="Pinned root Git checkout and Stage03 references; default GOAL/code")
    parser.add_argument("--engine-root", type=Path, help="Pinned engine Git checkout; default GOAL/engine")
    parser.add_argument("--output", type=Path, required=True, help="New JSON outside runs/ and both source checkouts; never overwritten")
    args = parser.parse_args(argv)
    args.goal_root = args.goal_root.resolve()
    args.reference_root = (args.reference_root or args.goal_root / "code").resolve()
    args.engine_root = (args.engine_root or args.goal_root / "engine").resolve()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(path) for path in
                            (args.goal_root / "runs", args.reference_root, args.engine_root)):
        parser.error("--output must be a new file outside artifact runs and source checkouts")
    audits, top_checks, pair, launch = {}, [], None, None
    try:
        need(args.root_commit == EXECUTION_ROOT, "--root-commit differs from the authorized frozen 2fc97b6 execution")
        need(args.goal_root.is_dir(), "Original --goal-root directory is missing")
        for arm in RUNS:
            audit = RepairAudit(args, arm)
            audits[arm] = audit
            for name, function in (("mandatory_artifacts", audit.audit_inventory), ("pinned_references", audit.audit_references),
                    ("formal_identity_and_execution_source", audit.audit_identity), ("complete_parsed_configuration", audit.audit_configuration),
                    ("twelve_input_bytes_and_SID_identity", audit.audit_inputs), ("full_curves_and_validation_selection", audit.audit_curves),
                    ("full_position_and_order_diagnostics", audit.audit_diagnostic_curves), ("once_only_test_and_results", audit.audit_test_and_results),
                    ("runtime_metadata", audit.audit_metadata), ("best_checkpoint_CPU_and_strict_reload", audit.audit_best_checkpoint),
                    ("terminal_Adam_scheduler_RNG_state", audit.audit_terminal_state), ("gradients_initialization_and_parameter_changes", audit.audit_training_integrity),
                    ("full_profile_and_actual_recovery_evidence", audit.audit_profile), ("unchanged_audited_bytes", audit.audit_unchanged_files)):
                audit.section(name, function)
        pair = compare_pair(audits)
        launch = audit_launch_receipts(audits, args)
        for audit in audits.values():
            audit.audit_unchanged_files()
        top_checks.append({"check": "paired_repaired_comparison", "status": "passed"})
    except Exception as error:
        top_checks.append({"check": "paired_repaired_comparison", "status": "failed", "error_type": type(error).__name__, "error": str(error)})
    receipts = {arm: audit.receipt() for arm, audit in audits.items()}
    passed = (len(receipts) == 2 and all(value["status"] == "PASS" for value in receipts.values())
              and all(row["status"] == "passed" for row in top_checks) and pair is not None and launch is not None)
    receipt = {"schema": "stage07_ocn_repair_pair_audit_v1", "status": "PASS" if passed else "FAIL",
               "campaign_id": CAMPAIGN, "root_commit": args.root_commit, "engine_commit": ENGINE,
               "audited_utc": datetime.now(timezone.utc).isoformat(), "artifact_audit_only": True, "goal_complete": False,
               "checks": top_checks, "runs": receipts, "paired_comparison": pair, "profile_launch_evidence": launch,
               "limitations": ["SID-level single-seed comparison; no statistical-significance or item-level claim.",
                   "Old Stage03 metrics are background only; its original full runtime config/source/RNG gaps remain.",
                   "CPU validates stored bytes and SHA-bound execution evidence; it does not execute CUDA, model inference, checkpoint restore or a continuation step.",
                   "The profile recovery evidence is a terminal-state manual one-update probe, not Trainer.fit resume or CUDA bitwise replay.",
                   "All 84 base tensor changes are independently hash-compared; six local initial hashes were not stored, so all-90 change coverage relies on the frozen runner attestation.",
                   "GPU release, platform terminal state, allocation duration and the cumulative 28-GPU-hour budget require separate platform evidence."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "campaign_id": CAMPAIGN, "output": str(output),
                      "failed_checks": {arm: value["failed_checks"] for arm, value in receipts.items()},
                      "pair_errors": [row.get("error") for row in top_checks if row["status"] == "failed"],
                      "goal_complete": False}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
