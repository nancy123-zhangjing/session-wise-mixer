"""Seed extension of the frozen Stage07 alternating joint-training runner.

The original runner and engine remain unchanged. This adapter changes only
seed/run identity, the explicitly recorded stopping policy, budget binding,
and the formal delivery tail (training only; evaluation is a separate step).
It preserves the original training, precision repair, checkpoint selection,
gradient audit, recovery, and saved-SID synchronization implementation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "stage07_item_local_joint_multiseed_20261021_r1"
SEEDS = (2025, 2026)
VARIANTS = ("control", "item_local")
PINNED_BASE_SOURCE_SHA256 = "61f6c36c02ee59498add74c28af8b3a15a54e2f38465bc65bf3c0d8353850475"
EXPECTED_ENGINE_COMMIT = "1c879252f6395d6e64462511823c3929b9c42c00"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def campaign_id(seed):
    need(seed in SEEDS, "Only the preregistered new seeds 2025 and 2026 are supported")
    return f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1"


def run_id(seed, variant):
    need(seed in SEEDS and variant in VARIANTS, "Unknown multiseed arm")
    return f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"


def source_sha256(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest()


def validate_config(config, seed):
    allowed = {"schema_version", "campaign_id", "seeds", "recipe_id", "variants",
               "max_gpu_hours", "prior_allocated_gpu_seconds", "gpu_type", "gpu_count_per_run",
               "training_budget_policy", "final_evaluation", "initialization", "training", "gpu_cap_scope",
               "selection", "global_codebook_mixer_enabled", "tokenizer_codebooks_trainable",
               "recommendation_gradient_bridge", "evaluation_level", "historical_results_role"}
    need(not set(config) - allowed, f"Unknown multiseed settings: {sorted(set(config) - allowed)}")
    need(config.get("schema_version") == "stage07_joint_multiseed_v1"
         and config.get("campaign_id") == CAMPAIGN and config.get("seeds") == list(SEEDS),
         "Exact multiseed campaign identity and seeds are required")
    need(seed in SEEDS and config.get("recipe_id") == "refined_joint", "Wrong seed or warm-start recipe")
    need(config.get("gpu_type") == "H100" and config.get("gpu_count_per_run") == 1,
         "Each arm must use one visible H100; parallelism is between arms")
    cap, prior = config.get("max_gpu_hours"), config.get("prior_allocated_gpu_seconds")
    need(isinstance(cap, (float, int)) and not isinstance(cap, bool) and cap == 14.0,
         "Preserve the new campaign execution safety cap of 14 GPU hours")
    need(config.get("gpu_cap_scope") == "execution_safety_cap_not_user_total_budget",
         "The execution safety cap must not be presented as a user-declared total budget")
    need(isinstance(prior, (float, int)) and not isinstance(prior, bool) and math.isfinite(prior)
         and 0 <= prior < cap * 3600, "Invalid prior allocation for this new campaign")
    expected = {name: {"item_local_attention": {"enabled": name == "item_local", "num_heads": 4, "dropout": 0.0}}
                for name in VARIANTS}
    need(config.get("variants") == expected, "Local attention must be the sole paired model difference")
    need(config.get("training_budget_policy") == "fixed_45", "All new arms require the fixed 45-epoch budget")
    need(config.get("final_evaluation") == "separate_validation_beam32_256", "This runner must not evaluate the test split")
    for key, expected_value in {"global_codebook_mixer_enabled": False, "tokenizer_codebooks_trainable": True,
                                "recommendation_gradient_bridge": "diger_st", "evaluation_level": "sid",
                                "historical_results_role": "seed2024 supplementary only"}.items():
        need(config.get(key) == expected_value, f"Campaign contradicts frozen training: {key}")
    study = copy.deepcopy(config)
    study.update(parent_campaign_id=CAMPAIGN, campaign_id=campaign_id(seed), protocol_id=campaign_id(seed),
                 seed=seed, gpu_count=1,
                 variants={name: {"run_id": run_id(seed, name), **copy.deepcopy(value)}
                           for name, value in config["variants"].items()})
    return study


def apply_stopping_policy(config, audit, policy):
    need(policy == "fixed_45", "All new arms require fixed training quantity")
    expected = {"patience": 5, "epochs": 45, "eval_start_epoch": 5,
                "eval_interval": 5, "val_metric": "weighted_score", "crosspse_seed": 2024}
    need(all(config.get(key) == value for key, value in expected.items()),
         "Frozen recipe stopping/selection/catalog settings changed")
    beam = config.get("vectorized_beam_search", {})
    need(beam.get("val") == {"beam_act": 32, "beam_max": 32}, "Checkpoint selection must retain validation beam32")
    result = copy.deepcopy(config)
    result["patience"] = 0
    return result, {"status": "passed", "frozen_recipe_audit": audit,
                    "adapter_overrides": {"patience": {"before": 5, "after": 0}},
                    "training_budget_policy": policy,
                    "checkpoint_selection_unchanged": True,
                    "selection_schedule": "Initial validation and every five recommender epochs; beam32 weighted_score",
                    "reason": "All four new arms complete 45 logical epochs (35 recommender and 10 tokenizer phases)"}


def adapted_prepare_ast(source):
    """Only seed and adapter/source identity additions to original prepare."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")

    class SeedAndIdentity(ast.NodeTransformer):
        seeds = 0
        identities = 0
        configurations = 0
        artifacts = 0

        def visit_Constant(self, item):
            if type(item.value) is int and item.value == 2024:
                self.seeds += 1
                return ast.copy_location(ast.Name(id="_MULTISEED_SEED", ctx=ast.Load()), item)
            return item

        def visit_Assign(self, statement):
            statement = self.generic_visit(statement)
            if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name == "identity":
                    self.identities += 1
                    return [statement, ast.parse("identity.update(multiseed_adapter=_MULTISEED_ADAPTER_IDENTITY, parent_campaign_id=_MULTISEED_PARENT_CAMPAIGN)").body[0]]
                if name == "values":
                    self.artifacts += 1
                    return [statement, ast.parse("values['code_identity.json']['multiseed_adapter'] = _MULTISEED_ADAPTER_IDENTITY").body[0]]
            return statement

        def visit_Expr(self, statement):
            statement = self.generic_visit(statement)
            call = statement.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "config"
                    and call.func.attr == "update"
                    and any(keyword.arg == "rand_seed" for keyword in call.keywords)):
                self.configurations += 1
                return [statement, ast.parse("config['training_state_identity'].update(seed=_MULTISEED_SEED, multiseed_adapter=_MULTISEED_ADAPTER_IDENTITY)").body[0]]
            return statement

    transform = SeedAndIdentity()
    node = transform.visit(node)
    need((transform.seeds, transform.identities, transform.configurations, transform.artifacts) == (2, 1, 1, 1),
         "Original prepare anchors changed")
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


TRAINING_ONLY_TAIL = """
result_name = 'results.json'
need(not (work / 'TEST_STARTED.json').exists() and not (work / 'TEST_RESULT.json').exists(),
     'Multiseed training must not consume test targets')
need(all(row['split'] != 'test' for row in timings), 'Unexpected test evaluation')
if prepared['study']['training_budget_policy'] == 'fixed_45':
    need(fit['completed_epochs'] == 45 and fit['training_stop_reason'] == 'max_epochs'
         and fit['cumulative_recommender_optimizer_updates'] == 3640
         and terminal['progress']['decoder_epochs_completed'] == 35
         and joint_progress['tokenizer_epochs_completed'] == 10
         and joint_progress['tokenizer_optimizer_updates'] == 60
         and joint_progress['sid_refresh_count'] == 10,
         'Fixed training budget was not completed')
    phase_counts = {phase: sum(row['phase'] == phase for row in audit.records)
                    for phase in ('recommender', 'tokenizer')}
    need(phase_counts == {'recommender': 35, 'tokenizer': 10}, 'Alternating phase budget differs')
result = {**common, 'status': 'formal_training_completed_validation_pending',
          'seed': _MULTISEED_SEED, 'parent_campaign_id': _MULTISEED_PARENT_CAMPAIGN,
          'initial_SID_sha256': initial['initial_SID_sha256'],
          'test_data_evaluated': False, 'final_validation_pending': [32, 256],
          'training_budget_policy': prepared['study']['training_budget_policy'],
          'phase_integrity_sha256': sha(work / 'phase_integrity.json'),
          'process_wall_seconds': time.perf_counter() - started}
"""


def adapted_run_ast(source):
    """Replace budget constant and formal test tail; retain training verbatim."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")

    class TrainingOnly(ast.NodeTransformer):
        budget = 0
        tail = 0

        def visit_BinOp(self, item):
            if (isinstance(item.op, ast.Mult) and isinstance(item.left, ast.Constant)
                    and item.left.value == 28 and isinstance(item.right, ast.Constant) and item.right.value == 3600):
                self.budget += 1
                return ast.copy_location(ast.parse("prepared['study']['max_gpu_hours'] * 3600", mode="eval").body, item)
            return self.generic_visit(item)

        def visit_If(self, item):
            item = self.generic_visit(item)
            # Only the final smoke/formal branch has this formal result marker.
            if (item.orelse and isinstance(item.orelse[0], ast.Assign)
                    and isinstance(item.orelse[0].value, ast.Constant)
                    and item.orelse[0].value.value == "results.json"):
                self.tail += 1
                item.orelse = ast.parse(TRAINING_ONLY_TAIL).body
            return item

    transform = TrainingOnly()
    node = transform.visit(node)
    need((transform.budget, transform.tail) == (1, 1), "Original run adaptation anchors changed")
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def load_runner(seed, base_runner_path=None):
    """Create an isolated module namespace for one seed; imports no torch."""
    campaign_id(seed)
    path = Path(base_runner_path or Path(__file__).with_name("run_item_local_joint.py")).resolve()
    source = path.read_text(encoding="utf-8")
    need(source_sha256(source) == PINNED_BASE_SOURCE_SHA256, "Frozen base runner source changed")
    spec = importlib.util.spec_from_file_location(f"frozen_item_local_joint_seed{seed}", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    adapter_identity = {"path": str(Path(__file__).resolve()),
                        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "base_runner_path": str(path), "base_runner_normalized_sha256": source_sha256(source),
                        "base_runner_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    runner.CAMPAIGN = campaign_id(seed)
    runner.run_id = lambda variant: run_id(seed, variant)
    runner.validate_study = lambda config: validate_config(config, seed)
    runner._MULTISEED_SEED = seed
    runner._MULTISEED_PARENT_CAMPAIGN = CAMPAIGN
    runner._MULTISEED_ADAPTER_IDENTITY = adapter_identity
    original_merge = runner.merge_recipe

    def merge_recipe(engine, package, recipe):
        config, audit, layers = original_merge(engine, package, recipe)
        # The stopping policy is applied by prepare after study validation.
        policy = runner._MULTISEED_VALIDATED_POLICY
        config, audit = apply_stopping_policy(config, audit, policy)
        return config, audit, layers

    def validate_study(config):
        study = validate_config(config, seed)
        runner._MULTISEED_VALIDATED_POLICY = study["training_budget_policy"]
        return study

    runner.validate_study = validate_study
    runner.merge_recipe = merge_recipe
    exec(compile(adapted_prepare_ast(source), str(path), "exec"), runner.__dict__)
    exec(compile(adapted_run_ast(source), str(path), "exec"), runner.__dict__)
    return runner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--base-runner")
    args, remaining = parser.parse_known_args()
    runner = load_runner(args.seed, args.base_runner)
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *remaining]
        runner.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
