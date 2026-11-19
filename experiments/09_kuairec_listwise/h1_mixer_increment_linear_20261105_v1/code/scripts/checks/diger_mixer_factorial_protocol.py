"""Locked common-start four-arm Mixer/update factorial, standard-library checks."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/05_crosspse_joint/pilots/diger_mixer_factorial_seed2024.json"
PROTOCOL_ID = "stage05_diger_mixer_factorial_once_freeze_recover_v1"
ARMS = ("no_mixer_frozen", "no_mixer_joint", "mixer_frozen", "mixer_joint")
MIXER_INITIALIZATION = {
    "base": "bound_common_initial_effective_codebooks", "n_heads": 4,
    "dropout": 0, "attention_scope": "full", "residual_scale_init": 1,
    "learn_residual_scale": False, "zero_init_attention_output": True,
    "train_base_codebooks": True,
}
CONTROLLED_TOKENIZER_SETTINGS = {
    "joint_codebook_lr": 0.00001, "joint_mixer_lr": 0.0001,
    "joint_weight_decay": 0.0001, "joint_balance_weight": 0.01,
    "joint_anchor_weight": 0.01, "joint_balance_temperature": 0.1,
    "max_grad_norm": 1.0, "crosspse_dropout": 0.0,
}

def _load_common():
    path = ROOT / "scripts/checks/diger_noise_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("_mixer_factorial_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_common = _load_common()
relative_path = _common.relative_path
phase_seed = _common.phase_seed
loss_divisor = _common.loss_divisor
file_sha256 = _common.file_sha256

def _projection(protocol):
    # Only reuse archived asset validation; never execute legacy noise arms.
    projected = dict(protocol)
    projected.update(protocol_id="stage05_diger_noise_once_freeze_recover_v1",
                     noise_start=1.0, noise_end=0.2)
    return projected

def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unknown Mixer factorial protocol")
    if protocol.get("arms") != list(ARMS):
        raise ValueError("All four factorial arms and their order are required")
    fixed_ints = {"seed": 2024, "gradient_accumulation_steps": 16,
                  "tokenizer_batches": 384, "decoder_epochs": 3,
                  "tokenizer_schedule_total_updates": 60, "mixer_init_seed": 2024}
    for key, expected in fixed_ints.items():
        if type(protocol.get(key)) is not int or protocol[key] != expected:
            raise ValueError(f"Fixed integer control differs: {key}")
    for key, expected in {"decoder_lr": 0.0001, "decoder_weight_decay": 0.0,
                          "assignment_temperature": 0.1, "softmax_temperature": 1.0,
                          "noise_start": 0.0, "noise_end": 0.0,
                          "expected_label_smoothing": 0.2}.items():
        if isinstance(protocol.get(key), bool) or protocol.get(key) != expected:
            raise ValueError(f"Fixed scalar control differs: {key}")
    if protocol.get("mixer_initialization") != MIXER_INITIALIZATION:
        raise ValueError("Common effective base, fixed nonzero residual and zero output are required")
    if protocol.get("controlled_tokenizer_settings") != CONTROLLED_TOKENIZER_SETTINGS:
        raise ValueError("Shared optimizer/regularizer controls differ")
    if protocol.get("recommendation_weights") != {arm: 1.0 for arm in ARMS}:
        raise ValueError("Both Joint arms must receive the same recommendation feedback")
    if protocol.get("tokenizer_probe_updates") != [6, 24]:
        raise ValueError("Keep fixed no-interference probes at update 6 and 24")
    if protocol.get("evaluation") != {
        "split": "val", "level": "sid", "primary_decoder_epoch": 3,
        "checkpoint_selection": "none_fixed_budget", "no_test": True,
    }:
        raise ValueError("Validation-only fixed final epoch, not best-of-arm selection")
    relative_path(ROOT, protocol.get("common_snapshot_run"))
    if not re.fullmatch(r"[0-9a-f]{64}", protocol.get("common_snapshot_summary_sha256", "")):
        raise ValueError("Bind the complete source run by summary SHA256")
    smoke = protocol.get("smoke", {})
    for key, expected in {"tokenizer_batches": 16, "decoder_batches": 16,
                          "eval_batches": 2, "decoder_epochs": 1}.items():
        if type(smoke.get(key)) is not int or smoke[key] != expected:
            raise ValueError(f"Smoke control differs: {key}")
    if smoke.get("tokenizer_probe_updates") != [1]:
        raise ValueError("Smoke probe must be one complete tokenizer update")
    for key in ("train_batch_size", "eval_batch_size"):
        if key in smoke and (type(smoke[key]) is not int or smoke[key] != 2):
            raise ValueError("Smoke uses batch size 2")
    _common.validate_protocol(_projection(protocol))

def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol

def arm_names(protocol):
    validate_protocol(protocol)
    return ARMS

def arm_settings(protocol, arm, smoke=False):
    validate_protocol(protocol)
    if arm not in ARMS:
        raise ValueError(f"Unknown factorial arm: {arm}")
    short = protocol["smoke"] if smoke else {}
    joint = arm.endswith("_joint")
    enabled = arm.startswith("mixer_")
    return {
        "arm": arm, "mixer_enabled": enabled, "tokenizer_variant": "cross_pse" if enabled else "refined_pse",
        "tokenizer_batches": short.get("tokenizer_batches", 384) if joint else 0,
        "decoder_epochs": short.get("decoder_epochs", 3),
        "decoder_batch_limit": short.get("decoder_batches", 0),
        "eval_batch_limit": short.get("eval_batches", 0),
        "noise_start": 0.0, "noise_end": 0.0, "noise_total_steps": 60,
        "decoder_lr": 0.0001, "gradient_accumulation_steps": 16,
        "recommendation_weight": 1.0,
        "tokenizer_probe_updates": list(short.get("tokenizer_probe_updates", [6, 24])) if joint else [],
        "audit_recommendation_feedback": joint,
    }

def verify_common_snapshot(project_root, protocol):
    validate_protocol(protocol)
    root = Path(project_root).resolve()
    run = relative_path(root, protocol["common_snapshot_run"])
    if run.parent != root / "outputs/pilots/stage05_feedback":
        raise ValueError("Common snapshot must be from the bound feedback pilot")
    summary_path = run / "summary.json"
    expected = protocol["common_snapshot_summary_sha256"]
    if file_sha256(summary_path) != expected:
        raise RuntimeError("Common source summary checksum mismatch")
    complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("summary_sha256") != expected or not complete.get("all_protocol_arms_complete"):
        raise RuntimeError("Common source run is not fully complete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "pilot" or summary.get("status") != "complete":
        raise RuntimeError("A smoke/partial run cannot supply a formal common start")
    frozen = next(row for row in summary["arms"] if row["arm"] == "frozen")
    snapshot = next(row for row in frozen["snapshots"] if row["label"] == "initial")
    if any(snapshot.get(key) != 0 for key in
           ("tokenizer_updates", "recommender_updates", "recommender_epochs")):
        raise RuntimeError("Only the pre-update initial snapshot may supply the common state")
    directory = run / "frozen/snapshots/initial"
    files = snapshot["files"]
    if set(files) != {"model.pt", "training_state.pt", "semantic_ids.npy", "crossed_codebooks.npy"}:
        raise RuntimeError("Incomplete canonical initial snapshot")
    for name, sha in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(root) or file_sha256(path) != sha:
            raise RuntimeError(f"Common initial input checksum mismatch: {name}")
    return {"run_dir": str(run), "snapshot_dir": str(directory), "files": files,
            "model_tensor_state_sha256": snapshot["model_tensor_state_sha256"],
            "semantic_ids_array_sha256": snapshot["semantic_ids_array_sha256"],
            "summary_sha256": expected}

def verify_assets(project_root, protocol):
    validate_protocol(protocol)
    root = Path(project_root).resolve()
    receipts = _common.verify_assets(root, _projection(protocol))
    common = verify_common_snapshot(root, protocol)
    for name, sha in common["files"].items():
        path = Path(common["snapshot_dir"]) / name
        receipts.append({"path": path.relative_to(root).as_posix(), "sha256": sha,
                         "size_bytes": path.stat().st_size, "binding": "common_initial_snapshot"})
    for name in ("summary.json", "COMPLETE.json"):
        path = Path(common["run_dir"]) / name
        receipts.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                         "size_bytes": path.stat().st_size, "binding": "common_run_completion"})
    return receipts

def source_receipt(project_root):
    result = _common.source_receipt(Path(project_root).resolve())
    paths = ["scripts/train/run_diger_mixer_factorial.py",
             "scripts/checks/diger_mixer_factorial_protocol.py",
             "tests/test_diger_mixer_factorial.py", DEFAULT_CONFIG]
    result["execution_source_sha256"] = {
        str(ROOT / name): file_sha256(ROOT / name) for name in paths if (ROOT / name).is_file()
    }
    result["execution_git_head"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    path = args.config if args.config.is_absolute() else ROOT / args.config
    protocol = load_protocol(path)
    result = {"protocol_id": PROTOCOL_ID, "arms": [arm_settings(protocol, arm) for arm in ARMS],
              "common_start": "shared effective codebooks/SID/full recommender; new zero-output Mixer",
              "primary_pair": "mixer_joint - no_mixer_joint",
              "interaction_increment": "(mixer_joint-mixer_frozen)-(no_mixer_joint-no_mixer_frozen)",
              "negative_control": "the two Frozen trajectories must agree",
              "scope": "warm-start once-update/freeze/recover; single seed, validation SID-level"}
    if not args.plan_only:
        result["assets"] = verify_assets(args.project_root, protocol)
        result["common_snapshot"] = verify_common_snapshot(args.project_root, protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
