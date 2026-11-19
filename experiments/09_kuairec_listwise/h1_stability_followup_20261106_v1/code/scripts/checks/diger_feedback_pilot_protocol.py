"""Read-only, standard-library protocol for the stage-05 feedback ablation.

Three shared-start arms isolate tokenizer reconstruction refinement from the
additional recommendation gradient. No PyTorch imports or asset mutations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json"
PROTOCOL_ID = "stage05_diger_feedback_once_freeze_recover_v1"
ARMS = ("frozen", "cross_recon", "cross_rec")
RECOMMENDATION_WEIGHTS = {"frozen": 1.0, "cross_recon": 0.0, "cross_rec": 1.0}
CONTROLLED_TOKENIZER_SETTINGS = {
    "joint_codebook_lr": 0.00001,
    "joint_mixer_lr": 0.0001,
    "joint_weight_decay": 0.0001,
    "joint_balance_weight": 0.01,
    "joint_anchor_weight": 0.01,
    "joint_balance_temperature": 0.1,
    "max_grad_norm": 1.0,
    "crosspse_dropout": 0.0,
}


def _load_common_helper():
    path = ROOT / "scripts/checks/diger_noise_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("_diger_feedback_common_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared runtime protocol helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_common_helper()
relative_path = _common.relative_path
phase_seed = _common.phase_seed
loss_divisor = _common.loss_divisor
file_sha256 = _common.file_sha256


def _resource_projection(protocol: dict) -> dict:
    """Reuse only common resource checks, never the noise arm implementation.

    The archived source was a noise-study checkpoint. Its helper also checks
    generic paths, asset hashes, and source schedule binding. Those shared
    checks require its legacy protocol label/nonzero-noise shape. This temporary
    projection is not returned, written, or passed to the training runner.
    """
    result = dict(protocol)
    result.update({
        "protocol_id": "stage05_diger_noise_once_freeze_recover_v1",
        "noise_start": 1.0,
        "noise_end": 0.2,
    })
    return result


def _require_fixed(protocol: dict, key: str, expected, *, integer: bool = False) -> None:
    value = protocol.get(key)
    if isinstance(value, bool) or value != expected or (integer and type(value) is not int):
        raise ValueError(f"{key} must be fixed to {expected!r} for this feedback protocol")


def validate_protocol(protocol: dict) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("Feedback pilot configuration must be an object")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unknown feedback pilot protocol")
    if protocol.get("arms") != list(ARMS):
        raise ValueError(f"arms must preserve the paired order {ARMS!r}")

    fixed = {
        "gradient_accumulation_steps": 16,
        "tokenizer_batches": 384,
        "decoder_epochs": 3,
        "tokenizer_schedule_total_updates": 60,
    }
    for key, expected in fixed.items():
        _require_fixed(protocol, key, expected, integer=True)
    for key, expected in {
        "decoder_lr": 0.0001,
        "decoder_weight_decay": 0.0,
        "assignment_temperature": 0.1,
        "softmax_temperature": 1.0,
        "noise_start": 0.0,
        "noise_end": 0.0,
    }.items():
        _require_fixed(protocol, key, expected)
    if protocol.get("tokenizer_probe_updates") != [6, 24] or any(
        type(value) is not int for value in protocol.get("tokenizer_probe_updates", [])
    ):
        raise ValueError("tokenizer_probe_updates must be the fixed [6, 24] diagnostic checkpoints")

    weights = protocol.get("recommendation_weights")
    if not isinstance(weights, dict) or set(weights) != set(ARMS):
        raise ValueError("recommendation_weights must bind exactly the three feedback arms")
    for arm, expected in RECOMMENDATION_WEIGHTS.items():
        _require_fixed(weights, arm, expected)
    controlled = protocol.get("controlled_tokenizer_settings")
    if not isinstance(controlled, dict) or set(controlled) != set(CONTROLLED_TOKENIZER_SETTINGS):
        raise ValueError("controlled_tokenizer_settings must explicitly pin the shared optimizer and regularizers")
    for key, expected in CONTROLLED_TOKENIZER_SETTINGS.items():
        _require_fixed(controlled, key, expected)

    smoke = protocol.get("smoke")
    if not isinstance(smoke, dict):
        raise ValueError("smoke must be an object")
    for key, expected in {
        "tokenizer_batches": 16, "decoder_batches": 16, "eval_batches": 2, "decoder_epochs": 1,
    }.items():
        _require_fixed(smoke, key, expected, integer=True)
    # The shared runtime permits optional smoke batch-size overrides. Keep
    # its default of 2 pinned: a hidden override must not enlarge the test.
    for key in ("train_batch_size", "eval_batch_size"):
        if key in smoke:
            _require_fixed(smoke, key, 2, integer=True)
    if smoke.get("tokenizer_probe_updates") != [1] or type(smoke["tokenizer_probe_updates"][0]) is not int:
        raise ValueError("smoke.tokenizer_probe_updates must be [1]")
    _common.validate_protocol(_resource_projection(protocol))


def load_protocol(config_path: Path) -> dict:
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def arm_names(protocol: dict) -> tuple[str, ...]:
    validate_protocol(protocol)
    return ARMS


def arm_settings(protocol: dict, arm: str, smoke: bool = False) -> dict:
    validate_protocol(protocol)
    if arm not in ARMS:
        raise ValueError(f"Unknown feedback arm: {arm}")
    short = protocol["smoke"] if smoke else {}
    frozen = arm == "frozen"
    return {
        "arm": arm,
        "tokenizer_batches": 0 if frozen else short.get("tokenizer_batches", protocol["tokenizer_batches"]),
        "decoder_epochs": short.get("decoder_epochs", protocol["decoder_epochs"]),
        "decoder_batch_limit": short.get("decoder_batches", 0),
        "eval_batch_limit": short.get("eval_batches", 0),
        "noise_start": 0.0,
        "noise_end": 0.0,
        "noise_total_steps": protocol["tokenizer_schedule_total_updates"],
        "decoder_lr": protocol["decoder_lr"],
        "gradient_accumulation_steps": protocol["gradient_accumulation_steps"],
        "recommendation_weight": protocol["recommendation_weights"][arm],
        "tokenizer_probe_updates": [] if frozen else list(
            short.get("tokenizer_probe_updates", protocol["tokenizer_probe_updates"])
        ),
        "audit_recommendation_feedback": True,
    }


def verify_assets(project_root: Path, protocol: dict) -> list[dict]:
    validate_protocol(protocol)
    return _common.verify_assets(Path(project_root), _resource_projection(protocol))


def source_receipt(project_root: Path) -> dict:
    """Record both the reused execution runtime and the new ablation entrypoints."""
    root = Path(project_root).resolve()
    result = _common.source_receipt(root)
    extra_paths = [
        "scripts/train/run_diger_feedback_pilot.py",
        "scripts/checks/diger_feedback_pilot_protocol.py",
        "tests/test_diger_feedback_protocol.py",
        "tests/test_diger_feedback_runner.py",
        DEFAULT_CONFIG,
    ]
    for value in extra_paths:
        path = relative_path(root, value)
        if path.is_file():
            result["source_sha256"][value] = file_sha256(path)
    result["source_sha256"] = dict(sorted(result["source_sha256"].items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--plan-only", action="store_true", help="Validate controls without reading large assets")
    args = parser.parse_args()
    root = args.project_root.resolve()
    path = args.config if args.config.is_absolute() else relative_path(root, args.config.as_posix())
    protocol = load_protocol(path)
    result = {
        "protocol_id": protocol["protocol_id"],
        "shared_checkpoint": protocol["shared_checkpoint"],
        "arms": [arm_settings(protocol, arm) for arm in arm_names(protocol)],
        "controlled_tokenizer_settings": protocol["controlled_tokenizer_settings"],
        "optimizer_start": "fresh, identical per arm; not exact historical resume",
        "evaluation": "validation only; SID-level; fixed budgets; no best-probe selection",
        "asset_verification": "not_requested" if args.plan_only else "passed",
    }
    if not args.plan_only:
        result["inputs"] = verify_assets(root, protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
