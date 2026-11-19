"""Static preflight for the matched frozen Cross-PSE control."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "diff_formal_crosspse_frozen_decoder_continuation_4x256_seed2024"


def merge(paths: list[Path]) -> dict:
    config: dict = {}
    for path in paths:
        config.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return config


def resolve_from_diff_root(value: str) -> Path:
    return (ROOT / "external" / "DiffGRM" / value).resolve()


def phase(config: dict, epoch_index: int) -> str:
    if config.get("joint_frozen_tokenizer_control", False):
        return "recommender"
    warmup = int(config.get("joint_recommender_warmup_epochs", 0))
    if epoch_index < warmup:
        return "recommender"
    cycle = int(config.get("joint_cycle_epochs", 4))
    tokenizer_epochs = int(config.get("joint_tokenizer_epochs_per_cycle", 1))
    phase_index = (epoch_index - warmup) // cycle
    if (
        phase_index < int(config.get("joint_max_tokenizer_phases", 0))
        and (epoch_index - warmup) % cycle < tokenizer_epochs
    ):
        return "tokenizer"
    return "recommender"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--recommender-checkpoint", type=Path)
    args = parser.parse_args()

    shared_paths = [
        ROOT / "configs/shared/diff_formal_common.yaml",
        ROOT / "configs/shared/diff_formal_repo_v6.yaml",
        ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger_v2.yaml",
    ]
    control_path = ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_frozen_decoder_v1.yaml"
    joint_config = merge(shared_paths)
    control_config = merge([*shared_paths, control_path])

    allowed_differences = {
        "run_id",
        "epochs",
        "reproduction_protocol",
        "protocol_note",
        "joint_frozen_tokenizer_control",
        "joint_recommender_warmup_epochs",
        "joint_max_tokenizer_phases",
    }
    actual_differences = {
        key
        for key in set(joint_config) | set(control_config)
        if joint_config.get(key) != control_config.get(key)
    }
    assert actual_differences == allowed_differences, (
        "Frozen control changed unexpected settings: "
        f"expected={sorted(allowed_differences)}, "
        f"actual={sorted(actual_differences)}"
    )

    assert control_config["run_id"] == RUN_ID
    assert control_config["joint_crosspse_enabled"] is True
    assert control_config["joint_frozen_tokenizer_control"] is True
    assert control_config["joint_max_tokenizer_phases"] == 0
    assert control_config["epochs"] == 35
    assert control_config["n_digit"] == 4
    assert control_config["codebook_size"] == 256
    assert control_config["mixed_precision"] == "bf16"
    assert control_config["evaluation_level"] == "sid"
    assert control_config["eval_expand_sid_to_items"] is False

    joint_decoder_epochs = sum(
        phase(joint_config, epoch) == "recommender"
        for epoch in range(int(joint_config["epochs"]))
    )
    control_decoder_epochs = sum(
        phase(control_config, epoch) == "recommender"
        for epoch in range(int(control_config["epochs"]))
    )
    assert joint_decoder_epochs == 35
    assert control_decoder_epochs == joint_decoder_epochs
    assert all(
        phase(control_config, epoch) == "recommender"
        for epoch in range(int(control_config["epochs"]))
    )

    required_code = [
        ROOT / "scripts/train/run_diff_crosspse_frozen.sh",
        ROOT / "scripts/checks/check_diff_crosspse_frozen_smoke.py",
        ROOT / "configs/05_crosspse_joint/diff_frozen_crosspse_smoke.yaml",
        ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py",
    ]
    for path in required_code:
        assert path.is_file(), f"Missing frozen-control pipeline file: {path}"

    artifact = resolve_from_diff_root(
        control_config["joint_crosspse_artifact_dir"]
    )
    required_artifacts = [
        artifact / "best.pt",
        artifact / "base_pse_codebooks.npy",
        artifact / "opq_residual_embeddings.npy",
        artifact / "semantic_ids.npy",
        artifact / "semantic_ids.json",
        artifact / "summary.json",
        resolve_from_diff_root(control_config["joint_crosspse_items_path"]),
    ]
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing and not args.allow_missing_artifacts:
        raise FileNotFoundError(
            "Missing server artifacts:\n" + "\n".join(map(str, missing))
        )
    if not args.allow_missing_artifacts:
        if args.recommender_checkpoint is None:
            raise ValueError("--recommender-checkpoint is required")
        if not args.recommender_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing pretrained recommender: {args.recommender_checkpoint}"
            )

    print("Frozen Cross-PSE matched-control preflight: OK")
    print(f"run_id: {control_config['run_id']}")
    print(f"joint Decoder epochs: {joint_decoder_epochs}")
    print(f"control Decoder epochs: {control_decoder_epochs}")
    print("Tokenizer phases: 0")
    print("Controlled variables: model, initialization, LR, batch, eval, seed")
    if missing:
        print("Artifact check skipped; missing locally:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
