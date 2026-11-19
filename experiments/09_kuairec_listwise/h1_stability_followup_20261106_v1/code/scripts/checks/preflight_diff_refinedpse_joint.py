"""Preflight for the Joint Refined-PSE DIGER matched control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "diff_formal_refinedpse_joint_diger_st_v1_slowanneal_4x256_seed2024"


def merge(paths):
    config = {}
    for path in paths:
        config.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return config


def resolve_from_diff_root(value):
    return (ROOT / "external" / "DiffGRM" / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--recommender-checkpoint", type=Path)
    args = parser.parse_args()

    common_paths = [
        ROOT / "configs/shared/diff_formal_common.yaml",
        ROOT / "configs/shared/diff_formal_repo_v6.yaml",
    ]
    cross_path = ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger_v2.yaml"
    refined_path = ROOT / "configs/05_crosspse_joint/diff_formal_refinedpse_joint_diger_v1.yaml"
    cross_config = merge([*common_paths, cross_path])
    refined_config = merge([*common_paths, refined_path])
    cross_config.setdefault("crosspse_mixer_enabled", True)
    cross_config.setdefault("crosspse_train_base_codebooks", True)

    allowed_differences = {
        "run_id",
        "reproduction_protocol",
        "protocol_note",
        "precomputed_semantic_ids_path",
        "crosspse_summary_path",
        "crosspse_mixer_enabled",
        "joint_recommender_source_run_id",
        "joint_crosspse_artifact_dir",
    }
    actual_differences = {
        key
        for key in set(cross_config) | set(refined_config)
        if cross_config.get(key) != refined_config.get(key)
    }
    assert actual_differences == allowed_differences, (
        "Joint Refined-PSE is no longer a matched method control: "
        f"expected={sorted(allowed_differences)}, "
        f"actual={sorted(actual_differences)}"
    )

    config = refined_config
    assert config["run_id"] == RUN_ID
    assert config["crosspse_mixer_enabled"] is False
    assert config["crosspse_train_base_codebooks"] is True
    assert config["joint_crosspse_enabled"] is True
    assert config["joint_gradient_mode"] == "diger_st"
    assert config["joint_recommender_source_run_id"] == (
        "diff_formal_refinedpse_4x256_v2_catalog_bf16"
    )
    assert config["n_digit"] == 4 and config["codebook_size"] == 256
    assert config["epochs"] == 45
    assert config["joint_recommender_warmup_epochs"] == 5
    assert config["joint_cycle_epochs"] == 4
    assert config["joint_tokenizer_epochs_per_cycle"] == 1
    assert config["joint_max_tokenizer_phases"] == 10
    assert config["joint_tokenizer_batches_per_phase"] == 96
    assert config["joint_codebook_lr"] == 1e-5
    assert config["joint_balance_weight"] == 0.01
    assert config["joint_anchor_weight"] == 0.01
    assert config["joint_diger_assignment_temperature"] == 0.1
    assert config["joint_diger_softmax_temperature_start"] == 1.0
    assert config["joint_diger_softmax_temperature_end"] == 1.0
    assert config["joint_diger_gumbel_noise_start"] == 1.0
    assert config["joint_diger_gumbel_noise_end"] == 0.2
    assert config["joint_diger_uncertainty_decay"] == "cosine"
    assert config["joint_recommender_sia_weight"] == 0.0
    assert config["joint_tokenizer_sia_weight"] == 0.0
    assert config["mixed_precision"] == "bf16"
    assert config["lr"] == 0.001 and config["warmup_steps"] == 500
    assert config["evaluation_level"] == "sid"
    assert config["eval_expand_sid_to_items"] is False

    required_code = [
        ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py",
        ROOT / "tests/test_diff_refinedpse_diger.py",
        ROOT / "scripts/checks/check_diff_refinedpse_joint_smoke.py",
        ROOT / "scripts/train/run_diff_refinedpse_joint.sh",
        ROOT / "scripts/checks/find_diff_checkpoint.py",
    ]
    for path in required_code:
        assert path.is_file(), f"Missing Joint Refined-PSE pipeline file: {path}"

    artifact = resolve_from_diff_root(config["joint_crosspse_artifact_dir"])
    required_artifacts = [
        artifact / "best.pt",
        artifact / "base_pse_codebooks.npy",
        artifact / "opq_residual_embeddings.npy",
        artifact / "semantic_ids.npy",
        artifact / "semantic_ids.json",
        artifact / "summary.json",
        resolve_from_diff_root(config["joint_crosspse_items_path"]),
    ]
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing and not args.allow_missing_artifacts:
        raise FileNotFoundError(
            "Missing server artifacts:\n" + "\n".join(map(str, missing))
        )
    if not missing:
        summary = json.loads(
            (artifact / "summary.json").read_text(encoding="utf-8")
        )
        assert summary.get("mixer_enabled") is False
        assert summary.get("tokenizer_type") == "refined_pse"
    if not args.allow_missing_artifacts:
        if args.recommender_checkpoint is None:
            raise ValueError("--recommender-checkpoint is required")
        if not args.recommender_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing pretrained Refined-PSE recommender: "
                f"{args.recommender_checkpoint}"
            )

    print("Joint Refined-PSE DIGER preflight: OK")
    print("mixer: disabled and excluded from optimizer")
    print("trainable tokenizer component: base codebooks only")
    print("matched to Joint Cross-PSE V2: schedule, budget, bridge, regularizers, eval")
    if missing:
        print("Artifact check skipped; missing locally:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
