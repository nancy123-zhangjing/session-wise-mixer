"""Static preflight for Cross-PSE + DIGER-ST alternating training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def merge(paths):
    config = {}
    for path in paths:
        config.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return config


def resolve_from_diff_root(value):
    return (ROOT / "external" / "DiffGRM" / value).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--recommender-checkpoint", type=Path)
    args = parser.parse_args()
    paths = [
        ROOT / "configs/shared/diff_formal_common.yaml",
        ROOT / "configs/shared/diff_formal_repo_v6.yaml",
        ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger.yaml",
    ]
    config = merge(paths)

    assert config["joint_crosspse_enabled"] is True
    assert config["joint_gradient_mode"] == "diger_st"
    assert config["joint_recommender_sia_weight"] == 0.0
    assert config["joint_tokenizer_sia_weight"] == 0.0
    assert config["joint_diger_recommendation_weight"] > 0
    assert config["n_digit"] == 4 and config["codebook_size"] == 256
    assert config["mixed_precision"] == "bf16"
    assert config["joint_cycle_epochs"] == 4
    assert config["joint_tokenizer_epochs_per_cycle"] == 1
    assert config["joint_recommender_warmup_epochs"] == 5
    assert config["joint_max_tokenizer_phases"] == 10
    assert config["joint_tokenizer_batches_per_phase"] == 96
    assert config["joint_tokenizer_batches_per_phase"] % config[
        "gradient_accumulation_steps"
    ] == 0
    assert config["evaluation_level"] == "sid"
    assert config["joint_recommender_source_run_id"] == (
        "diff_formal_crosspse_4x256_v3_catalog_bf16"
    )
    assert config["joint_diger_gumbel_noise_start"] >= (
        config["joint_diger_gumbel_noise_end"]
    )
    assert config["joint_diger_softmax_temperature_start"] >= (
        config["joint_diger_softmax_temperature_end"]
    )

    required_code = [
        ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py",
        ROOT / "tests/test_diff_crosspse_diger.py",
        ROOT / "scripts/checks/check_diff_crosspse_diger_smoke.py",
        ROOT / "configs/05_crosspse_joint/diff_diger_smoke.yaml",
        ROOT / "scripts/train/run_diff_crosspse_diger.sh",
        ROOT / "scripts/checks/find_diff_checkpoint.py",
    ]
    for path in required_code:
        assert path.is_file(), f"Missing pipeline file: {path}"

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
    if not args.allow_missing_artifacts:
        if args.recommender_checkpoint is None:
            raise ValueError("--recommender-checkpoint is required")
        if not args.recommender_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing pretrained recommender: {args.recommender_checkpoint}"
            )

    print("Cross-PSE + DIGER-ST preflight: OK")
    print(f"run_id: {config['run_id']}")
    print("layout: 4 x 256; initialization: Cross-PSE v3 + pretrained DiffGRM")
    print("alternation: 5 recommender warmup epochs, then 1 tokenizer : 3 recommender")
    print("gradient bridge: hard one-hot forward + soft/Gumbel backward")
    print("tokenizer objective: recommendation + reconstruction + balance + anchor")
    print("SIA: disabled (direct-gradient experiment)")
    if missing:
        print("Artifact check skipped; missing locally:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
