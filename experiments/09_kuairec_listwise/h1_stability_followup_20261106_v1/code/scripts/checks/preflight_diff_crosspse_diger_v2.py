"""Static preflight for the controlled DIGER-ST V2 annealing run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "diff_formal_crosspse_joint_diger_st_v2_slowanneal_4x256_seed2024"


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
        ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger_v2.yaml",
    ]
    config = merge(paths)

    v1_overlay = yaml.safe_load(
        (ROOT / "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger.yaml").read_text(
            encoding="utf-8"
        )
    )
    v2_overlay = yaml.safe_load(paths[-1].read_text(encoding="utf-8"))
    allowed_v2_differences = {
        "run_id",
        "reproduction_protocol",
        "protocol_note",
        "joint_diger_softmax_temperature_end",
        "joint_diger_gumbel_noise_end",
    }
    actual_v2_differences = {
        key
        for key in set(v1_overlay) | set(v2_overlay)
        if v1_overlay.get(key) != v2_overlay.get(key)
    }
    assert actual_v2_differences == allowed_v2_differences, (
        "V2 is no longer a controlled schedule-only ablation: "
        f"expected changed keys {sorted(allowed_v2_differences)}, "
        f"got {sorted(actual_v2_differences)}"
    )

    assert config["run_id"] == RUN_ID
    assert config["reproduction_protocol"] == (
        "alternating_crosspse_diffgrm_diger_st_v2_slowanneal"
    )
    assert config["joint_crosspse_enabled"] is True
    assert config["joint_gradient_mode"] == "diger_st"
    assert config["joint_recommender_sia_weight"] == 0.0
    assert config["joint_tokenizer_sia_weight"] == 0.0
    assert config["joint_diger_recommendation_weight"] == 1.0
    assert config["n_digit"] == 4 and config["codebook_size"] == 256
    assert config["mixed_precision"] == "bf16"

    # These are controlled variables inherited from the completed V1 run.
    assert config["epochs"] == 45
    assert config["joint_cycle_epochs"] == 4
    assert config["joint_tokenizer_epochs_per_cycle"] == 1
    assert config["joint_recommender_warmup_epochs"] == 5
    assert config["joint_max_tokenizer_phases"] == 10
    assert config["joint_tokenizer_batches_per_phase"] == 96
    assert config["joint_tokenizer_batches_per_phase"] % config[
        "gradient_accumulation_steps"
    ] == 0
    assert config["joint_codebook_lr"] == 1e-5
    assert config["joint_mixer_lr"] == 1e-4
    assert config["joint_balance_weight"] == 0.01
    assert config["joint_anchor_weight"] == 0.01
    assert config["evaluation_level"] == "sid"
    assert config["eval_expand_sid_to_items"] is False
    assert config["joint_recommender_source_run_id"] == (
        "diff_formal_crosspse_4x256_v3_catalog_bf16"
    )

    # V2 changes only these two schedule endpoints relative to V1.
    assert config["joint_diger_assignment_temperature"] == 0.1
    assert config["joint_diger_softmax_temperature_start"] == 1.0
    assert config["joint_diger_softmax_temperature_end"] == 1.0
    assert config["joint_diger_gumbel_noise_start"] == 1.0
    assert config["joint_diger_gumbel_noise_end"] == 0.2
    assert config["joint_diger_uncertainty_decay"] == "cosine"

    required_code = [
        ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py",
        ROOT / "tests/test_diff_crosspse_diger.py",
        ROOT / "scripts/checks/check_diff_crosspse_diger_smoke.py",
        ROOT / "configs/05_crosspse_joint/diff_diger_smoke.yaml",
        ROOT / "scripts/train/run_diff_crosspse_diger_v2.sh",
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

    print("Cross-PSE + DIGER-ST V2 preflight: OK")
    print(f"run_id: {config['run_id']}")
    print("controlled change: softmax temperature 1.0 -> 1.0")
    print("controlled change: Gumbel noise 1.0 -> 0.2 (cosine)")
    print("unchanged: initialization, alternation, optimizer, regularizers, eval")
    print("V1/V2 overlay diff guard: PASS (exactly five documented keys differ)")
    if missing:
        print("Artifact check skipped; missing locally:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
