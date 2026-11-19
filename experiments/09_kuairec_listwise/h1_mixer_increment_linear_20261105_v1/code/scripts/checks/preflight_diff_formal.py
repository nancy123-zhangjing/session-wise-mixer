"""Fast environment/config checks for the formal DiffGRM pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

import torch
import yaml


def merge_yaml(paths: list[Path]) -> dict:
    config: dict = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        config.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--tokenizer",
        choices=(
            "pse", "rq_kmeans", "rqvae", "cross_pse", "cross_pse_v2",
            "refined_pse", "cross_pse_v3", "refined_pse_catalog",
            "cross_pse_v3_catalog",
        ),
        default="pse",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    common_path = root / "configs" / "shared" / "diff_formal_common.yaml"
    pse_path = root / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_pse.yaml"
    formal_path = root / "configs" / "shared" / "diff_formal_repo_v6.yaml"
    rqk_path = root / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_rqkmeans_v6.yaml"
    rqvae_path = root / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_rqvae_v6.yaml"
    crosspse_path = root / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v1.yaml"
    crosspse_v2_path = root / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v2.yaml"
    refinedpse_path = root / "configs" / "04_crosspse_offline" / "diff_formal_refinedpse_v1.yaml"
    crosspse_v3_path = root / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v3.yaml"
    refinedpse_catalog_path = (
        root / "configs" / "04_crosspse_offline" / "diff_formal_refinedpse_v2_catalog.yaml"
    )
    crosspse_v3_catalog_path = (
        root / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v3_catalog.yaml"
    )
    if args.tokenizer == "pse":
        config = merge_yaml([common_path, pse_path, formal_path])
    elif args.tokenizer == "rq_kmeans":
        config = merge_yaml([common_path, formal_path, rqk_path])
    elif args.tokenizer == "rqvae":
        config = merge_yaml([common_path, formal_path, rqvae_path])
    elif args.tokenizer == "cross_pse":
        config = merge_yaml([common_path, formal_path, crosspse_path])
    elif args.tokenizer == "cross_pse_v2":
        config = merge_yaml([common_path, formal_path, crosspse_v2_path])
    elif args.tokenizer == "refined_pse":
        config = merge_yaml([common_path, formal_path, refinedpse_path])
    elif args.tokenizer == "cross_pse_v3":
        config = merge_yaml([common_path, formal_path, crosspse_v3_path])
    elif args.tokenizer == "refined_pse_catalog":
        config = merge_yaml([common_path, formal_path, refinedpse_catalog_path])
    else:
        config = merge_yaml([common_path, formal_path, crosspse_v3_catalog_path])

    assert config["category"] == "Beauty"
    assert config["n_digit"] == 4
    assert config["codebook_size"] == 256
    if args.tokenizer == "pse":
        assert config["sid_quantizer"] == "opq_pq"
        assert config["run_id"] == "diff_formal_pse_4x256_repo_repro_v6_bf16"
        assert config["reproduction_protocol"] == "author_repository_beauty_warmup_safe_early_stop"
    elif args.tokenizer == "rq_kmeans":
        assert config["sid_quantizer"] == "rq_kmeans"
        assert config["run_id"] == "diff_formal_rqkmeans_4x256_v6_bf16"
        assert config["reproduction_protocol"] == "controlled_tokenizer_swap_rq_kmeans_v6"
        assert config["rq_kmeans_seed"] == 2024
        assert config["rq_kmeans_niters"] == 20
        assert config["rq_kmeans_use_pca"] is True
        assert config["force_regenerate_opq"] is False
    elif args.tokenizer == "rqvae":
        assert config["sid_quantizer"] == "rqvae"
        assert config["run_id"] == "diff_formal_rqvae_4x256_v6_bf16"
        assert config["reproduction_protocol"] == "controlled_tokenizer_swap_rqvae_v6"
        assert config["semantic_n_digit"] == 4
        assert config["rqvae_seed"] == 2024
        assert config["rqvae_input_dim"] == 256
        assert config["rqvae_embed_dim"] == 32
        assert config["rqvae_max_steps"] == 60000
        assert config["rqvae_mixed_precision"] == "bf16"
        assert config["force_regenerate_opq"] is False
    elif args.tokenizer == "cross_pse":
        assert config["sid_quantizer"] == "cross_pse"
        assert config["run_id"] == "diff_formal_crosspse_4x256_v1_bf16"
        assert config["reproduction_protocol"] == "controlled_cross_codebook_pse_v1"
        assert config["semantic_n_digit"] == 4
        assert config["crosspse_seed"] == 2024
        assert config["crosspse_attention_scope"] == "full"
        assert config["crosspse_heads"] == 4
        assert config["crosspse_train_base_codebooks"] is False
        assert config["crosspse_mixed_precision"] == "bf16"
        assert config["force_regenerate_opq"] is False
    elif args.tokenizer == "cross_pse_v2":
        assert config["sid_quantizer"] == "cross_pse"
        assert config["run_id"] == "diff_formal_crosspse_4x256_v2_bf16"
        assert config["reproduction_protocol"] == "controlled_cross_codebook_pse_v2"
        assert config["semantic_n_digit"] == 4
        assert config["crosspse_seed"] == 2024
        assert config["crosspse_attention_scope"] == "full"
        assert config["crosspse_heads"] == 4
        assert config["crosspse_train_base_codebooks"] is False
        assert config["crosspse_residual_scale_init"] == 1.0
        assert config["crosspse_learn_residual_scale"] is False
        assert config["crosspse_zero_init_attention_output"] is True
        assert config["crosspse_validation_fraction"] == 0.1
        assert config["crosspse_split_seed"] == 2024
        assert config["crosspse_anchor_weight"] == 0.01
        assert config["crosspse_mixed_precision"] == "bf16"
        assert config["force_regenerate_opq"] is False
    else:
        expected = {
            "refined_pse": (
                "diff_formal_refinedpse_4x256_v1_bf16",
                "controlled_trainable_pse_no_cross_v1",
                False,
            ),
            "cross_pse_v3": (
                "diff_formal_crosspse_4x256_v3_bf16",
                "controlled_trainable_cross_codebook_pse_v3",
                True,
            ),
            "refined_pse_catalog": (
                "diff_formal_refinedpse_4x256_v2_catalog_bf16",
                "controlled_trainable_pse_catalog_selection_v2",
                False,
            ),
            "cross_pse_v3_catalog": (
                "diff_formal_crosspse_4x256_v3_catalog_bf16",
                "controlled_trainable_cross_codebook_pse_v3_catalog",
                True,
            ),
        }[args.tokenizer]
        assert config["sid_quantizer"] == "cross_pse"
        assert config["run_id"] == expected[0]
        assert config["reproduction_protocol"] == expected[1]
        assert config["semantic_n_digit"] == 4
        assert config["crosspse_seed"] == 2024
        assert config["crosspse_attention_scope"] == "full"
        assert config["crosspse_heads"] == 4
        assert config["crosspse_train_base_codebooks"] is True
        assert config["crosspse_mixer_enabled"] is expected[2]
        assert config["crosspse_residual_scale_init"] == 1.0
        assert config["crosspse_learn_residual_scale"] is False
        assert config["crosspse_zero_init_attention_output"] is True
        is_catalog_protocol = args.tokenizer in (
            "refined_pse_catalog", "cross_pse_v3_catalog"
        )
        assert config["crosspse_validation_fraction"] == (
            0.0 if is_catalog_protocol else 0.1
        )
        assert config["crosspse_split_seed"] == 2024
        assert config["crosspse_codebook_lr"] == (
            0.00001 if is_catalog_protocol else 0.0001
        )
        assert config["crosspse_lr"] == (
            0.0001 if is_catalog_protocol else 0.001
        )
        if is_catalog_protocol:
            assert config["crosspse_selection_protocol"] == (
                "fit_catalog_reconstruction"
            )
            assert config["crosspse_batch_size"] == 2048
            assert config["crosspse_eval_batch_size"] == 2048
            assert config["crosspse_max_steps"] == 3000
            assert config["crosspse_eval_interval"] == 50
        assert config["crosspse_anchor_weight"] == 0.01
        assert config["crosspse_mixed_precision"] == "bf16"
        assert config["force_regenerate_opq"] is False
    assert config["masking_strategy"] == "guided"
    assert config["guided_select"] == "least"
    assert not config["guided_refresh_each_step"]
    assert config["encoder_n_layer"] == 1
    assert config["decoder_n_layer"] == 4
    assert config["n_head"] == 4
    assert config["n_embd"] == 256
    assert config["n_inner"] == 1024
    assert config["diffusion_loss_reduction"] == "per_view"
    assert config["vectorized_beam_search"]["test"]["beam_act"] == 256
    assert config["beam_search_modes"] == ["confidence"]
    assert config["evaluation_level"] == "sid"
    assert config["mixed_precision"] == "bf16"
    assert config["numerical_guard"] is True
    assert config["numerical_guard_parameter_interval"] == 100
    assert config["train_batch_size"] == 64
    assert config["gradient_accumulation_steps"] == 16
    assert config["train_batch_size"] * config["gradient_accumulation_steps"] == 1024
    assert config["lr"] == 0.01
    assert config["warmup_steps"] == 10000
    assert config["epochs"] == 200
    assert config["label_smoothing"] == 0.2
    assert config["paper_repo_discrepancy"] == {
        "paper_max_epochs": 100,
        "repository_effective_epochs": 200,
        "paper_label_smoothing": 0.1,
        "repository_beauty_label_smoothing": 0.2,
        "repository_default_eval_interval": 1,
        "single_gpu_eval_interval": 5,
    }
    assert config["eval_start_epoch"] == 20
    assert config["eval_interval"] == 5
    assert config["patience"] == 5
    assert config["early_stop_after_warmup"] is True

    model_path = root / "external" / "DiffGRM" / "genrec" / "models" / "DIFF_GRM" / "model.py"
    trainer_path = root / "external" / "DiffGRM" / "genrec" / "models" / "DIFF_GRM" / "trainer.py"
    tokenizer_path = root / "external" / "DiffGRM" / "genrec" / "models" / "DIFF_GRM" / "tokenizer.py"
    model_text = model_path.read_text(encoding="utf-8")
    trainer_text = trainer_path.read_text(encoding="utf-8")
    tokenizer_text = tokenizer_path.read_text(encoding="utf-8")
    for marker in (
        "position_nll_sums",
        "noising_order_counts",
        "full_mask_diagnostics",
        "greedy_decode_diagnostics",
        "capture_attention",
    ):
        assert marker in model_text, f"Missing model diagnostic marker: {marker}"
    for marker in (
        "position_metrics_csv",
        "order_metrics_csv",
        "training_metrics_csv",
        "evaluation_metrics_csv",
        "cpd_greedy_decode_easy_to_hard",
        "_check_finite_tensor",
        "_check_model_parameters",
        "early_stop_after_warmup",
    ):
        assert marker in trainer_text, f"Missing trainer diagnostic marker: {marker}"
    for marker in (
        "rq_kmeans_use_pca",
        "precomputed_semantic_ids_path",
        "semantic_ids_sha256",
        "SID_CODEBOOK_STATS",
    ):
        assert marker in tokenizer_text, f"Missing tokenizer marker: {marker}"
    for required in (
        root / "scripts" / "legacy" / "archive_diff_pilot.py",
        root / "scripts" / "analysis" / "analyze_diff_cpd_attention.py",
        root / "scripts" / "analysis" / "summarize_diff_formal.py",
        root / "configs" / "shared" / "diff_formal_repo_v6.yaml",
        root / "docs" / "methods" / "diffgrm_repo_reproduction_v6.md",
    ):
        assert required.is_file(), f"Missing pipeline script: {required}"
    if args.tokenizer == "rq_kmeans":
        for required in (
            rqk_path,
            root / "scripts" / "train" / "run_diff_rqkmeans_v6.sh",
            root / "scripts" / "analysis" / "compare_diff_tokenizers.py",
            root / "docs" / "methods" / "diffgrm_rqkmeans_v6.md",
        ):
            assert required.is_file(), f"Missing RQ-KMeans pipeline file: {required}"
    if args.tokenizer == "rqvae":
        for required in (
            rqvae_path,
            root / "scripts" / "train" / "run_diff_rqvae_v6.sh",
            root / "scripts" / "prepare" / "prepare_diff_rqvae_inputs.py",
            root / "scripts" / "tokenizers" / "train_rqvae_tokenizer.py",
            root / "scripts" / "checks" / "verify_diff_rqvae_artifact.py",
            root / "scripts" / "analysis" / "analyze_tokenizer_geometry.py",
            root / "docs" / "methods" / "diffgrm_rqvae_v6.md",
        ):
            assert required.is_file(), f"Missing RQ-VAE pipeline file: {required}"
    cross_variants = {
        "cross_pse": (crosspse_path, "run_diff_crosspse_v1.sh"),
        "cross_pse_v2": (crosspse_v2_path, "run_diff_crosspse_v2.sh"),
        "refined_pse": (refinedpse_path, "run_diff_refinedpse_v1.sh"),
        "cross_pse_v3": (crosspse_v3_path, "run_diff_crosspse_v3.sh"),
        "refined_pse_catalog": (
            refinedpse_catalog_path, "run_diff_refinedpse_v2_catalog.sh"
        ),
        "cross_pse_v3_catalog": (
            crosspse_v3_catalog_path, "run_diff_crosspse_v3_catalog.sh"
        ),
    }
    if args.tokenizer in cross_variants:
        selected_crosspse_path, runner_name = cross_variants[args.tokenizer]
        selected_crosspse_runner = root / "scripts" / "train" / runner_name
        for required in (
            selected_crosspse_path,
            selected_crosspse_runner,
            root / "scripts" / "prepare" / "prepare_diff_crosspse_fit_mask.py",
            root / "scripts" / "tokenizers" / "train_cross_pse_tokenizer.py",
            root / "scripts" / "checks" / "verify_diff_crosspse_artifact.py",
            root / "tests" / "test_cross_pse_module.py",
            root / "scripts" / "tokenizers" / "inspect_crosspse_tokenizer.py",
            root / "scripts" / "analysis" / "analyze_tokenizer_geometry.py",
            root / "scripts" / "analysis" / "compare_diff_tokenizers.py",
            root / "external" / "DiffGRM" / "genrec" / "models"
            / "DIFF_GRM" / "cross_pse.py",
        ):
            assert required.is_file(), f"Missing Cross-PSE pipeline file: {required}"
        if args.tokenizer in (
            "refined_pse", "cross_pse_v3",
            "refined_pse_catalog", "cross_pse_v3_catalog",
        ):
            for required in (
                root / "scripts" / "train" / "run_diff_pse_refinement.sh",
                root / "scripts" / "analysis" / "compare_pse_refinement_tokenizers.py",
                root / "docs" / "methods" / "refinedpse_crosspse_v3.md",
            ):
                assert required.is_file(), f"Missing refinement pipeline file: {required}"

    default_path = root / "external" / "DiffGRM" / "genrec" / "default.yaml"
    dataset_path = (
        root / "external" / "DiffGRM" / "genrec" / "datasets"
        / "AmazonReviews2014" / "config.yaml"
    )
    model_config_path = (
        root / "external" / "DiffGRM" / "genrec" / "models"
        / "DIFF_GRM" / "config.yaml"
    )
    pse_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, pse_path, formal_path,
    ])
    rqk_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, rqk_path,
    ])
    rqvae_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, rqvae_path,
    ])
    crosspse_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, crosspse_path,
    ])
    crosspse_v2_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, crosspse_v2_path,
    ])
    refinedpse_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, refinedpse_path,
    ])
    crosspse_v3_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, crosspse_v3_path,
    ])
    refinedpse_catalog_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, refinedpse_catalog_path,
    ])
    crosspse_v3_catalog_full = merge_yaml([
        default_path, dataset_path, model_config_path,
        common_path, formal_path, crosspse_v3_catalog_path,
    ])
    selected_full = {
        "pse": pse_full,
        "rq_kmeans": rqk_full,
        "rqvae": rqvae_full,
        "cross_pse": crosspse_full,
        "cross_pse_v2": crosspse_v2_full,
        "refined_pse": refinedpse_full,
        "cross_pse_v3": crosspse_v3_full,
        "refined_pse_catalog": refinedpse_catalog_full,
        "cross_pse_v3_catalog": crosspse_v3_catalog_full,
    }[args.tokenizer]
    changed_keys = sorted(
        key for key in set(pse_full) | set(selected_full)
        if pse_full.get(key) != selected_full.get(key)
    )
    allowed_tokenizer_differences = {
        "run_id",
        "reproduction_protocol",
        "protocol_note",
        "sid_quantizer",
        "rq_kmeans_seed",
        "rq_kmeans_use_pca",
        "semantic_n_digit",
        "precomputed_semantic_ids_path",
        "rqvae_summary_path",
        "rqvae_seed",
        "rqvae_input_dim",
        "rqvae_hidden_dims",
        "rqvae_embed_dim",
        "rqvae_commitment_weight",
        "rqvae_codebook_normalize",
        "rqvae_batch_size",
        "rqvae_eval_batch_size",
        "rqvae_lr",
        "rqvae_weight_decay",
        "rqvae_warmup_ratio",
        "rqvae_max_steps",
        "rqvae_eval_interval",
        "rqvae_profile_steps",
        "rqvae_mixed_precision",
        "rqvae_fp16",
        "rqvae_min_code_utilization",
        "rqvae_min_unique_sid_ratio",
        "tokenizer_neighbor_queries",
        "sid_fit_val_fraction",
        "crosspse_seed",
        "crosspse_summary_path",
        "crosspse_attention_scope",
        "crosspse_heads",
        "crosspse_dropout",
        "crosspse_residual_scale_init",
        "crosspse_learn_residual_scale",
        "crosspse_zero_init_attention_output",
        "crosspse_train_base_codebooks",
        "crosspse_mixer_enabled",
        "crosspse_selection_protocol",
        "crosspse_validation_fraction",
        "crosspse_split_seed",
        "crosspse_batch_size",
        "crosspse_eval_batch_size",
        "crosspse_lr",
        "crosspse_codebook_lr",
        "crosspse_weight_decay",
        "crosspse_warmup_ratio",
        "crosspse_max_steps",
        "crosspse_eval_interval",
        "crosspse_patience",
        "crosspse_min_delta",
        "crosspse_profile_steps",
        "crosspse_mixed_precision",
        "crosspse_balance_weight",
        "crosspse_balance_temperature",
        "crosspse_anchor_weight",
    }
    unexpected = sorted(set(changed_keys) - allowed_tokenizer_differences)
    assert not unexpected, (
        "PSE/selected configs differ outside the tokenizer protocol: "
        f"{unexpected}"
    )

    dependencies = {}
    for module_name in (
        "accelerate",
        "datasets",
        "faiss",
        "gin",
        "einops",
        "numpy",
        "sentence_transformers",
        "transformers",
        "yaml",
    ):
        try:
            module = importlib.import_module(module_name)
            dependencies[module_name] = getattr(module, "__version__", "OK")
        except Exception as error:  # pragma: no cover - remote diagnosis
            dependencies[module_name] = f"ERROR: {error!r}"

    disk = shutil.disk_usage(root)
    result = {
        "protocol": f"DiffGRM {args.tokenizer}/OCN/CPD Beauty 4x256",
        "run_id": config["run_id"],
        "sid_quantizer": config["sid_quantizer"],
        "rq_kmeans": (
            {
                "tokenizer_seed": config["rq_kmeans_seed"],
                "niters": config["rq_kmeans_niters"],
                "shared_pca256_input": config["rq_kmeans_use_pca"],
            }
            if args.tokenizer == "rq_kmeans" else None
        ),
        "rqvae": (
            {
                "tokenizer_seed": config["rqvae_seed"],
                "input_dim": config["rqvae_input_dim"],
                "latent_dim": config["rqvae_embed_dim"],
                "max_steps": config["rqvae_max_steps"],
                "mixed_precision": config["rqvae_mixed_precision"],
            }
            if args.tokenizer == "rqvae" else None
        ),
        "cross_pse": (
            {
                "tokenizer_seed": config["crosspse_seed"],
                "attention_scope": config["crosspse_attention_scope"],
                "heads": config["crosspse_heads"],
                "base_codebooks_trainable": config[
                    "crosspse_train_base_codebooks"
                ],
                "max_steps": config["crosspse_max_steps"],
                "residual_scale": config["crosspse_residual_scale_init"],
                "learn_residual_scale": config.get(
                    "crosspse_learn_residual_scale", True
                ),
                "zero_init_attention_output": config.get(
                    "crosspse_zero_init_attention_output", False
                ),
                "validation_fraction": config.get(
                    "crosspse_validation_fraction", 0.0
                ),
            }
            if args.tokenizer in cross_variants else None
        ),
        "pse_vs_selected_changed_keys": changed_keys,
        "unexpected_control_differences": unexpected,
        "paper_alignment": {
            "semantic_encoder": config["sent_emb_model"],
            "architecture": "encoder1-decoder4-hidden256-heads4-ffn1024",
            "label_smoothing": config["label_smoothing"],
            "test_cpd_beam": config["vectorized_beam_search"]["test"]["beam_act"],
            "eval_interval": config["eval_interval"],
            "mixed_precision": config["mixed_precision"],
        },
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "physical_batch": config["train_batch_size"],
        "gradient_accumulation": config["gradient_accumulation_steps"],
        "effective_base_batch": (
            config["train_batch_size"] * config["gradient_accumulation_steps"]
        ),
        "reproduction_protocol": config["reproduction_protocol"],
        "official_beauty_base_batch": 1024,
        "expected_full_updates_per_epoch": 104,
        "expected_total_optimizer_steps": 20800,
        "expected_warmup_end_epoch": round(10000 / 104, 2),
        "expected_cosine_decay_epochs": round(200 - 10000 / 104, 2),
        "expected_max_validation_evaluations": 37,
        "early_stop_no_improvement_epochs": (
            config["eval_interval"] * config["patience"]
        ),
        "early_stop_after_warmup": config["early_stop_after_warmup"],
        "expected_first_early_stop_epoch": 120,
        "paper_repo_discrepancy": config["paper_repo_discrepancy"],
        "numerical_guard": config["numerical_guard"],
        "ocn_views_per_example": config["n_digit"],
        "free_disk_gb": round(disk.free / 1024**3, 2),
        "dependencies": dependencies,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    failures = [
        name for name, version in dependencies.items() if str(version).startswith("ERROR")
    ]
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or failures
    ):
        raise SystemExit(f"DiffGRM preflight failed; dependency failures={failures}")
    print("DiffGRM formal preflight: OK")


if __name__ == "__main__":
    main()
