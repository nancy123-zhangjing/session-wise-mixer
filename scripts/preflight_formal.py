"""Fast, non-training checks for the paired formal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def load(paths: list[Path]) -> dict:
    result: dict = {}
    for path in paths:
        result.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    common = load([root / "configs" / "ar_formal_common.yaml"])
    rqvae = load(
        [root / "configs" / "ar_formal_common.yaml", root / "configs" / "ar_formal_rqvae.yaml"]
    )
    rqkmeans = load(
        [root / "configs" / "ar_formal_common.yaml", root / "configs" / "ar_formal_rqkmeans.yaml"]
    )
    pse = load(
        [root / "configs" / "ar_formal_common.yaml", root / "configs" / "ar_formal_pse.yaml"]
    )
    rqk_ascending = load([
        root / "configs" / "ar_formal_common.yaml",
        root / "configs" / "ar_formal_rqkmeans_128_256_512.yaml",
    ])
    rqk_descending = load([
        root / "configs" / "ar_formal_common.yaml",
        root / "configs" / "ar_formal_rqkmeans_512_256_128.yaml",
    ])

    variants = {
        "rqvae_256x256x256": rqvae,
        "rqk_256x256x256": rqkmeans,
        "pse_256x256x256": pse,
        "rqk_128x256x512": rqk_ascending,
        "rqk_512x256x128": rqk_descending,
    }
    # These fields define the downstream AR experiment and must not change
    # when swapping the tokenizer or reallocating the three semantic codebooks.
    controlled_fields = (
        "reproducibility", "category", "metadata", "split", "train_sliding", "min_hist_len",
        "max_hist_len", "max_history_len", "semantic_n_digit",
        "use_collision_suffix", "require_unique_sid", "strict_item_eval",
        "collision_suffix_codebook_size", "codebook_size", "sent_emb_model",
        "sent_emb_dim", "sent_emb_batch_size", "sent_emb_fp16", "sent_emb_path",
        "sid_fit_scope", "sid_fit_val_fraction",
        "encoder_n_layer", "decoder_n_layer", "n_head", "n_embd", "n_inner",
        "dropout", "attn_pdrop", "resid_pdrop",
        "share_embeddings", "share_decoder_output_embedding", "mixed_precision",
        "gradient_accumulation_steps", "train_batch_size", "eval_batch_size",
        "lr", "weight_decay", "warmup_ratio", "warmup_steps", "epochs",
        "max_grad_norm", "label_smoothing", "eval_start_epoch", "eval_interval",
        "patience", "topk", "metrics", "val_metric", "ar_beam_search",
    )
    reference = rqkmeans
    for name, variant in variants.items():
        mismatches = {
            field: (reference.get(field), variant.get(field))
            for field in controlled_fields
            if reference.get(field) != variant.get(field)
        }
        assert not mismatches, f"Downstream controls differ for {name}: {mismatches}"

    assert rqvae["semantic_n_digit"] == rqkmeans["semantic_n_digit"] == 3
    assert rqvae["codebook_size"] == rqkmeans["codebook_size"] == 256
    assert rqvae["use_collision_suffix"] and rqkmeans["use_collision_suffix"]
    assert rqvae["sent_emb_model"] == rqkmeans["sent_emb_model"]
    assert rqvae["sent_emb_dim"] == rqkmeans["sent_emb_dim"] == 768
    assert rqvae["sid_fit_val_fraction"] == rqkmeans["sid_fit_val_fraction"]
    assert rqvae["sid_quantizer"] == "rqvae"
    assert rqkmeans["sid_quantizer"] == "rq_kmeans"
    assert rqkmeans["rq_kmeans_use_pca"] and rqkmeans["sent_emb_pca"] == 32
    assert pse["sid_quantizer"] == "opq_pq"
    assert pse["semantic_codebook_sizes"] == [256, 256, 256]
    assert pse["sent_emb_pca"] == 0
    assert pse["sent_emb_dim"] % pse["semantic_n_digit"] == 0
    assert rqk_ascending["semantic_codebook_sizes"] == [128, 256, 512]
    assert rqk_descending["semantic_codebook_sizes"] == [512, 256, 128]
    baseline_capacity = 256 ** 3
    for variant in (rqk_ascending, rqk_descending):
        sizes = variant["semantic_codebook_sizes"]
        assert sizes[0] * sizes[1] * sizes[2] == baseline_capacity
        assert variant["sid_quantizer"] == "rq_kmeans"
        assert variant["rq_kmeans_use_pca"]
        for field in (
            "sent_emb_pca", "normalize_after_pca", "pca_whiten",
            "rq_kmeans_use_pca", "rq_kmeans_niters",
        ):
            assert variant[field] == rqkmeans[field], (
                f"RQ-KMeans tokenizer control differs at {field}: "
                f"baseline={rqkmeans[field]!r}, variant={variant[field]!r}"
            )
    assert common["n_embd"] % common["n_head"] == 0
    assert int(common["gradient_accumulation_steps"]) >= 1

    dependencies = {}
    for module_name in (
        "accelerate", "datasets", "einops", "faiss", "gin",
        "sentence_transformers", "wandb",
    ):
        try:
            module = __import__(module_name)
            dependencies[module_name] = getattr(module, "__version__", "OK")
        except Exception as error:  # pragma: no cover - intended for remote diagnosis
            dependencies[module_name] = f"ERROR: {error!r}"

    result = {
        "protocol": (
            "AR internal controls: PSE 3x256 and RQ-KMeans "
            "128x256x512 / 512x256x128"
        ),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "effective_batch": common["train_batch_size"] * common["gradient_accumulation_steps"],
        "controlled_fields_verified": list(controlled_fields),
        "intentional_differences": {
            "pse_input": "raw sentence-T5-XL 768d; OPQ/PQ requires a dimension divisible by 3",
            "rqkmeans_input": "shared sentence-T5-XL followed by the same fitted PCA32 protocol",
            "semantic_codebook_layouts": {
                name: variant["semantic_codebook_sizes"]
                for name, variant in variants.items()
            },
            "model_vocab_size": {
                name: 3 + sum(variant["semantic_codebook_sizes"])
                + variant["collision_suffix_codebook_size"]
                for name, variant in variants.items()
            },
        },
        "dependencies": dependencies,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not torch.cuda.is_available() or any(str(value).startswith("ERROR") for value in dependencies.values()):
        raise SystemExit("Formal preflight failed")
    print("Formal preflight: OK")


if __name__ == "__main__":
    main()
