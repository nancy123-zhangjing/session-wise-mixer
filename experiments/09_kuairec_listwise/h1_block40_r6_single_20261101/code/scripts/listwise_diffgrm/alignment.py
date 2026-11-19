"""Construct the existing alternating PSE bridge from new dataset assets."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from .assets import PSEAssets


def build_alignment(assets: PSEAssets, mixer_enabled: bool, *, total_t_updates: int = 400,
                    config_overrides: dict | None = None):
    from .runtime import engine_module

    items = [str(value) for value in assets.original_item_ids[1:].tolist()]
    tokenizer = SimpleNamespace(id2item=["[PAD]", *items],
                                item2id={value: index + 1 for index, value in enumerate(items)})
    config = {
        "joint_crosspse_artifact_dir": str(assets.root),
        "joint_crosspse_checkpoint": str(assets.root / "initial_mixer.pt"),
        "joint_crosspse_items_path": str(assets.root / "items.json"),
        "joint_crosspse_slots_path": str(assets.root / "opq_residual_embeddings.npy"),
        "n_digit": assets.n_digit, "codebook_size": assets.codebook_size,
        "crosspse_mixer_enabled": bool(mixer_enabled), "crosspse_heads": 4,
        "crosspse_dropout": 0.0, "crosspse_attention_scope": "full",
        "crosspse_residual_scale_init": 1.0, "crosspse_learn_residual_scale": False,
        "crosspse_zero_init_attention_output": True, "crosspse_train_base_codebooks": True,
        "joint_gradient_mode": "diger_st", "joint_sia_temperature": 0.1,
        "joint_diger_assignment_temperature": 0.1,
        "joint_diger_softmax_temperature_start": 1.0, "joint_diger_softmax_temperature_end": 1.0,
        "joint_diger_gumbel_noise_start": 1.0, "joint_diger_gumbel_noise_end": 0.2,
        "joint_diger_uncertainty_decay": "cosine",
    }
    # Asset identity, hard-forward protocol and identity Mixer cannot be silently overridden.
    mutable = {key for key in config if key.startswith("joint_diger_")}
    for key, value in (config_overrides or {}).items():
        if key not in mutable:
            if key in config and value == config[key]:
                continue
            raise ValueError(f"Unsupported or conflicting alignment override: {key}")
        config[key] = value
    alignment = engine_module("joint_cross_pse").JointCrossPSEAlignment(config, tokenizer)
    alignment.configure_diger_schedule(total_t_updates)
    with torch.no_grad():
        effective, _ = alignment.mixer.crossed_codebooks(False)
        if (not torch.equal(effective.cpu(), assets.centroids)
                or not torch.equal(alignment.mixer.initial_codebooks.cpu(), assets.centroids)
                or not torch.equal(alignment.item_slots.cpu(), assets.residual_features)
                or not torch.equal(alignment.hard_codes.cpu(), assets.catalog_sids)):
            raise ValueError("Initial alignment does not match E0/anchor/SID asset")
    return alignment
