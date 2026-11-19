"""CPU unit test for the DIGER hard-forward/soft-backward SID bridge."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "DiffGRM"))

from genrec.models.DIFF_GRM.cross_pse import CrossPSECodebookMixer  # noqa: E402
from genrec.models.DIFF_GRM.joint_cross_pse import JointCrossPSEAlignment  # noqa: E402


class FakeTokenizer:
    def __init__(self, items):
        self.id2item = ["<pad>", *items]
        self.item2id = {item: index for index, item in enumerate(self.id2item)}


def grad_norm(parameters) -> float:
    total = torch.zeros(())
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    return float(total.sqrt())


def main() -> None:
    torch.manual_seed(2024)
    rng = np.random.default_rng(2024)
    n_digit, codebook_size, code_dim, n_items = 4, 8, 8, 7
    items = [f"item-{index}" for index in range(n_items)]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = rng.normal(size=(n_digit, codebook_size, code_dim)).astype("float32")
        slots = rng.normal(size=(n_items, n_digit * code_dim)).astype("float32")
        slot_view = slots.reshape(n_items, n_digit, code_dim)
        distances = ((slot_view[:, :, None, :] - base[None]) ** 2).sum(-1)
        codes = distances.argmin(-1).astype("int64")

        np.save(root / "base_pse_codebooks.npy", base)
        np.save(root / "opq_residual_embeddings.npy", slots)
        np.save(root / "semantic_ids.npy", codes)
        (root / "items.json").write_text(json.dumps(items), encoding="utf-8")

        initialized = CrossPSECodebookMixer(
            torch.from_numpy(base),
            n_heads=2,
            residual_scale_init=1.0,
            learn_residual_scale=False,
            zero_init_attention_output=True,
            train_base_codebooks=True,
            mixer_enabled=True,
        )
        torch.save({"model": initialized.state_dict()}, root / "best.pt")

        config = {
            "joint_crosspse_artifact_dir": str(root),
            "joint_crosspse_items_path": str(root / "items.json"),
            "n_digit": n_digit,
            "codebook_size": codebook_size,
            "crosspse_heads": 2,
            "crosspse_residual_scale_init": 1.0,
            "crosspse_learn_residual_scale": False,
            "crosspse_zero_init_attention_output": True,
            "joint_gradient_mode": "diger_st",
            "joint_diger_assignment_temperature": 0.2,
            "joint_diger_softmax_temperature_start": 1.0,
            "joint_diger_softmax_temperature_end": 0.2,
            "joint_diger_gumbel_noise_start": 0.0,
            "joint_diger_gumbel_noise_end": 0.0,
        }
        bridge = JointCrossPSEAlignment(config, FakeTokenizer(items))
        bridge.train()
        bridge.configure_diger_schedule(10)

        item_ids = torch.tensor([1, 2, 3, 4])
        assignments, diagnostics = bridge.diger_assignments(
            item_ids, stochastic=False
        )
        assert assignments.shape == (4, n_digit, codebook_size)
        assert torch.equal(
            assignments.detach(),
            torch.nn.functional.one_hot(
                assignments.detach().argmax(-1), codebook_size
            ).float(),
        ), "forward assignments must be exactly one-hot"

        # Frozen recommender head: recommendation CE must still update bridge.
        frozen_log_probs = torch.log_softmax(
            torch.randn(4, n_digit, codebook_size), dim=-1
        ).detach()
        recommendation_loss = -(assignments * frozen_log_probs).sum(-1).mean()
        recommendation_loss.backward()
        norm = grad_norm(bridge.parameters())
        assert norm > 0, "recommendation gradient did not reach Cross-PSE"
        assert torch.isfinite(recommendation_loss)
        assert torch.isfinite(diagnostics["entropy"])
        print("DIGER hard-forward/soft-backward gradient test: OK")
        print(f"direct recommendation gradient norm: {norm:.6g}")


if __name__ == "__main__":
    main()
