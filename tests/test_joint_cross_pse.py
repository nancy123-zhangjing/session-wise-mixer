"""CPU test for alternating Cross-PSE SIA gradient routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CROSS_PATH = ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/cross_pse.py"
spec = importlib.util.spec_from_file_location("cross_pse_joint_test", CROSS_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
CrossPSECodebookMixer = module.CrossPSECodebookMixer


def symmetric_kl(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left_log = left_logits.log_softmax(-1)
    right_log = right_logits.log_softmax(-1)
    return 0.5 * (
        (left_log.exp() * (left_log - right_log)).sum(-1)
        + (right_log.exp() * (right_log - left_log)).sum(-1)
    ).mean()


def logits(mixer, slots, temperature=0.2):
    codebooks, _ = mixer.crossed_codebooks(False)
    return -mixer.squared_distances(slots, codebooks) / temperature


def clear(module):
    for parameter in module.parameters():
        parameter.grad = None


def grad_sum(parameters):
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in parameters
        if parameter.grad is not None
    )


def main() -> None:
    torch.manual_seed(19)
    batch, n_digit, codebook_size, code_dim = 7, 4, 16, 8
    mixer = CrossPSECodebookMixer(
        torch.randn(n_digit, codebook_size, code_dim),
        n_heads=2,
        residual_scale_init=1.0,
        learn_residual_scale=False,
        train_base_codebooks=True,
        mixer_enabled=True,
    )
    adapter = nn.Linear(n_digit * code_dim, n_digit * code_dim)
    sequence = torch.randn(batch, n_digit * code_dim)
    targets = torch.randn(batch, n_digit, code_dim)

    # Recommender phase: frozen tokenizer, gradients reach only the adapter.
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)
    sequence_slots = adapter(sequence).view(batch, n_digit, code_dim)
    rec_sia = symmetric_kl(logits(mixer, sequence_slots), logits(mixer, targets))
    rec_sia.backward()
    assert grad_sum(adapter.parameters()) > 0
    assert grad_sum(mixer.parameters()) == 0

    # Tokenizer phase: frozen recommender, gradients reach only Cross-PSE.
    clear(adapter)
    clear(mixer)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in mixer.parameters():
        parameter.requires_grad_(True)
    frozen_sequence_slots = adapter(sequence).detach().view(batch, n_digit, code_dim)
    id_sia = symmetric_kl(logits(mixer, frozen_sequence_slots), logits(mixer, targets))
    id_sia.backward()
    assert grad_sum(adapter.parameters()) == 0
    assert grad_sum(mixer.parameters()) > 0
    assert torch.isfinite(rec_sia) and torch.isfinite(id_sia)
    print("Joint Cross-PSE alternating gradient test: OK")


if __name__ == "__main__":
    main()
