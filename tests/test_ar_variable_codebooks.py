"""Minimal functional test for position-specific AR codebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "DiffGRM"))

from genrec.models.AR_GRM.model import AR_GRM  # noqa: E402


class DummyTokenizer:
    codebook_sizes = (128, 256, 512, 256)

    def __init__(self) -> None:
        offsets = []
        current = 3
        for size in self.codebook_sizes:
            offsets.append(current)
            current += size
        self._offsets = tuple(offsets)
        self.vocab_size = current

    def codebook_offset_at(self, position: int) -> int:
        return self._offsets[position]


def main() -> None:
    tokenizer = DummyTokenizer()
    config = {
        "n_digit": 4,
        "codebook_size": 256,
        "n_embd": 32,
        "n_head": 4,
        "n_inner": 64,
        "dropout": 0.0,
        "attn_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "encoder_n_layer": 1,
        "decoder_n_layer": 1,
        "max_history_len": 3,
        "use_causal_mask": True,
        "share_decoder_output_embedding": True,
        "label_smoothing": 0.1,
    }
    model = AR_GRM(config, object(), tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_codes = torch.tensor(
        [[12, 200, 500, 7], [127, 255, 511, 255]], dtype=torch.long
    )
    token_rows = torch.stack(
        [
            raw_codes[:, position] + tokenizer.codebook_offset_at(position)
            for position in range(4)
        ],
        dim=1,
    )
    batch = {
        "history_sid": token_rows[:, None, :].repeat(1, 3, 1),
        "decoder_input_ids": raw_codes,
        "decoder_labels": raw_codes.clone(),
    }

    output = model(batch)
    assert output.position_nll_sums.shape == (4,)
    assert output.position_token_counts.tolist() == [2, 2, 2, 2]
    assert torch.isfinite(output.loss), output.loss
    assert torch.isfinite(output.position_nll_sums).all(), output.position_nll_sums

    expected_shapes = [(2, size) for size in tokenizer.codebook_sizes]
    actual_shapes = [
        tuple(model._compute_digit_logits(torch.zeros(2, 32, device=device), i).shape)
        for i in range(4)
    ]
    assert actual_shapes == expected_shapes, (actual_shapes, expected_shapes)
    print("AR variable-codebook functional test: OK")
    print("device:", device)
    print("layout:", tokenizer.codebook_sizes)
    print("offsets:", tokenizer._offsets)
    print("loss:", float(output.loss.detach().cpu()))


if __name__ == "__main__":
    main()
