"""RecTools PreLN SASRec and the eSASRec author's LiGR blocks, adapted for
one-next-item supervision. See experiments/06_continuous_baseline/SOURCES.md.

Block equations adapted from Apache-2.0 sources, Copyright 2025 MTS and
transformer_benchmark contributors. Changes: shared standalone item table,
last-position output, PAD-free logits and merged boolean attention masks.
"""
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


CUDA_ATTENTION_POLICY = "fp32_math"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    n_items: int = 12101
    max_length: int = 50
    dim: int = 64
    layers: int = 1
    heads: int = 1
    dropout: float = 0.2
    ffn_multiplier: int = 4
    negatives: int = 256

    def __post_init__(self):
        if self.model not in {"sasrec_fullce", "esasrec"}:
            raise ValueError("Unknown model")
        if min(self.n_items, self.max_length, self.dim, self.layers, self.heads) < 1:
            raise ValueError("Positive dimensions required")
        if self.dim % self.heads or not 0 <= self.dropout < 1 or self.negatives < 1:
            raise ValueError("Invalid heads, dropout or negatives")

    def to_dict(self):
        return asdict(self)


class FeedForward(nn.Module):
    def __init__(self, dim, multiplier, dropout, gated):
        super().__init__()
        self.ff_linear_1 = nn.Linear(dim, dim * multiplier, bias=not gated)
        self.ff_dropout_1 = nn.Dropout(dropout)
        self.ff_activation = nn.SiLU() if gated else nn.GELU()
        self.ff_linear_2 = nn.Linear(dim * multiplier, dim, bias=not gated)
        self.gated = gated
        if gated:
            self.ff_linear_3 = nn.Linear(dim, dim * multiplier, bias=False)

    def forward(self, seqs):
        output = self.ff_activation(self.ff_linear_1(seqs))
        if self.gated:
            output = output * self.ff_linear_3(seqs)
        return self.ff_linear_2(self.ff_dropout_1(output))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gated = config.model == "esasrec"
        self.multi_head_attn = nn.MultiheadAttention(
            config.dim, config.heads, config.dropout, batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(config.dim)
        self.dropout_1 = nn.Dropout(config.dropout)
        self.layer_norm_2 = nn.LayerNorm(config.dim)
        self.feed_forward = FeedForward(
            config.dim, config.ffn_multiplier, config.dropout, self.gated
        )
        self.dropout_2 = nn.Dropout(config.dropout)
        self.dropout_3 = nn.Dropout(config.dropout)
        if self.gated:
            self.gating_linear_1 = nn.Linear(config.dim, config.dim)
            self.gating_linear_2 = nn.Linear(config.dim, config.dim)

    def forward(self, seqs, attn_mask, key_padding_mask=None):
        normalized = self.layer_norm_1(seqs)
        if normalized.is_cuda:
            # The default BF16 efficient-attention backward produced enormous
            # incorrect gradients on a trained 512d/2-head checkpoint. Keep
            # projections, attention and its backward in FP32 Math on CUDA.
            # CPU/native double remains available for author-equation checks.
            with torch.autocast("cuda", enabled=False), sdpa_kernel([SDPBackend.MATH]):
                inputs = normalized.float()
                attended, _ = self.multi_head_attn(
                    inputs, inputs, inputs, attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask, need_weights=False
                )
        else:
            attended, _ = self.multi_head_attn(
                normalized, normalized, normalized, attn_mask=attn_mask,
                key_padding_mask=key_padding_mask, need_weights=False
            )
        gate = torch.sigmoid(self.gating_linear_1(seqs)) if self.gated else 1.0
        seqs = seqs + gate * self.dropout_1(attended)
        output = self.feed_forward(self.layer_norm_2(seqs))
        gate = torch.sigmoid(self.gating_linear_2(seqs)) if self.gated else 1.0
        seqs = seqs + gate * self.dropout_2(output)
        return seqs if self.gated else self.dropout_3(seqs)


class ContinuousRecommender(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.n_items + 1, config.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        self.register_buffer(
            "causal_mask", torch.ones(config.max_length, config.max_length,
                                      dtype=torch.bool).triu(1), persistent=False
        )

    def attention_mask(self, histories):
        batch, length = histories.shape
        mask = (self.causal_mask[:length, :length][None]
                | histories.eq(0)[:, None, :])
        # Match RecTools' merged mask: padded queries may attend their diagonal,
        # while real queries never attend padded keys or future real positions.
        mask = mask & ~torch.eye(length, device=histories.device, dtype=torch.bool)[None]
        return mask[:, None].expand(batch, self.config.heads, length, length).reshape(
            batch * self.config.heads, length, length
        )

    def encode_all(self, histories):
        if histories.ndim != 2 or not 1 <= histories.shape[1] <= self.config.max_length:
            raise ValueError("Expected nonempty [batch, length <= max_length]")
        # Inverse learned positions match the author's RecTools backbone.
        positions = torch.arange(histories.shape[1] - 1, -1, -1, device=histories.device)
        seqs = self.embedding_dropout(
            self.item_embedding(histories) + self.position_embedding(positions)[None]
        )
        mask = self.attention_mask(histories)
        for block in self.blocks:
            seqs = block(seqs, mask)
        return seqs

    def forward(self, histories):
        # The data boundary guarantees left padding and a real final item.
        return self.encode_all(histories)[:, -1, :]

    def score_catalog(self, queries):
        # Column 0 corresponds to real item 1; PAD never has a logit.
        return F.linear(queries, self.item_embedding.weight[1:])

    def loss(self, histories, targets, negative_ids=None):
        queries = self(histories)
        if self.config.model == "sasrec_fullce":
            return F.cross_entropy(self.score_catalog(queries).float(), targets - 1)
        if negative_ids is None:
            # Author's CatalogUniformSampler: iid uniform, with replacement,
            # no logQ correction and no rejection of accidental positive hits.
            negative_ids = torch.randint(
                1, self.config.n_items + 1,
                (len(targets), self.config.negatives), device=targets.device
            )
        candidates = torch.cat((targets[:, None], negative_ids), dim=1)
        # One batched dot-product path keeps positives and negatives under the
        # same autocast precision instead of mixing FP32 multiply with BF16 GEMM.
        logits = torch.einsum("bd,bkd->bk", queries, self.item_embedding(candidates)).float()
        return F.cross_entropy(logits, torch.zeros_like(targets))
