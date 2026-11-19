"""Opt-in H decoder prefixes. Original engine blocks and checkpoints stay intact."""
from __future__ import annotations

from torch import nn
import torch

MODES = ('history', 'asymmetric', 'cross_attention')


class PrefixCrossAttention(nn.Module):
    def __init__(self, config, attention_type, dim):
        super().__init__()
        # Same normalization and initialization conventions as the backbone.
        norm = nn.RMSNorm if config.get('norm_type', 'layernorm').lower() == 'rmsnorm' else nn.LayerNorm
        eps = float(config.get('norm_eps', 1e-6))
        self.query_norm = norm(dim, eps=eps)
        self.prefix_norm = norm(dim, eps=eps)
        self.attention = attention_type(dim, int(config['n_head']),
                                       float(config.get('attn_pdrop', 0.1)),
                                       float(config.get('resid_pdrop', 0.1)))

    def forward(self, hidden, prefix):
        if prefix.shape[1] == 0:
            return hidden
        dim = hidden.shape[-1]
        kv = self.attention.qkv(self.prefix_norm(prefix))[..., dim:]
        update, _ = self.attention(self.query_norm(hidden), key_value=kv)
        return hidden + update


def asymmetric_mask(prefix_length, current_length, device):
    """Rows read columns: bidirectional prefix, no prefix -> current reads."""
    length = prefix_length + current_length
    mask = torch.ones((1, length, length), dtype=torch.bool, device=device)
    mask[:, :prefix_length, prefix_length:] = False
    return mask


def decoder_layer(block, hidden, encoded, cross_kv, mask=None,
                  prefix_layer=None, prefix=None):
    """Original pre-norm block with explicit self mask / optional prefix branch.

    Do not change the engine's legacy mask handling or teacher raw-K/V convention.
    """
    update, _ = block.self_attn(block.ln_1(hidden), attention_mask=mask,
                                is_decoder_self_attn=True)
    hidden = hidden + update
    if prefix_layer is not None:
        hidden = prefix_layer(hidden, prefix)
    if cross_kv is None:
        cross_kv = torch.cat((encoded, encoded), dim=-1)
    update, _ = block.cross_attn(block.ln_2(hidden), key_value=cross_kv)
    hidden = hidden + update
    return hidden + block.mlp(block.ln_3(hidden))
