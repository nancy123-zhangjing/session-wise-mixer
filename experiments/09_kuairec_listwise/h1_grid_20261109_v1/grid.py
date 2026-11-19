"""History SID attention, inserted before the unchanged item MLP.

Uses the historical backbone's item_local_attention interface. No decoder,
Tokenizer, SID vocabulary, position embedding, or historical source is edited.
"""
import torch
from torch import nn


class ResidualAttention(nn.Module):
    def __init__(self, dim, heads=4, norm_eps=1e-5):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError('Embedding dimension must be divisible by heads')
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x, valid):
        # An empty sequence gets one dummy zero key. Invalid queries always
        # receive zero correction, including output-projection bias after learning.
        safe = valid.clone()
        safe[:, 0] |= ~valid.any(-1)
        z = self.norm(x).masked_fill(~valid.unsqueeze(-1), 0)
        update, _ = self.attention(z, z, z, key_padding_mask=~safe, need_weights=False)
        return x + update.masked_fill(~valid.unsqueeze(-1), 0)


class HistoryGridAttention(nn.Module):
    def __init__(self, dim=256, heads=4, mode='row', norm_eps=1e-5):
        super().__init__()
        if mode not in ('row', 'row_column'):
            raise ValueError('Expected row or row_column')
        self.dim, self.mode = dim, mode
        self.row = ResidualAttention(dim, heads, norm_eps)
        self.column = ResidualAttention(dim, heads, norm_eps) if mode == 'row_column' else None

    def forward(self, x, valid_mask=None):
        if x.ndim != 4 or x.shape[-1] != self.dim or min(x.shape[1:3]) < 1:
            raise ValueError('Grid expects nonempty [B,L,M,D] with the configured D')
        b, length, slots, dim = x.shape
        if valid_mask is None:
            valid = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        else:
            if valid_mask.shape != x.shape[:-1]:
                raise ValueError('Grid mask must have shape [B,L,M]')
            valid = valid_mask.to(device=x.device, dtype=torch.bool)
        x = self.row(x.reshape(b * length, slots, dim), valid.reshape(b * length, slots))
        x = x.reshape(b, length, slots, dim)
        if self.column is not None:
            columns = x.permute(0, 2, 1, 3).reshape(b * slots, length, dim)
            column_mask = valid.permute(0, 2, 1).reshape(b * slots, length)
            columns = self.column(columns, column_mask)
            x = columns.reshape(b, slots, length, dim).permute(0, 2, 1, 3).contiguous()
        return x


def attach_grid(trainer, spec):
    """Call AFTER strict R5 restoration, preserving Adam's old parameter order."""
    backbone = trainer.model.backbone
    if backbone.item_local_attention_enabled or backbone.item_local_attention is not None:
        raise ValueError('History attention already installed')
    if len(trainer.r_optimizer.param_groups) != 1:
        raise ValueError('Expected the historical single R optimizer group')
    # CPU-only isolated initialization does not advance training/CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec['initialization_seed'])
        grid = HistoryGridAttention(trainer.config['model']['n_embd'], spec['heads'],
                                    spec['mode'], spec['norm_eps'])
    grid.to(device=trainer.device, dtype=next(backbone.parameters()).dtype)
    backbone.item_local_attention = grid
    backbone.item_local_attention_enabled = True
    # Do NOT add a parameter group or recreate Adam: preserve inherited moments,
    # current LR and LambdaLR's original one-group, full-R35 schedule.
    trainer.r_optimizer.param_groups[0]['params'].extend(grid.parameters())
    grid.train(trainer.model.training)
    return grid
