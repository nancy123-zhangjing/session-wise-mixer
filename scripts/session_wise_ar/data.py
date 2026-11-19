"""Five next-item tasks per session, compact prefix appending, no future labels."""
from __future__ import annotations

import hashlib
import torch
from scripts.rqcd.data import PackedSplit


def user_buckets(users, count=2000):
    return torch.tensor([int(hashlib.sha256(str(u).encode()).hexdigest(), 16) % count
                         for u in users], dtype=torch.long)


def compact_context(history, prefix, prefix_lengths=None):
    """Right-pad histories AFTER appending; eliminate left-PAD gaps."""
    batch, h = history.shape
    if prefix_lengths is None:
        prefix_lengths = torch.full((batch,), prefix.shape[1], device=history.device)
    visible = torch.cat((history.ne(0), torch.arange(prefix.shape[1], device=history.device)[None]
                         < prefix_lengths[:, None]), dim=1)
    values = torch.cat((history, prefix), dim=1)
    positions = visible.long().cumsum(1)-1
    result = torch.zeros_like(values)
    rows = torch.arange(batch, device=history.device)[:, None].expand_as(values)
    result[rows[visible], positions[visible]] = values[visible]
    return result, result.ne(0)


class SessionTasks:
    def __init__(self, packed, list_length=5):
        self.packed, self.list_length = packed, list_length
        self.user_codes = user_buckets(packed.users)

    def __len__(self):
        return len(self.packed) * self.list_length

    def batch(self, indices, device):
        rows, position = indices // self.list_length, indices % self.list_length
        history, targets = self.packed.batch(rows, device)
        position = position.to(device)
        # Exclude current/future targets by mask; no leakage via PAD embedding.
        context, mask = compact_context(history, targets[:, :-1], position)
        target = targets.gather(1, position[:, None]).squeeze(1)
        return {"context": context, "mask": mask, "target": target,
                "users": self.user_codes[rows.cpu()].to(device), "position": position}
