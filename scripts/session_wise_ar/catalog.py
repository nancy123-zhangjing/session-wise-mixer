"""Unique four-digit catalog paths and batched per-item legal beam search."""
from __future__ import annotations

import torch
from .data import compact_context
from scripts.rqcd_mixer.catalog import distinct_history_ids


def add_suffix(semantic, codes):
    # Original DIGER assigns suffixes in catalog row order (internal ID 1..N).
    counts, rows = {}, []
    for prefix in semantic.detach().cpu().tolist():
        key = tuple(prefix)
        suffix = counts.get(key, 0)
        if suffix >= codes:
            raise ValueError(f'Collision bucket exceeds {codes}; no silent wrap/drop')
        counts[key] = suffix+1
        rows.append(prefix+[suffix])
    return torch.tensor([[-1]*4]+rows, dtype=torch.long, device=semantic.device)


class Catalog:
    def __init__(self, sid, codes, original_ids=None):
        self.sid, self.codes = sid, codes
        n = len(sid)-1
        if sid.shape != (n+1, 4) or not ((sid[1:] >= 0) & (sid[1:] < codes)).all():
            raise ValueError('Invalid catalog SID geometry')
        identities = list(range(1, n+1)) if original_ids is None else list(original_ids)
        if len(identities) != n or len(set(identities)) != n:
            raise ValueError('Original ID mapping mismatch')
        rank = {value: index for index, value in enumerate(sorted(identities))}
        children, count, minimum, leaf = [[-1]*codes], [0], [n+1], [0]
        paths = [[0]*5]
        for item, code in enumerate(sid[1:].cpu().tolist(), 1):
            node, path = 0, [0]
            item_rank = rank[identities[item-1]]
            count[0] += 1; minimum[0] = min(minimum[0], item_rank)
            for digit in code:
                if children[node][digit] < 0:
                    children[node][digit] = len(children)
                    children.append([-1]*codes); count.append(0); minimum.append(n+1); leaf.append(0)
                node = children[node][digit]
                count[node] += 1; minimum[node] = min(minimum[node], item_rank)
                path.append(node)
            if leaf[node]:
                raise ValueError('Non-unique full SID; suffix construction failed')
            leaf[node] = item
            paths.append(path)
        device = sid.device
        self.children = torch.tensor(children, device=device)
        self.counts = torch.tensor(count, device=device)
        self.minimum = torch.tensor(minimum, device=device)
        self.leaf = torch.tensor(leaf, device=device)
        self.paths = torch.tensor(paths, device=device)

    @torch.no_grad()
    def next_item(self, rec, context, users, selected, width):
        hidden, mask = rec.encode(context, self.sid, users)
        b, device = len(context), context.device
        prefixes = torch.empty((b, 1, 0), dtype=torch.long, device=device)
        nodes = torch.zeros((b, 1), dtype=torch.long, device=device)
        scores = torch.zeros((b, 1), device=device)
        for depth in range(4):
            beams = prefixes.shape[1]
            logp = rec.next_logprobs(hidden[:, None].expand(-1, beams, -1, -1).reshape(b*beams, *hidden.shape[1:]),
                                     mask[:, None].expand(-1, beams, -1).reshape(b*beams, -1),
                                     prefixes.reshape(b*beams, depth)).view(b, beams, self.codes)
            child = self.children[nodes]
            safe = child.clamp_min(0)
            remaining = self.counts[safe]
            if selected.shape[1]:
                used_paths = self.paths[selected, depth+1]
                remaining = remaining - ((safe[..., None] == used_paths[:, None, None]) &
                                         (selected[:, None, None, :] > 0)).sum(-1)
            candidate = (scores[..., None]+logp).masked_fill((child < 0) | (remaining <= 0), -torch.inf)
            candidate, child = candidate.flatten(1), child.flatten(1)
            # Stable score ordering; equal scores break by minimum original ID.
            tie = self.minimum[child.clamp_min(0)].argsort(dim=-1, stable=True)
            ordered = candidate.gather(1, tie).argsort(dim=-1, descending=True, stable=True)
            chosen = tie.gather(1, ordered[:, :min(width, candidate.shape[1])])
            parent, digit = chosen//self.codes, chosen%self.codes
            prefixes = torch.cat((prefixes.gather(1, parent[..., None].expand(-1, -1, depth)), digit[..., None]), -1)
            scores, nodes = candidate.gather(1, chosen), child.gather(1, chosen).clamp_min(0)
        if not torch.isfinite(scores[:, 0]).all():
            raise RuntimeError('No legal unseen item remains')
        return self.leaf[nodes[:, 0]]

    @torch.no_grad()
    def generate(self, rec, history, users, length=5, width=20, *, filter_history=False):
        selected = torch.empty((len(history), 0), dtype=torch.long, device=history.device)
        excluded = distinct_history_ids(history) if filter_history else selected
        for _ in range(length):
            context, _ = compact_context(history, selected)
            blocked = torch.cat((excluded, selected), 1) if filter_history else selected
            item = self.next_item(rec, context, users, blocked, width)
            selected = torch.cat((selected, item[:, None]), 1)
        return selected


@torch.no_grad()
def encoding_stats(tokenizer, sid, previous=None, previous_books=None):
    base, effective = tokenizer.base_books(), tokenizer.effective_books()
    n, codes = len(sid)-1, base.shape[1]
    counts = torch.stack([torch.bincount(sid[1:, i], minlength=codes) for i in range(3)]).float()
    p = counts/counts.sum(-1, keepdim=True)
    unique = torch.unique(sid[1:, :3], dim=0, return_counts=True)[1]
    out = {'utilization': (counts.gt(0).float().mean(-1)).tolist(),
           'perplexity': (-(p*p.clamp_min(1e-30).log()).sum(-1)).exp().tolist(),
           'collision_item_fraction': (unique[unique > 1].sum().float()/n).item(),
           'collision_redundancy': 1-len(unique)/n,
           'relative_correction_rms': ((effective-base).square().mean((-2, -1)).sqrt()
                                       /base.square().mean((-2, -1)).sqrt().clamp_min(1e-8)).tolist()}
    if previous is not None:
        out.update(semantic_sid_changed=(sid[1:, :3] != previous[1:, :3]).any(-1).float().mean().item(),
                   suffix_changed=(sid[1:, -1] != previous[1:, -1]).float().mean().item())
    if previous_books is not None:
        out['effective_relative_change'] = ((effective-previous_books).square().mean((-2, -1)).sqrt()
                      /previous_books.square().mean((-2, -1)).sqrt().clamp_min(1e-8)).tolist()
    return out
