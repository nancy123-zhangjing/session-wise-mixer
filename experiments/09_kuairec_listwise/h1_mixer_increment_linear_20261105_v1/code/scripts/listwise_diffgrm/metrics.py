"""True item-level multi-positive metrics for one list and list aggregation."""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from .decoding import GenerationOutput, aggregate_candidates


def ranking_metrics(predictions: Any, targets: Any, *, ks: Sequence[int] = (5,),
                    catalog_item_ids: Any | None = None) -> dict[str, Tensor]:
    """Return per-example metrics; do not turn SID equality into item credit.

    Targets are unique positive item IDs. Duplicate predictions receive credit
    at their first rank only; subsequent duplicates/PAD/invalid IDs keep their
    positions and count as misses. Requesting k beyond the actual output width
    is rejected, so five outputs cannot silently be advertised as Top50.
    """
    predicted = torch.as_tensor(predictions)
    actual = torch.as_tensor(targets, device=predicted.device)
    if predicted.ndim != 2 or actual.ndim != 2 or predicted.shape[0] != actual.shape[0] or not actual.shape[1]:
        raise ValueError('Predictions and targets must be aligned nonempty [batch, positions] matrices')
    if predicted.dtype not in (torch.int32, torch.int64) or actual.dtype not in (torch.int32, torch.int64):
        raise ValueError('Metrics require true integer item IDs')
    if (actual <= 0).any():
        raise ValueError('Targets must contain actual positive items, not PAD')
    target_equal = actual[:, :, None] == actual[:, None, :]
    if torch.tril(target_equal, diagonal=-1).any():
        raise ValueError('The target set must contain distinct items')
    width, positives = predicted.shape[1], actual.shape[1]
    ks = tuple(int(k) for k in ks)
    if not ks or any(k < 1 or k > width for k in ks):
        raise ValueError('Each k must fit the supplied output; use explicit candidate aggregation for larger k')
    valid = predicted > 0
    if catalog_item_ids is not None:
        catalog = torch.as_tensor(catalog_item_ids, device=predicted.device).reshape(-1)
        valid &= torch.isin(predicted, catalog)
    duplicates = torch.tril(predicted[:, :, None] == predicted[:, None, :], diagonal=-1).any(dim=-1) & (predicted > 0)
    relevant = ((predicted[:, :, None] == actual[:, None, :]).any(dim=-1) & valid & ~duplicates)
    discount = 1.0 / torch.log2(torch.arange(1, width + 1, device=predicted.device, dtype=torch.float32) + 1)
    result = {}
    for k in ks:
        hits = relevant[:, :k].sum(dim=-1).float()
        result[f'recall@{k}'] = hits / positives
        result[f'hr@{k}'] = (hits > 0).float()
        result[f'hit_rate@{k}'] = result[f'hr@{k}']
        ideal = discount[:min(k, positives)].sum()
        result[f'ndcg@{k}'] = (relevant[:, :k].float() * discount[:k]).sum(dim=-1) / ideal
        result[f'precision@{k}'] = hits / k
    result['completion_rate'] = (valid & ~duplicates).all(dim=-1).float()
    result['duplicate_rate'] = duplicates.float().mean(dim=-1)
    result['missing_rate'] = (predicted == 0).float().mean(dim=-1)
    result['invalid_rate'] = ((predicted != 0) & ~valid).float().mean(dim=-1)
    result['distinct_valid_items'] = (valid & ~duplicates).sum(dim=-1).float()
    if width == positives:
        result['position_accuracy'] = (predicted == actual).float().mean(dim=-1)
        result['exact_match'] = (predicted == actual).all(dim=-1).float()
    return result


def evaluate_generation(output: GenerationOutput, target_item_ids: Tensor, *,
                        catalog_item_ids: Any | None = None,
                        aggregate_ks: Sequence[int] = (10, 20, 50),
                        max_candidate_lists: int = 32) -> dict[str, Tensor]:
    """Keep single-list quality distinct from multi-list candidate retrieval."""
    length = output.item_ids.shape[-1]
    direct = ranking_metrics(output.best_item_ids, target_item_ids, ks=(length,), catalog_item_ids=catalog_item_ids)
    result = {'list/' + name: value for name, value in direct.items()}
    if aggregate_ks:
        aggregation = aggregate_candidates(output, topk=max(aggregate_ks), max_lists=max_candidate_lists)
        expanded = ranking_metrics(aggregation.item_ids, target_item_ids, ks=aggregate_ks, catalog_item_ids=catalog_item_ids)
        result.update({'aggregate/' + name: value for name, value in expanded.items()})
        result['aggregate/candidate_count'] = aggregation.candidate_counts.float()
        result['aggregate/lists_used'] = aggregation.lists_used.float()
        for k in aggregate_ks:
            result[f'aggregate/coverage@{k}'] = (aggregation.item_ids[:, :k] > 0).float().mean(dim=-1)
    return result
