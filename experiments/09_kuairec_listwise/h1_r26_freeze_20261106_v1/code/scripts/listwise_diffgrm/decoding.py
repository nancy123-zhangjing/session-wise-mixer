"""Target-free, catalog-constrained CPD for whole-list and hierarchical DiffGRM.

One beam is one *list* path. Both routes retain at most ``beam_size`` paths
across all item/code decisions. No historical-item exclusion is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)


@dataclass(frozen=True)
class _PartialInfo:
    sid_indices: np.ndarray
    allowed: np.ndarray


class SIDCatalog:
    """Multi-value SID catalog with explicit deterministic collision resolution.

    ``item_ids`` are the model's positive mapped IDs; 0 is PAD. Each item has
    exactly one SID. Within a SID bucket, reconstruction error, original item
    ID, then mapped ID determine the order. Build a *new* object after every
    tokenizer refresh so compatibility caches cannot retain old assignments.

    Compatibility uses inverted postings and caches partial SID patterns. It
    never Python-scans the full item catalog at every decoding step.
    """

    def __init__(self, item_ids: Any, sids: Any, reconstruction_errors: Any,
                 original_item_ids: Any | None = None, codebook_size: int | None = None):
        ids, codes = _numpy(item_ids), _numpy(sids)
        errors = _numpy(reconstruction_errors).astype(np.float64, copy=False)
        originals = ids if original_item_ids is None else _numpy(original_item_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids <= 0):
            raise ValueError('Catalog item_ids must be positive integers; PAD 0 is not an item')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Catalog item IDs must be unique')
        if codes.ndim != 2 or codes.shape[0] != len(ids) or codes.shape[1] < 1 or codes.dtype.kind not in 'iu':
            raise ValueError('Catalog SIDs must be an integer [items, digits] matrix')
        if originals.shape != ids.shape or originals.dtype.kind not in 'iu':
            raise ValueError('Original item IDs must be an integer vector aligned with item_ids')
        if errors.shape != ids.shape or not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError('Finite, nonnegative reconstruction errors are required')
        if codebook_size is None:
            if not len(ids):
                raise ValueError('An empty catalog requires an explicit codebook_size')
            codebook_size = int(codes.max()) + 1
        if codebook_size < 1 or np.any(codes < 0) or np.any(codes >= codebook_size):
            raise ValueError('Catalog contains out-of-range SID codes')
        self.item_ids = ids.astype(np.int64, copy=True)
        self.item_ids.flags.writeable = False
        self.n_digit = int(codes.shape[1])
        self.codebook_size = int(codebook_size)
        self.item_count = len(ids)
        buckets: dict[tuple[int, ...], list[tuple[float, int, int]]] = {}
        for item, original, sid, error in zip(ids, originals, codes, errors):
            key = tuple(int(code) for code in sid)
            buckets.setdefault(key, []).append((float(error), int(original), int(item)))
        self.buckets = {key: tuple(item for _, _, item in sorted(values)) for key, values in buckets.items()}
        self._sid_keys = sorted(self.buckets)
        self._sids = np.asarray(self._sid_keys, dtype=np.int64).reshape(-1, self.n_digit)
        self._postings = [
            {int(code): np.flatnonzero(self._sids[:, d] == code) for code in np.unique(self._sids[:, d])}
            for d in range(self.n_digit)
        ]
        self._partials: dict[tuple[int, ...], _PartialInfo] = {}
        self._matching_options: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}

    def _info(self, partial: Sequence[int]) -> _PartialInfo:
        key = tuple(int(code) for code in partial)
        if len(key) != self.n_digit or any(code < -1 or code >= self.codebook_size for code in key):
            raise ValueError('Invalid partial SID; use local code IDs or -1 for MASK')
        cached = self._partials.get(key)
        if cached is not None:
            return cached
        fixed = [(d, code) for d, code in enumerate(key) if code >= 0]
        if fixed:
            postings = [self._postings[d].get(code, np.empty(0, dtype=np.int64)) for d, code in fixed]
            indices = min(postings, key=len)
            for d, code in fixed:
                indices = indices[self._sids[indices, d] == code]
        else:
            indices = np.arange(len(self._sids), dtype=np.int64)
        allowed = np.zeros((self.n_digit, self.codebook_size), dtype=bool)
        if len(indices):
            for d, code in enumerate(key):
                if code == -1:
                    allowed[d, np.unique(self._sids[indices, d])] = True
        info = _PartialInfo(indices, allowed)
        self._partials[key] = info
        return info

    def legal_token_mask(self, partials: Any) -> np.ndarray:
        values = _numpy(partials)
        if values.ndim != 2 or values.shape[1] != self.n_digit:
            raise ValueError('Expected [partial_items, digits]')
        if not len(values):
            return np.empty((0, self.n_digit, self.codebook_size), dtype=bool)
        return np.stack([self._info(row).allowed for row in values])

    def _options(self, partial: Sequence[int], limit: int) -> tuple[int, ...]:
        key = tuple(int(code) for code in partial)
        cache_key = (key, limit)
        if cache_key not in self._matching_options:
            items: list[int] = []
            for index in self._info(key).sid_indices:
                items.extend(self.buckets[self._sid_keys[int(index)]][:limit - len(items)])
                if len(items) >= limit:
                    break
            self._matching_options[cache_key] = tuple(items)
        return self._matching_options[cache_key]

    def can_complete(self, partial_items: Any) -> bool:
        """Exact distinct-item feasibility for a small number of list slots.

        For K slots it suffices to retain any K distinct candidates per slot:
        a set with >=K candidates cannot cause a Hall violation, and smaller
        sets remain intact. Thus matching never scans 10k items in Python.
        """
        values = _numpy(partial_items)
        if values.ndim != 2 or values.shape[1] != self.n_digit:
            raise ValueError('Expected [list_slots, digits]')
        count = len(values)
        if count > self.item_count:
            return False
        options = [self._options(row, count) for row in values]
        if any(not row for row in options):
            return False
        owner: dict[int, int] = {}

        def augment(slot: int, seen: set[int]) -> bool:
            for item in options[slot]:
                if item in seen:
                    continue
                seen.add(item)
                if item not in owner or augment(owner[item], seen):
                    owner[item] = slot
                    return True
            return False

        return all(augment(slot, set()) for slot in sorted(range(count), key=lambda i: len(options[i])))

    def resolve(self, sid_list: Any) -> list[int]:
        """Resolve only complete slots, in final list order, leaving holes as PAD.

        Completion order is irrelevant. Two equal SIDs can yield two distinct
        items when their bucket contains enough items; this is not a duplicate.
        """
        values = _numpy(sid_list)
        if values.ndim != 2 or values.shape[1] != self.n_digit:
            raise ValueError('Expected [list_slots, digits]')
        seen: set[int] = set()
        result = []
        for row in values:
            bucket = self.buckets.get(tuple(int(code) for code in row), ())
            chosen = next((item for item in bucket if item not in seen), 0)
            result.append(chosen)
            if chosen:
                seen.add(chosen)
        return result


@dataclass
class GenerationOutput:
    item_ids: Tensor                 # [B, W, K], PAD=0
    sid: Tensor                      # [B, W, K, D], MASK/unfilled=-1
    scores: Tensor                   # [B, W], cumulative code log-probability
    beam_mask: Tensor                # actual paths, not repeated padding beams
    complete_mask: Tensor            # K distinct valid items
    counters: dict[str, int | str]

    @property
    def best_item_ids(self) -> Tensor:
        return self.item_ids[:, 0]


@dataclass
class CandidateAggregation:
    item_ids: Tensor                 # [B, topk], PAD=0 if underfilled
    candidate_counts: Tensor
    lists_used: Tensor


@dataclass(frozen=True)
class _Beam:
    sid: tuple[int, ...]
    score: float


@contextmanager
def _evaluating(model: Any):
    modes = [(module, module.training) for module in model.modules()] if isinstance(model, torch.nn.Module) else []
    if modes:
        model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training


def _next_beams(beams: list[_Beam], scores: Tensor, catalog: SIDCatalog,
                beam_size: int, list_length: int, required_slots: int,
                position_offset: int) -> tuple[list[_Beam], list[int]]:
    """Select a single global top-W frontier, max-path deduplicated.

    Log-probabilities are not renormalized after legality masking. Adaptive
    top-k expands the candidate window only when duplicate/infeasible paths
    would otherwise leave a beam underfilled. Ties use flattened index order.
    """
    width, positions, vocab = scores.shape
    flat = scores.reshape(-1)
    finite_count = int(torch.isfinite(flat).sum().item())
    if not finite_count:
        return [], []
    window = min(finite_count, max(beam_size * 4, 1))
    while True:
        threshold = torch.topk(flat, k=window, sorted=False).values.min()
        # Include every tie at the boundary; topk alone may choose arbitrary ties.
        indices = torch.nonzero(torch.isfinite(flat) & (flat >= threshold), as_tuple=False).flatten()
        ordering = torch.argsort(flat[indices], descending=True, stable=True)
        indices = indices[ordering]
        candidates = indices.detach().cpu().tolist()
        candidate_scores = flat[indices].detach().cpu().tolist()
        unique: set[tuple[int, ...]] = set()
        accepted: list[_Beam] = []
        parents: list[int] = []
        for index, score in zip(candidates, candidate_scores):
            parent, remainder = divmod(index, positions * vocab)
            position, code = divmod(remainder, vocab)
            state = list(beams[parent].sid)
            global_position = position + position_offset
            if state[global_position] != -1:
                raise AssertionError('A filled code position must never be expanded again')
            state[global_position] = code
            key = tuple(state)
            if key in unique:
                continue
            unique.add(key)
            items = np.asarray(state, dtype=np.int64).reshape(list_length, catalog.n_digit)
            if not catalog.can_complete(items[:required_slots]):
                continue
            accepted.append(_Beam(key, float(score)))
            parents.append(parent)
            if len(accepted) == beam_size:
                return accepted, parents
        if window == finite_count:
            return accepted, parents
        window = min(finite_count, window * 2)


def generate_lists(model: Any, history_sid: Tensor, history_mask: Tensor,
                   catalog: SIDCatalog, *, route: str = 'J', list_length: int = 5,
                   beam_size: int = 32) -> GenerationOutput:
    """Generate lists without a target/label argument or historical filtering.

    Model interface::

        decode_logits(history_sid, history_mask, partial_sid, item_index=None)

    J passes K*D positions. H passes D positions and appends completed generated
    prefixes to the original history; all branches are forwarded in one batch.
    Models may expose encode_context/decode_with_context for request-local
    caching. J encodes history once; H re-encodes the generated prefix once per
    item. Within an item, context rows follow their selected beam parents.
    Caller must supply only observed history.
    """
    if route not in ('J', 'H') or list_length < 1 or beam_size < 1:
        raise ValueError('route must be J/H and list_length/beam_size must be positive')
    if history_sid.ndim != 3 or history_sid.shape[-1] != catalog.n_digit:
        raise ValueError('history_sid must have shape [B, L, digits]')
    if history_mask.shape != history_sid.shape[:2] or history_sid.shape[0] == 0:
        raise ValueError('Invalid history_mask shape or empty batch')
    if history_sid.dtype not in (torch.int32, torch.int64):
        raise ValueError('History SID must use integer local codes')
    if history_sid.device != history_mask.device:
        raise ValueError('History and mask must share a device')
    valid_history = history_sid[history_mask.bool()]
    if ((valid_history < 0) | (valid_history >= catalog.codebook_size)).any():
        raise ValueError('Observed history contains an invalid SID code')
    batch, _, digits = history_sid.shape
    device = history_sid.device
    required_slots = min(list_length, catalog.item_count)
    initial = _Beam(tuple([-1] * (list_length * digits)), 0.0)
    active = [[initial] for _ in range(batch)]
    fallback: list[list[_Beam] | None] = [None] * batch
    forward_calls, state_rows, encoder_calls, encoder_rows = 0, 0, 0, 0
    use_context = callable(getattr(model, 'encode_context', None)) and callable(getattr(model, 'decode_with_context', None))
    context = None
    with _evaluating(model):
        for step in range(required_slots * digits):
            flattened = [(query, beam) for query, beams in enumerate(active) for beam in beams]
            if not flattened:
                break
            query_ids = torch.tensor([query for query, _ in flattened], dtype=torch.long, device=device)
            states_np = np.asarray([beam.sid for _, beam in flattened], dtype=np.int64).reshape(-1, list_length, digits)
            states = torch.as_tensor(states_np, device=device)
            histories = history_sid.index_select(0, query_ids)
            masks = history_mask.index_select(0, query_ids).bool()
            item_index = step // digits if route == 'H' else None
            if route == 'H':
                if item_index:
                    histories = torch.cat((histories, states[:, :item_index]), dim=1)
                    masks = torch.cat((masks, torch.ones((len(flattened), item_index), device=device, dtype=torch.bool)), dim=1)
                partial = states[:, item_index]
                legal = catalog.legal_token_mask(states_np[:, item_index])
                position_offset = item_index * digits
            else:
                partial = states.reshape(len(flattened), -1)
                legal = catalog.legal_token_mask(states_np.reshape(-1, digits)).reshape(len(flattened), list_length * digits, -1)
                legal[:, required_slots * digits:] = False
                position_offset = 0
            if use_context:
                if context is None or (route == 'H' and step % digits == 0):
                    context = model.encode_context(histories, masks)
                    if not isinstance(context, Tensor) or context.shape[0] != len(flattened) or context.device != device:
                        raise ValueError('encode_context returned an unexpected batch or device')
                    encoder_calls += 1
                    encoder_rows += len(flattened)
                logits = model.decode_with_context(context, partial, item_index=item_index)
            else:
                logits = model.decode_logits(histories, masks, partial, item_index=item_index)
            if not isinstance(logits, Tensor) or logits.shape != (*partial.shape, catalog.codebook_size):
                raise ValueError('decode_logits returned an unexpected shape')
            if logits.device != device or torch.isnan(logits).any() or torch.isposinf(logits).any():
                raise ValueError('decode_logits returned wrong-device, NaN, or positive-infinite logits')
            forward_calls += 1
            state_rows += len(flattened)
            logits = logits.float()
            available = torch.isfinite(logits).any(dim=-1, keepdim=True)
            logp = F.log_softmax(torch.where(available, logits, torch.zeros_like(logits)), dim=-1)
            logp = logp.masked_fill(~available, float('-inf'))
            logp = logp.masked_fill(~torch.as_tensor(legal, device=device), float('-inf'))
            offset = 0
            selected_context_rows: list[int] = []
            for query, beams in enumerate(active):
                if not beams:
                    continue
                count = len(beams)
                prior = torch.tensor([beam.score for beam in beams], dtype=logp.dtype, device=device)
                candidates = logp[offset:offset + count] + prior[:, None, None]
                next_beams, parents = _next_beams(beams, candidates, catalog, beam_size, list_length, required_slots, position_offset)
                if not next_beams:
                    fallback[query] = beams
                active[query] = next_beams
                selected_context_rows.extend(offset + parent for parent in parents)
                offset += count
            if use_context:
                parent_ids = torch.tensor(selected_context_rows, dtype=torch.long, device=device)
                context = context.index_select(0, parent_ids)
    # Catalog resolution already lives on CPU; assemble once and transfer each
    # batch field once instead of launching tiny GPU copies for every beam.
    item_ids = np.zeros((batch, beam_size, list_length), dtype=np.int64)
    sid = np.full((batch, beam_size, list_length, digits), -1, dtype=np.int64)
    scores = np.full((batch, beam_size), float('-inf'), dtype=np.float32)
    beam_mask = np.zeros((batch, beam_size), dtype=bool)
    complete = np.zeros_like(beam_mask)
    for query in range(batch):
        paths = active[query] if active[query] else fallback[query] or []
        if not required_slots:
            paths = []
        for rank, beam in enumerate(paths[:beam_size]):
            codes = np.asarray(beam.sid).reshape(list_length, digits)
            resolved = catalog.resolve(codes)
            item_ids[query, rank] = resolved
            sid[query, rank] = codes
            scores[query, rank] = beam.score
            beam_mask[query, rank] = True
            complete[query, rank] = all(item > 0 for item in resolved) and len(set(resolved)) == list_length
    return GenerationOutput(*(torch.as_tensor(value, device=device) for value in
                              (item_ids, sid, scores, beam_mask, complete)),
                            {'route': route, 'beam_size': beam_size, 'model_forward_calls': forward_calls,
                             'decoder_state_rows': state_rows, 'logical_steps_requested': required_slots * digits,
                             'encoder_forward_calls': encoder_calls, 'encoder_state_rows': encoder_rows,
                             'context_cache_enabled': int(use_context)})


def aggregate_candidates(output: GenerationOutput, *, topk: int = 50,
                         max_lists: int = 32) -> CandidateAggregation:
    """Rank-list-first then slot-first expansion of complete candidate lists.

    Equal path scores retain beam order. First item occurrence wins; no score
    uses target labels, future popularity, or repeated padding candidates.
    """
    if topk < 1 or max_lists < 1:
        raise ValueError('topk and max_lists must be positive')
    device = output.item_ids.device
    batch, width, _ = output.item_ids.shape
    ranked = np.zeros((batch, topk), dtype=np.int64)
    counts = np.zeros(batch, dtype=np.int64)
    used = np.zeros_like(counts)
    ids_cpu = output.item_ids.detach().cpu().tolist()
    scores = output.scores.detach().cpu()
    masks = (output.beam_mask & output.complete_mask & torch.isfinite(output.scores)).detach().cpu()
    for query in range(batch):
        order = torch.argsort(scores[query], descending=True, stable=True).tolist()
        seen: set[int] = set()
        items: list[int] = []
        for rank in order:
            if not masks[query, rank] or used[query] >= max_lists:
                continue
            used[query] += 1
            for item in ids_cpu[query][rank]:
                if item > 0 and item not in seen:
                    seen.add(item)
                    items.append(item)
                    if len(items) == topk:
                        break
            if len(items) == topk:
                break
        counts[query] = len(items)
        if items:
            ranked[query, :len(items)] = items
    return CandidateAggregation(*(torch.as_tensor(value, device=device) for value in (ranked, counts, used)))
