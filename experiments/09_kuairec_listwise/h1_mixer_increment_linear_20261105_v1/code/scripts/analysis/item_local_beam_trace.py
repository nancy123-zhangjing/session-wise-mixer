"""Observational probes for the pinned Stage07 confidence decoder.

The original function is AST-copied, with calls inserted at four decision
boundaries. No original expression, order, dtype, RNG call, or search decision
is changed. Labels are accessible only to the observer. This is evaluation
instrumentation, not a new decoder or a training implementation.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


TRACE_SCHEMA = "stage07_confidence_search_trace_v1"
PINNED_FUNCTION_AST_SHA256 = "2ddda8ef2a14bb072258f8a80059a7856472cf75785e605a569901cbed7673e4"


def _function_node(source):
    # Python 3.10 retains CR in some literal AST fields; normalize only source
    # newlines before parsing so Windows and Linux produce the same fingerprint.
    tree = ast.parse(textwrap.dedent(source.replace("\r\n", "\n").replace("\r", "\n")))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "iterative_mask_decode"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one iterative_mask_decode function")
    return functions[0]


def function_ast_sha256(source):
    node = _function_node(source)
    # AST dump formatting and empty fields changed in Python 3.13/3.14.
    # Canonicalize fields ourselves so the 3.10 remote runtime has the same pin.
    def canonical(value):
        if isinstance(value, ast.AST):
            return [type(value).__name__, {name: canonical(child)
                    for name, child in ast.iter_fields(value)
                    if child is not None and child != []}]
        if isinstance(value, list):
            return [canonical(child) for child in value]
        return value
    encoded = json.dumps(canonical(node), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def instrument_function_ast(source):
    """Fail closed on source drift, then add observer calls only."""
    if function_ast_sha256(source) != PINNED_FUNCTION_AST_SHA256:
        raise ValueError("Pinned decoder AST does not match; do not instrument changed code")
    node = _function_node(source)

    class Insert(ast.NodeTransformer):
        def __init__(self):
            self.anchors = {"initial": 0, "middle": 0, "last": 0, "candidates": 0}

        def visit_Assign(self, statement):
            # These exact anchors occur once in the pinned source. Each probe
            # precedes any subsequent in-place mutation of an aliased state.
            target = ast.unparse(statement.targets[0])
            if target == "kv_cache_final":
                self.anchors["initial"] += 1
                probe = "_trace_observer.initial(beam_ids[:, :BEAM_ACT, :], all_log_probs, MASK_ID, VOC)"
                return [statement, ast.parse(probe).body[0]]
            if target == "(best_logprobs, new_beam_ids)":
                self.anchors["middle"] += 1
                probe = "_trace_observer.middle(step + 1, active_beam_ids, new_beam_ids)"
                return [statement, ast.parse(probe).body[0]]
            if target == "(best_token_logprobs, best_tokens)":
                self.anchors["last"] += 1
                probe = "_trace_observer.last_before(active_beam_ids, active_beam_logp, final_logits, last_mask_pos, best_tokens)"
                return [statement, ast.parse(probe).body[0]]
            if target == "dedup_strategy" and isinstance(statement.value, ast.Constant):
                self.anchors["candidates"] += 1
                probe = "_trace_observer.candidates(active_beam_ids, final_beam_logp, TOP_K_FINAL)"
                return [ast.parse(probe).body[0], statement]
            return statement

    transform = Insert()
    node = transform.visit(node)
    if transform.anchors != {"initial": 1, "middle": 1, "last": 1, "candidates": 1}:
        raise ValueError(f"Unexpected decoder instrumentation anchors: {transform.anchors}")
    # Validate confidence before invoking the untouched original body.
    guard = ast.parse("_trace_observer.require_ready(mode, model)").body[0]
    node.body.insert(1 if isinstance(node.body[0], ast.Expr) else 0, guard)
    tree = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(tree)
    return tree


@contextmanager
def install_trace(beam_module, observer):
    """Temporarily instrument the module used by normal model.generate()."""
    original = beam_module.iterative_mask_decode
    cached = getattr(observer, "_compiled_trace", None)
    if cached is None or cached[0] is not original:
        source = inspect.getsource(original)
        tree = instrument_function_ast(source)
        namespace = dict(original.__globals__)
        namespace["_trace_observer"] = observer
        exec(compile(tree, inspect.getsourcefile(original) or "<pinned-beam>", "exec"), namespace)
        observer.source_identity = {
            "function_ast_sha256": function_ast_sha256(source),
            "module_file_sha256": hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest(),
            "module_file": inspect.getfile(original),
        }
        cached = observer._compiled_trace = (original, namespace["iterative_mask_decode"])
    beam_module.iterative_mask_decode = cached[1]
    try:
        yield observer
    finally:
        beam_module.iterative_mask_decode = original


class BeamTraceObserver:
    """Batch observer; one instance is used by one process on one GPU.

    catalog_sids: all catalog SID rows, allowing item collisions. start_batch
    accepts target SID rows, not item IDs. Per-user rows report SID-level
    search behaviour; they must not be presented as exact-item recall.
    """

    def __init__(self, catalog_sids, mask_id, codebook_size):
        import torch

        self.torch = torch
        catalog = torch.as_tensor(catalog_sids, dtype=torch.long, device="cpu")
        if catalog.ndim != 2 or not 2 <= catalog.shape[1] <= 6:
            raise ValueError("Expected catalog [items, 2..6 SID digits]")
        self.n_digit = int(catalog.shape[1])
        self.mask_id = int(mask_id)
        self.vocab = int(codebook_size)
        if 0 <= self.mask_id < self.vocab:
            raise ValueError("MASK must be distinct from codeword indices")
        if self.vocab < 2 or not bool(((catalog >= 0) & (catalog < self.vocab)).all()):
            raise ValueError("Catalog contains invalid SID values")
        if (self.vocab + 1) ** self.n_digit > 2 ** 63 - 1:
            raise ValueError("Partial-state encoding would overflow int64")
        self._powers_cpu = torch.tensor([(self.vocab + 1) ** d for d in range(self.n_digit)], dtype=torch.long)
        projected = []
        for filled_mask in range(1 << self.n_digit):
            mask = torch.tensor([bool(filled_mask & (1 << d)) for d in range(self.n_digit)])
            state = torch.where(mask, catalog, self.vocab)
            projected.append((state * self._powers_cpu).sum(-1))
        self._catalog_keys_cpu = torch.unique(torch.cat(projected), sorted=True)
        self._catalog_set = {tuple(row) for row in catalog.tolist()}
        self._device = None
        self._active = False
        self.source_identity = None

    def start_batch(self, target_sids):
        torch = self.torch
        if self._active:
            raise RuntimeError("finish_batch() required before starting another batch")
        targets = torch.as_tensor(target_sids, dtype=torch.long)
        if targets.ndim != 2 or targets.shape[1] != self.n_digit:
            raise ValueError("Labels must be [batch, SID digits]")
        # Clone isolates observation labels from caller mutation.
        self.targets = targets.detach().clone()
        self._rounds = []
        self._last = None
        self._final = None
        self._active = True

    def require_ready(self, mode, model):
        if not self._active:
            raise RuntimeError("start_batch(target_sids) must precede generate()")
        if mode != "confidence" or model.config.get("current_split", "val") != "val":
            raise ValueError("Trace supports validation confidence decoding only")
        if model.config.get("dedup_strategy", "simple") != "simple":
            raise ValueError("Pinned trace interpretation requires simple deduplication")
        if model.n_digit != self.n_digit or model.codebook_size != self.vocab:
            raise ValueError("Model/catalog dimensions disagree")

    def _prepare_device(self, ids):
        if ids.shape[0] != self.targets.shape[0]:
            raise ValueError("Target and decoded batch sizes differ")
        if self._device != ids.device:
            self._device = ids.device
            self._powers = self._powers_cpu.to(ids.device)
            self._catalog_keys = self._catalog_keys_cpu.to(ids.device)
        self.targets = self.targets.to(ids.device)

    def _keys(self, ids):
        return (self.torch.where(ids == self.mask_id, self.vocab, ids) * self._powers).sum(-1)

    def _compatible(self, ids):
        return ((ids == self.mask_id) | (ids == self.targets[:, None, :])).all(-1)

    def _unique_count(self, keys, valid=None):
        torch = self.torch
        if valid is not None:
            keys = torch.where(valid, keys, -1)
        sorted_keys = keys.sort(-1).values
        first = torch.ones_like(sorted_keys, dtype=torch.bool)
        first[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
        return (first & (sorted_keys >= 0)).sum(-1)

    def _state_metrics(self, ids):
        keys = self._keys(ids)
        compatible = self._compatible(ids)
        locations = self.torch.searchsorted(self._catalog_keys, keys)
        legal = (locations < len(self._catalog_keys)) & (
            self._catalog_keys[locations.clamp(max=len(self._catalog_keys) - 1)] == keys)
        unique = self._unique_count(keys)
        return {
            "retained_path_count": self.torch.full_like(unique, ids.shape[1]),
            "gt_compatible_after": compatible.sum(-1),
            "gt_compatible_unique_after": self._unique_count(keys, compatible),
            "retained_unique_states": unique,
            "retained_duplicate_paths": ids.shape[1] - unique,
            "catalog_completable_paths": legal.sum(-1),
            "catalog_completable_unique_states": self._unique_count(keys, legal),
        }

    def initial(self, ids, all_log_probs, mask_id, vocab):
        if mask_id != self.mask_id or vocab != self.vocab or self._rounds:
            raise ValueError("Unexpected decoder mask/vocabulary or repeated initial probe")
        self._prepare_device(ids)
        metrics = self._state_metrics(ids)
        # Exactly one target-valued expansion for each initially masked slot.
        metrics["gt_compatible_before"] = self.torch.full_like(metrics["gt_compatible_after"], self.n_digit)
        metrics["candidate_paths_before"] = self.torch.full_like(metrics["gt_compatible_after"], self.n_digit * self.vocab)
        self._rounds.append((1, "global_topk", metrics))

    def middle(self, round_number, parents, children):
        metrics = self._state_metrics(children)
        remaining = (parents == self.mask_id).sum(-1)
        metrics["gt_compatible_before"] = (self._compatible(parents) * remaining).sum(-1)
        metrics["candidate_paths_before"] = remaining.sum(-1) * self.vocab
        self._rounds.append((int(round_number), "global_topk", metrics))

    def last_before(self, parents, parent_logp, final_logits, last_mask_pos, chosen):
        torch = self.torch
        compatible = self._compatible(parents)
        target_tokens = self.targets.gather(1, last_mask_pos)
        truth_logp = final_logits.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        strict_rank = 1 + (final_logits > truth_logp.unsqueeze(-1)).sum(-1)
        worst_tie_rank = (final_logits >= truth_logp.unsqueeze(-1)).sum(-1)
        # All payloads are new tensors: original next line mutates parents.
        self._last = {
            "integer": torch.cat([parents.clone(), compatible.unsqueeze(-1).long(),
                                   last_mask_pos.unsqueeze(-1), target_tokens.unsqueeze(-1),
                                   chosen.unsqueeze(-1), strict_rank.unsqueeze(-1),
                                   worst_tie_rank.unsqueeze(-1)], dim=-1),
            "float": torch.stack([parent_logp.clone().float(), truth_logp.float(),
                                  final_logits.max(-1).values.float()], dim=-1),
            "compatible_before": compatible.sum(-1),
        }

    def candidates(self, ids, scores, top_k):
        if self._last is None:
            raise RuntimeError("Final argmax probe was not reached")
        metrics = self._state_metrics(ids)
        metrics["gt_compatible_before"] = self._last["compatible_before"]
        metrics["candidate_paths_before"] = self.torch.full_like(metrics["gt_compatible_after"], ids.shape[1] * self.vocab)
        self._rounds.append((self.n_digit, "per_branch_argmax", metrics))
        # Match the original one-dimensional sort invocation, including ties.
        order = self.torch.stack([self.torch.argsort(scores[b], descending=True)
                                  for b in range(len(scores))])
        self._final = {"ids": ids.clone(), "scores": scores.detach().clone(),
                       "order": order,
                       "top_k": int(top_k)}

    def finish_batch(self):
        """Transfer compact trace tensors once per batch and return user rows."""
        torch = self.torch
        if not self._active or self._final is None or len(self._rounds) != self.n_digit:
            raise RuntimeError("Incomplete traced decode; no accepted batch rows")
        fields = list(self._rounds[0][2])
        if any(set(m) != set(fields) for _, _, m in self._rounds):
            raise RuntimeError("Inconsistent round metrics")
        packed = torch.stack([torch.stack([m[key] for key in fields], -1)
                              for _, _, m in self._rounds], 1).cpu().tolist()
        targets = self.targets.cpu().tolist()
        final_ids = self._final["ids"].cpu().tolist()
        final_order = self._final["order"].cpu().tolist()
        last_integer = self._last["integer"].cpu().tolist()
        last_float = self._last["float"].cpu().tolist()
        rows = []
        for b, target_values in enumerate(targets):
            target = tuple(target_values)
            rounds = [{"round": number, "selection": selection,
                       **dict(zip(fields, packed[b][r]))}
                      for r, (number, selection, _) in enumerate(self._rounds)]
            first_lost = next((r["round"] for r in rounds if r["gt_compatible_after"] == 0), None)
            sequences = [tuple(s) for s in final_ids[b]]
            legal = [s in self._catalog_set for s in sequences]
            ordered = [sequences[idx] for idx in final_order[b]]
            raw_rank = next((i + 1 for i, sid in enumerate(ordered) if sid == target), None)
            legal_unique_order = list(dict.fromkeys(sid for sid in ordered if sid in self._catalog_set))
            legal_rank = legal_unique_order.index(target) + 1 if target in legal_unique_order else None
            branches = []
            for state, values in zip(last_integer[b], last_float[b]):
                d = self.n_digit
                if state[d]:
                    branches.append({
                        "partial_sid": state[:d], "remaining_position": state[d + 1],
                        "target_token": state[d + 2], "selected_token": state[d + 3],
                        "target_rank_strict": state[d + 4], "target_rank_worst_tie": state[d + 5],
                        "selected_is_target": state[d + 2] == state[d + 3],
                        "parent_logp": values[0], "target_token_logp": values[1], "best_token_logp": values[2],
                    })
            if first_lost is not None:
                fate = f"pruned_round_{first_lost}" if first_lost < self.n_digit else "final_argmax"
            elif legal_rank is None:
                fate = "generated_but_illegal"
            elif legal_rank > self._final["top_k"]:
                fate = "rank_cutoff"
            else:
                fate = "topk_hit"
            rows.append({
                "trace_schema": TRACE_SCHEMA, "target_sid": target_values,
                "target_sid_in_catalog": target in self._catalog_set,
                "rounds": rounds, "first_gt_extinction_round": first_lost,
                "gt_compatible_final_branches": branches,
                "final_candidates": {
                    "path_count": len(sequences), "unique_count": len(set(sequences)),
                    "legal_path_count": sum(legal),
                    "legal_unique_count": len(legal_unique_order),
                    "target_generated": raw_rank is not None,
                    "target_path_count": sum(s == target for s in sequences),
                    "target_raw_path_rank": raw_rank, "target_legal_unique_rank": legal_rank,
                    "top_k": self._final["top_k"], "no_legal_candidates": not any(legal),
                },
                "fate": fate,
            })
        self._active = False
        self._last = None
        self._final = None
        self._rounds = []
        return rows
