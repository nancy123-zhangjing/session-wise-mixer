"""CPU toy-oracle coverage for catalog-safe global list CPD."""
import inspect
import itertools
import math
import unittest

import numpy as np
import torch

from scripts.listwise_diffgrm.decoding import (
    SIDCatalog, GenerationOutput, aggregate_candidates, generate_lists,
)


def catalog(sids, ids=None, errors=None, originals=None, vocab=2):
    count = len(sids)
    return SIDCatalog(np.asarray(ids if ids is not None else range(1, count + 1), dtype=np.int64),
                      np.asarray(sids, dtype=np.int64),
                      np.asarray(errors if errors is not None else [0.0] * count),
                      original_item_ids=originals, codebook_size=vocab)


def rule(partial, item_index=None, prefix=()):
    """A tiny conditional model; oracle independently enumerates its paths."""
    visible = sum(code + 1 for code in partial if code >= 0)
    prefix_sum = sum(sum(item) for item in prefix)
    j = item_index or 0
    return [[0.4 * (position + 1) + 0.13 * visible,
             -0.2 * position + 0.19 * prefix_sum + 0.17 * j]
            for position in range(len(partial))]


class ToyModel(torch.nn.Module):
    def __init__(self, original_length=1):
        super().__init__()
        self.original_length = original_length
        self.dropout = torch.nn.Dropout(0.1)
        self.calls = []

    def decode_logits(self, history_sid, history_mask, partial_sid, item_index=None):
        self.calls.append((history_sid.clone(), history_mask.clone(), partial_sid.clone(), item_index))
        rows = []
        for history, partial in zip(history_sid.tolist(), partial_sid.tolist()):
            prefix = history[self.original_length:] if item_index is not None else ()
            rows.append(rule(partial, item_index, prefix))
        return torch.tensor(rows, dtype=torch.float32, device=partial_sid.device)


def oracle(route, cat, length):
    """Exhaust every fill order and code choice, with no decoder helpers."""
    digits = cat.n_digit
    valid_sids = set(cat.buckets)
    finals = {}

    def visit(state, score):
        if -1 not in state:
            pieces = [tuple(state[i:i + digits]) for i in range(0, len(state), digits)]
            if any(sid not in valid_sids for sid in pieces):
                return
            if any(pieces.count(sid) > len(cat.buckets[sid]) for sid in set(pieces)):
                return
            finals[state] = max(finals.get(state, -math.inf), score)
            return
        if route == 'H':
            item = next(i for i in range(length) if -1 in state[i * digits:(i + 1) * digits])
            start = item * digits
            part = state[start:start + digits]
            prefix = [state[i:i + digits] for i in range(0, start, digits)]
        else:
            item, start, part, prefix = None, 0, state, ()
        logits = rule(part, item, prefix)
        for local, code in enumerate(part):
            if code != -1:
                continue
            maximum = max(logits[local])
            denominator = sum(math.exp(value - maximum) for value in logits[local])
            for value in range(cat.codebook_size):
                updated = list(state)
                updated[start + local] = value
                pieces = [updated[i:i + digits] for i in range(0, len(updated), digits)]
                if any(not any(all(a == -1 or a == b for a, b in zip(piece, sid)) for sid in valid_sids) for piece in pieces):
                    continue
                logp = logits[local][value] - maximum - math.log(denominator)
                visit(tuple(updated), score + logp)

    visit(tuple([-1] * (length * digits)), 0.0)
    return finals


class CatalogTests(unittest.TestCase):
    def test_arbitrary_digit_compatibility_is_not_a_left_prefix_tree(self):
        cat = catalog([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]])
        mask = cat.legal_token_mask(np.array([[-1, 1, -1, 0]]))[0]
        self.assertEqual(mask[0].tolist(), [True, True])
        self.assertEqual(mask[2].tolist(), [True, True])
        self.assertFalse(mask[1].any())
        self.assertFalse(mask[3].any())
        self.assertFalse(cat.legal_token_mask(np.array([[1, -1, 1, -1]])).any())

    def test_collision_resolution_uses_error_then_original_id_and_slot_order(self):
        cat = catalog([[0, 0]] * 3, ids=[10, 20, 30], errors=[0.2, 0.1, 0.1], originals=np.array([10, 300, 200]))
        self.assertEqual(cat.resolve([[0, 0]] * 4), [30, 20, 10, 0])
        self.assertEqual(cat.resolve([[-1, 0], [0, 0], [0, -1], [0, 0]]), [0, 30, 0, 20])
        self.assertTrue(cat.can_complete(np.array([[0, 0]] * 3)))
        self.assertFalse(cat.can_complete(np.array([[0, 0]] * 4)))

    def test_matching_detects_shared_candidate_shortage(self):
        cat = catalog([[0, 0], [0, 1], [1, 1]])
        self.assertFalse(cat.can_complete(np.array([[0, -1], [0, 0], [0, 1]])))
        self.assertTrue(cat.can_complete(np.array([[-1, -1], [0, 0], [0, 1]])))

    def test_catalog_refresh_uses_new_assignments_not_stale_partial_cache(self):
        old = catalog([[0, 0]], ids=[9])
        old.legal_token_mask(np.array([[0, -1]]))
        new = catalog([[1, 1]], ids=[9])
        self.assertFalse(new.legal_token_mask(np.array([[0, -1]])).any())
        self.assertEqual(new.resolve([[0, 0], [1, 1]]), [0, 9])

    def test_rejects_pad_ids_invalid_codes_and_unusable_errors(self):
        for kwargs in [dict(ids=[0]), dict(sids=[[0, 2]]), dict(errors=[float('nan')])]:
            args = dict(sids=[[0, 0]])
            args.update(kwargs)
            with self.assertRaises(ValueError):
                catalog(**args)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.history = torch.tensor([[[0, 1]]])
        self.mask = torch.ones((1, 1), dtype=torch.bool)

    def test_both_routes_match_exhaustive_max_path_scores(self):
        cat = catalog(list(itertools.product(range(2), repeat=2)))
        for route in ('J', 'H'):
            with self.subTest(route=route):
                model = ToyModel()
                output = generate_lists(model, self.history, self.mask, cat, route=route, list_length=2, beam_size=32)
                expected = oracle(route, cat, 2)
                actual = {tuple(sid.reshape(-1).tolist()): float(score) for sid, score, present in
                          zip(output.sid[0], output.scores[0], output.beam_mask[0]) if present}
                self.assertEqual(actual.keys(), expected.keys())
                for sid in expected:
                    self.assertAlmostEqual(actual[sid], expected[sid], places=5)
                self.assertEqual(int(output.beam_mask.sum()), 12)
                self.assertTrue(output.complete_mask[output.beam_mask].all())
                self.assertTrue(torch.isneginf(output.scores[~output.beam_mask]).all())
                self.assertEqual(output.counters['model_forward_calls'], 4)

    def test_hierarchical_prefixes_are_generated_and_global_width_is_bounded(self):
        cat = catalog(list(itertools.product(range(2), repeat=2)))
        model = ToyModel()
        histories = self.history.repeat(2, 1, 1)
        output = generate_lists(model, histories, self.mask.repeat(2, 1), cat, route='H', list_length=2, beam_size=3)
        self.assertLessEqual(int(output.beam_mask.sum(dim=1).max()), 3)
        for history, masks, partial, item in model.calls:
            self.assertLessEqual(history.shape[0], 2 * 3)
            self.assertEqual(history.shape[1], 1 + item)
            self.assertEqual(partial.shape[1], 2)
            self.assertTrue((history[:, 0] == self.history[0, 0]).all())
            if item:
                self.assertTrue((history[:, 1:] >= 0).all())
                self.assertTrue(masks[:, 1:].all())
        self.assertNotIn('target', inspect.signature(generate_lists).parameters)
        self.assertNotIn('labels', inspect.signature(generate_lists).parameters)

    def test_seen_history_items_are_not_excluded(self):
        cat = catalog([[0, 1]], ids=[17])
        output = generate_lists(ToyModel(), self.history, self.mask, cat, list_length=1, beam_size=4)
        self.assertEqual(output.best_item_ids.tolist(), [[17]])
        self.assertEqual(int(output.beam_mask.sum()), 1)

    def test_equal_sid_slots_can_resolve_distinct_collision_items(self):
        cat = catalog([[0, 0], [0, 0]], ids=[9, 4], errors=[0.2, 0.1])
        for route in ('J', 'H'):
            output = generate_lists(ToyModel(), self.history, self.mask, cat, route=route, list_length=2, beam_size=8)
            self.assertEqual(output.best_item_ids.tolist(), [[4, 9]])
            self.assertTrue(output.complete_mask[0, 0])

    def test_too_small_catalog_pads_without_repeating_or_fabricating_items(self):
        cat = catalog([[0, 0]], ids=[7])
        for route in ('J', 'H'):
            output = generate_lists(ToyModel(), self.history, self.mask, cat, route=route, list_length=3, beam_size=4)
            self.assertEqual(output.best_item_ids.tolist(), [[7, 0, 0]])
            self.assertFalse(output.complete_mask.any())
            self.assertEqual(int(output.beam_mask.sum()), 1)
            aggregate = aggregate_candidates(output, topk=10)
            self.assertEqual(int(aggregate.candidate_counts[0]), 0)

    def test_no_finite_legal_path_is_reported_as_missing(self):
        class Impossible(ToyModel):
            def decode_logits(self, history_sid, history_mask, partial_sid, item_index=None):
                return torch.full((*partial_sid.shape, 2), float('-inf'))
        output = generate_lists(Impossible(), self.history, self.mask, catalog([[0, 0]]), list_length=1)
        self.assertEqual(output.best_item_ids.tolist(), [[0]])
        self.assertFalse(output.complete_mask.any())
        self.assertFalse(torch.isnan(output.scores).any())

    def test_more_beams_can_recover_a_lower_initial_probability_better_path(self):
        class Trap(ToyModel):
            def decode_logits(self, history_sid, history_mask, partial_sid, item_index=None):
                rows = []
                for first, second in partial_sid.tolist():
                    p = 0.05 if first < 0 else 0.1 if first == 0 else 0.9
                    rows.append([[math.log(0.7), math.log(0.3)], [math.log(p), math.log(1-p)]])
                return torch.tensor(rows)
        cat = catalog([[0, 0], [1, 0]])
        narrow = generate_lists(Trap(), self.history, self.mask, cat, list_length=1, beam_size=1)
        wider = generate_lists(Trap(), self.history, self.mask, cat, list_length=1, beam_size=2)
        self.assertEqual(narrow.best_item_ids.tolist(), [[1]])
        self.assertEqual(wider.best_item_ids.tolist(), [[2]])
        self.assertAlmostEqual(float(narrow.scores[0, 0]), math.log(.07), places=5)
        self.assertAlmostEqual(float(wider.scores[0, 0]), math.log(.27), places=5)

    def test_generation_restores_individual_training_modes(self):
        model = ToyModel().train()
        model.dropout.eval()
        generate_lists(model, self.history, self.mask, catalog([[0, 0]]), list_length=1)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    def test_request_local_context_cache_matches_uncached_parent_reordering(self):
        class ContextSensitive(ToyModel):
            def decode_logits(self, history_sid, history_mask, partial_sid, item_index=None):
                logits = super().decode_logits(history_sid, history_mask, partial_sid, item_index)
                total = history_sid.masked_fill(~history_mask[:, :, None], 0).sum((1, 2))
                logits[:, :, 1] += total[:, None] * 0.37
                return logits

        class Cached(ContextSensitive):
            def __init__(self):
                super().__init__()
                self.encodes = []

            def encode_context(self, history_sid, history_mask):
                self.encodes.append((history_sid.clone(), history_mask.clone()))
                return history_sid.masked_fill(~history_mask[:, :, None], -1)

            def decode_with_context(self, encoded, partial_sid, item_index=None):
                return ContextSensitive.decode_logits(self, encoded, encoded[:, :, 0] >= 0,
                                                       partial_sid, item_index)

            def decode_logits(self, *args, **kwargs):
                raise AssertionError('The uncached interface must not be called')

        cat = catalog(list(itertools.product(range(2), repeat=2)))
        histories = torch.tensor([[[0, 0]], [[1, 1]]])
        masks = torch.ones((2, 1), dtype=torch.bool)
        for route in ('J', 'H'):
            cached = Cached()
            expected = generate_lists(ContextSensitive(), histories, masks, cat,
                                      route=route, list_length=2, beam_size=3)
            actual = generate_lists(cached, histories, masks, cat,
                                    route=route, list_length=2, beam_size=3)
            torch.testing.assert_close(actual.item_ids, expected.item_ids)
            torch.testing.assert_close(actual.sid, expected.sid)
            torch.testing.assert_close(actual.scores, expected.scores)
            self.assertEqual(actual.counters['encoder_forward_calls'], 1 if route == 'J' else 2)
            self.assertEqual(len(cached.encodes), 1 if route == 'J' else 2)
            self.assertEqual(cached.encodes[0][0].shape[0], 2)
            if route == 'H':
                self.assertEqual(cached.encodes[1][0].shape, (6, 2, 2))
            # A second request builds fresh context instead of retaining a
            # previous call's history or generated prefix inside the model.
            other = generate_lists(cached, histories.flip(0), masks, cat,
                                   route=route, list_length=2, beam_size=3)
            torch.testing.assert_close(other.item_ids, actual.item_ids.flip(0))


class AggregationTests(unittest.TestCase):
    def test_order_list_budget_dedup_and_underfill(self):
        output = GenerationOutput(
            torch.tensor([[[1, 2], [3, 2], [2, 4], [9, 0]]]),
            torch.zeros((1, 4, 2, 2), dtype=torch.long),
            torch.tensor([[2.0, 3.0, 1.0, 100.0]]),
            torch.ones((1, 4), dtype=torch.bool),
            torch.tensor([[True, True, True, False]]), {},
        )
        self.assertEqual(aggregate_candidates(output, topk=6).item_ids.tolist(), [[3, 2, 1, 4, 0, 0]])
        limited = aggregate_candidates(output, topk=6, max_lists=2)
        self.assertEqual(limited.item_ids.tolist(), [[3, 2, 1, 0, 0, 0]])
        self.assertEqual(limited.candidate_counts.tolist(), [3])
        self.assertEqual(limited.lists_used.tolist(), [2])


if __name__ == '__main__':
    unittest.main()
