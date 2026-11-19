"""Structural checks run without torch; CPU semantic/parity checks need torch."""
from __future__ import annotations

import ast
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load(ROOT / "scripts/analysis/item_local_beam_trace.py", "beam_trace_under_test")
BEAM_PATH = ROOT / "external/DiffGRM/genrec/models/DIFF_GRM/beam.py"
try:
    import torch
except ImportError:
    torch = None


class StructuralTests(unittest.TestCase):
    def test_instrumentation_only_adds_observer_calls(self):
        source = BEAM_PATH.read_text(encoding="utf8")
        original = trace._function_node(source)
        instrumented = trace.instrument_function_ast(source)
        compile(instrumented, "<test>", "exec")

        class RemoveProbes(ast.NodeTransformer):
            removed = 0

            def visit_Expr(self, node):
                call = node.value
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "_trace_observer"):
                    self.removed += 1
                    return None
                return node

        remover = RemoveProbes()
        recovered = remover.visit(instrumented).body[0]
        self.assertEqual(remover.removed, 5)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(recovered, include_attributes=False))

    def test_source_drift_fails_closed(self):
        source = BEAM_PATH.read_text(encoding="utf8").replace("TOP_K_FINAL = min(", "TOP_K_FINAL = max(")
        with self.assertRaisesRegex(ValueError, "AST does not match"):
            trace.instrument_function_ast(source)

    def test_line_endings_do_not_change_pin(self):
        source = BEAM_PATH.read_text(encoding="utf8")
        self.assertEqual(trace.function_ast_sha256(source), trace.function_ast_sha256(source.replace("\n", "\r\n")))


@unittest.skipIf(torch is None, "torch unavailable; run these on the owned grm-exp CPU runtime")
class TorchTests(unittest.TestCase):
    def setUp(self):
        self.beam = load(BEAM_PATH, "frozen_beam_for_trace_test")

    @staticmethod
    def tokenizer(catalog):
        lookup = {tuple(row): i + 1 for i, row in enumerate(catalog)}
        return SimpleNamespace(mask_token=-1, codebooks_to_item_id=lambda row: lookup.get(tuple(row)))

    @staticmethod
    def model(beam_size=6):
        config = {"current_split": "val", "dedup_strategy": "simple", "vectorized_beam_search": {
            "beam_act": beam_size, "beam_max": beam_size, "top_k_final": 3,
            "neg_inf_fp32": -1e9, "neg_inf_fp16": -1e4}}
        model = SimpleNamespace(n_digit=4, codebook_size=4, config=config, calls=0)

        def forward(batch, **kwargs):
            model.calls += 1
            ids, masks = batch["decoder_input_ids"], batch["mask_positions"]
            user = batch["encoder_hidden"][:, 0, 0]
            token = torch.arange(4)[None, None, :].float()
            pos = torch.arange(4)[None, :, None].float()
            context = ((ids * (1 - masks)).sum(-1) + user)[:, None, None]
            desired = (context + pos + 1) % 4
            logits = -(token - desired).square() + pos * 0.013 + token * 0.001
            return SimpleNamespace(logits=logits, past_key_values=None)

        model.forward_decoder_only = forward
        return model

    def test_original_parity_target_independence_rng_and_restoration(self):
        catalog = list(itertools.product(range(4), repeat=4))
        tokenizer = self.tokenizer(catalog)
        model = self.model()
        hidden = torch.tensor([[[0.]], [[1.]]])
        original = self.beam.iterative_mask_decode
        expected = original(model, hidden, n_return_sequences=3, tokenizer=tokenizer)
        observer = trace.BeamTraceObserver(catalog, -1, 4)
        rng = torch.get_rng_state().clone()
        labels_sets = [torch.tensor([[1, 2, 3, 0], [2, 3, 0, 1]]), torch.zeros(2, 4, dtype=torch.long)]
        rows_list = []
        for labels in labels_sets:
            observer.start_batch(labels)
            with trace.install_trace(self.beam, observer):
                actual = self.beam.iterative_mask_decode(model, hidden, n_return_sequences=3, tokenizer=tokenizer)
            rows = observer.finish_batch()
            rows_list.append(rows)
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertEqual(expected[1:], actual[1:])
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["round"] for r in rows[0]["rounds"]], [1, 2, 3, 4])
            for row, prediction in zip(rows, actual[0].tolist()):
                target = row["target_sid"]
                output_rank = prediction.index(target) + 1 if target in prediction else None
                traced_rank = row["final_candidates"]["target_legal_unique_rank"]
                self.assertEqual(output_rank, traced_rank if traced_rank is not None and traced_rank <= 3 else None)
            self.assertIs(self.beam.iterative_mask_decode, original)
        self.assertNotEqual(rows_list[0], rows_list[1])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(model.calls, 12)
        observer.start_batch(labels_sets[0])
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with trace.install_trace(self.beam, observer):
                raise RuntimeError("test exception")
        self.assertIs(self.beam.iterative_mask_decode, original)

    def test_partial_keys_include_mask_and_count_duplicate_paths(self):
        observer = trace.BeamTraceObserver([[0, 1, 2, 3], [2, 0, 1, 3]], -1, 4)
        observer.start_batch([[0, 1, 2, 3]])
        states = torch.tensor([[[0, -1, -1, -1], [-1, 0, -1, -1], [0, -1, -1, -1]]])
        observer._prepare_device(states)
        metrics = observer._state_metrics(states)
        self.assertEqual(metrics["retained_unique_states"].item(), 2)
        self.assertEqual(metrics["retained_duplicate_paths"].item(), 1)
        self.assertEqual(metrics["gt_compatible_after"].item(), 2)
        self.assertEqual(metrics["gt_compatible_unique_after"].item(), 1)
        self.assertEqual(metrics["catalog_completable_paths"].item(), 3)
        self.assertEqual(metrics["catalog_completable_unique_states"].item(), 2)

    def staged_observer(self, catalog, target, parents):
        observer = trace.BeamTraceObserver(catalog, -1, 4)
        observer.start_batch([target])
        first = parents.clone()
        first[:, :, 1:] = -1
        second = parents.clone()
        second[:, :, 2:] = -1
        observer.initial(first, None, -1, 4)
        observer.middle(2, first, second)
        observer.middle(3, second, parents)
        return observer

    def test_last_argmax_tie_rank_one_can_still_miss(self):
        parents = torch.tensor([[[0, 1, 2, -1]]])
        observer = self.staged_observer([[0, 1, 2, 0], [0, 1, 2, 1]], [0, 1, 2, 1], parents)
        logits = torch.tensor([[[0., 0., -1., -2.]]])
        chosen = logits.argmax(-1)
        observer.last_before(parents, torch.zeros(1, 1), logits, torch.tensor([[3]]), chosen)
        parents.scatter_(2, torch.tensor([[[3]]]), chosen.unsqueeze(-1))
        observer.candidates(parents, torch.zeros(1, 1), 1)
        row = observer.finish_batch()[0]
        self.assertEqual(row["fate"], "final_argmax")
        self.assertEqual(row["first_gt_extinction_round"], 4)
        branch = row["gt_compatible_final_branches"][0]
        self.assertEqual(branch["partial_sid"], [0, 1, 2, -1])
        self.assertEqual(branch["target_rank_strict"], 1)
        self.assertEqual(branch["target_rank_worst_tie"], 2)
        self.assertFalse(branch["selected_is_target"])

    def test_terminal_rank_cutoff(self):
        catalog = [[0, 0, d, t] for d in range(3) for t in range(4)]
        full = torch.tensor([catalog])
        parents = full.clone()
        parents[:, :, 3] = -1
        observer = self.staged_observer(catalog, catalog[10], parents)
        logits = torch.full((1, 12, 4), -1.)
        logits.scatter_(2, full[:, :, 3:].clone(), 1.)
        observer.last_before(parents, torch.zeros(1, 12), logits, torch.full((1, 12), 3), full[:, :, 3])
        observer.candidates(full, -torch.arange(12)[None, :].float(), 10)
        row = observer.finish_batch()[0]
        self.assertEqual(row["fate"], "rank_cutoff")
        self.assertIsNone(row["first_gt_extinction_round"])
        self.assertEqual(row["final_candidates"]["target_legal_unique_rank"], 11)
        self.assertEqual(row["final_candidates"]["legal_unique_count"], 12)

    def test_no_legal_candidates_preserves_original_fallback(self):
        catalog = [[1, 1, 1, 1]]
        tokenizer = self.tokenizer(catalog)
        model = self.model(beam_size=4)

        def forward(batch, **kwargs):
            size = batch["decoder_input_ids"].shape[0]
            logits = torch.tensor([4., 3., 2., 1.]).view(1, 1, 4).expand(size, 4, 4).clone()
            return SimpleNamespace(logits=logits, past_key_values=None)

        model.forward_decoder_only = forward
        hidden = torch.zeros(2, 1, 1)
        original = self.beam.iterative_mask_decode
        expected = original(model, hidden, n_return_sequences=3, tokenizer=tokenizer)
        observer = trace.BeamTraceObserver(catalog, -1, 4)
        observer.start_batch([[1, 1, 1, 1]] * 2)
        with trace.install_trace(self.beam, observer):
            actual = self.beam.iterative_mask_decode(model, hidden, n_return_sequences=3, tokenizer=tokenizer)
        rows = observer.finish_batch()
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertEqual(expected[1:], actual[1:])
        for row in rows:
            self.assertTrue(row["final_candidates"]["no_legal_candidates"])
            self.assertEqual(row["final_candidates"]["legal_unique_count"], 0)
            self.assertEqual(row["first_gt_extinction_round"], 1)
            self.assertEqual(row["fate"], "pruned_round_1")

    def test_first_extinction_rounds_one_two_three(self):
        for lost_round in (1, 2, 3):
            with self.subTest(lost_round=lost_round):
                observer = trace.BeamTraceObserver([[0, 0, 0, 0]], -1, 4)
                observer.start_batch([[0, 0, 0, 0]])
                states = torch.full((1, 1, 4), -1)
                for round_number in (1, 2, 3):
                    children = states.clone()
                    children[:, :, round_number - 1] = int(round_number == lost_round)
                    if round_number == 1:
                        observer.initial(children, None, -1, 4)
                    else:
                        observer.middle(round_number, states, children)
                    states = children
                logits = torch.tensor([[[0., -1., -2., -3.]]])
                observer.last_before(states, torch.zeros(1, 1), logits, torch.tensor([[3]]), torch.zeros(1, 1, dtype=torch.long))
                full = states.clone()
                full[:, :, 3] = 0
                observer.candidates(full, torch.zeros(1, 1), 1)
                row = observer.finish_batch()[0]
                self.assertEqual(row["first_gt_extinction_round"], lost_round)
                self.assertEqual(row["fate"], f"pruned_round_{lost_round}")

    def test_requires_validation_and_simple_dedup(self):
        observer = trace.BeamTraceObserver([[0, 1, 2, 3]], -1, 4)
        observer.start_batch([[0, 1, 2, 3]])
        model = self.model()
        model.config["current_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            observer.require_ready("confidence", model)
        model.config["current_split"] = "val"
        model.config["dedup_strategy"] = "weighted"
        with self.assertRaisesRegex(ValueError, "simple"):
            observer.require_ready("confidence", model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
