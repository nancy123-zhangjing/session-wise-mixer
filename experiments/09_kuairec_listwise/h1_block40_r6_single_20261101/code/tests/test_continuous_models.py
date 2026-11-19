"""Real CPU tensor gates for stage 06; no optional imports or skipped tests."""
import ast
import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import typing
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.continuous.data import load_split
from scripts.continuous.metrics import metrics_from_ranks, target_ranks, stable_top_items
from scripts.continuous.model import ContinuousRecommender, ModelConfig, TransformerBlock
from scripts.continuous.protocol import COMMON, canonical_hash, generate_search
from scripts.continuous.trainer import evaluate, fit, load_checkpoint, restore_model, seed_everything

torch.set_num_threads(2)


def official_classes():
    """Execute only inspected standalone class/function ASTs from locked sources.

    Avoid importing upstream experiment entry points or its dependency graph.
    The reference equations remain original author code, not a rewritten oracle.
    """
    ns = {"torch": torch, "nn": nn, "tp": typing}
    reference = ROOT / "experiments/06_continuous_baseline/reference_upstream"
    names = {"PointWiseFeedForward", "TransformerLayersBase", "PreLNTransformerLayer",
             "SwigluFeedForward", "init_feed_forward", "LiGRLayer"}
    for filename in ("net_blocks.py", "swiglu.py", "ligr.py"):
        tree = ast.parse((reference / filename).read_text(encoding="utf-8"))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, filename, "exec"), ns)
    return ns


def tiny_data():
    history = torch.tensor([[0, 0, 1, 2], [0, 0, 4, 5], [0, 0, 7, 8]] * 4)
    targets = torch.tensor([3, 6, 9] * 4)
    return {"histories": history, "targets": targets, "users": torch.arange(12),
            "positions": torch.ones(12, dtype=torch.long) * 2, "sha256": "synthetic-only"}


def tiny_config(model="sasrec_fullce"):
    return {**COMMON, "model": model, "n_items": 9, "max_length": 4,
            "dim": 16, "layers": 1, "heads": 2, "dropout": 0.1,
            "negatives": 8, "lr": 0.01, "weight_decay": 0.0,
            "batch_size": 4, "eval_batch_size": 5, "max_epochs": 3}


class SourceTests(unittest.TestCase):
    def test_reference_hashes(self):
        lock = json.loads((ROOT / "experiments/06_continuous_baseline/upstream_sources.json").read_text())
        self.assertEqual(len(lock["sources"]), 3)
        for source in lock["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["reference_path"]).read_bytes()).hexdigest(),
                             source["sha256"])

    def test_author_block_forward_and_gradients(self):
        official = official_classes()
        for model, name in (("sasrec_fullce", "PreLNTransformerLayer"), ("esasrec", "LiGRLayer")):
            with self.subTest(model=model):
                seed_everything(11)
                config = ModelConfig(model, n_items=9, dim=16, heads=2, dropout=0.0)
                ours = TransformerBlock(config).double()
                reference = official[name](16, 2, 0.0, ff_factors_multiplier=4).double()
                reference.load_state_dict(ours.state_dict(), strict=True)
                x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
                a, b = ours(x, mask), reference(y, mask, None)
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
                a.square().sum().backward()
                b.square().sum().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-10)
                for (left_name, left), (right_name, right) in zip(
                        ours.named_parameters(), reference.named_parameters()):
                    self.assertEqual(left_name, right_name)
                    torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(2024)

    def models(self):
        return [ContinuousRecommender(ModelConfig(m, n_items=9, max_length=6,
                dim=16, heads=2, layers=2, dropout=0.0)) for m in ("sasrec_fullce", "esasrec")]

    def test_causal_future_invariance(self):
        for model in self.models():
            first = model.encode_all(torch.tensor([[0, 1, 2, 3, 4]]))
            changed = model.encode_all(torch.tensor([[0, 1, 2, 8, 9]]))
            torch.testing.assert_close(first[:, 1:3], changed[:, 1:3], rtol=0, atol=0)

    def test_padding_invariance_and_no_padding_attention(self):
        for model in self.models():
            short = model(torch.tensor([[1, 2, 3]]))
            padded = model(torch.tensor([[0, 0, 0, 1, 2, 3]]))
            torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-6)
            with torch.no_grad():
                model.item_embedding.weight[0].fill_(10000)
            changed = model(torch.tensor([[0, 0, 0, 1, 2, 3]]))
            torch.testing.assert_close(padded, changed, rtol=0, atol=0)

    def test_one_item_histories_finite_forward_backward(self):
        for model in self.models():
            histories = torch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2]])
            loss = model.loss(histories, torch.tensor([2, 3]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            self.assertTrue(torch.equal(model.item_embedding.weight.grad[0], torch.zeros(16)))

    def test_catalog_size_and_tied_weights(self):
        for model in self.models():
            q = model(torch.tensor([[0, 1, 2]]))
            scores = model.score_catalog(q)
            self.assertEqual(scores.shape, (1, 9))
            torch.testing.assert_close(scores, q @ model.item_embedding.weight[1:].T)

    def test_fullce_matches_manual_logsumexp(self):
        model = self.models()[0]
        histories, targets = torch.tensor([[0, 1, 2], [0, 4, 5]]), torch.tensor([3, 6])
        scores = model.score_catalog(model(histories))
        expected = (scores.logsumexp(1) - scores.gather(1, (targets - 1)[:, None]).squeeze(1)).mean()
        torch.testing.assert_close(model.loss(histories, targets), expected)

    def test_sampled_loss_multiset_and_positive_collision(self):
        model = self.models()[1]
        histories, targets = torch.tensor([[0, 1, 2], [0, 4, 5]]), torch.tensor([3, 6])
        negatives = torch.tensor([[3, 3, 4, 8], [6, 4, 4, 9]])
        q = model(histories)
        scores = model.score_catalog(q)
        logits = torch.stack([scores[row, torch.cat((targets[row:row+1], negatives[row])) - 1]
                              for row in range(2)])
        expected = (logits.logsumexp(1) - logits[:, 0]).mean()
        torch.testing.assert_close(model.loss(histories, targets, negatives), expected)

    def test_synthetic_examples_fit(self):
        data = tiny_data()
        for model in self.models():
            before = model.loss(data["histories"], data["targets"]).item()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            for _ in range(100):
                optimizer.zero_grad()
                loss = model.loss(data["histories"], data["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            after = model.loss(data["histories"], data["targets"]).item()
            self.assertLess(after, before * 0.65)
            self.assertEqual(evaluate(model, data)["HR@5"], 1.0)

    def test_invalid_config_and_shapes(self):
        with self.assertRaises(ValueError):
            ModelConfig("unknown")
        with self.assertRaises(ValueError):
            ModelConfig("esasrec", dim=10, heads=3)
        with self.assertRaises(ValueError):
            self.models()[0](torch.ones(1, 7, dtype=torch.long))


class MetricAndProtocolTests(unittest.TestCase):
    def test_fast_topk_matches_full_stable_sort_including_signed_zero_and_ties(self):
        seed_everything(91)
        scores = torch.randn(13, 101).bfloat16().float()
        scores[0].zero_()
        scores[1, :6] = torch.tensor([-0.0, 0.0, 1e-45, -1e-45, 1e20, -1e20])
        for k in (1, 5, 20, 101):
            actual = stable_top_items(scores, k)
            expected = scores.argsort(dim=1, descending=True, stable=True)[:, :k] + 1
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    def test_rank_and_metrics_against_independent_python_sort(self):
        scores = torch.tensor([[2., 2., 3., 0.], [1., 1., 1., 1.], [9., 3., 2., 0.]])
        targets = torch.tensor([2, 4, 1])
        expected = []
        for row, target in zip(scores.tolist(), targets.tolist()):
            ranking = sorted(range(1, 5), key=lambda item: (-row[item - 1], item))
            expected.append(ranking.index(target) + 1)
        ranks = target_ranks(scores, targets)
        self.assertEqual(ranks.tolist(), expected)
        metrics = metrics_from_ranks(ranks, ks=(1, 2, 3, 4))
        for k in (1, 2, 3, 4):
            self.assertAlmostEqual(metrics[f"HR@{k}"], sum(r <= k for r in expected) / 3)
            self.assertAlmostEqual(metrics[f"NDCG@{k}"],
                sum(1 / math.log2(r + 1) if r <= k else 0 for r in expected) / 3)

    def test_search_manifest_frozen_and_anchors(self):
        first, second = generate_search(), generate_search()
        self.assertEqual(first, second)
        self.assertEqual(first["budget"]["valid_training_runs"], 89)
        checked = json.loads((ROOT / "configs/06_continuous_baseline/search_manifest.json").read_text())
        self.assertEqual(checked, first)
        self.assertEqual(len(first["trials"]), 64)
        for model in ("sasrec_fullce", "esasrec"):
            trials = [t for t in first["trials"] if t["config"]["model"] == model]
            self.assertEqual(len({t["config_sha256"] for t in trials}), 32)
            self.assertEqual({t["seed"] for t in trials}, {2024})
            self.assertTrue(all(canonical_hash(t["config"]) == t["config_sha256"] for t in trials))
            self.assertEqual([(t["config"]["dim"], t["config"]["layers"], t["config"]["heads"])
                              for t in trials[:2]], [(64, 1, 1), (256, 2, 4)])

    def test_dataset_boundary_rejects_pad_and_empty(self):
        parent = ROOT / "tmp/continuous_model_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp) / "full_train_prefix_v1"
            directory.mkdir()
            path = directory / "train.jsonl"
            base = {"user_id": 1, "target_position": 2, "target": 3}
            for history in ([], [0, 1], [99]):
                path.write_text(json.dumps({**base, "history": history}) + "\n")
                with self.assertRaises(ValueError):
                    load_split(tmp, "full_train_prefix_v1", "train", n_items=9, max_length=4)


class NumericalGateTests(unittest.TestCase):
    def test_gradient_gate_rejects_wrong_backward_with_unchanged_forward(self):
        from scripts.continuous.gpu_smoke import check_training_gradients
        data = tiny_data()
        config = ModelConfig("esasrec", n_items=9, max_length=4, dim=16,
                             heads=2, negatives=8)
        clean = ContinuousRecommender(config)
        proof = check_training_gradients(clean, data["histories"], data["targets"])
        self.assertEqual(proof["relative_gradient_l2"], 0.0)
        broken = ContinuousRecommender(config)
        calls = {}

        def wrong_second_backward(name):
            def hook(gradient):
                calls[name] = calls.get(name, 0) + 1
                return -gradient if calls[name] == 2 else gradient
            return hook

        for name, parameter in broken.named_parameters():
            parameter.register_hook(wrong_second_backward(name))
        # Forward/loss stay unchanged; only the second backward is corrupted.
        with self.assertRaisesRegex(FloatingPointError, "gradients disagree"):
            check_training_gradients(broken, data["histories"], data["targets"])


class RecoveryTests(unittest.TestCase):
    def test_crash_between_aliases_and_commit_recovers_last_complete_epoch(self):
        from scripts.continuous import checkpoints
        parent = ROOT / "tmp/continuous_model_tests"
        parent.mkdir(parents=True, exist_ok=True)
        for boundary in ("CURRENT.json", "last.receipt.json"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(dir=parent) as tmp:
                config, data = tiny_config(), tiny_data()
                reference, broken = Path(tmp)/"reference", Path(tmp)/"broken"
                with contextlib.redirect_stdout(io.StringIO()):
                    fit(config, 44, "synthetic", data, data, reference, smoke=True)
                    fit(config, 44, "synthetic", data, data, broken, smoke=True, stop_after_epochs=1)
                original = checkpoints.atomic_json

                def interrupted(path, value):
                    if Path(path).name == boundary and value.get("epoch") == 2:
                        raise OSError("injected save interruption")
                    return original(path, value)

                with mock.patch.object(checkpoints, "atomic_json", side_effect=interrupted):
                    with self.assertRaises(OSError), contextlib.redirect_stdout(io.StringIO()):
                        fit(config, 44, "synthetic", data, data, broken, smoke=True, resume=True)
                self.assertEqual(checkpoints.repair_committed(broken)["epoch"], 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    fit(config, 44, "synthetic", data, data, broken, smoke=True, resume=True)
                expected, actual = load_checkpoint(reference/"last.pt"), load_checkpoint(broken/"last.pt")
                for key, value in expected["model"].items():
                    torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
                self.assertLessEqual(len(list((broken/"checkpoint_epochs").glob("*.pt"))), 2)

    def test_checkpoint_load_and_exact_epoch_resume(self):
        parent = ROOT / "tmp/continuous_model_tests"
        parent.mkdir(parents=True, exist_ok=True)
        for model in ("sasrec_fullce", "esasrec"):
            with self.subTest(model=model), tempfile.TemporaryDirectory(dir=parent) as tmp:
                config, data = tiny_config(model), tiny_data()
                with contextlib.redirect_stdout(io.StringIO()):
                    fit(config, 21, "synthetic", data, data, Path(tmp)/"full", smoke=True)
                    fit(config, 21, "synthetic", data, data, Path(tmp)/"resume",
                        smoke=True, stop_after_epochs=1)
                    fit(config, 21, "synthetic", data, data, Path(tmp)/"resume", smoke=True, resume=True)
                complete = load_checkpoint(Path(tmp)/"full/last.pt")
                resumed = load_checkpoint(Path(tmp)/"resume/last.pt")
                self.assertEqual(complete["updates"], resumed["updates"])
                for name, tensor in complete["model"].items():
                    torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
                for left, right in zip(complete["optimizer"]["state"].values(),
                                       resumed["optimizer"]["state"].values()):
                    for key in left:
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                a, _ = restore_model(Path(tmp)/"full/best.pt")
                b, _ = restore_model(Path(tmp)/"resume/best.pt")
                torch.testing.assert_close(a(data["histories"]), b(data["histories"]), rtol=0, atol=0)
                self.assertFalse((Path(tmp)/"resume/COMPLETE.json").exists())

    def test_tampering_and_wrong_resume_binding_rejected(self):
        parent = ROOT / "tmp/continuous_model_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            config, data = tiny_config(), tiny_data()
            output = Path(tmp)/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                fit(config, 1, "synthetic", data, data, output, smoke=True, stop_after_epochs=1)
            with self.assertRaises(ValueError):
                fit({**config, "lr": 0.02}, 1, "synthetic", data, data, output, smoke=True, resume=True)
            with (output/"last.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                load_checkpoint(output/"last.pt")

    def test_formal_completion_and_no_overwrite(self):
        parent = ROOT / "tmp/continuous_model_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            config, data = tiny_config(), tiny_data()
            output = Path(tmp)/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                result = fit(config, 1, "synthetic", data, data, output)
            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(result["validation_samples"], 12)
            self.assertTrue((output/"COMPLETE.json").exists())
            with self.assertRaises(FileExistsError):
                fit(config, 1, "synthetic", data, data, output)
            with self.assertRaises(FileExistsError):
                fit(config, 1, "synthetic", data, data, output, resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
