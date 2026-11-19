"""New RQCD numerical tests. Synthetic data only; no historical model inference."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rqcd.common import (load_config, seed_all, learning_rate, derive_seed,
                                 atomic_save, sha256, cpu_copy, source_binding)
from scripts.rqcd.model import ContinuousListModel, Diffusion
from scripts.rqcd.tokenizer import RQVAE, sinkhorn_ids, health, latent_scaling
from scripts.rqcd.catalog import Catalog, generate, evaluation_noise
from scripts.rqcd.data import PackedSplit
from scripts.rqcd.training import Trainer
from scripts.rqcd.run import fit_tokenizer, train_recommender, test_recommender
from scripts.rqcd.analysis import clustered_interval, analyze
from scripts.listwise_diffgrm.assets import BOUND_FILES, dataset_binding

torch.set_num_threads(2)


def configuration():
    cfg = load_config()
    cfg.update(history_max=7, catalog_size=24)
    cfg["split_counts"] = dict(train=11, validation=7, test=6)
    cfg["rq"].update(input_dim=8, hidden_dims=[16, 8], latent_dim=4, levels=4, codes=4,
                     batch_size=8, max_updates=4, eval_interval=2, warmup_updates=1,
                     min_utilization=0.0, min_unique_ratio=0.0,
                     sinkhorn_iterations=10, kmeans_iterations=10)
    cfg["model"].update(width=16, heads=4, ffn=32, dropout=0.0, decoder_layers=2)
    cfg["diffusion"].update(steps=20, sample_steps=4)
    cfg["training"].update(epochs=2, effective_batch=6, microbatch=2, warmup_updates=1,
                           precision="fp32", validation_epochs=[1, 2], probe_times=[2, 10, 18],
                           checkpoint_interval=1)
    cfg["runtime"].update(fused_adamw=False, data_device="cpu", sdpa_backend="math")
    cfg["evaluation"]["batch_size"] = 3
    return cfg


def asset(cfg):
    g = torch.Generator().manual_seed(72)
    c = cfg["rq"]
    z = torch.randn(cfg["catalog_size"], c["latent_dim"], generator=g)
    books = torch.randn(c["levels"], c["codes"], c["latent_dim"], generator=g)
    res, ids = z.clone(), []
    for book in books:
        code = (res[:, None]-book).square().sum(-1).argmin(-1)
        ids.append(code)
        res -= book[code]
    return {"z": z, "sid": torch.stack(ids, -1), "books": books,
            "mean": z.mean(0), "scale": z.std(), "errors": torch.rand(len(z), generator=g),
            "original_ids": torch.arange(len(z))*2}


def examples(cfg, b=3):
    n = cfg["catalog_size"]
    h = torch.arange(b*cfg["history_max"]).reshape(b, -1) % n + 1
    h[0, :3] = 0
    h[1:, :1] = 0
    target = torch.arange(b*cfg["list_length"]).reshape(b, -1) % n + 1
    return h, target


def fixture(root, cfg):
    root.mkdir()
    n = cfg["catalog_size"]
    (root/"config.json").write_text(json.dumps({"history_max": cfg["history_max"], "list_length": 5}))
    (root/"item_map.json").write_text(json.dumps({"pad_id": 0, "entries": [
        {"id": i+1, "original_id": i*2} for i in range(n)]}))
    (root/"item_text.jsonl").write_text("\n".join(json.dumps({"id": i}) for i in range(n)))
    (root/"train_seen.json").write_text(json.dumps({"user_ids": [1, 2], "item_ids": list(range(1, n+1))}))
    pq.write_table(pa.table({"item_id": list(range(1, n+1))}), root/"item_catalog.parquet")
    for split, count in cfg["split_counts"].items():
        rows = []
        for i in range(count):
            length = 3+i % 5
            rows.append({
                "sample_id": f"{split}-{i}", "user_id": i % 3+1,
                "history_item_ids": [1+(i+j) % n for j in range(length)],
                "history_timestamp_us": list(range(1, length+1)),
                "target_item_ids": [1+(i+j) % n for j in range(5)],
                "target_timestamp_us": list(range(100+i*5, 105+i*5)),
                "target_mask": [True]*5,
            })
        pq.write_table(pa.Table.from_pylist(rows), root/f"{split}.parquet", row_group_size=2)
    (root/"manifest.json").write_text(json.dumps({"artifacts": [
        {"path": name, "bytes": (root/name).stat().st_size, "sha256": sha256(root/name)}
        for name in BOUND_FILES]}))
    return root


class RQCDTests(unittest.TestCase):
    def setUp(self):
        seed_all(2024)
        self.cfg = configuration()
        self.catalog = Catalog(asset(self.cfg))

    def test_batched_views_match_sequential_outputs_and_gradients(self):
        a = ContinuousListModel(self.cfg)
        b = copy.deepcopy(a)
        history, target = examples(self.cfg)
        h, y = self.catalog.y[history], self.catalog.y[target]
        times = torch.randint(1, 21, target.shape)
        noisy = torch.randn_like(y)
        actual = a.training_prediction(h, history.ne(0), y, noisy, times)
        mem = b.encode_history(h, history.ne(0))
        expected = torch.stack([b.predict(noisy[:, j], times[:, j], y[:, :j], j, mem, history.ne(0))
                                for j in range(5)], 1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4, msg=name)

    def test_no_clean_current_or_future_target_leakage(self):
        model = ContinuousListModel(self.cfg).eval()
        history, target = examples(self.cfg)
        y = self.catalog.y[target]
        noisy, times = torch.randn_like(y), torch.full(target.shape, 10)
        before = model.training_prediction(self.catalog.y[history], history.ne(0), y, noisy, times)
        changed = y.clone()
        changed[:, 2:] += 100
        after = model.training_prediction(self.catalog.y[history], history.ne(0), changed, noisy, times)
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        self.assertGreater(float((before[:, 3:]-after[:, 3:]).abs().max()), 1e-3)
        # Padded historical vectors cannot influence the output either.
        h = self.catalog.y[history].clone()
        h[history == 0] = 1000
        padded = model.training_prediction(h, history.ne(0), y, noisy, times)
        torch.testing.assert_close(before, padded, rtol=0, atol=0)

    def test_cached_cross_attention_matches_uncached(self):
        model = ContinuousListModel(self.cfg).eval()
        history, target = examples(self.cfg)
        mem = model.encode_history(self.catalog.y[history], history.ne(0))
        noisy, times = torch.randn(3, 4), torch.full((3,), 8)
        prefix = self.catalog.y[target[:, :3]]
        a = model.predict(noisy, times, prefix, 3, mem, history.ne(0))
        b = model.predict(noisy, times, prefix, 3, mem, history.ne(0), model.cache_history(mem))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_diffusion_clipped_schedule_grid_and_oracle(self):
        cfg = load_config()
        diffusion = Diffusion(cfg)
        self.assertEqual(len(set(diffusion.grid())), 50)
        self.assertEqual(diffusion.grid()[0], 1000)
        self.assertEqual(diffusion.grid()[-1], 1)
        self.assertTrue(bool((diffusion.alpha[1:] < diffusion.alpha[:-1]).all()))
        self.assertGreater(float(diffusion.alpha[-1]), 0)
        clean = torch.randn(2, 32)
        result = diffusion.sample(torch.randn_like(clean), lambda _, t: clean)
        torch.testing.assert_close(result, clean, atol=0, rtol=0)
        self.assertEqual(learning_rate(500, 24255, 500, 1e-4), 1e-4)
        self.assertEqual(learning_rate(24255, 24255, 500, 1e-4), 0)

    def test_real_latent_quantizes_to_its_exported_sid(self):
        a = asset(self.cfg)
        empty = torch.empty(len(a["z"]), 0, dtype=torch.long)
        _, d = self.catalog.quantize(a["z"], empty)
        torch.testing.assert_close(d["sid"], a["sid"], atol=0, rtol=0)
        self.assertFalse(bool(d["changed"].any()))

    def test_catalog_matches_scalar_reference_and_deduplicates_collisions(self):
        a = asset(self.cfg)
        z = torch.randn(3, 4)*3
        selected = torch.tensor([[1, 2], [3, 4], [5, 6]])
        actual, diag = self.catalog.quantize(z, selected)
        for b in range(3):
            candidates = [i for i in range(len(a["z"])) if i+1 not in selected[b].tolist()]
            res, codes = z[b].clone(), []
            for level, book in enumerate(a["books"]):
                allowed = sorted(set(int(a["sid"][i, level]) for i in candidates))
                code = min(allowed, key=lambda c: (float((res-book[c]).square().sum()), c))
                codes.append(code)
                res -= book[code]
                candidates = [i for i in candidates if int(a["sid"][i, level]) == code]
            choice = min(candidates, key=lambda i: (float(a["errors"][i]), int(a["original_ids"][i]), i+1))
            self.assertEqual(int(actual[b]), choice+1)
            self.assertEqual(diag["sid"][b].tolist(), codes)
        model, diffusion = ContinuousListModel(self.cfg).eval(), Diffusion(self.cfg)
        history, _ = examples(self.cfg)
        result, diagnostics = generate(model, diffusion, self.catalog, history,
                                       ["a", "b", "c"], "validation", self.cfg)
        for row in result:
            self.assertEqual(len(set(row.tolist())), 5)
            self.assertTrue(bool((row > 0).all()))
        self.assertTrue(bool(diagnostics["complete"].all()))

    def test_noise_independent_of_batch_order(self):
        whole = evaluation_noise(self.cfg, ["a", "b", "c"], "validation", 2, 4)
        reordered = evaluation_noise(self.cfg, ["c", "a"], "validation", 2, 4)
        torch.testing.assert_close(whole[[2, 0]], reordered, atol=0, rtol=0)
        different = evaluation_noise(self.cfg, ["a", "b", "c"], "test", 2, 4)
        self.assertFalse(torch.equal(whole, different))

    def test_tail_accumulation_matches_whole_group(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"].update(microbatch=5, effective_batch=6)
        seed_all(17)
        whole = Trainer(cfg, copy.deepcopy(self.catalog), 11, {})
        cfg2 = copy.deepcopy(cfg)
        cfg2["training"]["microbatch"] = 2
        seed_all(17)
        chunked = Trainer(cfg2, copy.deepcopy(self.catalog), 11, {})
        history, targets = examples(cfg, 5)  # 2+2+1, not divide by three blindly
        draws = whole.group_noise(5)
        a = whole.update(history, targets, draws=draws)
        b = chunked.update(history, targets, draws=draws)
        torch.testing.assert_close(a["loss"], b["loss"], atol=1e-6, rtol=1e-5)
        for (name, p), (_, q) in zip(whole.model.named_parameters(), chunked.model.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4, msg=name)
        self.catalog.verify_frozen()
        whole.catalog.verify_frozen()
        self.assertEqual(list(whole.catalog.parameters()), [])

    def test_resume_exact_with_dropout_rng_optimizer_and_noise(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["dropout"] = 0.1
        trainer = Trainer(cfg, copy.deepcopy(self.catalog), 11, {"test": 1})
        history, targets = examples(cfg)
        trainer.update(history, targets)
        state = trainer.state()
        expected = trainer.update(history, targets)
        expected_state = cpu_copy(trainer.model.state_dict())
        recovered = Trainer(cfg, copy.deepcopy(self.catalog), 11, {"test": 1})
        recovered.restore(state)
        actual = recovered.update(history, targets)
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
        for key, value in recovered.model.state_dict().items():
            torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0, msg=key)
        bad = copy.deepcopy(state)
        bad["binding"] = {"wrong": 1}
        with self.assertRaises(ValueError):
            recovered.restore(bad)
        bad = copy.deepcopy(state)
        bad["updates"] = 999
        with self.assertRaises(ValueError):
            recovered.restore(bad)

    def test_rq_eval_is_deterministic_and_sinkhorn_is_training_only(self):
        model = RQVAE(self.cfg)
        content = F.normalize(torch.randn(24, 8), dim=-1)
        with self.assertRaises(RuntimeError):
            model(content[:8])
        iterations = model.initialize(content[:8], torch.Generator().manual_seed(19))
        self.assertTrue(all(i <= 10 for i in iterations))
        model.train()
        output = model(content[:8])
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.books.grad).all())
        self.assertTrue(torch.isfinite(model.encoder[0].weight.grad).all())
        model.eval()
        a = model(content)
        b = torch.cat([model(x)["sid"] for x in content.split(3)])
        torch.testing.assert_close(a["sid"], b, atol=0, rtol=0)
        res = model.encoder(content)
        expected = []
        for book in model.books:
            code = (res[:, None]-book).square().sum(-1).argmin(-1)
            expected.append(code)
            res = res-book[code]
        torch.testing.assert_close(a["sid"], torch.stack(expected, -1), atol=0, rtol=0)
        h = health(a["sid"], 4)
        self.assertTrue(0 <= h["collision_rate"] <= 1)

    def test_rq_loss_and_gradients_match_existing_residual_semantics(self):
        model = RQVAE(self.cfg)
        x = F.normalize(torch.randn(8, 8), dim=-1)
        model.initialize(x, torch.Generator().manual_seed(11))
        reference = copy.deepcopy(model)
        actual = model(x)
        res = reference.encoder(x)
        qloss, embs = torch.zeros(len(x)), []
        for l, book in enumerate(reference.books):
            # Existing GenRec uses expanded squared distances.
            distances = res.square().sum(1, keepdim=True)+book.square().sum(1)[None]-2*res @ book.T
            code = (sinkhorn_ids(distances.detach(), 10, 0.003) if l == 3
                    else distances.detach().argmin(1))
            value = book[code]
            qloss += (res.detach()-value).square().sum(1)+0.25*(res-value.detach()).square().sum(1)
            out = res+(value-res).detach()
            res = res-out
            embs.append(out)
        rec = F.normalize(reference.decoder(torch.stack(embs).sum(0)), dim=-1)
        expected = ((rec-x).square().sum(1)+qloss).mean()
        torch.testing.assert_close(actual["loss"], expected)
        actual["loss"].backward()
        expected.backward()
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_scaling_uses_target_counts_and_round_trips(self):
        z = torch.randn(7, 4)
        ids = torch.tensor([[1, 1, 2, 3, 4], [1, 2, 3, 4, 7]])
        mean, scale = latent_scaling(z, ids)
        expanded = z[ids.flatten()-1].double()
        torch.testing.assert_close(mean, expanded.mean(0))
        torch.testing.assert_close(scale, ((expanded-mean).square().mean()).sqrt())
        restored = ((z-mean.float())/scale.float())*scale.float()+mean.float()
        torch.testing.assert_close(restored, z, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            latent_scaling(torch.ones_like(z), ids)

    def test_parquet_packing_and_full_miniature_lifecycle(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.cfg
            dataset = fixture(root/"data", cfg)
            content = root/"content.npy"
            np.save(content, F.normalize(torch.randn(24, 8), dim=-1).numpy())
            cfg["content_sha256"] = sha256(content)
            packed = PackedSplit(dataset, "train", cfg)
            self.assertEqual(len(packed), 11)
            self.assertEqual(packed.history[0].tolist(), [0, 0, 0, 0, 1, 2, 3])
            self.assertEqual(packed.target[0].tolist(), [1, 2, 3, 4, 5])
            rq_args = Namespace(device="cpu", dataset=dataset, content=content, output=root/"rq", resume=False)
            fit_tokenizer(rq_args, cfg)
            self.assertTrue((root/"rq/COMPLETE.json").exists())
            binding = dataset_binding(dataset)
            probe_path = root/"probe.json"
            probe_path.write_text(json.dumps({
                "count": 2, "population": 7,
                "validation_sha256": binding["files"]["validation.parquet"]["sha256"],
                "rows": [{"index": 0, "sample_id": "validation-0"}, {"index": 5, "sample_id": "validation-5"}]}))
            args = Namespace(device="cpu", dataset=dataset, catalog=root/"rq/catalog.pt",
                             probe_manifest=probe_path, output=root/"train", resume=False)
            train_recommender(args, cfg)
            receipt = json.loads((root/"train/COMPLETE.json").read_text())
            self.assertEqual(receipt["updates"], 4)
            # Completed resume must perform no further model updates.
            before = sha256(root/"train/last.pt")
            args.resume = True
            train_recommender(args, cfg)
            self.assertEqual(sha256(root/"train/last.pt"), before)
            test_args = Namespace(device="cpu", dataset=dataset, catalog=args.catalog,
                                  run=root/"train", output=root/"test", resume=False)
            test_recommender(test_args, cfg)
            result = json.loads((root/"test/best/summary.json").read_text())
            self.assertEqual(result["count"], 6)
            self.assertEqual(result["metrics"]["completion_rate"], 1)
            self.assertEqual(result["candidate_lists"], 1)
            with self.assertRaises(FileExistsError):
                train_recommender(Namespace(**{**vars(args), "resume": False}), cfg)

    def test_frozen_asset_tampering_rejected(self):
        self.catalog.y[1, 0] += 1
        with self.assertRaises(RuntimeError):
            self.catalog.verify_frozen()

    def test_rq_interrupted_resume_matches_next_updates(self):
        from argparse import Namespace
        import scripts.rqcd.run as runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.cfg
            dataset = fixture(root/"data", cfg)
            content = root/"content.npy"
            np.save(content, F.normalize(torch.randn(24, 8), dim=-1).numpy())
            cfg["content_sha256"] = sha256(content)
            args = Namespace(device="cpu", dataset=dataset, content=content, output=root/"full", resume=False)
            fit_tokenizer(args, cfg)
            expected = torch.load(root/"full/last.pt", weights_only=False)
            # A process failure just after a complete saved U2 boundary.
            original_log = runner.append_log

            def fail_at_two(path, row):
                original_log(path, row)
                if row.get("stage") == "rq_validation" and row["update"] == 2:
                    raise RuntimeError("synthetic interruption")

            args.output = root/"resume"
            with patch.object(runner, "append_log", fail_at_two):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    fit_tokenizer(args, cfg)
            args.resume = True
            fit_tokenizer(args, cfg)
            actual = torch.load(root/"resume/last.pt", weights_only=False)
            for name in expected["model"]:
                torch.testing.assert_close(actual["model"][name], expected["model"][name], atol=0, rtol=0)
            self.assertEqual(actual["best_update"], expected["best_update"])
            self.assertEqual(actual["choices_rng"], expected["choices_rng"])

    def test_cluster_bootstrap_sample_weighting_and_pair_identity(self):
        values = [1.0, 1.0, 0.0]
        users = [1, 1, 2]
        a = clustered_interval(values, users)
        self.assertAlmostEqual(a["estimate"], 2/3)
        self.assertNotEqual(a["estimate"], .5)
        self.assertEqual(a, clustered_interval(values, users))
        rows = [{"sample_id": str(i), "user_id": users[i], "target": [1, 2, 3, 4, 5],
                 "prediction": [1, 6, 7, 8, 9],
                 "metrics": {"ndcg@5": values[i], "hr@5": values[i], "recall@5": values[i]}}
                for i in range(3)]
        result = analyze(rows, list(reversed(rows)))
        self.assertEqual(result["paired_reference"]["ndcg@5"]["ci95"], [0.0, 0.0])
        with self.assertRaises(ValueError):
            analyze(rows, rows[:2])

    def test_nonfinite_generation_fails_instead_of_becoming_item_zero(self):
        model = ContinuousListModel(self.cfg).eval()
        history, _ = examples(self.cfg)
        with patch.object(Diffusion, "sample", return_value=torch.full((3, 4), float("nan"))):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                generate(model, Diffusion(self.cfg), self.catalog, history, ["a", "b", "c"], "test", self.cfg)

    def test_packed_input_rejects_real_pad_and_future_history(self):
        with tempfile.TemporaryDirectory() as td:
            dataset = fixture(Path(td)/"data", self.cfg)
            file = dataset/"train.parquet"
            rows = pq.read_table(file).to_pylist()
            original = copy.deepcopy(rows)
            rows[0]["history_item_ids"][0] = 0
            pq.write_table(pa.Table.from_pylist(rows), file)
            with self.assertRaisesRegex(ValueError, "historical items"):
                PackedSplit(dataset, "train", self.cfg)
            rows = original
            rows[0]["history_timestamp_us"][-1] = 1000
            pq.write_table(pa.Table.from_pylist(rows), file)
            with self.assertRaisesRegex(ValueError, "chronology"):
                PackedSplit(dataset, "train", self.cfg)

    def test_cpu_bf16_attention_finite_backward_is_not_cuda_acceptance(self):
        # CPU autocast catches dtype promotion/SDPA integration mistakes only.
        model = ContinuousListModel(self.cfg).train()
        history, target = examples(self.cfg)
        y = self.catalog.y[target]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model.training_prediction(self.catalog.y[history], history.ne(0), y,
                                            torch.randn_like(y), torch.full(target.shape, 5))
        loss = (out.float()-y).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
