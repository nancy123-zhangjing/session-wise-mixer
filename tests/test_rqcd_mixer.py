"""CPU synthetic acceptance tests. No real dataset, checkpoint, network or GPU."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts.rqcd_mixer.config import load_config, load_runtime, RUNTIME_TEMPLATE, total_t_steps
from scripts.rqcd_mixer.common import (cpu_copy, seed_all, r_learning_rate, pre_learning_rate,
                                       gumbel_strength, atomic_save, read_state, write_json, sha256)
from scripts.rqcd_mixer.model import Recommender, Diffusion, relative_buckets
from scripts.rqcd_mixer.tokenizer import (RQTokenizer, CodebookMixer, hard_quantize, fixed_statistics,
                                         joint_quantize, distances, code_health)
from scripts.rqcd_mixer.catalog import Catalog, generate, prepare_noise, evaluation_noise
from scripts.rqcd_mixer.losses import recommendation_losses, alignment_loss
from scripts.rqcd_mixer.training import JointTrainer, Pretrainer
from scripts.rqcd_mixer.data import PermutationStream, PackedSplit
from scripts.rqcd_mixer.evaluation import evaluate, item_metrics


torch.set_num_threads(1)


def tiny_config():
    cfg = load_config()
    cfg.update(catalog_size=12, history_max=3, split_counts={"train": 7, "validation": 3, "test": 3})
    cfg["rq"].update(input_dim=8, hidden_dims=[16, 12, 8], latent_dim=8, codes=4,
                     epochs=2, batch_size=8, warmup_epochs=1, eval_interval=1, kmeans_iterations=3)
    cfg["model"].update(width=16, d_kv=4, ffn=32, time_hidden=16, dropout=0.0)
    cfg["diffusion"].update(steps=8, sample_steps=3)
    cfg["training"].update(epochs=3, preheat_epochs=1, effective_batch=4, warmup_updates=2, t_steps_per_cycle=2)
    cfg["evaluation"]["epochs"] = [1, 2, 3]
    return cfg


def cpu_runtime(micro=2):
    rt = json.loads(RUNTIME_TEMPLATE.read_text(encoding="utf-8"))
    rt.update(micro_r=micro, micro_t=micro, evaluation_batch=2, precision="fp32", data_device="cpu",
              fused_adamw=False, attention_backend="math", checkpoint_every_updates=2)
    return rt


class SyntheticData:
    def __init__(self, cfg, split="train"):
        self.split = split
        n, h, c = cfg["split_counts"][split], cfg["history_max"], cfg["catalog_size"]
        self.history = (torch.arange(n*h).reshape(n, h) % c)+1
        self.history[0, :h-1] = 0
        self.target = (torch.arange(n)[:, None]+torch.arange(5)[None]) % c+1
        self.sample_ids = [f"{split}:{i}" for i in range(n)]
        self.users = list(range(n))
        self.first_target_time = np.arange(n, dtype=np.int64)+100

    def __len__(self):
        return len(self.sample_ids)

    def batch(self, index, device):
        return self.history[index].to(device), self.target[index].to(device)


def fixture(mode="intra", cfg=None, micro=2):
    cfg = tiny_config() if cfg is None else cfg
    seed_all(17)
    content = F.normalize(torch.randn(cfg["catalog_size"], cfg["rq"]["input_dim"]), dim=-1)
    tokenizer = RQTokenizer(cfg, mode)
    tokenizer.initialize(content[:cfg["rq"]["batch_size"]], torch.Generator().manual_seed(112))
    with torch.no_grad():
        z, books = tokenizer.encoder(content), tokenizer.effective_books()
        asset = {"z": z.clone(), "books": books.clone(), **hard_quantize(z, books),
                 **fixed_statistics(z, books, torch.arange(1, len(z)+1)),
                 "original_ids": torch.arange(len(z)-1, -1, -1), "version": 0}
    model = Recommender(cfg)
    trainer = JointTrainer(cfg, cpu_runtime(micro), tokenizer, asset, cpu_copy(model.state_dict()), {"synthetic": True}, "cpu")
    return cfg, content, trainer


def assert_tree_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        case.assertTrue(torch.equal(a, b), f"Tensor state differs: shape {a.shape}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for k in a:
            assert_tree_equal(case, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class ProtocolTests(unittest.TestCase):
    def test_fixed_budgets_and_scheduler_endpoints(self):
        c = load_config()
        from scripts.rqcd_mixer.run import plan
        p = plan(c)
        self.assertEqual((p["pretraining_updates"], p["R_updates"], p["T_updates"]), (110000, 24255, 300))
        self.assertEqual(p["validation"],
                         [{"R": r, "T": (r - 5) * 10} for r in range(5, 36)])
        self.assertFalse(p["training_started"])
        self.assertEqual(pre_learning_rate(0, 11, c), 0)
        self.assertEqual(pre_learning_rate(550, 11, c), .001)
        self.assertAlmostEqual(pre_learning_rate(109999, 11, c), .001/109450)
        self.assertAlmostEqual(r_learning_rate(1, 24255, c), 6e-7)
        self.assertEqual(r_learning_rate(24255, 24255, c), 0)
        self.assertAlmostEqual(gumbel_strength(1, c), 1)
        self.assertAlmostEqual(gumbel_strength(300, c), .2)

    def test_config_mutation_and_unmeasured_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"bad.json"
            cfg = load_config()
            cfg["model"]["decoder_layers"] = 3
            write_json(path, cfg)
            with self.assertRaises(ValueError):
                load_config(path)
        with self.assertRaises(ValueError):
            load_runtime(RUNTIME_TEMPLATE, load_config())
        self.assertIsNone(load_runtime(RUNTIME_TEMPLATE, load_config(), require_batches=False)["micro_r"])

    def test_permutations_keep_tail_and_resume(self):
        stream = PermutationStream(7, 11)
        first, tail = stream.take(4), stream.take(4)
        self.assertEqual(len(tail), 3)
        self.assertEqual(len(torch.unique(torch.cat((first, tail)))), 7)
        saved = stream.state()
        nxt = stream.take(4)
        other = PermutationStream(7, 99)
        other.restore(saved)
        self.assertTrue(torch.equal(nxt, other.take(4)))

    def test_cosine_and_ddim_have_exact_update_budget(self):
        diffusion = Diffusion(load_config())
        self.assertEqual(len(diffusion.grid), 20)
        self.assertEqual((diffusion.grid[0], diffusion.grid[-1]), (1000, 1))
        self.assertTrue(all(x > y for x, y in zip(diffusion.grid, diffusion.grid[1:])))
        clean, calls = torch.randn(2, 32), []
        def predictor(x, times):
            calls.append(int(times[0]))
            return clean
        out = diffusion.sample(torch.randn_like(clean), predictor)
        self.assertTrue(torch.equal(out, clean))
        self.assertEqual(calls, diffusion.grid)
        self.assertEqual(float(diffusion.alpha[0]), 1)

    def test_noise_sample_identity_not_batch_or_model(self):
        cfg = tiny_config()
        a = evaluation_noise(cfg, ["a", "b", "c"], "test")
        b = evaluation_noise(cfg, ["c", "a"], "test")
        np.testing.assert_array_equal(a[[2, 0]], b)
        self.assertFalse(np.array_equal(a, evaluation_noise(cfg, ["a", "b", "c"], "validation")))


class MixerTests(unittest.TestCase):
    def test_intra_batched_matches_dense_block_reference_and_gradient(self):
        cfg = tiny_config()
        torch.manual_seed(12)
        mix = CodebookMixer(cfg, "intra")
        torch.nn.init.normal_(mix.out.weight, std=.1)
        books = torch.randn(4, 4, 8, requires_grad=True)
        actual = mix(books)
        x = (mix.norm(books)+mix.layer_embedding[:, None]).flatten(0, 1)[None]
        q, k, v = (mix._heads(m(x)) for m in (mix.q, mix.k, mix.v))
        layer = torch.arange(4).repeat_interleave(4)
        score = (q@k.transpose(-1, -2))/math.sqrt(2)
        score = score.masked_fill((layer[:, None] != layer[None])[None, None], -torch.inf)
        raw = mix.out((score.softmax(-1)@v).transpose(1, 2).reshape_as(x)).view_as(books)
        rms = books.square().mean((-1, -2)).sqrt()
        expected = raw*(rms/rms[0]).detach()[:, None, None]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
        ga = torch.autograd.grad(actual.square().sum(), books, retain_graph=True)[0]
        gb = torch.autograd.grad(expected.square().sum(), books)[0]
        torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-7)

    def test_causal_is_layer_blocks_not_center_order(self):
        mix = CodebookMixer(tiny_config(), "causal")
        torch.nn.init.normal_(mix.out.weight, std=.2)
        books = torch.randn(4, 4, 8)
        altered = books.clone()
        altered[2:] += torch.randn_like(altered[2:])*4
        torch.testing.assert_close(mix(books)[:2], mix(altered)[:2], rtol=0, atol=0)
        perm = torch.tensor([2, 0, 3, 1])
        permuted = books[:, perm]
        torch.testing.assert_close(mix(permuted), mix(books)[:, perm], rtol=3e-5, atol=2e-6)
        self.assertTrue(mix.causal_mask[0, 0, :4, :4].all())
        self.assertFalse(mix.causal_mask[0, 0, :4, 4:].any())

    def test_zero_output_shared_init_and_scale_reference(self):
        cfg, _, trainer = fixture()
        books = trainer.tokenizer.books
        self.assertTrue(torch.equal(books, trainer.tokenizer.effective_books()))
        mix = trainer.tokenizer.mixer
        with torch.no_grad():
            mix.out.bias.fill_(1)
        correction = mix(books)
        ratio = (books.square().mean((-1, -2)).sqrt()/books[0].square().mean().sqrt()).detach()
        torch.testing.assert_close(correction, ratio[:, None, None].expand_as(correction))

    def test_pretraining_reconstruction_and_vq_use_distinct_gradient_paths(self):
        _, content, trainer = fixture()
        token = trainer.tokenizer
        token.requires_grad_(True)
        losses = token.pretrain_forward(content)
        losses["reconstruction"].backward(retain_graph=True)
        self.assertIsNone(token.books.grad)
        self.assertGreater(sum(p.grad.abs().sum() for p in token.encoder.parameters()), 0)
        self.assertGreater(sum(p.grad.abs().sum() for p in token.decoder.parameters()), 0)
        token.zero_grad(set_to_none=True)
        losses["vq"].backward()
        self.assertGreater(token.books.grad.abs().sum(), 0)
        self.assertGreater(token.mixer.out.weight.grad.abs().sum(), 0)

    def test_weighted_statistics_match_explicit_all_center_distances(self):
        cfg, _, trainer = fixture()
        cat = trainer.catalog
        weight = torch.arange(1, len(cat.z)+1)
        stats = fixed_statistics(cat.z, cat.books, weight)
        residuals = hard_quantize(cat.z, cat.books)["residuals"].double()
        expected = (((residuals[:, :, None]-cat.books.double()[None]).square().sum(-1).mean(-1)) *
                    weight[:, None]).sum(0)/weight.sum()
        torch.testing.assert_close(stats["distance_scale"].double(), expected, rtol=1e-6, atol=1e-8)
        mean = (cat.z.double()*weight[:, None]).sum(0)/weight.sum()
        torch.testing.assert_close(stats["mean"].double(), mean, rtol=1e-6, atol=1e-8)


class ModelTests(unittest.TestCase):
    def test_five_parallel_views_match_sequential_and_cannot_read_future(self):
        cfg, _, trainer = fixture()
        model, cat = trainer.model.eval(), trainer.catalog
        data = SyntheticData(cfg)
        hist, target = data.history[:2], data.target[:2]
        h_emb, t_emb = model.lookup(cat.sid_pad[hist]), model.lookup(cat.sid_pad[target])
        noisy, times = torch.randn(2, 5, 8), torch.randint(1, 9, (2, 5))
        actual = model.training_prediction(h_emb, hist != 0, t_emb, noisy, times)
        mem, valid = model.encode_history(h_emb, hist != 0)
        expected = torch.stack([model.predict_uncached(noisy[:, j], times[:, j], t_emb[:, :j], mem, valid)
                                for j in range(5)], 1)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        changed = t_emb.clone()
        changed[:, 2:] += torch.randn_like(changed[:, 2:])*5
        modified = model.training_prediction(h_emb, hist != 0, changed, noisy, times)
        torch.testing.assert_close(actual[:, :3], modified[:, :3], rtol=0, atol=0)
        ga = torch.autograd.grad(actual.square().mean(), model.embedding, retain_graph=True)[0]
        gb = torch.autograd.grad(expected.square().mean(), model.embedding)[0]
        torch.testing.assert_close(ga, gb, rtol=1e-4, atol=2e-5)

    def test_cache_matches_uncached_for_all_five_prefix_lengths(self):
        cfg, _, trainer = fixture()
        model, cat = trainer.model.eval(), trainer.catalog
        data = SyntheticData(cfg)
        hist, target = data.history[:2], data.target[:2]
        mem, valid = model.encode_history(model.lookup(cat.sid_pad[hist]), hist != 0)
        cache = model.start_cache(mem, valid)
        for j in range(5):
            saved = cpu_copy(cache.prefix)
            for time in (8, 4, 1):
                noisy, times = torch.randn(2, 8), torch.full((2,), time)
                a = model.predict_cached(noisy, times, cache)
                b = model.predict_uncached(noisy, times, model.lookup(cat.sid_pad[target[:, :j]]), mem, valid)
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
            assert_tree_equal(self, cache.prefix, saved)
            self.assertEqual(cache.length, 4*j)
            if j < 4:
                model.append_prefix(model.lookup(cat.sid_pad[target[:, j]]), cache)

    def test_real_50_item_geometry_and_independent_sid_tables(self):
        cfg = load_config()
        model = Recommender(cfg).eval()
        sid = torch.randint(0, 256, (1, 50, 4))
        history = model.lookup(sid)
        with torch.no_grad():
            memory, valid = model.encode_history(history, torch.ones(1, 50, dtype=torch.bool))
            output = model.training_prediction(history, torch.ones(1, 50, dtype=torch.bool),
                                               model.lookup(sid[:, :5]), torch.randn(1, 5, 32),
                                               torch.ones(1, 5, dtype=torch.long))
        self.assertEqual(tuple(memory.shape), (1, 200, 256))
        self.assertEqual(tuple(output.shape), (1, 5, 32))
        self.assertFalse(torch.equal(model.embedding[0, 1], model.embedding[1, 1]))
        self.assertEqual(len(model.decoder), 4)

    def test_t5_stacks_match_official_transformers_reference(self):
        # Installed reference, no model download and no pretrained weights.
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Stack
        cfg = tiny_config()
        c = cfg["model"]
        model = Recommender(cfg).eval()
        def make_stack(decoder):
            cc = T5Config(d_model=c["width"], d_kv=c["d_kv"], d_ff=c["ffn"], num_heads=c["heads"],
                          num_layers=c["decoder_layers"] if decoder else c["encoder_layers"],
                          dropout_rate=0, layer_norm_epsilon=c["norm_eps"], is_decoder=decoder,
                          is_encoder_decoder=False, feed_forward_proj="relu",
                          relative_attention_num_buckets=32, relative_attention_max_distance=128,
                          use_cache=False)
            cc._attn_implementation = "eager"
            reference = T5Stack(cc).eval()
            blocks = model.decoder if decoder else model.encoder
            state = reference.state_dict()
            for i, b in enumerate(blocks):
                pre = f"block.{i}.layer."
                state[pre+"0.layer_norm.weight"] = b.self_norm.weight
                for name in ("q", "k", "v", "o"):
                    state[pre+"0.SelfAttention."+name+".weight"] = getattr(b.self_attn, name).weight
                if i == 0:
                    state[pre+"0.SelfAttention.relative_attention_bias.weight"] = (model.dec_bias if decoder else model.enc_bias).table.weight
                if decoder:
                    state[pre+"1.layer_norm.weight"] = b.cross_norm.weight
                    for name in ("q", "k", "v", "o"):
                        state[pre+"1.EncDecAttention."+name+".weight"] = getattr(b.cross, name).weight
                f = "2" if decoder else "1"
                state[pre+f+".layer_norm.weight"] = b.ff_norm.weight
                state[pre+f+".DenseReluDense.wi.weight"] = b.ff.wi.weight
                state[pre+f+".DenseReluDense.wo.weight"] = b.ff.wo.weight
            state["final_layer_norm.weight"] = (model.dec_norm if decoder else model.enc_norm).weight
            reference.load_state_dict(state)
            return reference
        enc, dec = make_stack(False), make_stack(True)
        embeddings = torch.randn(2, 3, 4, 16, requires_grad=True)
        valid_items = torch.tensor([[False, False, True], [True, True, True]])
        ours, valid = model.encode_history(embeddings, valid_items)
        reference = enc(inputs_embeds=embeddings.flatten(1, 2).masked_fill(~valid[..., None], 0),
                        attention_mask=valid).last_hidden_state.masked_fill(~valid[..., None], 0)
        torch.testing.assert_close(ours, reference, rtol=2e-5, atol=2e-6)
        ga = torch.autograd.grad(ours.square().sum(), embeddings, retain_graph=True)[0]
        gb = torch.autograd.grad(reference.square().sum(), embeddings)[0]
        torch.testing.assert_close(ga, gb, rtol=1e-3, atol=4e-6)
        prefix, noisy, times = torch.randn(2, 2, 4, 16), torch.randn(2, 8), torch.tensor([2, 7])
        output = model.predict_uncached(noisy, times, prefix, ours, valid)
        x = torch.cat((prefix.flatten(1, 2), model.noisy_query(noisy, times)[:, None]), 1)
        expected = model.output(dec(inputs_embeds=x, encoder_hidden_states=ours,
                                    encoder_attention_mask=valid, use_cache=False).last_hidden_state[:, -1])
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        from transformers.models.t5.modeling_t5 import T5Attention
        offsets = torch.arange(-220, 221)
        for bidirectional in (False, True):
            self.assertTrue(torch.equal(relative_buckets(offsets, bidirectional),
                                        T5Attention._relative_position_bucket(offsets, bidirectional)))


class GradientAndTrainingTests(unittest.TestCase):
    def test_denoise_only_crosses_sid_bridge_to_tokenizer(self):
        cfg, _, trainer = fixture()
        trainer._set_mode("T")
        data = SyntheticData(cfg)
        hist, tgt = data.history[:2], data.target[:2]
        ids = torch.cat((hist, tgt), 1)
        draw = trainer.draws(2, "T")
        books = trainer.tokenizer.effective_books()
        result = joint_quantize(trainer.catalog.z_pad[ids].reshape(-1, 8), books,
                                trainer.catalog.stats["distance_scale"], trainer.catalog.sid_pad[ids].reshape(-1, 4),
                                trainer.model.embedding, draw["gumbel"].reshape(-1, 4, 4), 1, cfg)
        emb = result["embedding"].view(2, 8, 4, 16)
        self.assertTrue(torch.equal(emb.detach(), trainer.model.lookup(trainer.catalog.sid_pad[ids])))
        clean = trainer.catalog.x_pad[tgt]
        noisy = trainer.diffusion.noise(clean, draw["times"], draw["epsilon"])
        pred = trainer.model.training_prediction(emb[:, :3], hist != 0, emb[:, 3:], noisy, draw["times"])
        F.mse_loss(pred, clean).backward()
        self.assertGreater(trainer.tokenizer.books.grad.abs().sum(), 0)
        self.assertGreater(trainer.tokenizer.mixer.out.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
        self.assertTrue(all(p.grad is None for p in trainer.tokenizer.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in trainer.tokenizer.decoder.parameters()))

    def test_kl_target_stopped_prediction_and_live_prefix_have_gradients(self):
        cfg, _, trainer = fixture()
        books = trainer.catalog.books.clone().requires_grad_()
        target = trainer.catalog.z[:5].clone().requires_grad_()
        predicted = (target.detach()+.03).requires_grad_()
        sid, scale = trainer.catalog.sid[:5], trainer.catalog.stats["distance_scale"]
        actual = alignment_loss(predicted, target, sid, books, scale, .1)
        prefix = torch.zeros_like(target)
        manual = []
        for level in range(4):
            p = (-(target.detach()-prefix.detach())[:, None].sub(books[level].detach()[None]).square().sum(-1)/(scale[level]*.1)).softmax(-1)
            logq = (-(predicted-prefix)[:, None].sub(books[level][None]).square().sum(-1)/(scale[level]*.1)).log_softmax(-1)
            manual.append((p*(p.clamp_min(1e-38).log()-logq)).sum(-1))
            prefix = prefix+books[level, sid[:, level]]
        expected = torch.stack(manual).mean()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
        actual.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(predicted.grad.abs().sum(), 0)
        self.assertGreater(books.grad.abs().sum(), 0)
        # The R phase also retains the generated-vector gradient.
        p2 = predicted.detach().requires_grad_()
        alignment_loss(p2, target.detach(), sid, books.detach(), scale, .1).backward()
        self.assertGreater(p2.grad.abs().sum(), 0)

    def test_r_and_t_only_change_their_authorized_parameters(self):
        cfg, _, trainer = fixture()
        d = SyntheticData(cfg)
        before = cpu_copy(trainer.tokenizer.state_dict())
        old_model = cpu_copy(trainer.model.state_dict())
        trainer.update(d.history[:4], d.target[:4], "R")
        assert_tree_equal(self, before, trainer.tokenizer.state_dict())
        self.assertTrue(any(not torch.equal(v, trainer.model.state_dict()[k]) for k, v in old_model.items()))
        after_r = cpu_copy(trainer.model.state_dict())
        cached = cpu_copy(trainer.catalog.state())
        trainer.update(d.history[:4], d.target[:4], "T")
        assert_tree_equal(self, after_r, trainer.model.state_dict())
        assert_tree_equal(self, cached, trainer.catalog.state())
        for k, v in trainer.tokenizer.state_dict().items():
            if k.startswith(("encoder.", "decoder.")):
                self.assertTrue(torch.equal(v, before[k]))
        self.assertFalse(torch.equal(before["books"], trainer.tokenizer.books))

    def test_microbatch_tail_accumulation_matches_full_gradient(self):
        for phase in ("R", "T"):
            cfg, _, a = fixture(micro=3)
            _, _, b = fixture(micro=1)
            d = SyntheticData(cfg)
            draw = a.draws(3, phase)
            first = a.update(d.history[-3:], d.target[-3:], phase, draws=draw)
            second = b.update(d.history[-3:], d.target[-3:], phase, draws=draw)
            torch.testing.assert_close(first["losses"], second["losses"], rtol=2e-5, atol=3e-6)
            params_a = a.model.parameters() if phase == "R" else a.tokenizer.joint_parameters()
            params_b = b.model.parameters() if phase == "R" else b.tokenizer.joint_parameters()
            for pa, pb in zip(params_a, params_b):
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=3e-4, atol=4e-6)
            self.assertEqual(a.r_updates if phase == "R" else a.t_updates, 1)

    def test_partial_t_resume_exact_including_dropout_and_cached_catalog(self):
        cfg = tiny_config()
        cfg["model"]["dropout"] = .1
        cfg, _, a = fixture(cfg=cfg)
        data = SyntheticData(cfg)
        while not (a.phase == "T" and a.t_in_block == 1):
            a.advance(data)
        saved = a.state()
        old_sid = saved["catalog"]["sid"].clone()
        events_a = [a.advance(data) for _ in range(4)]
        final_a = a.state()
        _, _, b = fixture(cfg=cfg)
        b.restore(saved)
        self.assertTrue(torch.equal(b.catalog.sid, old_sid))
        events_b = [b.advance(data) for _ in range(4)]
        assert_tree_equal(self, events_a, events_b)
        assert_tree_equal(self, final_a, b.state())
        bad = cpu_copy(saved)
        bad["r_updates"] += 1
        with self.assertRaises(ValueError):
            b.restore(bad)

    def test_full_tiny_schedule_and_pretraining_resume(self):
        cfg, content, trainer = fixture()
        data = SyntheticData(cfg)
        t_counts = []
        while trainer.phase != "done":
            event = trainer.advance(data)
            if event["kind"] == "R_epoch":
                t_counts.append((event["R"], event["T"]))
        self.assertEqual(t_counts, [(1, 0), (2, 2), (3, 4)])
        self.assertEqual(trainer.r_updates, 6)
        _, _, other = fixture()
        other.restore(trainer.state())
        self.assertEqual(other.catalog.version, 2)
        token = RQTokenizer(cfg, "intra")
        token.load_state_dict(trainer.tokenizer.state_dict())
        pre = Pretrainer(cfg, cpu_runtime(), token, content, {"test": True})
        pre.advance()
        saved = pre.state()
        a = pre.advance()
        state = pre.state()
        token2 = RQTokenizer(cfg, "intra")
        pre2 = Pretrainer(cfg, cpu_runtime(), token2, content, {"test": True})
        pre2.restore(saved)
        b = pre2.advance()
        assert_tree_equal(self, a, b)
        assert_tree_equal(self, state, pre2.state())


class LandingAndEvaluationTests(unittest.TestCase):
    def test_legal_rq_landing_matches_scalar_reference(self):
        _, _, trainer = fixture()
        cat = trainer.catalog
        z = torch.randn(6, 8)*.3
        selected = torch.tensor([[1, 2], [3, 4], [2, 9], [4, 7], [8, 3], [11, 2]])
        actual, diag = cat.land(z, selected)
        expected = []
        for row, used in zip(z, selected):
            remaining = [i for i in range(1, cat.count+1) if i not in used.tolist()]
            residual = row.clone()
            for level in range(4):
                legal = sorted({int(cat.sid[i-1, level]) for i in remaining})
                distances_ref = [(float((residual-cat.books[level, k]).square().sum()), k) for k in legal]
                _, code = min(distances_ref)
                remaining = [i for i in remaining if int(cat.sid[i-1, level]) == code]
                residual -= cat.books[level, code]
            _, _, item = min((float((row-cat.z[i-1]).square().sum()), int(cat.original_ids[i-1]), i) for i in remaining)
            expected.append(item)
        self.assertEqual(actual.tolist(), expected)
        self.assertTrue(diag["available_bucket_size"].min() > 0)

    def test_collisions_use_original_id_tie_and_exclude_items_not_sid(self):
        z = torch.tensor([[1., 0.]]*6)
        books = torch.zeros(4, 2, 2)
        books[0, 0] = torch.tensor([1., 0.])
        asset = {"z": z, "books": books, **hard_quantize(z, books),
                 "original_ids": torch.tensor([9, 3, 8, 1, 7, 2]),
                 "mean": torch.zeros(2), "scale": torch.tensor(1.), "distance_scale": torch.ones(4)}
        cat = Catalog(asset, "cpu")
        selected = torch.empty(1, 0, dtype=torch.long)
        result = []
        for _ in range(5):
            item, diag = cat.land(z[:1], selected)
            result.append(int(item[0]))
            selected = torch.cat((selected, item[:, None]), 1)
        self.assertEqual(result, [4, 6, 2, 5, 3])
        self.assertEqual(code_health(cat.sid, 2)["collision_redundancy"], 5/6)

    def test_metrics_match_existing_item_protocol_including_invalid_duplicates(self):
        from scripts.listwise_diffgrm.metrics import ranking_metrics
        pred = torch.tensor([[1, 1, 0, 99, 3], [5, 4, 3, 2, 1], [6, 7, 8, 9, 10]])
        truth = torch.tensor([[1, 2, 3, 4, 5]]*3)
        actual = item_metrics(pred, truth, 12)
        reference = ranking_metrics(pred, truth, catalog_item_ids=torch.arange(1, 13))
        for k, value in actual.items():
            torch.testing.assert_close(value, reference[k], rtol=0, atol=0)

    def test_generation_uses_landed_prefix_and_batch_invariance(self):
        cfg, _, trainer = fixture()
        data = SyntheticData(cfg, "validation")
        noise = torch.from_numpy(evaluation_noise(cfg, data.sample_ids, "validation"))
        model = trainer.model.eval()
        original = model.append_prefix
        appended = []
        def record(tokens, cache):
            appended.append(tokens.clone())
            return original(tokens, cache)
        with patch.object(model, "append_prefix", side_effect=record), \
                patch.object(model, "encode_history", wraps=model.encode_history) as history_call:
            predicted, _ = generate(model, trainer.diffusion, trainer.catalog, data.history, noise, cpu_runtime())
        self.assertEqual(history_call.call_count, 1)
        self.assertEqual(len(appended), 4)
        for j in range(4):
            self.assertTrue(torch.equal(appended[j], model.lookup(trainer.catalog.sid_pad[predicted[:, j]])))
        singles = torch.cat([generate(model, trainer.diffusion, trainer.catalog, data.history[i:i+1], noise[i:i+1],
                                      cpu_runtime())[0] for i in range(len(data))])
        self.assertTrue(torch.equal(predicted, singles))
        self.assertTrue(all(len(row.unique()) == 5 for row in predicted))

    def test_evaluation_receipt_hash_rng_and_reuse(self):
        cfg, _, trainer = fixture()
        data = SyntheticData(cfg, "validation")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare_noise(root/"noise.npy", cfg, data.sample_ids, "validation")
            state = torch.get_rng_state().clone()
            result = evaluate(trainer.model, trainer.diffusion, trainer.catalog, data, cfg, cpu_runtime(),
                              root/"noise.npy", root/"eval", {"test_identity": 1})
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertTrue(trainer.model.training)
            self.assertEqual(result["count"], 3)
            with gzip.open(root/"eval/predictions.jsonl.gz", "rt", encoding="utf-8") as f:
                predictions = [json.loads(line) for line in f]
            self.assertEqual([x["sample_id"] for x in predictions], data.sample_ids)
            with patch("scripts.rqcd_mixer.evaluation.generate", side_effect=AssertionError("must reuse")):
                again = evaluate(trainer.model, trainer.diffusion, trainer.catalog, data, cfg, cpu_runtime(),
                                 root/"noise.npy", root/"eval", {"test_identity": 1})
            self.assertEqual(result, again)
            with self.assertRaises(ValueError):
                evaluate(trainer.model, trainer.diffusion, trainer.catalog, data, cfg, cpu_runtime(),
                         root/"noise.npy", root/"eval", {"test_identity": 2})

    def test_catalog_rejects_mixed_sid_or_quantized_vector_versions(self):
        _, _, trainer = fixture()
        asset = cpu_copy(trainer.catalog.state())
        asset["sid"][0, 0] = (asset["sid"][0, 0]+1) % 4
        with self.assertRaises(ValueError):
            Catalog(asset, "cpu")
        asset = cpu_copy(trainer.catalog.state())
        asset["q"][0, 0] += 1
        with self.assertRaises(ValueError):
            Catalog(asset, "cpu")


class RunnerIntegrationTests(unittest.TestCase):
    def test_combined_benchmark_numerical_and_resume_helpers_on_cpu(self):
        from scripts.rqcd_mixer.benchmark import numerical_check, resume_check
        cfg, _, trainer = fixture("causal")
        initial = {"recommender": cpu_copy(trainer.model.state_dict()),
                   "tokenizers": {"B2": cpu_copy(trainer.tokenizer.state_dict())},
                   "benchmark_asset": cpu_copy(trainer.catalog.state())}
        data = SyntheticData(cfg)
        result = numerical_check(cfg, cpu_runtime(), initial, data.history[:4], data.target[:4], "cpu")
        self.assertTrue(result["passed"])
        with tempfile.TemporaryDirectory() as temp:
            resumed = resume_check(cfg, cpu_runtime(), initial, data, "cpu", Path(temp))
            self.assertTrue(resumed["passed"])

    def test_three_arm_synthetic_pipeline_interrupt_resume_test_and_report(self):
        # Tiny synthetic fixture is injected directly into functions. The
        # public CLI still rejects changed scientific settings.
        import pyarrow as pa
        import pyarrow.parquet as pq
        from scripts.listwise_diffgrm.assets import BOUND_FILES
        from scripts.rqcd_mixer import run
        from scripts.rqcd_mixer.report import report
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset = root/"dataset"
            dataset.mkdir()
            content = F.normalize(torch.randn(12, 8), dim=-1).numpy()
            np.save(root/"content.npy", content)
            cfg["content_sha256"] = sha256(root/"content.npy")
            write_json(dataset/"config.json", {"history_max": 3, "list_length": 5})
            write_json(dataset/"item_map.json", {"pad_id": 0, "entries": [
                {"id": i+1, "original_id": 12-i} for i in range(12)]})
            write_json(dataset/"train_seen.json", {})
            (dataset/"item_text.jsonl").write_text("{}\n", encoding="utf-8")
            pq.write_table(pa.table({"item_id": list(range(1, 13))}), dataset/"item_catalog.parquet")
            for split in cfg["split_counts"]:
                data = SyntheticData(cfg, split)
                histories = [[int(x) for x in row if x] for row in data.history]
                table = pa.table({"sample_id": data.sample_ids, "user_id": data.users,
                                  "history_item_ids": histories,
                                  "history_timestamp_us": [list(range(1, len(h)+1)) for h in histories],
                                  "target_item_ids": data.target.tolist(),
                                  "target_timestamp_us": [list(range(10, 15)) for _ in histories],
                                  "target_mask": [[True]*5 for _ in histories]})
                pq.write_table(table, dataset/f"{split}.parquet")
            write_json(dataset/"manifest.json", {"artifacts": [
                {"path": n, "bytes": (dataset/n).stat().st_size, "sha256": sha256(dataset/n)} for n in BOUND_FILES]})
            args = SimpleNamespace(dataset=dataset, content=root/"content.npy", output=root/"prepared",
                                   device="cpu", prepared=root/"prepared", resume=False)
            runtime = cpu_runtime()
            with patch.object(run, "_runtime", return_value=(runtime, {"synthetic_cpu": True})):
                run.prepare(args, cfg)
                shared = read_state(root/"prepared/initial.pt")
                for name, parameter in shared["tokenizers"]["B0"].items():
                    self.assertTrue(torch.equal(parameter, shared["tokenizers"]["B1"][name]))
                    self.assertTrue(torch.equal(parameter, shared["tokenizers"]["B2"][name]))
                for name, parameter in shared["tokenizers"]["B1"].items():
                    if name.startswith("mixer."):
                        self.assertTrue(torch.equal(parameter, shared["tokenizers"]["B2"][name]))
                paths = []
                for arm in ("B0", "B1", "B2"):
                    args.arm, args.pretrained = arm, root/(arm+"_rq")
                    args.output, args.resume = args.pretrained, False
                    pre = run.pretrain(args, cfg)
                    self.assertEqual((pre["epochs"], pre["updates"]), (2, 4))
                    args.output, args.run = root/arm, root/arm
                    if arm == "B1":
                        with patch("scripts.rqcd_mixer.evaluation.evaluate", side_effect=RuntimeError("synthetic interruption")):
                            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                                run.train(args, cfg)
                        node = root/arm/"checkpoints/R01_T000.pt"
                        original_sha = sha256(node)
                        args.resume = True
                    result = run.train(args, cfg)
                    if arm == "B1":
                        self.assertEqual(original_sha, sha256(node))
                    self.assertEqual((result["R"], result["T"], result["R_updates"]), (3, 4, 6))
                    self.assertEqual(len(list((root/arm/"checkpoints").glob("*.pt"))), 3)
                    run.test(args, cfg)
                    tests = json.loads((root/arm/"TEST_COMPLETE.json").read_text())
                    self.assertEqual(tests["best"]["count"], 3)
                    self.assertEqual(tests["final"]["count"], 3)
                    paths.append(root/arm)
                report_result = report(paths, root/"report")
                self.assertFalse(report_result["model_executed"])
                self.assertEqual(report_result["rows"], 6)
                self.assertTrue((root/"report/T_align.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
