"""Resumable effective-batch R/T updates. No launches or platform side effects."""
from __future__ import annotations

import math
import torch

from .catalog import Catalog
from .common import (adamw, autocast, attention_backend, cpu_copy, r_learning_rate,
                     gumbel_strength, seed_for, rng_state, restore_rng, pre_learning_rate)
from .config import total_t_steps
from .data import PermutationStream
from .losses import recommendation_losses
from .model import Recommender, Diffusion
from .tokenizer import joint_quantize, hard_quantize, code_health
from .screen import switches, validate as validate_screen

LOSS_KEYS = ("total", "denoise", "align", "latent", "vq")


class JointTrainer:
    def __init__(self, cfg, runtime, tokenizer, asset, initial_recommender, binding, device):
        self.cfg, self.runtime, self.binding = cfg, runtime, binding
        validate_screen(cfg)
        self.screen = switches(cfg)
        self.loss_keys = (("total", "denoise", "landing", "path", "bucket", "latent", "vq")
                          if cfg['loss'].get('mode') == 'rq_landing' else LOSS_KEYS)
        if self.screen.get("history_head"):
            self.loss_keys = (*self.loss_keys, "history")
        self.item_vectors = self.gen_x_pad = None
        self.device = torch.device(device)
        self.tokenizer = tokenizer.to(device)
        self.catalog = Catalog(asset, device)
        if not torch.allclose(self.tokenizer.effective_books().detach(), self.catalog.books, rtol=0, atol=0):
            raise ValueError("Tokenizer checkpoint and catalog effective books mismatch")
        if not self.screen.keys() - {"name"}:
            self.model = Recommender(cfg).to(device)
            self.model.load_state_dict(initial_recommender)
        else:
            self.model = self._screen_model(cfg, initial_recommender, device)
        self.diffusion = Diffusion(cfg).to(device)
        t = cfg["training"]
        self.opt_r = adamw(self.model.trainable_parameters(), cfg, runtime, device,
                          lr=t["r_lr"], weight_decay=t["r_weight_decay"])
        groups = [{"params": [self.tokenizer.books], "lr": t["base_lr"], "name": "books"}]
        if self.tokenizer.mixer is not None:
            groups.append({"params": list(self.tokenizer.mixer.parameters()), "lr": t["mixer_lr"], "name": "mixer"})
        self.opt_t = adamw(groups, cfg, runtime, device, lr=t["base_lr"], weight_decay=t["t_weight_decay"])
        self.train_count = cfg["split_counts"]["train"]
        self.steps_per_epoch = t.get('r_updates_per_unit', math.ceil(self.train_count/t["effective_batch"]))
        if type(self.steps_per_epoch) is not int or not 1 <= self.steps_per_epoch <= math.ceil(
                self.train_count/t['effective_batch']):
            raise ValueError('Invalid CF R updates per evaluation unit')
        self.total_r = self.steps_per_epoch*t["epochs"]
        epoch_size = min(self.train_count, self.steps_per_epoch * t['effective_batch'])
        self.r_stream = PermutationStream(self.train_count, seed_for(cfg["seed"], "R_order"), epoch_size=epoch_size)
        self.t_stream = PermutationStream(self.train_count, seed_for(cfg["seed"], "T_order"))
        names = ("R_times", "R_epsilon", "T_times", "T_epsilon", "T_gumbel")
        if self.screen.get("guidance"):
            names = (*names, "R_uncond")
        self.generators = {
            name: torch.Generator(device=device).manual_seed(seed_for(cfg["seed"], name))
            for name in names}
        self.r_epoch = self.r_updates = self.t_updates = self.t_in_block = 0
        self.phase, self.active_mode = "R", None
        self.validated_epochs = []
        self.best_score, self.best_epoch = -1.0, None
        self.sums = {phase: torch.zeros(len(self.loss_keys)+1, dtype=torch.float64, device=device) for phase in ("R", "T")}
        self.counts = {"R": 0, "T": 0}
        self._set_mode("R")

    def _screen_model(self, cfg, initial_recommender, device):
        """Screening model: every original parameter from the common initial state;
        parameters that are new (or change shape, teacher-space generation) keep
        their constructor initialization under a fixed seed."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(cfg["seed"], "screen_initial_recommender"))
            model = Recommender(cfg)
        own = model.state_dict()
        allowed_new = {k for k in own if k.split(".")[0] in ("history_query", "history_head", "null_memory",
                                                             "to_latent", "gen_mean", "gen_scale")}
        # Generation width changes these (teacher space): the time-frequency buffer is
        # recomputed by the constructor, the layers start from its seeded initialization.
        reshaped = {"input.weight", "output.weight", "output.bias", "time.0.weight", "time.2.weight", "time.2.bias",
                    "time_frequency"}
        loaded = {}
        for key, value in initial_recommender.items():
            if key not in own:
                raise ValueError(f"Initial recommender key {key} is missing from the screening model")
            if own[key].shape == value.shape:
                loaded[key] = value
            elif not (model.generation_space == "teacher" and key in reshaped):
                raise ValueError(f"Initial recommender key {key} changed shape unexpectedly")
        missing = set(own) - set(loaded) - allowed_new - (reshaped if model.generation_space == "teacher" else set())
        if missing:
            raise ValueError(f"Screening model keys without an initial value: {sorted(missing)}")
        model.load_state_dict({**own, **loaded})
        return model.to(device)

    def attach_item_vectors(self, features):
        """Fixed 256-d CF target vectors (teacher items / global RMS), 1-based item IDs -> row id-1."""
        if features.shape != (self.cfg["catalog_size"], self.cfg["rq"]["input_dim"]):
            raise ValueError("Item vectors do not match the catalog")
        self.item_vectors = features.detach().float().to(self.device)

    def attach_latent_encoder(self, encoder):
        """Teacher-space generation lands through a frozen copy of the RQ Encoder."""
        if self.model.to_latent is None:
            raise ValueError("Only teacher-space generation uses the latent encoder")
        self.model.to_latent.load_state_dict(encoder.state_dict())
        self.model.to_latent.requires_grad_(False)

    @torch.no_grad()
    def configure_generation(self, target_counts):
        """Teacher-space targets: standardize the item vectors with the training-target
        weighted mean and one global scale (the same rule as fixed_statistics)."""
        if self.model.to_latent is None:
            return
        weight, value = target_counts.double().to(self.device), self.item_vectors.double()
        mean = (weight[:, None]*value).sum(0)/weight.sum()
        scale = ((weight[:, None]*(value-mean).square()).sum()/(weight.sum()*value.shape[-1])).sqrt()
        if not torch.isfinite(scale) or scale <= 1e-8:
            raise ValueError("Degenerate teacher-space statistics")
        self.model.gen_mean.copy_(mean.float())
        self.model.gen_scale.copy_(scale.float())
        self._build_generation_targets()

    def _build_generation_targets(self):
        x = (self.item_vectors-self.model.gen_mean)/self.model.gen_scale
        self.gen_x_pad = torch.cat((x.new_zeros(1, x.shape[1]), x))

    def _set_mode(self, phase):
        if self.active_mode == phase:
            return
        self.model.zero_grad(set_to_none=True)
        self.tokenizer.zero_grad(set_to_none=True)
        self.model.requires_grad_(phase == "R")
        for name, parameter in self.model.named_parameters():
            if name in self.model.fixed_parameter_names():
                parameter.requires_grad_(False)
        self.model.train(phase == "R")
        self.tokenizer.requires_grad_(False)
        for p in self.tokenizer.joint_parameters():
            p.requires_grad_(phase == "T")
        self.tokenizer.eval()  # Mixer has dropout=0 in both phases.
        self.active_mode = phase

    def draws(self, count, phase):
        shape = (count, self.cfg["list_length"], self.model.dim)
        result = {
            "times": torch.randint(1, self.diffusion.steps+1, shape[:2], device=self.device, generator=self.generators[phase+"_times"]),
            "epsilon": torch.randn(shape, device=self.device, generator=self.generators[phase+"_epsilon"]),
        }
        if phase == "R" and self.screen.get("guidance"):
            result["uncond"] = torch.rand(count, device=self.device, generator=self.generators["R_uncond"]) < \
                self.screen["guidance"]["p_uncond"]
        if phase == "T":
            r = self.cfg["rq"]
            shape_g = (count, self.cfg["history_max"]+self.cfg["list_length"], r["levels"], r["codes"])
            uniform = torch.rand(shape_g, device=self.device, generator=self.generators["T_gumbel"]).clamp_(1e-6, 1-1e-6)
            result["gumbel"] = -torch.log(-torch.log(uniform))
        return result

    def update(self, history, target, phase, *, draws=None):
        t = self.cfg["training"]
        count = len(history)
        if phase not in ("R", "T") or not 0 < count <= t["effective_batch"]:
            raise ValueError("Invalid phase or effective batch size")
        if history.shape[1] != self.cfg["history_max"] or target.shape[1] != self.cfg["list_length"]:
            raise ValueError("Batch geometry differs from protocol")
        self._set_mode(phase)
        optimizer = self.opt_r if phase == "R" else self.opt_t
        if phase == "R":
            lr = r_learning_rate(self.r_updates+1, self.total_r, self.cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            sigma = None
        else:
            sigma = gumbel_strength(self.t_updates+1, self.cfg)
            lr = t["base_lr"]
        optimizer.zero_grad(set_to_none=True)
        draws = self.draws(count, phase) if draws is None else draws
        sums = torch.zeros(len(self.loss_keys), device=self.device)
        micro = self.runtime["micro_r" if phase == "R" else "micro_t"]
        hlen, levels, dim = self.cfg["history_max"], self.cfg["rq"]["levels"], self.cfg["rq"]["latent_dim"]
        for start in range(0, count, micro):
            stop = min(start+micro, count)
            hist, tgt = history[start:stop], target[start:stop]
            n = stop-start
            true_z = self.catalog.z_pad[tgt]
            clean = self.catalog.x_pad[tgt] if self.gen_x_pad is None else self.gen_x_pad[tgt]
            times = draws["times"][start:stop]
            noisy = self.diffusion.noise(clean, times, draws["epsilon"][start:stop])
            rq_target = None
            with attention_backend(self.runtime["attention_backend"]):
                if phase == "T":
                    books = self.tokenizer.effective_books()
                    ids = torch.cat((hist, tgt), 1)
                    result = joint_quantize(
                        self.catalog.z_pad[ids].reshape(-1, dim), books,
                        self.catalog.stats["distance_scale"], self.catalog.sid_pad[ids].reshape(-1, levels),
                        self.model.embedding, draws["gumbel"][start:stop].reshape(-1, levels, self.cfg["rq"]["codes"]),
                        sigma, self.cfg)
                    embeddings = result["embedding"].view(n, ids.shape[1], levels, self.model.width)
                    history_embedding, targets_embedding = embeddings[:, :hlen], embeddings[:, hlen:]
                    rq_target = {
                        "q": result["q"].view(n, ids.shape[1], dim)[:, hlen:].reshape(-1, dim),
                        "vq": result["vq"].view(n, ids.shape[1])[:, hlen:]}
                else:
                    books = self.catalog.books
                    history_embedding = self.model.lookup(self.catalog.sid_pad[hist])
                    targets_embedding = self.model.lookup(self.catalog.sid_pad[tgt])
                head = self.screen.get("history_head")
                screen_args = {}
                if phase == "R" and "uncond" in draws:
                    screen_args["uncond"] = draws["uncond"][start:stop]
                if head:
                    screen_args["return_memory"] = True
                with autocast(self.runtime, self.device):
                    predicted = self.model.training_prediction(history_embedding, hist != 0, targets_embedding,
                                                               noisy, times, **screen_args)
                extra = None
                if head:
                    predicted, memory, valid = predicted
                    value = predicted.new_zeros((), dtype=torch.float32)
                    if phase == "R":
                        logits = self.model.history_logits(memory, valid, self.item_vectors, head["temperature"])
                        value = torch.nn.functional.cross_entropy(
                            logits.repeat_interleave(tgt.shape[1], 0), (tgt-1).reshape(-1))
                    extra = {"history": (head["weight"], value)}
                to_latent = None
                if self.model.to_latent is not None:
                    to_latent = lambda x: self.model.output_to_latent(x, self.catalog)
                losses = recommendation_losses(predicted, clean, true_z, self.catalog.sid_pad[tgt],
                                                books, self.catalog.stats, self.cfg, rq_target,
                                                catalog=self.catalog, target_ids=tgt,
                                                to_latent=to_latent, extra=extra)
                (losses["total"]*(n/count)).backward()
                sums += torch.stack([losses[k].detach() for k in self.loss_keys])*n
        params = self.model.trainable_parameters() if phase == "R" else self.tokenizer.joint_parameters()
        norm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"], error_if_nonfinite=True, foreach=True)
        optimizer.step()
        if phase == "R":
            self.r_updates += 1
        else:
            self.t_updates += 1
            self.t_in_block += 1
        self.sums[phase][:-1] += sums.double()
        self.sums[phase][-1] += norm.detach().double()*count
        self.counts[phase] += count
        return {"losses": sums/count, "grad_norm": norm.detach(), "lr": lr, "sigma": sigma}

    def summary(self, phase, reset=False):
        values = (self.sums[phase]/max(self.counts[phase], 1)).cpu().tolist()
        result = dict(zip((*self.loss_keys, "grad_norm"), values))
        supervision = 'landing' if 'landing' in result else 'align'
        result['weighted_'+supervision] = self.screen.get(supervision+'_weight', self.cfg['loss'][supervision])*result[supervision]
        if 'denoise_weight' in self.screen:
            result['weighted_denoise'] = self.screen['denoise_weight']*result['denoise']
        if 'history' in result:
            result['weighted_history'] = self.screen['history_head']['weight']*result['history']
        result.update(sessions=self.counts[phase],
                      weighted_latent=self.cfg["loss"]["latent"]*result["latent"],
                      weighted_vq=self.cfg["loss"]["vq"]*result["vq"])
        if reset:
            self.sums[phase].zero_()
            self.counts[phase] = 0
        return result

    @torch.no_grad()
    def refresh_catalog(self):
        old = self.catalog
        books, scales = self.tokenizer.books_with_diagnostics()
        hard = hard_quantize(old.z, books)
        asset = {**old.state(), "books": books, "sid": hard["sid"], "q": hard["q"], "version": old.version+1}
        new = Catalog(asset, self.device)  # Build fully, then swap atomically.
        change = old.sid != new.sid
        relative = (books-old.books).flatten(1).norm(dim=1)/old.books.flatten(1).norm(dim=1).clamp_min(1e-8)
        stats = {"sid_change": float(change.any(-1).float().mean()),
                 "sid_change_by_level": change.float().mean(0).cpu().tolist(),
                 "effective_relative_change": relative.cpu().tolist(),
                 **scales,
                 **code_health(hard["sid"], self.cfg["rq"]["codes"])}
        if any(scales["near_zero_base_scale"]):
            raise RuntimeError(f"Degenerate codebook scale at refresh: {scales}")
        self.catalog = new
        return stats

    def advance(self, data):
        """One optimizer update or one atomic refresh; caller handles validation."""
        if self.phase == "done":
            return {"kind": "done"}
        if self.phase == "refresh":
            diagnostics = self.refresh_catalog()
            losses = self.summary("T", reset=True)
            self.t_in_block = 0
            self.phase = "R"
            return {"kind": "T_block", "R": self.r_epoch, "T": self.t_updates,
                    "updates_in_block": self.cfg["training"]["t_steps_per_cycle"],
                    "sigma_first": gumbel_strength(self.t_updates-self.cfg["training"]["t_steps_per_cycle"]+1, self.cfg),
                    "sigma_last": gumbel_strength(self.t_updates, self.cfg),
                    "lr": self.cfg["training"]["base_lr"], "losses": losses, "encoding": diagnostics}
        phase = self.phase
        stream = self.r_stream if phase == "R" else self.t_stream
        ids = stream.take(self.cfg["training"]["effective_batch"])
        history, target = data.batch(ids, self.device)
        update = self.update(history, target, phase)
        event = {"kind": "update", "phase": phase, "R": self.r_epoch, "T": self.t_updates,
                 "R_updates": self.r_updates, "lr": update["lr"], "sigma": update["sigma"]}
        if phase == "R" and stream.at_end:
            self.r_epoch += 1
            event.update(kind="R_epoch", R=self.r_epoch, updates_in_epoch=self.steps_per_epoch,
                         losses=self.summary("R", reset=True))
            if self.r_epoch == self.cfg["training"]["epochs"]:
                self.phase = "done"
            elif self.r_epoch >= self.cfg["training"]["preheat_epochs"] and not self.frozen_after(self.r_epoch):
                self.phase = "T"
        elif phase == "T" and self.t_in_block == self.cfg["training"]["t_steps_per_cycle"]:
            self.phase = "refresh"
        return event

    def frozen_after(self, r_epoch):
        """In-run freeze: no tokenizer block (and no SID refresh) after R epoch
        training.freeze_after_epoch; R continues on the unchanged schedule. The
        quantization noise clock keeps the unfrozen total, like the accepted R9 freeze."""
        freeze = self.cfg["training"].get("freeze_after_epoch")
        if freeze is None:
            return False
        t = self.cfg["training"]
        if type(freeze) is not int or not t["preheat_epochs"] <= freeze <= t["epochs"]:
            raise ValueError("freeze_after_epoch must lie between preheat and the last R epoch")
        return r_epoch >= freeze

    def state(self):
        return cpu_copy({"format": "rqcd_mixer_joint_v1", "config": self.cfg, "runtime": self.runtime,
                         "binding": self.binding, "tokenizer": self.tokenizer.state_dict(),
                         "model": self.model.state_dict(), "diffusion": self.diffusion.state_dict(),
                         "catalog": self.catalog.state(), "optimizer_r": self.opt_r.state_dict(),
                         "optimizer_t": self.opt_t.state_dict(), "phase": self.phase,
                         "r_epoch": self.r_epoch, "r_updates": self.r_updates, "t_updates": self.t_updates,
                         "t_in_block": self.t_in_block, "r_stream": self.r_stream.state(),
                         "t_stream": self.t_stream.state(), "generators": {k: g.get_state() for k, g in self.generators.items()},
                         "rng": rng_state(), "sums": self.sums, "counts": self.counts,
                         "validated_epochs": self.validated_epochs, "best_score": self.best_score, "best_epoch": self.best_epoch})

    def restore(self, state):
        if (state["format"] != "rqcd_mixer_joint_v1" or state["config"] != self.cfg or
                state["runtime"] != self.runtime or state["binding"] != self.binding):
            raise ValueError("Joint resume config/runtime/source/asset mismatch")
        if not 0 <= state["r_updates"] <= self.total_r or not 0 <= state["t_updates"] <= total_t_steps(self.cfg):
            raise ValueError("Invalid optimizer counters")
        # Clocks must describe the same point as the saved data permutations.
        batch = self.cfg["training"]["effective_batch"]
        for prefix in ("r", "t"):
            stream = state[prefix+"_stream"]
            live_stream = getattr(self, prefix + '_stream')
            updates = stream["epoch"]*math.ceil(live_stream.epoch_size/batch) + math.ceil(stream["cursor"]/batch)
            if updates != state[prefix+"_updates"]:
                raise ValueError("Data cursor and optimizer update count disagree")
            saved_steps = [int(v["step"]) for v in state["optimizer_"+prefix]["state"].values()]
            if (updates and (not saved_steps or any(s != updates for s in saved_steps))) or (not updates and saved_steps):
                raise ValueError("Optimizer state has a different update clock")
        completed_r = state["r_stream"]["epoch"] + int(state["r_stream"]["cursor"] == self.r_stream.epoch_size)
        block = self.cfg["training"]["t_steps_per_cycle"]
        if (completed_r != state["r_epoch"] or not 0 <= state["t_in_block"] <= block or
                state["t_updates"] != state["catalog"]["version"]*block+state["t_in_block"] or
                state["phase"] not in ("R", "T", "refresh", "done") or
                (state["phase"] == "refresh" and state["t_in_block"] != block) or
                (state["phase"] in ("R", "done") and state["t_in_block"] != 0)):
            raise ValueError("Invalid phase/catalog/epoch clock")
        self.tokenizer.load_state_dict(state["tokenizer"])
        self.model.load_state_dict(state["model"])
        if self.model.to_latent is not None:
            self._build_generation_targets()
        self.diffusion.load_state_dict(state["diffusion"])
        self.catalog = Catalog(state["catalog"], self.device)
        if self.catalog.version_hash != state["catalog"]["version_hash"]:
            raise ValueError("Catalog state hash mismatch")
        # During a partially completed T block the cached catalog intentionally
        # differs from live codebooks. Do not rebuild it on resume.
        self.opt_r.load_state_dict(state["optimizer_r"])
        self.opt_t.load_state_dict(state["optimizer_t"])
        for key in ("phase", "r_epoch", "r_updates", "t_updates", "t_in_block",
                    "validated_epochs", "best_score", "best_epoch"):
            setattr(self, key, state[key])
        self.r_stream.restore(state["r_stream"])
        self.t_stream.restore(state["t_stream"])
        for key, value in state["generators"].items():
            self.generators[key].set_state(value.cpu())
        self.sums = {k: v.to(self.device) for k, v in state["sums"].items()}
        self.counts = dict(state["counts"])
        self.active_mode = None
        self._set_mode("T" if self.phase in ("T", "refresh") else "R")
        restore_rng(state["rng"])


class Pretrainer:
    def __init__(self, cfg, runtime, tokenizer, content, binding, target_data=None):
        self.cfg, self.runtime, self.binding = cfg, runtime, binding
        self.tokenizer, self.content = tokenizer, content
        self.target_data = target_data
        if cfg.get("target", {}).get("mode") == "hybrid" and target_data is None:
            raise ValueError("Missing hybrid pretraining teacher assets")
        self.loss_keys = ("loss", "reconstruction", "vq", "cf") if "target" in cfg else ("loss", "reconstruction", "vq")
        self.device = content.device
        self.stream = PermutationStream(len(content), seed_for(cfg["seed"], "pretrain_order"))
        self.optimizer = adamw(tokenizer.parameters(), cfg, runtime, self.device,
                               lr=cfg["rq"]["lr"], weight_decay=cfg["rq"]["weight_decay"])
        self.steps_per_epoch = math.ceil(len(content)/cfg["rq"]["batch_size"])
        self.updates = self.epochs = 0
        self.best_loss, self.best_collision = float("inf"), float("inf")
        self.best_loss_epoch = self.best_collision_epoch = None
        self.best_objective, self.best_objective_epoch = float("inf"), None
        self.sums = torch.zeros(len(self.loss_keys), device=self.device, dtype=torch.float64)
        self.samples = 0

    def losses(self, ids):
        kwargs = {}
        if self.cfg.get("target", {}).get("mode") == "hybrid":
            kwargs = {k: self.target_data[k][ids] for k in ("cf", "groups", "supported")}
        return self.tokenizer.pretrain_forward(self.content[ids], **kwargs)

    @torch.no_grad()
    def evaluate_objective(self):
        self.tokenizer.eval()
        # Same batches/negatives at every checkpoint, independent of training cursor/RNG.
        order = torch.randperm(len(self.content), generator=torch.Generator().manual_seed(
            seed_for(self.cfg['seed'], 'pretrain_eval'))).to(self.device)
        total = torch.zeros(len(self.loss_keys), dtype=torch.float64, device=self.device)
        for ids in order.split(self.cfg['rq']['batch_size']):
            losses = self.losses(ids)
            total += torch.stack([losses[k] for k in self.loss_keys]).double()*len(ids)
        return dict(zip(self.loss_keys, (total/len(order)).cpu().tolist()))

    def advance(self):
        if self.epochs >= self.cfg["rq"]["epochs"]:
            return {"kind": "done"}
        ids = self.stream.take(self.cfg["rq"]["batch_size"]).to(self.device)
        lr = pre_learning_rate(self.updates, self.steps_per_epoch, self.cfg)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.tokenizer.train()
        self.optimizer.zero_grad(set_to_none=True)
        with attention_backend(self.runtime["attention_backend"]):
            losses = self.losses(ids)
            losses["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.cfg["training"]["max_grad_norm"],
                                             error_if_nonfinite=True, foreach=True)
        self.optimizer.step()
        self.updates += 1
        self.sums += torch.stack([losses[k].detach() for k in self.loss_keys]).double()*len(ids)
        self.samples += len(ids)
        if not self.stream.at_end:
            return {"kind": "update", "update": self.updates}
        self.epochs += 1
        vals = (self.sums/self.samples).cpu().tolist()
        self.sums.zero_()
        count, self.samples = self.samples, 0
        return {"kind": "pretrain_epoch", "epoch": self.epochs, "update": self.updates,
                **dict(zip(self.loss_keys, vals)),
                "items": count, "lr": lr, "grad_norm": float(norm)}

    def state(self):
        return cpu_copy({"format": "rqcd_mixer_pretrain_v1", "config": self.cfg, "runtime": self.runtime,
                         "binding": self.binding, "tokenizer": self.tokenizer.state_dict(),
                         "optimizer": self.optimizer.state_dict(), "stream": self.stream.state(),
                         "updates": self.updates, "epochs": self.epochs, "sums": self.sums, "samples": self.samples,
                         "best_loss": self.best_loss, "best_collision": self.best_collision,
                         "best_loss_epoch": self.best_loss_epoch, "best_collision_epoch": self.best_collision_epoch,
                         "best_objective": self.best_objective, "best_objective_epoch": self.best_objective_epoch,
                         "rng": rng_state()})

    def restore(self, state):
        if (state["format"] != "rqcd_mixer_pretrain_v1" or state["config"] != self.cfg or
                state["runtime"] != self.runtime or state["binding"] != self.binding):
            raise ValueError("Pretraining resume identity differs")
        stream = state["stream"]
        updates = stream["epoch"]*self.steps_per_epoch+math.ceil(stream["cursor"]/self.cfg["rq"]["batch_size"])
        epochs = stream["epoch"]+int(stream["cursor"] == len(self.content))
        saved_steps = [int(v["step"]) for v in state["optimizer"]["state"].values()]
        if (updates != state["updates"] or epochs != state["epochs"] or
                (updates and (not saved_steps or any(s != updates for s in saved_steps)))):
            raise ValueError("Pretraining optimizer/data/epoch clocks disagree")
        self.tokenizer.load_state_dict(state["tokenizer"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.stream.restore(state["stream"])
        for key in ("updates", "epochs", "samples", "best_loss", "best_collision", "best_loss_epoch", "best_collision_epoch"):
            setattr(self, key, state[key])
        self.best_objective = state.get("best_objective", float("inf"))
        self.best_objective_epoch = state.get("best_objective_epoch")
        self.sums = state["sums"].to(self.device)
        restore_rng(state["rng"])
