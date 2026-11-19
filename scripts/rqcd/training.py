"""Update-boundary training with deterministic full-group diffusion draws."""
from __future__ import annotations

import math
import torch

from .common import (optimizer, learning_rate, derive_seed, cpu_copy, rng_state,
                     restore_rng, attention_backend)
from .model import ContinuousListModel, Diffusion


class Trainer:
    def __init__(self, cfg, catalog, train_count, binding):
        self.cfg, self.catalog, self.binding = cfg, catalog, binding
        self.device = catalog.y.device
        self.model = ContinuousListModel(cfg).to(self.device)
        self.diffusion = Diffusion(cfg).to(self.device)
        self.opt = optimizer(self.model.parameters(), cfg, self.device)
        self.train_count = train_count
        self.groups = math.ceil(train_count/cfg["training"]["effective_batch"])
        self.total = self.groups*cfg["training"]["epochs"]
        self.updates = 0
        self.completed_epochs = []
        self.best_score, self.best_epoch = -1.0, None
        self.noise_rng = torch.Generator(device=self.device).manual_seed(derive_seed(cfg["seed"], "epsilon"))
        self.time_rng = torch.Generator(device=self.device).manual_seed(derive_seed(cfg["seed"], "tau"))
        self.bf16 = cfg["training"]["precision"] == "bf16"
        if self.bf16 and self.device.type != "cuda":
            raise ValueError("Formal BF16 mode requires CUDA; use explicit FP32 for CPU tests")
        self.statistics = torch.zeros(1+cfg["list_length"], device=self.device, dtype=torch.float64)
        self.stat_count = 0

    def group_noise(self, count):
        shape = (count, self.cfg["list_length"], self.cfg["rq"]["latent_dim"])
        epsilon = torch.randn(shape, generator=self.noise_rng, device=self.device)
        times = torch.randint(1, self.diffusion.steps+1, shape[:2],
                              generator=self.time_rng, device=self.device)
        return times, epsilon

    def update(self, history_ids, target_ids, *, draws=None):
        t = self.cfg["training"]
        count = len(history_ids)
        if not 0 < count <= t["effective_batch"] or self.updates >= self.total:
            raise ValueError("Update outside fixed budget/batch")
        self.model.train()
        lr = learning_rate(self.updates+1, self.total, t["warmup_updates"], t["lr"])
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        times, epsilon = self.group_noise(count) if draws is None else draws
        sums = torch.zeros(self.cfg["list_length"], device=self.device)
        # Generate all random labels before splitting: changing microbatch does
        # not change which timestep/noise is paired with each training example.
        for start in range(0, count, t["microbatch"]):
            end = min(start+t["microbatch"], count)
            history = self.catalog.y[history_ids[start:end]]
            targets = self.catalog.y[target_ids[start:end]]
            tau = times[start:end]
            noisy = self.diffusion.noise(targets, tau, epsilon[start:end])
            with attention_backend(self.cfg["runtime"]["sdpa_backend"]):
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                    predicted = self.model.training_prediction(
                        history, history_ids[start:end].ne(0), targets, noisy, tau)
                error = (predicted.float()-targets.float()).square().mean(-1)
                # Mean over actual group, including the 835-row last group.
                loss = error.sum()/(count*self.cfg["list_length"])
                loss.backward()
            sums += error.detach().sum(0)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), t["max_grad_norm"], error_if_nonfinite=True,
            foreach=True,
        )
        self.opt.step()
        self.updates += 1
        position_mean = sums/count
        self.statistics[0] += sums.mean().double()
        self.statistics[1:] += sums.double()
        self.stat_count += count
        return {"loss": position_mean.mean().detach(), "position_loss": position_mean.detach(),
                "grad_norm": norm.detach(), "lr": lr}

    def summary(self, reset=False):
        result = (self.statistics/max(self.stat_count, 1)).cpu().tolist()
        out = {"mse": result[0], "position_mse": result[1:], "lists": self.stat_count}
        if reset:
            self.statistics.zero_()
            self.stat_count = 0
        return out

    def state(self):
        return cpu_copy({
            "format": "rqcd_trainer_v1", "config": self.cfg, "binding": self.binding,
            "model": self.model.state_dict(), "diffusion": self.diffusion.state_dict(),
            "optimizer": self.opt.state_dict(), "updates": self.updates,
            "train_count": self.train_count, "total_updates": self.total,
            "scheduler": {"completed_updates": self.updates, "total": self.total,
                          "warmup": self.cfg["training"]["warmup_updates"]},
            "rng": rng_state(), "noise_rng": self.noise_rng.get_state(),
            "time_rng": self.time_rng.get_state(), "completed_epochs": self.completed_epochs,
            "best_score": self.best_score, "best_epoch": self.best_epoch,
            "statistics": self.statistics, "stat_count": self.stat_count,
            "frozen_hash": self.catalog.frozen_hash,
        })

    def restore(self, state):
        if (state["format"] != "rqcd_trainer_v1" or state["config"] != self.cfg or
                state["binding"] != self.binding or state["train_count"] != self.train_count or
                state["frozen_hash"] != self.catalog.frozen_hash or state["total_updates"] != self.total):
            raise ValueError("Checkpoint identity/configuration mismatch")
        updates = state["updates"]
        if not 0 <= updates <= self.total or state["scheduler"]["completed_updates"] != updates:
            raise ValueError("Invalid checkpoint clock")
        for parameter in state["optimizer"]["state"].values():
            if float(parameter["step"]) != updates:
                raise ValueError("Adam clock mismatch")
        self.model.load_state_dict(state["model"])
        self.diffusion.load_state_dict(state["diffusion"])
        self.opt.load_state_dict(state["optimizer"])
        self.updates = updates
        self.noise_rng.set_state(state["noise_rng"].cpu())
        self.time_rng.set_state(state["time_rng"].cpu())
        self.completed_epochs = list(state["completed_epochs"])
        self.best_score, self.best_epoch = state["best_score"], state["best_epoch"]
        self.statistics.copy_(state["statistics"])
        self.stat_count = state["stat_count"]
        restore_rng(state["rng"])
