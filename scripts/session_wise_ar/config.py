"""Immutable scientific presets; CPU test configurations are never CLI presets."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "configs/session_wise_ar"
DESIGN = ROOT / "docs/research/session_wise_ar_mixer/FIVE_ARM_DESIGN_20261113.md"
ARMS = {"AR-T0": ("tiger", False), "AR-T1": ("tiger", True),
        "AR-D0": ("diger", False), "AR-D1": ("diger", True),
        # 2026-11-14 versioned TIGER rerun: only the RQ pretraining optimizer differs.
        "AR-T0A": ("tiger_adamw", False), "AR-T1A": ("tiger_adamw", True)}


def load(arm):
    family, mixer = ARMS[arm]
    cfg = json.loads((DIRECTORY / f"{family}.json").read_text())
    if hashlib.sha256(DESIGN.read_bytes()).hexdigest() != cfg["design_sha256"]:
        raise ValueError("Approved design bytes changed; version the protocol before execution")
    cfg.update(arm=arm, mixer_enabled=mixer)
    return copy.deepcopy(cfg)


def plan(cfg):
    tasks = cfg["split_counts"]["train"] * cfg["list_length"]
    per_epoch = math.ceil(tasks / cfg["training"]["batch_size"])
    steps = (cfg["training"]["max_steps"] if cfg["family"] == "tiger" else
             cfg["training"]["max_epochs"] * per_epoch)
    return {"arm": cfg["arm"], "tasks_per_session_epoch": tasks, "updates_per_epoch": per_epoch,
            "max_recommender_updates": steps,
            "max_joint_tokenizer_updates": steps if cfg["family"] == "diger" else 0,
            "pretrain_updates": math.ceil(cfg["catalog_size"]/cfg["rq"]["batch_size"])*cfg["rq"]["epochs"],
            "training_started": False}
