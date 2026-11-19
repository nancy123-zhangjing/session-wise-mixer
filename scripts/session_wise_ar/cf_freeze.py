"""R9 -> R35 CF continuation. Original CF engine and scientific config stay immutable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.rqcd_mixer import run, training
from scripts.rqcd_mixer.common import atomic_save, read_state, sha256, write_json
from scripts.rqcd_mixer.config import load_config
from scripts.session_wise_ar.cf import CONFIG, REFERENCE, verify_parent

BaseTrainer = training.JointTrainer
POLICY = {"name": "cf_freeze_r9_v1", "start_R": 9, "final_R": 35, "fixed_T": 40}


class FrozenTrainer(BaseTrainer):
    def update(self, history, target, phase, **kwargs):
        if phase != "R":
            raise ValueError("Frozen continuation forbids tokenizer optimizer updates")
        return super().update(history, target, phase, **kwargs)

    def refresh_catalog(self):
        raise ValueError("Frozen continuation forbids SID/catalog refresh")

    def advance(self, data):
        if self.phase == "T":
            self.phase = "R"
        event = super().advance(data)
        if self.phase == "T":
            self.phase = "R"
        if self.t_updates != 40 or self.t_in_block != 0:
            raise ValueError("Tokenizer clock changed")
        return event

    def state(self):
        result = super().state()
        result["freeze_policy"] = POLICY
        return result

    def restore(self, state):
        marker = state.get("freeze_policy")
        if marker != POLICY and not (marker is None and state["r_epoch"] == 9):
            raise ValueError("Checkpoint is not from this freeze protocol or its R9 parent")
        if state["t_updates"] != 40 or state["t_in_block"] != 0:
            raise ValueError("Expected complete R9/T40 tokenizer state")
        super().restore(state)
        if self.phase == "T":
            self.phase = "R"
        self.active_mode = None
        self._set_mode("R")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["B0", "B1"], required=True)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--parent-sha", required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, default=REFERENCE / "runtime.locked.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = load_config(CONFIG)
    assert cfg["training"]["epochs"] == 35
    verify_parent(args.prepared, args.runtime)
    parent = args.parent_run / "checkpoints/R09_T040.pt"
    if sha256(parent) != args.parent_sha:
        raise ValueError("Parent R9 checkpoint checksum mismatch")
    best = json.loads((args.parent_run / "BEST.json").read_text())
    if best["R"] != 9 or best["sha256"] != args.parent_sha:
        raise ValueError("Parent validation selection differs from approved R9")
    root = args.output.resolve()
    parent_root = args.parent_run.resolve()
    if root == parent_root or root in parent_root.parents or parent_root in root.parents:
        raise ValueError("Use an independent branch output directory")
    training.JointTrainer = FrozenTrainer
    runtime, environment = run._runtime(args, cfg)
    trainer, _ = run._joint(args, cfg, runtime, environment)
    provenance = {"policy": POLICY, "parent_checkpoint": str(parent),
                  "parent_sha256": args.parent_sha, "arm": args.arm,
                  "wrapper_sha256": sha256(__file__)}
    if (root / "BRANCH.json").exists():
        if json.loads((root / "BRANCH.json").read_text()) != provenance:
            raise ValueError("Branch provenance changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing to initialize a nonempty output")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "BRANCH.json", provenance)
    if not (root / "last.pt").exists():
        trainer.restore(read_state(parent))
        if trainer.r_epoch != 9 or trainer.r_updates != 9 * trainer.steps_per_epoch:
            raise ValueError("Parent is not an R9 boundary")
        trainer.validated_epochs = sorted(set(trainer.validated_epochs + [9]))
        trainer.best_score, trainer.best_epoch = best["validation_ndcg@5"], 9
        (root / "checkpoints").mkdir(exist_ok=True)
        shutil.copyfile(parent, root / "checkpoints/R09_T040.pt")
        write_json(root / "BEST.json", best)
        for name, value in (("config.json", cfg), ("runtime.json", runtime),
                            ("binding.json", trainer.binding)):
            write_json(root / name, value)
        write_json(root / "INHERITED_VALIDATION.json", {
            "source": str(args.parent_run / "validation/R09/summary.json"),
            "selection": best, "note": "R9 score inherited; no new evaluation"})
        atomic_save(root / "last.pt", trainer.state())
        # A real GPU full-state restore, without an extra training update.
        trainer.restore(read_state(root / "last.pt"))
        write_json(root / "RESTORE_CHECK.json", {"passed": True, "R": trainer.r_epoch,
                   "T": trainer.t_updates, "catalog_hash": trainer.catalog.version_hash,
                   "scope": "strict full-state load; no additional optimizer update"})
    del trainer
    args.resume = True
    if not (root / "TRAINING_COMPLETE.json").exists():
        run.train(args, cfg)
    args.run = root
    if not (root / "TEST_COMPLETE.json").exists():
        run.test(args, cfg)


if __name__ == "__main__":
    main()
