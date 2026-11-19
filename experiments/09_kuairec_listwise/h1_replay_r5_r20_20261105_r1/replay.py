"""Run the unchanged, checkpoint-bound H1 trainer; retain validation boundaries.

The original 35R/400T plan is never shortened. The only training-loop hook is
checkpoint publication: retain complete validation boundaries and stop at R20.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ORIGINAL = Path("/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1")
DEFAULT_ROOT = ORIGINAL.parent / "h1_replay_r5_r20_20261105_r1"
PARENT_SHA = "991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1"
BOUNDARIES = {9: (8, 40), 13: (11, 80), 17: (14, 120), 21: (17, 160), 25: (20, 200)}
FIELDS = ("model", "alignment", "r_optimizer", "t_optimizer", "scheduler", "rng",
          "r_updates", "t_updates", "binding", "precision", "arm")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def differences(a, b, name="", out=None):
    """Exact nested state comparison, including CPU/CUDA tensor values and RNG."""
    import numpy as np
    import torch
    if out is None:
        out = []
    if len(out) >= 30:
        return out
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            out.append(name)
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
            out.append(name)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            out.append(name + ".keys")
        else:
            for key in a:
                differences(a[key], b[key], name + "." + str(key), out)
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            out.append(name)
        else:
            for index, (left, right) in enumerate(zip(a, b)):
                differences(left, right, name + f"[{index}]", out)
    elif a != b:
        out.append(name)
    return out


def import_original():
    os.environ["DIFFGRM_ENGINE_ROOT"] = str(ORIGINAL / "code/external/DiffGRM")
    sys.path.insert(0, str(ORIGINAL / "code"))
    from scripts.listwise_diffgrm import run, training
    if Path(run.__file__).resolve() != ORIGINAL / "code/scripts/listwise_diffgrm/run.py":
        raise RuntimeError("Wrong trainer imported")
    return run, training


def preflight(root, device):
    import torch
    run, training = import_original()
    parent_path = ORIGINAL / "runs/H1/initial.pt"
    if sha256(parent_path) != PARENT_SHA:
        raise ValueError("Original H1 R5 parent changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    config = json.loads((ORIGINAL / "runs/H1/resolved_config.json").read_text())
    if parent["binding"]["config"] != config:
        raise ValueError("Resolved original config differs from the parent binding")
    if (parent["r_updates"], parent["t_updates"], parent["progress"]["phase"],
            parent["progress"]["update"], parent["arm"]) != (3465, 0, 5, 0, "H1"):
        raise ValueError("Not a completed H1 R5/T0 parent")
    trainer, assets, data, binding = run.build_context(config, device, "bf16", True)
    if binding != parent["binding"]:
        raise ValueError("Frozen source/config/data/assets binding mismatch")
    if len(trainer.plan) != 45 or sum(p["updates"] for p in trainer.plan if p["kind"] == "T") != 400:
        raise ValueError("Original schedule horizon changed")
    for phase, (epoch, t_updates) in BOUNDARIES.items():
        row = trainer.plan[phase - 1]
        counts = training.expected_update_counts(trainer.plan, {"phase": phase, "update": 0})
        if (row["kind"], row["epoch"], row["validate"], counts["R"], counts["T"]) != (
                "R", epoch, True, epoch * 693, t_updates):
            raise ValueError("Boundary mapping differs from the original phase plan")
    saved_restore_rng = training.restore_rng
    try:
        if device == "cpu":
            # CPU preflight cannot restore a GPU stream. This exception is only
            # for inspection; the GPU gate and training use unmodified restore.
            training.restore_rng = lambda state: saved_restore_rng(dict(state, cuda=None))
        progress = trainer.restore(parent, binding, arm="H1")
        restored = trainer.checkpoint(binding, progress, "H1")
        fields = FIELDS if device.startswith("cuda") else tuple(k for k in FIELDS if k != "rng")
        delta = differences({k: parent[k] for k in fields}, {k: restored[k] for k in fields})
        if device == "cpu":
            delta += differences({k: v for k, v in parent["rng"].items() if k != "cuda"},
                                 {k: v for k, v in restored["rng"].items() if k != "cuda"}, "cpu_rng")
        if delta:
            raise ValueError("Parent restore differs: " + repr(delta))
        if device.startswith("cuda") and (torch.cuda.device_count() != 1
                                           or "H100" not in torch.cuda.get_device_name(0)):
            raise ValueError("Expected exactly one visible H100")
        result = dict(status="passed", parent=str(parent_path), parent_sha256=PARENT_SHA,
                      source=str(ORIGINAL / "code"), binding_exact=True,
                      model_optimizer_scheduler_restore_exact=True,
                      cuda_rng_restore_exact=device.startswith("cuda"),
                      total_r_epochs=35, total_t_updates=400, stop_r_epoch=20,
                      microbatch=config["training"]["microbatch_size"],
                      effective_batch=config["training"]["effective_batch_size"],
                      torch_version=torch.__version__, device=device,
                      wrapper_sha256=sha256(__file__), checked_unix=time.time())
        write_json(root / ("GPU_PREFLIGHT.json" if device.startswith("cuda") else "CPU_PREFLIGHT.json"), result)
        print(json.dumps(result), flush=True)
    finally:
        training.restore_rng = saved_restore_rng
    del trainer, assets, data, parent, restored
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return config


class StopAtR20(Exception):
    pass


class BoundaryPublisher:
    def __init__(self, original_publish, root):
        self.original_publish = original_publish
        self.root = Path(root)

    def __call__(self, output, state, *, save_best):
        output = Path(output)
        phase = state["progress"]["phase"]
        if phase > 25:
            raise RuntimeError("Refusing to advance beyond R20")
        self.original_publish(output, state, save_best=save_best)
        print(json.dumps(dict(event="phase_saved", phase=phase,
                              r_updates=state["r_updates"], t_updates=state["t_updates"])), flush=True)
        if phase not in BOUNDARIES:
            return
        epoch, t_updates = BOUNDARIES[phase]
        if (state["progress"]["update"], state["r_updates"], state["t_updates"]) != (0, 693 * epoch, t_updates):
            raise ValueError("Incomplete or mismatched validation boundary")
        evaluation = Path(state["progress"]["validations"][-1])
        if evaluation.parent != output or not (evaluation / "summary.json").is_file():
            raise ValueError("Validation for this boundary is not complete")
        summary = json.loads((evaluation / "summary.json").read_text())
        if (summary["r_updates"], summary["t_updates"], summary["examples"],
                summary["full_split"], summary["beam"]) != (693 * epoch, t_updates, 71657, True, 32):
            raise ValueError("Full-validation protocol mismatch")
        destination = output / "checkpoints" / f"R{epoch:02d}_T{t_updates:03d}.pt"
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Immutable checkpoint already exists: {destination}")
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copyfile(output / "last.pt", temporary)
        if sha256(temporary) != sha256(output / "last.pt"):
            raise ValueError("Checkpoint copy failed byte verification")
        temporary.replace(destination)
        receipt = dict(epoch=epoch, t_updates=t_updates, phase=phase, checkpoint=str(destination),
                       checkpoint_sha256=sha256(destination), validation=str(evaluation),
                       summary_sha256=sha256(evaluation / "summary.json"),
                       ndcg5=summary["metrics"]["list/ndcg@5"])
        write_json(output / "checkpoints" / f"R{epoch:02d}.json", receipt)
        print(json.dumps(dict(event="validation_checkpoint_preserved", **receipt)), flush=True)
        if epoch == 20:
            raise StopAtR20()


def compare_predictions(left, right):
    unequal = 0
    rows = 0
    # gzip headers contain timestamps; compare uncompressed content.
    with gzip.open(left, "rt", encoding="utf-8") as a, gzip.open(right, "rt", encoding="utf-8") as b:
        for x, y in itertools.zip_longest(a, b):
            rows += 1
            if x is None or y is None or (x != y and json.loads(x) != json.loads(y)):
                unequal += 1
    return dict(rows=rows, differing_rows=unequal, exact=rows == 71657 and unequal == 0)


def verify(root):
    import torch
    run, _ = import_original()
    output = root / "runs/H1-Replay-R20"
    records = []
    for phase, (epoch, t_updates) in BOUNDARIES.items():
        receipt = json.loads((output / "checkpoints" / f"R{epoch:02d}.json").read_text())
        checkpoint = Path(receipt["checkpoint"])
        if sha256(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("Immutable checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (state["progress"]["phase"], state["progress"]["update"],
                state["r_updates"], state["t_updates"]) != (phase, 0, epoch * 693, t_updates):
            raise ValueError("Saved checkpoint cursor mismatch")
        reference = ORIGINAL / "runs/H1" / f"validation_R{epoch:02d}"
        current = Path(receipt["validation"])
        a = json.loads((reference / "summary.json").read_text())
        b = json.loads((current / "summary.json").read_text())
        keys = ("split", "examples", "full_split", "dataset_binding", "beam", "inference_precision",
                "architecture", "seed", "r_updates", "t_updates", "metrics", "metric_protocol")
        delta = differences({k: a[k] for k in keys}, {k: b[k] for k in keys})
        predictions = compare_predictions(reference / "per_sample.jsonl.gz", current / "per_sample.jsonl.gz")
        records.append(dict(epoch=epoch, t_updates=t_updates,
                            reference_ndcg5=a["metrics"]["list/ndcg@5"],
                            replay_ndcg5=b["metrics"]["list/ndcg@5"],
                            summary_differences=delta, predictions=predictions,
                            checkpoint_sha256=receipt["checkpoint_sha256"]))
        del state
    # The original R20 best is available, so verify actual model and training
    # state as well as outputs. Output-directory paths in progress intentionally differ.
    original = torch.load(ORIGINAL / "runs/H1/best.pt", map_location="cpu", weights_only=False)
    replay = torch.load(output / "checkpoints/R20_T200.pt", map_location="cpu", weights_only=False)
    state_delta = differences({k: original[k] for k in FIELDS}, {k: replay[k] for k in FIELDS})
    del original, replay
    gc.collect()
    # Restore all saved states with the original strict loader, including CUDA
    # RNG, so saved boundaries are usable parents rather than just loadable files.
    config = json.loads((output / "resolved_config.json").read_text())
    trainer, assets, data, binding = run.build_context(config, "cuda:0", "bf16", True)
    restore_receipts = []
    for phase, (epoch, t_updates) in BOUNDARIES.items():
        path = output / "checkpoints" / f"R{epoch:02d}_T{t_updates:03d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        run.validate_evaluation_checkpoint(state, trainer.plan)
        progress = trainer.restore(state, binding, arm="H1")
        restored = trainer.checkpoint(binding, progress, "H1")
        delta = differences({k: state[k] for k in FIELDS}, {k: restored[k] for k in FIELDS})
        if delta:
            raise ValueError(f"R{epoch} actual GPU state restore mismatch: {delta}")
        restore_receipts.append(dict(epoch=epoch, strict_restore_exact=True))
        del state, restored
    exact = not state_delta and all(not row["summary_differences"] and row["predictions"]["exact"]
                                    for row in records)
    result = dict(status="exact_replay_verified" if exact else "replay_completed_with_differences",
                  exact_replay_verified=exact, boundaries=records, r20_state_differences=state_delta,
                  gpu_restore_checks=restore_receipts, parent_sha256=PARENT_SHA,
                  r_updates=13860, t_updates=200, stop_epoch=20)
    write_json(root / "REPLAY_VERIFICATION.json", result)
    print(json.dumps(result), flush=True)
    if exact:
        # User authorized removal of this replay's redundant artifacts only
        # AFTER exact verification. Keep every immutable boundary and validation.
        protected_hashes = {row["checkpoint_sha256"] for row in records}
        deleted = []
        for filename in ("best.pt", "last.pt"):
            path = output / filename
            if path.is_file() and sha256(path) in protected_hashes:
                deleted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
                path.unlink()
        write_json(root / "CLEANUP_RECEIPT.json",
                   dict(status="completed", gate="exact_replay_verified",
                        deleted=deleted, all_validation_checkpoints_retained=True))
    return exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "train", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        preflight(args.root, args.device)
        return
    if args.mode == "verify":
        raise SystemExit(0 if verify(args.root) else 3)
    output = args.root / "runs/H1-Replay-R20"
    if output.exists():
        raise FileExistsError("Existing run must be inspected before any explicit resume")
    config = preflight(args.root, "cuda:0")
    run, _ = import_original()
    original_publish = run.publish_checkpoint
    run.publish_checkpoint = BoundaryPublisher(original_publish, args.root)
    try:
        try:
            run.train(SimpleNamespace(command="train", arm="H1", output=str(output),
                                      resume=str(ORIGINAL / "runs/H1/initial.pt"),
                                      shared_warmup=None, device="cuda:0", precision="bf16"), config)
        except StopAtR20:
            write_json(output / "complete.json", dict(status="training_validation_complete",
                       stop_epoch=20, r_updates=13860, t_updates=200, original_horizon_retained=True))
        else:
            raise RuntimeError("Original trainer returned without the R20 stop gate")
    finally:
        run.publish_checkpoint = original_publish
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    raise SystemExit(0 if verify(args.root) else 3)


if __name__ == "__main__":
    main()
