"""Atomic best/last transaction over at most two committed immutable checkpoints."""
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import torch

from .data import sha256
from .protocol import atomic_json


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    receipt = {"sha256": sha256(path), "epoch": state["epoch"], "binding": state["binding"]}
    atomic_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    if sha256(path) != receipt["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    state = torch.load(path, map_location=device, weights_only=False)
    if state["binding"] != receipt["binding"] or state["epoch"] != receipt["epoch"]:
        raise ValueError("Checkpoint receipt mismatch")
    return state


def publish_alias(source, target):
    source, target = Path(source), Path(target)
    temporary = target.with_name(target.name + ".alias-" + uuid.uuid4().hex)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    receipt = json.loads(source.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    atomic_json(target.with_suffix(".receipt.json"), receipt)


def checked_storage(output, relative):
    output = Path(output)
    path = output / relative
    if (not path.resolve().is_relative_to((output / "checkpoint_epochs").resolve())
            or path.is_symlink() or not path.is_file()):
        raise ValueError("Invalid checkpoint transaction storage path")
    return path


def commit_epoch(output, state, improved):
    output = Path(output)
    storage = output / "checkpoint_epochs"
    storage.mkdir(exist_ok=True)
    relative = f"checkpoint_epochs/epoch_{state['epoch']:04d}_{uuid.uuid4().hex}.pt"
    state["epoch_storage_file"] = relative
    if improved:
        state["best_storage_file"] = relative
    receipt = atomic_checkpoint(output / relative, state)
    best = checked_storage(output, state["best_storage_file"])
    best_receipt = json.loads(best.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    publish_alias(output / relative, output / "last.pt")
    if improved:
        publish_alias(best, output / "best.pt")
        atomic_json(output / "best_validation.json", {"epoch": state["epoch"],
                    "metrics": state["validation_metrics"], "checkpoint_sha256": receipt["sha256"],
                    "binding": state["binding"]})
    # This is the sole commit point. A crash before it keeps the previous
    # immutable best/last pair recoverable even if an alias was partly updated.
    atomic_json(output / "CURRENT.json", {
        "epoch": state["epoch"], "binding": state["binding"],
        "last": {"file": relative, "sha256": receipt["sha256"]},
        "best": {"file": state["best_storage_file"], "sha256": best_receipt["sha256"]}})
    keep = {relative, state["best_storage_file"]}
    for path in storage.glob("epoch_*.pt"):
        if (path.relative_to(output).as_posix() not in keep and not path.is_symlink()
                and re.fullmatch(r"epoch_\d{4}_[0-9a-f]{32}\.pt", path.name)):
            path.unlink()
            path.with_suffix(".receipt.json").unlink(missing_ok=True)
    return state["best_storage_file"]


def repair_committed(output):
    output = Path(output)
    current = json.loads((output / "CURRENT.json").read_text(encoding="utf-8"))
    states = {}
    for kind in ("last", "best"):
        source = checked_storage(output, current[kind]["file"])
        if sha256(source) != current[kind]["sha256"]:
            raise ValueError("Committed immutable checkpoint changed")
        state = load_checkpoint(source)
        if state["binding"] != current["binding"]:
            raise ValueError("Checkpoint transaction binding differs")
        states[kind] = state
        publish_alias(source, output / (kind + ".pt"))
    if (states["last"]["epoch_storage_file"] != current["last"]["file"]
            or states["last"]["best_storage_file"] != current["best"]["file"]):
        raise ValueError("Checkpoint transaction references differ")
    best = states["best"]
    atomic_json(output / "best_validation.json", {"epoch": best["epoch"],
                "metrics": best["validation_metrics"], "checkpoint_sha256": current["best"]["sha256"],
                "binding": best["binding"]})
    return states["last"]
