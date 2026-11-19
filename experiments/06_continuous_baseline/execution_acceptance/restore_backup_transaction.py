"""Rebuild checkpoint transaction names in a new local copy, then CPU-load it.

The immutable archives are never edited. This does not train or resume a finished
run. CUDA RNG is preserved and hashed but cannot be executed on the CPU host.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", default="formal__sasrec_fullce_13__seed3031")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"(?:formal|legacy|confirmation|search)__[a-z0-9_]+__seed\d+", args.task_id)
    assert args.attempt >= 0
    manifest = read(args.manifest)
    assert manifest["status"] == "BACKUP_ARCHIVES_CREATED" and manifest["source_sha256"] == SOURCE
    destination = Path(args.output).resolve()
    assert destination.is_relative_to(ROOT / "outputs/stage06_continuous") and not destination.exists()
    prefix = f"campaign/runs/{args.task_id}/attempt_{args.attempt}/"
    names = {n for n in manifest["files"] if n.startswith(prefix)}
    assert {prefix + n for n in ("CURRENT.json", "run.json", "best.pt", "last.pt", "best.receipt.json", "last.receipt.json")} <= names
    archives = Path(args.archives).resolve()
    needed_parts = {manifest["files"][n]["archive"] for n in names}
    verified_parts = []
    for part in manifest["archives"]:
        if part["file"] not in needed_parts:
            continue
        path = archives / part["file"]
        assert path.stat().st_size == part["bytes"] and sha(path) == part["sha256"]
        verified_parts.append(part["file"])
    destination.mkdir(parents=True)
    extracted = set()
    for part in verified_parts:
        with tarfile.open(archives / part, "r|gz") as stream:
            for member in stream:
                if member.name not in names:
                    continue
                suffix = member.name.removeprefix(prefix)
                assert PurePosixPath(suffix).name == suffix and member.isfile()
                assert member.name not in extracted
                target = destination / suffix
                with target.open("xb") as output:
                    shutil.copyfileobj(stream.extractfile(member), output, 1024 * 1024)
                assert target.stat().st_size == manifest["files"][member.name]["bytes"]
                assert sha(target) == manifest["files"][member.name]["sha256"]
                extracted.add(member.name)
    assert extracted == names
    current, run = read(destination / "CURRENT.json"), read(destination / "run.json")
    assert current["binding"] == run["binding"] and current["binding"]["source_sha256"] == SOURCE
    # Only the two transaction targets are created. No arbitrary archive path is followed.
    aliases = {}
    for label in ("best", "last"):
        relative = current[label]["file"]
        assert re.fullmatch(r"checkpoint_epochs/epoch_\d{4}_[0-9a-f]{32}\.pt", relative)
        source, receipt = destination / (label + ".pt"), destination / (label + ".receipt.json")
        expected = current[label]["sha256"]
        assert sha(source) == read(receipt)["sha256"] == expected
        target = destination / relative
        assert target.resolve().is_relative_to(destination) and not target.is_symlink()
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            assert sha(target) == expected
            assert read(target.with_suffix(".receipt.json")) == read(receipt)
        else:
            shutil.copyfile(source, target)
            shutil.copyfile(receipt, target.with_suffix(".receipt.json"))
        aliases[label] = {"file": relative, "sha256": expected}
    # Bind the original consumer to the exact frozen source before importing it.
    for relative, expected in run["source"]["files"].items():
        assert sha(ROOT / relative) == expected
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from scripts.continuous import checkpoints
    from scripts.continuous.model import ContinuousRecommender
    from scripts.continuous.trainer import model_config, random_state, restore_random_state
    assert Path(checkpoints.__file__).resolve() == ROOT / "scripts/continuous/checkpoints.py"
    torch.set_num_threads(2)
    state = checkpoints.repair_committed(destination)
    assert state["binding"] == current["binding"] and state["epoch"] == current["epoch"]
    assert state["epoch_storage_file"] == current["last"]["file"] and state["best_storage_file"] == current["best"]["file"]
    model = ContinuousRecommender(model_config(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    assert all(torch.equal(value, state["model"][key]) for key, value in model.state_dict().items())
    optimizer = torch.optim.Adam(model.parameters(), lr=state["config"]["lr"],
        betas=tuple(state["config"]["adam_betas"]), weight_decay=state["config"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    assert len(optimizer.state_dict()["state"]) == len(state["optimizer"]["state"])
    assert scheduler.state_dict() == state["scheduler"]
    for key, stored in state["optimizer"]["state"].items():
        for name, value in stored.items():
            actual = optimizer.state_dict()["state"][key][name]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value
    original_rng = random_state()
    stored_rng = state["random_state"]
    try:
        restore_random_state({**stored_rng, "cuda": None})
        actual_rng = random_state()
        assert actual_rng["python"] == stored_rng["python"]
        assert actual_rng["numpy"][0] == stored_rng["numpy"][0]
        assert np.array_equal(actual_rng["numpy"][1], stored_rng["numpy"][1])
        assert actual_rng["numpy"][2:] == stored_rng["numpy"][2:]
        assert torch.equal(actual_rng["torch"], stored_rng["torch"])
    finally:
        restore_random_state(original_rng)
    cuda = stored_rng["cuda"]
    assert isinstance(cuda, list) and len(cuda) == 1
    assert all(v.dtype == torch.uint8 and v.ndim == 1 and v.numel() > 0 for v in cuda)
    for label in ("best", "last"):
        assert sha(destination / (label + ".pt")) == current[label]["sha256"]
    proof = {"status": "PASS", "task_id": args.task_id, "attempt": args.attempt, "source_sha256": SOURCE,
        "recovery_directory": str(destination), "verified_archive_parts": verified_parts,
        "extracted_files": len(extracted), "rebuilt_transaction_files": aliases,
        "epoch": state["epoch"], "updates": state["updates"], "best_epoch": state["best_epoch"],
        "original_repair_committed_passed": True, "strict_model_restore": True,
        "optimizer_state_restore": True, "scheduler_state_restore": True, "cpu_random_state_restore": True,
        "cuda_rng_bytes_sha256": [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in cuda],
        "completed_marker_preserved": (destination / "COMPLETE.json").is_file(), "training_started": False,
        "manifest_sha256": sha(args.manifest), "recovery_script_sha256": sha(__file__),
        "scope": "New copy reconstructed from checksum-verified archive members. Original frozen repair_committed, strict model, Adam, scheduler and Python/NumPy/CPU RNG consumers executed on CPU. CUDA RNG byte states preserved and hashed, not executed. COMPLETE marker retained: this does not authorize or launch extra training."}
    with Path(args.report).open("x", encoding="utf8") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
