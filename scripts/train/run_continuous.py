"""Stage-06 single-worker command surface; campaign orchestration is separate."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.continuous.data import load_cached_split, prepare_cache, sha256
from scripts.continuous.protocol import atomic_json, canonical_hash
from scripts.continuous.trainer import evaluate, fit, restore_model


def guarded_output(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / "outputs/stage06_continuous").resolve()):
        raise ValueError("New outputs must stay under stage06_continuous")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="Verify accepted JSONL and prepare a new CPU tensor cache")
    cache.add_argument("--data", required=True)
    cache.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train a task; never load test data")
    train.add_argument("--task", required=True)
    train.add_argument("--cache", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--max-steps", type=int)
    for name in ("evaluate", "export", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cuda:0")
        if name in {"evaluate", "benchmark"}:
            command.add_argument("--cache", required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=["validation", "test"], default="validation")
            command.add_argument("--selection-lock")
    args = parser.parse_args(argv)
    output = guarded_output(args.output)
    if args.command == "cache":
        result = prepare_cache(args.data, output)
        print(json.dumps({"status": "CACHE_COMPLETE", "splits": list(result["splits"])}))
        return
    if args.command == "train":
        task = json.loads(Path(args.task).read_text(encoding="utf-8"))
        config = task["config"]
        if config["n_items"] != 12101 or config["max_length"] != 50:
            raise ValueError("Production dataset protocol changed")
        if canonical_hash(config) != task["config_sha256"]:
            raise ValueError("Task config hash mismatch")
        if args.max_steps is not None and (not args.smoke or args.max_steps < 1):
            raise ValueError("Step limit requires a positive explicit smoke")
        if args.smoke and args.max_steps is None:
            raise ValueError("Every GPU smoke needs an explicit step bound")
        train_data = load_cached_split(args.cache, task["protocol"], "train")
        validation = load_cached_split(args.cache, task["protocol"], "validation")
        result = fit(config, task["seed"], task["protocol"], train_data, validation,
                     output, device=args.device,
                     precision=config["precision"] if args.device.startswith("cuda") else "float32",
                     resume=args.resume, smoke=args.smoke, max_steps=args.max_steps, root=ROOT)
    else:
        if output.exists():
            raise FileExistsError(output)
        model, state = restore_model(args.checkpoint, args.device)
        config, protocol = state["config"], state["binding"]["protocol"]
        binding = {"checkpoint_sha256": sha256(args.checkpoint), "binding": state["binding"]}
        if args.command == "evaluate":
            if args.split == "test":
                if not args.selection_lock:
                    raise ValueError("Test metrics require frozen validation-only selection")
                from scripts.continuous.delivery import validate_test_unlock
                validate_test_unlock(args.selection_lock, state, args.checkpoint)
            output.mkdir(parents=True)
            data = load_cached_split(args.cache, protocol, args.split)
            result = evaluate(model, data, config["eval_batch_size"], config["precision"], predictions=True)
            rows = [{"user_id": int(user), "target_position": int(position), "target": int(target),
                     "rank": rank, "top_items": top, "nll": row_nll}
                    for user, position, target, rank, top, row_nll in zip(data["users"], data["positions"],
                       data["targets"], result.pop("ranks"), result.pop("top_items"), result.pop("nlls"))]
            with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            result.update(binding)
            result["prediction_sha256"] = sha256(output / "predictions.jsonl")
            result["split"] = args.split
            result["evaluation_device"] = args.device
            result["evaluation_precision"] = config["precision"] if args.device.startswith("cuda") else "float32"
        elif args.command == "benchmark":
            output.mkdir(parents=True)
            from scripts.continuous.benchmark import benchmark
            validation = load_cached_split(args.cache, protocol, "validation")
            result = {**benchmark(model, validation, config["precision"]), **binding}
        else:
            import numpy as np
            import torch
            output.mkdir(parents=True)
            vectors = model.item_embedding.weight.detach().float().cpu().numpy()
            np.save(output / "input_item_embeddings_with_pad.npy", vectors, allow_pickle=False)
            np.save(output / "output_item_embeddings.npy", vectors[1:], allow_pickle=False)
            with (output / "recommender.pt").open("xb") as stream:
                torch.save({"model_config": model.config.to_dict(), "model": model.state_dict(),
                            "training_binding": state["binding"],
                            "training_checkpoint_sha256": binding["checkpoint_sha256"]}, stream)
            result = {**binding, "config": config, "input_output_tied": True,
                      "output_item_ids": [1, 12101], "pad_output_included": False,
                      "recommender_checkpoint": str(Path(args.checkpoint).resolve()),
                      "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "measurements"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
