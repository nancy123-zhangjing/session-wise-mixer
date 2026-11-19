"""Explicit RQCD entry points. Nothing starts without a subcommand and paths."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.listwise_diffgrm.assets import dataset_binding, catalog_original_ids
from .common import (DEFAULT_CONFIG, ROOT, load_config, configure_runtime, seed_all,
                     derive_seed, optimizer, learning_rate, atomic_save, cpu_copy,
                     rng_state, restore_rng, sha256, source_binding, write_json,
                     output_lock, preserve_rng)
from .tokenizer import RQVAE, encode_catalog, health, latent_scaling
from .catalog import Catalog
from .data import PackedSplit, bind_inputs
from .training import Trainer
from .evaluation import evaluate, probe


def read_state(path):
    # Only load trusted locally produced experiment state, never untrusted uploads.
    return torch.load(path, map_location="cpu", weights_only=False)


def append_log(root, row):
    with (Path(root)/"events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def fit_tokenizer(args, cfg):
    device = torch.device(args.device)
    runtime = configure_runtime(cfg, device)
    binding = bind_inputs(args.dataset, args.content, cfg)
    binding.update(source=source_binding(), runtime=runtime)
    seed_all(cfg["seed"])
    content_array = np.load(args.content, allow_pickle=False)
    if (content_array.shape != (cfg["catalog_size"], cfg["rq"]["input_dim"]) or
            not np.isfinite(content_array).all() or
            not np.allclose(np.linalg.norm(content_array, axis=1), 1, atol=1e-4)):
        raise ValueError("Invalid normalized PCA content")
    content = torch.tensor(content_array, device=device, dtype=torch.float32)
    c = cfg["rq"]
    permutation = np.random.default_rng(cfg["seed"]).permutation(len(content))
    n_val = round(len(content)*c["val_fraction"])
    validation = np.sort(permutation[:n_val])
    train_indices = np.sort(permutation[n_val:])
    choices = np.random.default_rng(cfg["seed"])
    model = RQVAE(cfg).to(device)
    opt = optimizer(model.parameters(), cfg, device, rq=True)
    init_rng = torch.Generator(device=device).manual_seed(derive_seed(cfg["seed"], "kmeans"))
    update, best_loss, best_update = 0, None, None
    with output_lock(args.output, args.resume) as root:
        if args.resume:
            state = read_state(root/"last.pt")
            if state["config"] != cfg or state["binding"] != binding or state["kind"] != "rq":
                raise ValueError("RQ resume identity mismatch")
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            choices.bit_generator.state = state["choices_rng"]
            init_rng.set_state(state["init_rng"].cpu())
            restore_rng(state["rng"])
            update, best_loss, best_update = state["update"], state["best_loss"], state["best_update"]
            for parameter in state["optimizer"]["state"].values():
                if float(parameter["step"]) != update:
                    raise ValueError("RQ optimizer clock mismatch")
        else:
            write_json(root/"binding.json", binding)
            write_json(root/"config.json", cfg)
            write_json(root/"content_split.json", {"training": train_indices.tolist(), "validation": validation.tolist()})
        started = time.perf_counter()
        while update < c["max_updates"]:
            chosen = choices.choice(train_indices, size=c["batch_size"], replace=False)
            batch = content[torch.tensor(chosen, device=device)]
            if not bool(model.initialized):
                iterations = model.initialize(batch, init_rng)
                atomic_save(root/"initial.pt", {"model": cpu_copy(model.state_dict()), "iterations": iterations})
            model.train()
            lr = learning_rate(update+1, c["max_updates"], c["warmup_updates"], c["lr"])
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss = model(batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=True)
            opt.step()
            update += 1
            if update % 100 == 0:
                append_log(root, {"stage": "rq", "update": update, "loss": float(loss.detach()), "lr": lr})
            if update % c["eval_interval"] and update != c["max_updates"]:
                continue
            with torch.inference_mode():
                model.eval()
                total = torch.zeros(3, device=device, dtype=torch.float64)
                for indices in torch.tensor(validation, device=device).split(c["batch_size"]):
                    result = model(content[indices])
                    total += torch.stack([result[k] for k in
                                          ("loss", "reconstruction_loss", "quantization_loss")]).double()*len(indices)
                metrics = (total/len(validation)).cpu().tolist()
                encoded = encode_catalog(model, content, c["batch_size"])
                h = health(encoded["sid"], c["codes"])
            if not all(math.isfinite(v) for v in metrics) or not torch.isfinite(encoded["z"]).all():
                raise RuntimeError("Nonfinite RQ validation")
            healthy = h["min_utilization"] >= c["min_utilization"] and h["unique_ratio"] >= c["min_unique_ratio"]
            improved = healthy and (best_loss is None or metrics[0] < best_loss)
            if improved:
                best_loss, best_update = metrics[0], update
            state = cpu_copy({"kind": "rq", "config": cfg, "binding": binding, "model": model.state_dict(),
                              "optimizer": opt.state_dict(), "update": update, "best_loss": best_loss,
                              "best_update": best_update, "rng": rng_state(), "choices_rng": choices.bit_generator.state,
                              "init_rng": init_rng.get_state(), "scheduler": {"update": update, "total": c["max_updates"]}})
            atomic_save(root/"last.pt", state)
            if improved:
                atomic_save(root/"best.pt", state)
            append_log(root, {"stage": "rq_validation", "update": update, "losses": metrics,
                              "health": h, "healthy": healthy, "best_update": best_update,
                              "seconds": time.perf_counter()-started})
        if best_update is None:
            raise RuntimeError("No healthy RQ checkpoint; downstream training prohibited")
        model.load_state_dict(read_state(root/"best.pt")["model"])
        encoded = encode_catalog(model, content, c["batch_size"])
        data = PackedSplit(args.dataset, "train", cfg, device="cpu")
        mean, scale = latent_scaling(encoded["z"], data.target)
        asset = {**encoded, "books": model.books.detach().cpu(), "mean": mean, "scale": scale,
                 "original_ids": torch.from_numpy(catalog_original_ids(Path(args.dataset))),
                 "config": cfg, "binding": binding, "rq_update": best_update,
                 "rq_checkpoint_sha256": sha256(root/"best.pt")}
        # The complete RQ encoder/decoder stays in best.pt, only frozen numeric
        # tables are needed by the recommender GPU.
        atomic_save(root/"catalog.pt", asset)
        write_json(root/"COMPLETE.json", {"stage": "rq", "updates": update, "best_update": best_update,
                                        "catalog_sha256": sha256(root/"catalog.pt")})


def load_catalog(args, cfg, device):
    asset = read_state(args.catalog)
    if asset["config"] != cfg:
        raise ValueError("Asset was built under a different protocol/config")
    actual = dataset_binding(Path(args.dataset), verify=True)
    if actual != asset["binding"]["dataset"]:
        raise ValueError("Dataset differs from frozen RQ assets")
    catalog = Catalog(asset).to(device)
    return catalog, {"asset_sha256": sha256(args.catalog), "dataset": actual,
                     "source": source_binding(), "config": cfg}


def load_probe(path, data, binding):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value["count"] != len(value["rows"]) or
            value["population"] != len(data) or
            value["validation_sha256"] != binding["dataset"]["files"]["validation.parquet"]["sha256"]):
        raise ValueError("Probe manifest does not match validation")
    indices = data.subset_indices([r["sample_id"] for r in value["rows"]])
    if indices.tolist() != [r["index"] for r in value["rows"]]:
        raise ValueError("Probe row indices changed")
    return indices


def train_recommender(args, cfg):
    device = torch.device(args.device)
    runtime = configure_runtime(cfg, device)
    seed_all(cfg["seed"])
    catalog, binding = load_catalog(args, cfg, device)
    binding["runtime"] = runtime
    placement = device if cfg["runtime"]["data_device"] == "cuda" and device.type == "cuda" else "cpu"
    train = PackedSplit(args.dataset, "train", cfg, placement)
    validation = PackedSplit(args.dataset, "validation", cfg, placement)
    indices = load_probe(args.probe_manifest, validation, binding)
    binding["probe_sha256"] = sha256(args.probe_manifest)
    trainer = Trainer(cfg, catalog, len(train), binding)
    t = cfg["training"]
    with output_lock(args.output, args.resume) as root:
        if args.resume:
            trainer.restore(read_state(root/"last.pt"))
        else:
            write_json(root/"binding.json", binding)
            write_json(root/"config.json", cfg)
            atomic_save(root/"initial.pt", trainer.state())
            atomic_save(root/"last.pt", trainer.state())
        for epoch in range(1, t["epochs"]+1):
            if epoch in trainer.completed_epochs:
                continue
            g = torch.Generator(device="cpu").manual_seed(derive_seed(cfg["protocol_id"], cfg["seed"], epoch))
            order = torch.randperm(len(train), generator=g)
            completed_groups = trainer.updates-(epoch-1)*trainer.groups
            if not 0 <= completed_groups <= trainer.groups:
                raise RuntimeError("Invalid resume epoch cursor")
            started = time.perf_counter()
            for group in range(completed_groups, trainer.groups):
                selected = order[group*t["effective_batch"]:(group+1)*t["effective_batch"]]
                history, target = train.batch(selected, device)
                result = trainer.update(history, target)
                if trainer.updates % t["checkpoint_interval"] == 0:
                    append_log(root, {"stage": "train", "epoch": epoch, "updates": trainer.updates,
                                      "lr": result["lr"], "grad_norm": float(result["grad_norm"]),
                                      **trainer.summary()})
                    atomic_save(root/"last.pt", trainer.state())
            # Save the epoch boundary before expensive evaluation. On resume
            # completed_groups == groups means evaluation is still pending.
            catalog.verify_frozen()
            atomic_save(root/"last.pt", trainer.state())
            probe_path = root/f"probe_R{epoch:02d}.json"
            if not probe_path.exists():
                write_json(probe_path, probe(trainer.model, trainer.diffusion, catalog, validation, cfg, indices))
            improved = False
            if epoch in t["validation_epochs"]:
                val_dir = root/f"validation/R{epoch:02d}"
                if (val_dir/"summary.json").exists():
                    if not (val_dir/"predictions.jsonl.gz").is_file():
                        raise ValueError("Evaluation receipt lacks predictions")
                    metrics = json.loads((val_dir/"summary.json").read_text(encoding="utf-8"))
                else:
                    metrics = evaluate(trainer.model, trainer.diffusion, catalog, validation, cfg, val_dir)
                score = metrics["metrics"][f"ndcg@{cfg['list_length']}"]
                if score > trainer.best_score:
                    trainer.best_score, trainer.best_epoch = score, epoch
                    improved = True
            append_log(root, {"stage": "epoch_end", "epoch": epoch, "updates": trainer.updates,
                              "seconds": time.perf_counter()-started, **trainer.summary(reset=True)})
            trainer.completed_epochs.append(epoch)
            state = trainer.state()
            atomic_save(root/f"R{epoch:02d}.pt", state)
            if improved:
                atomic_save(root/"best.pt", state)
            atomic_save(root/"last.pt", state)
        if trainer.best_epoch is None:
            raise RuntimeError("No full-validation best checkpoint")
        receipt = {"format": "rqcd_train_complete_v1", "best_epoch": trainer.best_epoch,
                   "best_validation_ndcg": trainer.best_score, "updates": trainer.updates,
                   "best_sha256": sha256(root/"best.pt"), "last_sha256": sha256(root/"last.pt"),
                   "catalog_sha256": sha256(args.catalog), "config": cfg,
                   "source": binding["source"]}
        write_json(root/"COMPLETE.json", receipt)


def test_recommender(args, cfg):
    """Only evaluate a sealed completed run, never arbitrary test checkpoints."""
    device = torch.device(args.device)
    configure_runtime(cfg, device)
    run = Path(args.run)
    receipt = json.loads((run/"COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["config"] != cfg or receipt["source"] != source_binding():
        raise ValueError("Completed training source/config mismatch")
    catalog, binding = load_catalog(args, cfg, device)
    if sha256(args.catalog) != receipt["catalog_sha256"]:
        raise ValueError("Frozen catalog changed")
    data = PackedSplit(args.dataset, "test", cfg, device="cpu")
    from .model import ContinuousListModel, Diffusion
    model, diffusion = ContinuousListModel(cfg).to(device), Diffusion(cfg).to(device)
    with output_lock(args.output, args.resume) as root:
        test_binding = {"training_receipt_sha256": sha256(run/"COMPLETE.json"),
                        "catalog_sha256": sha256(args.catalog), "config": cfg,
                        "source": source_binding()}
        if args.resume:
            if json.loads((root/"binding.json").read_text(encoding="utf-8")) != test_binding:
                raise ValueError("Test resume does not match sealed run")
        else:
            write_json(root/"binding.json", test_binding)
        outputs = {}
        for role, expected in (("best", receipt["best_sha256"]), ("last", receipt["last_sha256"])):
            checkpoint = run/f"{role}.pt"
            if sha256(checkpoint) != expected:
                raise ValueError("Sealed checkpoint changed")
            if role == "last" and receipt["best_epoch"] == cfg["training"]["epochs"]:
                outputs[role] = {"same_as": "best"}
                continue
            state = read_state(checkpoint)
            if state["binding"]["asset_sha256"] != binding["asset_sha256"]:
                raise ValueError("Checkpoint asset mismatch")
            model.load_state_dict(state["model"])
            diffusion.load_state_dict(state["diffusion"])
            out = root/role
            if (out/"summary.json").exists():
                outputs[role] = json.loads((out/"summary.json").read_text(encoding="utf-8"))
            else:
                outputs[role] = evaluate(model, diffusion, catalog, data, cfg, out)
        write_json(root/"COMPLETE.json", {"training_receipt_sha256": sha256(run/"COMPLETE.json"),
                                        "catalog_sha256": sha256(args.catalog), "roles": outputs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Print config only; no asset loading/training")
    for name in ("fit-tokenizer", "train", "test"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--device", required=True, help="Explicit cpu or cuda:0; never allocate platform resources")
        s.add_argument("--resume", action="store_true")
        if name == "fit-tokenizer":
            s.add_argument("--content", type=Path, required=True)
        else:
            s.add_argument("--catalog", type=Path, required=True)
        if name == "train":
            s.add_argument("--probe-manifest", type=Path, default=ROOT/"experiments/09_kuairec_listwise/h1_recovery_20261028/validation_subset_8192.json")
        if name == "test":
            s.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.command == "inspect":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    {"fit-tokenizer": fit_tokenizer, "train": train_recommender, "test": test_recommender}[args.command](args, cfg)


if __name__ == "__main__":
    main()
