"""Explicit local entry points. No scheduler, remote connection or auto-launch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .config import ARMS, DEFAULT_CONFIG, RUNTIME_TEMPLATE, design_path, load_config, load_runtime, total_t_steps


def plan(cfg):
    count = math.ceil(cfg["split_counts"]["train"]/cfg["training"]["effective_batch"])
    return {"protocol": cfg["protocol_id"], "arms": ARMS, "seed": cfg["seed"],
            "target": cfg.get("target", {"mode": "content_v1"}),
            "teacher_required": "target" in cfg,
            "pretraining_item_epochs": cfg["rq"]["epochs"],
            "pretraining_updates": math.ceil(cfg["catalog_size"]/cfg["rq"]["batch_size"])*cfg["rq"]["epochs"],
            "R_epochs": cfg["training"]["epochs"], "T_updates": total_t_steps(cfg),
            "R_updates": count*cfg["training"]["epochs"],
            "validation": [{"R": r, "T": max(0, r-cfg["training"]["preheat_epochs"])*cfg["training"]["t_steps_per_cycle"]}
                           for r in cfg["evaluation"]["epochs"]],
            "training_started": False}


def _runtime(args, cfg, require_batches=True):
    from .common import configure_runtime
    runtime = load_runtime(args.runtime, cfg, require_batches=require_batches)
    environment = configure_runtime(runtime, args.device)
    return runtime, environment


def _prepared(args, cfg, *, allow_l5o_parent=False):
    from .common import read_state, sha256, source_binding
    from .data import bind_inputs
    root = Path(args.prepared)
    meta = json.loads((root/"PREPARED.json").read_text(encoding="utf-8"))
    source = source_binding()
    # First-wave RQ assets predate the joint landing-mask fix. Neither
    # changed file affects their initialization or RQ pretraining.
    compatible_sources = [source]
    if allow_l5o_parent and cfg.get('calibrated_teacher') is not None:
        compatible_sources.append({**source, **{
            'scripts/rqcd_mixer/run.py': 'd1e1f2d278bd0cd3a4a1e770739928e89996c8ac8126f7c4bf19666b21763066',
            'scripts/rqcd_mixer/losses.py': 'eaaa2697deaece12ee13d308ed0bb2300d898a24554b91cf8499e970ee758468',
        }})
    from .screen import ACCEPTED_PARENT_SOURCES, asset_view, source_digest
    # Screening switches are training-only; a recorded first-wave parent may be reused
    # only through the explicit, documented list in screen.py.
    accepted = (meta["source"] in compatible_sources or
                (allow_l5o_parent and source_digest(meta["source"]) in ACCEPTED_PARENT_SOURCES))
    if asset_view(meta["config"]) != asset_view(cfg) or not accepted:
        raise ValueError("Prepared bundle belongs to different design/code/configuration")
    if meta["initial_sha256"] != sha256(root/"initial.pt"):
        raise ValueError("Common initial-state integrity error")
    actual = bind_inputs(args.dataset, meta["content_path"], cfg)
    if actual != meta["data_binding"]:
        raise ValueError("Dataset/content changed since preparation")
    if "target" in cfg:
        from .targets import load_teacher
        from .data import catalog_original_ids
        if (meta["teacher_sha256"] != sha256(meta["teacher_path"]) or
                meta["target_inputs_sha256"] != sha256(root/"target_inputs.pt")):
            raise ValueError("Frozen teacher/target inputs changed since preparation")
        load_teacher(meta["teacher_path"], cfg, actual["dataset"], catalog_original_ids(Path(args.dataset)))
    return root, meta, read_state(root/"initial.pt")


def prepare(args, cfg):
    import torch
    from .common import (output_lock, atomic_save, write_json, source_binding, sha256,
                         seed_all, seed_for, tensor_digest)
    from .data import bind_inputs, load_content, PackedSplit, target_counts, PermutationStream, catalog_original_ids
    from .tokenizer import RQTokenizer, hard_quantize, fixed_statistics
    from .model import Recommender
    from .catalog import prepare_noise
    runtime, environment = _runtime(args, cfg, require_batches=False)
    if cfg["design_sha256"] != sha256(design_path(cfg)):
        raise ValueError("Approved design changed; update version explicitly before implementation/run")
    binding = bind_inputs(args.dataset, args.content, cfg)
    content = load_content(args.content, cfg, args.device)
    inputs = None
    if "target" in cfg:
        from .targets import load_teacher, target_inputs
        if getattr(args, "teacher", None) is None:
            raise ValueError("Both v2 targets require the same explicit --teacher items.pt")
        teacher = load_teacher(args.teacher, cfg, binding["dataset"], catalog_original_ids(Path(args.dataset)))
        inputs = target_inputs(content, teacher, cfg)
        content = inputs["features"]
    train = PackedSplit(args.dataset, "train", cfg, device="cpu")
    counts = target_counts(train, cfg["catalog_size"]).to(args.device)
    order = PermutationStream(len(content), seed_for(cfg["seed"], "pretrain_order"))
    first = content[order.order[:cfg["rq"]["batch_size"]].to(args.device)]
    seed_all(seed_for(cfg["seed"], "initial_RQ"))
    common = RQTokenizer(cfg, "none").to(args.device)
    common.initialize(first, torch.Generator(device=args.device).manual_seed(seed_for(cfg["seed"], "kmeans")))
    common_state = common.state_dict()
    states, mixer_state = {"B0": common_state}, None
    seed_all(seed_for(cfg["seed"], "initial_Mixer"))
    for arm in ("B1", "B2"):
        model = RQTokenizer(cfg, ARMS[arm]).to(args.device)
        model.load_state_dict(common_state, strict=False)
        if mixer_state is None:
            mixer_state = {k: v.detach().clone() for k, v in model.mixer.state_dict().items()}
        model.mixer.load_state_dict(mixer_state)
        states[arm] = model.state_dict()
    seed_all(seed_for(cfg["seed"], "initial_recommender"))
    rec = Recommender(cfg)
    with torch.no_grad():
        z = common.encoder(content)
        books = common.effective_books()
        asset = {"z": z, "books": books, **hard_quantize(z, books), **fixed_statistics(z, books, counts),
                 "original_ids": torch.from_numpy(catalog_original_ids(Path(args.dataset))).to(args.device), "version": 0}
    with output_lock(args.output) as root:
        if inputs is not None:
            atomic_save(root/"target_inputs.pt", inputs)
        atomic_save(root/"initial.pt", {"tokenizers": states, "recommender": rec.state_dict(),
                                       "benchmark_asset": asset, "config": cfg})
        noises = {}
        for split in ("validation", "test"):
            data = PackedSplit(args.dataset, split, cfg, device="cpu")
            noises[split] = prepare_noise(root/f"{split}_noise.npy", cfg, data.sample_ids, split)
        meta = {"config": cfg, "source": source_binding(), "data_binding": binding, "environment": environment,
                "content_path": str(Path(args.content).resolve()), "dataset_path": str(Path(args.dataset).resolve()),
                "initial_sha256": sha256(root/"initial.pt"), "noise": noises,
                "common_RQ_hash": tensor_digest(common_state), "common_recommender_hash": tensor_digest(rec.state_dict()),
                "common_Mixer_hash": tensor_digest(mixer_state)}
        if inputs is not None:
            meta.update(teacher_path=str(Path(args.teacher).resolve()), teacher_sha256=sha256(args.teacher),
                        target_inputs_sha256=sha256(root/"target_inputs.pt"),
                        train_unsupported_items=int((~inputs["supported"]).sum()),
                        cf_global_scale=float(inputs["cf_global_scale"]))
        write_json(root/"PREPARED.json", meta)
    return {"prepared": str(root), "training_started": False}


def pretrain(args, cfg):
    import torch
    from .common import (output_lock, atomic_save, write_json, read_state, sha256,
                         seed_all, seed_for, append_log)
    from .data import load_content, PackedSplit, target_counts, catalog_original_ids
    from .tokenizer import RQTokenizer, hard_quantize, fixed_statistics, code_health
    from .training import Pretrainer
    runtime, env = _runtime(args, cfg, require_batches=False)
    prepared, meta, initial = _prepared(args, cfg)
    content = load_content(meta["content_path"], cfg, args.device)
    inputs = None
    if "target" in cfg:
        inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                  for k, v in read_state(prepared/"target_inputs.pt").items()}
        content = inputs["features"]
    model = RQTokenizer(cfg, ARMS[args.arm]).to(args.device)
    model.load_state_dict(initial["tokenizers"][args.arm])
    binding = {"prepared_sha256": sha256(prepared/"PREPARED.json"), "arm": args.arm,
               "source": meta["source"], "data": meta["data_binding"], "environment": env}
    if "target" in cfg:
        binding.update(teacher_sha256=meta['teacher_sha256'], target_mode=cfg['target']['mode'])
    trainer = Pretrainer(cfg, runtime, model, content, binding, target_data=inputs)
    seed_all(seed_for(cfg["seed"], "pretrain_runtime"))
    with output_lock(args.output, args.resume) as root:
        if args.resume:
            trainer.restore(read_state(root/"last.pt"))
        else:
            write_json(root/"config.json", cfg)
            write_json(root/"binding.json", binding)
            atomic_save(root/"last.pt", trainer.state())
        clock = time.perf_counter()
        while trainer.epochs < cfg["rq"]["epochs"]:
            event = trainer.advance()
            if event["kind"] == "pretrain_epoch":
                if event["loss"] < trainer.best_loss:
                    trainer.best_loss, trainer.best_loss_epoch = event["loss"], trainer.epochs
                    # Diagnostic model only; resumable state remains last/best_collision.
                    atomic_save(root/"best_loss.pt", {"kind": "pretrain_loss_diagnostic", "epoch": trainer.epochs,
                                                     "loss": event["loss"], "tokenizer": model.state_dict(), "config": cfg})
                append_log(root, {**event, "wall_seconds": time.perf_counter()-clock})
                if trainer.epochs % cfg["rq"]["eval_interval"] == 0:
                    objective = trainer.evaluate_objective() if "target" in cfg else None
                    if objective is not None and objective["loss"] < trainer.best_objective:
                        trainer.best_objective, trainer.best_objective_epoch = objective["loss"], trainer.epochs
                        atomic_save(root/"best_objective.pt", trainer.state())
                    model.eval()
                    with torch.no_grad():
                        z = model.encoder(content)
                        books, scales = model.books_with_diagnostics()
                        health = code_health(hard_quantize(z, books)["sid"], cfg["rq"]["codes"])
                    if not torch.isfinite(z).all() or not torch.isfinite(books).all():
                        raise RuntimeError("Nonfinite pretraining model")
                    if health["collision_redundancy"] < trainer.best_collision:
                        trainer.best_collision, trainer.best_collision_epoch = health["collision_redundancy"], trainer.epochs
                        atomic_save(root/"best_collision.pt", trainer.state())
                    append_log(root, {"kind": "pretrain_validation", "epoch": trainer.epochs,
                                      "fixed_objective": objective,
                                      "encoding": {**health, **scales},
                                      "z_rms": float(z.square().mean().sqrt()),
                                      "z_variance_by_dim": z.var(0, unbiased=False).cpu().tolist(),
                                      "best_collision_epoch": trainer.best_collision_epoch})
                    atomic_save(root/"last.pt", trainer.state())
            if trainer.updates % runtime["checkpoint_every_updates"] == 0:
                atomic_save(root/"last.pt", trainer.state())
        atomic_save(root/"last.pt", trainer.state())
        selected_name = "best_objective.pt" if "target" in cfg else "best_collision.pt"
        selected = read_state(root/selected_name)
        model.load_state_dict(selected["tokenizer"])
        model.eval()
        train = PackedSplit(args.dataset, "train", cfg, device="cpu")
        with torch.no_grad():
            z, books = model.encoder(content), model.effective_books()
            hard = hard_quantize(z, books)
            statistics = fixed_statistics(z, books, target_counts(train, cfg["catalog_size"]).to(args.device))
        asset = {"z": z, "books": books, "sid": hard["sid"], "q": hard["q"], **statistics,
                 "original_ids": torch.from_numpy(catalog_original_ids(Path(args.dataset))), "version": 0,
                 "config": cfg, "arm": args.arm, "binding": binding,
                 "pretrain_checkpoint_sha256": sha256(root/selected_name), "selected_checkpoint": selected_name}
        atomic_save(root/"catalog.pt", asset)
        result = {"kind": "pretraining_complete", "arm": args.arm, "epochs": trainer.epochs,
                  "updates": trainer.updates, "best_collision_epoch": trainer.best_collision_epoch,
                  "best_loss_epoch": trainer.best_loss_epoch, "best_objective_epoch": trainer.best_objective_epoch,
                  "selected_checkpoint": selected_name, "catalog_sha256": sha256(root/"catalog.pt")}
        write_json(root/"COMPLETE.json", result)
    return result


def _joint(args, cfg, runtime, environment):
    from .common import read_state, sha256, source_binding
    from .tokenizer import RQTokenizer
    from .training import JointTrainer
    prepared, meta, initial = _prepared(args, cfg, allow_l5o_parent=True)
    rq_root = Path(args.pretrained)
    receipt = json.loads((rq_root/"COMPLETE.json").read_text(encoding="utf-8"))
    asset = read_state(rq_root/"catalog.pt")
    selected_name = "best_objective.pt" if "target" in cfg else "best_collision.pt"
    checkpoint = read_state(rq_root/selected_name)
    shared_cf_parent = cfg.get('calibrated_teacher') is not None and args.arm == 'B1'
    parent_arm = 'B0' if shared_cf_parent else args.arm
    from .screen import asset_view, generation_space, switches
    if (receipt["catalog_sha256"] != sha256(rq_root/"catalog.pt") or asset_view(asset["config"]) != asset_view(cfg) or
            asset["arm"] != parent_arm or asset_view(checkpoint["config"]) != asset_view(cfg) or
            asset["pretrain_checkpoint_sha256"] != sha256(rq_root/selected_name) or
            asset["binding"]["prepared_sha256"] != sha256(prepared/"PREPARED.json")):
        raise ValueError("Pretrained artifact identity mismatch")
    tokenizer = RQTokenizer(cfg, ARMS[args.arm]).to(args.device)
    if shared_cf_parent:
        # The paired arms share the same pretrained plain RQ. The B1 Mixer
        # starts from its common initialization with a zero output projection.
        mixer = {key: value for key, value in initial['tokenizers']['B1'].items() if key.startswith('mixer.')}
        tokenizer.load_state_dict({**checkpoint['tokenizer'], **mixer})
    else:
        tokenizer.load_state_dict(checkpoint["tokenizer"])
    binding = {"prepared_sha256": sha256(prepared/"PREPARED.json"), "catalog_sha256": sha256(rq_root/"catalog.pt"),
               "arm": args.arm, "source": source_binding(), "data": meta["data_binding"], "environment": environment}
    if "target" in cfg:
        binding.update(teacher_sha256=meta['teacher_sha256'], target_mode=cfg['target']['mode'])
    if shared_cf_parent:
        binding['shared_pretrained_arm'] = parent_arm
    trainer = JointTrainer(cfg, runtime, tokenizer, asset, initial["recommender"], binding, args.device)
    if switches(cfg).get("history_head") or generation_space(cfg) == "teacher":
        if meta["target_inputs_sha256"] != sha256(prepared/"target_inputs.pt"):
            raise ValueError("Target inputs changed since preparation")
        trainer.attach_item_vectors(read_state(prepared/"target_inputs.pt")["features"])
    if generation_space(cfg) == "teacher":
        trainer.attach_latent_encoder(tokenizer.encoder)
    return trainer, prepared


def _validation_noise(cfg, prepared, root, data):
    """The prepared table, or (generation width != RQ latent) a run-local table built by the same rule."""
    from .catalog import prepare_noise
    from .screen import generation_dim
    if generation_dim(cfg) == cfg["rq"]["latent_dim"]:
        return prepared/"validation_noise.npy"
    path = root/"noise"/"validation_noise.npy"
    prepare_noise(path, cfg, data.sample_ids, "validation")
    return path


def train(args, cfg):
    import torch
    from .common import (output_lock, atomic_save, write_json, read_state, sha256,
                         seed_all, seed_for, append_log)
    from .data import PackedSplit, target_counts
    from .evaluation import evaluate
    runtime, environment = _runtime(args, cfg)
    trainer, prepared = _joint(args, cfg, runtime, environment)
    data_device = args.device if runtime["data_device"] == "cuda" else "cpu"
    training = PackedSplit(args.dataset, "train", cfg, device=data_device)
    validation = PackedSplit(args.dataset, "validation", cfg, device=data_device)
    stop_after = getattr(args, "stop_after", None)
    if stop_after is not None and not 1 <= stop_after <= cfg["training"]["epochs"]:
        raise ValueError("--stop-after must be an R epoch of the schedule")
    seed_all(seed_for(cfg["seed"], "recommender_dropout"))
    with output_lock(args.output, args.resume) as root:
        validation_noise = _validation_noise(cfg, prepared, root, validation)
        if args.resume:
            trainer.restore(read_state(root/"last.pt"))
        else:
            trainer.configure_generation(target_counts(training, cfg["catalog_size"]))
            write_json(root/"config.json", cfg)
            write_json(root/"runtime.json", runtime)
            write_json(root/"binding.json", trainer.binding)
            write_json(root/"artifact_paths.json", {"pretrained": str(Path(args.pretrained).resolve()),
                                                    "prepared": str(prepared.resolve()),
                                                    "dataset": str(Path(args.dataset).resolve())})
            atomic_save(root/"last.pt", trainer.state())
        clock = time.perf_counter()
        phase_clock = clock
        while True:
            epoch = trainer.r_epoch
            if epoch in cfg["evaluation"]["epochs"] and epoch not in trainer.validated_epochs:
                checkpoint = root/f"checkpoints/R{epoch:02d}_T{trainer.t_updates:03d}.pt"
                if not checkpoint.exists():
                    atomic_save(checkpoint, trainer.state())
                else:
                    # An interrupted validation may already have its immutable
                    # node. Evaluate that exact saved state, not an assumption
                    # that a replay from last.pt produced identical weights.
                    trainer.restore(read_state(checkpoint))
                    if trainer.r_epoch != epoch:
                        raise ValueError("Immutable validation checkpoint has the wrong epoch")
                identity = {"checkpoint_sha256": sha256(checkpoint), "binding": trainer.binding,
                            "split": "validation", "R": epoch, "T": trainer.t_updates}
                result = evaluate(trainer.model, trainer.diffusion, trainer.catalog, validation, cfg, runtime,
                                  validation_noise, root/f"validation/R{epoch:02d}", identity)
                metric = 'ev_ndcg@5' if hasattr(validation, 'watch_ratios') else 'ndcg@5'
                score = result['metrics'][metric]
                if score > trainer.best_score:
                    trainer.best_score, trainer.best_epoch = score, epoch
                    write_json(root/"BEST.json", {"checkpoint": str(checkpoint.relative_to(root)),
                                                 "sha256": identity["checkpoint_sha256"], "R": epoch,
                                                 "T": trainer.t_updates, "validation_ndcg@5": score})
                trainer.validated_epochs.append(epoch)
                atomic_save(root/"last.pt", trainer.state())
                append_log(root, {"kind": "validation", "R": epoch, "T": trainer.t_updates,
                                  "metrics": result["metrics"], "best_R": trainer.best_epoch,
                                  "seconds": result["seconds"]})
                phase_clock = time.perf_counter()
            if stop_after is not None and trainer.r_epoch >= stop_after and epoch in trainer.validated_epochs:
                # Screening budget: the schedule (learning rate, T clock) stays that of the
                # full run; training simply ends after this validated R epoch.
                atomic_save(root/"last.pt", trainer.state())
                result = {"kind": "stopped_after", "R": trainer.r_epoch, "T": trainer.t_updates,
                          "R_updates": trainer.r_updates, "best_R": trainer.best_epoch}
                write_json(root/"STOPPED.json", result)
                return result
            if trainer.phase == "done":
                break
            event = trainer.advance(training)
            if event["kind"] != "update":
                append_log(root, {**event, "phase_wall_seconds": time.perf_counter()-phase_clock,
                                  "attempt_wall_seconds": time.perf_counter()-clock})
                atomic_save(root/"last.pt", trainer.state())
                phase_clock = time.perf_counter()
            elif (trainer.r_updates+trainer.t_updates) % runtime["checkpoint_every_updates"] == 0:
                atomic_save(root/"last.pt", trainer.state())
        final = root/f"checkpoints/R{trainer.r_epoch:02d}_T{trainer.t_updates:03d}.pt"
        result = {"kind": "joint_training_complete", "arm": args.arm, "R": trainer.r_epoch, "T": trainer.t_updates,
                  "R_updates": trainer.r_updates, "best_R": trainer.best_epoch,
                  "final_checkpoint": str(final.relative_to(root)), "final_sha256": sha256(final),
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.device(args.device).type == "cuda" else None}
        write_json(root/"TRAINING_COMPLETE.json", result)
    return result


def test(args, cfg):
    if "screen" in cfg:
        raise ValueError("Screening runs are validation-only; the test split is reserved for the formal runs")
    import torch
    from .common import read_state, sha256, write_json
    from .data import PackedSplit
    from .evaluation import evaluate
    runtime, environment = _runtime(args, cfg)
    trainer, prepared = _joint(args, cfg, runtime, environment)
    root = Path(args.run)
    done = json.loads((root/"TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    best = json.loads((root/"BEST.json").read_text(encoding="utf-8"))
    data_device = args.device if runtime["data_device"] == "cuda" else "cpu"
    data = PackedSplit(args.dataset, "test", cfg, device=data_device)
    results, evaluated = {}, {}
    nodes = [("best", root/best["checkpoint"], best["sha256"]),
             ("final", root/done["final_checkpoint"], done["final_sha256"])]
    if cfg["evaluation"].get("selection_rule") == "smoothed3":
        from scripts.kuairec_data.selection import smoothed_choice
        metric = 'ev_ndcg@5' if hasattr(data, 'watch_ratios') else 'ndcg@5'
        points = []
        for epoch in cfg["evaluation"]["epochs"]:
            summary = json.loads((root/f"validation/R{epoch:02d}/summary.json").read_text(encoding="utf-8"))
            [checkpoint] = sorted(root.glob(f"checkpoints/R{epoch:02d}_T*.pt"))
            points.append(((epoch, checkpoint, summary["identity"]["checkpoint_sha256"]), summary["metrics"][metric]))
        (epoch, checkpoint, digest), smoothed, raw, _ = smoothed_choice(points)
        nodes.append(("smoothed", checkpoint, digest))
        write_json(root/"SMOOTHED_SELECTION.json", {"R": epoch, "checkpoint": str(checkpoint.relative_to(root)),
                                                    "sha256": digest, "smoothed_validation": smoothed,
                                                    "raw_validation": raw})
    for name, path, expected in nodes:
        actual = sha256(path)
        if actual != expected:
            raise ValueError("Selected checkpoint checksum mismatch")
        if actual in evaluated:
            results[name] = {**evaluated[actual], "reused_same_checkpoint": True}
            continue
        state = read_state(path)
        trainer.restore(state)
        identity = {"checkpoint_sha256": actual, "binding": trainer.binding, "split": "test",
                    "R": trainer.r_epoch, "T": trainer.t_updates}
        result = evaluate(trainer.model, trainer.diffusion, trainer.catalog, data, cfg, runtime,
                          prepared/"test_noise.npy", root/f"test/{name}", identity)
        results[name] = evaluated[actual] = result
    write_json(root/"TEST_COMPLETE.json", results)
    return {k: v["metrics"] for k, v in results.items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--l5o-calibration', type=Path, help='Completed SASRec calibration receipt for the L5O wave')
    p.add_argument('--resolved-config', type=Path, help='Protocol v2: a complete per-seed config written by l5o_v2')
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Print fixed budgets only; no torch/model/resource execution")
    for name in ("prepare", "pretrain", "train", "test", "benchmark"):
        s = sub.add_parser(name)
        s.add_argument("--runtime", type=Path, default=RUNTIME_TEMPLATE)
        s.add_argument("--device", default="cuda")
        s.add_argument("--dataset", type=Path, required=True)
        if name == "prepare":
            s.add_argument("--content", type=Path, required=True)
            s.add_argument("--teacher", type=Path, help="Shared frozen teacher items.pt; required by v2")
            s.add_argument("--output", type=Path, required=True)
        else:
            s.add_argument("--prepared", type=Path, required=True)
        if name in ("pretrain", "train", "test"):
            s.add_argument("--arm", choices=ARMS, required=True)
        if name in ("pretrain", "train"):
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--resume", action="store_true")
        if name == "train":
            s.add_argument("--stop-after", type=int, help="screening: end after this validated R epoch")
        if name in ("train", "test"):
            s.add_argument("--pretrained", type=Path, required=True)
        if name == "test":
            s.add_argument("--run", type=Path, required=True)
        if name == "benchmark":
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--microbatches", type=int, nargs="+", default=[64, 128, 256])
            s.add_argument("--repeats", type=int, default=3)
    report = sub.add_parser("report")
    report.add_argument("--runs", nargs="+", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        cfg = load_config(args.config)
        if args.resolved_config:
            if args.l5o_calibration:
                raise ValueError('--resolved-config already contains the calibrated L5O settings')
            resolved = json.loads(args.resolved_config.read_text(encoding='utf-8'))
            if (('protocol_v2' not in resolved and 'screen' not in resolved) or
                    resolved.get('target', {}).get('mode') != 'cf'):
                raise ValueError('Resolved config must be a protocol v2 or screening CF config')
            from .screen import validate
            cfg = validate(resolved)
        elif args.l5o_calibration:
            if cfg.get('target', {}).get('mode') != 'cf':
                p.error('The L5O wave uses the CF target preset')
            from scripts.kuairec_data.l5o_launch import wave_config
            cfg = wave_config(cfg, args.l5o_calibration, 'cf')
        if args.command == "plan":
            result = plan(cfg)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(args, cfg)
        elif args.command == "report":
            from .report import report
            result = report(args.runs, args.output)
        else:
            result = globals()[args.command](args, cfg)
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        root = getattr(args, "output", getattr(args, "run", None))
        if root is not None and Path(root).is_dir():
            from .common import write_json
            write_json(Path(root)/"ERROR.json", {"command": args.command, "type": type(error).__name__,
                                                "message": str(error), "traceback": traceback.format_exc(),
                                                "resume": "Use the last complete checkpoint; no partial update was promoted"})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
