"""Four-arm stage05 long factorial with paired RNG, real noise and boundary recovery.

Does not submit jobs. Uses the existing model, OCN objective, phase loop, and
evaluator; implements the new explicitly paired long-study orchestration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/checks"))
import diger_mixer_long_protocol as protocol_lib

need, sha, read, write = protocol_lib.need, protocol_lib.sha, protocol_lib.read, protocol_lib.write
ARMS = protocol_lib.ARMS


def module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


def utc():
    return datetime.now(timezone.utc).isoformat()


def summary(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        item = value.detach().cpu().contiguous()
        return dict(dtype=str(item.dtype), shape=list(item.shape),
                    sha256=hashlib.sha256(item.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(key): summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summary(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(summary(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def configure(model, arm):
    enabled = arm.startswith("mixer_")
    model.joint_alignment.mixer.mixer_enabled = enabled
    model.joint_alignment.tokenizer_variant = "cross_pse" if enabled else "refined_pse"
    model.config["crosspse_mixer_enabled"] = enabled
    model.joint_alignment.set_recommender_cache(False)
    model.sync_joint_tokenizer_mapping()


def scheduler_factor(step, protocol):
    if step < protocol["decoder_warmup_updates"]:
        return max(step, 1) / protocol["decoder_warmup_updates"]
    progress = min(max((step - protocol["decoder_warmup_updates"]) /
                      (protocol["decoder_total_updates"] - protocol["decoder_warmup_updates"]), 0), 1)
    return .5 * (1 + math.cos(math.pi * progress))


def optimizers(model, arm, protocol):
    import torch
    controls = dict(betas=tuple(protocol["adam_betas"]), eps=protocol["adam_eps"])
    recommender = torch.optim.AdamW(list(model.joint_recommender_parameters()),
                                   lr=protocol["decoder_lr"], weight_decay=0, **controls)
    groups = []
    if arm.endswith("_joint"):
        for name, parameters in model.joint_tokenizer_parameter_groups():
            if parameters:
                groups.append(dict(params=parameters, group_name=name,
                                   lr=protocol["codebook_lr"] if name == "joint_codebook" else protocol["mixer_lr"],
                                   weight_decay=protocol["tokenizer_weight_decay"]))
    tokenizer = torch.optim.AdamW(groups, **controls) if groups else None
    schedule = torch.optim.lr_scheduler.LambdaLR(recommender, lambda step: scheduler_factor(step, protocol))
    return {"recommender": recommender, "tokenizer": tokenizer}, schedule


def restore_rng(saved, device):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"])
    if device.type == "cuda" and saved["torch_cuda"] is not None:
        torch.cuda.set_rng_state(saved["torch_cuda"], device)


def payload(context, arm, optim, schedule, progress):
    runtime, model = context["runtime"], context["pipeline"].model
    return dict(format="stage05_long_phase_boundary_v1", protocol_id=protocol_lib.PROTOCOL,
                arm=arm, mode=context["mode"], identity=context["identity"], model=runtime.cpu_state(model),
                E0_array_sha256=context["protocol"]["canonical_E0_array_sha256"],
                optimizers={key: None if value is None else value.state_dict() for key, value in optim.items()},
                scheduler=schedule.state_dict(), rng=runtime.phase_rng_state(context["device"]), progress=copy.deepcopy(progress))


def restore(context, arm, state, optim, schedule):
    model = context["pipeline"].model
    need(state["format"] == "stage05_long_phase_boundary_v1" and state["arm"] == arm
         and state["identity"] == context["identity"] and state["mode"] == context["mode"],
         "Checkpoint source/input/config/arm identity differs")
    need(state["E0_array_sha256"] == context["protocol"]["canonical_E0_array_sha256"], "Wrong nonpersistent anchor")
    model.load_state_dict(state["model"], strict=True)
    configure(model, arm)
    for key, optimizer in optim.items():
        if optimizer is not None:
            need(state["optimizers"][key] is not None, "Missing complete optimizer")
            # On CPU, AdamW may reuse input state tensors rather than copying.
            # Replaying one boundary must never advance the saved payload itself.
            optimizer.load_state_dict(copy.deepcopy(state["optimizers"][key]))
    schedule.load_state_dict(copy.deepcopy(state["scheduler"]))
    restore_rng(state["rng"], context["device"])
    need(context["runtime"].tensor_state_hash(model.state_dict()) == context["runtime"].tensor_state_hash(state["model"]),
         "Strict checkpoint reload differs")
    need(digest(schedule.state_dict()) == digest(state["scheduler"]), "Scheduler restore differs")
    for key, optimizer in optim.items():
        if optimizer is not None:
            need(digest(optimizer.state_dict()) == digest(state["optimizers"][key]), "Optimizer restore differs")
    return copy.deepcopy(state["progress"])


class PhaseAudit:
    """Audit a real accumulated update; retain the exact production phase loop."""
    def __init__(self, context, optimizer, phase, schedule):
        self.context, self.optimizer, self.phase, self.schedule = context, optimizer, phase, schedule
        self.steps, self.evidence = 0, None

    def __enter__(self):
        import torch
        self.original = self.optimizer.step
        model = self.context["pipeline"].model
        def step(*args, **kwargs):
            if time.time() > self.context["deadline"] - 30:
                raise TimeoutError("Own process deadline; keep the last committed boundary")
            if self.steps == 0:
                active = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
                expected = {name: parameter for name, parameter in model.named_parameters()
                            if id(parameter) in active and not name.startswith("joint_sequence_adapter.")}
                need(expected, "No expected phase parameters")
                missing = [name for name, parameter in expected.items() if parameter.grad is None]
                need(not missing, "Expected parameters have no gradient: " + str(missing))
                need(all(bool(torch.isfinite(parameter.grad).all()) for parameter in expected.values()),
                     "Nonfinite per-parameter gradient")
                before = {name: parameter.detach().clone() for name, parameter in expected.items()}
            result = self.original(*args, **kwargs)
            if self.phase == "recommender":
                self.schedule.step()
            if self.steps == 0:
                changed = [name for name, parameter in expected.items() if not torch.equal(before[name], parameter.detach())]
                need(changed, "Optimizer did not change any expected parameter")
                need(all(parameter in self.optimizer.state for parameter in expected.values()),
                     "Adam state missing for an active expected parameter")
                self.evidence = dict(expected_parameter_count=len(expected), all_expected_gradients_finite=True,
                                     all_expected_Adam_states_present=True, actual_changed_parameters=changed,
                                     phase=self.phase, audited_optimizer_update=1)
            self.steps += 1
            return result
        self.optimizer.step = step
        step._wrapped_by_lr_sched = getattr(self.original, "_wrapped_by_lr_sched", False)
        return self

    def __exit__(self, *_):
        self.optimizer.step = self.original


def execute_phase(context, arm, row, optim, schedule):
    runtime, pipeline, protocol = context["runtime"], context["pipeline"], context["protocol"]
    model, kind = pipeline.model, row["phase"]
    before = runtime.cpu_state(model)
    inactive = {name: value for name, value in before.items()
                if (name.startswith("joint_alignment.") if kind == "recommender" else not name.startswith("joint_alignment."))}
    batch_size = protocol["smoke"]["train_batch_size"] if context["mode"] == "smoke" else protocol["train_batch_size"]
    started_lr = optim[kind].param_groups[0]["lr"]
    started_schedule = model.joint_alignment.diger_schedule_state()
    with PhaseAudit(context, optim[kind], kind, schedule) as audit:
        result = runtime.train_phase(pipeline, optim[kind], kind, row["batches"], row["logical_index"],
            protocol, context["device"], batch_size, protocol["gradient_accumulation_steps"],
            gumbel_seed=runtime.phase_seed(protocol["gumbel_seed"], "gumbel", row["logical_index"]) if kind == "tokenizer" else None,
            feedback_weight=1.0 if kind == "tokenizer" else None)
    need(result["optimizer_updates"] == row["optimizer_updates"] == audit.steps, "Actual phase update budget differs")
    after = runtime.cpu_state(model)
    need(runtime.tensor_state_hash(inactive) == runtime.tensor_state_hash({name: after[name] for name in inactive}),
         "Frozen side changed during phase")
    if arm.startswith("no_mixer"):
        names = [name for name in before if name.startswith("joint_alignment.mixer.")
                 and name != "joint_alignment.mixer.base_codebooks"]
        need(all(bool((before[name] == after[name]).all()) for name in names), "Disabled Mixer tensors changed")
    if kind == "tokenizer":
        model.refresh_joint_codes()
    current_codes = model.joint_alignment.hard_codes[1:].detach().cpu().numpy()
    result.update(parameter_audit=audit.evidence, frozen_side_unchanged=True,
                  lr_start=started_lr, lr_end=optim[kind].param_groups[0]["lr"],
                  tokenizer_schedule_start=started_schedule,
                  tokenizer_schedule_end=model.joint_alignment.diger_schedule_state(),
                  formal_SID_sha256=hashlib.sha256(current_codes.tobytes()).hexdigest(),
                  formal_SID_change_since_initial=runtime.compare_codes(context["initial_codes"], current_codes),
                  formal_SID_statistics=runtime.sid_catalog_stats(current_codes, [256] * 4),
                  gumbel_stream_seed=runtime.phase_seed(protocol["gumbel_seed"], "gumbel", row["logical_index"]) if kind == "tokenizer" else None)
    return result


def evaluate(context, arm, epoch, *, split="val", beam_profile=False):
    import torch
    pipeline, runtime, protocol = context["pipeline"], context["runtime"], context["protocol"]
    pipeline.model.set_joint_phase("recommender")
    pipeline.model.generate_w_decoding_graph = True
    runtime.set_phase_seed(runtime.phase_seed(protocol["seed"], "evaluation", epoch))
    batch = protocol["smoke"]["eval_batch_size"] if context["mode"] == "smoke" else protocol["eval_batch_size"]
    dataset_split = "val" if beam_profile else split
    loader = torch.utils.data.DataLoader(pipeline.tokenized_datasets[dataset_split], batch_size=batch,
        shuffle=False, collate_fn=pipeline.tokenizer.collate_fn[dataset_split], num_workers=0,
        generator=torch.Generator().manual_seed(runtime.phase_seed(protocol["seed"], "evaluation", epoch)))
    loader = pipeline.accelerator.prepare_data_loader(loader)
    start = time.perf_counter()
    with runtime.autocast_context(context["device"], True):
        values = pipeline.trainer.evaluate(loader, split=split, epoch=epoch)
    metrics = {key: float(value) for key, value in values.items()}
    need(metrics and all(math.isfinite(value) for value in metrics.values()), "Nonfinite/empty evaluation")
    weighted = .8 * metrics["ndcg@10"] + .2 * metrics["hit_rate@10"]
    need(abs(weighted - metrics["weighted_score"]) < 1e-7, "Metric aliases/weighted score differ")
    result = dict(arm=arm, decoder_epoch=epoch, split=split, metrics=metrics,
                  seconds=time.perf_counter() - start, actual_dataset_split=dataset_split,
                  beam_profile_not_test=beam_profile)
    print("[LONG_EVAL] " + json.dumps(result, sort_keys=True), flush=True)
    return result


def context_for(args):
    import numpy as np
    import torch
    protocol = protocol_lib.validate(read(args.config))
    goal, engine = args.goal_root.resolve(), args.engine_root.resolve()
    need(goal.name == protocol_lib.CAMPAIGN and not goal.is_symlink(), "Wrong campaign output root")
    device = torch.device("cpu" if args.allow_cpu else "cuda:0")
    need(not args.allow_cpu or args.mode == "smoke", "CPU is only an explicit plumbing smoke, never formal/profile")
    if not args.allow_cpu:
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and "H100" in torch.cuda.get_device_name(0).upper() and torch.cuda.is_bf16_supported(),
             "GPU execution requires exactly one BF16-capable H100")
        need(args.expected_code_commit and args.expected_engine_commit, "GPU modes need both frozen commits")
    for directory, expected in ((ROOT, args.expected_code_commit), (engine, args.expected_engine_commit)):
        if expected:
            head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
            status = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
            need(head == expected and not status, "Frozen source checkout is wrong or dirty: " + str(directory))
    manifest = read(goal / "delivery/INPUTS.json")
    need(manifest["protocol"] == protocol, "Prepared input protocol differs")
    for row in manifest["records"]:
        need(sha(protocol_lib.relative(goal / "inputs", row["path"])) == row["sha256"], "Owned immutable input changed")
    source = {name: sha(engine / name) for name in protocol_lib.ENGINE_FILES}
    root_sources = {name: sha(ROOT / name) for name in (
        "scripts/train/run_diger_mixer_long.py", "scripts/checks/diger_mixer_long_protocol.py",
        "scripts/train/run_diger_noise_pilot.py", "scripts/train/run_diger_mixer_factorial.py",
        "scripts/checks/diger_mixer_factorial_protocol.py", "scripts/checks/diger_noise_pilot_protocol.py")}
    identity = dict(campaign_id=protocol_lib.CAMPAIGN, execution_backend=device.type, protocol_sha256=sha(args.config),
                    inputs_sha256=sha(goal / "delivery/INPUTS.json"), engine_sources=source, root_sources=root_sources)
    output = goal / ("cpu_smoke" if args.allow_cpu else "smoke" if args.mode == "smoke" else "formal")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists():
        need(read(output / "identity.json") == identity, "Existing run source/config/input identity differs")
    else:
        write(output / "identity.json", identity)
    work = output / "runtime" / uuid.uuid4().hex
    work.mkdir(parents=True)
    os.environ.update(HF_DATASETS_CACHE=str(work / "hf_datasets"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        os.environ.pop(name, None)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = module("_stage05_long_runtime", ROOT / "scripts/train/run_diger_noise_pilot.py")
    short = module("_stage05_short_factorial", ROOT / "scripts/train/run_diger_mixer_factorial.py")
    seeds = module("_stage05_long_seeds", ROOT / "scripts/checks/diger_noise_pilot_protocol.py")
    runtime.torch, runtime.np = torch, np
    runtime.phase_seed, runtime.loss_divisor = seeds.phase_seed, seeds.loss_divisor
    sys.path.insert(0, str(engine))
    from genrec.utils import get_config
    from genrec.pipeline import Pipeline
    from genrec.models.DIFF_GRM.trainer import DIFF_GRMTrainer
    from datasets import disable_caching
    disable_caching()
    layers = [str(ROOT / name) for name in ("configs/shared/diff_formal_common.yaml",
              "configs/shared/diff_formal_repo_v6.yaml", "configs/05_crosspse_joint/diff_formal_crosspse_joint_diger_v2.yaml")]
    base = runtime.absolute_engine_paths(get_config("DIFF_GRM", "AmazonReviews2014", layers, {}), goal / "inputs")
    shadow, _ = runtime.shadow_dataset_cache(base["cache_dir"], work, "Beauty")
    base.update(cache_dir=str(shadow), mixed_precision="no", rand_seed=2024, num_proc=1,
                run_id=protocol_lib.CAMPAIGN + "_" + args.mode, item_local_attention={"enabled": False},
                joint_recommender_checkpoint=str(goal / "inputs" / protocol["initial_decoder"]),
                crosspse_mixer_enabled=True, joint_crosspse_enabled=True, joint_gradient_mode="diger_st",
                joint_diger_recommendation_weight=1, joint_recommender_sia_weight=0, joint_tokenizer_sia_weight=0,
                joint_diger_gumbel_noise_start=1, joint_diger_gumbel_noise_end=.2,
                joint_diger_softmax_temperature_start=1, joint_diger_softmax_temperature_end=1,
                joint_diger_assignment_temperature=.1, joint_balance_temperature=.1,
                joint_balance_weight=.01, joint_anchor_weight=.01, joint_diger_uncertainty_decay="cosine",
                train_batch_size=64, eval_batch_size=8, gradient_accumulation_steps=16,
                label_smoothing=.2, max_grad_norm=1, force_regenerate_opq=False,
                max_eval_batches=protocol["smoke"]["eval_batches"] if args.mode == "smoke" else None,
                patience=0, lr=.001, weight_decay=0, warmup_steps=500, evaluation_level="sid", eval_expand_sid_to_items=False)
    base = runtime.configure_paths(base, output, "common")
    expected_config = dict(n_digit=4, codebook_size=256, encoder_n_layer=1, decoder_n_layer=4,
                           n_head=4, n_embd=256, n_inner=1024, dropout=.1, label_smoothing=.2,
                           masking_strategy="guided", guided_select="least", guided_refresh_each_step=False,
                           diffusion_loss_reduction="per_view", train_sliding=True, min_hist_len=2,
                           max_hist_len=50, evaluation_level="sid", eval_expand_sid_to_items=False)
    need(all(base.get(key) == value for key, value in expected_config.items()), "Parsed backbone/data/OCN controls differ")
    with runtime.working_directory(work):
        pipeline = Pipeline("DIFF_GRM", "AmazonReviews2014", config_dict=base, trainer=object())
    need(pipeline.accelerator.num_processes == 1 and not pipeline.model.item_local_attention_enabled,
         "Unexpected process count or extra architecture")
    for split, count in (("train", 105668), ("val", 22363), ("test", 22363)):
        need(len(pipeline.tokenized_datasets[split]) == count, "Actual tokenized split size differs: " + split)
    model = pipeline.model.to(device)
    offline = torch.load(goal / "inputs" / protocol["initial_decoder"], map_location="cpu", weights_only=False)
    need(all(name in model.state_dict() and torch.equal(value, model.state_dict()[name].cpu())
             for name, value in offline.items()), "Complete original offline recommender was not loaded")
    model.eval()
    initial_codes = model.joint_alignment.hard_codes[1:].detach().cpu().numpy().copy()
    need(hashlib.sha256(initial_codes.tobytes()).hexdigest() == protocol["canonical_SID_array_sha256"], "Initial catalogue SID differs")
    effective = np.load(goal / "inputs/canonical/E0.npy", allow_pickle=False)
    need(effective.shape == (4, 256, 64) and effective.dtype == np.float32
         and hashlib.sha256(effective.tobytes()).hexdigest() == protocol["canonical_E0_array_sha256"], "Canonical E0 array differs")
    short.rebase_common_mixer(model, torch.from_numpy(effective).to(device), 2024)
    model.joint_alignment.configure_diger_schedule(60)
    model.refresh_joint_codes()
    need(np.array_equal(model.joint_alignment.hard_codes[1:].cpu().numpy(), initial_codes), "E0 rebasing changed official nearest SID")
    common = runtime.cpu_state(model)
    model.eval()
    for arm in ARMS:
        configure(model, arm)
        with torch.no_grad():
            books, _ = model.joint_alignment.mixer.crossed_codebooks(return_attention=False)
        need(torch.equal(books, torch.from_numpy(effective).to(device)), "An initial arm differs from common E0")
    configure(model, ARMS[0])
    pipeline.trainer = DIFF_GRMTrainer(pipeline.config, model, pipeline.tokenizer)
    common_path = output / "common/initial.pt"
    if common_path.exists():
        saved = torch.load(common_path, map_location="cpu", weights_only=False)
        need(runtime.tensor_state_hash(saved["model"]) == runtime.tensor_state_hash(common), "Reconstructed initial state differs")
    else:
        atomic_torch(common_path, dict(model=common, identity=identity,
                    offline_recommender_loaded_all_tensors=True, E0_file_sha256=protocol["canonical_E0_file_sha256"]))
    environment = dict(utc=utc(), python=sys.version, torch=torch.__version__, numpy=np.__version__,
                       cuda_runtime=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                       code_commit=args.expected_code_commit, engine_commit=args.expected_engine_commit,
                       initial_model_tensor_sha256=runtime.tensor_state_hash(common),
                       initial_offline_recommender_sha256=runtime.tensor_state_hash(offline))
    write(output / ("environment_" + args.mode + "_" + args.arm + ".json"), environment)
    write(output / ("resolved_config_" + args.mode + "_" + args.arm + ".json"), {key: str(value) if key == "device" else value
          for key, value in pipeline.config.items() if key != "accelerator"})
    return dict(protocol=protocol, identity=identity, runtime=runtime, pipeline=pipeline, device=device,
                common=common, initial_codes=initial_codes, output=output, goal=goal,
                mode="smoke" if args.mode == "smoke" else "formal", deadline=args.deadline_unix or time.time() + 3600)


def train_arm(context, arm, resume):
    import torch
    pipeline, runtime, protocol = context["pipeline"], context["runtime"], context["protocol"]
    directory = context["output"] / arm
    directory.mkdir(exist_ok=True)
    done = directory / "TRAIN_COMPLETE.json"
    if done.exists():
        value = read(done)
        need(value["identity"] == context["identity"] and sha(directory / "last.pt") == value["last_sha256"]
             and sha(directory / "best.pt") == value["best_sha256"], "Completed arm damaged or wrong identity")
        return value
    model = pipeline.model
    model.load_state_dict(context["common"], strict=True)
    configure(model, arm)
    optim, schedule = optimizers(model, arm, protocol)
    plan = protocol_lib.phases(protocol, arm, smoke=context["mode"] == "smoke")
    progress = dict(next_stage=0, decoder_epochs=0, decoder_updates=0, tokenizer_updates=0,
                    noisy_calls=0, training=[], validation=[], best_score=-1, best_decoder_epoch=None)
    last = directory / "last.pt"
    if last.exists():
        need(resume, "Existing arm boundary requires explicit --resume")
        progress = restore(context, arm, torch.load(last, map_location="cpu", weights_only=False), optim, schedule)
    else:
        initial = evaluate(context, arm, 0)
        progress["validation"].append(initial)
        progress["best_score"], progress["best_decoder_epoch"] = initial["metrics"]["weighted_score"], 0
        state = payload(context, arm, optim, schedule, progress)
        atomic_torch(directory / "best.pt", state)
        atomic_torch(last, state)
    for stage in range(progress["next_stage"], len(plan)):
        row = plan[stage]
        if time.time() > context["deadline"] - 120:
            raise TimeoutError("Insufficient reserved process time to enter another full phase")
        result = execute_phase(context, arm, row, optim, schedule)
        progress["training"].append(result)
        progress.update(next_stage=stage + 1, decoder_epochs=row["decoder_epochs"], tokenizer_updates=row["tokenizer_updates"])
        if row["phase"] == "recommender":
            progress["decoder_updates"] += result["optimizer_updates"]
        else:
            progress["noisy_calls"] += result["noise_applied_calls"]
        need(int(model.joint_alignment.diger_optimizer_step) == progress["tokenizer_updates"], "Tokenizer counter drifted")
        need(schedule.state_dict()["last_epoch"] == progress["decoder_updates"], "Recommender scheduler clock drifted")
        improved = False
        validate = row["phase"] == "recommender" and (context["mode"] == "smoke"
                   or row["decoder_epochs"] in protocol["evaluation"]["epochs"])
        if validate:
            evaluated = evaluate(context, arm, row["decoder_epochs"])
            progress["validation"].append(evaluated)
            if evaluated["metrics"]["weighted_score"] > progress["best_score"]:
                progress["best_score"] = evaluated["metrics"]["weighted_score"]
                progress["best_decoder_epoch"] = row["decoder_epochs"]
                improved = True
        state = payload(context, arm, optim, schedule, progress)
        if improved:
            atomic_torch(directory / "best.pt", state)
        atomic_torch(last, state)
        if context["mode"] == "smoke" and row["phase"] == "tokenizer":
            atomic_torch(directory / "tokenizer_boundary.pt", state)
        write(directory / "progress.json", progress)
    need(progress["decoder_epochs"] == plan[-1]["decoder_epochs"], "Incomplete recommender budget")
    expected_calls = sum(2 * row["batches"] for row in plan if row["phase"] == "tokenizer")
    need(progress["noisy_calls"] == expected_calls, "Actual noisy assignment count differs")
    need(progress["decoder_updates"] == sum(row["optimizer_updates"] for row in plan if row["phase"] == "recommender"),
         "Incomplete recommender optimizer budget")
    result = dict(status="smoke_trained" if context["mode"] == "smoke" else "trained_not_tested",
                  arm=arm, identity=context["identity"], progress=progress, last_sha256=sha(last),
                  best_sha256=sha(directory / "best.pt"), final_model_tensor_sha256=runtime.tensor_state_hash(model.state_dict()),
                  final_SID_sha256=hashlib.sha256(model.joint_alignment.hard_codes[1:].cpu().numpy().tobytes()).hexdigest(),
                  final_SID_change=runtime.compare_codes(context["initial_codes"], model.joint_alignment.hard_codes[1:].cpu().numpy()),
                  selected_decoder_epoch=progress["best_decoder_epoch"])
    write(done, result)
    return result


def controls(context, results):
    by_arm = {row["arm"]: row for row in results}
    need(set(by_arm) == set(ARMS), "All four arms are required for campaign control checks")
    reference = [row for row in by_arm[ARMS[0]]["progress"]["training"] if row["phase"] == "recommender"]
    keys = ("logical_index", "batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256")
    for value in results:
        actual = [row for row in value["progress"]["training"] if row["phase"] == "recommender"]
        need([{key: row[key] for key in keys} for row in actual] == [{key: row[key] for key in keys} for row in reference],
             "Recommender RNG/data/order/update budgets are not paired")
    joints = [[row for row in by_arm[arm]["progress"]["training"] if row["phase"] == "tokenizer"] for arm in (ARMS[1], ARMS[3])]
    tokenizer_keys = keys + ("gumbel_stream_seed", "gumbel_calls", "assignment_training_enabled_calls", "noise_applied_calls")
    need([{key: row[key] for key in tokenizer_keys} for row in joints[0]] == [{key: row[key] for key in tokenizer_keys} for row in joints[1]],
         "Tokenizer RNG/data/order/update budgets are not paired")
    left, right = by_arm[ARMS[0]], by_arm[ARMS[2]]
    need(left["final_model_tensor_sha256"] == right["final_model_tensor_sha256"], "Frozen final models differ")
    need([row["metrics"] for row in left["progress"]["validation"]] == [row["metrics"] for row in right["progress"]["validation"]],
         "Frozen validation trajectories differ")
    metrics = {arm: by_arm[arm]["progress"]["validation"][-1]["metrics"] for arm in ARMS}
    increment = {key: (metrics[ARMS[3]][key] - metrics[ARMS[2]][key]) - (metrics[ARMS[1]][key] - metrics[ARMS[0]][key])
                 for key in ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "weighted_score")}
    value = dict(status="passed", all_four_arms_complete=True, frozen_models_and_metrics_identical=True,
                 paired_recommender_and_tokenizer_data_controls=True, fixed_final_validation=metrics,
                 Mixer_increment=increment, formal=context["mode"] == "formal")
    write(context["output"] / "CONTROL_PASS.json", value)
    return value


def recovery_probe(context, arm):
    import torch
    directory = context["output"] / arm
    source = directory / "tokenizer_boundary.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)
    protected = {name: sha(directory / name) for name in ("last.pt", "best.pt", "tokenizer_boundary.pt")}
    plan = protocol_lib.phases(context["protocol"], arm, smoke=True)
    next_row = plan[state["progress"]["next_stage"]]
    need(next_row["phase"] == "recommender", "Recovery probe must cross the real Tokenizer->recommender boundary")
    outputs = []
    for _ in range(2):
        model = context["pipeline"].model
        configure(model, arm)
        optim, schedule = optimizers(model, arm, context["protocol"])
        restored = restore(context, arm, state, optim, schedule)
        need(digest(context["runtime"].phase_rng_state(context["device"])) == digest(state["rng"]), "RNG restore differs")
        row = execute_phase(context, arm, next_row, optim, schedule)
        outputs.append(dict(model=context["runtime"].cpu_state(model),
                            optimizers={key: None if value is None else value.state_dict() for key, value in optim.items()},
                            scheduler=schedule.state_dict(), rng=context["runtime"].phase_rng_state(context["device"]),
                            data_order_sha256=row["data_order_sha256"]))
        outputs[-1] = summary(outputs[-1])
    if outputs[0] != outputs[1]:
        write(directory / "RECOVERY_FAILURE.json", dict(source_sha256=sha(source),
              first=outputs[0], second=outputs[1], device=str(context["device"])))
        raise ValueError("Boundary restore/replayed real update differs; see RECOVERY_FAILURE.json")
    need(protected == {name: sha(directory / name) for name in protected}, "Recovery probe changed original committed checkpoints")
    result = dict(status="passed", arm=arm, source_sha256=sha(source),
                  model_optimizer_scheduler_RNG_restored=True, repeated_real_next_phase_identical=True,
                  original_checkpoints_unchanged=True, device=str(context["device"]))
    write(directory / "RECOVERY_PASS.json", result)
    return result


def profile(context):
    """Private full-size training and validation timing; never consumes test data."""
    import torch
    old_mode = context["mode"]
    context["mode"] = "formal"
    pipeline, protocol = context["pipeline"], context["protocol"]
    model = pipeline.model
    model.load_state_dict(context["common"], strict=True)
    configure(model, "mixer_joint")
    optim, schedule = optimizers(model, "mixer_joint", protocol)
    recommender = execute_phase(context, "mixer_joint", dict(phase="recommender", logical_index=1,
               batches=1652, optimizer_updates=104), optim, schedule)
    tokenizer = execute_phase(context, "mixer_joint", dict(phase="tokenizer", logical_index=0,
               batches=96, optimizer_updates=6), optim, schedule)
    pipeline.config["max_eval_batches"] = pipeline.trainer.config["max_eval_batches"] = None
    validation = evaluate(context, "profile_not_formal", 1)
    pipeline.config["max_eval_batches"] = pipeline.trainer.config["max_eval_batches"] = 16
    test_beam_on_validation = evaluate(context, "profile_not_formal", 1, split="test", beam_profile=True)
    temporary = context["output"] / "profile_boundary.pt"
    beginning = time.perf_counter()
    atomic_torch(temporary, payload(context, "mixer_joint", optim, schedule, dict(profile_only=True)))
    temporary_sha = sha(temporary)
    checkpoint_seconds = time.perf_counter() - beginning
    estimated_test = test_beam_on_validation["seconds"] * math.ceil(22363 / 8) / 16
    base_estimate = 4 * (35 * recommender["elapsed_seconds"] + 8 * validation["seconds"] + estimated_test)
    base_estimate += 20 * tokenizer["elapsed_seconds"] + 192 * checkpoint_seconds
    training_estimates = {
        arm: 1.2 * (35 * recommender["elapsed_seconds"] + 8 * validation["seconds"]
                     + (10 * tokenizer["elapsed_seconds"] if arm.endswith("_joint") else 0)
                     + (58 if arm.endswith("_joint") else 48) * checkpoint_seconds)
        for arm in ARMS
    }
    evaluation_estimate = 1.2 * (4 * estimated_test + 8 * checkpoint_seconds)
    result = dict(status="profiled_not_formal", measured_full_recommender_epoch=recommender,
                  measured_full_tokenizer_phase=tokenizer, measured_full_validation=validation,
                  test_beam_on_validation_examples=test_beam_on_validation,
                  estimated_full_test_per_arm_seconds=estimated_test,
                  estimated_four_arm_seconds=base_estimate,
                  estimated_four_arm_seconds_with_20_percent_margin=max(base_estimate * 1.2,
                       sum(training_estimates.values()) + evaluation_estimate),
                  estimated_seconds_by_task={**{"train_arm:" + arm: seconds for arm, seconds in training_estimates.items()},
                      "evaluate:all": evaluation_estimate},
                  checkpoint_seconds=checkpoint_seconds, checkpoint_sha256=temporary_sha,
                  actual_test_data_evaluated=False, formal_updates_executed=0)
    write(context["output"] / "PROFILE.json", result)
    pipeline.config["max_eval_batches"] = pipeline.trainer.config["max_eval_batches"] = protocol["smoke"]["eval_batches"]
    context["mode"] = old_mode
    return result


def test_selected(context):
    import torch
    output = context["output"]
    need(context["mode"] == "formal", "Only complete formal arms may select and test")
    results = [read(output / arm / "TRAIN_COMPLETE.json") for arm in ARMS]
    controls(context, results)
    selection = dict(identity=context["identity"], arms={row["arm"]: dict(best_sha256=row["best_sha256"],
                     selected_decoder_epoch=row["selected_decoder_epoch"], last_sha256=row["last_sha256"]) for row in results})
    sealed = output / "SELECTION_SEALED.json"
    if sealed.exists():
        need(read(sealed) == selection, "Previously sealed selection differs")
    else:
        write(sealed, selection)
    tests = {}
    for arm in ARMS:
        directory = output / arm
        target, started = directory / "TEST_RESULT.json", directory / "TEST_STARTED.json"
        if target.exists():
            result = read(target)
            need(result["selection_sha256"] == sha(sealed) and result["best_sha256"] == sha(directory / "best.pt"),
                 "Saved test has wrong selected weights/source")
            tests[arm] = result
            continue
        need(not started.exists(), "Test already started without a committed result; inspect rather than silently rerun")
        profile_path = context["goal"] / "smoke/PROFILE.json"
        estimate = read(profile_path)["estimated_full_test_per_arm_seconds"]
        need(time.time() + estimate * 1.25 + 120 < context["deadline"], "Insufficient measured process time before starting final test")
        state = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        optim, schedule = optimizers(context["pipeline"].model, arm, context["protocol"])
        restore(context, arm, state, optim, schedule)
        need(state["progress"]["decoder_epochs"] == selection["arms"][arm]["selected_decoder_epoch"], "Selected checkpoint epoch differs")
        receipt = dict(utc=utc(), selection_sha256=sha(sealed), best_sha256=sha(directory / "best.pt"),
                       arm=arm, actual_split="test", identity=context["identity"])
        with started.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        evaluated = evaluate(context, arm, state["progress"]["decoder_epochs"], split="test")
        evaluated.update(selection_sha256=sha(sealed), best_sha256=sha(directory / "best.pt"),
                         started_sha256=sha(started), selected_tokenizer_updates=state["progress"]["tokenizer_updates"])
        write(target, evaluated)
        tests[arm] = evaluated
    complete = dict(status="four_arm_training_and_testing_complete_not_delivery", identity=context["identity"],
                    selection_sha256=sha(sealed), tests=tests, controls_sha256=sha(output / "CONTROL_PASS.json"))
    write(output / "COMPUTE_COMPLETE.json", complete)
    return complete


@contextmanager
def file_lock(path, *, shared=False, nonblocking=True):
    """Kernel lock on the shared filesystem; process death releases it on every host.

    Never unlink lock files: removing a locked inode would admit duplicate workers.
    Linux-only execution is intentional; GPU Jobs and the real CPU gate use Linux.
    """
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(handle.fileno(), operation | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as error:
            raise ValueError("Campaign execution lock is already held: " + path.name) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def execution_lock(goal, arm="all"):
    directory = Path(goal) / "locks"
    with file_lock(directory / "campaign.lock", shared=arm != "all"):
        with file_lock(directory / ("arm_" + arm + ".lock")):
            yield


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=protocol_lib.DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--expected-engine-commit")
    parser.add_argument("--deadline-unix", type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    need(not args.profile or args.mode == "smoke" and not args.allow_cpu, "Profile is only authorized on the H100 smoke")
    need(args.allow_cpu or args.deadline_unix is not None, "GPU work requires a bounded process deadline")
    if args.mode != "smoke":
        smoke = read(args.goal_root / "smoke/SMOKE_PASS.json")
        need(smoke["status"] == "H100_SMOKE_PASS_NOT_FORMAL" and smoke["controls"]["status"] == "passed"
             and len(smoke["recovery"]) == 2 and all(row["status"] == "passed" and row["device"].startswith("cuda")
                                                   for row in smoke["recovery"]), "Formal requires four-arm H100 and recovery evidence")
    scope = args.arm if args.mode == "train" else "all"
    with execution_lock(args.goal_root, scope):
        # Short, serialized initialization protects common identity and E0 files.
        # Training then runs independently; no inter-arm gradients or DDP.
        with file_lock(args.goal_root / "locks/initialize.lock", nonblocking=False):
            context = context_for(args)
        if args.mode != "smoke":
            need(read(args.goal_root / "smoke/identity.json") == context["identity"], "H100 gate ran different sources/inputs")
        if args.mode == "evaluate":
            result = test_selected(context)
        else:
            arms = ARMS if args.arm == "all" else (args.arm,)
            results = [train_arm(context, arm, args.resume) for arm in arms]
            result = dict(status="trained", arms=list(arms))
            if len(arms) == 4:
                result["controls"] = controls(context, results)
            if args.mode == "smoke":
                need(len(arms) == 4, "Smoke must exercise all four authorized arms")
                result["recovery"] = [recovery_probe(context, arm) for arm in (ARMS[1], ARMS[3])]
                if args.profile:
                    result["profile"] = profile(context)
                result["status"] = "CPU_SMOKE_PASS_NOT_H100" if args.allow_cpu else "H100_SMOKE_PASS_NOT_FORMAL"
                write(context["output"] / "SMOKE_PASS.json", result)
        print("[LONG_DONE] " + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print("[LONG_FAILED] " + type(exception).__name__ + ": " + str(exception), file=sys.stderr, flush=True)
        raise
