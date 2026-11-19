"""Final GPU evaluation/export windows and scientific report generation."""
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .campaign import freeze_validation_selection, verify_result
from .data import sha256
from .protocol import MODELS, atomic_json, canonical_hash

METRICS = [f"{kind}@{k}" for kind in ("HR", "NDCG") for k in (5, 10, 20)]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_test_unlock(selection_path, checkpoint, checkpoint_path):
    selection_path = Path(selection_path)
    lock, state = read_json(selection_path), read_json(selection_path.parent / "queue.json")
    if freeze_validation_selection(state) != lock or lock.get("validation_only") is not True:
        raise ValueError("Validation selection is not fully frozen")
    config = checkpoint["config"]
    if (lock.get("test_unlocked") is not True or lock["source_sha256"] != checkpoint["binding"]["source_sha256"]
            or lock["models"][config["model"]]["config_sha256"] != checkpoint["binding"]["config_sha256"]):
        raise ValueError("Test checkpoint differs from frozen configuration/source")
    eligible = [t for t in state["tasks"] if t["phase"] in {"formal", "legacy"}
                and (Path(t["output"]) / "best.pt").resolve() == Path(checkpoint_path).resolve()]
    if len(eligible) != 1 or sha256(checkpoint_path) != eligible[0]["result"]["best_sha256"]:
        raise ValueError("Only the 13 designated fresh-seed best checkpoints may access test")
    return lock


def run_command(command, log, root, environment=None):
    with Path(log).open("xb") as stream:
        result = subprocess.run(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def run_evaluations(root, campaign, cache, inventory):
    """13 tests, at most eight independent GPUs. Selection was already sealed."""
    root, campaign = Path(root), Path(campaign)
    state = read_json(campaign / "queue.json")
    lock = read_json(campaign / "FROZEN_SELECTION.json")
    if freeze_validation_selection(state) != lock:
        raise ValueError("Frozen selection disagrees with completed validation results")
    tasks = [t for t in state["tasks"] if t["phase"] in {"formal", "legacy"}]
    if len(tasks) != 13 or state["workers"]:
        raise ValueError("Training must be complete before final evaluation")
    destination = campaign / "evaluations"
    destination.mkdir(exist_ok=True)
    pending, running, completed = list(tasks), {}, {}
    try:
        while pending or running:
            for gpu, record in list(running.items()):
                if record["process"].poll() is None:
                    continue
                record["log"].close()
                if record["process"].returncode:
                    raise RuntimeError(f"Final test failed: {record['task']['task_id']}")
                result = read_json(record["output"] / "result.json")
                if result["samples"] != 22363 or result["split"] != "test":
                    raise ValueError("Incomplete final evaluation")
                if result["checkpoint_sha256"] != record["task"]["result"]["best_sha256"]:
                    raise ValueError("Wrong checkpoint evaluated")
                completed[record["task"]["task_id"]] = {"output": str(record["output"]), "result": result}
                del running[gpu]
                atomic_json(campaign / "evaluation_index.json", completed)
            for gpu in range(8):
                if gpu in running or not pending:
                    continue
                task = pending.pop(0)
                output = destination / task["task_id"]
                if (output / "result.json").is_file():
                    # Reuse only identical, intact deterministic evaluation artifacts.
                    result = read_json(output / "result.json")
                    if (result["checkpoint_sha256"] != task["result"]["best_sha256"]
                            or sha256(output / "predictions.jsonl") != result["prediction_sha256"]):
                        raise ValueError("Existing final evaluation differs")
                    completed[task["task_id"]] = {"output": str(output), "result": result}
                    continue
                if output.exists():
                    raise RuntimeError("Partial evaluation exists; reconcile before rerun")
                command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"), "evaluate",
                           "--checkpoint", str(Path(task["output"]) / "best.pt"), "--cache", str(cache),
                           "--split", "test", "--selection-lock", str(campaign / "FROZEN_SELECTION.json"),
                           "--output", str(output)]
                log = (destination / (task["task_id"] + ".log")).open("xb")
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": inventory[gpu]["uuid"],
                               "CONTINUOUS_GPU_UUID": inventory[gpu]["uuid"], "OMP_NUM_THREADS": "2",
                               "MKL_NUM_THREADS": "2"}
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = {"task": task, "output": output, "log": log, "process": process}
            time.sleep(1)
    finally:
        for record in running.values():
            if record["process"].poll() is None:
                record["process"].terminate()
            try:
                record["process"].wait(timeout=20)
            except subprocess.TimeoutExpired:
                record["process"].kill()
                record["process"].wait()
            record["log"].close()
    atomic_json(campaign / "evaluation_index.json", completed)
    return completed


def benchmark_checkpoints(root, destination, cache, checkpoints, gpu_uuid):
    """Sequential on one otherwise idle device. Caller has drained all workers."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(exist_ok=True)
    results = {}
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid, "CONTINUOUS_GPU_UUID": gpu_uuid,
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for model in MODELS:
        output = destination / model
        if (output / "result.json").is_file():
            result = read_json(output / "result.json")
            if result["checkpoint_sha256"] != sha256(checkpoints[model]):
                raise ValueError("Existing benchmark checkpoint differs")
        else:
            command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"), "benchmark",
                       "--checkpoint", str(checkpoints[model]), "--cache", str(cache), "--output", str(output)]
            run_command(command, destination / (model + ".log"), root, environment)
            result = read_json(output / "result.json")
        results[model] = result
    return results


def export_checkpoints(root, campaign):
    root, campaign = Path(root), Path(campaign)
    state = read_json(campaign / "queue.json")
    exports = {}
    for task in state["tasks"]:
        if task["phase"] != "formal":
            continue
        output = campaign / "exports" / task["task_id"]
        if not (output / "result.json").is_file():
            output.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-B", str(root / "scripts/train/run_continuous.py"), "export",
                       "--checkpoint", str(Path(task["output"]) / "best.pt"), "--device", "cpu",
                       "--output", str(output)]
            run_command(command, output.parent / (task["task_id"] + ".log"), root)
        exports[task["task_id"]] = read_json(output / "result.json")
    return exports


def mean_and_std(results):
    return {metric: {"mean": statistics.mean(row[metric] for row in results),
                     "std": statistics.stdev(row[metric] for row in results), "n": len(results)}
            for metric in METRICS}


def build_summary(state, evaluations):
    models, paired = {}, []
    for model in MODELS:
        tasks = sorted((t for t in state["tasks"] if t["phase"] == "formal" and
                        t["config"]["model"] == model), key=lambda t: t["seed"])
        if len(tasks) != 5 or [t["seed"] for t in tasks] != list(range(3031, 3036)):
            raise ValueError("Five fresh formal seeds required")
        metrics = [evaluations[t["task_id"]]["result"] for t in tasks]
        models[model] = {"seeds": [t["seed"] for t in tasks], "test": mean_and_std(metrics),
                         "raw": [{"seed": t["seed"], **evaluations[t["task_id"]]["result"]} for t in tasks],
                         "config": state["selected_configs"][model]["config"]}
    for seed in (3031, 3032, 3033):
        full = next(t for t in state["tasks"] if t["phase"] == "formal"
                    and t["config"]["model"] == "sasrec_fullce" and t["seed"] == seed)
        legacy = next(t for t in state["tasks"] if t["phase"] == "legacy" and t["seed"] == seed)
        left, right = evaluations[full["task_id"]]["result"], evaluations[legacy["task_id"]]["result"]
        if full["config_sha256"] != legacy["config_sha256"]:
            raise ValueError("Legacy comparison changed hyperparameters")
        paired.append({"seed": seed, "full": {m: left[m] for m in METRICS},
                       "legacy": {m: right[m] for m in METRICS},
                       "full_minus_legacy": {m: left[m] - right[m] for m in METRICS},
                       "full_updates": full["result"]["updates"], "legacy_updates": legacy["result"]["updates"],
                       "full_targets_per_epoch": 131413, "legacy_targets_per_epoch": 105668,
                       "full_target_exposures": full["result"]["epochs"] * 131413,
                       "legacy_target_exposures": legacy["result"]["epochs"] * 105668,
                       "full_training_seconds": full["result"]["elapsed_seconds"],
                       "legacy_training_seconds": legacy["result"]["elapsed_seconds"]})
    return {"models": models, "legacy_paired": paired,
            "legacy_paired_delta_summary": mean_and_std([p["full_minus_legacy"] for p in paired]),
            "training_runs": len(state["tasks"]), "standard_deviation_ddof": 1,
            "runs_reaching_epoch_cap": [t["task_id"] for t in state["tasks"]
                                        if t["result"]["epochs"] == t["config"]["max_epochs"]],
            "late_best_validation_at_cap": [t["task_id"] for t in state["tasks"]
                if t["result"]["epochs"] == t["config"]["max_epochs"] and
                t["result"]["best_epoch"] >= t["config"]["max_epochs"]-t["config"]["patience"]],
            "selection": freeze_validation_selection(state),
            "limitations": ["Leave-one-out is not a global temporal split or online evaluation.",
                            "Legacy/full protocols change target exposure and optimizer update counts.",
                            "SASRec/eSASRec differ in architecture and training loss, not a causal one-factor ablation.",
                            "This is a continuous retrieval baseline, not learned Decoder-adapted encoding."]}


def write_training_table(state, path):
    rows = []
    for task in state["tasks"]:
        result = task["result"]
        rows.append({"task_id": task["task_id"], "model": task["config"]["model"],
                     "phase": task["phase"], "seed": task["seed"], "protocol": task["protocol"],
                     "config_sha256": task["config_sha256"], "epochs": result["epochs"],
                     "best_epoch": result["best_epoch"], "updates": result["updates"],
                     "training_seconds": result["elapsed_seconds"], "parameters": result["parameter_count"],
                     **{f"validation_{metric}": result["best_validation"][metric] for metric in METRICS}})
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(state, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    for model in MODELS:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
        for task in state["tasks"]:
            if task["phase"] != "formal" or task["config"]["model"] != model:
                continue
            with (Path(task["output"]) / "curves.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            epochs = [int(row["epoch"]) for row in rows]
            for axis, key in zip(axes, ("train_loss", "validation_NLL", "validation_NDCG@10")):
                axis.plot(epochs, [float(row[key]) for row in rows], label=f"seed {task['seed']}")
                axis.set(xlabel="Epoch", ylabel=key)
                axis.grid(alpha=0.2)
        axes[0].set_title(model + " / five fresh seeds")
        axes[0].legend(ncol=3)
        fig.savefig(destination / (model + ".png"), dpi=160)
        fig.savefig(destination / (model + ".pdf"))
        plt.close(fig)


def finalize_delivery(root, campaign, data):
    """CPU audit after all GPU work. Fails closed on any incomplete deliverable."""
    root, campaign = Path(root), Path(campaign)
    state, evaluations = read_json(campaign / "queue.json"), read_json(campaign / "evaluation_index.json")
    if state["workers"] or freeze_validation_selection(state) != read_json(campaign / "FROZEN_SELECTION.json"):
        raise ValueError("Nonterminal training or selection changed")
    report = campaign / "delivery"
    report.mkdir(exist_ok=False)
    campaign_command = [sys.executable, "-B", str(root / "scripts/checks/verify_continuous_campaign.py"),
                        "--campaign", str(campaign), "--manifest",
                        str(root / "configs/06_continuous_baseline/search_manifest.json"),
                        "--report", str(report / "independent_campaign_audit.json")]
    run_command(campaign_command, report / "independent_campaign_audit.log", root)
    audits = {}
    for task in state["tasks"]:
        if verify_result(task, task["output"], state["source_sha256"]) != task["result"]:
            raise ValueError("Stored queue result differs from actual training artifacts")
        if task["phase"] in {"formal", "legacy"}:
            entry = evaluations[task["task_id"]]
            command = [sys.executable, "-B", str(root / "scripts/checks/verify_continuous_predictions.py"),
                       "--evaluation", entry["output"], "--data", str(data), "--protocol", task["protocol"],
                       "--report", str(report / (task["task_id"] + "_audit.json"))]
            run_command(command, report / (task["task_id"] + "_audit.log"), root)
            audits[task["task_id"]] = read_json(report / (task["task_id"] + "_audit.json"))
    summary = build_summary(state, evaluations)
    summary["independent_campaign_audit"] = read_json(report / "independent_campaign_audit.json")
    summary["independent_evaluation_audits"] = audits
    summary["latency"] = {m: read_json(campaign / "final_benchmarks" / m / "result.json") for m in MODELS}
    summary["initial_latency"] = {m: read_json(campaign.parent / "initial_benchmarks" / m / "result.json") for m in MODELS}
    usage_files = list((campaign.parent / "resource_attempts").glob("*/usage_summary.json"))
    if not usage_files:
        raise ValueError("Missing actual resource-use observations")
    usage = [read_json(path) for path in usage_files]
    summary["resource_usage"] = {
        "attempts": usage, "business_allocated_gpu_hours": sum(r["business_allocated_gpu_hours"] for r in usage),
        "business_points_calculated": sum(r["business_runtime_points_calculated"] for r in usage),
        "successful_training_worker_gpu_hours": sum(t["result"]["elapsed_seconds"] for t in state["tasks"])/3600,
        "platform_accounting": "Pending external Job terminal-state and allocation-duration reconciliation"}
    for model, result in summary["latency"].items():
        measurements = result["measurements"]
        if len(measurements) != 6 or {(r["batch_size"], r["component"]) for r in measurements} != {
                (b, c) for b in (1, 128) for c in ("query_forward", "catalog_top20", "end_to_end")}:
            raise ValueError("Missing final latency measurements")
    exports = read_json(campaign / "export_index.json")
    if len(exports) != 10:
        raise ValueError("Missing formal checkpoint/vector exports")
    for task in (t for t in state["tasks"] if t["phase"] == "formal"):
        entry = exports[task["task_id"]]
        for name, expected in entry["files"].items():
            if sha256(campaign / "exports" / task["task_id"] / name) != expected:
                raise ValueError("Exported item vectors changed")
    write_training_table(state, report / "training_results.csv")
    test_rows = []
    for task in state["tasks"]:
        if task["phase"] not in {"formal", "legacy"}:
            continue
        result = evaluations[task["task_id"]]["result"]
        test_rows.append({"task_id": task["task_id"], "model": task["config"]["model"],
                          "phase": task["phase"], "seed": task["seed"], "protocol": task["protocol"],
                          **{metric: result[metric] for metric in METRICS + ["NLL", "samples"]},
                          "checkpoint_sha256": result["checkpoint_sha256"]})
    with (report / "test_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    plot_curves(state, report / "curves")
    atomic_json(report / "summary.json", summary)
    chosen = summary["selection"]["selected_model"]
    lines = ["# Stage 06 continuous retrieval baselines", "", f"Validation-only selected baseline: `{chosen}`.",
             "", "89 verified training runs; full-catalog item-level evaluation, no history filtering.",
             "Sample standard deviations use ddof=1. No test metric influenced selection.", "",
             "| Model | Test HR@10 (mean ± SD) | Test NDCG@10 (mean ± SD) |", "|---|---:|---:|"]
    for model in MODELS:
        hr, ndcg = summary["models"][model]["test"]["HR@10"], summary["models"][model]["test"]["NDCG@10"]
        lines.append(f"| {model} | {hr['mean']:.6f} ± {hr['std']:.6f} | {ndcg['mean']:.6f} ± {ndcg['std']:.6f} |")
    lines += ["", "Legacy protocol paired differences, all seed-level metrics and configurations: `summary.json`.",
              "All 89 training curves remain in their run directories; five-seed figures are in `curves/`.", "",
              *["- " + item for item in summary["limitations"]]]
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = {path.relative_to(report).as_posix(): sha256(path) for path in report.rglob("*") if path.is_file()}
    proof = {"status": "PASS", "verified_training_runs": 89, "test_evaluations": 13,
             "independent_metric_audits": len(audits), "formal_exports": len(exports),
             "selected_model": chosen, "summary_sha256": sha256(report / "summary.json"),
             "files": files, "source_sha256": state["source_sha256"]}
    atomic_json(report / "DELIVERY_PASS.json", proof)
    return proof
