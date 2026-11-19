"""Offline, fail-closed validation summary. Standard library only; never trains.

Requires four complete new runs, the historical validation grid and historical
training delivery. Remote receipt paths are remapped only within explicit roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


N_USERS = 22363
SEEDS = (2024, 2025, 2026)
VARIANTS = ("control", "item_local")
BEAMS = (32, 256)
ENGINE = "1c879252f6395d6e64462511823c3929b9c42c00"
METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "weighted_score", "duplicate@10")
INITIAL = ("initial_base_state_sha256", "initial_SID_sha256", "initial_rng_sha256")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        need(path.is_file(), f"Missing evidence: {path}")
        item = {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}
        if expected is not None:
            need(item["sha256"] == expected["sha256"] and item["size_bytes"] == expected["size_bytes"],
                 f"Evidence SHA/size differs: {path}")
        need(str(path) not in self.files or self.files[str(path)] == item, "Evidence changed during summary")
        self.files[str(path)] = item
        return path

    def mapped(self, item, local_root, remote_root):
        remote = PurePosixPath(item["path"].replace("\\", "/"))
        base = PurePosixPath(str(remote_root).replace("\\", "/"))
        need(remote.is_relative_to(base), "Receipt escapes its declared remote root")
        relative = remote.relative_to(base)
        need(".." not in relative.parts and relative.parts, "Unsafe receipt relative path")
        root = Path(local_root).resolve()
        path = root.joinpath(*relative.parts).resolve()
        need(path.is_relative_to(root), "Receipt escapes its explicit local root")
        return self.bind(path, item)

    def document(self, path):
        return read(self.bind(path))


def unique_run(root, seed, variant):
    date = "20261020" if seed == 2024 else "20261021"
    name = f"stage07_item_local_joint_{variant}_seed{seed}_{date}_r1"
    root = Path(root).resolve()
    matches = ([root] if root.name == name else []) + [p for p in root.rglob(name) if p.is_dir()]
    need(len(matches) == 1, f"Expected exactly one complete run directory for {seed}/{variant}, found {len(matches)}")
    return matches[0]


def check_plan(plan):
    need(plan["identity_sha256"] == json_sha({k: v for k, v in plan.items() if k != "identity_sha256"}),
         "Evaluation plan identity digest differs")
    need(plan["validation_count"] == N_USERS and plan["split"] == "val", "Wrong validation population")


def selection(csv_path, fit):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "val"]
    scores = [(int(r["epoch"]), float(r["value"])) for r in rows if r["metric"] == "weighted_score"]
    need(scores and scores[0][0] == 0 and len({e for e, _ in scores}) == len(scores)
         and scores == sorted(scores) and all(math.isfinite(v) for _, v in scores), "Invalid selection score history")
    # The frozen recipe uses min_delta=0, strict improvement and earliest ties.
    chosen = max(scores, key=lambda row: row[1])
    need(chosen[0] == fit["best_epoch"] and abs(chosen[1] - fit["best_val_score"]) <= 1e-12,
         "Saved best is not the original validation32 selection")
    chosen_rows = [r for r in rows if int(r["epoch"]) == chosen[0]]
    metrics = {r["metric"]: float(r["value"]) for r in chosen_rows}
    need(len(metrics) == len(chosen_rows) and all(k in metrics and math.isfinite(metrics[k]) for k in METRICS),
         "Selected validation metrics missing or duplicated")
    return metrics


def audit_training(run, seed, variant, evidence):
    names = ("identity.json", "resolved_config.json", "results.json", "COMPLETE.json", "FIT_COMPLETE.json",
             "initial_state_identity.json", "checkpoint_reload.json", "phase_integrity.json", "input_identities.json")
    identity, config, result, complete, fit, initial, reload, phases, inputs = [evidence.document(run / n) for n in names]
    need(identity["seed"] == config["rand_seed"] == seed and identity["variant"] == result["variant"] == variant
         and identity["run_id"] == result["run_id"] == complete["run_id"] == config["run_id"] == run.name,
         "Training identity differs")
    need(identity["engine_commit"] == ENGINE and complete["mode"] == "formal"
         and complete["result_file"] == "results.json" and complete["result_sha256"] == sha(run / "results.json")
         and result["fit"] == fit, "Training completion/fit binding differs")
    need(fit["completed_epochs"] == 45 and fit["training_stop_reason"] == "max_epochs"
         and fit["cumulative_recommender_optimizer_updates"] == 3640, "Incomplete recommender training budget")
    progress = fit["joint_progress"]
    need(progress["tokenizer_epochs_completed"] == 10 and progress["tokenizer_optimizer_updates"] == 60
         and progress["sid_refresh_count"] == 10, "Incomplete tokenizer training budget")
    expected = ["recommender"] * 5 + [p for _ in range(10) for p in ("tokenizer", "recommender", "recommender", "recommender")]
    need(len(phases) == 45 and [r["epoch"] for r in phases] == list(range(1, 46))
         and [r["phase"] for r in phases] == expected, "Phase schedule differs")
    for phase in phases:
        need(phase["optimizer_updates"] == (104 if phase["phase"] == "recommender" else 6)
             and phase["frozen_parameters_unchanged"] is True and phase["all_active_gradients_finite"] is True,
             "Phase update/freeze/gradient check failed")
    need(result["phase_integrity_sha256"] == sha(run / "phase_integrity.json"), "Phase records not bound to result")
    need(config["patience"] == (5 if seed == 2024 else 0) and config["epochs"] == 45
         and config["val_metric"] == "weighted_score" and config.get("min_delta", 0) == 0
         and config["eval_start_epoch"] == config["eval_interval"] == 5
         and config["vectorized_beam_search"]["val"] == {"beam_act": 32, "beam_max": 32}
         and config["item_local_attention"]["enabled"] == (variant == "item_local")
         and config["crosspse_mixer_enabled"] is False and config["joint_gradient_mode"] == "diger_st"
         and config["eval_expand_sid_to_items"] is False, "Training/selection protocol changed")
    if seed != 2024:
        need(result["status"] == complete["status"] == "formal_training_completed_validation_pending"
             and result["test_data_evaluated"] is False and result["final_validation_pending"] == [32, 256]
             and not (run / "TEST_STARTED.json").exists() and not (run / "TEST_RESULT.json").exists(),
             "New campaign unexpectedly evaluated test")
    else:
        need(result["status"] == complete["status"] == "formal_completed", "Historical training is incomplete")
    for key in INITIAL:
        need(isinstance(initial[key], str) and len(initial[key]) == 64, "Missing initial state digest")
        if key in result:
            need(result[key] == initial[key], "Initial state differs from result")
    checkpoint = evidence.bind(run / "training_state/best_model.bin")
    digest = sha(checkpoint)
    need(digest == complete["checkpoint_sha256"] == result["checkpoint_sha256"] == reload["checkpoint_sha256"]
         and reload["best_epoch"] == fit["best_epoch"] and reload["strict_all_tensor_match"] is True
         and reload["mapping_synchronized_without_requantization"] is True
         and reload["matching_committed_SID_sha256"] == result["selected_SID_sha256"], "Best checkpoint/SID differs")
    files = list((run / "diagnostics").rglob("*_evaluation_metrics.csv"))
    need(len(files) == 1, "Expected one original selection CSV")
    selected = selection(evidence.bind(files[0]), fit)
    artifacts = {r["path"]: r["sha256"] for r in inputs["artifacts"]}
    need(len(artifacts) == len(inputs["artifacts"]) == 13
         and artifacts == config["training_state_identity"]["inputs"], "Warm-start input identity differs")
    return {"seed": seed, "variant": variant, "run": str(run), "best_epoch": fit["best_epoch"],
            "best_val_score": fit["best_val_score"], "selected_validation": selected,
            "checkpoint_sha256": digest, "selected_SID_sha256": result["selected_SID_sha256"],
            "initial": {k: initial[k] for k in INITIAL}, "inputs": artifacts,
            "engine_sources": identity["engine_source_sha256"], "budget": {"logical_epochs": 45,
            "recommender_epochs": 35, "tokenizer_phases": 10, "recommender_updates": 3640, "tokenizer_updates": 60}}


def recompute_rows(path, expected_count=N_USERS):
    sums = {key: [] for key in METRICS}
    order, targets = [], []
    ranks, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            need(row["index"] == index and row["user"] not in seen, "Per-user order/index is duplicated or incomplete")
            seen.add(row["user"])
            predicted, target = row["predicted_SIDs"], row["target_SID"]
            need(len(predicted) == 10 and all(len(sid) == 4 and all(type(x) is int and 0 <= x < 256 for x in sid)
                 for sid in [target, *predicted]), "Invalid SID prediction geometry")
            rank = predicted.index(target) + 1 if target in predicted else None
            unique = len({tuple(sid) for sid in predicted})
            need(row["target_rank"] == rank and row["top10_unique_count"] == unique
                 and row["inferred_repeat_fill_slots"] == 10 - unique, "Per-user diagnostic disagrees with predictions")
            calculated = {"duplicate@10": 1 - unique / 10}
            for k in (5, 10):
                hit = rank is not None and rank <= k
                calculated[f"hit_rate@{k}"] = float(hit)
                calculated[f"ndcg@{k}"] = 1 / math.log2(rank + 1) if hit else 0.0
            calculated["weighted_score"] = .8 * calculated["ndcg@10"] + .2 * calculated["hit_rate@10"]
            for key, expected in calculated.items():
                actual = row["metrics"][key]
                # Torch's float32 log/reciprocal and weighted sum differ slightly
                # from stdlib float64; this is numeric validation, not significance.
                need(type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= 1e-7,
                     f"Per-user metric does not follow predictions: {key}, user {index}")
                sums[key].append(actual)
            for k in (5, 10):
                need(row["metrics"][f"recall@{k}"] == row["metrics"][f"hit_rate@{k}"], "Recall/HR identity differs")
            order.append([index, row["user"], row["target_item_id"]])
            targets.append(target)
            ranks.append(rank)
    need(len(order) == expected_count, f"Per-user evaluation count differs: {len(order)} != {expected_count}")
    digest = hashlib.sha256()
    for row in order:
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return {"count": len(order), "metrics": {k: math.fsum(v) / len(order) for k, v in sums.items()},
            "hits10": sum(r is not None for r in ranks), "order_sha256": digest.hexdigest(),
            "targets_sha256": json_sha(targets), "order": order, "ranks": ranks}


def audit_cell(files, stored, training, beam, local_root, remote_root, evidence):
    paths = {key: evidence.mapped(value, local_root, remote_root) for key, value in files.items()}
    metrics = read(paths["metrics"])
    need(metrics == stored and metrics["variant"] == training["variant"] and metrics["count"] == N_USERS
         and metrics["split"] == "val" and metrics["beam_act"] == metrics["beam_max"] == beam
         and metrics["checkpoint_sha256"] == training["checkpoint_sha256"]
         and metrics["selected_SID_sha256"] == training["selected_SID_sha256"]
         and metrics["model_state_unchanged"] is True and metrics["selected_SID_unchanged"] is True
         and metrics["test_data_evaluated"] is False
         and metrics["model_state_before_sha256"] == metrics["model_state_after_sha256"], "Cell state/protocol differs")
    gradients = "gradients_created" if training["seed"] == 2024 else "training_gradients_created"
    need(metrics[gradients] is False, "Evaluation produced gradients")
    if training["seed"] != 2024:
        need(metrics["seed"] == training["seed"] and metrics["best_epoch"] == training["best_epoch"], "Cell reselected best/seed")
    else:
        need(metrics["partial_diagnostic"] is False, "Historical grid is partial")
    calculated = recompute_rows(paths["per_user"], N_USERS)
    need(calculated["order_sha256"] == metrics["user_target_order_sha256"], "User order digest differs")
    need(all(abs(value - metrics["metrics"][key]) <= 1e-12 for key, value in calculated["metrics"].items()),
         "Aggregate metrics differ from raw per-user results")
    if beam == 32:
        need(metrics["original_val32_reproduction"]["status"] == "matched"
             and all(abs(metrics["metrics"][key] - value) <= 1e-9 for key, value in training["selected_validation"].items()),
             "Original selected validation32 metrics did not reproduce")
    return {"seed": training["seed"], "variant": training["variant"], "beam": beam,
            "best_epoch": training["best_epoch"], "checkpoint_sha256": training["checkpoint_sha256"],
            "selected_SID_sha256": training["selected_SID_sha256"],
            "history_sha256": metrics["validation_user_history_sha256"], **calculated}


def read_new(run, training, evidence):
    root = run / "validation"
    plan, complete = [evidence.document(root / n) for n in ("EVALUATION_PLAN.json", "EVALUATION_COMPLETE.json")]
    check_plan(plan)
    need(complete["status"] == "selected_validation_completed" and complete["identity_sha256"] == plan["identity_sha256"]
         and complete["seed"] == training["seed"] and complete["variant"] == training["variant"]
         and complete["best_epoch"] == training["best_epoch"] and complete["test_data_evaluated"] is False
         and complete["training_performed"] is False and set(complete["cells"]) == {"beam32", "beam256"},
         "New validation completion differs")
    need(plan["checkpoint"]["sha256"] == training["checkpoint_sha256"]
         and plan["selected_SID_sha256"] == training["selected_SID_sha256"] and plan["reselection_allowed"] is False,
         "Plan changed selected model/SID")
    for item in plan["original_files"].values():
        evidence.mapped(item, run, plan["run"])
    evidence.mapped(plan["checkpoint"], run, plan["run"])
    cells = []
    for beam in BEAMS:
        entry = complete["cells"][f"beam{beam}"]
        receipt = read(evidence.mapped(entry["receipt"], run, plan["run"]))
        need(receipt["status"] == "cell_completed" and receipt["identity_sha256"] == plan["identity_sha256"]
             and receipt["beam"] == beam and entry["files"] == receipt["files"], "Cell receipt binding differs")
        start = read(evidence.mapped(receipt["started_receipt"], run, plan["run"]))
        need(start["identity_sha256"] == plan["identity_sha256"] and start["beam"] == beam, "Cell start identity differs")
        cells.append(audit_cell(receipt["files"], entry["metrics"], training, beam, run, plan["run"], evidence))
    return cells


def read_old(root, training_root, evidence):
    root = Path(root).resolve()
    if not (root / "GRID_PLAN.json").exists():
        root /= "results"
    plan, complete = [evidence.document(root / n) for n in ("GRID_PLAN.json", "COMPLETE.json")]
    check_plan(plan)
    need(complete["status"] == "validation_beam_grid_completed" and plan["max_eval_batches"] is None
         and complete["grid_identity_sha256"] == plan["identity_sha256"]
         and complete["test_data_evaluated"] is False and complete["training_performed"] is False
         and plan["sources"]["engine"]["commit"] == ENGINE, "Historical validation grid differs")
    trainings, cells = {}, []
    for variant in VARIANTS:
        run = unique_run(training_root, 2024, variant)
        training = audit_training(run, 2024, variant, evidence)
        arm = plan["arms"][variant]
        for item in arm["files"].values():
            evidence.mapped(item, run, arm["run"])
        evidence.mapped(arm["checkpoint"], run, arm["run"])
        need(arm["initial_base_state_sha256"] == training["initial"]["initial_base_state_sha256"]
             and arm["initial_rng_sha256"] == training["initial"]["initial_rng_sha256"]
             and training["best_epoch"] == 25 and arm["selected_SID_sha256"] == training["selected_SID_sha256"],
             "Historical grid is not the accepted best25 pair")
        trainings[(2024, variant)] = training
        for beam in BEAMS:
            name = f"{variant}_beam{beam}"
            entry = complete["cells"][name]
            remote_root = str(PurePosixPath(entry["receipt"]["path"]).parent.parent)
            receipt = read(evidence.mapped(entry["receipt"], root, remote_root))
            need(receipt["status"] == "cell_completed" and receipt["cell"] == name
                 and receipt["grid_identity_sha256"] == plan["identity_sha256"], "Historical cell identity differs")
            cells.append(audit_cell(receipt["files"], entry["metrics"], training, beam, root, remote_root, evidence))
    return trainings, cells


def paired_statistics(cells, seeds):
    lookup = {(r["seed"], r["variant"], r["beam"]): r for r in cells}
    output = {}
    for beam in BEAMS:
        output[str(beam)] = {}
        for metric in ("hit_rate@10", "ndcg@10"):
            a = [lookup[(s, "control", beam)]["metrics"][metric] for s in seeds]
            b = [lookup[(s, "item_local", beam)]["metrics"][metric] for s in seeds]
            delta = [right - left for left, right in zip(a, b)]
            describe = lambda values: {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
            output[str(beam)][metric] = {"seeds": list(seeds), "control": describe(a), "item_local": describe(b),
                "paired_delta": {**describe(delta), "by_seed": dict(zip(map(str, seeds), delta)),
                                 "positive_seeds": sum(value > 0 for value in delta)}}
    return output


def build_summary(new_root, old_grid_root, old_training_root):
    evidence = Evidence()
    trainings, cells = {}, []
    # Fail before historical scans if any of the four new completions is missing.
    for seed in (2025, 2026):
        for variant in VARIANTS:
            run = unique_run(new_root, seed, variant)
            need((run / "validation/EVALUATION_COMPLETE.json").is_file(), f"Missing complete new evaluation: {run}")
            training = audit_training(run, seed, variant, evidence)
            trainings[(seed, variant)] = training
            cells.extend(read_new(run, training, evidence))
    old_training, old_cells = read_old(old_grid_root, old_training_root, evidence)
    trainings.update(old_training); cells.extend(old_cells)
    for seed in SEEDS:
        a, b = [trainings[(seed, variant)] for variant in VARIANTS]
        need(a["initial"] == b["initial"] and a["inputs"] == b["inputs"] and a["engine_sources"] == b["engine_sources"],
             f"Paired initialization/input/source differs for seed {seed}")
    need(len({json_sha(t["inputs"]) for t in trainings.values()}) == 1
         and len({json_sha(t["engine_sources"]) for t in trainings.values()}) == 1,
         "Seeds used different warm-start assets or engines")
    need(all(c["order"] == cells[0]["order"] for c in cells)
         and len({c["history_sha256"] for c in cells}) == 1, "Cells used different validation users/targets/history")
    for seed in SEEDS:
        for variant in VARIANTS:
            pair = [c for c in cells if c["seed"] == seed and c["variant"] == variant]
            need(len(pair) == 2 and pair[0]["targets_sha256"] == pair[1]["targets_sha256"], "Beam widths used different saved target SID")
    paired = []
    for seed in SEEDS:
        for beam in BEAMS:
            a, b = [next(c for c in cells if c["seed"] == seed and c["variant"] == v and c["beam"] == beam) for v in VARIANTS]
            gains = sum(x is None and y is not None for x, y in zip(a["ranks"], b["ranks"]))
            losses = sum(x is not None and y is None for x, y in zip(a["ranks"], b["ranks"]))
            need(gains - losses == b["hits10"] - a["hits10"], "Paired net hit accounting differs")
            paired.append({"seed": seed, "beam": beam, "delta": {k: b["metrics"][k] - a["metrics"][k] for k in METRICS},
                           "new_hit_users": gains, "lost_hit_users": losses, "net_hit_users": gains - losses})
    return {"schema": "stage07_multiseed_offline_summary_v1", "status": "ALL_RESULTS_VERIFIED",
            "count_per_cell": N_USERS, "cell_count": 12, "new_training_seeds": [2025, 2026], "historical_seed": 2024,
            "trainings": [trainings[key] for key in sorted(trainings)],
            "cells": [{k: v for k, v in c.items() if k not in ("order", "ranks")} for c in sorted(cells, key=lambda c: (c["seed"], c["beam"], c["variant"]))],
            "paired": paired, "new_seeds_statistics": paired_statistics(cells, (2025, 2026)),
            "all_three_descriptive_statistics": paired_statistics(cells, SEEDS),
            "verified_files": list(evidence.files.values()),
            "limitations": ["Same validation set and warm-start assets; not independently pretrained end-to-end seeds",
                            "New seeds fixed 45 epochs; historical seed patience5 also actually completed45",
                            "Cross-seed base hashes may differ because inactive SIA adapter is randomly initialized",
                            "SID-level metrics; no item-level, test generalization, serving latency or significance claim",
                            "Raw input contents and model tensors are not reloaded; local checkpoint bytes and saved receipts are SHA-verified"],
            "metric_recomputation": "Recompute first matching SID rank, HR/NDCG/weighted-score/duplicates per user; float32 numeric tolerance 1e-7, aggregate stored-row mean tolerance1e-12"}


def report(summary):
    lookup = {(r["seed"], r["variant"], r["beam"]): r for r in summary["cells"]}
    lines = ["# 商品内交互：多 seed 验证集结果", "", "A 为原版，B 为商品内交互。四组新增训练和两个历史模型均已核对完整训练量；每个模型使用原 val32 选出的 best，同时评估 beam32/256，没有重新选模。每格覆盖 22,363 个相同验证用户。", "",
             "| seed | beam | A HR@10 | B HR@10 | B−A | A NDCG@10 | B NDCG@10 | B−A |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["paired"]:
        seed, beam = row["seed"], row["beam"]
        a, b = [lookup[(seed, v, beam)]["metrics"] for v in VARIANTS]
        lines.append(f"| {seed}{'（历史）' if seed == 2024 else ''} | {beam} | {a['hit_rate@10']:.6f} | {b['hit_rate@10']:.6f} | {row['delta']['hit_rate@10']:+.6f} | {a['ndcg@10']:.6f} | {b['ndcg@10']:.6f} | {row['delta']['ndcg@10']:+.6f} |")
    lines += ["", "下表按训练 seed 统计配对增益 B−A，± 后是样本标准差。新增两 seed 是本轮主要证据；三 seed 合并只作描述性参考。", "",
              "| 范围 | beam | HR@10 增益均值 ± SD | NDCG@10 增益均值 ± SD |", "|---|---:|---:|---:|"]
    for label, key in (("新增2025/2026", "new_seeds_statistics"), ("含历史的三seed", "all_three_descriptive_statistics")):
        for beam in BEAMS:
            h, n = [summary[key][str(beam)][m]["paired_delta"] for m in ("hit_rate@10", "ndcg@10")]
            lines.append(f"| {label} | {beam} | {h['mean']:+.6f} ± {h['sample_sd']:.6f} | {n['mean']:+.6f} ± {n['sample_sd']:.6f} |")
    lines += ["", "本轮新增两 seed 的方向："]
    for beam in BEAMS:
        values = summary["new_seeds_statistics"][str(beam)]
        lines.append(f"- beam{beam}：HR@10 在 {values['hit_rate@10']['paired_delta']['positive_seeds']}/2 个 seed 上提高，NDCG@10 在 {values['ndcg@10']['paired_delta']['positive_seeds']}/2 个 seed 上提高。")
    lines += ["", "同 seed 的 A/B 初始共同参数、SID、RNG 和13份输入一致，均完成35个推荐器训练轮、60次 tokenizer 更新。跨 seed 的整体 base SHA 可能受未使用的 SIA 辅助层随机初始化影响，不能据此认定有效共同起点不同。这些是同一预训练起点后的训练 seed，不能称为完全独立的端到端重训。", "",
              "2024 属于历史实验，配置仍是 patience=5，但实际也完成45轮；新增两 seed 显式关闭早停。三 seed 尚不足以作强稳定性或显著性结论，22,363 个用户也不是22,363次独立训练。结果仍是同一验证集上的 SID 级指标，本轮未评估测试集，也没有测线上延迟。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True, help="Local verified delivery containing exactly four new run directories")
    parser.add_argument("--old-grid-root", required=True, help="Local historical results directory or its verified parent")
    parser.add_argument("--old-training-root", required=True, help="Local historical verified training delivery")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = build_summary(args.new_root, args.old_grid_root, args.old_training_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    print("ALL_RESULTS_VERIFIED: SUMMARY.json and REPORT.md written")


if __name__ == "__main__":
    main()
