"""Verify a downloaded four-cell validation grid and write a local summary.

Usage: python summarize_results.py --verified PATH_TO_VERIFIED --output REPORT_DIR
Only the supplied local archive is read; no model, GPU or external package is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

COUNT = 22363
ANCHOR = "/beam_validation_20261021_r1/"
VARIANTS = ("control", "item_local")
CELLS = tuple(f"{variant}_beam{beam}" for variant in VARIANTS for beam in (32, 256))
METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "duplicate@10", "legal@10", "weighted_score")
LOCKS = {
    "control": ("55960f52906b0c4972a90b0ad06c078ef12fceac10fa4400b113e41d9a5c1f5f", "a5483ad7d06a0076e035e5746f5c5d85ecc92eac511fa05e87ab91c257c1d0c1"),
    "item_local": ("f093c22e4e0cb354870b467e41f4ebab96d90047daf9d0e3d33991e3108b348b", "075bf705f54deaba794324193de95e6d7f0ae3f985166b39652f90d15859b6e2"),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound_file(root, record):
    remote = record["path"]
    need(remote.count(ANCHOR) == 1, f"Unexpected remote record path: {remote}")
    relative = remote.split(ANCHOR, 1)[1]
    path = (root / relative).resolve()
    need(path.is_relative_to(root) and path.is_file(), f"Missing or unsafe local file: {relative}")
    need(path.stat().st_size == record["size_bytes"] and sha(path) == record["sha256"], f"Size/SHA mismatch: {relative}")
    return path


def close(actual, expected, label, tolerance=1e-12):
    need(math.isfinite(float(actual)) and math.isfinite(float(expected))
         and abs(actual - expected) <= tolerance, f"Value mismatch: {label}: {actual} versus {expected}")


def bucket(rank):
    return "miss" if rank is None else "top5" if rank <= 5 else "rank6_10"


def user_rows(path, cell_metrics):
    rows, digest = [], hashlib.sha256()
    values = {key: [] for key in METRICS}
    unique = legal_unique = repeats = no_legal = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            need(row["index"] == index, f"User index/order mismatch in {path}")
            predictions = [tuple(item) for item in row["predicted_SIDs"]]
            target = tuple(row["target_SID"])
            need(len(predictions) == 10 and len(target) == 4 and all(len(item) == 4 for item in predictions), "Wrong prediction geometry")
            rank = predictions.index(target) + 1 if target in predictions else None
            need(rank == row["target_rank"], "Saved target rank does not match predictions")
            n_unique = len(set(predictions))
            n_legal = row["top10_unique_legal_count"]
            need(n_unique == row["top10_unique_count"] and 0 <= n_legal <= n_unique
                 and row["inferred_repeat_fill_slots"] == 10 - n_unique
                 and row["no_legal_top10_fallback"] == (n_legal == 0), "Per-user candidate diagnostics differ")
            for k in (5, 10):
                hit = int(rank is not None and rank <= k)
                close(row["metrics"][f"hit_rate@{k}"], hit, "per-user hit")
                close(row["metrics"][f"ndcg@{k}"], 1 / math.log2(rank + 1) if hit else 0,
                      "per-user NDCG versus rank", tolerance=1e-7)
            close(row["metrics"]["duplicate@10"], 1 - n_unique / 10, "per-user duplicate ratio", tolerance=5e-8)
            for key in METRICS:
                need(math.isfinite(row["metrics"][key]), f"Nonfinite per-user metric: {key}")
                values[key].append(row["metrics"][key])
            identity = [index, row["user"], row["target_item_id"]]
            digest.update(json.dumps(identity, separators=(",", ":")).encode() + b"\n")
            rows.append({"identity": identity, "rank": rank, "target_SID": target})
            unique += n_unique
            legal_unique += n_legal
            repeats += 10 - n_unique
            no_legal += int(n_legal == 0)
    need(len(rows) == COUNT and len({row["identity"][1] for row in rows}) == COUNT, "Full unique validation-user count differs")
    need(digest.hexdigest() == cell_metrics["user_target_order_sha256"], "Per-user order/target SHA differs")
    for key in METRICS:
        close(math.fsum(values[key]) / COUNT, cell_metrics["metrics"][key], f"Aggregate {key}")
    close(unique / COUNT, cell_metrics["mean_top10_unique_count"], "Unique Top10 count")
    close(legal_unique / COUNT, cell_metrics["mean_top10_unique_legal_count"], "Unique legal Top10 count")
    need(repeats == cell_metrics["inferred_repeat_fill_slots"] and no_legal == cell_metrics["no_legal_top10_fallback_users"], "Aggregate candidate counts differ")
    return rows


def paired(left, right, left_name, right_name):
    need([r["identity"] for r in left] == [r["identity"] for r in right], "Paired users/target items differ")
    transitions = {a: {b: 0 for b in ("miss", "top5", "rank6_10")} for a in ("miss", "top5", "rank6_10")}
    answer = {"from": left_name, "to": right_name, "count": COUNT}
    for a, b in zip(left, right):
        transitions[bucket(a["rank"])][bucket(b["rank"])] += 1
    for k in (5, 10):
        counts = dict(new_hits=0, lost_hits=0, both_hit=0, both_miss=0)
        for a, b in zip(left, right):
            ah = a["rank"] is not None and a["rank"] <= k
            bh = b["rank"] is not None and b["rank"] <= k
            key = "both_hit" if ah and bh else "lost_hits" if ah else "new_hits" if bh else "both_miss"
            counts[key] += 1
        counts["net_hits"] = counts["new_hits"] - counts["lost_hits"]
        counts["absolute_hit_rate_change"] = counts["net_hits"] / COUNT
        answer[f"top{k}"] = counts
    answer["rank_bucket_transition_counts"] = transitions
    answer["from_rank_counts"] = {key: sum(value.values()) for key, value in transitions.items()}
    answer["to_rank_counts"] = {key: sum(value[key] for value in transitions.values()) for key in transitions}
    return answer


def deltas(left, right):
    return {key: {"absolute": right[key] - left[key], "relative_percent": (right[key] / left[key] - 1) * 100 if left[key] else None}
            for key in METRICS}


def summarize(root):
    results = root / "results"
    complete_path, plan_path, preflight_path = (results / name for name in ("COMPLETE.json", "GRID_PLAN.json", "GRID_PREFLIGHT.json"))
    complete, plan, preflight = map(read, (complete_path, plan_path, preflight_path))
    identity_payload = {key: value for key, value in plan.items() if key != "identity_sha256"}
    need(json_sha(identity_payload) == plan["identity_sha256"] == complete["grid_identity_sha256"] == preflight["grid_identity_sha256"], "Grid identities differ")
    need(complete["status"] == "validation_beam_grid_completed" and set(complete["cells"]) == set(CELLS), "Full four-cell completion missing")
    need(complete["original_artifacts_unchanged"] is True and complete["training_performed"] is False
         and complete["test_data_evaluated"] is False and plan["max_eval_batches"] is None
         and plan["validation_count"] == COUNT and plan["split"] == "val", "Wrong scope or incomplete validation")
    need(preflight["status"] == "CPU_PREFLIGHT_PASSED_NO_INFERENCE" and preflight["cuda_initialized"] is False, "Preflight did not pass")
    cells, users = {}, {}
    for name in CELLS:
        variant, beam_text = name.rsplit("_beam", 1)
        beam = int(beam_text)
        receipt = read(bound_file(root, complete["cells"][name]["receipt"]))
        need(receipt["cell"] == name and receipt["status"] == "cell_completed" and receipt["grid_identity_sha256"] == plan["identity_sha256"], "Cell receipt identity differs")
        files = {key: bound_file(root, value) for key, value in receipt["files"].items()}
        metrics = read(files["metrics"])
        need(metrics == complete["cells"][name]["metrics"], "Embedded and stored cell metrics differ")
        need(metrics["cell"] == name and metrics["variant"] == variant and metrics["beam_act"] == metrics["beam_max"] == beam
             and metrics["return_count"] == 10 and metrics["count"] == COUNT and metrics["split"] == "val"
             and metrics["test_data_evaluated"] is False and metrics["partial_diagnostic"] is False, "Cell evaluation scope differs")
        need(metrics["model_state_unchanged"] is True and metrics["selected_SID_unchanged"] is True
             and metrics["gradients_created"] is False and metrics["model_state_before_sha256"] == metrics["model_state_after_sha256"]
             == preflight["best_states"][variant]["model_state_sha256"], "Model state changed or original state was not loaded")
        need((metrics["checkpoint_sha256"], metrics["selected_SID_sha256"]) == LOCKS[variant], "Selected best25 checkpoint or saved SID differs")
        need(metrics["validation_user_history_sha256"] == plan["validation_user_history_sha256"], "Validation history identity differs")
        need(math.isfinite(metrics["seconds"]) and metrics["seconds"] > 0, "Invalid validation elapsed time")
        if beam == 32:
            reproduction = metrics["original_val32_reproduction"]
            need(reproduction["status"] == "matched", "Original beam32 validation was not reproduced")
            for key, expected in plan["arms"][variant]["original_best25_validation"].items():
                close(metrics["metrics"][key], expected, f"{name} original best25 {key}", tolerance=1e-9)
                close(reproduction["metric_differences"][key], metrics["metrics"][key] - expected, "Reproduction delta")
        users[name] = user_rows(files["per_user"], metrics)
        cells[name] = metrics
    need(len({row["user_target_order_sha256"] for row in cells.values()}) == 1, "Cross-cell user/target SHA differs")
    comparisons, effects = {}, {}
    for beam in (32, 256):
        a, b = f"control_beam{beam}", f"item_local_beam{beam}"
        comparisons[str(beam)] = {"metric_deltas_B_minus_A": deltas(cells[a]["metrics"], cells[b]["metrics"]),
                                 "paired_users": paired(users[a], users[b], a, b)}
    for variant in VARIANTS:
        a, b = f"{variant}_beam32", f"{variant}_beam256"
        need([row["target_SID"] for row in users[a]] == [row["target_SID"] for row in users[b]], "Within-model target SID changed with beam")
        effects[variant] = {"metric_deltas_256_minus_32": deltas(cells[a]["metrics"], cells[b]["metrics"]),
                            "paired_users": paired(users[a], users[b], a, b),
                            "full_validation_seconds_ratio_256_over_32": cells[b]["seconds"] / cells[a]["seconds"]}
    for beam, comparison in comparisons.items():
        for k in (5, 10):
            close(comparison["paired_users"][f"top{k}"]["absolute_hit_rate_change"], comparison["metric_deltas_B_minus_A"][f"hit_rate@{k}"]["absolute"], "Paired hit delta")
    interaction = {key: comparisons["256"]["metric_deltas_B_minus_A"][key]["absolute"] - comparisons["32"]["metric_deltas_B_minus_A"][key]["absolute"] for key in METRICS}
    return {"status": "LOCAL_GRID_VERIFICATION_PASSED", "count_per_cell": COUNT,
            "input_files": {str(path.relative_to(root)): sha(path) for path in (complete_path, plan_path, preflight_path)},
            "grid_identity_sha256": plan["identity_sha256"], "cells": cells, "B_minus_A_by_beam": comparisons,
            "within_model_beam_effect": effects, "absolute_gain_change_delta256_minus_delta32": interaction,
            "checks": {"four_complete_cells": True, "all_bound_files_SHA_match": True, "per_user_metrics_reconstructed": True,
                       "same_user_target_order": True, "original_beam32_reproduced": True, "original_model_SID_unchanged": True,
                       "source_and_original_bytes_unchanged_remote_receipt": True},
            "limitations": ["Fixed best25 checkpoints selected on this validation set; one training seed.",
                            "SID-level hit metrics, not collision-resolved item-level metrics.",
                            "Timing includes metrics and per-user JSON writes; it is full validation time, not serving latency.",
                            "The beam comparison cannot by itself explain the entire previous validation/test gap or establish significance.",
                            "Local checks verify downloaded evidence; remote original-byte immutability is attested by the completion receipt."]}


def markdown(summary):
    lines = ["# 同一验证集 beam32 / beam256 对照（结果草案）", "",
             "A 为交替联合训练原版，B 为加入商品内注意力的版本。固定各自第25轮 best 权重及保存的 SID；每格均评估同一批22,363名用户，未重新训练、未评估测试集。", "",
             "| 模型 | beam | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Top10重复率 | 平均不重复合法候选数 | 完整验证耗时（秒） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in CELLS:
        cell = summary["cells"][name]
        metric = cell["metrics"]
        label = "A 原版" if cell["variant"] == "control" else "B 商品内交互"
        lines.append(f"| {label} | {cell['beam_act']} | {metric['hit_rate@5']:.6f} | {metric['hit_rate@10']:.6f} | {metric['ndcg@5']:.6f} | {metric['ndcg@10']:.6f} | {metric['duplicate@10']:.2%} | {cell['mean_top10_unique_legal_count']:.3f} | {cell['seconds']:.1f} |")
    lines += ["", "以下差值均为 B−A；相对变化的分母为同一 beam 下的 A。", "",
              "| beam | HR@10绝对差 | HR@10相对变化 | NDCG@10绝对差 | NDCG@10相对变化 |",
              "|---|---:|---:|---:|---:|"]
    def relative(value):
        return "—" if value is None else f"{value:+.2f}%"
    for beam in ("32", "256"):
        delta = summary["B_minus_A_by_beam"][beam]["metric_deltas_B_minus_A"]
        hr, ndcg = delta["hit_rate@10"], delta["ndcg@10"]
        lines.append(f"| {beam} | {hr['absolute']:+.6f} | {relative(hr['relative_percent'])} | {ndcg['absolute']:+.6f} | {relative(ndcg['relative_percent'])} |")
    gain = summary["absolute_gain_change_delta256_minus_delta32"]
    lines += ["", f"搜索宽度由32变为256后，B−A的绝对差变化：HR@10 {gain['hit_rate@10']:+.6f}，NDCG@10 {gain['ndcg@10']:+.6f}。负值表示B−A的差值变小，正值表示变大；这只是本次固定权重的数值比较。", "",
              "| beam | A前5命中人数 | B前5命中人数 | A第6–10名命中人数 | B第6–10名命中人数 | B新增Top10命中用户 | B丢失Top10命中用户 | Top10净增 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for beam in ("32", "256"):
        pair = summary["B_minus_A_by_beam"][beam]["paired_users"]
        a, b, top = pair["from_rank_counts"], pair["to_rank_counts"], pair["top10"]
        lines.append(f"| {beam} | {a['top5']} | {b['top5']} | {a['rank6_10']} | {b['rank6_10']} | {top['new_hits']} | {top['lost_hits']} | {top['net_hits']:+d} |")
    lines += ["", "新增/丢失按同一用户及真实目标商品配对统计；两组各用自己的SID标签，因此仍是SID级命中。完整的前5/第6–10名/未命中转移计数，以及每个模型从beam32增至256的配对结果，见SUMMARY.json。", "",
              "本地校验通过：四格文件SHA、用户顺序、逐用户指标与汇总一致；beam32复现原第25轮验证结果；权重和SID摘要保持不变。远程原文件不变的检查由下载的完成回执记录。", "",
              "耗时包含指标计算和逐用户结果写盘，不等于线上单请求推理延迟。这里只诊断搜索宽度是否影响两组差距；固定在验证集选出的权重、单个训练种子，不能据此认定统计显著，也不能把此前验证集与测试集的全部差距归因于beam。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.verified.resolve(), args.output.resolve()
    summary = summarize(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "COMPARISON.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
