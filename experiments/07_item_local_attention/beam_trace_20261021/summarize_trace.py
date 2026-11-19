"""Verify downloaded Stage07 validation traces and summarize observed search fates.

Only the Python standard library is required. This tool never invokes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


CELLS = tuple(f"{arm}_beam{beam}" for arm in ("control", "item_local") for beam in (32, 256))
FATES = ("pruned_round_1", "pruned_round_2", "pruned_round_3", "final_argmax", "rank_cutoff", "topk_hit")
FATE_CN = dict(zip(FATES, ("第1轮丢失目标路径", "第2轮丢失目标路径", "第3轮丢失目标路径", "最后一位argmax未选中", "已生成但排在Top10外", "Top10命中")))
N = 22363


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_bound_file(root, cell, record):
    parts = PurePosixPath(record["path"].replace("\\", "/")).parts
    need(parts.count(cell) == 1, f"Cannot unambiguously remap {record['path']}")
    relative = Path(*parts[parts.index(cell):])
    path = (root / relative).resolve()
    need(path.is_relative_to(root) and path.is_file(), f"Missing local evidence {path}")
    need(path.stat().st_size == record["size_bytes"] and sha(path) == record["sha256"], f"SHA/size mismatch {path}")
    return path


def rank_bucket(rank):
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 32:
        return "11-32"
    return "33-256"


def validate_row(row, beam):
    trace = row["trace"]
    rounds, final = trace["rounds"], trace["final_candidates"]
    need(trace["target_sid"] == row["target_SID"] and trace["target_sid_in_catalog"], "Trace target/catalog mismatch")
    need([r["round"] for r in rounds] == [1, 2, 3, 4], "Trace round sequence differs")
    for round_index, r in enumerate(rounds):
        need(r["retained_path_count"] == beam, "Retained beam width differs")
        need(0 <= r["gt_compatible_unique_after"] <= r["gt_compatible_after"] <= r["catalog_completable_paths"] <= beam, "Invalid compatible-path counts")
        need(0 <= r["catalog_completable_unique_states"] <= r["catalog_completable_paths"], "Invalid legal unique counts")
        need(1 <= r["retained_unique_states"] <= beam and r["retained_duplicate_paths"] == beam - r["retained_unique_states"], "Invalid duplicate counts")
        need(r["gt_compatible_unique_after"] <= r["catalog_completable_unique_states"] <= r["retained_unique_states"], "Unique-state containment differs")
        need(r["gt_compatible_after"] <= r["gt_compatible_before"] <= r["candidate_paths_before"], "Before/after target count differs")
        remaining = 4 - round_index
        expected_gt_before = 4 if round_index == 0 else rounds[round_index - 1]["gt_compatible_after"] * remaining
        expected_candidates = 1024 if round_index == 0 else beam * remaining * 256
        need(r["gt_compatible_before"] == expected_gt_before and r["candidate_paths_before"] == expected_candidates, "Candidate expansion count differs")
        need(r["gt_compatible_unique_after"] <= math.comb(4, round_index + 1), "Too many unique target-compatible partial states")
    first_lost = next((r["round"] for r in rounds if not r["gt_compatible_after"]), None)
    need(trace["first_gt_extinction_round"] == first_lost, "First extinction differs")
    if first_lost:
        need(all(not r["gt_compatible_after"] for r in rounds[first_lost - 1:]), "Extinct target path reappeared")
    need(final["path_count"] == beam and final["target_path_count"] == rounds[-1]["gt_compatible_after"], "Final target count differs")
    need(final["target_generated"] == bool(final["target_path_count"]), "Final target presence differs")
    need(final["legal_unique_count"] == rounds[-1]["catalog_completable_unique_states"] and final["legal_path_count"] == rounds[-1]["catalog_completable_paths"], "Final legal-state count differs")
    need(final["unique_count"] == rounds[-1]["retained_unique_states"] and final["top_k"] == 10, "Final candidate geometry differs")
    need(min(10, final["legal_unique_count"]) == row["top10_unique_legal_count"], "Final candidate/output diversity differs")
    need(final["no_legal_candidates"] == (final["legal_path_count"] == 0) == row["no_legal_top10_fallback"], "Fallback accounting differs")
    target = row["target_SID"]
    actual_rank = row["predicted_SIDs"].index(target) + 1 if target in row["predicted_SIDs"] else None
    need(row["target_rank"] == actual_rank, "Output rank differs")
    legal_rank = final["target_legal_unique_rank"]
    need((legal_rank is not None) == final["target_generated"], "Generated legal target lacks rank")
    expected_fate = (f"pruned_round_{first_lost}" if first_lost and first_lost < 4 else
                     "final_argmax" if first_lost == 4 else
                     "rank_cutoff" if legal_rank > 10 else "topk_hit")
    need(trace["fate"] == expected_fate and trace["fate"] in FATES, "Fate classification differs")
    need((trace["fate"] == "topk_hit") == (actual_rank is not None), "Fate/output hit differs")
    if actual_rank is not None:
        need(legal_rank == actual_rank, "Trace and output target rank differ")
    need(float(row["metrics"]["hit_rate@10"]) == float(actual_rank is not None), "Stored HR differs")
    expected_ndcg = 1 / math.log2(actual_rank + 1) if actual_rank else 0.0
    need(abs(row["metrics"]["ndcg@10"] - expected_ndcg) <= 1e-7, "Stored NDCG differs")
    branches = trace["gt_compatible_final_branches"]
    need(len(branches) == rounds[2]["gt_compatible_after"], "Final compatible branch count differs")
    need(sum(b["selected_is_target"] for b in branches) == final["target_path_count"], "Final argmax target count differs")
    for branch in branches:
        need(1 <= branch["target_rank_strict"] <= branch["target_rank_worst_tie"] <= 256, "Final token rank differs")
        need(not branch["selected_is_target"] or branch["target_rank_strict"] == 1, "Argmax selected a nonmaximal target")


def summarize_cell(root, cell, grid_identity):
    receipt = read(root / cell / "CELL_COMPLETE.json")
    need(receipt["status"] == "cell_completed" and receipt["cell"] == cell and receipt["grid_identity_sha256"] == grid_identity, "Cell completion identity differs")
    files = {name: local_bound_file(root, cell, record) for name, record in receipt["files"].items()}
    metrics = read(files["metrics"])
    beam = int(cell.rsplit("beam", 1)[1])
    need(metrics["count"] == N and metrics["split"] == "val" and metrics["beam_act"] == beam, "Cell evaluation protocol differs")
    need(metrics["model_state_unchanged"] and metrics["selected_SID_unchanged"] and not metrics["gradients_created"] and not metrics["test_data_evaluated"] and not metrics["partial_diagnostic"], "Cell is partial or mutated state")
    need(metrics["original_predictions_reproduction"]["status"] == "matched" and metrics["original_predictions_reproduction"]["users"] == N, "Original prediction parity is missing")
    need(read(files["trace_parity"])["status"] == "matched", "Warmup parity missing")
    fates, round_sums = Counter(), [Counter() for _ in range(4)]
    final_sums, branch_rank_buckets, miss_min_rank_buckets = Counter(), Counter(), Counter()
    min_ranks, branch_ranks, miss_min_ranks = [], [], []
    legal_ranks, compact, metric_totals = [], [], Counter()
    for index, line in enumerate(files["per_user"].open(encoding="utf-8")):
        row = json.loads(line)
        need(row["index"] == index, "Per-user order/index differs")
        validate_row(row, beam)
        trace, final = row["trace"], row["trace"]["final_candidates"]
        fates[trace["fate"]] += 1
        for r, sums in zip(trace["rounds"], round_sums):
            for key, value in r.items():
                if key not in ("round", "selection"):
                    sums[key] += value
            sums["users_with_gt_path"] += int(r["gt_compatible_after"] > 0)
            sums["users_with_dead_catalog_path"] += int(r["catalog_completable_paths"] < beam)
        for key in ("path_count", "unique_count", "legal_path_count", "legal_unique_count", "target_path_count"):
            final_sums[key] += final[key]
        final_sums["target_generated_users"] += int(final["target_generated"])
        final_sums["no_legal_candidates_users"] += int(final["no_legal_candidates"])
        if final["target_legal_unique_rank"] is not None:
            legal_ranks.append(final["target_legal_unique_rank"])
        ranks = [b["target_rank_strict"] for b in trace["gt_compatible_final_branches"]]
        branch_ranks.extend(ranks)
        branch_rank_buckets.update(rank_bucket(rank) for rank in ranks)
        if ranks:
            minimum = min(ranks)
            min_ranks.append(minimum)
            if trace["fate"] == "final_argmax":
                miss_min_ranks.append(minimum)
                miss_min_rank_buckets[rank_bucket(minimum)] += 1
        for key, value in row["metrics"].items():
            metric_totals[key] += value
        compact.append({"index": index, "user": row["user"], "target_item_id": row["target_item_id"],
                        "target_SID": row["target_SID"], "fate": trace["fate"], "hit": int(row["target_rank"] is not None),
                        "ndcg": row["metrics"]["ndcg@10"], "rank": row["target_rank"]})
    need(len(compact) == N, "Per-user file is incomplete")
    for key, total in metric_totals.items():
        need(abs(total / N - metrics["metrics"][key]) <= 1e-9, f"Reconstructed metric differs: {cell} {key}")
    rounds = []
    for index, sums in enumerate(round_sums):
        rounds.append({"round": index + 1, "users_with_gt_path": sums["users_with_gt_path"],
                       "gt_survival_rate": sums["users_with_gt_path"] / N,
                       "mean_gt_compatible_paths": sums["gt_compatible_after"] / N,
                       "mean_gt_compatible_unique_states": sums["gt_compatible_unique_after"] / N,
                       "mean_unique_states": sums["retained_unique_states"] / N,
                       "duplicate_path_fraction": sums["retained_duplicate_paths"] / (N * beam),
                       "mean_catalog_completable_paths": sums["catalog_completable_paths"] / N,
                       "mean_catalog_completable_unique_states": sums["catalog_completable_unique_states"] / N,
                       "catalog_incompletable_path_fraction": 1 - sums["catalog_completable_paths"] / (N * beam),
                       "users_with_catalog_incompletable_path": sums["users_with_dead_catalog_path"]})
    return {"cell": cell, "count": N, "beam": beam,
            "checkpoint_sha256": metrics["checkpoint_sha256"], "selected_SID_sha256": metrics["selected_SID_sha256"],
            "metrics": metrics["metrics"], "trace_loop_seconds": metrics["seconds"],
            "fates": {fate: {"count": fates[fate], "rate": fates[fate] / N} for fate in FATES},
            "rounds": rounds,
            "final_candidates": {**dict(final_sums), "target_generated_rate": final_sums["target_generated_users"] / N,
                                 "mean_legal_unique_count": final_sums["legal_unique_count"] / N,
                                 "target_legal_unique_rank_median_when_generated": statistics.median(legal_ranks) if legal_ranks else None},
            "last_token_conditional_on_gt_path_surviving_round3": {
                "users": len(min_ranks), "compatible_branches": len(branch_ranks),
                "per_user_best_truth_rank_median": statistics.median(min_ranks) if min_ranks else None,
                "per_branch_truth_rank_median": statistics.median(branch_ranks) if branch_ranks else None,
                "per_branch_truth_rank_buckets": dict(branch_rank_buckets),
                "final_argmax_miss_users": len(miss_min_ranks),
                "miss_users_best_truth_rank_median": statistics.median(miss_min_ranks) if miss_min_ranks else None,
                "miss_users_best_truth_rank_buckets": dict(miss_min_rank_buckets),
                "rank_one_miss_users_due_to_tie": miss_min_rank_buckets["1"]},
            "verified_files": {str(path.relative_to(root)): receipt["files"][name]["sha256"] for name, path in files.items()}}, compact


def paired_comparison(source, target, rows):
    left, right = rows[source], rows[target]
    transitions = {fate: Counter() for fate in FATES}
    new_hits, lost_hits, by_category = 0, 0, {fate: Counter() for fate in FATES if fate != "topk_hit"}
    sid_changes = 0
    for a, b in zip(left, right):
        need(all(a[key] == b[key] for key in ("index", "user", "target_item_id")), "Four cells use different users or targets")
        transitions[a["fate"]][b["fate"]] += 1
        sid_changes += a["target_SID"] != b["target_SID"]
        if not a["hit"] and b["hit"]:
            new_hits += 1
            by_category[a["fate"]]["new_hits"] += 1
        if a["hit"] and not b["hit"]:
            lost_hits += 1
            by_category[b["fate"]]["lost_hits"] += 1
    decomposition = {fate: {"new_hits": counts["new_hits"], "lost_hits": counts["lost_hits"],
                            "net_hits": counts["new_hits"] - counts["lost_hits"]} for fate, counts in by_category.items()}
    need(sum(item["net_hits"] for item in decomposition.values()) == new_hits - lost_hits, "Net hit decomposition differs")
    return {"from": source, "to": target, "count": N, "target_sid_changed_users": sid_changes,
            "new_hits": new_hits, "lost_hits": lost_hits, "net_hits": new_hits - lost_hits,
            "delta_hr10": (new_hits - lost_hits) / N,
            "fate_transitions": {f: {g: transitions[f][g] for g in FATES} for f in FATES},
            "hit_change_by_nonhit_fate": decomposition,
            "interpretation": "Accounting of observed nonhit categories; not causal attribution to attention, duplicates or legality."}


def normal_ci(values):
    count = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(count)
    return {"mean": mean, "standard_error": se, "normal_95_interval": [mean - 1.96 * se, mean + 1.96 * se],
            "scope": "Paired user-level descriptive normal interval on selected validation checkpoints; not independent training-seed evidence."}


def report(summary):
    cells = summary["cells"]
    out = ["# Stage07 验证集逐轮搜索诊断", "", "本次固定两组 best25 权重及各自保存的 SID，完整评估同一批 22,363 名验证用户，只增加搜索记录。候选选择、剪枝、末位 argmax 和最终排序策略都没有改动。四组输出逐用户复现此前结果；未重新训练、未使用测试集。", "",
           "| 模型 | beam | HR@10 | NDCG@10 | 最终候选含目标的用户数 | 返回Top10前，每人不同合法候选数 |", "|---|---:|---:|---:|---:|---:|"]
    for cell in CELLS:
        c = cells[cell]
        out.append(f"| {'A 原版' if cell.startswith('control') else 'B 商品内交互'} | {c['beam']} | {c['metrics']['hit_rate@10']:.6f} | {c['metrics']['ndcg@10']:.6f} | {c['final_candidates']['target_generated_users']} | {c['final_candidates']['mean_legal_unique_count']:.2f} |")
    low = summary["paired"]["A_to_B_beam32"]
    high = summary["paired"]["A_to_B_beam256"]
    high_rank_net = high["hit_change_by_nonhit_fate"]["rank_cutoff"]["net_hits"]
    a32, b32 = cells["control_beam32"], cells["item_local_beam32"]
    a256, b256 = cells["control_beam256"], cells["item_local_beam256"]
    generated = lambda cell: cell["final_candidates"]["target_generated_users"]
    out += ["",
            f"beam32 下，B 比 A 净多命中 {low['net_hits']} 人，最终候选里包含目标的用户也从 {generated(a32)} 增至 {generated(b32)}，多了 {generated(b32) - generated(a32)} 人。已经生成目标、却没排进Top10的用户，A只有 {a32['fates']['rank_cutoff']['count']} 人，B只有 {b32['fates']['rank_cutoff']['count']} 人。因此，小 beam 下的差别主要发生在目标能不能走到最终候选这一步。",
            "",
            f"beam256 下，最终候选包含目标的用户从 {generated(a256)} 变为 {generated(b256)}，只多了 {generated(b256) - generated(a256)} 人，命中却净增加 {high['net_hits']} 人。配对账目里，其中 {high_rank_net:+d} 人来自‘已生成但排在Top10外↔命中’的净变化，其他类别合计只有 {high['net_hits'] - high_rank_net:+d} 人。大 beam 让更多目标进入候选后，两组之间的差别更多体现在Top10排序。这个划分是当前用户结果的账目，不能当作交互模块的因果贡献。",
            "", "下面按每个用户目标 SID 在搜索中最终停在哪里划分，各列合计均为 22,363。这里的‘丢失’指与目标兼容的所有路径均被剪掉，不能据此直接判断是哪一种模型机制导致。", "",
            "| 目标的最终状态 | A/32 | B/32 | A/256 | B/256 |", "|---|---:|---:|---:|---:|"]
    order = ("control_beam32", "item_local_beam32", "control_beam256", "item_local_beam256")
    for fate in FATES:
        out.append("| " + FATE_CN[fate] + " | " + " | ".join(str(cells[cell]["fates"][fate]["count"]) for cell in order) + " |")
    out += ["", "A→B 的命中变化可以进一步拆成以下账目。正数表示该类未命中转成命中的人数更多，负数表示相反；这不是机制的因果贡献。", "",
            "| 对照 | 新命中 | 丢失命中 | 净增加 |", "|---|---:|---:|---:|"]
    for beam in (32, 256):
        p = summary["paired"][f"A_to_B_beam{beam}"]
        out.append(f"| beam{beam}，A→B | {p['new_hits']} | {p['lost_hits']} | {p['net_hits']:+d} |")
    out += ["", "| 未命中类别 | beam32 净变化 | beam256 净变化 |", "|---|---:|---:|"]
    for fate in FATES[:-1]:
        vals = [summary["paired"][f"A_to_B_beam{beam}"]["hit_change_by_nonhit_fate"][fate]["net_hits"] for beam in (32, 256)]
        out.append(f"| {FATE_CN[fate]} | {vals[0]:+d} | {vals[1]:+d} |")
    out += ["", "扩大 beam 的变化：", ""]
    for arm, name in (("control", "A"), ("item_local", "B")):
        p = summary["paired"][f"{arm}_32_to_256"]
        low, high = cells[f"{arm}_beam32"], cells[f"{arm}_beam256"]
        largest = max(FATES[:-1], key=lambda f: p["hit_change_by_nonhit_fate"][f]["net_hits"])
        out.append(f"- {name}：净多命中 {p['net_hits']} 人（新增 {p['new_hits']}，丢失 {p['lost_hits']}）。最终候选含目标的人数从 {low['final_candidates']['target_generated_users']} 变为 {high['final_candidates']['target_generated_users']}；命中变化账目中，‘{FATE_CN[largest]}’对应的净变化最大，为 {p['hit_change_by_nonhit_fate'][largest]['net_hits']:+d} 人。")
    out += ["", "每轮的目标存活用户数（第4轮是最后一位 argmax 完成后）：", "", "| 模型/beam | 第1轮 | 第2轮 | 第3轮 | 第4轮 |", "|---|---:|---:|---:|---:|"]
    for cell in order:
        out.append("| " + cell.replace("control", "A").replace("item_local", "B") + " | " + " | ".join(str(r["users_with_gt_path"]) for r in cells[cell]["rounds"]) + " |")
    out += ["", "搜索过程中确实有很多重复路径。同一组已填位置和取值，可以由不同填充顺序走到；这些重复项在最后去重前一直占着 beam 名额。下面的重复率按当前保留路径计算，不是返回Top10的重复率。", "",
            "| 模型/beam | 第3轮重复路径 | 第3轮无法合法补全的路径 | 末轮重复路径 | 末轮不同合法候选/人 |", "|---|---:|---:|---:|---:|"]
    for cell in order:
        c = cells[cell]
        third, final = c["rounds"][2], c["rounds"][3]
        label = cell.replace("control", "A").replace("item_local", "B")
        out.append(f"| {label} | {third['duplicate_path_fraction']:.2%} | {third['catalog_incompletable_path_fraction']:.2%} | {final['duplicate_path_fraction']:.2%} | {c['final_candidates']['mean_legal_unique_count']:.2f} |")
    out += ["", "beam32 到第3轮时，约57%的路径是重复项；填完4位后约77%重复，32条路径最终平均只有约6.7～6.8个不同的合法SID。B没有降低这些重复率，合法候选也没有更多。因此，这次结果不支持‘交互靠减少重复、增加候选多样性带来提升’的解释。重复状态是否真的挤掉了有价值的候选，还需要改搜索策略做一次对照才能判断。重复路径与无法合法补全的路径可能重叠，两列不能相加。", "",
            "最后一位只取 argmax 也是一个值得保留的线索：", "",
            "| 模型/beam | 第3轮仍有目标路径的用户 | 最后一位未选中目标的用户 | 其中目标最好排名为2～5 |", "|---|---:|---:|---:|"]
    for cell in order:
        last = cells[cell]["last_token_conditional_on_gt_path_surviving_round3"]
        near = last["miss_users_best_truth_rank_buckets"].get("2-5", 0)
        misses = last["final_argmax_miss_users"]
        label = cell.replace("control", "A").replace("item_local", "B")
        fraction = near / misses if misses else 0
        out.append(f"| {label} | {last['users']} | {misses} | {near}（{fraction:.1%}） |")
    ci = summary["width_interaction_delta_BminusA_256_minus_32"]["hit"]["normal_95_interval"]
    out += ["", "这里的‘最好排名’是同一用户所有目标兼容分支中，剩余真实token的最低名次。例如 A/32 有671名用户在最后一步丢失目标，其中509名至少有一条分支把真实token排在第2～5名。它说明末位只取第一名可能限制覆盖；不能直接理解成改取Top5就会多命中509人，因为扩展后的候选仍要竞争最终Top10。", "",
            "下一步先只验证一个问题：**beam32 中重复的部分状态，是否占用了本来能留给其他有效候选的位置？** 最小方案是固定这两组权重和SID，在同一验证集做 A/B × 原搜索/中间状态去重，共四格。只在第2、3轮全局候选选取时，把‘已填位置相同、取值相同’的状态合并，保留最高累计分数，再从所有扩展候选中取32个不同状态；不要先取32条后直接删掉重复项。最后一位仍取argmax，合法性过滤、Top10排序及其余设置保持原样。此方案仅为下一步建议，本次没有执行。", "",
            "判断时同时看第3轮不同状态数、最终目标覆盖、HR@10/NDCG@10及相同计时方式下的计算开销。若两组都提高，说明这是共用搜索环节值得改进的地方；若A追上得更多、A/B差距缩小，说明原先交互的优势对搜索限制较敏感；若去重后B仍领先，才有理由继续检验这部分收益。仅重复率下降、推荐指标不变，不足以说明方法有效。末位Top5留作另一项独立实验，避免一次改动两个因素。", "",
            f"‘小beam下交互增益更大’目前也只是这批结果的现象：增益差（beam256减beam32）的逐用户配对95%描述性区间为 [{ci[0]:.6f}, {ci[1]:.6f}]，仍跨过0。",
            "",
            "这组诊断能定位当前两组模型的搜索瓶颈，但仍是同一个训练种子、同一验证集选出的权重，且指标按 SID 匹配，不能直接证明独立商品级收益或跨种子稳定性。观测到的路径重复/非法比例与提升之间的关系，也需要后续受控改动才能确认。", "",
            "记录中的耗时包含逐轮观测、指标计算和 JSON 写入，只用于本次评估进度核对，不是线上推理时延，也不等同于平台计费时长。"]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    plan = read(root / "GRID_PLAN.json")
    complete = read(root / "COMPLETE.json")
    preflight = read(root / "GRID_PREFLIGHT.json")
    need(plan["split"] == "val" and plan["validation_count"] == N and plan["max_eval_batches"] is None, "Expected complete validation-only plan")
    need(preflight["grid_identity_sha256"] == plan["identity_sha256"] and preflight["status"] == "CPU_PREFLIGHT_PASSED_NO_INFERENCE", "Preflight identity differs")
    need(complete["status"] == "validation_beam_trace_completed", "Trace campaign is not complete")
    need(complete["grid_identity_sha256"] == plan["identity_sha256"] and set(complete["cells"]) == set(CELLS), "Grid completion differs")
    need(not complete["test_data_evaluated"] and not complete["training_performed"], "Unexpected test or training work")
    cells, rows = {}, {}
    for cell in CELLS:
        local_bound_file(root, cell, complete["cells"][cell]["receipt"])
        cells[cell], rows[cell] = summarize_cell(root, cell, plan["identity_sha256"])
        print(f"VERIFIED {cell}: {N} users", flush=True)
    pairs = {f"A_to_B_beam{beam}": paired_comparison(f"control_beam{beam}", f"item_local_beam{beam}", rows) for beam in (32, 256)}
    pairs.update({f"{arm}_32_to_256": paired_comparison(f"{arm}_beam32", f"{arm}_beam256", rows) for arm in ("control", "item_local")})
    width = {}
    for key in ("hit", "ndcg"):
        values = [b256[key] - a256[key] - b32[key] + a32[key] for a32, a256, b32, b256 in zip(*(rows[c] for c in CELLS))]
        width[key] = normal_ci(values)
    summary = {"status": "LOCAL_TRACE_VERIFICATION_PASSED", "count_per_cell": N,
               "grid_identity_sha256": plan["identity_sha256"], "input_manifests": {name: sha(root / name) for name in ("GRID_PLAN.json", "COMPLETE.json", "GRID_PREFLIGHT.json")},
               "cells": cells, "paired": pairs, "width_interaction_delta_BminusA_256_minus_32": width,
               "checks": {"four_complete_sha_bound_cells": True, "all_user_target_orders_aligned": True,
                          "trace_and_output_invariants_passed": True, "per_user_metrics_reconstructed": True,
                          "full_prediction_parity_with_prior_grid_remote_receipt": True},
               "limitations": ["Single training seed; checkpoints selected on this same validation set.", "SID-level targets, not collision-resolved item-level recall.", "Observed fate transitions and occupancy are descriptive, not causal attribution.", "Trace-loop time includes observer and output overhead, not serving latency."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    print("LOCAL_TRACE_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
