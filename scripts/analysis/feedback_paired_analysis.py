"""Read-only, standard-library analysis of the three-arm feedback pilot.

The stored evaluator float32 scores are the observations. Analytical scores
derived from the raw first SID hit only validate their interpretation; they
never replace them. All subgroup comparisons use identical sample membership.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path


ARMS = ("frozen", "cross_recon", "cross_rec")
GROUPS = ("target_changed", "history_only", "unchanged")
PAIR_SPECS = (
    ("cross_rec_minus_cross_recon", "cross_rec", "cross_recon"),
    ("cross_rec_minus_frozen", "cross_rec", "frozen"),
    ("cross_recon_minus_frozen", "cross_recon", "frozen"),
)
REQUIRED_METRICS = (
    "hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "weighted_score",
    "legal@10", "duplicate@10",
)
SCORE_TOLERANCE = 1e-6


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sid(value, width=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("SID must be a nonempty integer list")
    result = tuple(_integer(v, "SID code") for v in value)
    if width is not None and len(result) != width:
        raise ValueError("Inconsistent SID width")
    return result


def _catalog(codes):
    if not isinstance(codes, (list, tuple)) or not codes:
        raise ValueError("Catalogue SID array must be nonempty")
    width = len(_sid(codes[0]))
    return tuple(_sid(row, width) for row in codes)


def changed_items(initial_codes, arm_codes_map):
    """Return the shared union of changed 1-based item IDs across arms."""
    initial = _catalog(initial_codes)
    if not isinstance(arm_codes_map, dict) or not arm_codes_map:
        raise ValueError("At least one arm catalogue is required")
    changed = set()
    for arm, codes in arm_codes_map.items():
        current = _catalog(codes)
        if len(current) != len(initial) or len(current[0]) != len(initial[0]):
            raise ValueError(f"Arm {arm}: catalogue shape differs from initial")
        changed.update(i + 1 for i, (a, b) in enumerate(zip(initial, current)) if a != b)
    return changed


def sample_identity(sample_index, target, history):
    """Canonical ordered identity; pad item 0 does not represent a history item."""
    _integer(sample_index, "sample_index")
    _integer(target, "target_item_id", minimum=1)
    if not isinstance(history, (list, tuple)):
        raise ValueError("history_item_ids must be an ordered list")
    ordered = [_integer(v, "history item ID") for v in history]
    return {"sample_index": sample_index, "target_item_id": target,
            "history_item_ids": [v for v in ordered if v != 0]}


def classify_sample(target, history, changed):
    """Target changes take precedence over concurrent history changes."""
    if target in changed:
        return "target_changed"
    if any(item != 0 and item in changed for item in history):
        return "history_only"
    return "unchanged"


def collision_sizes(codes):
    """Map every 1-based item ID to its complete-SID catalogue group size."""
    catalogue = _catalog(codes)
    counts = Counter(catalogue)
    return {index + 1: counts[code] for index, code in enumerate(catalogue)}


def _collision_summary(codes):
    counts = Counter(codes)
    collided_items = sum(size for size in counts.values() if size > 1)
    return {"unique_semantic_ids": len(counts), "collided_items": collided_items,
            "collision_rate": collided_items / len(codes),
            "collision_groups": sum(size > 1 for size in counts.values()),
            "largest_group": max(counts.values())}


def _close(actual, expected, description):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=SCORE_TOLERANCE):
        raise ValueError(f"{description}: stored {actual} differs from expected {expected}")


def _validate_record(record, catalogue, arm, expected_keys=None, legal_catalogue=None):
    if not isinstance(record, dict):
        raise ValueError(f"Arm {arm}: invalid record")
    try:
        identity = sample_identity(record["sample_index"], record["target_item_id"],
                                   record["history_item_ids"])
        target = identity["target_item_id"]
        if target > len(catalogue) or any(v > len(catalogue) for v in identity["history_item_ids"]):
            raise ValueError("Sample item ID is outside the catalogue")
        label = _sid(record["target_sid"], len(catalogue[0]))
        if label != catalogue[target - 1]:
            raise ValueError("Target SID does not match the official arm catalogue")
        predictions = record["top10_sids"]
        if not isinstance(predictions, list) or len(predictions) != 10:
            raise ValueError("Exactly 10 raw, ordered predicted SIDs are required")
        predictions = tuple(_sid(v, len(label)) for v in predictions)
        first = next((i + 1 for i, code in enumerate(predictions) if code == label), 0)
        reported = _integer(record["first_hit_rank"], "first_hit_rank")
        if reported != first:
            raise ValueError("first_hit_rank is not the first raw SID match (do not deduplicate)")
        metrics = record["metrics"]
        if not isinstance(metrics, dict) or any(key not in metrics for key in REQUIRED_METRICS):
            raise ValueError("Required per-user evaluator metrics are missing")
        if expected_keys is not None and set(metrics) != expected_keys:
            raise ValueError("Per-user metric keys differ between records or arms")
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Metric {name} must be a finite number")
        for k in (5, 10):
            hit = float(first != 0 and first <= k)
            ndcg = 1 / math.log2(first + 1) if hit else 0.0
            _close(metrics[f"hit_rate@{k}"], hit, f"HR@{k}")
            _close(metrics[f"ndcg@{k}"], ndcg, f"NDCG@{k}")
            if f"recall@{k}" in metrics:
                _close(metrics[f"recall@{k}"], hit, f"Recall@{k}")
        _close(metrics["weighted_score"],
               0.8 * metrics["ndcg@10"] + 0.2 * metrics["hit_rate@10"], "weighted_score")
        if legal_catalogue is None:
            legal_catalogue = set(catalogue)
        _close(metrics["legal@10"], sum(code in legal_catalogue for code in predictions) / 10,
               "legal@10")
        _close(metrics["duplicate@10"], 1 - len(set(predictions)) / 10, "duplicate@10")
        if "dup@10" in metrics:
            _close(metrics["dup@10"], metrics["duplicate@10"], "dup@10 alias")
        for name in REQUIRED_METRICS:
            if metrics[name] < -SCORE_TOLERANCE or metrics[name] > 1 + SCORE_TOLERANCE:
                raise ValueError(f"Metric {name} is outside [0, 1]")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Arm {arm}: malformed record: {error}") from error
    return identity


def _paired(rows_left, rows_right, indices, left, right):
    size = len(indices)
    result = {"left": left, "right": right, "count": size}
    for k in (5, 10):
        hr_name, ndcg_name = f"hit_rate@{k}", f"ndcg@{k}"
        gained = sum(rows_left[i]["metrics"][hr_name] > 0.5 and
                     rows_right[i]["metrics"][hr_name] < 0.5 for i in indices)
        lost = sum(rows_left[i]["metrics"][hr_name] < 0.5 and
                   rows_right[i]["metrics"][hr_name] > 0.5 for i in indices)
        hr_deltas = [rows_left[i]["metrics"][hr_name] - rows_right[i]["metrics"][hr_name]
                     for i in indices]
        ndcg_deltas = [rows_left[i]["metrics"][ndcg_name] - rows_right[i]["metrics"][ndcg_name]
                       for i in indices]
        net = gained - lost
        if not math.isclose(math.fsum(hr_deltas), net, abs_tol=SCORE_TOLERANCE):
            raise ValueError("Pairwise net hit count does not equal summed HR change")
        result.update({f"new_hits_at{k}": gained, f"lost_hits_at{k}": lost,
                       f"net_hits_at{k}": net,
                       f"delta_hr_at{k}": math.fsum(hr_deltas) / size if size else None,
                       f"delta_ndcg_at{k}": math.fsum(ndcg_deltas) / size if size else None,
                       f"ndcg_improved_at{k}": sum(v > 1e-12 for v in ndcg_deltas),
                       f"ndcg_worse_at{k}": sum(v < -1e-12 for v in ndcg_deltas),
                       f"ndcg_tied_at{k}": sum(abs(v) <= 1e-12 for v in ndcg_deltas)})
    return result


def _section(records, indices, metric_keys):
    size = len(indices)
    return {"count": size,
            "arms": {arm: {key: math.fsum(records[arm][i]["metrics"][key] for i in indices) / size
                           if size else None for key in metric_keys} for arm in ARMS},
            "pairs": {name: _paired(records[left], records[right], indices, left, right)
                      for name, left, right in PAIR_SPECS}}


def _assert_partition(overall, sections):
    if sum(value["count"] for value in sections.values()) != overall["count"]:
        raise ValueError("Subgroup membership does not partition the full validation set")
    for arm in ARMS:
        for metric, expected in overall["arms"][arm].items():
            regrouped = math.fsum(section["count"] * section["arms"][arm][metric]
                                 for section in sections.values() if section["count"]) / overall["count"]
            if not math.isclose(regrouped, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Weighted subgroup metric does not reproduce overall metric")
    for pair in overall["pairs"]:
        for k in (5, 10):
            for key in (f"new_hits_at{k}", f"lost_hits_at{k}", f"net_hits_at{k}",
                        f"ndcg_improved_at{k}", f"ndcg_worse_at{k}", f"ndcg_tied_at{k}"):
                if sum(section["pairs"][pair][key] for section in sections.values()) != overall["pairs"][pair][key]:
                    raise ValueError("Subgroup paired counts do not reproduce overall paired counts")


def analyze_records(records_by_arm, initial_codes, arm_codes_map, expected_count):
    """Validate all records and report paired SID-level, shared-subset results."""
    _integer(expected_count, "expected_count", minimum=1)
    if set(records_by_arm) != set(ARMS) or set(arm_codes_map) != set(ARMS):
        raise ValueError("Exactly frozen, cross_recon and cross_rec arms are required")
    initial = _catalog(initial_codes)
    catalogues = {arm: _catalog(arm_codes_map[arm]) for arm in ARMS}
    union = changed_items(initial_codes, arm_codes_map)
    if catalogues["frozen"] != initial:
        raise ValueError("Frozen arm catalogue differs from the shared initial catalogue")
    records = {arm: list(records_by_arm[arm]) for arm in ARMS}
    if any(len(rows) != expected_count for rows in records.values()):
        raise ValueError("Unexpected validation record count; partial evaluation is not final")
    metric_keys = sorted(records[ARMS[0]][0].get("metrics", {}))
    identities = {}
    for arm in ARMS:
        legal_catalogue = set(catalogues[arm])
        identities[arm] = [_validate_record(row, catalogues[arm], arm, set(metric_keys), legal_catalogue)
                           for row in records[arm]]
        if [row["sample_index"] for row in identities[arm]] != list(range(expected_count)):
            raise ValueError("Sample indices must be unique, ordered and cover 0..expected_count-1")
        if identities[arm] != identities[ARMS[0]]:
            raise ValueError("Arm sample order, target or truncated history differs")
    initial_collision = collision_sizes(initial_codes)
    arm_collision = {arm: collision_sizes(arm_codes_map[arm]) for arm in ARMS}
    group_indices = {group: [] for group in GROUPS}
    baseline_indices = {"singleton": [], "collided": []}
    union_indices = {"singleton": [], "collided": []}
    for index, row in enumerate(identities[ARMS[0]]):
        target = row["target_item_id"]
        group_indices[classify_sample(target, row["history_item_ids"], union)].append(index)
        baseline_indices["collided" if initial_collision[target] > 1 else "singleton"].append(index)
        ambiguous = any(arm_collision[arm][target] > 1 for arm in ARMS)
        union_indices["collided" if ambiguous else "singleton"].append(index)
    overall = _section(records, list(range(expected_count)), metric_keys)
    groups = {name: _section(records, indices, metric_keys) for name, indices in group_indices.items()}
    collision_strata = {
        "baseline": {name: _section(records, indices, metric_keys) for name, indices in baseline_indices.items()},
        "union": {name: _section(records, indices, metric_keys) for name, indices in union_indices.items()},
    }
    for partition in (groups, collision_strata["baseline"], collision_strata["union"]):
        _assert_partition(overall, partition)
    return {
        "protocol_id": "stage05_feedback_paired_sample_analysis_v1",
        "evaluation_level": "SID-level, single-seed validation; not strict item-level",
        "validation_count": expected_count,
        "catalogue": {
            "items": len(initial), "sid_width": len(initial[0]),
            "changed_union_items": len(union), "changed_union_item_ids": sorted(union),
            "changed_items_by_arm": {arm: sorted(changed_items(initial_codes, {arm: arm_codes_map[arm]}))
                                     for arm in ARMS},
            "collisions": {"initial": _collision_summary(initial),
                           **{arm: _collision_summary(catalogues[arm]) for arm in ARMS}},
        },
        "group_definition": "union of official SID changes across update arms; target takes precedence; actual truncated nonpadding history",
        "collision_definition": "full catalogue complete-SID tuple multiplicity; baseline initial and shared union of the three arms",
        "primary_pair": "cross_rec_minus_cross_recon",
        "overall": overall, "groups": groups, "collision_strata": collision_strata,
        "checks": {"count_and_order": True, "official_target_sids": True,
                   "raw_first_hit_and_float32_scores": True, "finite_metrics": True,
                   "shared_partitions_regroup_to_overall": True},
        "interpretation_limits": [
            "Stored original evaluator float32 scores are preserved; analytical scores only validate them.",
            "Subgroups localize observed gains/losses; they are not new randomized causal controls.",
            "An unchanged SID does not imply an unchanged decoder response.",
            "Collision-free target subsets do not convert the complete protocol to strict item-level evaluation.",
            "This analysis does not establish across-seed stability or isolate the Mixer contribution.",
        ],
    }


def _read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank record at {path}:{line_number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON record at {path}:{line_number}") from error


def _record_path(root, manifest, arm):
    name = manifest.get("records", {}).get(arm, f"{arm}/records.jsonl.gz")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Record path must stay inside the inference evidence directory")
    if not path.is_file():
        alternative = root / f"{arm}.records.jsonl.gz"
        if name == f"{arm}/records.jsonl.gz" and alternative.is_file():
            return alternative
        raise FileNotFoundError(path)
    return path


def _number(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(result):
    lines = ["# 推荐反馈逐样本配对诊断", "",
             "使用三组统一的 Decoder 第 3 轮快照；完整验证集、单 seed、SID-level 评估，不是严格商品级结果。", "",
             "## 总体结果", "", "| 组别 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = result["overall"]["arms"][arm]
        lines.append("| " + arm + " | " + " | ".join(_number(row[key]) for key in
                     ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10")) + " |")
    changed = result["catalogue"]
    lines += ["", f"共享变化集合包含 {changed['changed_union_items']} 件商品；所有模型使用同一批样本分组。", "",
              "## 主比较：加入推荐反馈减去仅重构更新", "",
              "| 共享分组 | 样本数 | 新增 HR@10 命中 | 丢失 HR@10 命中 | 净命中 | ΔHR@10 | ΔNDCG@10 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, section in [("overall", result["overall"]), *result["groups"].items()]:
        pair = section["pairs"][result["primary_pair"]]
        lines.append(f"| {name} | {section['count']} | {pair['new_hits_at10']} | "
                     f"{pair['lost_hits_at10']} | {pair['net_hits_at10']} | "
                     f"{_number(pair['delta_hr_at10'])} | {_number(pair['delta_ndcg_at10'])} |")
    lines += ["", "target_changed：目标商品在变化并集中（无论历史是否也变）；history_only：仅历史包含变化商品；unchanged：两者都不包含。", "",
              "## 碰撞敏感性分析", "",
              "| 固定分层依据 | 目标类别 | 样本数 | 推荐反馈净新增 HR@10 命中 | ΔNDCG@10 |",
              "|---|---|---:|---:|---:|"]
    for basis, strata in result["collision_strata"].items():
        for name, section in strata.items():
            pair = section["pairs"][result["primary_pair"]]
            lines.append(f"| {basis} | {name} | {section['count']} | {pair['net_hits_at10']} | {_number(pair['delta_ndcg_at10'])} |")
    lines += ["", "baseline 按共同起点的完整 SID 是否对应多个商品分层；union 按三组任意一组是否存在目标 SID 歧义分层。", "",
              "## 解释边界", "",
              "- SID 命中不等于碰撞商品的身份命中，没有从 SID 单一代表反推目标商品。",
              "- 各分组用于定位收益与损失，不直接证明机制因果；编码没变的样本也可能受 Decoder 更新影响。",
              "- 三组都保留 Mixer，这轮不能判断交互模块的独立作用；单 seed 也不代表结果稳定。",
              "- 原 evaluator 的逐样本 float32 指标原样聚合；原始排名只用于核对，不以重新计算的指标替代它们。", "",
              "完整分组、两组相对 Frozen 的辅助比较和核验见 [paired_summary.json](paired_summary.json)。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.input_dir.resolve()
    manifest = json.loads((root / "catalog_ids.json").read_text(encoding="utf-8"))
    records = {arm: _read_jsonl(_record_path(root, manifest, arm)) for arm in ARMS}
    result = analyze_records(records, manifest["initial"], manifest["arms"], args.expected_count)
    destination = args.output_dir.resolve() if args.output_dir else root
    targets = [destination / "paired_summary.json", destination / "README.md"]
    if any(target.exists() for target in targets):
        raise FileExistsError("Analysis outputs already exist; use a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with targets[1].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(result))
    print(json.dumps({"status": "complete", "validation_count": args.expected_count,
                      "output": str(destination), "primary_pair": result["overall"]["pairs"][result["primary_pair"]]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
