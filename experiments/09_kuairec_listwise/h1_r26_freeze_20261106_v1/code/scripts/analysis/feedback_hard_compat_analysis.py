"""Standard-library paired analysis of fixed-decoder hard-SID compatibility.

All four target positions are masked. Records contain saved per-position
scores, not Top-K recommendation results. The original-label scores are a
same-logit diagnostic anchor, never a replacement for the arm's official SID.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path


ARMS = ("frozen", "cross_recon", "cross_rec")
SPLITS = ("train", "val")
GROUPS = ("target_changed", "history_only", "unchanged")
PAIRS = (
    ("cross_rec_minus_cross_recon", "cross_rec", "cross_recon"),
    ("cross_rec_minus_frozen", "cross_rec", "frozen"),
    ("cross_recon_minus_frozen", "cross_recon", "frozen"),
)
POSITIONS = 4
CODEBOOK_SIZE = 256
SCORE_TOLERANCE = 1e-6
TIE_TOLERANCE = 1e-6
SCORE_FIELDS = ("position_nll", "position_loss", "initial_label_position_nll",
                "initial_label_position_loss")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sid(values):
    if not isinstance(values, (list, tuple)) or len(values) != POSITIONS:
        raise ValueError("Exactly four SID positions are required")
    result = tuple(_integer(value, "SID code") for value in values)
    if any(value >= CODEBOOK_SIZE for value in result):
        raise ValueError("SID code is outside the 256-entry codebook")
    return result


def _catalog(codes):
    if not isinstance(codes, (list, tuple)) or not codes:
        raise ValueError("Official SID catalogue must be nonempty")
    return tuple(_sid(row) for row in codes)


def changed_items(initial_codes, arm_codes_map):
    """Same shared, 1-based change-union definition as the paired Top-K audit."""
    initial = _catalog(initial_codes)
    if not isinstance(arm_codes_map, dict) or not arm_codes_map:
        raise ValueError("At least one updated catalogue is required")
    result = set()
    for arm, codes in arm_codes_map.items():
        current = _catalog(codes)
        if len(current) != len(initial):
            raise ValueError(f"Arm {arm}: catalogue length differs from initial")
        result.update(index + 1 for index, (old, new) in enumerate(zip(initial, current))
                      if old != new)
    return result


def sample_identity(sample_index, target, history):
    _integer(sample_index, "sample_index")
    _integer(target, "target_item_id", 1)
    if not isinstance(history, (list, tuple)):
        raise ValueError("Actual truncated history must be an ordered list")
    values = [_integer(value, "history item ID") for value in history]
    return {"sample_index": sample_index, "target_item_id": target,
            "history_item_ids": [value for value in values if value != 0]}


def classify_sample(target, history, changed):
    if target in changed:
        return "target_changed"
    return "history_only" if any(item in changed for item in history if item != 0) else "unchanged"


def collision_sizes(codes):
    catalogue = _catalog(codes)
    sizes = Counter(catalogue)
    return {index + 1: sizes[code] for index, code in enumerate(catalogue)}


def _collision_summary(codes):
    counts = Counter(codes)
    collided = sum(value for value in counts.values() if value > 1)
    return {"unique_semantic_ids": len(counts), "collided_items": collided,
            "collision_rate": collided / len(codes),
            "collision_groups": sum(value > 1 for value in counts.values()),
            "largest_group": max(counts.values())}


def _scores(row, name):
    values = row[name]
    if not isinstance(values, (list, tuple)) or len(values) != POSITIONS:
        raise ValueError(f"{name} must contain four scores")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative numbers")
    return values


def _validate(row, catalogue, initial, arm, smoothing):
    if not isinstance(row, dict):
        raise ValueError(f"Arm {arm}: malformed sample record")
    try:
        identity = sample_identity(row["sample_index"], row["target_item_id"], row["history_item_ids"])
        target = identity["target_item_id"]
        if target > len(catalogue) or any(item > len(catalogue) for item in identity["history_item_ids"]):
            raise ValueError("Item identity is outside the catalogue")
        label, predicted = _sid(row["target_sid"]), _sid(row["predicted_sid"])
        if label != catalogue[target - 1]:
            raise ValueError("Official hard-SID target label mismatch")
        correct = row["position_correct"]
        if (not isinstance(correct, (list, tuple)) or len(correct) != POSITIONS
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or value not in (0, 1) for value in correct)):
            raise ValueError("position_correct must contain four numeric 0/1 values")
        if list(correct) != [int(a == b) for a, b in zip(predicted, label)]:
            raise ValueError("Stored position correctness differs from the raw argmax SID")
        scores = {name: _scores(row, name) for name in SCORE_FIELDS}
        initial_label = initial[target - 1]
        for index in range(POSITIONS):
            if label[index] == initial_label[index]:
                for own, anchor in (("position_nll", "initial_label_position_nll"),
                                    ("position_loss", "initial_label_position_loss")):
                    if abs(scores[own][index] - scores[anchor][index]) > SCORE_TOLERANCE:
                        raise ValueError("Unchanged label position has different own/initial-label scores")
            for nll, loss in (("position_nll", "position_loss"),
                              ("initial_label_position_nll", "initial_label_position_loss")):
                lower_bound = (1 - smoothing) * scores[nll][index]
                if scores[loss][index] + SCORE_TOLERANCE < lower_bound:
                    raise ValueError("Smoothed CE is below its nonnegative NLL component")
                if smoothing == 0 and abs(scores[loss][index] - scores[nll][index]) > SCORE_TOLERANCE:
                    raise ValueError("Zero-smoothing CE must equal NLL")
        # The smoothing term is independent of the label for the same logits.
        # Therefore own - anchor smoothed CE = (1-epsilon)*(own-anchor NLL).
        for index in range(POSITIONS):
            difference = scores["position_loss"][index] - scores["initial_label_position_loss"][index]
            expected = (1 - smoothing) * (
                scores["position_nll"][index] - scores["initial_label_position_nll"][index])
            if not math.isclose(difference, expected, rel_tol=0, abs_tol=2 * SCORE_TOLERANCE):
                raise ValueError("Own and initial-label scores do not use the same logits/smoothing")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Arm {arm}: malformed sample record: {error}") from error
    return identity


def _sample_mean(row, field):
    return math.fsum(row[field]) / POSITIONS


def _arm_means(rows, indices):
    size = len(indices)
    names = {"position_nll": "mean_nll", "position_loss": "mean_loss",
             "position_correct": "mean_accuracy",
             "initial_label_position_nll": "mean_initial_label_nll",
             "initial_label_position_loss": "mean_initial_label_loss"}
    result = {}
    for field, scalar in names.items():
        destination = "position_accuracy" if field == "position_correct" else field
        result[destination] = [math.fsum(rows[i][field][digit] for i in indices) / size
                               if size else None for digit in range(POSITIONS)]
        result[scalar] = math.fsum(result[destination]) / POSITIONS if size else None
    for suffix, own, anchor in (
        ("nll", "position_nll", "initial_label_position_nll"),
        ("loss", "position_loss", "initial_label_position_loss"),
    ):
        result[f"mean_own_minus_initial_{suffix}"] = math.fsum(
            _sample_mean(rows[i], own) - _sample_mean(rows[i], anchor) for i in indices) / size if size else None
    return result


def _paired(left_rows, right_rows, indices, left, right):
    size = len(indices)
    result = {"left": left, "right": right, "count": size,
              "count_tie_absolute_tolerance": TIE_TOLERANCE}
    names = {"position_nll": "mean_nll", "position_loss": "mean_loss",
             "position_correct": "mean_accuracy",
             "initial_label_position_nll": "mean_initial_label_nll",
             "initial_label_position_loss": "mean_initial_label_loss"}
    for field, scalar in names.items():
        destination = "position_accuracy" if field == "position_correct" else field
        deltas = [_sample_mean(left_rows[i], field) - _sample_mean(right_rows[i], field)
                  for i in indices]
        result[f"delta_{scalar}"] = math.fsum(deltas) / size if size else None
        result[f"delta_{destination}"] = [math.fsum(
            left_rows[i][field][digit] - right_rows[i][field][digit] for i in indices) / size
            if size else None for digit in range(POSITIONS)]
        if field in ("position_nll", "initial_label_position_nll"):
            result[f"lower_{scalar}_count"] = sum(value < -TIE_TOLERANCE for value in deltas)
            result[f"higher_{scalar}_count"] = sum(value > TIE_TOLERANCE for value in deltas)
            result[f"tied_{scalar}_count"] = sum(abs(value) <= TIE_TOLERANCE for value in deltas)
    for suffix, own, anchor in (
        ("nll", "position_nll", "initial_label_position_nll"),
        ("loss", "position_loss", "initial_label_position_loss"),
    ):
        deltas = [(_sample_mean(left_rows[i], own) - _sample_mean(left_rows[i], anchor))
                  - (_sample_mean(right_rows[i], own) - _sample_mean(right_rows[i], anchor))
                  for i in indices]
        result[f"delta_mean_own_minus_initial_{suffix}"] = math.fsum(deltas) / size if size else None
        if size and not math.isclose(
            result[f"delta_mean_{suffix}"],
            result[f"delta_mean_initial_label_{suffix}"] + result[f"delta_mean_own_minus_initial_{suffix}"],
            rel_tol=0, abs_tol=1e-10,
        ):
            raise ValueError("Own-label delta does not decompose into anchor and label-rescoring deltas")
    return result


def _section(records, indices):
    return {"count": len(indices),
            "arms": {arm: _arm_means(records[arm], indices) for arm in ARMS},
            "pairs": {name: _paired(records[left], records[right], indices, left, right)
                      for name, left, right in PAIRS}}


def _assert_partition(overall, sections):
    if sum(section["count"] for section in sections.values()) != overall["count"]:
        raise ValueError("Shared groups do not partition the split")
    for arm in ARMS:
        for key, reference in overall["arms"][arm].items():
            positions = range(POSITIONS) if isinstance(reference, list) else (None,)
            for position in positions:
                expected = reference[position] if position is not None else reference
                observed = math.fsum(section["count"] * (
                    section["arms"][arm][key][position] if position is not None else section["arms"][arm][key])
                    for section in sections.values() if section["count"]) / overall["count"]
                if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("Weighted per-position/mean subgroup score does not reproduce overall")
    for pair, values in overall["pairs"].items():
        for key, expected in values.items():
            if key.endswith("_count"):
                if sum(section["pairs"][pair][key] for section in sections.values()) != expected:
                    raise ValueError("Subgroup lower/higher/tie counts do not reproduce overall")


def _unchanged_consistency(records, indices, include_own=True):
    fields = SCORE_FIELDS if include_own else SCORE_FIELDS[2:]
    maximum = {field: 0.0 for field in fields}
    mismatches = 0
    for index in indices:
        if len({tuple(records[arm][index]["predicted_sid"]) for arm in ARMS}) != 1:
            mismatches += 1
        for field in fields:
            for position in range(POSITIONS):
                values = [records[arm][index][field][position] for arm in ARMS]
                maximum[field] = max(maximum[field], max(values) - min(values))
    result = {"count": len(indices), "absolute_tolerance": SCORE_TOLERANCE,
              "predicted_sid_mismatches": mismatches,
              **{f"max_{field}_discrepancy": value for field, value in maximum.items()}}
    if mismatches or any(value > SCORE_TOLERANCE for value in maximum.values()):
        raise ValueError(f"Fixed-decoder unchanged-context invariance failed: {result}")
    return result


def analyze_records(records_by_split, initial_codes, arm_codes_map, expected_counts, label_smoothing=0.1):
    """Analyze two fixed-context splits without replacing the saved scores."""
    if (isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float))
            or not math.isfinite(label_smoothing) or not 0 <= label_smoothing <= 1):
        raise ValueError("label_smoothing must be finite and between 0 and 1")
    if set(records_by_split) != set(SPLITS) or set(expected_counts) != set(SPLITS):
        raise ValueError("Exactly train and val splits and expected counts are required")
    if set(arm_codes_map) != set(ARMS):
        raise ValueError("Exactly frozen, cross_recon and cross_rec official catalogues are required")
    initial = _catalog(initial_codes)
    catalogues = {arm: _catalog(arm_codes_map[arm]) for arm in ARMS}
    union = changed_items(initial_codes, arm_codes_map)
    if catalogues["frozen"] != initial:
        raise ValueError("Frozen catalogue must equal the shared initial catalogue")
    initial_collision = collision_sizes(initial_codes)
    arm_collision = {arm: collision_sizes(arm_codes_map[arm]) for arm in ARMS}
    splits = {}
    for split in SPLITS:
        expected = _integer(expected_counts[split], f"expected {split} count", 1)
        if set(records_by_split[split]) != set(ARMS):
            raise ValueError(f"Split {split}: exactly three arms are required")
        records = {arm: list(records_by_split[split][arm]) for arm in ARMS}
        if any(len(rows) != expected for rows in records.values()):
            raise ValueError(f"Split {split}: incomplete or unexpected record count")
        identities = {}
        for arm in ARMS:
            identities[arm] = [_validate(row, catalogues[arm], initial, arm, label_smoothing)
                               for row in records[arm]]
            if [row["sample_index"] for row in identities[arm]] != list(range(expected)):
                raise ValueError(f"Split {split}: sample order must cover 0..expected_count-1")
            if identities[arm] != identities[ARMS[0]]:
                raise ValueError(f"Split {split}: paired target or actual truncated history differs")
        group_indices = {name: [] for name in GROUPS}
        baseline_indices = {"singleton": [], "collided": []}
        union_indices = {"singleton": [], "collided": []}
        context_unchanged_indices = []
        for index, identity in enumerate(identities[ARMS[0]]):
            target, history = identity["target_item_id"], identity["history_item_ids"]
            group_indices[classify_sample(target, history, union)].append(index)
            if not any(item in union for item in history):
                context_unchanged_indices.append(index)
            baseline_indices["collided" if initial_collision[target] > 1 else "singleton"].append(index)
            ambiguous = any(arm_collision[arm][target] > 1 for arm in ARMS)
            union_indices["collided" if ambiguous else "singleton"].append(index)
        overall = _section(records, list(range(expected)))
        groups = {name: _section(records, indices) for name, indices in group_indices.items()}
        collision = {
            "baseline": {name: _section(records, indices) for name, indices in baseline_indices.items()},
            "union": {name: _section(records, indices) for name, indices in union_indices.items()},
        }
        for partition in (groups, collision["baseline"], collision["union"]):
            _assert_partition(overall, partition)
        splits[split] = {"count": expected, "overall": overall, "groups": groups,
                         "collision_strata": collision,
                         "unchanged_consistency": _unchanged_consistency(records, group_indices["unchanged"]),
                         "context_unchanged_consistency": _unchanged_consistency(
                             records, context_unchanged_indices, include_own=False)}
    return {
        "analysis_protocol": "fixed_initial_decoder_hard_sid_compatibility_v1",
        "inference_only": True, "masking": "all four target positions masked for every sample",
        "evaluation_level": "fixed-decoder per-position hard-SID compatibility; not Top-K or item-level",
        "label_smoothing": label_smoothing, "primary_pair": "cross_rec_minus_cross_recon",
        "expected_counts": dict(expected_counts), "splits": splits,
        "catalogue": {"items": len(initial), "sid_width": POSITIONS,
                      "changed_union_items": len(union), "changed_union_item_ids": sorted(union),
                      "changed_items_by_arm": {arm: sorted(changed_items(initial_codes, {arm: arm_codes_map[arm]}))
                                               for arm in ARMS},
                      "collisions": {"initial": _collision_summary(initial),
                                     **{arm: _collision_summary(catalogues[arm]) for arm in ARMS}}},
        "group_definition": "one shared union of official post-tokenizer SID changes; target takes precedence; actual truncated nonpadding history",
        "anchor_definition": "same arm logits/context scored against the common initial target SID; diagnostic only",
        "lower_higher_tie_definition": "paired four-position sample-mean NLL difference, using absolute 1e-6 tie tolerance",
        "checks": {"count_identity_and_order": True, "official_sid_labels_and_raw_accuracy": True,
                   "finite_nonnegative_four_position_scores": True, "same_logit_label_smoothing_identity": True,
                   "unchanged_label_own_and_anchor_equal": True, "unchanged_context_invariance": True,
                   "shared_groups_regroup_to_overall": True, "own_anchor_delta_decomposition": True},
        "interpretation_limits": [
            "Both the DIGER training forward and official evaluation use hard assignments; soft probabilities provide only the ST backward derivative.",
            "This fully-masked fixed condition is not an exact replay of random/guided training loss.",
            "Lower own-label NLL can reflect easier replacement labels or coarser collision groups, not better item representation.",
            "Higher own-label NLL can reflect an old decoder not yet adapted to the new labels, not intrinsically worse tokenization.",
            "Anchored NLL isolates score changes under changed history contexts with fixed initial labels in this condition; it is not an official recommendation metric.",
            "Own-minus-anchor NLL is same-logit label-rescoring, not a tokenizer quality measure.",
            "This is a compatibility/direction diagnostic, not a mandatory pass gate before continued research.",
        ],
    }


def _read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank record at {path}:{number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed record at {path}:{number}") from error


def _record_path(root, manifest, split, arm):
    mapping = manifest.get("records", {})
    name = mapping.get(split, {}).get(arm, f"{arm}/{split}.records.jsonl.gz")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Record path cannot escape the input evidence directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _number(value):
    return "—" if value is None else f"{value:.6f}"


def markdown_report(result):
    lines = ["# 固定初始 Decoder 的硬 SID 兼容性诊断", "",
             "同一初始 Encoder、Decoder、SID embedding 和输出层；只切换正式硬 SID 映射。每个样本四个目标位置全部遮蔽，不更新任何参数。", "",
             "本报告是特定条件下的 NLL／平滑 CE／逐位置准确率诊断，不是随机训练 loss 的精确重放，也不是 HR、NDCG 或严格商品级结果。", "",
             f"本次评分的 label smoothing ε={result['label_smoothing']:g}。同 logits 下应满足："
             "own CE − anchor CE = (1−ε) × (own NLL − anchor NLL)。", ""]
    for split in SPLITS:
        section = result["splits"][split]
        lines += [f"## {split}：{section['count']} 个样本", "",
                  "| 组别 | 正式标签平均 NLL | 平滑 CE | 逐位置平均准确率 | 初始标签 anchored NLL | 正式标签 − 初始标签 NLL |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            values = section["overall"]["arms"][arm]
            keys = ("mean_nll", "mean_loss", "mean_accuracy", "mean_initial_label_nll", "mean_own_minus_initial_nll")
            lines.append("| " + arm + " | " + " | ".join(_number(values[key]) for key in keys) + " |")
        lines += ["", "主比较：Cross-Rec − Cross-Recon，负的 ΔNLL 表示该固定条件下损失较低，不等同编码质量更好。", "",
                  "| 共享分组 | 样本数 | Δ正式标签 NLL | NLL 更低 / 更高 / 相同 | Δ初始标签 anchored NLL | Δ标签换位评分项 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, current in [("overall", section["overall"]), *section["groups"].items()]:
            pair = current["pairs"][result["primary_pair"]]
            counts = f"{pair['lower_mean_nll_count']} / {pair['higher_mean_nll_count']} / {pair['tied_mean_nll_count']}"
            lines.append(f"| {name} | {current['count']} | {_number(pair['delta_mean_nll'])} | {counts} | "
                         f"{_number(pair['delta_mean_initial_label_nll'])} | {_number(pair['delta_mean_own_minus_initial_nll'])} |")
        lines += ["", "共享分组以两种更新 SID 相对起点变化的商品并集确定；目标变化优先，history_only 只改变实际截断历史。", "",
                  "逐位置原始均值、相对 Frozen 的辅助比较和固定 baseline／union 碰撞层见 JSON，不按各组自己选不同子集。", "",
                  f"unchanged 样本最大 NLL 差异：{section['unchanged_consistency']['max_position_nll_discrepancy']:.9g}；"
                  f"预测 SID 不一致样本数：{section['unchanged_consistency']['predicted_sid_mismatches']}。", ""]
    lines += ["## 解释边界", "",
              "- 本轮 DIGER 训练前向也是硬 one-hot；soft 只提供 ST 反向导数，不是 train-soft／eval-hard 的前向切换。",
              "- 正式标签损失下降可能来自换成较容易预测的标签、碰撞后的粗粒度编码；上升可能来自旧 Decoder 尚未适配。两者都不能直接判定 Tokenizer 内在质量。",
              "- anchored NLL 用同一套 logits 对初始目标标签评分，帮助定位历史编码上下文的影响；它不是正式推荐指标。",
              "- own−anchor 只是同 logits 下的目标标签换位评分差异，不称为表征质量收益。",
              "- 该诊断不是继续研究、延长统一恢复预算或做 Mixer 消融必须通过的门槛；方向判断须结合推荐结果和适配条件。", "",
              "完整材料：[compatibility_summary.json](compatibility_summary.json)。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train", required=True, type=int)
    parser.add_argument("--expected-val", required=True, type=int)
    parser.add_argument("--label-smoothing", type=float, required=True,
                        help="Exact value resolved from the locked training config layers; do not assume a default")
    args = parser.parse_args(argv)
    root, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError("Analysis destination must be a new directory; evidence is never overwritten")
    manifest = json.loads((root / "catalog_ids.json").read_text(encoding="utf-8"))
    records = {split: {arm: _read_jsonl(_record_path(root, manifest, split, arm)) for arm in ARMS}
               for split in SPLITS}
    result = analyze_records(records, manifest["initial"], manifest["arms"],
                             {"train": args.expected_train, "val": args.expected_val}, args.label_smoothing)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "compatibility_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (destination / "README.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(result))
    print(json.dumps({"status": "complete", "output": str(destination),
                      "expected_counts": result["expected_counts"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
