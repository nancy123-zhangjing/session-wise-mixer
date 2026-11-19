"""Read saved best-checkpoint SID collisions locally, on CPU, without inference.

The two seed2024 rows must reproduce the accepted historical SID statistics.
All six selected checkpoints and their item-number mapping are SHA-bound.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HELPER = HERE / "postprocess/checkpoint_payload_checks.py"
SPEC = importlib.util.spec_from_file_location("sid_collision_cpu_checks", HELPER)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)
need, read, sha, tensor_digest = checks.need, checks.read, checks.sha, checks.tensor_digest
MAPPING = "data/cache/AmazonReviews2014/Beauty/processed/id_mapping.json"
SEEDS, VARIANTS = (2024, 2025, 2026), ("control", "item_local")


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def find_run(root, seed, variant):
    date = "20261020" if seed == 2024 else "20261021"
    name = f"stage07_item_local_joint_{variant}_seed{seed}_{date}_r1"
    root = Path(root).resolve()
    paths = ([root] if root.name == name else []) + [p for p in root.rglob(name) if p.is_dir()]
    need(len(paths) == 1, f"Expected exactly one local run: {name}; found {len(paths)}")
    return paths[0]


def catalog_mapping(runs):
    # The historical backup omits input bytes. New full delivery includes the
    # exact same SHA-bound mapping, so it can establish numbering for all six.
    source = runs[(2025, "control")] / "workspace" / MAPPING
    source_record = record(source)
    for run in runs.values():
        identities = read(run / "input_identities.json")
        entries = [row for row in identities["artifacts"] if row["path"] == MAPPING]
        need(len(entries) == 1 and entries[0]["sha256"] == source_record["sha256"]
             and entries[0]["size_bytes"] == source_record["size_bytes"], "Item-number mapping differs between runs")
    mapping = read(source)
    item2id, id2item = mapping["item2id"], mapping["id2item"]
    need(isinstance(id2item, list) and len(id2item) == 12102 and id2item[0] == "[PAD]"
         and item2id.get("[PAD]") == 0 and len(item2id) == len(id2item)
         and all(type(item2id.get(item)) is int and item2id[item] == index for index, item in enumerate(id2item)),
         "Catalog is not the expected contiguous item IDs with PAD at zero")
    return list(range(1, len(id2item))), source_record


def count_collisions(sids):
    counts = Counter(tuple(row) for row in sids)
    n, unique = len(sids), len(counts)
    need(n > 0, "Empty real-item SID catalog")
    colliding = sum(size for size in counts.values() if size > 1)
    excess = n - unique
    return {"item_count": n, "unique_sid_count": unique,
            "colliding_item_count": colliding, "collision_group_count": sum(size > 1 for size in counts.values()),
            "colliding_item_ratio": colliding / n, "duplicate_excess_count": excess,
            "duplicate_excess_ratio": excess / n, "maximum_group_size": max(counts.values())}


def inspect_best(run, seed, variant, item_ids):
    import torch
    files = {name: record(run / name) for name in
             ("identity.json", "results.json", "COMPLETE.json", "checkpoint_reload.json", "input_identities.json", "training_state/best_model.bin")}
    identity, result, complete, reload = [read(run / name) for name in
                                        ("identity.json", "results.json", "COMPLETE.json", "checkpoint_reload.json")]
    checkpoint = files["training_state/best_model.bin"]
    need(identity["seed"] == seed and identity["variant"] == result["variant"] == complete["variant"] == variant
         and identity["run_id"] == result["run_id"] == complete["run_id"] == run.name
         and complete["mode"] == "formal" and complete["result_file"] == "results.json"
         and complete["result_sha256"] == files["results.json"]["sha256"], "Run completion/identity differs")
    need(checkpoint["sha256"] == result["checkpoint_sha256"] == complete["checkpoint_sha256"] == reload["checkpoint_sha256"]
         and reload["best_epoch"] == result["fit"]["best_epoch"] and reload["strict_all_tensor_match"] is True
         and reload["mapping_synchronized_without_requantization"] is True, "Selected best checkpoint binding differs")
    expected = "formal_completed" if seed == 2024 else "formal_training_completed_validation_pending"
    need(result["status"] == complete["status"] == expected, "Training is incomplete")
    if seed != 2024:
        path = run / "validation/EVALUATION_COMPLETE.json"
        evaluation = read(path)
        files["validation/EVALUATION_COMPLETE.json"] = record(path)
        need(evaluation["status"] == "selected_validation_completed" and evaluation["seed"] == seed
             and evaluation["variant"] == variant and evaluation["best_epoch"] == result["fit"]["best_epoch"]
             and evaluation["checkpoint_sha256"] == checkpoint["sha256"]
             and evaluation["selected_SID_sha256"] == result["selected_SID_sha256"]
             and evaluation["test_data_evaluated"] is False and result["test_data_evaluated"] is False,
             "New validation is incomplete or references another best checkpoint")
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    codes = state["joint_alignment.hard_codes"]
    need(codes.device.type == "cpu" and codes.dtype == torch.int64 and tuple(codes.shape) == (len(item_ids) + 1, 4),
         "Saved SID tensor geometry differs from actual catalog numbering")
    need(bool((codes[0] == -1).all()) and bool(((codes[1:] >= 0) & (codes[1:] < 256)).all()),
         "Invalid saved SID values or PAD sentinel")
    sid_digest = tensor_digest(codes[1:])
    need(sid_digest == result["selected_SID_sha256"] == reload["matching_committed_SID_sha256"], "Saved SID SHA differs")
    statistics = count_collisions(codes[item_ids].tolist())
    for name, expected_record in files.items():
        need(record(run / name) == expected_record, "Evidence changed during CPU SID readback")
    return {"seed": seed, "variant": variant, "best_epoch": result["fit"]["best_epoch"],
            "checkpoint_sha256": checkpoint["sha256"], "SID_sha256": sid_digest,
            **statistics, "files": files}


def compare_historical(arms, historical):
    need(historical["status"] == "BEST_SID_STATISTICS_VERIFIED", "Historical SID statistics are not accepted evidence")
    for arm in arms:
        if arm["seed"] != 2024:
            continue
        old = historical["arms"][arm["variant"]]
        fields = ("best_epoch", "checkpoint_sha256", "SID_sha256", "item_count", "unique_sid_count",
                  "colliding_item_count", "collision_group_count", "maximum_group_size")
        need(all(arm[field] == old[field] for field in fields)
             and abs(arm["colliding_item_ratio"] - old["collision_rate"]) <= 1e-15,
             "Historical best SID statistics did not reproduce")


def summarize(verified_root, old_training_root, old_statistics):
    # Verify that all required local completions exist before loading any tensors.
    runs = {(seed, variant): find_run(old_training_root if seed == 2024 else verified_root, seed, variant)
            for seed in SEEDS for variant in VARIANTS}
    for (seed, _), run in runs.items():
        need((run / "COMPLETE.json").is_file() and (run / "training_state/best_model.bin").is_file(), "Missing completed best checkpoint")
        if seed != 2024:
            need((run / "validation/EVALUATION_COMPLETE.json").is_file(), "New selected validation has not completed")
    os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import torch
    need(not torch.cuda.is_initialized(), "This diagnostic is CPU only")
    torch.set_num_threads(2)
    item_ids, mapping = catalog_mapping(runs)
    reference = record(old_statistics)
    arms = [inspect_best(run, seed, variant, item_ids) for (seed, variant), run in runs.items()]
    compare_historical(arms, read(old_statistics))
    need(record(old_statistics) == reference and record(mapping["path"]) == mapping
         and not torch.cuda.is_initialized(), "Source evidence changed or CUDA was initialized")
    return {"status": "SAVED_BEST_SID_COLLISIONS_VERIFIED", "GPU_initialized": False,
            "source_field": "joint_alignment.hard_codes", "item_id_range": [1, len(item_ids)], "excluded_padding_item_id": 0,
            "item_mapping": mapping, "historical_reference": reference, "historical_statistics_reproduced": True,
            "tools": {"diagnostic": record(__file__), "cpu_helper": record(HELPER)}, "arms": arms,
            "definitions": {"colliding_item_ratio": "Items in SID groups of size >1 / real item count",
                            "duplicate_excess_ratio": "(Real item count - unique SID count) / real item count"},
            "scope": "Saved selected best only; no re-quantization, inference, training or test evaluation",
            "limitation": "Equal global collision counts do not imply identical collision memberships or equal target-weighted collision effects; this diagnostic alone cannot attribute recommendation gains"}


def table(value):
    lines = ["# 所选 best 的商品 SID 碰撞", "", "直接统计 checkpoint 保存的四位 SID。按真实商品编号1—12101计数，仅排除 PAD=0；没有重新量化。", "",
             "| seed | 模型 | best轮 | 商品数 | 唯一SID | 碰撞商品占比 | 重复超额占比 | 最大组 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in value["arms"]:
        name = "A 原版" if arm["variant"] == "control" else "B 商品内交互"
        lines.append(f"| {arm['seed']} | {name} | {arm['best_epoch']} | {arm['item_count']} | {arm['unique_sid_count']} | {arm['colliding_item_ratio']:.3%} | {arm['duplicate_excess_ratio']:.3%} | {arm['maximum_group_size']} |")
    lines += ["", "碰撞商品占比计算所有落在多人共用 SID 组里的商品；重复超额占比计算每组保留一个商品后剩余的数量，两者含义不同。", "",
              "2024两组已与原验收记录逐项一致；每行都绑定 best checkpoint、保存 SID 和商品编号映射的 SHA。全局碰撞数量相近，不代表碰撞成员相同，也不能单凭此表归因推荐增益；SID级命中仍可能包含同组商品的影响。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-root", type=Path, required=True)
    parser.add_argument("--old-training-root", type=Path, required=True)
    parser.add_argument("--old-statistics", type=Path,
                        default=HERE.parent / "joint_alternating/BEST_SID_STATISTICS_20261020_final_r3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.verified_root, args.old_training_root, args.old_statistics)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SID_COLLISIONS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "SID_COLLISIONS.md").write_text(table(result), encoding="utf-8")
    print("SAVED_BEST_SID_COLLISIONS_VERIFIED")


if __name__ == "__main__":
    main()
