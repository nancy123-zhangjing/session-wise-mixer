"""Read-only independent receipt and raw-latency audit; writes derived proofs."""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    gate = read(HERE / "smoke/GPU_GATE.json")
    cpu = HERE.parent / "remote_cpu_20261016_02/CPU_GATE.json"
    assert gate["status"] == "PASS" and gate["source_sha256"] == SOURCE
    assert read(cpu)["source_sha256"] == SOURCE and sha(cpu) == gate["cpu_gate_sha256"]
    assert gate["cache_manifest_sha256"] == "9d229a275d0c476e9c8d64a7797a0d2d1eada535cfe5ad81dc8118a518ab5519"
    inventory = gate["inventory"]
    assert len(inventory) == 8 and len({r["uuid"] for r in inventory}) == 8
    assert {int(r["index"]) for r in inventory} == set(range(8))
    assert all("H100" in r["name"] and int(r["memory_mib"]) >= 80000 for r in inventory)
    for phase, updates in (("base", 20), ("resumed", 21), ("parallel", 2)):
        records = gate[phase]
        assert len(records) == (10 if phase == "parallel" else 2)
        for record in records:
            result = record["result"]
            assert result["status"] == "SMOKE_COMPLETE" and result["updates"] == updates
            assert result["validation_samples"] == result["best_validation"]["samples"] == 22363
            assert result["binding"]["source_sha256"] == SOURCE and result["binding"]["smoke"]
            assert record["gpu_uuid"] == inventory[record["gpu_slot"]]["uuid"]
    assert {r["gpu_slot"] for r in gate["parallel"]} == set(range(8))
    dispatches = [r for r in gate["dispatches"] if r["name"].startswith("parallel_")]
    assert len(dispatches) == 10 and max(r["concurrent_worker_count"] for r in dispatches) == 8
    assert len(gate["resume_checks"]) == 2 and all(
        r["updates_before"] == 20 and r["updates_after"] == 21 and r["optimizer_steps_restored"]
        for r in gate["resume_checks"])
    assert len(gate["precision_checks"]) == 2 and all(
        r["minimum_query_cosine_fp32_bf16"] >= 0.98 and r["gpu_topk_matches_full_stable_sort"]
        for r in gate["precision_checks"])
    checks = gate["gradient_checks"]
    assert {(r["model"], r["dim"], r["heads"]) for r in checks} == {
        (m, d, h) for m in ("esasrec", "sasrec_fullce") for d, h in ((64, 1), (512, 2))}
    assert all(r["training_mode"] and r["dropout_disabled"] and r["fixed_negatives"]
               and 0 <= r["relative_gradient_l2"] < 0.05 and r["gradient_cosine"] > 0.999 for r in checks)
    stress = read(HERE / "smoke/NUMERICAL_STRESS.json")
    manifest = read(ROOT / "configs/06_continuous_baseline/search_manifest.json")
    critical = next(t for t in manifest["trials"] if t["trial_id"] == "esasrec_17")
    assert stress == gate["numerical_stress"] and stress["status"] == "PASS"
    assert fingerprint(stress["config"]) == critical["config_sha256"]
    assert stress["seed"] == 2024 and stress["not_formal_training"] and not stress["validation_or_test_used"]
    assert stress["attention_precision_policy"] == "fp32_math" and len(stress["epochs"]) == 35
    for epoch, row in enumerate(stress["epochs"], 1):
        assert row["epoch"] == epoch and row["updates"] == 514 and row["samples"] == 131413
        assert math.isfinite(row["train_loss"]) and 0 <= row["maximum_gradient_norm"] < 1e6
    candidate = read(HERE.parent / "failure_20261016_01/gpu_precision_diagnostic/stress.json")
    assert len(candidate["epochs"]) == 35
    for actual, previous in zip(stress["epochs"], candidate["epochs"]):
        assert actual["train_loss"] == previous["train_loss"]
        assert actual["maximum_gradient_norm"] == previous["max_grad_norm"]
    result = {"status": "PASS", "source_sha256": SOURCE, "independent_receipt_recheck": True,
              "scope": "GPU gate receipts, not 89-run completion or full checkpoint payload audit",
              "distinct_h100s": 8, "parallel_tasks": 10, "actual_resume_updates": [20, 21],
              "gradient_checks": checks, "stress_full_epochs": 35, "stress_updates": 17990,
              "stress_max_gradient_norm": max(r["maximum_gradient_norm"] for r in stress["epochs"]),
              "actual_source_matches_candidate_all_35_loss_and_norm_rows": True,
              "gate_sha256": sha(HERE / "smoke/GPU_GATE.json"), "audit_script_sha256": sha(__file__)}
    (HERE / "GPU_GATE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8", newline="\n")
    latency = []
    for model in ("sasrec_fullce", "esasrec"):
        path = HERE / "initial_benchmarks" / model / "result.json"
        record = read(path)
        base = next(r["result"] for r in gate["resumed"] if r["name"] == "base_" + model)
        assert record["checkpoint_sha256"] == base["best_sha256"]
        assert record["binding"] == base["binding"]
        assert record["precision"] == "bf16" and record["attention_precision_policy"] == "fp32_math"
        assert len(record["measurements"]) == 6
        assert {(r["batch_size"], r["component"]) for r in record["measurements"]} == {
            (b, c) for b in (1, 128) for c in ("query_forward", "catalog_top20", "end_to_end")}
        for measurement in record["measurements"]:
            samples = measurement["milliseconds"]
            assert measurement["warmup"] == 50 and measurement["repeats"] == len(samples) == 200
            assert all(math.isfinite(v) and v > 0 for v in samples)
            expected = {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
                        "throughput_items_per_second": measurement["batch_size"] * 1000 / statistics.mean(samples)}
            assert all(math.isclose(measurement[k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in expected.items())
            assert 0 < measurement["peak_allocated_bytes"] <= measurement["peak_reserved_bytes"]
            latency.append({"model": model, "batch_size": measurement["batch_size"],
                            "component": measurement["component"], **expected})
    proof = {"status": "PASS", "source_sha256": SOURCE, "raw_measurement_groups": 12,
             "timed_samples": 2400, "independently_recomputed": latency,
             "scope": "Initial 64d smoke checkpoints; final selected-model timing still required",
             "audit_script_sha256": sha(__file__)}
    (HERE / "INITIAL_BENCHMARK_AUDIT.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({"GPU_GATE_AUDIT": "PASS", "stress_updates": 17990,
                      "INITIAL_BENCHMARK_AUDIT": "PASS", "timed_samples": 2400}))


if __name__ == "__main__":
    main()
