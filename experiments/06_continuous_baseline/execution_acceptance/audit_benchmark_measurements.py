"""CPU-only recheck of raw latency samples and checkpoint provenance.

This postprocessing tool lives outside the frozen training source. It never
loads weights, reads test metrics, runs inference, or changes the campaign.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
MODELS = ("sasrec_fullce", "esasrec")
COMPONENTS = ("query_forward", "catalog_top20", "end_to_end")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def audit(benchmarks, *, gate=None, campaign=None):
    if (gate is None) == (campaign is None):
        raise ValueError("Choose exactly one initial gate or final campaign")
    bindings = {}
    if gate is not None:
        evidence = read(gate)
        assert evidence["status"] == "PASS" and evidence["source_sha256"] == SOURCE
        for model in MODELS:
            matches = [r["result"] for r in evidence["resumed"] if r["name"] == "base_" + model]
            assert len(matches) == 1
            bindings[model] = matches[0]
        mode, seed = "initial_smoke", None
    else:
        campaign = Path(campaign)
        evidence = read(campaign / "queue.json")
        lock = read(campaign / "FROZEN_SELECTION.json")
        tasks = evidence["tasks"]
        assert len(tasks) == len({t["task_id"] for t in tasks}) == 89
        assert all(t["status"] == "succeeded" for t in tasks) and not evidence["workers"]
        assert evidence["source_sha256"] == lock["source_sha256"] == SOURCE
        assert lock["validation_only"] is True and lock["test_unlocked"] is True
        assert lock["training_results_sha256"] == fingerprint([
            {"task_id": t["task_id"], "result": t["result"]} for t in tasks])
        for model in MODELS:
            matches = [t for t in tasks if t["phase"] == "formal" and
                       t["config"]["model"] == model and t["seed"] == 3031]
            assert len(matches) == 1
            task = matches[0]
            assert task["config_sha256"] == lock["models"][model]["config_sha256"]
            bindings[model] = task["result"]
        mode, seed = "final_locked_configuration", 3031
    rows, hashes = [], {}
    for model in MODELS:
        path = Path(benchmarks) / model / "result.json"
        record, expected = read(path), bindings[model]
        hashes[model] = sha(path)
        assert record["checkpoint_sha256"] == expected["best_sha256"]
        assert record["binding"] == expected["binding"]
        assert record["binding"]["source_sha256"] == SOURCE
        assert record["binding"]["protocol"] == "full_train_prefix_v1"
        assert record["binding"]["smoke"] is (gate is not None)
        assert record["precision"] == "bf16" and record["attention_precision_policy"] == "fp32_math"
        assert "H100" in record["gpu"] and record["parameters"] > 0
        if campaign is not None:
            assert record["parameters"] == expected["parameter_count"]
        measurements = record["measurements"]
        assert len(measurements) == 6
        assert {(r["batch_size"], r["component"]) for r in measurements} == {
            (b, component) for b in (1, 128) for component in COMPONENTS}
        assert set(COMPONENTS) <= record["scope"].keys()
        for measurement in measurements:
            samples = measurement["milliseconds"]
            assert measurement["warmup"] == 50 and measurement["repeats"] == len(samples) == 200
            assert all(math.isfinite(value) and value > 0 for value in samples)
            values = {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
                      "throughput_items_per_second": measurement["batch_size"] * 1000 / statistics.mean(samples)}
            assert all(math.isclose(measurement[key], value, rel_tol=1e-12, abs_tol=1e-12)
                       for key, value in values.items())
            assert 0 < measurement["peak_allocated_bytes"] <= measurement["peak_reserved_bytes"]
            rows.append({"model": model, "batch_size": measurement["batch_size"],
                         "component": measurement["component"], **values})
    return {"status": "PASS", "mode": mode, "representative_seed": seed,
            "source_sha256": SOURCE, "raw_measurement_groups": len(rows), "timed_samples": 200 * len(rows),
            "benchmark_files_sha256": hashes, "independently_recomputed": rows,
            "scope": "CPU arithmetic and checkpoint/selection binding only; exclusivity, GPU UUID and actual kernel execution need separate Job/window evidence",
            "audit_script_sha256": sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gate")
    source.add_argument("--campaign")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = audit(args.benchmarks, gate=args.gate, campaign=args.campaign)
    with Path(args.report).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "independently_recomputed"}))


if __name__ == "__main__":
    main()
