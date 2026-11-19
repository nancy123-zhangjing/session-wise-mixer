"""Second implementation of the locked Stage09 statistics, CPU and local files only.

Does not import either experiment analyzer, production metrics, Torch or decoding.
Requires completed primary C and final reports; does not regenerate A/B reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pyarrow.parquet as pq

ROW_TOL = 2e-6
SECOND_IMPLEMENTATION_TOL = 5e-12
PRIMARY = ["list/" + m + "@5" for m in ("ndcg", "recall", "hr")]
PRIMARY += [f"aggregate/{m}@{k}" for k in (10, 20, 50) for m in ("ndcg", "recall", "hr")]
GRID = (20, 23, 26, 29, 32, 35)
BOUND = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def bind(path, expected=None):
    p = Path(path).resolve()
    actual = digest(p)
    if expected is not None and actual != expected:
        raise AssertionError(f"SHA mismatch: {p}")
    if str(p) in BOUND and BOUND[str(p)] != actual:
        raise AssertionError(f"Input changed: {p}")
    BOUND[str(p)] = actual
    return p


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text(encoding="utf-8"))


def close(actual, expected, label, tolerance=SECOND_IMPLEMENTATION_TOL):
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"Keys differ: {label}")
        return max((close(actual[k], expected[k], label + "/" + k, tolerance) for k in actual), default=0.0)
    if isinstance(actual, (list, tuple, np.ndarray)):
        if len(actual) != len(expected):
            raise AssertionError(f"Lengths differ: {label}")
        return max((close(a, b, label, tolerance) for a, b in zip(actual, expected)), default=0.0)
    if actual is None or isinstance(actual, (str, bool)):
        if actual != expected:
            raise AssertionError(f"Values differ: {label}")
        return 0.0
    error = abs(float(actual) - float(expected))
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(f"Numerical mismatch {label}: {actual} != {expected}; error={error}")
    return error


def ranking(items, target, cutoffs):
    unique = set()
    good, duplicates, hit_positions = [], [], []
    target_set = set(target)
    for rank, item in enumerate(items):
        valid = type(item) is int and 1 <= item <= 10728
        repeated = item > 0 and item in unique
        good.append(valid)
        duplicates.append(repeated)
        if valid and not repeated and item in target_set:
            hit_positions.append(rank)
        unique.add(item)
    result = {}
    for k in cutoffs:
        ranks = [i for i in hit_positions if i < k]
        n = len(ranks)
        discount = sum(1.0 / math.log2(i + 2) for i in ranks)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(target))))
        result.update({f"ndcg@{k}": discount / ideal, f"recall@{k}": n / len(target),
                       f"hr@{k}": float(bool(n)), f"hit_rate@{k}": float(bool(n)), f"precision@{k}": n / k})
    n = len(items)
    result.update(completion_rate=float(all(good) and not any(duplicates)),
                  duplicate_rate=sum(duplicates) / n,
                  missing_rate=sum(i == 0 for i in items) / n,
                  invalid_rate=sum(i != 0 and not v for i, v in zip(items, good)) / n,
                  distinct_valid_items=float(sum(v and not d for v, d in zip(good, duplicates))))
    if n == len(target):
        result.update(position_accuracy=sum(i == j for i, j in zip(items, target)) / n,
                      exact_match=float(items == target))
    return result


def contributions(row):
    target = row["targets"]
    assert len(target) == 5 and len(set(target)) == 5 and all(type(i) is int and 1 <= i <= 10728 for i in target)
    candidates = row["predictions"]
    assert candidates and all(len(row[k]) == len(candidates) for k in ("scores", "beam_mask", "complete_mask"))
    viable = []
    for i, (items, score, present, declared_complete) in enumerate(zip(candidates, row["scores"], row["beam_mask"], row["complete_mask"])):
        assert len(items) == 5 and all(type(x) is int for x in items)
        finite = score is not None and isinstance(score, (int, float)) and math.isfinite(score)
        assert type(present) is bool and type(declared_complete) is bool and present == finite
        derived = present and len(set(items)) == 5 and all(1 <= x <= 10728 for x in items)
        assert declared_complete == derived
        if derived:
            viable.append((i, score, items))
    # Tuple sorting makes equal path scores retain their original beam index.
    viable.sort(key=lambda v: (-v[1], v[0]))
    aggregate, used, seen = [], 0, set()
    for _, _, items in viable[:32]:
        used += 1
        for item in items:
            if item not in seen:
                seen.add(item)
                aggregate.append(item)
                if len(aggregate) == 50:
                    break
        if len(aggregate) == 50:
            break
    count = len(aggregate)
    aggregate += [0] * (50 - count)
    values = {"list/" + k: v for k, v in ranking(candidates[0], target, (5,)).items()}
    values.update({"aggregate/" + k: v for k, v in ranking(aggregate, target, (10, 20, 50)).items()})
    values.update({"aggregate/candidate_count": float(count), "aggregate/lists_used": float(used)})
    values.update({f"aggregate/coverage@{k}": min(count, k) / k for k in (10, 20, 50)})
    return values


def source(dataset, split):
    p = bind(dataset / (split + ".parquet"))
    records = pq.read_table(p, columns=["sample_id", "user_id", "target_item_ids"]).to_pylist()
    seen_record = read(dataset / "train_seen.json")
    seen = set(seen_record["item_ids"])
    assert len(seen) == len(seen_record["item_ids"])
    users = sorted(set(r["user_id"] for r in records))
    uid = {u: i for i, u in enumerate(users)}
    truth = {}
    counts = np.zeros(len(users), dtype=np.int64)
    groups = np.zeros(2, dtype=np.int64)
    for i, row in enumerate(records):
        key = row["sample_id"]
        assert key not in truth
        cold = any(item not in seen for item in row["target_item_ids"])
        truth[key] = (i, row["user_id"], row["target_item_ids"], cold)
        counts[uid[row["user_id"]]] += 1
        groups[int(cold)] += 1
    return dict(truth=truth, users=users, uid=uid, counts=counts, groups=groups,
                split_sha=BOUND[str(p)], seen_sha=BOUND[str((dataset / "train_seen.json").resolve())])


def replay(label, point, truth, full_audit):
    prediction = bind(point["predictions"], point["predictions_sha256"])
    meta = read(point["summary"], point["summary_sha256"])
    split = point["split"]
    assert meta["catalog_size"] == 10728 and meta["beam"] == 32 and meta["seed"] == 2024
    assert meta["inference_precision"] == "fp32" and meta["architecture"] == "H"
    assert meta["full_split"] is True and meta["subset"] is None
    assert meta["split"] == split and meta["examples"] == len(truth["truth"])
    assert meta["dataset_binding"]["files"][split + ".parquet"]["sha256"] == truth["split_sha"]
    assert meta["dataset_binding"]["files"]["train_seen.json"]["sha256"] == truth["seen_sha"]
    protocol = meta["metric_protocol"]
    assert protocol == dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                           target_relevance="unordered_set_of_five_future_distinct_items",
                           aggregation="list_score_order_then_item_position_first_occurrence")
    names = sorted(meta["metrics"]) if full_audit else PRIMARY
    idx = {name: i for i, name in enumerate(names)}
    sums = np.zeros(len(names), dtype=np.float64)
    user_sums = np.zeros((len(truth["users"]), len(names)), dtype=np.float64)
    group_sums = np.zeros((2, len(names)), dtype=np.float64)
    consumed = set()
    maximum = 0.0
    with gzip.open(prediction, "rt", encoding="utf-8") as f:
        for line in f:
            assert line.strip()
            row = json.loads(line)
            sample = row["sample_id"]
            assert sample not in consumed
            consumed.add(sample)
            expected = truth["truth"][sample]
            assert (row["dataset_index"], row["user_id"], row["targets"], row["cold_target"]) == expected
            assert type(row["cold_target"]) is bool and type(row["user_id"]) is int
            values = contributions(row)
            if full_audit:
                maximum = max(maximum, close(values, row["metrics"], label + "/row", ROW_TOL))
            vector = np.array([values[n] for n in names], dtype=np.float64)
            sums += vector
            user_sums[truth["uid"][row["user_id"]]] += vector
            group_sums[int(expected[3])] += vector
    assert consumed == set(truth["truth"])
    averages = {n: float(sums[i] / len(consumed)) for i, n in enumerate(names)}
    per_user = (user_sums / truth["counts"][:, None]).mean(axis=0)
    user_average = {n: float(per_user[i]) for i, n in enumerate(names)}
    group_metrics = {name: dict(examples=int(truth["groups"][g]), metrics={n: float(group_sums[g, i] / truth["groups"][g]) for i, n in enumerate(names)})
                     for g, name in enumerate(("seen_targets", "cold_target"))}
    error_against_main = 0.0
    if full_audit:
        main_audit = read(point["audit_path"], point["audit_sha256"])
        assert main_audit["passed"] and main_audit["mismatches"] == 0
        assert main_audit["predictions_sha256"] == point["predictions_sha256"]
        assert main_audit["summary_sha256"] == point["summary_sha256"]
        assert main_audit["tolerance"] == ROW_TOL
        assert main_audit["source_labels_verified"] and main_audit["source_cold_groups_verified"]
        for field, actual in (("metrics", averages), ("user_average", user_average), ("groups", group_metrics)):
            maximum = max(maximum, close(actual, meta[field], label + "/summary/" + field, ROW_TOL))
            error_against_main = max(error_against_main, close(actual, main_audit[field], label + "/main/" + field))
            close(actual, point[field], label + "/report/" + field)
    cols = [idx[n] for n in PRIMARY]
    detail = dict(scope="all saved per-sample metrics and aggregates" if full_audit else "necessary contributions for new A-C pairs only",
                  examples=len(consumed), users=len(truth["users"]), predictions_sha256=point["predictions_sha256"],
                  summary_sha256=point["summary_sha256"], metrics=averages, max_error_vs_saved=maximum if full_audit else None,
                  max_error_vs_main_independent_audit=error_against_main if full_audit else None,
                  source_labels_and_cold_groups_verified=True)
    print(json.dumps(dict(progress="replayed", model=label, examples=len(consumed), full_audit=full_audit)), flush=True)
    return dict(sums=user_sums[:, cols], groups=group_sums[:, cols], truth=truth, details=detail,
                predictions_sha256=point["predictions_sha256"])


def paired(label, baseline, treatment, receipt_spec):
    receipt = read(receipt_spec["path"], receipt_spec["sha256"])
    counts = baseline["truth"]["counts"]
    assert np.array_equal(counts, treatment["truth"]["counts"])
    assert baseline["truth"]["users"] == treatment["truth"]["users"]
    assert receipt["inputs"] == dict(baseline_sha256=baseline["predictions_sha256"], treatment_sha256=treatment["predictions_sha256"])
    assert receipt["bootstrap"]["repeats"] == 2000 and receipt["bootstrap"]["seed"] == 2024
    assert receipt["bootstrap"]["confidence"] == .95 and receipt["bootstrap"]["quantile"] == "linear_percentile"
    assert receipt["examples"] == int(counts.sum()) and receipt["users"] == len(counts)
    assert set(receipt["primary"]) == set(PRIMARY) == set(receipt["user_average"])
    a, b = baseline["sums"], treatment["sums"]
    delta = b - a
    equal_user_delta = delta / counts[:, None]
    rng = np.random.default_rng(2024)
    draws_sample = np.empty((2000, len(PRIMARY)))
    draws_user = np.empty_like(draws_sample)
    # Same RNG stream and estimands, independently assembled from source-verified user sums.
    for begin in range(0, 2000, 50):
        end = min(begin + 50, 2000)
        sampled_users = rng.integers(0, len(counts), size=(end - begin, len(counts)))
        denominator = counts[sampled_users].sum(axis=1)
        for j in range(len(PRIMARY)):
            draws_sample[begin:end, j] = delta[sampled_users, j].sum(axis=1) / denominator
            draws_user[begin:end, j] = equal_user_delta[sampled_users, j].mean(axis=1)
    bounds = ((1 - .95) / 2, 1 - (1 - .95) / 2)
    primary, secondary = {}, {}
    for output, ua, ub, ud, draws in ((primary, a.sum(axis=0)/counts.sum(), b.sum(axis=0)/counts.sum(), delta.sum(axis=0)/counts.sum(), draws_sample),
                                      (secondary, (a/counts[:,None]).mean(axis=0), (b/counts[:,None]).mean(axis=0), equal_user_delta.mean(axis=0), draws_user)):
        ci = np.quantile(draws, bounds, axis=0)
        for j, name in enumerate(PRIMARY):
            output[name] = dict(baseline=float(ua[j]), treatment=float(ub[j]), delta=float(ud[j]),
                                relative_delta=float(ud[j] / ua[j]) if ua[j] else None,
                                confidence_interval=ci[:,j].tolist())
    error = max(close(primary, receipt["primary"], label + "/sample_mean"),
                close(secondary, receipt["user_average"], label + "/user_mean"))
    group_results = {}
    for gi, group in enumerate(("seen_targets", "cold_target")):
        n = int(baseline["truth"]["groups"][gi])
        ga, gb = baseline["groups"][gi], treatment["groups"][gi]
        group_results[group] = dict(examples=n, metrics={m: dict(baseline=float(ga[j]/n), treatment=float(gb[j]/n), delta=float((gb[j]-ga[j])/n)) for j,m in enumerate(PRIMARY)})
    error = max(error, close(group_results, receipt["groups"], label + "/groups"))
    print(json.dumps(dict(progress="bootstrap_verified", comparison=label, maximum_error=error)), flush=True)
    return dict(passed=True, repeats=2000, seed=2024, confidence=.95, examples=int(counts.sum()), users=len(counts),
                max_error_vs_primary_analysis=error, primary=primary, user_average=secondary,
                receipt_sha256=receipt_spec["sha256"])


def selections(models, c_report, final_report):
    result = {}
    for branch in ("A", "C"):
        points = {e: models[f"{branch}_R{e}_validation"] for e in GRID}
        values = {e: read(p["summary"], p["summary_sha256"])["metrics"]["list/ndcg@5"] for e,p in points.items()}
        best = min(GRID, key=lambda e: (-values[e], e))
        expected = 32 if branch == "A" else 23
        assert best == expected and final_report["selected_epochs"][branch] == best
        if branch == "C": assert c_report["selected_epoch"] == best
        fixed_point = models[f"{branch}_R35_validation"]
        branch_dir = next(p for p in Path(fixed_point["summary"]).parents if p.name == ("H1-Freeze20-R35" if branch == "A" else "H0-Freeze20-R35"))
        locked = read(branch_dir / "selection.json")
        assert locked["branch"] == branch and locked["criterion"] == "validation/list/ndcg@5; tie earlier"
        assert locked["fixed_endpoint"] == 35 and set(locked["candidates"]) == {str(e) for e in GRID}
        assert locked["selected"] == locked["candidates"][str(best)]
        for e, point in points.items():
            candidate = locked["candidates"][str(e)]
            assert candidate["epoch"] == e and candidate["ndcg5"] == values[e]
            for key in ("summary_sha256", "predictions_sha256", "checkpoint_sha256"):
                assert candidate[key] == point[key]
        # Test endpoint uses the validation-locked checkpoint and model fingerprint.
        for e in (best, 35):
            validation, test = points[e], models[f"{branch}_R{e}_test"]
            assert validation["checkpoint_sha256"] == test["checkpoint_sha256"]
            assert validation["model_fingerprint"] == test["model_fingerprint"]
        result[branch] = dict(best_epoch=best, rule="max validation list/ndcg@5 on R20/23/26/29/32/35; earliest tie", raw_validation_scores=values,
                              test_metrics_not_used=True, selection_sha256=BOUND[str((branch_dir/"selection.json").resolve())], selection_keys=sorted(locked))
    assert final_report["selected_epochs"]["B"] == 35
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-report", required=True)
    parser.add_argument("--final-report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert not output.exists(), "Receipt must be a fresh file"
    bind(__file__)
    c = read(args.c_report)
    final = read(args.final_report)
    assert c["status"] == final["status"] == "passed" and c["branch"] == "C"
    assert c["bootstrap"] == final["bootstrap"] == dict(repeats=2000, seed=2024, confidence=.95, unit="user cluster")
    assert c["inputs_unchanged"] and final["inputs_unchanged"]
    assert not c["torch_imported"] and not final["torch_imported"]
    expected_c = {f"C_R{e}_validation" for e in (5,8,11,14,17,20,23,26,29,32,35)} | {"C_R23_test", "C_R35_test"}
    assert set(c["models"]) == expected_c
    models = final["models"]
    for name, point in c["models"].items():
        assert point["predictions_sha256"] == models[name]["predictions_sha256"]
    selected = selections(models, c, final)
    data = {split: source(Path(args.dataset).resolve(), split) for split in ("validation", "test")}
    cache, model_checks = {}, {}
    for name in sorted(expected_c):
        cache[name] = replay(name, models[name], data[models[name]["split"]], True)
        model_checks[name] = cache[name]["details"]
    # A is read only for the four newly required A-C comparisons, never as a repeat branch audit.
    for epoch in (32,35):
        for split in ("validation", "test"):
            name = f"A_R{epoch}_{split}"
            cache[name] = replay(name, models[name], data[split], False)
            model_checks[name] = cache[name]["details"]
    specs = [("C_R35_minus_R5_validation", "C_R5_validation", "C_R35_validation", c),
             ("C_R35_minus_R20_validation", "C_R20_validation", "C_R35_validation", c)]
    for split in ("test", "validation"):
        specs += [(f"A_minus_C_R35_{split}", f"C_R35_{split}", f"A_R35_{split}", final),
                  (f"A_minus_C_validation_best_{split}", f"C_R23_{split}", f"A_R32_{split}", final)]
    comparisons = {label: paired(label, cache[a], cache[b], report["comparisons"][label]) for label,a,b,report in specs}
    for path, sha in BOUND.items():
        assert digest(path) == sha, f"Input changed during independent review: {path}"
    assert "torch" not in sys.modules
    receipt = dict(status="passed", format="freeze35_second_implementation_statistics_review_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                   source_and_implementation_sha256=BOUND, inputs_unchanged=True, source_metric_tolerance=ROW_TOL,
                   second_implementation_comparison_tolerance=SECOND_IMPLEMENTATION_TOL, selections=selected,
                   models=model_checks, comparisons=comparisons, torch_imported=False, gpu_or_model_inference=False,
                   c_models_fully_recomputed=13, a_models_used_only_for_new_pairs=4, existing_A_B_branch_audits_repeated=False,
                   final_A_B_pair_independently_recomputed=False,
                   scope="C full metric replays; C own-origin validation pairs; A-C fixed35 and validation-selected best pairs. Existing A/B branch audits and final A-B pair receipts are outside this second numerical replay.",
                   limitations="One training seed; user clustering does not measure training-seed variability. Best checkpoints use validation only; C R5/R20 origin comparisons use validation only.")
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", receipt=str(output), sha256=digest(output))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
