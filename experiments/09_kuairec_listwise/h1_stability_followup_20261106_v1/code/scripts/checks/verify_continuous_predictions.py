"""Independent stdlib metric audit from actual top-20 item lists and raw labels.

Does not import the trainer, model, cache producer or their metric routines.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_rows(predictions, truth, *, n_items=12101, expected_rows=22363):
    sums = {f"{metric}@{k}": 0.0 for metric in ("HR", "NDCG") for k in (5, 10, 20)}
    total_nll, count, seen = 0.0, 0, set()
    sentinel = object()
    from itertools import zip_longest
    for prediction, label in zip_longest(predictions, truth, fillvalue=sentinel):
        if prediction is sentinel or label is sentinel:
            raise ValueError("Prediction/label row count differs")
        identity = (label["user_id"], label["target_position"])
        if identity in seen:
            raise ValueError("Duplicate held-out sample")
        seen.add(identity)
        if any(prediction[key] != label[key] for key in ("user_id", "target_position", "target")):
            raise ValueError("Prediction identities/targets differ from original held-out rows")
        top = prediction["top_items"]
        if len(top) != min(20, n_items) or len(set(top)) != len(top):
            raise ValueError("Top-K list must be complete and contain unique items")
        if any(type(item) is not int or not 1 <= item <= n_items for item in top):
            raise ValueError("PAD or invalid item recommended")
        rank = top.index(label["target"]) + 1 if label["target"] in top else n_items + 1
        reported = prediction["rank"]
        if type(reported) is not int or not 1 <= reported <= n_items:
            raise ValueError("Invalid reported full-catalog rank")
        if (rank <= 20 and rank != reported) or (rank > 20 and reported <= 20):
            raise ValueError("Reported rank contradicts actual Top-20 recommendations")
        for k in (5, 10, 20):
            if rank <= k:
                sums[f"HR@{k}"] += 1
                sums[f"NDCG@{k}"] += 1 / math.log2(rank + 1)
        row_nll = prediction["nll"]
        if not isinstance(row_nll, (int, float)) or not math.isfinite(row_nll) or row_nll < 0:
            raise ValueError("Nonfinite/negative per-sample NLL")
        total_nll += row_nll
        count += 1
    if count != expected_rows:
        raise ValueError("Incomplete held-out evaluation")
    return {"samples": count, "NLL": total_nll / count,
            **{key: value / count for key, value in sums.items()}}


def audit_predictions(directory, data, protocol, split):
    directory, data = Path(directory), Path(data)
    accepted = "6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa"
    if digest(data / "manifest.json") != accepted:
        raise ValueError("Original dataset identity changed")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    relative = protocol + "/" + split + ".jsonl"
    truth_path = data / relative
    if digest(truth_path) != manifest["files"][relative]["sha256"]:
        raise ValueError("Held-out labels differ from accepted data")
    reported = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if digest(directory / "predictions.jsonl") != reported["prediction_sha256"]:
        raise ValueError("Prediction bytes differ from evaluation receipt")
    if reported["split"] != split or reported["binding"]["protocol"] != protocol:
        raise ValueError("Evaluation protocol/split binding mismatch")
    with (directory / "predictions.jsonl").open(encoding="utf-8") as left, truth_path.open(encoding="utf-8") as right:
        recomputed = audit_rows((json.loads(line) for line in left), (json.loads(line) for line in right))
    for key, value in recomputed.items():
        tolerance = 1e-5 if key == "NLL" else 1e-12
        if not math.isfinite(reported[key]) or abs(reported[key] - value) > tolerance:
            raise ValueError(f"Independent recomputation differs: {key}")
    return {"status": "PASS", "split": split, "protocol": protocol,
            "recomputed": recomputed, "prediction_sha256": reported["prediction_sha256"],
            "truth_sha256": digest(truth_path), "metrics_derived_from_top_items": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--protocol", choices=["full_train_prefix_v1", "legacy_prefix_v1"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = audit_predictions(args.evaluation, args.data, args.protocol, args.split)
    if args.report:
        with Path(args.report).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))
