"""Read-only CPU comparison of committed SIDs with FP32 and FP64 distances."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    result = {
        "status": "DIAGNOSTIC_ONLY",
        "utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "states": [],
        "model_or_SID_modified": False,
    }
    for variant in ("control", "item_local"):
        run = args.goal_root / "runs" / (
            f"stage07_item_local_joint_{variant}_seed2024_20261020_r1"
        )
        for kind, name in (("best", "best_model.bin"), ("last", "last.pt")):
            path = run / "training_state" / name
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            state = loaded if kind == "best" else loaded["model"]
            codes = state["joint_alignment.hard_codes"]
            books = state["joint_alignment.mixer.base_codebooks"]
            slots = state["joint_alignment.item_slots"]
            row = {
                "variant": variant,
                "kind": kind,
                "path": str(path),
                "sha256": sha(path),
                "dtypes": [str(value.dtype) for value in (books, slots)],
                "code_positions": int(codes[1:].numel()),
                "saved_vs_fp64_argmin": 0,
                "saved_vs_fp32_cpu_argmin": 0,
                "old_tolerance_failures": 0,
                "max_old_tolerance_ratio": 0.0,
                "examples": [],
            }
            for start in range(1, len(slots), 2048):
                original = slots[start:start + 2048]
                saved = codes[start:start + len(original)]
                x, c = original.double(), books.double()
                distances = (
                    x.square().sum(-1, keepdim=True)
                    + c.square().sum(-1).unsqueeze(0)
                    - 2 * torch.einsum("bmd,mkd->bmk", x, c)
                )
                nearest = distances.argmin(-1)
                chosen = distances.gather(-1, saved.unsqueeze(-1)).squeeze(-1)
                best = distances.min(-1).values
                gap = chosen - best
                tolerance = 1e-6 * (1 + chosen.abs())
                f = (
                    original.square().sum(-1, keepdim=True)
                    + books.square().sum(-1).unsqueeze(0)
                    - 2 * torch.einsum("bmd,mkd->bmk", original, books)
                ).clamp_min(0)
                nearest32 = f.argmin(-1)
                bad = gap > tolerance
                row["saved_vs_fp64_argmin"] += int((saved != nearest).sum())
                row["saved_vs_fp32_cpu_argmin"] += int((saved != nearest32).sum())
                row["old_tolerance_failures"] += int(bad.sum())
                row["max_old_tolerance_ratio"] = max(
                    row["max_old_tolerance_ratio"], float((gap / tolerance).max())
                )
                for offset, digit in torch.nonzero(bad).tolist():
                    if len(row["examples"]) >= 32:
                        break
                    observed, optimal = int(saved[offset, digit]), int(nearest[offset, digit])
                    v = x[offset, digit]
                    saved_book, best_book = c[digit, observed], c[digit, optimal]
                    row["examples"].append({
                        "item_id": start + offset,
                        "digit": digit,
                        "saved_code": observed,
                        "fp64_argmin": optimal,
                        "fp32_cpu_argmin": int(nearest32[offset, digit]),
                        "chosen_distance64": float(chosen[offset, digit]),
                        "min_distance64": float(best[offset, digit]),
                        "gap64": float(gap[offset, digit]),
                        "old_tolerance": float(tolerance[offset, digit]),
                        "chosen_distance32": float(f[offset, digit, observed]),
                        "optimal_distance32": float(f[offset, digit, optimal]),
                        "direct_chosen_distance64": float((v - saved_book).square().sum()),
                        "direct_optimal_distance64": float((v - best_book).square().sum()),
                        "arithmetic_scale_chosen": float(
                            v.square().sum() + saved_book.square().sum()
                            + 2 * (v * saved_book).abs().sum()
                        ),
                        "arithmetic_scale_optimal": float(
                            v.square().sum() + best_book.square().sum()
                            + 2 * (v * best_book).abs().sum()
                        ),
                    })
            result["states"].append(row)
            del loaded, state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
