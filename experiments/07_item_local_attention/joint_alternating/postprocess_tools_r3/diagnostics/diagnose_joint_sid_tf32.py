"""CPU-only SID precision diagnosis; never refresh, restore, evaluate or train.

TF32 variants round only dot-product operands to ten fraction bits; norm terms
retain original FP32 inputs. CPU accumulation is not a CUDA kernel replay.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_tf32(values, mode):
    """Return fresh FP32 ndarray, rounding magnitude; preserve signed zero."""
    import numpy as np
    need(mode in ("nearest_even", "nearest_ties_away"), "Unknown rounding rule")
    values = np.ascontiguousarray(values, dtype=np.float32)
    need(bool(np.isfinite(values).all()), "Nonfinite input to mantissa rounding")
    bits = values.view(np.uint32)
    remainder = bits & np.uint32(0x1FFF)
    greater = remainder > np.uint32(0x1000)
    tie = remainder == np.uint32(0x1000)
    odd = ((bits >> np.uint32(13)) & np.uint32(1)) != 0
    up = greater | (tie & odd) if mode == "nearest_even" else greater | tie
    rounded = (bits & np.uint32(0xFFFFE000)) + up.astype(np.uint32) * np.uint32(0x2000)
    result = rounded.view(np.float32).copy()
    need(bool(np.isfinite(result).all()), "Mantissa rounding overflowed")
    return result


def selfcheck_rounding():
    """Hand-specified IEEE-754 bit cases, independent of distance data."""
    import numpy as np
    cases = [
        ("positive_zero", 0x00000000, 0x00000000, 0x00000000),
        ("negative_zero", 0x80000000, 0x80000000, 0x80000000),
        ("positive_exact", 0x3F800000, 0x3F800000, 0x3F800000),
        ("negative_exact", 0xBF800000, 0xBF800000, 0xBF800000),
        ("positive_below_half", 0x3F800FFF, 0x3F800000, 0x3F800000),
        ("positive_above_half", 0x3F801001, 0x3F802000, 0x3F802000),
        ("negative_below_half", 0xBF800FFF, 0xBF800000, 0xBF800000),
        ("negative_above_half", 0xBF801001, 0xBF802000, 0xBF802000),
        ("positive_even_tie", 0x3F801000, 0x3F800000, 0x3F802000),
        ("negative_even_tie", 0xBF801000, 0xBF800000, 0xBF802000),
        ("positive_odd_tie", 0x3F803000, 0x3F804000, 0x3F804000),
        ("negative_odd_tie", 0xBF803000, 0xBF804000, 0xBF804000),
        ("fractional_intermediate", 0x3FA66666, 0x3FA66000, 0x3FA66000),
        ("positive_subnormal_tie", 0x00001000, 0x00000000, 0x00002000),
        ("negative_subnormal_tie", 0x80001000, 0x80000000, 0x80002000),
    ]
    bits = np.array([row[1] for row in cases], dtype=np.uint32)
    before = bits.tobytes()
    for mode, index in (("nearest_even", 2), ("nearest_ties_away", 3)):
        observed = round_tf32(bits.view(np.float32), mode).view(np.uint32)
        need(np.array_equal(observed, np.array([row[index] for row in cases], dtype=np.uint32)), "Hand-specified rounding case failed: " + mode)
        need(np.array_equal(round_tf32(observed.view(np.float32), mode).view(np.uint32), observed), "Rounding was not idempotent")
    need(before == bits.tobytes(), "Rounding mutated its input")
    return {"status": "CPU_BIT_ROUNDING_SELFCHECK_PASS", "cases": [
        {"name": name, "input_bits": f"{original:08x}", "nearest_even_bits": f"{even:08x}", "nearest_ties_away_bits": f"{away:08x}"}
        for name, original, even, away in cases], "input_unchanged": True,
        "both_modes_idempotent": True, "GPU_or_platform_used": False,
        "scope": "Mantissa rounding unit cases only; no real checkpoint or CUDA acceptance."}


def inspect_checkpoint(path, variant, kind, torch):
    before_hash = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if kind == "last" else payload
    slots, books, codes = (state["joint_alignment." + key] for key in ("item_slots", "mixer.base_codebooks", "hard_codes"))
    need(slots.dtype == books.dtype == torch.float32 and codes.dtype == torch.int64, "Unexpected dtype")
    need(tuple(slots.shape) == (12102, 4, 64) and tuple(books.shape) == (4, 256, 64) and tuple(codes.shape) == (12102, 4), "Wrong catalogue/quantizer geometry")
    need(all(value.device.type == "cpu" for value in (slots, books, codes)), "Non-CPU checkpoint")
    need(bool(torch.isfinite(slots).all() and torch.isfinite(books).all()) and bool((codes[0] == -1).all()) and bool(((codes[1:] >= 0) & (codes[1:] < 256)).all()), "Malformed checkpoint values")
    methods = ("strict_fp32_cpu", "fp64_reference", "tf32_nearest_even_cpu", "tf32_nearest_ties_away_cpu")
    totals = {name: {"argmin_mismatches": 0, "saved_not_minimum_even_allowing_exact_ties": 0,
              "max_saved_distance_gap": 0.0, "minimum_top2_margin": None} for name in methods}
    rounded_books = {mode: torch.from_numpy(round_tf32(books.numpy(), mode)) for mode in ("nearest_even", "nearest_ties_away")}
    books64 = books.double()
    book_norm32 = books.square().sum(-1).unsqueeze(0)
    examples, positions, fp32_vs_fp64 = [], 0, 0
    for start in range(1, slots.shape[0], 2048):
        x = slots[start:start + 2048]
        saved = codes[start:start + len(x)]
        x64 = x.double()
        norm32 = x.square().sum(-1, keepdim=True) + book_norm32
        distances = {
            "strict_fp32_cpu": (norm32 - 2.0 * torch.einsum("bmd,mkd->bmk", x, books)).clamp_min(0),
            "fp64_reference": x64.square().sum(-1, keepdim=True) + books64.square().sum(-1).unsqueeze(0) - 2.0 * torch.einsum("bmd,mkd->bmk", x64, books64),
        }
        for mode, quantized_books in rounded_books.items():
            quantized_x = torch.from_numpy(round_tf32(x.numpy(), mode))
            distances["tf32_" + mode + "_cpu"] = (norm32 - 2.0 * torch.einsum("bmd,mkd->bmk", quantized_x, quantized_books)).clamp_min(0)
        minima = {name: distance.min(-1) for name, distance in distances.items()}
        fp32_vs_fp64 += int((minima[methods[0]].indices != minima[methods[1]].indices).sum())
        mismatched = torch.zeros_like(saved, dtype=torch.bool)
        details = {}
        for name, distance in distances.items():
            need(bool(torch.isfinite(distance).all()), "Nonfinite distance arithmetic")
            minimum, argmin = minima[name]
            chosen = distance.gather(-1, saved.unsqueeze(-1)).squeeze(-1)
            margin = distance.topk(2, dim=-1, largest=False).values.diff(dim=-1).squeeze(-1)
            gap = chosen - minimum
            need(bool((gap >= 0).all()), "Invalid distance gap")
            mismatch = argmin != saved
            mismatched |= mismatch
            summary = totals[name]
            summary["argmin_mismatches"] += int(mismatch.sum())
            summary["saved_not_minimum_even_allowing_exact_ties"] += int((gap > 0).sum())
            summary["max_saved_distance_gap"] = max(summary["max_saved_distance_gap"], float(gap.max()))
            current = float(margin.min())
            summary["minimum_top2_margin"] = current if summary["minimum_top2_margin"] is None else min(summary["minimum_top2_margin"], current)
            details[name] = (argmin, chosen, minimum, gap, margin)
        positions += saved.numel()
        for row, digit in mismatched.nonzero().tolist():
            exact_code = int(minima["fp64_reference"].indices[row, digit])
            saved_code = int(saved[row, digit])
            record = {"item_id": start + row, "digit_zero_based": digit, "saved_code": saved_code,
                      "fp64_direct_saved_distance": float((x64[row, digit] - books64[digit, saved_code]).square().sum()),
                      "fp64_direct_optimal_distance": float((x64[row, digit] - books64[digit, exact_code]).square().sum()),
                      "arithmetic": {}}
            for name, (argmin, chosen, minimum, gap, margin) in details.items():
                record["arithmetic"][name] = {"argmin": int(argmin[row, digit]), "saved_distance": float(chosen[row, digit]),
                    "minimum_distance": float(minimum[row, digit]), "saved_minus_minimum": float(gap[row, digit]),
                    "top2_margin": float(margin[row, digit]),
                    "strict_fp64_optimal_code_distance": float(distances[name][row, digit, exact_code])}
            examples.append(record)
    need(sha(path) == before_hash, "Source checkpoint bytes changed during read-only diagnosis")
    return {"variant": variant, "kind": kind, "path": str(path), "sha256": before_hash, "size_bytes": path.stat().st_size,
            "code_positions": positions, "diger_optimizer_step": int(state["joint_alignment.diger_optimizer_step"]),
            "strict_fp32_vs_fp64_argmin_mismatches": fp32_vs_fp64, "methods": totals,
            "all_union_mismatch_examples": examples, "union_mismatch_examples_count": len(examples),
            "source_sha256_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not os.path.lexists(args.output), "Choose a new diagnostic receipt")
    os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none")
    import torch
    import numpy as np
    torch.set_num_threads(4)
    result = {"status": "DIAGNOSTIC_ONLY", "utc": datetime.now(timezone.utc).isoformat(), "tool_sha256": sha(__file__),
              "goal_root": str(args.goal_root.resolve()), "torch": str(torch.__version__), "numpy": str(np.__version__),
              "CPU_threads": torch.get_num_threads(), "states": [], "model_or_SID_modified": False,
              "scope": "CPU operand-rounding emulation only; not exact CUDA/TF32 replay or proof of historical CUDA flags. No tolerance-based acceptance.",
              "arithmetic": "Original FP32 squared norms; only dot operands rounded from 23 to 10 fraction bits; CPU FP32 dot/combination then clamp_min(0). Strict FP64 reference retains the prior auditor's unclamped formula.",
              "rounding_selfcheck": selfcheck_rounding(), "GPU_used": False}
    code = 0
    try:
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            for variant in ("control", "item_local"):
                run = args.goal_root / "runs" / ("stage07_item_local_joint_" + variant + "_seed2024_20261020_r1")
                for kind, name in (("best", "best_model.bin"), ("last", "last.pt")):
                    result["states"].append(inspect_checkpoint(run / "training_state" / name, variant, kind, torch))
    except Exception as error:
        result.update(status="DIAGNOSTIC_FAILED", error_type=type(error).__name__, error=str(error))
        code = 1
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha(args.output),
                      "states": [{"variant": state["variant"], "kind": state["kind"], "methods": state["methods"]} for state in result["states"]]}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
