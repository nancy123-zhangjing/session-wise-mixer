"""Read-only CPU diagnostic for DIGER noisy and deterministic SID assignments.

No model training, dataset cache refresh, or historical artifact mutation.
The exported joint codebooks belong to the validation-selected BEST checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_catalog(array, source_keys, catalog_keys):
    if len(set(source_keys)) != len(source_keys) or len(set(catalog_keys)) != len(catalog_keys):
        raise ValueError("Duplicate catalogue keys")
    if set(source_keys) != set(catalog_keys) or len(array) != len(source_keys):
        raise ValueError("Catalogue identity or row count mismatch")
    index = {key: row for row, key in enumerate(source_keys)}
    return array[np.asarray([index[key] for key in catalog_keys])]


def squared_distances(slots, books):
    if slots.ndim != 3 or books.ndim != 3:
        raise ValueError("Expected slots [B,M,D], books [M,K,D]")
    if slots.shape[1] != books.shape[0] or slots.shape[2] != books.shape[2]:
        raise ValueError("Slot/codebook shape mismatch")
    item_sq = slots.square().sum(dim=-1, keepdim=True)
    code_sq = books.square().sum(dim=-1).unsqueeze(0)
    dot = torch.einsum("bmd,mkd->bmk", slots, books)
    return (item_sq + code_sq - 2.0 * dot).clamp_min(0.0)


def sample_hard(logits, noise_scale, softmax_temperature, generator):
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative")
    if not math.isfinite(softmax_temperature) or softmax_temperature <= 0:
        raise ValueError("Softmax temperature must be finite and positive")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    noisy_logits = logits
    if noise_scale > 0:
        uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device,
                             generator=generator).clamp_(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        noisy_logits = logits + noise_scale * gumbel
    # Follow the actual runtime softmax -> argmax, not a shortcut under ties.
    return torch.softmax(noisy_logits / softmax_temperature, dim=-1).argmax(dim=-1)


def analytic_agreement(logits, noise_scale):
    """Ideal, unclipped Gumbel reference; empirical clipped sampling is primary."""
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    if noise_scale == 0:
        digit = torch.ones(logits.shape[:-1], dtype=torch.float64)
        pair = digit.clone()
    else:
        probabilities = torch.softmax(logits.double() / noise_scale, dim=-1)
        digit = probabilities.gather(-1, logits.argmax(-1).unsqueeze(-1)).squeeze(-1)
        pair = probabilities.square().sum(-1)
    return {"deterministic_digit_match": digit,
            "deterministic_item_match": digit.prod(-1),
            "pair_digit_match": pair, "pair_item_match": pair.prod(-1)}


def compare_codes(initial, current):
    if initial.shape != current.shape or initial.ndim != 2 or initial.size == 0:
        raise ValueError("Expected matching nonempty SID arrays [N,M]")
    changed = initial != current
    return {"changed_items": int(changed.any(-1).sum()),
            "item_change_rate": float(changed.any(-1).mean()),
            "changed_digits": int(changed.sum()),
            "digit_change_rate": float(changed.mean()),
            "per_position_change_rate": changed.mean(0).tolist()}


def validation_coverage(sequences, catalog_keys, changed_mask,
                        max_history_len=50, min_seq_len=2):
    if len(set(catalog_keys)) != len(catalog_keys):
        raise ValueError("Duplicate catalogue keys")
    if changed_mask.shape != (len(catalog_keys),) or max_history_len <= 0 or min_seq_len < 2:
        raise ValueError("Invalid coverage mask or validation protocol")
    changed_by_key = dict(zip(catalog_keys, changed_mask.tolist()))
    counts = {"examples": 0, "target_changed_examples": 0,
              "history_changed_examples": 0, "either_changed_examples": 0}
    for seq in sequences.values():
        if len(seq) < min_seq_len:
            continue
        target = seq[-2]
        history = seq[:-2][-max_history_len:]
        if any(key not in changed_by_key for key in [target, *history]):
            raise ValueError("Validation item missing from catalogue")
        target_changed = bool(changed_by_key[target])
        history_changed = any(changed_by_key[key] for key in history)
        counts["examples"] += 1
        counts["target_changed_examples"] += int(target_changed)
        counts["history_changed_examples"] += int(history_changed)
        counts["either_changed_examples"] += int(target_changed or history_changed)
    denominator = counts["examples"]
    if not denominator:
        raise ValueError("No validation examples")
    for name in ("target", "history", "either"):
        counts[name + "_changed_rate"] = counts[name + "_changed_examples"] / denominator
    return counts


def git_state(root):
    def run(args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run(["rev-parse", "HEAD"]), "status": run(["status", "--short"])}


def summarize_probability(reference):
    return {"digit_disagreement_rate": float(1 - reference["deterministic_digit_match"].mean()),
            "per_position_disagreement_rate": (1 - reference["deterministic_digit_match"].mean(0)).tolist(),
            "item_disagreement_rate": float(1 - reference["deterministic_item_match"].mean()),
            "independent_pair_digit_disagreement_rate": float(1 - reference["pair_digit_match"].mean()),
            "independent_pair_item_disagreement_rate": float(1 - reference["pair_item_match"].mean())}


def sample_statistics(logits, noise_scale, softmax_temperature, pairs, batch_size, seed):
    deterministic = logits.argmax(-1)
    n_item, n_digit, _ = logits.shape
    mismatch = np.zeros((pairs * 2, n_digit), dtype=np.int64)
    item_mismatch = np.zeros(pairs * 2, dtype=np.int64)
    pair_mismatch = np.zeros((pairs, n_digit), dtype=np.int64)
    pair_item_mismatch = np.zeros(pairs, dtype=np.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, n_item, batch_size):
        chunk = logits[start:start + batch_size]
        base = deterministic[start:start + batch_size]
        for pair in range(pairs):
            a = sample_hard(chunk, noise_scale, softmax_temperature, generator)
            b = sample_hard(chunk, noise_scale, softmax_temperature, generator)
            for draw, indices in ((pair * 2, a), (pair * 2 + 1, b)):
                diff = indices != base
                mismatch[draw] += diff.sum(0).numpy()
                item_mismatch[draw] += int(diff.any(-1).sum())
            diff_pair = a != b
            pair_mismatch[pair] += diff_pair.sum(0).numpy()
            pair_item_mismatch[pair] += int(diff_pair.any(-1).sum())
    digit_rate = mismatch.sum(-1) / (n_item * n_digit)
    item_rate = item_mismatch / n_item
    return {"seed": seed, "draws_per_item": pairs * 2, "independent_pairs": pairs,
            "digit_disagreement_rate": float(digit_rate.mean()),
            "per_position_disagreement_rate": (mismatch.sum(0) / (n_item * pairs * 2)).tolist(),
            "item_disagreement_rate": float(item_rate.mean()),
            "digit_rate_monte_carlo_standard_error": float(digit_rate.std(ddof=1) / math.sqrt(len(digit_rate))),
            "item_rate_monte_carlo_standard_error": float(item_rate.std(ddof=1) / math.sqrt(len(item_rate))),
            "independent_pair_digit_disagreement_rate": float(pair_mismatch.sum() / (pairs * n_item * n_digit)),
            "independent_pair_item_disagreement_rate": float(pair_item_mismatch.mean() / n_item)}


def run_diagnostic(root, config_path, output_dir):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pairs = int(config.get("sampling_pairs", 16))
    batch_size = int(config.get("batch_size", 256))
    threads = int(config.get("cpu_threads", 4))
    if pairs < 2 or batch_size <= 0 or threads <= 0:
        raise ValueError("Invalid sampling budget")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    inputs = {}
    lock = json.loads((root / config["asset_lock"]).read_text(encoding="utf-8"))
    locked = lock["artifacts"]

    def input_path(relative):
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if relative not in inputs:
            digest = sha256(path)
            expected = locked.get(relative, {}).get("sha256")
            if expected and digest != expected:
                raise ValueError(f"Historical asset hash mismatch: {relative}")
            inputs[relative] = {"sha256": digest, "size_bytes": path.stat().st_size,
                                "historical_lock_match": True if expected else None}
        return path

    keys = json.loads(input_path(config["item_keys"]).read_text(encoding="utf-8"))
    mapping = json.loads(input_path(config["id_mapping"]).read_text(encoding="utf-8"))
    catalog = mapping["id2item"][1:]
    if len(catalog) != int(config["expected_items"]):
        raise ValueError("Unexpected catalogue size")
    sequences = json.loads(input_path(config["sequences"]).read_text(encoding="utf-8"))
    families = {}
    for name, family in config["families"].items():
        slots = np.load(input_path(family["slots"]), allow_pickle=False).astype(np.float32)
        slots = align_catalog(slots, keys, catalog)
        initial_codes = align_catalog(np.load(input_path(family["initial_codes"]), allow_pickle=False), keys, catalog)
        families[name] = {"slots": slots, "initial_codes": initial_codes}
    results = []
    baseline_books = {}
    output_dir.mkdir(parents=True, exist_ok=False)
    for snapshot in config["snapshots"]:
        name, family_name = snapshot["name"], snapshot["family"]
        print(f"[DIAGNOSTIC] {name}: read and verify BEST export", flush=True)
        meta = json.loads(input_path(snapshot["metadata"]).read_text(encoding="utf-8"))
        books_np = np.load(input_path(snapshot["codebooks"]), allow_pickle=False).astype(np.float32)
        saved_codes = np.load(input_path(snapshot["saved_codes"]), allow_pickle=False)
        family = families[family_name]
        n_digit, n_code, code_dim = books_np.shape
        if books_np.shape != (4, 256, 64) or family["slots"].shape != (len(catalog), n_digit * code_dim):
            raise ValueError("Unexpected formal tokenizer shape")
        books = torch.from_numpy(books_np)
        slots = torch.from_numpy(family["slots"].reshape(len(catalog), n_digit, code_dim))
        if not torch.isfinite(books).all() or not torch.isfinite(slots).all():
            raise ValueError("Non-finite codebooks or slots")
        distances = torch.cat([squared_distances(chunk, books) for chunk in slots.split(batch_size)])
        recomputed = distances.argmin(-1).numpy()
        # Tiny CPU/GPU rounding differences can affect a boundary; do not silently ignore them.
        recompute_check = compare_codes(saved_codes, recomputed)
        if recompute_check["changed_items"]:
            raise ValueError(f"Saved/recomputed SID mismatch for {name}: {recompute_check}")
        changed_mask = (family["initial_codes"] != saved_codes).any(-1)
        coverage = validation_coverage(sequences, catalog, changed_mask,
                                       config["max_history_len"], 2)
        if coverage["examples"] != int(config["expected_validation_examples"]):
            raise ValueError("Validation denominator differs from formal dataset")
        if snapshot.get("baseline"):
            if changed_mask.any():
                raise ValueError("Frozen baseline differs from offline initial SID")
            baseline_books[family_name] = books_np
        if family_name not in baseline_books:
            raise ValueError("Family frozen baseline must precede joint snapshots")
        book_delta = books_np.astype(np.float64) - baseline_books[family_name].astype(np.float64)
        baseline_rms = np.sqrt(np.mean(baseline_books[family_name].astype(np.float64) ** 2))
        state = meta["joint_alignment"]["diger_final_schedule_state"]
        assignment_temp = float(meta["joint_alignment"]["diger_assignment_temperature"])
        softmax_temp = float(state["softmax_temperature"])
        if assignment_temp <= 0 or not math.isfinite(assignment_temp):
            raise ValueError("Invalid assignment temperature")
        logits = -distances / assignment_temp
        nearest = distances.topk(2, largest=False).values
        margin = (nearest[..., 1] - nearest[..., 0]) / assignment_temp
        row = {"name": name, "family": family_name, "run_id": meta["run_id"],
               "best_epoch": int(meta["best_epoch"]), "schedule_at_best_export": state,
               "export_semantics": "validation_selected_best_not_last_training_state",
               "saved_sid_recompute_check": recompute_check,
               "deterministic_change_vs_offline_initial": compare_codes(family["initial_codes"], saved_codes),
               "relative_effective_codebook_delta_rms": float(np.sqrt(np.mean(book_delta ** 2)) / max(baseline_rms, 1e-12)),
               "validation_coverage": coverage,
               "normalized_nearest_second_margin": {"p05": float(torch.quantile(margin, .05)),
                    "median": float(torch.quantile(margin, .5)), "p95": float(torch.quantile(margin, .95)),
                    "per_position_median": torch.quantile(margin, .5, dim=0).tolist()},
               "noise_statistics": []}
        noise_specs = [{"scale": 0.0, "role": "no_noise_control", "sample": False}]
        if not snapshot.get("baseline"):
            noise_specs.append({"scale": float(state["gumbel_noise_scale"]),
                               "role": "best_snapshot_conditional_noise_probe" if state["optimizer_step"] else "hypothetical_first_tokenizer_phase",
                               "sample": True})
            for noise in snapshot.get("additional_noise_scales", []):
                noise_specs.append({"scale": float(noise), "role": "fixed_snapshot_counterfactual_not_new_training", "sample": True})
        for probe_index, probe in enumerate(noise_specs):
            noise = probe["scale"]
            print(f"[DIAGNOSTIC] {name}: noise={noise:.6f}, role={probe['role']}", flush=True)
            stats = {"noise_scale": noise, "role": probe["role"],
                     "assignment_temperature": assignment_temp, "softmax_temperature": softmax_temp,
                     "ideal_gumbel_reference": summarize_probability(analytic_agreement(logits, noise))}
            if probe["sample"]:
                seed = int(config["sampling_seed"]) + len(results) * 100 + probe_index
                stats["empirical_clipped_gumbel"] = sample_statistics(logits, noise, softmax_temp, pairs, batch_size, seed)
            else:
                hard = sample_hard(logits, 0, softmax_temp, torch.Generator().manual_seed(0))
                stats["zero_noise_exact_match"] = bool(torch.equal(hard, torch.from_numpy(saved_codes)))
                if not stats["zero_noise_exact_match"]:
                    raise ValueError("No-noise control failed")
            row["noise_statistics"].append(stats)
        row["changed_item_keys"] = [key for key, flag in zip(catalog, changed_mask) if flag]
        results.append(row)
    result = {"schema_version": 1, "diagnostic": "diger_sid_consistency_cpu",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "items": len(catalog), "sid_positions": 4, "codes_per_position": 256,
              "config": config, "input_files": inputs,
              "provenance": {"project_root": str(root), "config_sha256": sha256(config_path),
                  "script_sha256": sha256(Path(__file__)), "python": platform.python_version(),
                  "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                  "cpu_threads": threads, "root_git": git_state(root),
                  "engine_git": git_state(root / "external/DiffGRM"),
                  "source_delivery": "standalone_diagnostic_copy_not_full_repository_sync"},
              "sampling_scope": "catalogue_uniform_not_training_occurrence_weighted; independent calls sampled independently",
              "analytic_reference_limit": "ideal unclipped Gumbel; actual training clips Uniform and empirical sampler follows that clipping",
              "limitations": ["No reconstruction of unsaved post-best training codebooks",
                  "Fixed-snapshot simulation, not replay of historical RNG or training batches",
                  "No recommender forward pass or new HR/NDCG evaluation",
                  "Monte Carlo error is sampling error, not across-training-seed uncertainty",
                  "Validation coverage excludes reverse-map/legal-graph effects on otherwise unchanged examples"],
              "elapsed_seconds": time.perf_counter() - started, "snapshots": results}
    for filename, content in (("summary.json", result), ("config.json", config)):
        with (output_dir / filename).open("x", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    manifest = {name: sha256(output_dir / name) for name in ("summary.json", "config.json")}
    with (output_dir / "checksums.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"[DONE] {output_dir}; {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostic(args.project_root.resolve(), args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
