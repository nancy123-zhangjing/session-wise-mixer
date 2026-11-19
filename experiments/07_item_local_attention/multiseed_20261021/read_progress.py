"""Read this campaign's progress without importing Torch or changing artifacts."""
import argparse
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def tail_lines(path):
    if not path.is_file():
        return []
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 3000))
        return [s for s in source.read().decode("utf-8", "replace").splitlines() if s.strip()][-3:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    rows = []
    for seed in (2025, 2026):
        for arm in ("control", "item_local"):
            key = (f"smoke_seed{seed}_attempt1" if args.mode == "smoke"
                   else f"formal_seed{seed}_{arm}_attempt1")
            delivery = args.goal_root / f"seeds/seed{seed}/delivery" / key
            receipt = read(delivery / "receipt.json", {})
            run = args.goal_root / f"seeds/seed{seed}/runs/stage07_item_local_joint_{arm}_seed{seed}_20261021_r1"
            if args.mode == "smoke":
                run /= "smoke"
            phases = read(run / "phase_integrity.json", [])
            tokenizer = [{k: phase.get(k) for k in ("epoch", "optimizer_updates", "direct_recommendation_gradient_norm",
                          "frozen_parameters_unchanged", "all_active_gradients_finite", "SID_integrity")}
                         for phase in phases if phase.get("phase") == "tokenizer"]
            evaluations = read(run / "evaluation_timings.json", [])
            result = read(run / ("smoke_results.json" if args.mode == "smoke" else "results.json"), {})
            initial = read(run / "initial_state_identity.json", {})
            log = delivery / f"train_{arm}.log"
            validation_cells = []
            for beam in (32, 256):
                cell = run / f"validation/beam{beam}"
                if cell.is_dir():
                    complete = read(cell / "CELL_COMPLETE.json", {})
                    validation_cells.append(dict(beam=beam, status=complete.get("status", "pending_or_running"),
                                                 log_tail=tail_lines(cell / "worker.log")))
            rows.append(dict(seed=seed, arm=arm, job_status=receipt.get("status", "not_started"),
                             children=[{k: c.get(k) for k in ("label", "status", "returncode")} for c in receipt.get("children", [])],
                             completed_phases=len(phases), last_phase=phases[-1].get("phase") if phases else None,
                             recommender_updates=sum(p.get("optimizer_updates", 0) for p in phases if p.get("phase") == "recommender"),
                             tokenizer_checks=tokenizer,
                             evaluation_epochs=[v.get("epoch") for v in evaluations],
                             initial={k: v for k, v in initial.items() if k.startswith("initial_")},
                             training_status=result.get("status"), best_epoch=result.get("fit", {}).get("best_epoch"),
                             validation_complete=(run / "validation/EVALUATION_COMPLETE.json").is_file(),
                             validation_cells=validation_cells,
                             error=receipt.get("error"), log_tail=tail_lines(log)))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
