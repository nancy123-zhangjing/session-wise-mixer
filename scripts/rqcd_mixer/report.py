"""Read completed local run artifacts; never run a model to produce a report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import sha256, write_json


def report(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, curve_rows, event_sets, pretrain_events = [], [], {}, {}
    shared_teacher = None
    for path in runs:
        root = Path(path)
        done = json.loads((root/"TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
        tests = json.loads((root/"TEST_COMPLETE.json").read_text(encoding="utf-8"))
        best = json.loads((root/"BEST.json").read_text(encoding="utf-8"))
        arm = done["arm"]
        config_path = root/"config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if "target" in config:
            arm = config['target']['mode']+'/'+arm
        if arm in event_sets:
            raise ValueError("Supply at most one completed run per target/arm")
        events = [json.loads(x) for x in (root/"events.jsonl").read_text(encoding="utf-8").splitlines() if x]
        event_sets[arm] = events
        paths = json.loads((root/"artifact_paths.json").read_text(encoding="utf-8"))
        rq_root = Path(paths["pretrained"])
        binding = json.loads((root/"binding.json").read_text(encoding="utf-8"))
        if 'target' in config:
            teacher = binding.get('teacher_sha256')
            if not teacher or (shared_teacher is not None and teacher != shared_teacher):
                raise ValueError('Target comparison requires the same frozen teacher asset')
            shared_teacher = teacher
        if sha256(rq_root/"catalog.pt") != binding["catalog_sha256"]:
            raise ValueError("Pretraining assets changed since the joint run")
        pretrain_events[arm] = [json.loads(x) for x in (rq_root/"events.jsonl").read_text(encoding="utf-8").splitlines() if x]
        joint_seconds = sum(e.get("phase_wall_seconds", 0) for e in events if e["kind"] in ("R_epoch", "T_block"))
        validations = {}
        for p in sorted((root/"validation").glob("R*/summary.json")):
            value = json.loads(p.read_text(encoding="utf-8"))
            epoch = value["identity"]["R"]
            validations[epoch] = value
            curve_rows.append({"arm": arm, "R": epoch, "T": value["identity"]["T"], **value["metrics"]})
        for name, epoch in (("best", best["R"]), ("final", done["R"])):
            val, test = validations[epoch], tests[name]
            expected = best["sha256"] if name == "best" else done["final_sha256"]
            checkpoint = root/(best["checkpoint"] if name == "best" else done["final_checkpoint"])
            if (val["identity"]["checkpoint_sha256"] != expected or
                    test["identity"]["checkpoint_sha256"] != expected or sha256(checkpoint) != expected):
                raise ValueError("Validation/test checkpoint identity mismatch")
            rows.append({"arm": arm, "point": name, "R": epoch, "T": val["identity"]["T"],
                         "validation_ndcg@5": val["metrics"]["ndcg@5"], "test_ndcg@5": test["metrics"]["ndcg@5"],
                         "validation_recall@5": val["metrics"]["recall@5"], "test_recall@5": test["metrics"]["recall@5"],
                         "test_hr@5": test["metrics"]["hr@5"], "test_lists_per_second": test["count"]/test["seconds"],
                         "joint_phase_wall_seconds": joint_seconds,
                         "validation_seconds": sum(v["seconds"] for v in validations.values()),
                         "peak_gpu_bytes": done["peak_gpu_bytes"], "checkpoint_sha256": expected})
    if not rows:
        raise ValueError("No completed runs")
    for name, values in (("comparison.csv", rows), ("validation_nodes.csv", curve_rows)):
        with (output/name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(output/"comparison.json", rows)
    # Standalone PNGs; each T loss is separate instead of an unreadable
    # four-panel plot. All arms have distinct colors and solid lines.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"B0": "#555555", "B1": "#0072B2", "B2": "#D55E00"}
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    colors.update({arm: palette[i % len(palette)] for i, arm in enumerate(event_sets) if '/' in arm})
    def plot(name, ylabel, series, xlabel):
        fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        for arm, (x, y) in series.items():
            if x:
                ax.plot(x, y, marker="o", markersize=3, linewidth=1.7, color=colors[arm], label=arm)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
        fig.savefig(output/name, dpi=180)
        plt.close(fig)
    for phase in ("R", "T"):
        for loss in ("total", "denoise", "align", "landing", "path", "bucket", "latent", "vq"):
            if phase == "R" and loss in ("latent", "vq"):
                continue
            series = {}
            for arm, events in event_sets.items():
                selected = [e for e in events if e["kind"] == ("R_epoch" if phase == "R" else "T_block")
                            and loss in e['losses']]
                series[arm] = ([e[phase] for e in selected], [e["losses"][loss] for e in selected])
            if any(x for x, _ in series.values()):
                plot(f"{phase}_{loss}.png", loss+" loss", series, phase+" completed")
    plot("validation_ndcg5.png", "Validation NDCG@5",
         {arm: ([r["R"] for r in curve_rows if r["arm"] == arm],
                [r["ndcg@5"] for r in curve_rows if r["arm"] == arm]) for arm in event_sets}, "Recommender epochs")
    for loss in ("loss", "reconstruction", "vq", "cf"):
        series = {}
        for arm, events in pretrain_events.items():
            selected = [e for e in events if e["kind"] == "pretrain_epoch" and loss in e]
            series[arm] = ([e["epoch"] for e in selected], [e[loss] for e in selected])
        if any(x for x, _ in series.values()):
            plot("pretrain_"+loss+".png", loss, series, "RQ item epochs")
    for field in ("collision_redundancy", "collision_items", "sid_change"):
        series = {}
        for arm, events in event_sets.items():
            selected = [e for e in events if e["kind"] == "T_block"]
            series[arm] = ([e["T"] for e in selected], [e["encoding"][field] for e in selected])
        plot("encoding_"+field+".png", field, series, "T optimizer updates")
    lines = ["# RQCD Mixer comparison", "", "Best is selected only by validation NDCG@5.", "",
             "| Arm | Point | R/T | Validation NDCG@5 | Test NDCG@5 |", "|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["arm"]} | {r["point"]} | {r["R"]}/{r["T"]} | {r["validation_ndcg@5"]:.6f} | {r["test_ndcg@5"]:.6f} |')
    lines.extend(["", "A single seed does not establish significance. See comparison.csv for checkpoint identities and performance.",
                  "", "![Validation curve](validation_ndcg5.png)", ""])
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output.resolve()), "rows": len(rows), "model_executed": False}
