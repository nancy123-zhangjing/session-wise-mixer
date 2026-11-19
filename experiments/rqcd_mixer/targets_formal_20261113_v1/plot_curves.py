"""Plot the two accepted runs from saved logs; never executes training."""
import argparse
import json
from pathlib import Path
import shlex

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"cf": "#1765b4", "hybrid": "#df6c20"}
LABELS = {"cf": "CF target", "hybrid": "Content + CF target"}


def fetch():
    import control
    script = r'''
import json
from pathlib import Path
root = Path.cwd()
out = {}
for mode in ("cf", "hybrid"):
    one = {}
    for name, folder in (("joint", root / mode), ("pretrain", root / (mode + "_rq"))):
        events = [json.loads(s) for s in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = {"validation", "R_epoch", "T_block"} if name == "joint" else {"pretrain_validation"}
        keep = []
        for e in events:
            if e["kind"] not in kinds:
                continue
            row = {k: e[k] for k in ("kind", "R", "T", "epoch") if k in e}
            if name == "joint":
                source = e["metrics"] if e["kind"] == "validation" else e["losses"]
                key = "metrics" if e["kind"] == "validation" else "losses"
                wanted = ("ndcg@5",) if key == "metrics" else (
                    "total", "denoise", "weighted_landing", "latent", "vq")
                row[key] = {k: source[k] for k in wanted if k in source}
            else:
                row["fixed_objective"] = {"loss": e["fixed_objective"]["loss"]}
            keep.append(row)
        one[name] = keep
    out[mode] = one
print(json.dumps(out, ensure_ascii=False))
'''
    command = f"cd {control.REMOTE} && {control.PYTHON} -B -c {shlex.quote(script)}"
    data = json.loads(control.remote(command, "CURVE_FETCH_REMOTE.json", timeout=90))
    control.write(HERE / "CURVE_DATA.json", data)
    return data


def series(data, mode, kind, metric, phase="joint"):
    rows = [e for e in data[mode][phase] if e["kind"] == kind]
    xkey = "epoch" if phase == "pretrain" else ("T" if kind == "T_block" else "R")
    def value(row):
        item = row
        for key in metric.split("."):
            item = item[key]
        return item
    return [e[xkey] for e in rows], [value(e) for e in rows]


def draw(ax, data, kind, metric, title, ylabel, phase="joint", selected=None):
    for mode in ("cf", "hybrid"):
        x, y = series(data, mode, kind, metric, phase)
        ax.plot(x, y, color=COLORS[mode], linewidth=2, marker="o" if phase == "joint" else None,
                markersize=3, label=LABELS[mode])
        if selected is not None:
            idx = x.index(selected[mode])
            ax.scatter([x[idx]], [y[idx]], s=75, facecolor="white", edgecolor=COLORS[mode],
                       linewidth=2, zorder=5)
    ax.set_title(title)
    xlabel = "RQ pretrain item epochs" if phase == "pretrain" else (
        "Tokenizer updates (T)" if kind == "T_block" else "Recommender epoch (R)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="read existing remote logs")
    args = p.parse_args()
    data = fetch() if args.fetch else json.loads((HERE / "CURVE_DATA.json").read_text(encoding="utf-8"))
    out = HERE / "curves"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})
    figures = [
        ("validation_ndcg5", "validation", "metrics.ndcg@5", "Full validation NDCG@5", "NDCG@5", "joint", {"cf": 9, "hybrid": 31}),
        ("R_total_loss", "R_epoch", "losses.total", "Recommender total training loss", "Mean loss", "joint", None),
        ("R_denoise_loss", "R_epoch", "losses.denoise", "Recommender denoising MSE", "Mean MSE", "joint", None),
        ("R_landing_loss", "R_epoch", "losses.weighted_landing", "Recommender weighted RQ landing loss", "Mean weighted loss", "joint", None),
        ("T_total_loss", "T_block", "losses.total", "Tokenizer total training loss", "Mean loss", "joint", None),
        ("T_denoise_loss", "T_block", "losses.denoise", "Tokenizer denoising MSE", "Mean MSE", "joint", None),
        ("T_landing_loss", "T_block", "losses.weighted_landing", "Tokenizer weighted RQ landing loss", "Mean weighted loss", "joint", None),
        ("T_latent_loss", "T_block", "losses.latent", "Tokenizer latent reconstruction loss", "Mean loss", "joint", None),
        ("T_vq_loss", "T_block", "losses.vq", "Tokenizer VQ loss", "Mean loss", "joint", None),
        ("rq_pretrain_objective", "pretrain_validation", "fixed_objective.loss", "RQ pretraining fixed objective", "Fixed objective", "pretrain", {"cf": 50, "hybrid": 50}),
    ]
    for filename, kind, metric, title, ylabel, phase, selected in figures:
        fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
        draw(ax, data, kind, metric, title, ylabel, phase, selected)
        fig.savefig(out / (filename + ".png"))
        plt.close(fig)
    print(json.dumps({"figures": len(figures), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
