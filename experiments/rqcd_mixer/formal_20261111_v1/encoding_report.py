"""Supplement existing report with per-layer codebook diagnostics from logs only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    colors = {'B0': '#555555', 'B1': '#0072B2', 'B2': '#D55E00'}
    metrics = ('utilization', 'perplexity', 'correction_relative_rms', 'effective_relative_change')
    titles = ('Codebook utilization', 'Code usage perplexity', 'Mixer correction / base RMS', 'Effective codebook relative change per T block')
    rows, series = [], {}
    for arm in colors:
        # Resume may repeat an event; keep the last completed record for each clock.
        events = {}
        for line in (root / arm / 'events.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('kind') == 'T_block':
                events[e['T']] = e
        assert sorted(events) == list(range(10, 301, 10)), (arm, 'Missing T diagnostic')
        series[arm] = [events[t] for t in sorted(events)]
        for e in series[arm]:
            enc = e['encoding']
            for layer in range(4):
                rows.append({'arm': arm, 'R_before_next_epoch': e['R'], 'T': e['T'], 'layer': layer + 1,
                             **{key: enc[key][layer] for key in metrics},
                             'sid_change': enc['sid_change'], 'collision_redundancy': enc['collision_redundancy'],
                             'collision_items': enc['collision_items'], 'unique_sids': enc['unique_sids']})
    with (output / 'encoding_by_layer.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric, title in zip(metrics, titles):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for layer, ax in enumerate(axes.flat):
            for arm, events in series.items():
                ax.plot([e['T'] for e in events], [e['encoding'][metric][layer] for e in events],
                        color=colors[arm], label=arm, linewidth=1.8, linestyle='-')
            ax.set_title(f'RQ layer {layer + 1}')
            ax.set_xlabel('Cumulative Tokenizer updates')
            ax.grid(alpha=.25)
            if metric == 'utilization':
                ax.set_ylim(0, 1.05)
            ax.legend()
        fig.suptitle(title)
        fig.savefig(output / f'{metric}.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.output)
