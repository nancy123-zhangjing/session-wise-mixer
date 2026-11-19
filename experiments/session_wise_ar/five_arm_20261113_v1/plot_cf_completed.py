"""Plot existing completed CF logs only, without training or inference."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT/'cf_completed_curves'
    output.mkdir(exist_ok=True)
    files = ['CF_FINAL_CURVE_DATA.json', 'CF_M1_CURVE_DATA.json']
    groups = []
    for file, label, color in zip(files, ['No Mixer', 'Intra-layer Mixer'], ['#0072B2', '#D55E00']):
        events = json.loads((ROOT/file).read_text())['events']
        r = [e for e in events if e['kind'] == 'R_epoch']
        t = [e for e in events if e['kind'] == 'T_block']
        assert [e['R'] for e in r] == list(range(1, 36))
        assert [e['T'] for e in t] == list(range(10, 301, 10))
        groups.append((label, color, r, t))
    outputs = []
    specs = [('recommender_loss', 2, 'R', 'losses', [('total', 'Total loss'), ('denoise', 'Denoising MSE'), ('weighted_landing', 'Weighted RQ landing loss')]),
             ('tokenizer_loss', 3, 'T', 'losses', [('total', 'Total loss'), ('weighted_landing', 'Weighted RQ landing loss'), ('weighted_latent', 'Weighted latent loss'), ('weighted_vq', 'Weighted VQ loss')]),
             ('encoding', 3, 'T', 'encoding', [('sid_change', 'SID changed fraction per 10 T'), ('collision_items', 'Collision item fraction'), ('max_bucket', 'Maximum collision bucket')])]
    for name, index, xkey, section, metrics in specs:
        for key, title in metrics:
            fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
            for group in groups:
                label, color = group[:2]
                events = group[index]
                ax.plot([e[xkey] for e in events], [e[section][key] for e in events], color=color, label=label, linewidth=2)
            ax.set(xlabel='Recommender epochs' if xkey == 'R' else 'Cumulative Tokenizer updates', ylabel=title, title=title)
            ax.grid(alpha=.25)
            ax.legend()
            target = output/f'{name}_{key}.png'
            fig.savefig(target, dpi=170)
            plt.close(fig)
            outputs.append(target)
    manifest = {'scope': 'completed CF logs; no rerun', 'sources': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}, 'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (output/'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'figures': len(outputs), 'R_epochs': 35, 'T_blocks': 30}))


if __name__ == '__main__':
    main()
