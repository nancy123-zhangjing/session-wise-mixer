"""Plot completed validation summaries only; no training or inference."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/'five_arm_20261113_v1/CF_INTERIM_COMPARISON.json'


def main():
    rows = json.loads(REFERENCE.read_text())['rows']
    inputs = [REFERENCE]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout='constrained')
    counts = {}
    for ax, arm, title in zip(axes, ('M0', 'M1'), ('CF without Mixer', 'CF with intra-layer Mixer')):
        baseline = [r for r in rows if r['R'] >= 9]
        assert all(r['count'] == 71657 for r in baseline)
        key = f'CF_{arm}_metrics'
        ax.plot([r['R'] for r in baseline], [r[key]['ndcg@5'] for r in baseline],
                color='#0072B2', marker='o', markersize=3, label='Continue joint training')
        points = {9: next(r[key]['ndcg@5'] for r in baseline if r['R'] == 9)}
        for p in sorted((HERE/'nodes').glob(arm+'_R*.json')):
            s = json.loads(p.read_text())
            assert s['count'] == 71657 and s['identity']['T'] == 40
            points[s['identity']['R']] = s['metrics']['ndcg@5']
            inputs.append(p)
        epochs = sorted(points)
        ax.plot(epochs, [points[r] for r in epochs], color='#D55E00', marker='o',
                markersize=4, label='Freeze tokenizer after R9')
        counts[arm] = len(epochs)-1
        ax.set(title=title, xlabel='Recommender epoch', xlim=(8.5,35.5))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Full-validation NDCG@5')
    fig.suptitle('CF R9 freeze comparison — completed nodes only, seed 2024')
    fig.savefig(HERE/'validation_comparison.png', dpi=170)
    plt.close(fig)
    manifest = {'scope': 'interim, not final acceptance', 'new_validation_nodes': counts,
                'sources': {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (HERE/'PLOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
