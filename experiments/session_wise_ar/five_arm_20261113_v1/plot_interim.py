"""Plot archived full-validation receipts only; never invokes training/inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'interim_curves'
    output.mkdir(exist_ok=True)
    inputs = [ROOT / 'CF_INTERIM_COMPARISON.json']
    rows = json.loads(inputs[0].read_text(encoding='utf-8'))['rows']
    assert rows and all(r['count'] == 71657 for r in rows)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for key, label, color in [('CF_M0_metrics', 'CF without Mixer', '#0072B2'),
                              ('CF_M1_metrics', 'CF with Mixer (reference)', '#D55E00')]:
        ax.plot([r['R'] for r in rows], [r[key]['ndcg@5'] for r in rows],
                marker='o', markersize=4, color=color, label=label)
    ax.set(xlabel='Recommender epochs', ylabel='Full-validation NDCG@5',
           title='CF matched epochs only — interim results')
    ax.grid(alpha=.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.savefig(output / 'cf_validation.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    counts = {}
    for prefix, label, color in [('AR_T0', 'TIGER without Mixer', '#0072B2'),
                                 ('AR_D0', 'DIGER without Mixer', '#009E73'),
                                 ('AR_D1', 'DIGER with Mixer', '#D55E00')]:
        records = {}
        for path in sorted(ROOT.glob(prefix + '_*_VALIDATION.json')):
            obj = json.loads(path.read_text(encoding='utf-8'))
            rec = obj['record']
            assert rec['validation']['count'] == 71657
            assert rec['checkpoint_sha256'] == rec['validation']['identity']['checkpoint_sha256']
            if rec['step'] in records:
                assert records[rec['step']] == rec, 'Conflicting receipts'
            records[rec['step']] = rec
            inputs.append(path)
        ordered = [records[s] for s in sorted(records)]
        counts[prefix] = len(ordered)
        if ordered:
            ax.plot([r['session_epochs'] for r in ordered],
                    [r['validation']['metrics']['ndcg@5'] for r in ordered],
                    marker='o', color=color, label=label)
    ax.set(xlabel='Session-equivalent training epochs (method schedules differ)',
           ylabel='Full-validation NDCG@5', title='AR completed nodes only — interim results')
    ax.grid(alpha=.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.savefig(output / 'ar_validation.png', dpi=170)
    plt.close(fig)
    manifest = {'status': 'interim_not_acceptance', 'cf_last_R': rows[-1]['R'],
                'ar_node_counts': counts,
                'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in inputs},
                'limits': ['No missing-node interpolation or final-test results.',
                           'AR-T1 failed before recommender training; no score is plotted.',
                           'AR and CF schedules differ; epochs are not equal compute budgets.']}
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'cf_last_R': rows[-1]['R'], 'ar_node_counts': counts}))


if __name__ == '__main__':
    main()
