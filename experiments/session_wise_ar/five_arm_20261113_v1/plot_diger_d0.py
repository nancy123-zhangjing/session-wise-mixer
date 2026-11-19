"""Standalone D0 full-validation curve from immutable node receipts."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    records = {}
    sources = {}
    for path in ROOT.glob('AR_D0_*_VALIDATION.json'):
        r = json.loads(path.read_text())['record']
        assert r['validation']['count'] == 71657
        assert r['checkpoint_sha256'] == r['validation']['identity']['checkpoint_sha256']
        epoch = int(r['session_epochs'])
        if epoch in records:
            assert records[epoch] == r
        records[epoch] = r
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    epochs = sorted(records)
    assert epochs == list(range(1, max(epochs)+1))
    values = [records[e]['validation']['metrics']['ndcg@5'] for e in epochs]
    best = max(range(len(values)), key=values.__getitem__)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    ax.plot(epochs, values, color='#0072B2', marker='o', linewidth=2, markersize=5)
    ax.scatter([epochs[best]], [values[best]], color='#D55E00', s=100, zorder=5)
    for e, v in zip(epochs, values):
        offset = -19 if e == 17 else 9
        ax.annotate(f'{v:.6f}', (e,v), xytext=(0,offset), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.annotate(f'最佳：第{epochs[best]}轮，{values[best]:.6f}',
                (epochs[best],values[best]), xytext=(12,values[best]+.001),
                arrowprops={'arrowstyle':'->','color':'#D55E00'}, color='#B34D00', fontsize=11)
    ax.set(title=f'DIGER 无 Mixer：第1—{epochs[-1]}轮完整验证曲线（seed 2024）',
           xlabel='训练轮次（session epoch）', ylabel='验证 NDCG@5',
           xlim=(.4,epochs[-1]+.6), ylim=(0,max(values)+.0015))
    ax.set_xticks(epochs)
    ax.grid(alpha=.25)
    out = ROOT/'diger_d0_curve'
    out.mkdir(exist_ok=True)
    fig.savefig(out/'validation.png',dpi=180)
    plt.close(fig)
    (out/'DATA.json').write_text(json.dumps({'count_per_node':71657,'seed':2024,
        'scope':'completed validation nodes only; training ongoing',
        'points':[{'epoch':e,'ndcg@5':v} for e,v in zip(epochs,values)],
        'sources':sources},indent=2),encoding='utf-8')
    print(json.dumps({'nodes':len(epochs),'best_epoch':epochs[best],'best':values[best]}))


if __name__ == '__main__':
    main()
