"""Plot archived full-validation results only; no model execution."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
with (ROOT.parent / 'multiseed_freeze_20261109_v1/comparison_20261110/validation_all_seeds.csv').open(encoding='utf-8-sig') as f:
    reference = {int(r['epoch']): float(r['2024_H1_R20_freeze']) for r in csv.DictReader(f)}
first = read('R08_VERIFIED.json')
rows = [{'R': 8, 'A': reference[8], 'B': first['B']['checkpoint']['ndcg5'],
         'C': first['C']['checkpoint']['ndcg5']}]
for epoch in range(11, 36, 3):
    record = read(f'R{epoch:02d}_COMPARISON.json')
    if 'rows' in record:
        values = {'A': record['A_ndcg5'], **{r['arm']: r['ndcg5'] for r in record['rows']}}
    else:
        values = {arm: record[f'{arm}_ndcg5'] if f'{arm}_ndcg5' in record else record[arm]
                  for arm in 'ABC'}
    assert abs(values['A'] - reference[epoch]) < 1e-10
    rows.append({'R': epoch, **values})
summary = read('COMPARISON.json')
for arm in 'ABC':
    best = max(rows, key=lambda r: r[arm])
    assert best['R'] == summary[arm]['best']['epoch']
    assert abs(best[arm] - summary[arm]['best']['validation_ndcg5']) < 1e-10
    assert abs(rows[-1][arm] - summary[arm]['R35']['validation_ndcg5']) < 1e-10

out = ROOT / 'figures'
out.mkdir(exist_ok=True)
with (out / 'validation_grid_vs_freeze.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['R', 'A', 'B', 'C'])
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                     'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.axvspan(20, 35, color='#F1F4F7', zorder=0)
ax.axvline(20, color='#888888', linewidth=1)
for arm, label, color, marker in [
    ('A', '原 H1 Freeze', '#009E73', 'D'),
    ('B', 'Grid：横向', '#0072B2', 'o'),
    ('C', 'Grid：横向＋纵向', '#D55E00', 's'),
]:
    ax.plot([r['R'] for r in rows], [r[arm] for r in rows],
            label=label, color=color, marker=marker, linewidth=2.5, markersize=5)
    best = max(rows, key=lambda r: r[arm])
    ax.scatter(best['R'], best[arm], marker='*', s=230, color=color,
               edgecolors='white', zorder=5)
ax.set(title='Grid Attention 与原 H1 Freeze 对照（seed 2024）',
       xlabel='推荐器累计训练轮次 R', ylabel='完整验证集 NDCG@5',
       xlim=(7.5, 35.5), ylim=(.0053, .0112))
ax.set_xticks([r['R'] for r in rows])
ax.grid(alpha=.2)
ax.legend(loc='upper left', frameon=False)
fig.text(.5, .035, '星号：各组验证最佳点；灰色区域：R20 后冻结 Tokenizer，推荐器继续训练。\n'
         '各点为已有完整验证结果，连线仅帮助阅读；测试结果见周报表格。',
         ha='center', fontsize=10, color='#555555')
fig.subplots_adjust(left=.12, right=.97, bottom=.20, top=.90)
fig.savefig(out / 'validation_grid_vs_freeze.png', dpi=180)
plt.close(fig)
print('Verified 10 validation nodes and best/final summaries; saved', out)
