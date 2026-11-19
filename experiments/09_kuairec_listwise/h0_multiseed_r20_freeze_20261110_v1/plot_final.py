import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                     'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
with (ROOT.parent / 'multiseed_freeze_20261109_v1/comparison_20261110/validation_all_seeds.csv').open(encoding='utf-8-sig') as f:
    old = list(csv.DictReader(f))
with (ROOT / 'VALIDATION_COMPARISON.csv').open(encoding='utf-8-sig') as f:
    post = list(csv.DictReader(f))
x = np.array([int(r['epoch']) for r in old])
colors = ['#0072B2', '#D55E00', '#009E73']
labels = ['H0：持续交替', 'H0 Freeze：R20 后冻结', 'H1 Freeze：R20 后冻结']
data = {}
for seed in (2024, 2025, 2026):
    lookup = {int(r['epoch']): r for r in post if int(r['seed']) == seed}
    h0 = [float(r[f'{seed}_H0']) for r in old]
    h1 = [float(r[f'{seed}_H1_R20_freeze']) for r in old]
    hf = [h0[i] if e < 20 else float(lookup[e]['H0_freeze']) for i, e in enumerate(x)]
    data[seed] = np.array([h0, hf, h1])
    for i, e in enumerate(x):
        if e >= 20:
            assert abs(h0[i] - float(lookup[e]['H0_continued'])) < 1e-9
            assert abs(h1[i] - float(lookup[e]['H1_freeze'])) < 1e-9

def draw(ax, seed, zoom=False):
    ax.axvspan(20, 35, color='#F1F4F7', zorder=0)
    ax.axvline(20, color='#888888', linewidth=1)
    for j, (y, c, label) in enumerate(zip(data[seed], colors, labels)):
        # Before the fork H0 Freeze shares exactly the H0 trajectory; show it once.
        mask = x >= 20 if j == 1 else np.ones(len(x), dtype=bool)
        ax.plot(x[mask], y[mask], color=c, lw=2.5, marker=['o','s','D'][j], ms=5, label=label)
        k = int(np.argmax(y))
        ax.scatter(x[k], y[k], marker='*', s=210, color=c, edgecolors='white', zorder=6)
    ax.set_title(f'Seed {seed}', fontweight='bold')
    ax.set_xlim(19.5 if zoom else 4.5, 35.7)
    ax.set_xticks(x[x >= 20] if zoom else x)
    ax.set_ylim(.0044 if zoom else .0027, .0112)
    ax.set_xlabel('推荐器累计训练轮次 R')
    ax.grid(alpha=.2)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)

for zoom, name, title in [(False, 'validation_full', '三个 seed 的完整训练轨迹'), (True, 'validation_after_r20', 'R20 之后：同样冻结，H0 与 H1 的表现')]:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for ax, seed in zip(axes, data): draw(ax, seed, zoom)
    axes[0].set_ylabel('验证集 NDCG@5（越高越好）')
    fig.suptitle(title, fontsize=19, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.91), ncol=3, frameon=False)
    fig.text(.5,.025, '星号：各组验证最佳点   |   灰色区域：R20 后   |   H0 Freeze 在 R20 前与 H0 共用轨迹，仅画一次。\n各点为实际验证节点；连线只用于阅读。这里展示验证指标，不是训练 loss，也不是逐轮测试结果。', ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(top=.78,bottom=.22,wspace=.12,left=.065,right=.98)
    fig.savefig(OUT / (name+'.png'), dpi=180)
    fig.savefig(OUT / (name+'.svg'))
    plt.close(fig)
# Separate, readable figures for the weekly report; identical axes across seeds.
for seed in data:
    fig, ax = plt.subplots(figsize=(9.6, 6.1))
    draw(ax, seed)
    ax.set_ylabel('完整验证集 NDCG@5')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .94), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .035,
             '星号：验证最佳点；灰色区域：R20 后。\n'
             'H0 Freeze 在 R20 前与 H0 共用轨迹，仅画一次；折线为验证指标，不是训练 loss。',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(top=.81, bottom=.20, left=.12, right=.97)
    fig.savefig(OUT / f'validation_seed{seed}.png', dpi=180)
    plt.close(fig)
print(OUT)

