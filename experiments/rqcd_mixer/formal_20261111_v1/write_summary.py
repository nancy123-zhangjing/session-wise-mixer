import json
from pathlib import Path

root = Path(__file__).resolve().parent
p = json.loads((root / 'PROGRESS.json').read_text(encoding='utf-8-sig'))
rows = json.loads((root / 'report/comparison.json').read_text())
lines = ['# RQCD Mixer 三组正式实验总结（seed=2024）', '',
         '三组均完成 R35/T300、31 次完整验证及 best/final 完整测试。best 仅按验证 NDCG@5 选择；不能用测试集重新选点。', '',
         '## 设置与对照边界', '',
         'B0 不使用 Mixer；B1 使用层内四头注意力；B2 使用按 RQ 层块的因果注意力。每组独占一张 H100，采用共同初始化与相同 seed。每组各自完成 10000 商品 epoch 的 RQ-VAE 预训练，按最低碰撞冗余率选点，再冻结 RQ Encoder，按 5R 预热＋30 次“10 次 T 更新→刷新目录→1 个 R epoch”完成联合训练。', '',
         'R/T microbatch=256，累积4次，有效 batch=1024；评测 batch=256，bf16，固定20步 DDIM、单候选列表、固定初始噪声。验证节点为 R5 至 R35，共31次。验证/测试样本分别为71657/53194。此次未改变既定模型、损失或学习率，也没有重跑短测。', '',
         '这一对照比较的是三条完整训练路线，包含 Mixer 对预训练及后续联合训练的共同影响。三组预训练选点和产生的潜空间不同，不能把差异全部归因于联合训练某一次交互操作。', '',
         '## 最佳点与最终点', '',
         '|组别|节点|R/T|验证 NDCG@5|测试 NDCG@5|测试 Recall@5|测试 HR@5|',
         '|---|---|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['arm']}|{r['point']}|{r['R']}/{r['T']}|{r['validation_ndcg@5']:.9f}|{r['test_ndcg@5']:.9f}|{r['test_recall@5']:.9f}|{r['test_hr@5']:.9f}|")
lines += ['', '全部逐轮验证结果见 [validation_nodes.csv](validation_nodes.csv)。', '',
          '![验证曲线](validation_ndcg5.png)', '', '## 已观察到什么', '']
for point in ('best', 'final'):
    subset = {r['arm']: r for r in rows if r['point'] == point}
    for arm in ('B1', 'B2'):
        change = (subset[arm]['test_ndcg@5'] / subset['B0']['test_ndcg@5'] - 1) * 100
        lines.append(f'- {point} 对比：{arm} 的测试 NDCG@5 相对 B0 为 {change:+.2f}%。')
lines += ['', 'B1 的验证和测试方向较一致，且 R35 测试略高于其验证所选 best 的测试。B2 验证最佳值低于 B0，测试 best 却略高于 B0；这说明验证和测试排序并不完全一致，不能只看验证集就判定 B2 无效，也不能用测试集挑选有利节点。', '',
          '这些数值仍来自单 seed，尚不能证明统计显著或稳定收益；绝对 NDCG@5 仍约为0.001，不能仅凭组间百分比就称为已经建立强基线。本次不额外运行随机/热门基线或新实验。', '',
          '## 收敛与编码状态', '',
          'R 阶段去噪损失整体下降、后期趋平；T 阶段潜空间重建及逐层 VQ 损失下降，去噪/对齐损失仍有波动。B1 的 R35 去噪损失并非三组最低，却具有最高测试 NDCG，说明连续向量误差与最终推荐命中并不等价。', '',
          '预训练原始特征重建损失下降，但 VQ 损失上升，并带动总损失上升。不能把总损失曲线解释成良好收敛，也不能仅据此证明程序错误。当前按预先约定的碰撞指标选点；其与推荐效果的联系仍是待验证问题。', '',
          '|组别|预训练选点 epoch|选点碰撞冗余率|T300碰撞冗余率|最后10T的SID变化比例|四层利用率|',
          '|---|---:|---:|---:|---:|---|']
for arm, d in p['arms'].items():
    pre = d['pretrain']; enc = d['joint']['latest_by_kind']['T_block']['encoding']
    lines.append(f"|{arm}|{pre['COMPLETE.json']['best_collision_epoch']}|{pre['selection_audit']['selected_encoding']['collision_redundancy']:.4%}|{enc['collision_redundancy']:.4%}|{enc['sid_change']:.4%}|均100%|")
lines += ['', '碰撞冗余率为1−独立SID数/商品数；SID变化比例比较相邻两次目录刷新，不是累计变化率。利用率100%仅表示每个码字均有商品使用，不代表使用均匀或编码一定有效。', '',
          'B2 的后层有效码本变化较大，后期变化幅度回升，同时验证结果波动；“跨层交互造成更大漂移、增加推荐器适配难度”是可检验的猜想，不是本次已证明的因果结论。B1/B2 的碰撞冗余率都略高于 B0，因此本次 Mixer 收益也不能简单解释成碰撞减少。', '',
          '## 分开的损失图与诊断图', '']
for filename in ('R_total','R_denoise','R_align','T_total','T_denoise','T_align','T_latent','T_vq','pretrain_loss','pretrain_reconstruction','pretrain_vq','encoding_sid_change','encoding_collision_redundancy','encoding_collision_items'):
    lines += [f'### {filename}', '', f'![{filename}]({filename}.png)', '']
for filename in ('utilization','perplexity','correction_relative_rms','effective_relative_change'):
    lines += [f'### {filename}', '', f'![{filename}](encoding/{filename}.png)', '']
lines += ['修正/基础 RMS 是有效修正相对基础码本的幅度，不是每次优化器更新的幅度；逐层有效码本变化图才反映相邻 T 块之间的变化。19张图已进行视觉检查，三组采用不同颜色的实线；利用率曲线完全重叠是数据相等。', '',
          '## 执行效率和验收', '',
          '每组预训练约12–19分钟；R/T实际训练合计约79–80分钟，31次验证合计约119–123分钟，best/final测试合计约6分钟。该实验的主要耗时是逐件、多步扩散生成的完整验证，不能把全部用时当成反向传播训练耗时。峰值已分配显存约4.86GiB；显存小本身不等于GPU利用率低。', '',
          '93个不可覆盖验证checkpoint已核对大小、SHA、模型/优化器计数、数据流及随机状态字段；RQ Encoder和保留的RQ Decoder在联合训练中逐节点与预训练选点一致。共同初始化、源码绑定、配置、推理噪声及验证/测试checkpoint身份均通过核验。精确GPU下一步恢复沿用已验收短测证据；本次进行了状态完整性审计，没有重复训练恢复测试。', '',
          '全部大权重、预测文件和原始日志保留远端；本地仅保留源码、小摘要、图表和SHA清单。资源关闭与最终验收状态以同目录上级 FINAL_ACCEPTANCE.json 为准。', '',
          '## 下一步建议（未启动）', '',
          '优先保留B1为有希望的交互方向，但先查清预训练VQ曲线和生成向量落地质量，再决定多seed复现。B2可进一步研究后层变化与适配的关系；不要同时改交互结构、监督信号和量化方式，否则无法识别收益来源。']
(root / 'report/REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Wrote complete Chinese report')
