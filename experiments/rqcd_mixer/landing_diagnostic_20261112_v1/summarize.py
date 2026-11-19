"""Build the human-readable report from completed diagnostic summaries only."""
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

root = Path(__file__).parent
p = json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
lines = ['# RQCD 固定验证子集落地诊断', '',
         '仅推理，未训练；B0/B1/B2各自使用正式验证选出的best。三组共用按固定哈希抽取的8192条验证样本、原始推理噪声及20步DDIM。', '',
         'NN为对原RQ滚动生成的同一批连续向量进行的事后读出，使用全目录潜向量z的平方欧氏距离，各自排除列表内重复商品。NN不改变后续生成前缀，因此不是独立NN滚动推荐的完整评测。', '',
         '## 真实商品向量回查', '',
         '|组别|目录商品数|RQ商品恢复率|NN商品恢复率|标准化再反变换后RQ恢复率|',
         '|---|---:|---:|---:|---:|']
for arm, r in p['results'].items():
    a, b = r['oracle']['original_z'], r['oracle']['normalize_inverse_z']
    lines.append(f"|{arm}|{a['count']}|{a['rq_accuracy']:.4%}|{a['nn_accuracy']:.4%}|{b['rq_accuracy']:.4%}|")
lines += ['', '## 同向量落地对照', '',
          '|组别/最佳节点|RQ NDCG@5|NN NDCG@5|相对变化|RQ/NN命中session数|共同命中session数|',
          '|---|---:|---:|---:|---:|---:|']
for arm, r in p['results'].items():
    a, b = r['rq']['ndcg@5'], r['nn_same_vectors']['ndcg@5']
    lines.append(f"|{arm}/R{r['R']}|{a:.9f}|{b:.9f}|{b/a-1:+.2%}|{r['rq_hit_sessions']}/{r['nn_hit_sessions']}|{r['both_hit_sessions']}|")
lines += ['', '## 其余指标与不确定性', '',
          '|组别|RQ Recall@5|NN Recall@5|RQ HR@5|NN HR@5|NN−RQ NDCG配对用户bootstrap 95%区间|',
          '|---|---:|---:|---:|---:|---|']
for arm, r in p['results'].items():
    a,b = r['rq'],r['nn_same_vectors']; lo,hi=r['paired_user_bootstrap_95pct']
    lines.append(f"|{arm}|{a['recall@5']:.9f}|{b['recall@5']:.9f}|{a['hr@5']:.9f}|{b['hr@5']:.9f}|[{lo:.9f}, {hi:.9f}]|")
lines += ['', 'bootstrap按用户聚类配对重采样3000次；这是固定子集及单seed的描述性诊断，不替代全量验证或多seed结论。', '',
          '## 逐位置检查', '',
          '|组别|RQ/NN第1件命中任一目标的比例|五个位置RQ与NN商品一致率|',
          '|---|---|---|']
for arm,r in p['results'].items():
    f=r['first_position_hit_any_target']; agreement=', '.join(f'{v:.2%}' for v in r['prediction_agreement_by_position'])
    lines.append(f"|{arm}|{f['rq']:.4%} / {f['nn']:.4%}|{agreement}|")
lines += ['', '## 核验', '']
for arm,r in p['results'].items():
    lines.append(f"- {arm}: checkpoint `{r['checkpoint_sha256']}`；与原完整验证逐商品预测一致率 {r['original_validation_prediction_agreement']:.6%}，完整列表一致率 {r['original_validation_exact_list_agreement']:.6%}；模型未变={r['model_unchanged']}；目录未变={r['catalog_unchanged']}；诊断耗时{r['seconds']:.1f}秒。")
lines += ['', f"三组诊断程序总耗时 {p['seconds']:.1f}秒，不含平台排队和启动。", '',
          '详细真实向量错误项、带真实前缀的目标恢复率、噪声与源码哈希见 COMPLETE.json / PROTOCOL.json。生成向量和逐样本预测仅保留远端诊断目录。', '']
audit = json.loads((root/'ORACLE_AUDIT.json').read_text(encoding='utf-8'))
assert audit['distinct_content_vectors'] == 9832
lines += ['## 如何解释这些结果', '',
    '1. 真实向量SID恢复率三组均为100%，标准化反变换前后结果相同。目录中的10728件商品只有9832个不同的输入特征向量，三组也都只有9832个不同的潜向量。B0/B2的896次商品ID未恢复，全部由输入与潜向量完全相同的商品之间平局导致；不能误称为量化把本来可区分的896件商品弄丢。这里896是冗余数量，不是处于重复组内的全部商品数。', '',
    '2. B1另有2例近重复向量未恢复，精确平方距离约3.54e-7和4.01e-7，SID仍相同。与FP32展开平方距离的舍入/抵消误差相符，是应记录的数值边界；本次没有修改距离实现，也没有证明其对生成推荐指标的总体影响。', '',
    '3. 在验证子集的40960个真实目标位置上，带真实前缀回查的商品恢复率三组均为98.2031%。这比按全目录均匀统计更贴近当前验证数据，但不是推荐模型的NDCG上限。', '',
    '4. B1同向量改用全目录NN，NDCG提高49.11%，命中session由41增至60；B0仅提高4.55%，B2下降3.94%。B1用户配对bootstrap区间不含零，B0/B2均包含零；属于单seed固定子集的探索性证据，未作三组多重比较校正。不能推广为NN对所有组都更好。', '',
    '5. RQ与NN的商品选择差异很大，B1各位置只有约21%-25%一致，但绝对指标仍低。结果支持“B1的RQ落地会损失一部分有用信号”，同时不支持“只换最近邻就解释并解决相对H0/H1的全部差距”。这里与H0/H1只是量级参照，没有在本次固定子集重新评测它们。', '',
    '6. 下一步优先检查高噪声生成质量与历史条件是否真正有效（真实历史/打乱历史、按噪声等级拆分误差和目标排名）。如需确认B1落地收益，可在同checkpoint扩大验证范围，或另做NN自主滚动前缀评测；本次均未执行。', '']
(root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
