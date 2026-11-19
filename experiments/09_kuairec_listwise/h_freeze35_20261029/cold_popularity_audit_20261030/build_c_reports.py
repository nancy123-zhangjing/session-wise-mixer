"""Render Chinese deliverables from the verified, immutable CPU statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE / 'c_supplement_r1'
NAMES = {'H0_best_R32': '旧 H0 best R32', 'H1_R20': '旧 H1 R20',
         'H1_freeze_R32': '冻结 H1 R32', 'C_freeze_R32': '冻结 C R32',
         'C_best_R23': 'C best R23'}
SEG = {'early': '前段', 'late': '后段', 'full': '全量'}
CLASSES = {'never_seen': '训练完全未见', 'history_only': '仅训练历史出现',
           'target_seen': '曾作为训练目标', 'target_head': 'target head',
           'target_mid': 'target mid', 'target_tail': 'target tail'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(value)


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers) + '|\n'
            + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows))


def pct(v, digits=2):
    return f'{100*v:.{digits}f}%'


def interval(value):
    return f'[{value["ci_low"]:+.6f}, {value["ci_high"]:+.6f}]'


def main():
    verify = read(HERE / 'VERIFY_C_SUPPLEMENT.json')
    complete = read(RESULT / 'COMPLETE.json')
    assert verify['status'] == complete['status'] == 'passed'
    assert verify['analysis_complete_sha256'] == sha(RESULT / 'COMPLETE.json')
    for name, digest in complete['files'].items():
        assert sha(RESULT / name) == digest
    originals = read(HERE / 'initial_r1/COMPLETE.json')
    for name, digest in originals['files'].items():
        assert sha(HERE / 'initial_r1' / name) == digest
    pinned = read(HERE / 'C_EXTRA_MODELS.json')['initial_artifacts_unchanged']
    for name, digest in pinned.items():
        assert sha(HERE / name) == digest
    models = {name: read(RESULT / ('MODEL_' + name + '.json'))['segments'] for name in NAMES}
    paired = read(RESULT / 'PAIRED_BOOTSTRAP.json')
    decomp = read(RESULT / 'DECOMPOSITION.json')
    composition = read(RESULT / 'DATA_COMPOSITION.json')
    fair = 'H1_freeze_R32_minus_C_freeze_R32'
    best = 'H1_freeze_R32_minus_C_best_R23'
    frozen = 'H1_freeze_R32_minus_H1_R20'
    old = 'H1_freeze_R32_minus_H0_best_R32'
    feature = 'list/ndcg@5/overall'
    fair_comp = decomp[fair]['5']['composition_change'] / decomp[fair]['5']['observed_advantage_change']
    best_comp = decomp[best]['5']['composition_change'] / decomp[best]['5']['observed_advantage_change']
    for name in NAMES:
        for s in ('early', 'late'):
            for group in ('never_seen', 'history_only', 'target_tail'):
                assert models[name][s]['classes'][group]['metrics']['5']['target_hits'] == 0
    h1_top_classes = [[v['fine_class'] for v in models[n][s]['predictions']['list5']['top10_items']]
                      for n in ('H1_R20', 'H1_freeze_R32') for s in ('early', 'late')]
    assert all(set(row) == {'target_mid'} for row in h1_top_classes)

    def metrics_table(names, segments=('early', 'late', 'full')):
        return table(['模型', '时段', 'NDCG@5', 'HR@5', '聚合 NDCG@50'], [
            [NAMES[n], SEG[s], *[f'{models[n][s]["metrics"][k]:.6f}'
                              for k in ('list/ndcg@5', 'list/hr@5', 'aggregate/ndcg@50')]]
            for n in names for s in segments])

    def class_table(names):
        return table(['模型', '时段', '暖目标命中率@5', 'head 命中率@5', 'mid 命中率@5',
                      'head 的 NDCG@5 贡献', 'mid 的 NDCG@5 贡献'], [
            [NAMES[n], SEG[s], *[pct(models[n][s]['classes'][g]['metrics']['5']['target_hit_rate'], 3)
                                for g in ('target_seen', 'target_head', 'target_mid')],
             *[f'{models[n][s]["classes"][g]["metrics"]["5"]["ndcg_contribution"]:.6f}'
               for g in ('target_head', 'target_mid')]]
            for n in names for s in ('early', 'late')])

    def cold_table(names):
        return table(['模型', '前段：未见 / 仅历史 / tail', '后段：未见 / 仅历史 / tail'], [
            [NAMES[n], *[' / '.join(str(models[n][s]['classes'][g]['metrics']['50']['target_hits'])
                                   for g in ('never_seen', 'history_only', 'target_tail'))
                        for s in ('early', 'late')]] for n in names])

    def bias_table(names, protocol):
        return table(['模型', '时段', '训练 head 推荐份额', '前十件集中度', '不同推荐商品数', 'HHI', '平均有效商品数'], [
            [NAMES[n], SEG[s], pct(models[n][s]['predictions'][protocol]['class_shares']['target_head']),
             pct(models[n][s]['predictions'][protocol]['top10_concentration']),
             models[n][s]['predictions'][protocol]['unique_recommended_items'],
             f'{models[n][s]["predictions"][protocol]["hhi"]:.5f}',
             f'{models[n][s]["predictions"][protocol]["coverage"] * (5 if protocol == "list5" else 50):.2f}']
            for n in names for s in ('early', 'late')])

    def comparisons_table(pairs, segments=('early', 'late', 'full')):
        return table(['比较（H1 − 对照）', '时段', 'NDCG@5 差值', '相对对照', '95% 用户区间',
                      '聚合 NDCG@50 差值', '95% 用户区间'], [
            [label, SEG[s], f'{paired[s][p][feature]["delta"]:+.6f}',
             f'{paired[s][p][feature]["delta"] / paired[s][p][feature]["baseline"]:+.2%}',
             interval(paired[s][p][feature]),
             f'{paired[s][p]["aggregate/ndcg@50/overall"]["delta"]:+.6f}',
             interval(paired[s][p]['aggregate/ndcg@50/overall'])]
            for p, label in pairs for s in segments])

    def decomposition_table(pairs):
        return table(['比较', '后段优势 − 前段优势', '95% 用户区间', '组成变化项', '组内变化项'], [
            [label, f'{decomp[p]["5"]["observed_advantage_change"]:+.8f}',
             interval(paired['late_minus_early'][p][feature]),
             f'{decomp[p]["5"]["composition_change"]:+.8f}',
             f'{decomp[p]["5"]["within_class_change"]:+.8f}'] for p, label in pairs])

    audit_text = f'''五个模型各 71,657 条保存验证预测全量解析，合计 358,285 条；训练目标/历史频次重新统计 709,443 条训练样本。12 项整体指标均与主逐样本审计复算一致，最大差为 0。原 `verify.py` 独立复算全部数组、分组贡献、推荐份额/集中度、2,000 次 bootstrap 抽样值与对称分解；每模型另从原始预测固定抽查 1,027 条候选聚合并全量计数。bootstrap 最大复算误差为 {verify['max_bootstrap_draw_abs_error']:.3g}。

统计、验证、表格和报告均只读取现有预测；未加载模型、未做前向推理、未训练、未申请 GPU。原初版全部输出 SHA、原 `REPORT.md`、`analyze.py`、`verify.py` 在本次报告生成前再次核验不变。C best R23 来自原共同节点规则，未按本诊断、时间分段或测试结果重选。

- [固定方案](PROTOCOL.md)、[新增 C 输入及审计绑定](C_EXTRA_MODELS.json)、[五模型统计输入](c_supplement_r1/REQUEST.json)
- [完整整体指标](c_supplement_r1/METRICS.csv)、[目标组成](c_supplement_r1/DATA_COMPOSITION.json)、[原三模型报告](REPORT.md)
- [C 补充目标命中/DCG/贡献表](c_supplement_tables_r1/TARGET_CLASSES.csv)、[推荐份额与集中度](c_supplement_tables_r1/RECOMMENDATIONS.csv)、[推荐 top10 明细](c_supplement_tables_r1/TOP10_ITEMS.csv)
- [全部用户配对区间](c_supplement_r1/PAIRED_BOOTSTRAP.json)、[五类分解](c_supplement_r1/DECOMPOSITION.json)、[独立复核](VERIFY_C_SUPPLEMENT.json)
- [统计完成清单](c_supplement_r1/COMPLETE.json)、[表格来源及 SHA 清单](c_supplement_tables_r1/EXPORT_RECEIPT.json)

统计 COMPLETE SHA256：`{sha(RESULT / 'COMPLETE.json')}`；独立复核 SHA256：`{sha(HERE / 'VERIFY_C_SUPPLEMENT.json')}`。

本报告完成的是这项授权离线诊断，不单独宣告原训练 Goal 的完整备份、实际恢复、Git 或释放验收完成；这些由主阶段验收记录统一确认。
'''

    limitations = '''- 本次为已经参与选模的完整验证集上的事后诊断；用户聚类区间不覆盖训练 seed、checkpoint 选择、冻结时点选择的不确定性，辅助比较未做多重检验校正。
- A/C 固定 R32 都是 R32/T200、R20 冻结，消除了本轮比较中冻结时点与更新量不同的问题；但仍只有一个 seed，且 R20 冻结时点源于先前 H1 观察，不能据此宣称跨 seed 稳定的 Mixer 因果收益。
- 各自 best 比较为 H1 R32 对 C R23；R 更新量分别 22,176 和 15,939，不能称为相同训练终点。两者 T 均为 200。它用于检查结果是否仅由选了较弱的 C R32 引起。
- “组内变化”仍混合了不同用户、不同商品和时间分布；粗分类分解是描述性恒等式，不是因果中介分析，不能排除更细的冷启动、曝光或偏好变化。
- H1 R20 到冻结 H1 R32 同时包含推荐器继续训练；本轮没有同 R32、继续更新编码的反事实，不能把所有改善单独归因于冻结。
- 所有值都来自 validation，不能与 test 的绝对值混算。原固定 R35 测试主比较由主实验报告处理，本诊断没有改变训练、选模或测试协议。
'''

    supplement = f'''# C 公平对照补充：冷目标、暖目标与推荐集中度

2026-10-30。独立复核已通过；保留原三模型报告与结果不变。本补充复用 C 的完整验证预测，加入相同 R32/T200 的 C，以及原规则选出的 C best R23。

## 新证据带来的判断

**原判断经 C 公平对照仍成立：后段冷目标增加，但它不是 H1 相对优势缩小的主要组成解释；变化主要位于训练中频目标内部。H1 更集中于少数商品，不能改写成“比 H0 更多推荐训练最热门商品”。**

相同 R32/T200、相同 R20 冻结的比较中，H1 后段 NDCG@5 高于 C {paired['late'][fair][feature]['delta']/paired['late'][fair][feature]['baseline']:.2%}，用户区间在零以上；其优势仍从前段 {paired['early'][fair][feature]['delta']/paired['early'][fair][feature]['baseline']:.2%} 缩小。五类目标组成项仅占绝对优势缩小量约 {pct(fair_comp)}。换成原 C best R23 后结论方向不变，组成项约 {pct(best_comp)}，但该比较的 R 更新量不等。

## 1. 相同终点与各自 best 分开看

{metrics_table(('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23'))}

{comparisons_table(((fair, 'H1 R32 − C R32'), (best, 'H1 R32 − C best R23')))}

相同终点的前、后段 NDCG@5 与聚合 NDCG@50 区间均在零以上；各自 best 的对应区间也在零以上。不能说 H1 后段仍输给 C；准确表述是后段仍领先，但领先幅度缩小。

## 2. 暖目标内部也在变化

三覆盖类定义、head/mid/tail 定义与原报告一致。下面“暖目标”专指曾作为训练目标的 head+mid+tail；目标命中率以该类目标出现次数为分母，不是列表 HR@5。贡献以原五目标 IDCG 和全段样本数为分母。

{class_table(('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23'))}

三个模型在两段的未见、仅历史、tail 的 @5 命中数及 NDCG 贡献均为 0；因此上表 head+mid 贡献恰好构成总分。H1 暖目标命中率从 1.294% 降到 0.905%；主要是 mid 从 1.537% 降到 0.955%，head 反而从 1.259% 升到 1.314%。不能笼统说每一种暖目标都变差。

聚合 @50 对未见 / 仅历史 / tail 的目标命中数如下；少量命中不能证明已解决冷启动：

{cold_table(('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23'))}

{decomposition_table(((fair, 'H1 R32 − C R32'), (best, 'H1 R32 − C best R23')))}

公平 R32 的组内项 −0.00186733 中，mid 为 −0.00199411，head 为 +0.00012678，其他三类为 0。mid 相对效果的变化主导了优势减小；这不等于已识别出唯一训练机制。

## 3. 训练热门与推荐集中度分开看

真实目标的训练 head 占比：前段 15.01%，后段 12.49%。Top5 统计：

{bias_table(('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23'), 'list5')}

H1 的 head 份额低于两份 C；相对真实目标组成，H1 自身仍有 head 过代表。后段 H1 的 top10 集中度、HHI 都高于两份 C，但不能因此称为更强的训练热门偏置。H1 早晚两段最常推荐的十件商品全部是训练 mid；相对 C best 的前段 top10 集中度则更低，不能写成 H1 在所有模型/时段都更集中。

聚合 @50：

{bias_table(('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23'), 'aggregate50')}

@50 是原 beam32 五件候选列表聚合，平均只有约 19–20 件有效商品，余下位置为 PAD；份额分母排除 PAD，计分仍沿用原协议。

## 4. 对冻结和 session 留出的含义

旧 H1 R20 到冻结 H1 R32，后段 top10 集中度 36.88%→27.74%，不同商品数 582→648，NDCG@5 0.007316→0.008374（+14.46%）；原用户配对区间支持这段改善。公平 C 加入后，H1 后段的性能优势仍在，但集中度差距与前后收益差尚未消失。

**现有证据不支持把“冷目标变多”当成 H1 相对衰减的主要原因，也不足以优先改成每用户最后两个 session 留出。**以后可以把 session 留出作为另一种评估问题的稳健性检查，保留原时间留出并让 H0/H1 使用同一新协议；它不是本轮已实施的修复方案。本轮没有新增切分、训练、seed 或推理。

## 公平性与因果限制

{limitations}
## 验收证据

{audit_text}
'''

    composition_rows = [[CLASSES[g], composition['early']['classes'][g]['target_occurrences'],
                         pct(composition['early']['classes'][g]['target_share']),
                         composition['late']['classes'][g]['target_occurrences'],
                         pct(composition['late']['classes'][g]['target_share'])]
                        for g in ('never_seen', 'history_only', 'target_seen', 'target_head', 'target_mid', 'target_tail')]
    final = f'''# H1 后段收益减小：冷启动与热门偏置完整分析

2026-10-30，五模型 CPU 离线诊断完成并通过独立复核。完整保留旧 H1 R20、冻结 H1 R32、旧 H0 best 的初版分析，现补齐相同 R/T 的 C R32 和按原规则选出的 C best R23。

## 给学长的直接回答

1. **后段确实有更多冷目标，但目前不支持把它当作 H1 相对收益减小的主要解释。**未作为训练目标的商品占比由 7.71% 增至 12.41%；然而在同 R32/T200 的 H1−C 对照中，五类目标组成变化仅解释优势缩小量约 **{pct(fair_comp)}**，其余主要来自 mid 类内部相对效果减弱。这是固定粗分层下的描述性分解，不是说冷启动没有影响。
2. **暖目标内部也有下降；“更多推训练热门”与“集中推少数商品”必须分开。**冻结 H1 的暖目标命中率@5 从 1.294% 降到 0.905%，mid 从 1.537% 降到 0.955%，head 则小幅上升。后段 H1 训练 head 推荐份额 22.42%，低于公平 C 的 27.05%；但 top10 推荐集中度 27.74%，高于 C 的 21.96%，其 top10 都是训练 mid。因此支持“少数商品集中”，不支持“H1 比 H0 更多推训练最热门商品”。
3. **冻结后续训伴随集中程度缓解和后段性能恢复，但没有消除时间差。**旧 H1 到冻结 H1，后段 NDCG@5 提升 14.46%，top10 集中度 36.88%→27.74%；公平 C 对照下后段仍领先 14.95%，同时前段领先 30.52%，优势差依然存在。推荐器继续训练与冻结同时发生，不能把改善全部单独归因于冻结。

据此，**暂不把“每用户最后两个 session 留出”作为优先修复方向或替代当前时间划分**。它以后可用于检验另一种评估问题下的稳健性，但这批结果更指向暖目标、特别是 mid 内部的推荐与真实目标匹配变化；尚不足以认定简单换切分能解决问题。本轮不新增实验。

## 1. 问题、模型与比较边界

| 模型 | Mixer | R / T 更新量 | 用途 |
|---|---|---|---|
| 旧 H0 best R32 | 无 | R32=22,176 / T360 | 旧结果参照，T 与 H1 不同 |
| 旧 H1 R20 | 有 | R20=13,860 / T200 | 冻结前状态 |
| 冻结 H1 R32 | 有 | R32=22,176 / T200，R20 冻结 | 本次 H1 主诊断点，也是原规则选出的 best |
| 冻结 C R32 | 无 | R32=22,176 / T200，R20 冻结 | 与 H1 相同 R/T 和冻结时点的主对照 |
| C best R23 | 无 | R23=15,939 / T200，R20 冻结 | 原共同节点规则 best，R 更新量不等的补充比较 |

C 两个 checkpoint 不同，均纳入而非任选有利者；best 仍由原 R20/R23/R26/R29/R32/R35 共同节点完整验证 NDCG@5 决定。本次既不按冷暖分组重选，也不按 test 重选。

所有模型使用同一 71,657 条完整验证样本、4,733 名用户、beam32、FP32、五件目标和原候选聚合协议。沿用第一件目标时间切点 **2020-08-30 07:57:21.188500（北京时间）**：前段 38,644 条 / 4,011 名用户，后段 33,013 条 / 3,827 名用户；95 个跨切点目标块仍按第一件目标归类。

全篇的 @5 为一份五件列表；聚合 @50 是多份候选五件列表按分数顺序、商品首次出现去重后的结果，含 PAD，不是生成一份完整五十件列表。逐目标命中率与列表 HR@5 使用不同分母；贡献统一使用原五目标 IDCG。

## 2. 冷目标更多，但所有模型都有共同的冷尾短板

“训练完全未见”是保留训练样本的历史和目标均未出现；商品文本和静态目录可能已知，不能称为完全未知商品。“仅训练历史出现”没有直接目标监督，不代表没有梯度。“暖目标”在本报告特指曾作为训练目标的商品。

训练目标共覆盖 8,015 件商品。只按训练目标出现次数排序，前 10% 为 head（802 件），之后至前 50% 为 mid（3,206 件），其余 tail（4,007 件）；并列按内部 ID。未见的 1,914 件和仅历史的 799 件不混入 tail，也不拿重复历史窗口出现次数定义热门。

{table(['类别', '前段目标次数', '前段占比', '后段目标次数', '后段占比'], composition_rows)}

前三行是互斥三覆盖类；后三行是“曾作为训练目标”的细分，不能把六行再相加。未见+仅历史占比上升 4.70 个百分点，曾作为训练目标占比相应下降。

**五模型在前、后段对未见、仅历史、tail 的单列表 @5 均零命中、零 DCG、零 NDCG 贡献。**这指出共同能力短板；因为差异项也为零，不能仅凭这些目标更多就推出 H1 的相对收益下降是由它们造成。

扩大到原聚合 @50 后，三类实际命中次数如下：

{cold_table(tuple(NAMES))}

前段三类目标出现次数为 4,255 / 10,639 / 22,957，后段为 5,022 / 15,460 / 15,316。零或几次命中应作为覆盖短板报告，不宜据此宣称某模型已改善冷启动。推荐中出现冷商品也不等于命中冷目标。

## 3. 暖目标内部也下降，主要变化来自 mid

{class_table(tuple(NAMES))}

表中的暖目标命中率以 head+mid+tail 的目标次数为分母，已排除未见与仅历史目标；它仍从前段到后段下降，所以不能把整体下降全部归结为冷目标占比变化。head/mid 的贡献按全段样本数归一化，两项相加等于本次 @5 总分。完整各类命中数、原 DCG、@20/@50 贡献在机器可读结果中保留。

H1 的 mid 命中率下降尤为明显。冻结 H1 的 head 条件命中率反而略升，head 的整体贡献仍会因目标 head 占比下降而减少；条件效果与整体贡献不能混为一谈。

对 H1−对照的“后段优势 − 前段优势”作固定五类对称分解：组成项表示类别占比变化，组内项表示类别内部的逐目标相对 DCG 变化；两项精确闭合到总体差值。

{decomposition_table((('H1_R20_minus_H0_best_R32', '旧 H1 − 旧 H0'), (old, '冻结 H1 − 旧 H0'), (fair, '冻结 H1 R32 − C R32'), (best, '冻结 H1 R32 − C best R23')))}

相同 R32/T200 对照的优势缩小为 −0.00187318，组成项 −0.00000585（约 **{pct(fair_comp)}**），组内项 −0.00186733。具体为 mid 组内项 −0.00199411，head 组内项 +0.00012678，其他三类为 0。换 C best 后组成项约 {pct(best_comp)}；旧 H1 对旧 H0 的组成项甚至抵消了一部分下降。

因此，**在这套固定分层下，冷暖与频次层的目标占比改变不是相对收益衰减的主要组成解释**。但“mid 组内”还包含不同商品、用户和时间的组成变化；不能进一步断言已经证实偏好漂移、曝光变化或某个训练机制，也不能排除更细粒度的冷启动影响。

## 4. H1 的少数商品集中更明显，但训练 head 份额更低

训练 head 推荐份额统计固定 802 件高频训练目标商品占有效推荐位置的比例；top10 集中度统计该模型在该时段推荐最多的十件商品占比。两者定义不同。HHI 为有效推荐物品份额平方和，越高表示更集中；PAD 从这些分母中排除。

单列表 @5：

{bias_table(tuple(NAMES), 'list5')}

训练 head 的真实目标占比为前段 15.01%、后段 12.49%。冻结 H1 的 head 推荐份额高于真实目标构成，存在总体过代表；但**它低于旧 H0 和两份 C，不能支持“H1 比 H0 更偏训练热门”这个相对解释**。总体份额也没有调整个性化可推荐性，不能直接当作公平性或因果偏置度量。

旧 H1 R20 与冻结 H1 R32 在早晚两段各自推荐最多的十件商品全部属于训练 mid。冻结 H1 与公平 C R32 比较，在前后两段的 top10 集中度、HHI 都更高，推荐商品数更少；这个“少数商品集中”判断得到两个集中度指标支持。与 C best R23 相比，H1 前段 top10 集中度较低、后段较高，因此不能扩展为任何模型/时段都更集中。

另外，H1 自身后段集中度低于前段，不能说“越到后段越集中”。前后样本数也不同，商品覆盖数主要应在同一时段的模型间比较。

聚合 @50 同样分别查看训练热门与集中度：

{bias_table(tuple(NAMES), 'aggregate50')}

每条样本平均约 19–21 件有效聚合商品，其余为 PAD。原 @50 指标不改变，解释收益时同时保留候选覆盖。这些现象说明集中和准确率可以同时变化，但不能单凭相关性把集中当成唯一性能原因，更不能直接推出加热门惩罚会改善。

## 5. 冻结后续训改善了表现，但后段相对收益仍更小

完整指标如下；仅在 validation 内比较：

{metrics_table(tuple(NAMES))}

冻结 H1 相对自身 R20、旧 H0 的原比较：

{comparisons_table(((frozen, '冻结 H1 − 旧 H1 R20'), (old, '冻结 H1 − 旧 H0')), ('early', 'late'))}

H1 后段从 0.007316 到 0.008374（+14.46%），top10 集中度从 36.88% 降到 27.74%，覆盖商品数从 582 增至 648，显示冻结后的推荐器续训轨迹伴随性能和集中程度改善。相对旧 H0，旧 H1 后段略低而冻结 H1 后段更高；旧 H0 的 T360 使该比较不能单独归因于 Mixer。

补齐 C 后，同 R32/T200 的比较和各自 best 补充如下：

{comparisons_table(((fair, 'H1 R32 − C R32'), (best, 'H1 R32 − C best R23')))}

公平 C 对照下，后段 H1 的 NDCG@5 和聚合 NDCG@50 区间均在零以上；原 C best 对照也如此，不能把结果归结为只选择了较弱的 C R32。**后段仍领先与前后差距仍存在同时成立**：公平比较的后段减前段优势区间为 {interval(paired['late_minus_early'][fair][feature])}，不跨零。冻结相对自身 R20 的提升在前后段都存在，其提升的前后差区间为 {interval(paired['late_minus_early'][frozen][feature])}，跨零，不能说它专门修复了后段。

## 6. 哪些说法被支持，哪些还不能成立

| 问题 | 现有证据支持 | 不支持或不能推出 |
|---|---|---|
| 冷启动 | 后段未作为训练目标的占比增加，五模型都有严重冷尾覆盖短板 | 不能据此把 H1 相对优势衰减主要归因于冷目标增加 |
| 暖目标 | 暖目标条件命中率仍下降，主要集中于 mid 的相对变化 | 不是每类暖目标都下降；head 条件命中率可上升 |
| 训练热门 | H1 自身的 head 推荐份额高于目标构成 | H1 比 H0/C 更多推训练最热门商品；top10 集中就是训练热门偏置 |
| 推荐集中 | 旧 H1 集中，冻结后续训缓解；公平 R32 下 H1 仍比 C 集中 | 不能认定集中是唯一原因，不能直接断言热门惩罚有效 |
| 冻结后续训 | 后段性能、覆盖与集中度改善，公平 C 后段比较仍领先 | 不能分离冻结与额外 R 训练作用，也未消除时间差或证明跨 seed 稳定 |
| session 留出 | 以后可回答另一种时间/用户留出问题，作为稳健性检查有价值 | 现有证据不足以把它列为首要修复或替换原时间留出；改切分变好不等于原时间泛化已解决 |

若未来安排后续工作，当前更值得先明确的是：同一批 mid 商品在前后段的真实目标出现、推荐出现和命中变化是否一致，从而区分商品组合变化与模型集中推荐。该方向只是依据本轮结果给出的后续优先级判断，本轮没有执行新增诊断协议或实验。若以后做每用户 session 留出，应另行固定协议、保留旧划分，让 H0/H1 使用一致训练与评价设置，避免仅对某个模型更换问题。

## 7. 公平性与因果限制

{limitations}
## 8. 验收与可复核产物

{audit_text}
'''
    write(HERE / 'REPORT_C_SUPPLEMENT.md', supplement)
    write(HERE / 'FINAL_REPORT.md', final)
    receipt = dict(status='reports_generated_from_verified_statistics',
                   complete_sha256=sha(RESULT / 'COMPLETE.json'),
                   verify_sha256=sha(HERE / 'VERIFY_C_SUPPLEMENT.json'),
                   initial_files_checked=len(originals['files']), initial_artifacts_unchanged=pinned,
                   composition_fraction_fair_r32=fair_comp, composition_fraction_selected_best=best_comp,
                   h1_top10_classes_checked=h1_top_classes,
                   files={name: sha(HERE / name) for name in ('REPORT_C_SUPPLEMENT.md', 'FINAL_REPORT.md')},
                   script_sha256=sha(__file__), independent_report_review_pending=True)
    write(HERE / 'C_REPORT_BUILD_RECEIPT.json', json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
