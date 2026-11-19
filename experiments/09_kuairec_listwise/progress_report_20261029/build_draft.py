"""Build the user's Stage09 research report from audited local result files."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/09_kuairec_listwise'
OUT = Path(__file__).resolve().parent
DRAFT = ROOT / 'draft_c2fe40f7_folder/draft.xml'
OUT.mkdir(exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)
sources = set()

def load_json(path):
    path = ROOT / path
    sources.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def load_csv(path):
    path = ROOT / path
    sources.add(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def rel(path):
    return path.relative_to(ROOT).as_posix()

prefix = 'experiments/09_kuairec_listwise/'
first = {arm: load_json(prefix + f'final_analysis_20261027/{arm}_test_beam32_audit.json')['metrics']
         for arm in ['J0', 'J1', 'H0', 'H1']}
curves = load_csv(prefix + 'final_analysis_20261027/curves/validation_curve.csv')
second_val = load_csv(prefix + 'h1_recovery_20261028/METRICS.csv')
second_test = {r['model']: r for r in load_csv(prefix + 'h1_recovery_test_20261029/METRICS.csv')}
a = {r['model']: r for r in load_csv(prefix + 'h_freeze35_20261029/final_analysis/A/METRICS.csv')}
b = {r['model']: r for r in load_csv(prefix + 'h_freeze35_20261029/final_analysis/B/METRICS.csv')}
for path in [
    'docs/research/onerec_listwise/kuairec_preparation/REPORT.md',
    'docs/research/onerec_listwise/kuairec_preparation/MANIFEST.JSON',
    prefix + 'REPORT.md', prefix + 'SUMMARY.json',
    prefix + 'generalization_audit_20261028/REPORT.md',
    prefix + 'generalization_audit_20261028/RESULTS.json',
    prefix + 'formal_execution_20261027/CHECKPOINT_SID_DIAGNOSTICS_20261028.json',
    prefix + 'formal_execution_20261027/CHECKPOINT_GEOMETRY_CHECK_20261028.json',
    prefix + 'h1_recovery_20261028/REPORT.md',
    prefix + 'h1_recovery_20261028/ACTUAL_CONFIG.json',
    prefix + 'h1_recovery_test_20261029/REPORT.md',
    prefix + 'h_freeze35_20261029/H1_TEST_REPORT.md',
    prefix + 'h_freeze35_20261029/final_analysis/A/analysis_report.json',
    prefix + 'h_freeze35_20261029/final_analysis/B/analysis_report.json',
    prefix + 'h_freeze35_20261029/C_R17_REPLAY_EVIDENCE.json',
]:
    sources.add(ROOT / path)

copied_figures = []
for src, name in [
    ('final_analysis_20261027/curves/training_validation_curves.png', '01_first_training_validation.png'),
    ('h1_recovery_20261028/figures/validation_ndcg5.png', '02_decay_freeze_validation.png'),
    ('h1_recovery_20261028/figures/catalog_diagnostics.png', '03_decay_representation_diagnostics.png'),
]:
    srcpath = BASE / src
    sources.add(srcpath)
    dest = OUT / 'figures' / name
    shutil.copyfile(srcpath, dest)
    copied_figures.append(dest)

# Draw from accepted full-validation CSV, not the earlier provisional stage plot.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
epochs = [20, 23, 26, 29, 32, 35]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
for ax, key, label in zip(axes, ['list/ndcg@5', 'aggregate/ndcg@50'], ['Single-list NDCG@5', 'Aggregated NDCG@50']):
    vals = [float(a[f'A_R{e}_validation'][key]) for e in epochs]
    ax.plot(epochs, vals, 'o-', color='#c55a11', label='H1: freeze at R20')
    ax.plot([32, 35], [float(b[f'B_R{e}_validation'][key]) for e in [32, 35]], 's--', color='#3a7d44', label='H0: freeze at R32')
    ax.scatter([32], [vals[4]], marker='*', s=140, color='#c55a11', zorder=4)
    ax.set_xlabel('Completed recommender epochs (R)')
    ax.set_ylabel(label)
    ax.set_xticks(epochs)
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)
fig.suptitle('Full validation: 71,657 samples | beam 32 | seed 2024\nStar: H1 best selected by single-list NDCG@5; H0 freeze-at-R20 control pending', fontsize=11)
newfig = OUT / 'figures/04_freeze35_validation.png'
fig.savefig(newfig, dpi=190)
fig.savefig(newfig.with_suffix('.pdf'))
plt.close(fig)

xml, md = [], []

def inline(s):
    out = html.escape(str(s), quote=False)
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', out)

def title(s):
    xml.append('<title>' + inline(s) + '</title>')
    md.append('# ' + s)

def h(s, level=1):
    xml.append(f'<h{level} seq="auto">' + inline(s) + f'</h{level}>')
    md.append('#' * (level + 1) + ' ' + s)

def p(s):
    xml.append('<p>' + inline(s) + '</p>')
    md.append(s)

def bullets(items):
    xml.append('<ul>' + ''.join('<li>' + inline(x) + '</li>' for x in items) + '</ul>')
    md.append('\n'.join('- ' + x for x in items))

def table(headers, rows):
    xml.append('<table><thead><tr>' + ''.join('<th><p>' + inline(x) + '</p></th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td><p>' + inline(x) + '</p></td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>')
    md.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join('| ' + ' | '.join(str(x).replace('\n', '<br>') for x in row) + ' |' for row in rows))

def image(path, caption):
    xml.append(f'<img path="@./{rel(path)}" caption="{html.escape(caption, quote=True)}"/>')
    md.append(f'![{caption}]({path.as_posix()})')
    p(caption)

def f(row, key):
    return f'{float(row[key]):.6f}'

def main_metrics(models):
    table(['模型／状态', '单列表 NDCG@5', 'HR@5', 'Recall@5', '聚合 NDCG@50'],
          [[name, f(row, 'list/ndcg@5'), f(row, 'list/hr@5'), f(row, 'list/recall@5'), f(row, 'aggregate/ndcg@50')] for name, row in models])

def aggregate_metrics(models):
    table(['模型／状态', '聚合 K', 'NDCG@K', 'HR@K', 'Recall@K', '候选覆盖@K'],
          [[name, str(k), f(row, f'aggregate/ndcg@{k}'), f(row, f'aggregate/hr@{k}'), f(row, f'aggregate/recall@{k}'), f"{float(row[f'aggregate/coverage@{k}'])*100:.2f}%"] for name, row in models for k in [10, 20, 50]])

title('KuaiRec 列表生成与码本交互：三轮实验阶段报告（初稿）')
p('2026年10月29日。本文按“构造数据 → 比较两种列表生成结构 → 检查后期退化 → 冻结后继续训练”的顺序整理。前两轮已经完成；第三轮 H1 与 H0 晚冻结分支的训练和测试已完成，统一在 R20 冻结的 H0 对照仍在运行。C 组进度暂记录到北京时间20:28，尚未形成最终结论。')
p('本轮的问题始终是：**在一次请求生成五件商品的列表场景下，DiffGRM 加入全局码本交互，能否改善推荐效果？** 这里的 session-wise 指列表生成；本轮没有改成历史条件 SID，也没有混入之前的商品内注意力实验。')

h('数据：把 KuaiRec 整理成五商品列表任务')
h('为什么选这份数据', 2)
p('使用 KuaiRec 2.0 的 big_matrix 和作者提供的商品静态文本。原始数据有12,530,806条行为，覆盖7,176名用户和10,728个视频。它有连续的用户行为、精确时间和播放反馈，适合在当前项目中构造短列表，并复用“文本向量 → PSE → SID”的处理流程。')
p('需要先说清一个边界：KuaiRec 没有真实推荐请求 ID，所以这里构造的是**人工会话内的五商品消费续接任务**，不是 OneRec 的真实请求级数据复现，也不是预测整段自然会话。推荐结果是否在这个代理任务上有效，与是否能直接在线上生效，是两件事。')
h('清洗和列表构造', 2)
p('先去除968,004条重复事件，再隔离22条同一用户、同一时刻但无法确定顺序的事件，得到11,562,780条无歧义记录。不同时间的重复观看保留，不按商品 ID 猜测并列事件的顺序。')
bullets([
    '同一用户相邻事件间隔超过30分钟，划为新的人工会话；从会话段首按不重叠的五条行为切块。',
    '目标块必须包含五件不同商品，每件都有正播放，首末跨度不超过600秒；不合格块整块丢弃，不跳过中间行为重新拼接。',
    '输入取目标发生之前最近5—50条有效播放历史，不足50条补 PAD；历史可以跨会话。已经发生但没有成为目标的有效播放，也能进入后续历史。',
    '没有额外筛“五件总播放≥60秒”，也没有删除冷商品。正播放只代表发生过消费，不能保证五件都是强偏好正例。',
])
h('时间划分与最终规模', 2)
p('按清洗后事件时间的80%和90%分位确定两个全局切点，统一作用于所有用户；不是每个用户各拿最后两条。时间均为北京时间（UTC+8）。跨切点的整个人工会话不贡献目标块，避免同一目标会话横跨训练和评估。')
table(['集合', '目标所属时间区间', '五商品样本', '有样本用户', '历史／目标覆盖商品'], [
    ['训练', '2020-08-28 18:06:07.593 之前', '709,443', '5,849', '8,814'],
    ['验证', '上述切点至 2020-08-31 21:49:45.624 之前', '71,657', '4,733', '5,074'],
    ['测试', '2020-08-31 21:49:45.624 起', '53,194', '4,153', '5,318'],
])
p('最终共有834,294条合格列表样本，检索目录保留全部10,728件商品。清洗和会话过滤后，样本比例约为85.04%／8.59%／6.38%，不是严格的80%／10%／10%。')
p('验证和测试采用滚动历史：例如预测9月1日的一份列表时，可以使用该时刻之前已经发生的8月31日行为，但模型参数不会因此在验证或测试上更新；当前这五件目标及未来行为都不能进入输入。这既保留时间顺序，也不会因为中间隔了验证期就让测试没有可用历史。')
p('全量样本已检查历史早于目标、五目标互异、三集合目标事件不重复和会话边界；另从原始文件独立重建32名用户的样本，并进行了两次完整处理的哈希复核。当前数据满足本项目预定的代理任务规则。仍需保留限制：全目录静态文本可以离线使用，但未证实这些文本就是2020年行为发生时的版本。')

h('第一轮：两种列表生成方式，各做有码本交互／无码本交互对照')
h('模型怎样生成一份五商品列表', 2)
p('两种设计沿用 DiffGRM、PSE 和交替联合训练。冻结的 BGE-M3 把商品文本编码成1024维向量，再经 PCA 白化、归一化和 OPQ/PQ 残差量化，得到一个由四个码本索引组成的 SID。四套量化码本的形状是4×256×64；推荐器内部把每个 SID token 映射为256维向量，再汇成商品表示。这两种“维度”对应不同参数，不能混为一套码本。PCA、OPQ/PSE使用全部10,728件商品的静态目录拟合，不使用验证或测试行为反馈。')
table(['设计', '训练时学什么', '推理时怎么得到五件商品'], [
    ['J：整份列表联合建模', '把五件商品的4个码展开为20个位置，加入商品位置标识，对20码共同做遮盖与去噪。', '从20个 MASK 开始，每次按置信度选择码位并扩展候选；已填码逐步补齐为五件商品。码位选择可以跨商品位置。'],
    ['H：外层逐商品，内层四码 DiffGRM', '第j件的条件是历史＋前j−1件商品；训练使用真实前缀，每件内部仍学四码去噪。', '补齐当前商品的四码SID，将生成的SID追加到历史条件，再生成下一件，直到五件；完整列表最后查目录并消歧。推理不能使用真实目标前缀。'],
])
p('J 的“联合”指20个位置共同建模，**不代表一次前向就生成整份列表**。当前 J/H 都有20个逻辑填码阶段，都用全列表 beam32；H 没有把每件的32条路径简单扩成32⁵条。生成时约束合法 SID、处理碰撞并做列表内去重，允许推荐历史看过的商品。')
table(['分组', 'J 结构', 'H 结构', '训练时的差别'], [
    ['无交互基线', 'J0', 'H0', '更新基础码本，不使用 Mixer'],
    ['码本交互版', 'J1', 'H1', '更新基础码本，并训练全局 Full Mixer'],
])
p('Full Mixer 让全局码本向量通过注意力交互，再用修正后的码本为商品分配 SID。它改变的是全局编码，不为每个会话另造一套 SID。注意力模块的输出投影零初始化，使分叉时有效码本与基线一致。')
h('训练与评价规则', 2)
p('每种结构先共享5轮推荐器暖身，再复制完整状态给0／1两组；J与H分别暖身，不要求跨结构权重相同。之后每组执行10个“40次 Tokenizer 更新（T）＋3轮推荐器训练（R）”周期，总计35R／400T。**H0/J0也做交替联合训练，关闭 Mixer 并不等于冻结编码器。**')
table(['设置', '本轮实际值'], [
    ['推荐器骨干', '1层 Encoder＋4层 Decoder，hidden256，4头，FFN1024，dropout0.1'],
    ['批量与精度', 'microbatch256×梯度累积4，有效batch1024；训练BF16'],
    ['R 学习率', '峰值1e−3，前500次更新warmup，按原35轮日程余弦衰减'],
    ['T 学习率', '基础码本1e−5，Mixer 1e−4；首轮不衰减'],
    ['T 目标', '推荐损失＋重构损失＋0.01均衡项＋0.01锚定项；推荐梯度经过可导选码路径传入编码端'],
    ['评估与选模', 'FP32、关闭TF32、beam32、batch32；R5/8/11/…/35完整验证，按单列表NDCG@5选best，平局取早轮次'],
])
p('R阶段固定 Tokenizer，只更新推荐器；T阶段固定推荐器参数，但保留损失对编码输入的梯度，用它更新码本和 Mixer，随后刷新商品 SID。后文的“冻结”是停止T更新，仍保留此前学到的交互表示。')
p('所有正式结果都按真实商品 ID 评价，目标相关集合是接下来的五件商品，不要求预测位置和实际消费位置逐一相同。各指标定义如下，本文数值均用0—1小数表示，只有“相对变化”和覆盖率使用百分比。')
table(['指标', '含义'], [
    ['单列表 HR@5', '最高分五件列表中至少命中一个真实目标，记1，否则0；最后对样本平均。'],
    ['单列表 Recall@5', '命中的不同目标数÷5，再对样本平均。'],
    ['单列表 NDCG@5（主指标）', '相关商品排得越靠前，得分越高；按log₂(rank＋1)折扣，以五件相关商品的理想排序归一化。'],
    ['聚合 HR／Recall／NDCG@10/20/50', '最多32份完整五件列表按分数排序，按列表名次、列表内位置展开，首次出现去重后取TopK；不足补PAD。'],
    ['候选覆盖@K', '每个请求实际得到的不同有效候选数（最多K）÷K，再平均。不是全目录覆盖率。'],
])
p('因此，聚合 NDCG@50 不是“直接生成一份50件列表”的效果。聚合指标仍以相同的五件真实目标为相关集合，Recall的分母仍是5。所有验证和测试都使用同一规则；首轮训练seed只有2024。')
h('训练中先看到什么', 2)
table(['R轮次', 'J0 验证NDCG@5', 'J1', 'H0', 'H1'], [
    [str(e)] + [f(next(r for r in curves if r['arm'] == arm and int(r['epoch']) == e), 'list/ndcg@5') for arm in ['J0', 'J1', 'H0', 'H1']]
    for e in [5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
])
p('H1在R8、11、14、17、20都高于同期H0，R20时差距达到+21.27%。但此后H1明显下降，H0还在继续改善。比较各自验证best时，H1 R20为0.009372，H0 R32为0.008920，差距只有+5.07%。这些阶段点来自同一条训练轨迹，不能算作五次独立复现。')
p('J1没有在后续交替训练中超过暖身状态，选出的best仍是R5/T0；J0的best是R20。H1到R35降至0.001685，较R20下降82.02%，但中间R26有小幅回升，不是每轮都单调下降。')
image(copied_figures[0], '图1｜首轮四组完整验证与训练loss。上排分别是单列表NDCG@5和聚合NDCG@50，下排是R推荐损失和T总损失。星标统一表示按单列表NDCG@5选出的best。')
p('Loss也提供了线索：H1在T更新后的R训练会反复出现损失跳升，例如R20平均推荐loss为2.951，R21升至3.058，R23回到2.977，R24又升至3.325。J1则出现“loss明显下降、商品级指标却变差”的情况。因此，不能只凭SID预测loss下降就认定模型学得更好了。')
h('四个验证best的测试结果', 2)
p('四个模型按各自验证best锁定后，在同一53,194条测试样本、4,153名用户上评估。下表和后续测试表都已逐样本复算；J1的R5模型还没有做过T更新，不能解释成“已经学会交互的模型”。')
main_metrics([(name, first[arm]) for name, arm in [('J0 R20', 'J0'), ('J1 R5', 'J1'), ('H0 R32', 'H0'), ('H1 R20', 'H1')]])
p('同一架构下，J1相对J0的测试NDCG@5下降20.53%；H1相对H0下降7.71%，聚合NDCG@50下降6.84%。首轮没有证明交互能提高测试质量。H1−H0的测试NDCG@5绝对差95%用户聚类区间为[−0.001103，−0.000237]，但这只描述本次固定模型在这些用户上的差异，不代替多seed检验。')
aggregate_metrics([(name, first[arm]) for name, arm in [('J0 R20', 'J0'), ('J1 R5', 'J1'), ('H0 R32', 'H0'), ('H1 R20', 'H1')]])
h('验证领先、测试落后：再按数据时间检查', 2)
p('这里固定使用H0 R32和H1 R20，不重新选模型，把验证样本按目标时间中点2020年8月30日07:57:21分成前后两段。**前后段指行为发生的日期，不是训练轮次。**')
table(['数据时段', '样本数', 'H0 NDCG@5', 'H1 NDCG@5', 'H1相对变化'], [
    ['验证前段', '38,644', '0.009963', '0.011128', '+11.70%'],
    ['验证后段', '33,013', '0.007699', '0.007316', '−4.97%'],
    ['完整测试', '53,194', '0.008607', '0.007944', '−7.71%'],
])
p('H1的完整验证优势主要来自前段，到验证后段已经消失；测试发生得更晚，下降与这个方向相符。它提示时间泛化可能有问题，但不能仅凭这一张表确定原因。前后两段都已参与原来的best选择，所以后段不是新的独立验证集。')
bullets([
    '不是简单的用户换了一批：只看验证和测试共有用户，H1仍从验证+4.49%变成测试−8.36%。',
    '也不能主要归因于冷商品增加：全已见目标组从+4.90%变成−8.62%；含冷目标组的点估计反而仍为正。',
    'H1的首选输出更集中：测试时最常推荐的10件商品占全部推荐位置的45.23%，H0为29.78%；实际推荐过的不同商品为613件和771件。它可能更依赖部分商品或时期中的模式，但这还不是致因证明。',
    '改用“前段选模、后段检查”会选出不同的H0，后段H1仍可能略占优。因此不能说所有选模方式都会反转，更不能事后改选模规则替换正式结果。',
])
p('当前保留两个猜想：一是H1学到的排序更偏向局部时段和少数商品，后续数据变化时收益难以保持；二是H1在编码继续变化时没能稳定训练到更好的推荐器状态。两者可能同时存在，需要分别验证。')
h('后期为什么退化：检查码字和表示', 2)
p('对保存的best/last检查点重算商品SID，并检查有效码本、碰撞、重构和交互修正幅度。CPU FP32与独立FP64最近中心重算都与保存SID一致，排除了这些端点“只是SID缓存没刷新”的解释。')
table(['模型状态', '不同SID数', '碰撞桶内商品比例', '四码位使用码字数', '交互修正／基础码本范数'], [
    ['H0 best R32', '9,513', '14.02%', '256 / 256 / 256 / 256', '0'],
    ['H0 R35', '9,513', '14.01%', '256 / 256 / 256 / 256', '0'],
    ['H1 best R20', '9,600', '12.91%', '181 / 137 / 190 / 218', '0.824'],
    ['H1 R35', '2,551', '91.41%', '8 / 17 / 15 / 13', '4.230'],
    ['J1 R35', '1', '100.00%', '1 / 1 / 1 / 1', '4.763'],
])
p('H1后期不是只有指标波动，而是商品表示发生了明显退化：R20到R35，所有商品至少一个码位变化，最近中心重构SSE由0.74464升到0.97866。J1甚至只剩一类SID。这里的“碰撞比例”是处在共享SID桶中的商品占比，不等于推荐错误率；整体范数比也不是每个中心都放大同样倍数。')
p('一个值得检验的解释是：后期R学习率一直下降，T的码本和Mixer学习率却保持不变，编码还在明显变化，推荐器可能难以及时适应；同时也可能存在T目标与最终推荐质量不完全一致的问题。端点诊断不能定位第一次塌缩的准确更新步，更不能证明学习率是唯一原因。')
p('还要把两件事分开：**后期R35塌缩，不会倒过来影响已经保存的R20测试模型。** 固定H1 R20编码下，目标出现次数中落入碰撞桶的比例从验证5.79%降到测试5.06%；这个分母不同于表中全目录商品比例。无碰撞目标样本里仍出现验证到测试的反转。后续冻结实验是在验证“停止编码变化后，推荐器还有没有继续改善的空间”，不是预先认定已经解释了全部泛化问题。')
p('首轮还在固定1,024条验证样本上做过beam32/256检查，没有重新选模。更大的beam没有稳定改善H组质量，不能把完整测试下降简单归为搜索宽度不足；这项子集诊断也不替代完整测试。')

h('第二轮：从同一 H1 R20 出发，减弱编码更新还是直接冻结')
h('为什么做、具体改什么', 2)
p('第一轮已经有原设置继续训练的历史轨迹，所以没有再原样重训一支。这一轮直接把同一个H1 R20/T200完整状态分为Decay和Freeze，预先固定到R26结束，比较“降低后续编码更新力度”和“完全停止编码更新”。旧原设置轨迹只作历史参照。')
table(['分支', '编码端', '推荐器', '固定终点'], [
    ['Decay', '基础码本和Mixer继续更新；再做两次40T，学习率同时余弦衰减', '继续原日程训练6R', 'R26/T280'],
    ['Freeze', '固定基础码本、已学到的Mixer和商品→SID映射；跳过T，不推进T噪声时钟', '继续原日程训练6R', 'R26/T200'],
])
p('Decay的T学习率采用原学习率×0.5×[1＋cos(πu/400)]，u为累计T更新数200—279；第一次实际使用5e−6／5e−5，最后一次约2.09e−6／2.09e−5，分别对应基础码本和Mixer。它是在R20后立即减半再继续衰减，不是整段训练重新开始。')
p('两组都继承完整优化器、随机状态和原35轮R学习率日程，不重新warmup，不改变数据顺序。Freeze继续训练的是推荐器内部SID embedding、Encoder和Decoder，**保留交互，不等于删除Mixer或变成H0**。R23看过程，R26是预先固定终点，没有在这一轮重新挑一个best。')
h('完整验证和诊断结果', 2)
main_metrics([(name, row) for name, row in zip(['共同 H1 R20', 'Decay R23', 'Decay R26', 'Freeze R23', 'Freeze R26'], second_val)])
p('到固定R26，Freeze的完整验证NDCG@5较R20提高10.59%，Decay下降34.28%。Freeze−R20的绝对差95%用户配对聚类区间为[+0.000642，+0.001320]。旧原设置R26为0.006438；Decay为0.006159，因此不能因为中间某个点稍好，就认定本次学习率衰减已经解决退化。')
image(copied_figures[1], '图2｜R20→R26：左侧为71,657条完整验证，右侧为固定8,192条诊断子集，两个总体分开展示。灰线是旧训练轨迹，不是新重跑的第三组。')
p('进一步在每次T块结束、推荐器还没继续适应时检查固定子集：Decay在T240、T280后NDCG@5分别立即下降10.19%和18.42%；接着训练R会有局部恢复，但最后仍没回到起点。Freeze的全量验证则从R20到R23、R26逐步上升。')
table(['Decay状态', '不同SID数', '碰撞商品比例', '四码位使用数', '修正／基础范数'], [
    ['R20/T200', '9,600', '12.91%', '181 / 137 / 190 / 218', '0.824'],
    ['T240', '9,630', '12.50%', '161 / 122 / 159 / 184', '1.163'],
    ['T280', '9,643', '12.35%', '144 / 101 / 134 / 158', '1.490'],
])
image(copied_figures[2], '图3｜第二轮的编码诊断。Decay交互修正和重构误差持续增加，码字使用减少；但SID组合数没有塌缩，碰撞率略降。Freeze停在同一个T200编码状态。')
p('这一点很关键：**本次Decay推荐效果下降时，并未观测到大规模SID碰撞塌缩。** 只监控“用了多少码字”或“碰撞率是否增加”还不够。编码变化是否破坏了推荐器已经学到的对应关系，可能比碰撞数本身更值得关注；但目前还没有把码本几何变化、SID切换和推荐器适应能力分别隔离出来。')
h('固定R26的测试结果', 2)
second_models = [('原 H1 R20', second_test['R20']), ('Decay R26', second_test['decay']), ('Freeze R26', second_test['freeze'])]
main_metrics(second_models)
p('Freeze测试NDCG@5较原R20提高15.27%，绝对差95%区间为[+0.000834，+0.001594]；Decay下降42.51%。因此，这次结果支持“先保留已学到的编码，让推荐器继续适应”，而本次衰减设置没有做到这一点；不代表所有衰减方案都无效，也不能据此称R20的Tokenizer已经收敛。')
aggregate_metrics(second_models)
p('Freeze R26测试NDCG@5也比旧H0 best R32高约6.39%，但两组冻结时点和T更新量不同，H0当时还没获得同等冻结续训机会。这只能算一个值得补对照的信号，不能直接把差值都归给码本交互。')

h('第三轮：把冻结续训延长到R35，并补齐H0对照')
h('三组各自回答什么问题', 2)
table(['分支', '实际路线', '想回答的问题'], [
    ['A：H1在R20冻结', '沿用Freeze R26/T200，再训R27—35；编码一直保持R20状态', '已有改善能否保住，继续训练还有多少收益？'],
    ['B：H0在R32冻结', '从旧H0 best R32/T360出发，冻结编码，再训R33—35', '无交互模型从自身较好状态出发，晚期冻结还能否改善？'],
    ['C：H0在R20冻结', '从共享R5重放原H0到R20/T200，验收后冻结，继续到R35', '统一R20冻结后，H1相比H0是否仍有优势？这是主对照。'],
])
p('之所以需要C，是因为旧H0的R20完整权重没有保留下来，不能从R32倒推，也不能把H1关闭Mixer当作H0。因此从已保存的R5完整状态重放，并逐节点核对原训练轨迹和预测；后续每个R轮和T边界都保留完整检查点，避免再丢失关键对照状态。')
p('三组沿用原35轮R学习率日程、原阶段编号和数据顺序，不重新warmup，也不额外拉长终点。A/C主比较是固定R35；另外在共同R20/23/26/29/32/35节点上按完整验证NDCG@5各选best。B只把父R32与固定R35并列。B启动时R学习率约1.88e−5，已在原日程尾部，它的改善机会与A/C不同。')
h('A：H1冻结后能否继续改善', 2)
main_metrics([(f'H1 Freeze R{e}', a[f'A_R{e}_validation']) for e in epochs])
p('上表全部是完整验证。H1从R26到R32继续上升，验证best选在R32；R35小幅回落，但没有重新出现原训练中R35的严重塌缩。')
image(newfig, '图4｜第三轮已完成分支的完整验证。H1在R20冻结后改善到R32，再略回落；H0在R32才冻结的B分支没有继续改善。C组尚未完成，未画出任何预测结果。')
h('已完成模型的测试结果', 2)
third_models = [('H1 Freeze R26（起点参照）', a['A_R26_test']), ('A：H1 Freeze R32（验证best）', a['A_R32_test']), ('A：H1 Freeze R35（固定终点）', a['A_R35_test']), ('旧 H0 best R32', b['B_R32_test']), ('B：H0 Freeze R35', b['B_R35_test'])]
main_metrics(third_models)
table(['测试NDCG@5比较', '相对变化', '绝对差95%用户聚类区间'], [
    ['H1 Freeze R32 − Freeze R26', '+2.99%', '[−0.000050，+0.000610]'],
    ['H1 Freeze R35 − Freeze R26', '−2.19%', '[−0.000525，+0.000142]'],
    ['H1 Freeze R32 − 原H1 R20', '+18.72%', '[+0.001087，+0.001897]'],
    ['H1 Freeze R32 − 旧H0 best R32', '+9.57%', '[+0.000391，+0.001284]'],
    ['H1 Freeze R35 − 旧H0 best R32', '+4.06%', '[−0.000101，+0.000793]'],
    ['B H0 Freeze R35 − 旧H0 best R32', '−1.74%', '[−0.000353，+0.000053]'],
])
p('目前能说的是：冻结后的较好表现保留到了后续训练，H1验证best R32的测试也高于原R20和旧H0 best。但R32相对Freeze R26的增量很小，区间跨零；固定R35反而略低于R26，也不能确认这种小下降超出了用户抽样波动。因此，“冻结避免了原来的严重退化”已有支持，“越往后训越好”没有得到支持。')
p('B的验证NDCG@5从0.008920降到0.008634，测试从0.008607降到0.008457；这次从自身best出发的晚冻结没有带来进一步改善。不过，**A优于B仍不能单独证明Mixer有效**，因为冻结点和Tokenizer更新量不同。')
aggregate_metrics(third_models)
h('C：当前完成到哪里，最终还缺什么', 2)
p('截至20:28，C已完成原H0的R20训练，正在进行该节点的全量验证；当时已处理10,016／71,657条样本。R8、R11、R14、R17均已完成全量验证，并与原H0对应节点逐样本核对：71,657条样本的Top5商品完全一致，主指标差值为0。这支持当前重放没有发生可见偏离，但不等于找回了原R20的逐字节权重。R20分界验收及冻结后R23—35结果尚待完成。')
table(['C已核验节点', '完整验证NDCG@5', '与原H0同节点的Top5一致率'], [
    ['R8', '0.006381', '100%'], ['R11', '0.005829', '100%'], ['R14', '0.008307', '100%'], ['R17', '0.007687', '100%'],
])
p('等C完成后，首先比较A/C固定R35的测试，再补充各自按共同验证网格选出的best。只有把冻结时间、R训练预算和评价规则对齐，才能更清楚地区分“冻结本身有帮助”与“冻结后仍保留了码本交互的收益”。')

h('目前的判断与下一步')
p('这一段实验的收获，不是已经证明了交互稳定有效，而是把问题收窄了：**原交互版的后期训练确实有问题；从R20保留编码、继续训练推荐器，可以在当前数据上明显改善验证和测试；但收益中有多少来自交互，还需要同冻结点的H0来确认。**')
p('下一步先完成C和预定比较，不再根据已经看到的测试结果改学习率、冻结点或终点。若同设置下H1仍有优势，再用预先固定的冻结计划补其他seed和额外时间窗口；若优势消失，则应优先把冻结策略视为共同训练改进，而不是给Mixer归功。')
p('报告中的因果解释均保留为猜想。首轮用户聚类bootstrap为10,000次，第二、三轮为2,000次，均按样本平均计算95%区间；这些区间不包含训练随机性和选择R20冻结点的不确定性。测试集已经在前几轮分析中被查看，后续复用只能用于这轮探索性比较，不能再称全新未见测试。')
p('资料来源：项目的数据准备验收、首轮四组结果与时间泛化审计、R20→R26续训及测试报告、第三轮A/B逐样本复算结果。所有正式测试使用同一53,194条样本与beam32协议；本初稿附带的本地Markdown、图表和来源哈希清单单独保存，原实验报告不覆盖。')
xml.append('<p>数据原始说明：<a href="https://kuairec.com/">KuaiRec 官方页面</a>。</p>')
md.append('数据原始说明：[KuaiRec 官方页面](https://kuairec.com/)。')

DRAFT.write_text('\n'.join(xml) + '\n', encoding='utf-8')
(OUT / 'REPORT_DRAFT.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
(OUT / 'SOURCE_MANIFEST.json').write_text(json.dumps({
    'status': 'draft', 'scope': 'stage09_three_experiment_rounds',
    'current_C_snapshot': '2026-10-29T20:28:42+08:00',
    'source_files': [{'path': rel(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(sources)],
    'draft_xml': rel(DRAFT),
    'figures': [{'path': rel(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in copied_figures + [newfig]],
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'draft': rel(DRAFT), 'markdown': rel(OUT / 'REPORT_DRAFT.md'), 'characters': len('\n'.join(md)), 'tables': sum(s.startswith('<table>') for s in xml), 'figures': 4}, ensure_ascii=False))
