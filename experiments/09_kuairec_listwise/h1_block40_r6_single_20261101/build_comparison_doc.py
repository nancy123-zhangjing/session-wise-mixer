"""Write the authorized Feishu release candidate only from passed E audits."""
import hashlib
import json
from html import escape
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
D=ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031'
FREEZE=ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029'
WORK=ROOT/'draft_3cf6198c_folder'
KEY='list/ndcg@5'
sources={}

def read(path):
    data=path.read_bytes();sources[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    return json.loads(data)

def p(text):return '<p>'+escape(str(text))+'</p>'
def h(text):return '<h1 seq="auto">'+escape(text)+'</h1>'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th background-color="light-gray">'+p(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+p(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def fmt(x):return f'{x:.6f}'
def pct(x):return f'{100*x:+.2f}%'
def pic(name,caption):
    path=HERE/'final_report'/f'{name}.png'
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return '<img path="@./'+path.relative_to(ROOT).as_posix()+'" caption="'+escape(caption,quote=True)+'"/>'

def main():
    a=read(HERE/'final_analysis/ANALYSIS.json');pr=read(HERE/'final_process/PROCESS.json');te=read(HERE/'final_temporal/TEMPORAL.json')
    restore=read(HERE/'COLD_RESTORE.json');op=read(HERE/'OPERATIONAL_AUDIT.json')
    if any(x['status']!='passed' for x in (a,pr,te,restore)):raise ValueError('Passed scientific audits and restoration required')
    if op['status']!='passed_async_operational_delivery_audit':raise ValueError('Readonly execution audit required')
    read(ROOT/'outputs/stage09_block40_single_20261101_fresh_r1/LOCAL_EXTRACTION_VERIFIED.json')
    d=read(D/'final_analysis/ANALYSIS.json');old=read(FREEZE/'final_analysis/A/analysis_report.json');c=read(FREEZE/'final_analysis/C/analysis_report.json');b=read(FREEZE/'final_analysis/B/analysis_report.json')
    if any(x['status']!='passed' for x in (d,old,c,b)):raise ValueError('Historical accepted sources required')
    e=a['points'];dm=d['points'];am=old['models'];cm=c['models'];best=a['selected_epoch']
    fixed={'A':am['A_R35_test']['metrics'],'D':dm['D_R35_test']['metrics'],'E':e['E_R35_test']['metrics']}
    ce=a['comparisons']['E_minus_D_fixed_test'][KEY];ca=a['comparisons']['E_minus_A_fixed_test'][KEY]
    selected={'A':am['A_R32_test']['metrics'][KEY],'D':dm['D_R23_test']['metrics'][KEY],'E':e[f'E_R{best:02d}_test']['metrics'][KEY]}
    paragraphs=['<title>Tokenizer更新节奏对照：20T＋3R、40T＋6R与原H1／冻结方案</title>',
        p('KuaiRec五商品列表推荐｜2026年11月2日｜单训练seed2024。本文比较已完成实验；事实、解释假设与仍未回答的问题分开表述。'),
        p(f"固定R35/T200时，E（40T＋6R）的测试NDCG@5为{fmt(fixed['E'][KEY])}，相对D（20T＋3R）{pct(ce['relative_delta'])}，相对冻结A{pct(ca['relative_delta'])}。各自最佳节点另列，不能用最佳结果替代固定终点结论。"),
        h('为什么做这两组对照'),
        p('原H1在后期继续更新Tokenizer时推荐效果明显下降；在R20/T200冻结Tokenizer、只继续训练推荐器的路线取得了更好的结果。由此尝试：能否减少T更新、把它分散到训练后期，既保留编码学习，又让推荐器有时间适应？D和E分别采用下面两种安排。'),
        p('这里T表示Tokenizer的一次优化器更新，R表示推荐器完整训练一轮；T200不是200个epoch。Tokenizer包含基础码本和Full Mixer，推荐器自己的SID embedding仍随R更新。'),
        table(['方案','共享5R之后的训练安排','总终点'],[
            ['原H1','10次（40T＋3R）','R35/T400'],
            ['A：R20冻结','前5次（40T＋3R）后冻结T，再训练15R','R35/T200'],
            ['D：20T＋3R','10次（20T＋3R）','R35/T200'],
            ['E：40T＋6R','5次（40T＋6R）','R35/T200']]),
        p('A、D、E沿用同一个H1 R5/T0起点、相同的前200次有序T样本、相同R样本顺序、数据切分、模型和评价规则；保留原35R学习率日程与400T噪声退火时钟。E最终采用与D一致的单卡训练方式、全局batch1024。'),
        p('D与E总更新量相同，但同时改变了T块大小、SID刷新次数和连续R长度。这是训练安排整体的对照，不能把差异唯一归因于“多给推荐器3轮适应”。'),
        h('固定R35/T200：同预算主对照'),
        p('以下均为同一测试集53,194条列表、4,153名用户。NDCG@5评价最高分单份五商品列表的真实商品排序；HR@5表示至少命中一件，Recall@5是命中目标数除以5。'),
        table(['方案','测试NDCG@5','HR@5','Recall@5'],[[n,fmt(v[KEY]),fmt(v['list/hr@5']),fmt(v['list/recall@5'])] for n,v in fixed.items()]),
        table(['比较','NDCG@5绝对差','相对变化','95%用户配对区间'],[[label,f"{z['delta']:+.6f}",pct(z['relative_delta']),f"[{z['confidence_interval'][0]:+.6f}, {z['confidence_interval'][1]:+.6f}]"] for label,z in [('E−D',ce),('E−A',ca),('D−A',d['comparisons']['fixed_test'][KEY])]]),
        p('区间采用2000次用户聚类配对bootstrap，按样本平均计分。它刻画固定模型在用户样本上的不确定性，不覆盖训练seed差异、选模与观察旧结果后提出方案的不确定性；没有据此追加实验。'),
        h('各自最佳节点与此前实验：单列补充'),
        table(['方案','验证选出的best R/T','测试NDCG@5'],[
            ['A：冻结','R32/T200',fmt(selected['A'])],['D：20T＋3R','R23/T120',fmt(selected['D'])],['E：40T＋6R',f'R{best}/T{(1+(best-6)//6)*40}',fmt(selected['E'])]]),
        p('A、D、E共同从R20/23/26/29/32/35按完整验证NDCG@5选best，平局取较早节点，测试不参与选模。各自best的R/T不同，不能称作等预算比较。'),
        p(f"E的best测试点估计比D的best高{pct(selected['E']/selected['D']-1)}，但固定R35却低于D。因此不能笼统说E在整个训练过程中都更差，也不能用较好的best掩盖末期下降；E与D的best之间尚未计算用户配对区间。"),
        table(['此前实验／节点','测试NDCG@5','与本次的关系'],[
            ['原H1 best：R20/T200',fmt(am['A_R20_test']['metrics'][KEY]),'T总量相同，但R总量与T发生时机不同'],
            ['H1 Freeze：R26/T200',fmt(am['A_R26_test']['metrics'][KEY]),'A冻结路线的较早节点，并非另一种冻结机制'],
            ['H1 Decay：R26/T280','0.004567','降低T学习率后继续更新；不是降低更新频率'],
            ['旧H0 best：R32/T360',fmt(am['B_R32_test']['metrics'][KEY]),'无Mixer；R/T预算不同的历史基线'],
            ['B：H0在R32冻结至R35/T360',fmt(b['models']['B_R35_test']['metrics'][KEY]),'晚期冻结补充；冻结时点和T总量与A不同'],
            ['C：H0在R20冻结至R35/T200',fmt(cm['C_R35_test']['metrics'][KEY]),'与A同冻结点、同终点，用于考察Mixer路线']]),
        p('此前A固定R35对C固定R35提高8.48%，用户配对区间为[+0.000251,+0.001151]。这支持本次冻结方案下保留已学Mixer的价值，仍不是跨seed稳定性的证明。'),
        h('同样是T200，为什么更多R未必更好'),
        pic('validation_curve','完整验证集71,657条；横轴为R轮数。原H1最终T400，其余图示冻结或调度分支最终T200，不能把同一R点都当成相同T预算。'),
        table(['相同R／T节点','D验证NDCG@5','E验证NDCG@5','E−D'],[[f'R{r}/T{t}',fmt(dm[f'D_R{r:02d}_validation']['metrics'][KEY]),fmt(e[f'E_R{r:02d}_validation']['metrics'][KEY]),f"{e[f'E_R{r:02d}_validation']['metrics'][KEY]-dm[f'D_R{r:02d}_validation']['metrics'][KEY]:+.6f}"] for r,t in [(11,40),(17,80),(23,120),(29,160),(35,200)]]),
        p('E在R29/T160的完整验证表现高于D，到了R35/T200两者完整验证分数非常接近，而测试上E更低。验证与测试来自不同时间段，不能把两者的差异当作统计口径错误；也不能据此直接确定哪一种分布变化造成差距。'),
        p('确定的设计事实是：训练并不是先造好一套编码，再让推荐器在完全相同的编码上多学几轮。T更新时接收的推荐器梯度会随推荐器状态变化；R更新时面对的SID又取决于此前T更新。因此，即便更新次数和样本总量相同，训练路径仍然不同。'),
        p('A完成最后一次T后有15R保持编码不变，E有6R，D只有3R；而后期R的学习率沿原35轮日程下降。于是“累计R更多”不自动等于“对最终编码有更多有效适应”。这是一个有根据的解释方向，但尚未单独操纵学习率或最终编码后的训练长度来识别因果。'),
        h('过程证据：T后的下降能否在6R内恢复'),
        pic('cycle_recovery','固定8192条诊断样本；每周期比较T前、40T后与随后1R／3R／6R。不能与完整验证或测试数值混用。')]
    rows=[]
    for cycle in range(1,6):
        values=[v for v in pr['curves'] if v['cycle']==cycle]
        if len(values)!=5:raise ValueError('Five matched diagnostic positions required')
        rows.append([cycle,cycle*40,*[fmt(v[KEY]) for v in values],f"{values[4][KEY]-values[3][KEY]:+.6f}"])
    paragraphs += [table(['周期','累计T','T前','40T后','1R后','3R后','6R后','6R减3R'],rows),
        p('事实：五个周期只有第三周期的6R分数高于3R，其余四次更低。第3R至第6R之间没有T更新，SID保持不变；所以这些区段的下降不能归因于期间SID再次刷新。最后一个周期从40T后到首R后继续下降，随后6R结束也未恢复到T前水平。'),
        p('这一表直接观察“多给3R以后是否继续改善”。它能区分短期掉分与后续恢复，但无法单独证明下降来自Tokenizer过拟合，或恢复只来自对新SID的适应。'),
        pic('sid_change_40T','全目录10,728件商品，任意一个SID槽变化即计为改变。D与E都按实际相同累计40T端点比较，未相加两个20T百分比。'),
        table(['累计T','E该40T区间SID改变商品比例','D同40T区间'],[[z['cumulative_T'],f"{100*z['E_sid_item_change_fraction']:.2f}%",f"{100*z['D_sid_item_change_fraction']:.2f}%"] for z in pr['paired_40T_endpoints']]),
        p('E最后40T改变51.53%的商品SID，D同40T端点为76.38%；E末期碰撞商品比例约12.90%，也没有重现原H1末期91.41%的严重碰撞塌缩。但E固定测试仍低于D：移动更少、没有塌缩，并不自动保证推荐更好。'),
        p('原H1仅保留T0、T200、T400快照，无法恢复逐20T或40T完整变化率。原H1后期发生严重碰撞塌缩；D没有重现这种塌缩，却仍然掉分。编码区分能力、SID移动与推荐质量必须分别观察，不能把“碰撞更少”直接当成“推荐更好”。'),
        h('时间泛化与结论边界'),
        p('时间分析使用固定R35的验证预测，沿用原切点1598745441188500微秒、训练期频次分类；前段38,644条、后段33,013条。它是静态数据上的时间切片，不能证明模型获得或适应了线上新到达数据。'),
        table(['固定R35模型','前段验证NDCG@5','后段验证NDCG@5'],[[{'A_Freeze_R35':'A：冻结','D_Spread_R35':'D：20T＋3R','E_Block40_R35':'E：40T＋6R'}.get(n,n),fmt(v['early']['metrics'][KEY]),fmt(v['late']['metrics'][KEY])] for n,v in te['models'].items()]),
        p('后段绝对差变小并不自动表示时间泛化更好，还要看前后段各自水平、相对差及其区间。完整分解和区间保留在本地报告；本节不把分布变化解释为唯一原因。'),
        p('本次E相对D的点估计是前段略高、后段更低；但“E−D差值的后段减前段”95%用户区间为[−0.001346,+0.000067]，跨零，尚不能认定这一相对时间衰减稳定存在。E相对A在前后两段都更低，负差缩小也不等于超过A。'),
        table(['已有证据支持／能检验','仍不能断言'],[
            ['相同总R/T下更新安排会对应不同结果','不能只凭次数判断收敛；不能将调度差异唯一归于R适应时间'],
            ['冻结后继续R已有正向结果，D/E提供不同的继续更新对照','没有证明Tokenizer已经收敛，也没有证明所有降低T频率方案均无效'],
            ['过程诊断可观察T后掉分和随后恢复','没有确认MEDA所指的one-epoch overfitting；本次也没有实施随机重置']]),
        h('结果来源与验收范围'),
        p('新E预测经独立指标复算，更新计数、样本顺序、选模和只读评估已核验；完整备份经SHA与独立解压，关键checkpoint完成CPU真实恢复。A、D及此前实验直接复用已验收结果，没有重跑旧实验。资源收尾以单独最终交付收据为准。'),
        p('来源：原第09阶段REPORT；h1_recovery_test_20261029/REPORT；h_freeze35_20261029/REPORT；D的final_report_r2/REPORT；原H1保存SID补充统计；本次E的ANALYSIS、PROCESS、TEMPORAL与final_report/REPORT。所有路径均位于C:/Users/nancy/Documents/experiment1/experiments/09_kuairec_listwise。')]
    # Record prose source bindings as well as machine-readable result bindings.
    for path in [ROOT/'experiments/09_kuairec_listwise/REPORT.md',ROOT/'experiments/09_kuairec_listwise/h1_recovery_test_20261029/REPORT.md',FREEZE/'REPORT.md',ROOT/'experiments/09_kuairec_listwise/h1_sid_change_20261101/REPORT.md',HERE/'final_report/REPORT.md']:
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    draft=WORK/'draft.xml'
    with draft.open('x',encoding='utf-8') as f:f.write('\n'.join(paragraphs)+'\n')
    (WORK/'SOURCE_BINDINGS.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'draft':str(draft),'sources':len(sources),'images':3},ensure_ascii=False))

if __name__=='__main__':main()
