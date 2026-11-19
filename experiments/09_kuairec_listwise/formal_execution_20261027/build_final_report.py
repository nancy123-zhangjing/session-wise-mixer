"""Create Stage09 reports from independently replayed, frozen evaluations."""
import csv,hashlib,json,os,re
from datetime import datetime,timezone
from pathlib import Path

F=Path(__file__).resolve().parent
E=F.parent
ROOT=E.parents[1]
A=E/'final_analysis_20261027'
D=ROOT/'outputs/stage09_delivery/goal_20261027_r1'
R=ROOT/'outputs/stage09_restore/goal_20261027_r1'
G='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(v,msg):
    if not v:raise RuntimeError(msg)
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def num(x):return f'{x:.6f}'
def percent(x):return f'{100*x:.2f}%'
def ci(x):return '['+', '.join(f'{v:+.6f}' for v in x)+']'
inventory=read(F/'FINAL_LOCAL_ASSET_VERIFICATION_20261028.json')
need(inventory['status']=='PASS','Assets not accepted')
freeze=read(F/'FOUR_BEST_FREEZE_20261028.json')
release=read(F/'ALL_GPU_RELEASE_ACCOUNTING_20261028.json')
geometry=read(F/'CHECKPOINT_GEOMETRY_CHECK_20261028.json')
arms=['J0','J1','H0','H1'];variants=['test_beam32','validation_subset_beam32','validation_subset_beam256']
audits={a:{v:read(A/f'{a}_{v}_audit.json') for v in variants} for a in arms}
pairs={a:{v:read(A/f'{a}_{v}_paired_users_10000.json') for v in variants} for a in ['J','H']}
summaries={a:{v:read(R/'sidecars/evaluations/evaluations'/(a+'_'+v+('_r2' if a=='H1' else ''))/'summary.json') for v in variants} for a in arms}
for a in arms:
    for v in variants:need(audits[a][v]['passed'],'Metric replay failed')
index=read(A/'full_validation_selected_audits.json')
valid={}
for row in index['rows']:
    d=read(Path(row['audit_path']))
    for role in row['roles']:valid[role]=d
results={a:dict(best_state=freeze['arms'][a]['best_state'],best_sha256=freeze['arms'][a]['best']['sha256'],
    best_validation=valid[a+':selected_best'],fixed_R35_validation=valid[a+':fixed_R35'],
    evaluations=audits[a],inference_seconds={v:summaries[a][v]['seconds'] for v in variants},
    endpoint_diagnostics=geometry['arms'][a]) for a in arms}
conclusions=[
    'H1 best validation NDCG@5 exceeds H0 by 5.07%, but full-test NDCG@5 is lower by 7.71%; no demonstrated test gain.',
    'J1 selected checkpoint is the shared R5/0T warmup; its test result is not evidence of trained Mixer effectiveness.',
    'H1 and J1 exhibit late code-use collapse. Saved SIDs match independent FP32 and FP64 nearest-code calculations.',
    'Larger beam does not consistently improve item relevance; H pair subset differences have intervals crossing zero.',
    'Prioritize a bounded R20 continuation diagnostic, not an unbounded H1-only hyperparameter search.',
    'User bootstrap conditions on one seed and fixed trained models; it does not establish training-seed stability.'
]
summary=dict(schema='stage09_final_summary_v1',status='accepted',created_at_utc=datetime.now(timezone.utc).isoformat(),
    seed=2024,task='KuaiRec artificial five-item consumption continuation lists',
    counts=dict(catalog=10728,train=709443,validation=71657,test=53194,test_users=4153,validation_subset=1024),
    execution_sources=dict(main='1c423662d9a25a926328ee6466c30c0f2857b4f6',diffgrm='98b834cc5ce4428d1a95b3e4f9651833b5fbd5cd',genrec='b6c8b41d1d20a812590dae11408b69b1ed2a8313'),
    runtime_config_sha256='d8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb',
    freeze_sha256=sha(F/'FOUR_BEST_FREEZE_20261028.json'),
    protocol=dict(selection='full validation single-list item-level NDCG@5, earlier tie',test_beam=32,
        inference_precision='fp32',training_precision='bf16',max_candidate_lists=32,
        aggregation='list score order then item position first occurrence',
        cold_group='at least one target absent from actual train_seen item IDs',
        bootstrap='10000 paired user-cluster draws; sample-mean primary, user-mean secondary',
        test_used_for_tuning=False),
    actual_work=dict(recommender_epochs_excluding_duplicated_warmup=130,recommender_updates=90090,tokenizer_updates=1600,
        full_validations=42,final_test_runs=4,validation_beam_runs=8,failed_eval_attempts=1),
    results=results,paired_comparisons=pairs,conclusions=conclusions,
    resources=dict(gpu_hours=release['allocated_gpu_hours'],gpu_seconds=release['allocated_gpu_seconds'],job_count=13,
        all_gpu_released=True,price_points_per_gpu_hour=1,estimated_points=release['estimated_points'],settlement_verified=False,budget_gpu_hours=75),
    locations=dict(remote=G,archive=str(D),restore=str(R),analysis=str(A)),
    evidence=dict(asset_verification_sha256=sha(F/'FINAL_LOCAL_ASSET_VERIFICATION_20261028.json'),release_sha256=sha(F/'ALL_GPU_RELEASE_ACCOUNTING_20261028.json'),geometry_sha256=sha(F/'CHECKPOINT_GEOMETRY_CHECK_20261028.json')))
save(E/'SUMMARY.json',summary)

draft=(F/'REPORT_DRAFT.md').read_text(encoding='utf-8')
protocol='## 1.'+draft.split('## 1.',1)[1].split('## 6.',1)[0]
def relink(m):
    target=m.group(2)
    if '://' in target or target.startswith('#'):return m.group(0)
    path,sep,frag=target.partition('#')
    return '['+m.group(1)+']('+Path(os.path.relpath((F/path).resolve(),E)).as_posix()+(sep+frag if sep else '')+')'
protocol=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',relink,protocol)
report='# 09｜KuaiRec五商品列表生成与码本交互：最终报告\n\n'
report+='2026-10-28。四组训练、统一测试、逐样本复算、用户配对区间及本地恢复均已验收。所有GPU已释放；本轮未追加调参或新seed。\n\n'
report+='**本轮没有证明码本交互提高测试推荐质量。H1验证best有小幅优势，但测试NDCG@5比H0低7.71%；同时发现H1/J1后期码字使用塌缩。下一步值得做的是一个控制后期编码更新的小诊断，而不是直接认定交互有效或扩大模型。**\n\n'+protocol
report+='## 6. 最终测试结果\n\n四组按完整验证集预先选出的best，在同一53,194条测试样本、4,153名用户上评估。下表来自独立逐样本复算。\n\n'
report+=table(['模型','best R/T','单列表NDCG@5','HR@5','Recall@5','聚合NDCG@50','候选覆盖@50'],[
    [a,f"{results[a]['best_state']['epoch']} / {results[a]['best_state']['t_updates']}"]+
    [num(audits[a]['test_beam32']['metrics'][k]) for k in ['list/ndcg@5','list/hr@5','list/recall@5','aggregate/ndcg@50']]+
    [percent(audits[a]['test_beam32']['metrics']['aggregate/coverage@50'])] for a in arms])
report+='\n所有模型单份五件列表的完整输出率为100%，重复率和非法率为0。候选覆盖@K是每条样本实际不同候选数（最多K）除以K后取平均，不是全目录覆盖率。聚合@50的平均有效候选少于50，缺项按既定PAD规则处理。\n\n'
report+=table(['模型','K','聚合HR@K','Recall@K','NDCG@K','候选覆盖@K'],[
    [a,k]+[num(audits[a]['test_beam32']['metrics'][f'aggregate/{n}@{k}']) for n in ['hr','recall','ndcg']]+
    [percent(audits[a]['test_beam32']['metrics'][f'aggregate/coverage@{k}'])] for a in arms for k in [10,20,50]])
report+='\n### 配对差值与证据边界\n\n在同一用户内保留全部样本，按用户有放回抽样10,000次。下表区间对应样本平均差值，均为未校正的逐指标95%区间。\n\n'
report+=table(['比较','指标','绝对差值','相对差值','95%用户聚类区间'],[
    [arch+'1−'+arch+'0',metric,f"{p['delta']:+.6f}",f"{100*p['relative_delta']:+.2f}%",ci(p['confidence_interval'])]
    for arch in ['J','H'] for metric in ['list/ndcg@5','aggregate/ndcg@50'] for p in [pairs[arch]['test_beam32']['primary'][metric]]])
report+='\nH组两项区间都在零以下，在这两个固定模型和本次测试用户分布下，下降不只是点估计的微小波动。但这是单个训练seed，不能用用户抽样代替多seed复验。J1的best来自共同暖身、T=0；J1测试低于J0，说明这次带Mixer的训练轨迹没有选出优于暖身的状态，不能把该best视为已学会交互的模型。\n\n'
report+='### 用户平均与冷目标分组\n\n冷目标组指五件目标中至少有一件未出现在实际训练样本中；这不是只对冷商品计算指标。所有目标均已核对原始split和train_seen。\n\n'
report+=table(['模型','用户等权NDCG@5','含冷目标7,639条','全已见目标45,555条'],[
    [a,num(audits[a]['test_beam32']['user_average']['list/ndcg@5'])]+[num(audits[a]['test_beam32']['groups'][g]['metrics']['list/ndcg@5']) for g in ['cold_target','seen_targets']] for a in arms])
report+='\nH1在含冷目标组的点估计略高，全部已见目标组较低；这只是探索性分组结果，不构成冷启动收益结论。各指标、用户平均差值与区间详见[完整机器可读结果](SUMMARY.json)。\n\n'
report+='## 7. 验证轨迹与编码退化\n\n'
report+=table(['模型','best轮次','best验证NDCG@5','固定R35验证NDCG@5','best / R35聚合NDCG@50'],[
    [a,results[a]['best_state']['epoch'],num(valid[a+':selected_best']['metrics']['list/ndcg@5']),num(valid[a+':fixed_R35']['metrics']['list/ndcg@5']),num(valid[a+':selected_best']['metrics']['aggregate/ndcg@50'])+' / '+num(valid[a+':fixed_R35']['metrics']['aggregate/ndcg@50'])] for a in arms])
report+='\n![训练与验证曲线](final_analysis_20261027/curves/training_validation_curves.png)\n\nH1在R8、11、14、17、20超过同期H0，R20以后明显退化。这些点来自同一训练轨迹，不能当作五次独立实验。H1与H0各自best的验证差为+5.07%，不能用同轮R20的+21.27%替代。J1损失下降却指标恶化，也说明只看SID预测损失可能掩盖表示区分能力的损失。\n\n'
report+=table(['状态','不同SID数','碰撞商品比例','四槽使用码字数','Mixer修正/基础码本整体范数'],[
    [a+' '+role+' R'+str(results[a]['best_state']['epoch'] if role=='best' else 35),v['unique_saved_sids'],percent(v['collision_rate']),'/'.join(map(str,v['used_codes_per_slot'])),f"{v['mixer_delta_over_base_global_norm']:.3f}"]
    for a in ['H0','H1','J1'] for role in ['best','last'] for v in [geometry['arms'][a][role]]])
report+='\n碰撞商品比例是位于多个商品共享的SID桶中的商品占比，不是这些商品全部无法推荐。CPU FP32和独立FP64距离重算都与checkpoint SID一致，排除了这些端点的缓存未刷新。端点只能确认晚期退化，不能定位第一次塌缩的T更新，也不能单凭它判定学习率就是根因。\n\n'
report+='## 8. 固定验证子集的beam敏感性\n\n所有模型沿用冻结best，在事先固定的1,024条验证样本上比较beam32/256；均最多聚合32份列表，不重新选模。\n\n'
report+=table(['模型','beam','单列表NDCG@5','聚合NDCG@50','候选覆盖@50','推理秒'],[
    [a,b,num(audits[a][v]['metrics']['list/ndcg@5']),num(audits[a][v]['metrics']['aggregate/ndcg@50']),percent(audits[a][v]['metrics']['aggregate/coverage@50']),f"{summaries[a][v]['seconds']:.2f}"]
    for a in arms for b in [32,256] for v in [f'validation_subset_beam{b}']])
report+='\nH1相对H0的子集NDCG@5点估计在两种beam下都为正，但两个用户配对区间都跨零；不能据此推翻完整测试下降。beam256使H0/H1自身的单列表NDCG@5都下降，也没有稳定提高聚合指标。扩大搜索更充分地优化模型分数，不保证分数排序更符合真实商品相关性。子集较小且没有做同模型跨beam的配对区间，不作显著性结论。\n\n'
report+='## 9. 下一步最值得验证的问题\n\n**从H1 R20保留已学到的交互表示，停止或减弱后续编码更新，能否避免塌缩并保持推荐质量？**\n\n建议同一个完整R20状态分叉到R26：A保持原设置，B只把Mixer学习率1e-4降为3e-5，C固定基础码本、Mixer和SID，只继续推荐器。若只做一支干预，先C。原A轨迹只有在恢复一致性通过后才可复用。3e-5是待验证的诊断值，不是论文给出的最优值。\n\n保留优化器状态、R的35轮调度和数据阶段编号；A/B继续原400T噪声时钟，C明确停止T。R26完整验证NDCG@5为预先固定主判据，R23看轨迹；逐T边界记录有效码字/SID数、碰撞率、编码变化和残差范数。不要同时改学习率、损失、噪声和交互结构。\n\n这首先验证能否止跌；超过旧H1峰值、跨seed稳定超过H0是另外两步。H1测试best仍低于H0，后续参数只能按验证选择，不能把当前测试变成调参依据；正式新结论需要公平给H0调参机会及额外seed。方案与ETEGRec、BLOGER、DACT、DIGER的可借鉴部分和区别见[训练设置调研](../../docs/research/onerec_listwise/H1_TRAINING_STABILITY_REVIEW_20261028.md)。本轮没有执行上述新增分支。\n\n'
report+='## 10. 资源、恢复和验收\n\n'
report+=f"累计13个Job分配148,316 GPU秒，即 **{release['allocated_gpu_hours']:.6f} GPU小时**，低于75小时授权上限。含内容编码、短验收、共享GPU测速、暖身、训练、评估及失败H1 r1的522秒；所有GPU原生释放事件均已核实。按已核实1点/GPU小时估算 **{release['estimated_points']:.2f}点，未核实结算**。\n\n"
report+='正式四分支在2张H100上运行4个独立进程，GPU0=J0/H0，GPU1=J1/H1，无DDP；共享暖身2个单卡Job，最终四组评估各一个单卡Job。H1首次评估因NodeNotReady失败，r2仅改任务和产物路径，模型、参数和数据不变，失败证据及0字节输出保留。\n\n'
report+=table(['模型','完整测试推理秒','完整测试分钟'],[[a,f"{summaries[a]['test_beam32']['seconds']:.2f}",f"{summaries[a]['test_beam32']['seconds']/60:.2f}"] for a in arms])
report+='\n上述是实际批量离线评估耗时，不是单请求线上延迟，不包含整个Job分配时间。\n\n'
report+=f'- 远端来源：`{G}`。\n- 本地归档：`{D.as_posix()}`。\n- 独立恢复：`{R.as_posix()}`。\n- 两份暖身、四组initial/best/last、42份验证、12份最终评估、运行日志、共同资产和三份源码bundle均有SHA绑定；模型checkpoint包含优化器、调度、RNG及SID。代表J1/H1严格CPU恢复与无目标前向通过。\n- 12份最终预测和8个best/R35角色完整验证已独立复算；6份用户配对bootstrap已完成。已有本地数据只核对SHA，没有重新迁移。\n\n'
report+='验收入口：[资产与恢复复核](formal_execution_20261027/FINAL_LOCAL_ASSET_VERIFICATION_20261028.json)、[四best冻结](formal_execution_20261027/FOUR_BEST_FREEZE_20261028.json)、[GPU释放与费用](formal_execution_20261027/ALL_GPU_RELEASE_ACCOUNTING_20261028.json)、[交接说明](HANDOFF.md)。\n'
with (E/'REPORT.md').open('x',encoding='utf-8') as f:f.write(report)

handoff=f'''# 09阶段交接

2026-10-28。本轮四组seed2024实验及最终交付已验收。先读[报告](REPORT.md)、[结构化结果](SUMMARY.json)，不要再启动既有提交脚本。

- H0测试NDCG@5=0.008607，H1=0.007944；H1相对-7.71%，单seed用户配对区间在零以下。J1的best是共享R5/0T，不代表训练后的Mixer收益。
- H1验证best为R20，晚期存在真实码字使用塌缩；J1 R35全目录仅1种SID。当前优先问题是编码更新稳定性；没有追加调参。
- 四组全部35R/400T。两份5R暖身各计算一次，总130R/90090R更新/1600T更新；42次完整验证，4次完整测试，8次固定验证子集评估。
- 冻结科学源码main=1c423662d9a25a926328ee6466c30c0f2857b4f6，DiffGRM=98b834cc5ce4428d1a95b3e4f9651833b5fbd5cd，GenRec=b6c8b41d1d20a812590dae11408b69b1ed2a8313。报告/分析收尾源码与训练源码分别保留。
- 实际配置为preparation_20261027/runtime_efficiency_r2.json：微批256×4、eval batch32；base.json的16/4是早期配置，不是正式实测。
- 远端：{G}
- 本地归档：{D.as_posix()}
- 独立恢复：{R.as_posix()}
- 模型在runs/warmup_J、warmup_H、J0、J1、H0、H1；每份manifest绑定initial/best/last、状态、验证和日志。共享数据沿用outputs/kuairec_listwise_v1/run1，生成资产在恢复根assets/。
- 最终预测在sidecars/evaluations/evaluations/。H1正式采用H1_test_beam32_r2、H1_validation_subset_beam32_r2、H1_validation_subset_beam256_r2；原H1 r1为节点失败证据，不能用于分析。
- 操作侧包在sidecars/operations；四run包不能单独替代源码/共同资产/侧包。完整索引见formal_execution_20261027/FINAL_LOCAL_ASSET_VERIFICATION_20261028.json。
- 已下载归档完整SHA和成员SHA均复核；代表J1/H1已实际CPU加载模型、alignment、优化器/调度/RNG并前向。只用推理权重不能称为完整续训恢复。
- 原训练resume强校验源码、配置和阶段时钟。后续R20诊断须建立新分叉记录，保留原优化器矩和R调度，不伪改旧binding，不覆盖原目录。
- 所有GPU已释放，累计41.198889 GPU小时。按1点/GPU小时估41.20点，未核实结算。CPU Notebook是临时文件入口，启停状态不影响共享盘资产。

下一步仅为建议：R20原设置 / 降Mixer LR / 固定编码三支到R26。以验证NDCG@5、SID变化、碰撞率和残差范数判断；没有授权在本Goal追加此实验。详见[训练设置调研](../../docs/research/onerec_listwise/H1_TRAINING_STABILITY_REVIEW_20261028.md)。
'''
with (E/'HANDOFF.md').open('x',encoding='utf-8') as f:f.write(handoff)
print(json.dumps(dict(status='written',files=['REPORT.md','SUMMARY.json','HANDOFF.md'],h_test_delta=pairs['H']['test_beam32']['primary']['list/ndcg@5']['relative_delta'])),flush=True)
