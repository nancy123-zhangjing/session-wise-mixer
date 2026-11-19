"""Build the Chinese scientific report and standalone figures from passed audits.

Delivery/resource closure remains an explicit separate acceptance requirement.
"""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE))
from analyze_d import read,write,bind,sha,INPUTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('analysis','temporal','process','operational-audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    bind(__file__)
    analysis=read(a.analysis/'ANALYSIS.json');temporal=read(a.temporal/'TEMPORAL.json');process=read(a.process/'PROCESS.json')
    if any(x['status']!='passed' for x in (analysis,temporal,process)):raise ValueError('Passed independent science audits required')
    operational=read(a.operational_audit)
    if (operational['status']!='passed_async_operational_delivery_audit'
            or len(operational['readonly_tasks'])!=11
            or not all(x['readonly_state_exact'] for x in operational['readonly_tasks'])):
        raise ValueError('Passed complete operational continuation audit required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'font.size':10})
    def figure(name,fig):
        fig.tight_layout()
        for suffix in ('png','svg','pdf'):fig.savefig(out/(name+'.'+suffix),bbox_inches='tight')
        plt.close(fig)
    olda=read(ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis/A/analysis_report.json',
        '66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937')
    oldc=read(ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis/C/analysis_report.json',
        'a4f1315e1bf4b7dcb71d14bdcbbfdb48609a732428919480d321ad4711789955')
    if olda['status']!='passed' or oldc['status']!='passed':raise ValueError('Historical branch audit not passed')
    fig,ax=plt.subplots(figsize=(8,4))
    for label,points in [('D: spread 200T',analysis['points']),('A: freeze 200T',olda['models']),('C: H0 freeze 200T',oldc['models'])]:
        rows=sorted((x for x in points.values() if x['split']=='validation'),key=lambda x:x['epoch'])
        ax.plot([x['epoch'] for x in rows],[x['metrics']['list/ndcg@5'] for x in rows],marker='o',label=label)
    oldcurve=[]
    for epoch in (8,11,14,17,20,23,26,29,32,35):
        value=read(ROOT/f'outputs/stage09_restore/goal_20261027_r1/runs/H1/validation_R{epoch:02d}/summary.json')
        if value['r_updates']!=epoch*693 or value['t_updates']!=(epoch-5)//3*40:raise ValueError('Historical H1 curve clocks differ')
        oldcurve.append(value['metrics']['list/ndcg@5'])
    ax.plot((8,11,14,17,20,23,26,29,32,35),oldcurve,'--',color='gray',label='Original H1: 400T')
    ax.set(xlabel='Recommender epochs',ylabel='Validation NDCG@5');ax.grid(alpha=.2);ax.legend(fontsize=8)
    figure('validation_curve',fig)
    key='list/ndcg@5'
    fig,ax=plt.subplots(figsize=(8,4))
    cycles=range(1,11); deltas=[];recovered=[]
    for cycle in cycles:
        point={r['stage']:r[key] for r in process['curves'] if r['cycle']==cycle}
        deltas.append(point['after_20T']-point['before_T']);recovered.append(point['after_first_R']-point['before_T'])
    ax.plot(cycles,deltas,'o-',label='After 20T minus before T')
    ax.plot(cycles,recovered,'s-',label='After next R minus before T');ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='T block',ylabel='Fixed-subset NDCG@5 change',xticks=list(cycles));ax.legend(fontsize=8);ax.grid(alpha=.2)
    figure('T_block_response',fig)
    catalogs=sorted(process['catalogs'].values(),key=lambda x:x['t_updates'])
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
    axes[0].plot([x['t_updates'] for x in catalogs],[x['unique_sid_count'] for x in catalogs],'o-');axes[0].set(ylabel='Distinct SIDs')
    axes[1].plot([x['t_updates'] for x in catalogs],[x['collision_item_fraction'] for x in catalogs],'o-',label='Collision item fraction')
    axes[1].plot([x['t_updates'] for x in catalogs[1:]],[x['previous_change']['sid_item_change_fraction'] for x in catalogs[1:]],'s-',label='SID changed since prior block')
    axes[1].set(ylabel='Fraction');axes[1].legend(fontsize=8)
    axes[2].plot([x['t_updates'] for x in catalogs],[x['mixer_delta_over_base_global_norm'] for x in catalogs],'o-')
    axes[2].set(xlabel='Tokenizer updates',ylabel='Mixer correction / base norm')
    for ax in axes:ax.grid(alpha=.2)
    figure('tokenizer_stability',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for name,values in temporal['models'].items():axes[0].plot([0,1],[values[s]['metrics'][key] for s in ('early','late')],'o-',label=name)
    historical=temporal['old_H0_best_R32_historical_reference']['segments']
    axes[0].plot([0,1],[historical[s]['metrics'][key] for s in ('early','late')],'--',color='gray',label='H0 best R32 (historical)')
    axes[0].set(xticks=[0,1],xticklabels=['Early','Late'],ylabel='Validation NDCG@5');axes[0].legend(fontsize=7)
    for i,s in enumerate(('early','late')):
        v=temporal['D_minus_A'][s]['list/ndcg@5/overall'];lo,hi=v['ci'];delta=v['delta']
        axes[1].errorbar(i,delta,yerr=[[delta-lo],[hi-delta]],fmt='o',capsize=4)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(xticks=[0,1],xticklabels=['Early','Late'],ylabel='D - A; paired-user 95% interval')
    for ax in axes:ax.grid(alpha=.2)
    figure('temporal_generalization',fig)
    primary=analysis['comparisons']['fixed_test'];v=primary[key];low,high=v['confidence_interval']
    if low>0: verdict='本次固定终点的D高于冻结对照A，用户重采样区间位于0以上。'
    elif high<0:verdict='本次固定终点的D低于冻结对照A，用户重采样区间位于0以下。'
    else:verdict='本次固定终点的配对区间包含0，现有结果不足以支持明确的样本层面差异。'
    table=['| 指标 | A：冻结200T | D：分散200T | D−A | 用户聚类95%区间 |','|---|---:|---:|---:|---|']
    for metric in ('list/ndcg@5','list/hr@5','list/recall@5'):
        m=primary[metric];ci=m['confidence_interval']
        table.append(f'| {metric} | {m["baseline"]:.8f} | {m["treatment"]:.8f} | {m["delta"]:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] |')
    best=analysis['selected_epoch'];bestcmp=analysis['comparisons']['best_test'][key]
    aggregate_rows=['| 聚合指标 | A R35 | D R35 | D−A | 用户聚类95%区间 |','|---|---:|---:|---:|---|']
    for cutoff in (10,20,50):
        for name in ('ndcg','hr','recall'):
            metric=f'aggregate/{name}@{cutoff}';m=primary[metric];ci=m['confidence_interval']
            aggregate_rows.append(f'| {metric} | {m["baseline"]:.8f} | {m["treatment"]:.8f} | {m["delta"]:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] |')
    coverage_rows=['| 候选指标（描述性均值） | A R35 | D R35 |','|---|---:|---:|']
    for metric in ('aggregate/coverage@10','aggregate/coverage@20','aggregate/coverage@50','aggregate/candidate_count','aggregate/lists_used'):
        coverage_rows.append(f'| {metric} | {olda["models"]["A_R35_test"]["metrics"][metric]:.6f} | {analysis["points"]["D_R35_test"]["metrics"][metric]:.6f} |')
    best_rows=['| 测试指标 | A best R32/T200 | D best R23/T120 | D−A | 用户聚类95%区间 |','|---|---:|---:|---:|---|']
    for metric in ('list/ndcg@5','list/hr@5','list/recall@5'):
        m=analysis['comparisons']['best_test'][metric];ci=m['confidence_interval']
        best_rows.append(f'| {metric} | {m["baseline"]:.8f} | {m["treatment"]:.8f} | {m["delta"]:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] |')
    stability_rows=['| T累计更新 | 该20T块SID改变商品比例 | 碰撞商品比例 | Mixer修正/基础码本范数 |','|---|---:|---:|---:|']
    for c in catalogs[-4:]:
        stability_rows.append(f'| {c["t_updates"]} | {c["previous_change"]["sid_item_change_fraction"]:.2%} | {c["collision_item_fraction"]:.2%} | {c["mixer_delta_over_base_global_norm"]:.4f} |')
    response_rows=['| T块 | T前NDCG@5 | 20T后 | 随后首R后 | 随后三R后 |','|---|---:|---:|---:|---:|']
    for cycle in (9,10):
        values={r['stage']:r[key] for r in process['curves'] if r['cycle']==cycle}
        response_rows.append(f'| {cycle} | {values["before_T"]:.8f} | {values["after_20T"]:.8f} | {values["after_first_R"]:.8f} | {values["cycle_end"]:.8f} |')
    final_catalog=catalogs[-1]
    slot_rows=['| 最终码槽 | 已使用码字/总数 | 有效码字数 exp(entropy) | 归一化熵 |','|---|---:|---:|---:|']
    for row in final_catalog['per_slot']:
        slot_rows.append(f'| {row["slot"]} | {row["used_code_count"]}/{final_catalog["codebook_size"]} | {row["effective_code_count"]:.2f} | {row["normalized_entropy"]:.4f} |')
    journal_path=next(Path(path) for path in analysis['input_sha256'] if Path(path).name=='training.jsonl')
    bind(journal_path,analysis['input_sha256'][str(journal_path)])
    late_lr={}
    with journal_path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['phase']=='R' and row['epoch'] in (33,34,35) and row['update']==1:
                late_lr[row['epoch']]=row['learning_rates_used']['R']
    if set(late_lr)!={33,34,35}:raise ValueError('Late R learning-rate evidence missing')
    time_rows=['| 验证时段 | A NDCG@5 | D NDCG@5 | D−A及95%区间 |','|---|---:|---:|---|']
    for seg,cn in (('early','前段：38644条'),('late','后段：33013条')):
        values=temporal['models'];m=temporal['D_minus_A'][seg]['list/ndcg@5/overall']
        time_rows.append(f'| {cn} | {values["A_Freeze_R35"][seg]["metrics"][key]:.8f} | {values["D_Spread_R35"][seg]["metrics"][key]:.8f} | {m["delta"]:+.8f} [{m["ci"][0]:+.8f}, {m["ci"][1]:+.8f}] |')
    shift=temporal['D_minus_A']['late_minus_early']['list/ndcg@5/overall']
    relative_time={seg:temporal['models']['D_Spread_R35'][seg]['metrics'][key]/temporal['models']['A_Freeze_R35'][seg]['metrics'][key]-1 for seg in ('early','late')}
    lines=['# H1低T预算分散更新实验报告','',verdict,'',
        '本次新增唯一D分支；负结果也按原协议交付，没有据测试结果追加训练。下述区间只反映固定模型在用户样本上的不确定性，不包含训练seed、选模和事后设计的不确定性。','',
        '## 实验回答的问题','',
        'A与D均到R35/T200。A在前期完成200T后冻结，D把相同有序T样本分成10个20T块，分散到35R。R样本顺序、原35R学习率日程、400T噪声退火时钟、模型、损失和解码规则保持原绑定。后期T仍使用同一训练集，其接收梯度时推荐器已经处于更后期的状态。','',
        '因此本次主要考察相同更新预算下的时间安排。它没有实施MEDA式随机重置，也不能单凭本实验认定存在one-epoch overfitting。与原H1/T400的曲线比较同时涉及T总量与安排变化，应作辅助观察。','',
        '## 执行中断与恢复','',
        '正式作业曾中断，后续从R24第400步的完整checkpoint继续，没有从头重训，也没有把短测更新并入正式轨迹。恢复后将训练与阶段checkpoint上的只读验证放到同一张H100并行执行；模型、损失和解码源码仍使用原冻结版本，科学更新序列和预算口径保持原协议。','',
        f'最终独立运维审计核对了{len(operational["readonly_tasks"])}个恢复后只读任务：指定checkpoint、完整状态指纹和完成记录绑定一致，验证前后状态相同；首30次真实续跑也保留了原日志前缀及CUDA重放比对证据。运维包SHA为{operational["package_sha256"]}，详情见[独立运维审计](../ASYNC_DELIVERY_AUDIT.json)。这些证据用于核对执行正确性，不代表模型效果提升。','',
        '## 固定R35主测试','',*table,'',
        f'主指标的相对差为{primary[key]["relative_delta"]:.2%}。区间来自既定2000次用户聚类配对bootstrap，估计量仍为样本均值；它不包含跨seed训练不确定性。','',
        'HR@5表示五件列表至少命中一个目标；Recall@5按五件目标计算。聚合@10/20/50来自最多32份候选列表按分数和位置去重汇总，不表示单次生成一份50件列表。','',*aggregate_rows,'',*coverage_rows,'',
        '这里的coverage@K是去重候选对K个位置的平均填充比例，不是整个商品目录的覆盖率；candidate_count是最多50位中的不同有效商品数，lists_used是实际用于聚合的完整候选列表数。覆盖表仅作描述，不对其差值附加显著性结论。','',
        '## 共同网格best补充','',
        f'D从R20/23/26/29/32/35按验证NDCG@5选出R{best}/T{(best-5)//3*20}；A已锁best为R32/T200。补充测试NDCG@5差值为{bestcmp["delta"]:+.8f}，区间{bestcmp["confidence_interval"]}。补充best的R/T预算可能不同，等预算结论依据固定R35。测试只执行固定R35和已锁best，重合时复用。','',
        *best_rows,'',
        '![验证曲线](validation_curve.png)','',
        '## Tokenizer扰动与推荐器恢复','',
        f'固定8192条诊断集共31个状态：20份新子集推理、10份新完整验证的子集提取、1份旧R5提取复用。每个T块记录更新前、20T后、随后第一个R后及周期末；T后相对更新前的NDCG变化范围为[{min(deltas):+.8f}, {max(deltas):+.8f}]，随后第一个R后相对更新前范围为[{min(recovered):+.8f}, {max(recovered):+.8f}]。','',
        '曲线描述短期扰动和恢复，不能直接把一次下降等同于编码器退化。SID变化还要结合碰撞、码字使用、有效码本位移和Mixer幅度判断。R阶段Tokenizer状态相同已逐checkpoint核验，因此T前码本可复用上一T后的认证快照；这没有增加模型推理。','',
        *stability_rows,'',*response_rows,'',
        '上表效果来自同一8192条固定诊断集，与完整验证和测试的样本集合不同。第9块之后首个R已恢复到T前水平以上；第10块之后三R结束仍低于T前。不同块的表现不能概括为每次T更新都必然有害。','',
        f'全目录不同SID数从{catalogs[0]["unique_sid_count"]}增至{final_catalog["unique_sid_count"]}，碰撞商品比例从{catalogs[0]["collision_item_fraction"]:.2%}降至{final_catalog["collision_item_fraction"]:.2%}。与此同时，最后20T改变了{final_catalog["previous_change"]["sid_item_change_fraction"]:.2%}商品的SID，最终Mixer修正全局范数约为当时基础码本范数的{final_catalog["mixer_delta_over_base_global_norm"]:.2f}倍。碰撞减少与推荐性能下降在本次运行中同时出现，不能把码的分散程度直接当作推荐效果。','',
        *slot_rows,'',
        f'已验证的日程事实：T的基础码本/Mixer学习率仍为1e-5/1e-4；R33、R34、R35首步学习率依次为{late_lr[33]:.3e}、{late_lr[34]:.3e}、{late_lr[35]:.3e}，沿用原35R余弦衰减。','',
        '解释性猜想：后期SID大幅移动，而R的学习率已经较低，可能使推荐器来不及适应新的编码。过程曲线与这一解释相符，但本实验没有单独操纵学习率、SID移动或收敛速度，不能据此证明该机制，也不能据此确认Tokenizer已经发生过拟合。','',
        '![T块响应](T_block_response.png)','', '![编码稳定性](tokenizer_stability.png)','',
        '## 时间泛化','',*time_rows,'',
        f'后段减前段的D−A差值变化为{shift["delta"]:+.8f}，联合用户抽样95%区间为{shift["ci"]}。这表示绝对负差缩小；D在两个时段都低于A，两个区间均低于0。点估计的相对差前段为{relative_time["early"]:.2%}、后段为{relative_time["late"]:.2%}，因此不能将绝对负差缩小解读为D获得了更好的时间泛化。','',
        '切点保持1598745441188500微秒，按首目标时间划分；冷暖、head/mid/tail均沿用训练期定义。该切片来自验证集，属于过程诊断，不能替代主测试结论。','',
        '旧H0 best R32/T360单列为历史参照，其R/T预算不同。目标组成与组内表现分解只作描述，不能由组内项推断唯一原因；完整分组、推荐集中度及分解闭合证据见[时间分析](../final_temporal/TEMPORAL.json)。静态训练与时间切片结果也不能证明模型适应了线上新到达数据。','',
        '![时间泛化](temporal_generalization.png)','',
        '## 证据与交付状态','',
        '独立指标、逐更新batch与时钟、选模、31个过程状态、SID快照及时间分解的审计已通过，来源SHA保存在各JSON。归档下载、独立解压、真实CPU冷恢复、GPU释放和临时CPU关闭必须另有最终验收收据；本报告生成不代表Goal完成。','',
        '本次仅一个seed，且调度设计受到此前H1结果启发。用户bootstrap不覆盖跨seed稳定性或这种事后选择。结果支持到哪一步，应以固定终点、区间和过程证据共同判断。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    for path,digest in INPUTS.items():
        if sha(path)!=digest:raise ValueError('Report input changed')
    write(out/'REPORT_BINDING.json',dict(status='generated_visual_review_pending',input_sha256=INPUTS,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},goal_completion=False))

if __name__=='__main__':main()
