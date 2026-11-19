"""Render E report from completed independent audits; no training/inference."""
import argparse
import csv
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from analyze_e import read,write,sha,INPUTS

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    analysis=read(HERE/'final_analysis/ANALYSIS.json')
    process=read(HERE/'final_process/PROCESS.json')
    temporal=read(HERE/'final_temporal/TEMPORAL.json')
    restore=read(HERE/'COLD_RESTORE.json')
    operation=read(HERE/'OPERATIONAL_AUDIT.json')
    budget=read(HERE/'GPU_RELEASE_AND_BUDGET.json')
    if any(x['status']!='passed' for x in (analysis,process,temporal,restore)):
        raise ValueError('Independent science/restore audits required')
    if operation['status']!='passed_async_operational_delivery_audit' or len(operation['readonly_tasks'])!=20:
        raise ValueError('All twenty readonly tasks required')
    d=read(ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/final_analysis/ANALYSIS.json','5cc04a637140d94d198c1a2e0132cf09b83d03cac152e82ac9e5da675add44e8')
    olda=read(ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis/A/analysis_report.json','66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937')
    oldc=read(ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis/C/analysis_report.json','a4f1315e1bf4b7dcb71d14bdcbbfdb48609a732428919480d321ad4711789955')
    if olda['status']!='passed' or oldc['status']!='passed':raise ValueError('Accepted historical reports required')
    e=analysis['points'];best=analysis['selected_epoch']
    metric='list/ndcg@5'
    fixed={'E':e['E_R35_test']['metrics'],'D':d['points']['D_R35_test']['metrics'],'A':olda['models']['A_R35_test']['metrics']}
    selected={'E':e[f'E_R{best:02d}_test']['metrics'],'D':d['points']['D_R23_test']['metrics'],'A':olda['models']['A_R32_test']['metrics']}
    primary={b:analysis['comparisons'][f'E_minus_{b}_fixed_test'][metric] for b in ('D','A')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'font.size':10})
    def save(name,fig):
        fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,points in [('E: 40T + 6R',e),('D: 20T + 3R',d['points']),('A: H1 freeze at R20',olda['models']),('C: H0 freeze at R20',oldc['models'])]:
        rows=sorted((v for v in points.values() if v['split']=='validation'),key=lambda x:x['epoch'])
        ax.plot([v['epoch'] for v in rows],[v['metrics'][metric] for v in rows],marker='o',label=label)
    original=[]
    for epoch in (8,11,14,17,20,23,26,29,32,35):
        v=read(ROOT/f'outputs/stage09_restore/goal_20261027_r1/runs/H1/validation_R{epoch:02d}/summary.json')
        if v['r_updates']!=epoch*693 or v['t_updates']!=(epoch-5)//3*40:raise ValueError('Original H1 clock mismatch')
        original.append(v['metrics'][metric])
    ax.plot((8,11,14,17,20,23,26,29,32,35),original,'--',color='gray',label='Original H1: 400T')
    ax.set(xlabel='Recommender epochs',ylabel='Full validation NDCG@5');ax.legend(fontsize=8);ax.grid(alpha=.25)
    save('validation_curve',fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for cycle in range(1,6):
        rows=[v for v in process['curves'] if v['cycle']==cycle]
        ax.plot(range(5),[v[metric] for v in rows],marker='o',label=f'Cycle {cycle}, T={cycle*40}')
    ax.set_xticks(range(5),['Before T','After 40T','After 1R','After 3R','After 6R'])
    ax.set_ylabel('Fixed 8192-sample validation NDCG@5');ax.grid(alpha=.25);ax.legend(fontsize=8)
    save('cycle_recovery',fig)
    fig,ax=plt.subplots(figsize=(8,4))
    pairs=process['paired_40T_endpoints']
    for branch in ('E','D'):
        ax.plot([v['cumulative_T'] for v in pairs],[100*v[f'{branch}_sid_item_change_fraction'] for v in pairs],marker='o',label=branch)
    ax.set(xlabel='Cumulative T updates',ylabel='Items with changed SID over 40T (%)');ax.grid(alpha=.25);ax.legend()
    save('sid_change_40T',fig)
    fmt=lambda x:f'{x:.6f}'
    rows=[]
    for name,epoch,t in [('E',35,200),('D',35,200),('A',35,200)]:
        m=fixed[name];rows.append([name,epoch,t,fmt(m[metric]),fmt(m['list/hr@5']),fmt(m['list/recall@5'])])
    paragraphs=['# E：40T＋6R 实验结果与验收',
        '## 1. 实验改变与对照',
        '从原H1完整R5/T0状态出发，E执行5次“40T＋6R”，最终R35/T200。D是10次“20T＋3R”，总R/T相同；A在原H1的R20/T200冻结Tokenizer，推荐器继续到R35。E只改变交替安排，沿用D单卡实现、全局batch1024（256×4）、seed2024、原样本/阶段种子映射、双优化器、35R学习率时钟及400T噪声时钟。',
        '最初八卡尝试没有进入正式训练；本报告E结果来自唯一单H100正式分支。因此不再将“八卡与单卡执行后端不同”作为本次E对D的混杂因素。但40T块大小和6R间隔同时改变，仍不能把差异唯一归因于推荐器适应时间。',
        table(['组别','共享5R后的安排','终点','对照用途'],[
            ['原H1','10×(40T＋3R)','R35/T400','原始持续联合训练'],
            ['A','5×(40T＋3R)，随后冻结T继续15R','R35/T200','前期完成T、后期保留编码'],
            ['D','10×(20T＋3R)','R35/T200','较小T块分散更新'],
            ['E','5×(40T＋6R)','R35/T200','40T块与更长连续R窗口'],
            ['C','H0无Mixer，R20/T200后冻结T','R35/T200','与A同冻结点的交互对照']]),
        '## 2. 固定R35/T200：主比较',
        table(['模型','R','T','测试NDCG@5','HR@5','Recall@5'],rows)]
    for b,c in primary.items():
        ci=c['confidence_interval']
        paragraphs.append(f"E相对{b}的固定终点NDCG@5变化为{100*c['relative_delta']:+.2f}%，绝对差{c['delta']:+.6f}，用户聚类配对bootstrap 95%区间[{ci[0]:+.6f}, {ci[1]:+.6f}]。")
    paragraphs+=['## 3. 按共同验证网格选择的best（次要结果）',
        '只从R20/23/26/29/32/35按验证NDCG@5选模，平局取较早；测试不参与选模。固定R35与best共两次测试，未因结果较差追加模型。',
        table(['模型','best R','T','best测试NDCG@5'],[[n,r,t,fmt(selected[n][metric])] for n,r,t in [('E',best,(1+(best-6)//6)*40),('D',23,120),('A',32,200)]]),
        'best比较同时存在R/T进度差异，只反映各自共同网格选模后的表现，不能替代固定R35/T200主比较。',
        '![完整验证曲线](validation_curve.png)',
        '图中各轮次的T预算不始终相同；E与D在R11/17/23/29/35的累计T相同，分别为40/80/120/160/200。',
        table(['相同R','相同累计T','D完整验证NDCG@5','E完整验证NDCG@5','E−D'],[
            [r,t,fmt(d['points'][f'D_R{r:02d}_validation']['metrics'][metric]),fmt(e[f'E_R{r:02d}_validation']['metrics'][metric]),f"{e[f'E_R{r:02d}_validation']['metrics'][metric]-d['points'][f'D_R{r:02d}_validation']['metrics'][metric]:+.6f}"] for r,t in [(11,40),(17,80),(23,120),(29,160),(35,200)]]),
        '相同更新总量并不意味着相同优化路径：T发生时推荐器的参数已经不同，R发生时商品SID也不同；A最后一次T之后有15R保持编码不变，E为6R，D为3R。因此“R总量更多”不等于“对最终同一套SID的有效适应更多”。这是调度事实，不是已经验证的唯一掉分原因。',
        table(['历史状态（测试）','R/T','NDCG@5','比较边界'],[
            ['原H1 best', '20/200',fmt(olda['models']['A_R20_test']['metrics'][metric]),'同T，不同R；原全验证网格选模'],
            ['H1 Freeze R26','26/200',fmt(olda['models']['A_R26_test']['metrics'][metric]),'与A同一冻结路线的较早节点'],
            ['旧H0 best','32/360',fmt(olda['models']['B_R32_test']['metrics'][metric]),'无Mixer、R/T不同'],
            ['C固定终点','35/200',fmt(oldc['models']['C_R35_test']['metrics'][metric]),'无Mixer，R20冻结']]),
        '## 4. 40T之后，1R、3R和6R如何变化',
        '以下使用固定8192条诊断样本；3R/6R从对应完整验证预测中提取，没有重复解码。完整验证曲线与此子集曲线不能混用。',
        '![周期内恢复](cycle_recovery.png)']
    recovery=[]
    for cycle in range(1,6):
        v=[x for x in process['curves'] if x['cycle']==cycle]
        recovery.append([cycle,cycle*40,*[fmt(x[metric]) for x in v],f"{v[4][metric]-v[3][metric]:+.6f}"])
    paragraphs.append(table(['周期','累计T','T前','40T后','1R后','3R后','6R后','6R减3R'],recovery))
    improved=sum(float(r[-1])>0 for r in recovery)
    paragraphs.append(f'五个周期中，3R到6R的诊断NDCG@5有{improved}次上升、{5-improved}次没有上升。这是保存状态的描述性结果，不代表多给3R总能恢复，也不能独立证明Tokenizer退化或过拟合。')
    paragraphs.append('第3R至第6R之间没有T更新，SID保持不变，因此这些区段的下降不能直接归因于期间SID再次刷新。最后周期从40T后到首R后继续下降，6R结束仍未恢复到T前；推荐器继续优化也不保证推荐指标单调恢复。')
    paragraphs+=['## 5. SID变化：按相同累计40T端点比较','![每40T的SID变化](sid_change_40T.png)']
    cat=list(process['catalogs'].values())[1:]
    paragraphs.append(table(['累计T','E变化商品%','D变化商品%','E碰撞商品%','E四槽使用数','E Mixer相对幅度'],[
        [v['cumulative_T'],f"{100*v['E_sid_item_change_fraction']:.2f}",f"{100*v['D_sid_item_change_fraction']:.2f}",f"{100*c['collision_item_fraction']:.2f}",'/'.join(str(z['used_code_count']) for z in c['per_slot']),f"{c['mixer_delta_over_base_global_norm']:.6f}"] for v,c in zip(pairs,cat)]))
    paragraphs.append('D的数值由两个实际端点快照直接比较，未相加两个20T变化率。SID变化率高不等同于编码质量下降；还需结合碰撞、码字使用、码本位移与推荐指标。完整槽位分布、位移和训练损失/梯度/学习率保留于PROCESS.json与training.jsonl。')
    paragraphs+=['## 6. 时间泛化（固定R35验证集）','沿用原时间切点1598745441188500微秒及训练频次五类；早期38644、后期33013样本。A/D原统计与数组直接复用，只处理E新预测。此为固定数据切片，不是在线新数据学习。']
    display={'A_Freeze_R35':'A：冻结','D_Spread_R35':'D：20T＋3R','E_Block40_R35':'E：40T＋6R'}
    paragraphs.append(table(['模型','早期NDCG@5','后期NDCG@5','后期减早期'],[[display.get(n,n),fmt(m['early']['metrics'][metric]),fmt(m['late']['metrics'][metric]),f"{m['late']['metrics'][metric]-m['early']['metrics'][metric]:+.6f}"] for n,m in temporal['models'].items()]))
    tkey='list/ndcg@5/overall'
    for name,c in temporal['comparisons'].items():
        z=c['late_minus_early'][tkey]
        label=name.replace('E_minus_A','E−A').replace('E_minus_D','E−D')
        paragraphs.append(f"{label}差值的后期减早期变化为{z['delta']:+.6f}，95%区间[{z['ci'][0]:+.6f}, {z['ci'][1]:+.6f}]。频次占比变化和类内效率变化的描述性分解见TEMPORAL.json；不能从分解直接断言唯一原因。")
    paragraphs.append('E相对A在前后两段均更低，后段绝对负差缩小不能解释为获得更好的时间泛化。E相对D的点估计前段略高、后段更低，但上述联合时间差值区间跨零，尚不能认定相对时间衰减稳定存在。')
    paragraphs+=['## 7. 评价口径、局限与验收','所有新完整预测经过独立CPU复算，核验样本、标签、用户、冷启动分组及指标。保持FP32、TF32关闭、beam32、batch32、原映射/碰撞消歧/去重/PAD定义。2000次用户聚类配对bootstrap，seed2024、样本均值口径；单训练seed和多项比较仍限制结论，用户bootstrap不等于跨seed稳定性。',
        f"已核验全部R/T样本顺序、R24255/T200计数及原双时钟；20个只读任务完成且状态指纹一致；{len(restore['all_boundary_checkpoint_fields_and_clocks_verified'])}个边界完整状态核验，关键状态从独立解压代码真实CPU恢复。CPU恢复未应用CUDA RNG，也未新增GPU下一步恢复测试；复用D既有实现依据，原CUDA RNG字节保留。",
        f"全部E GPU已释放；包含九次旧尝试总消耗{budget['total_gpu_hours']:.4f}GPU小时，上限24小时。完整科学备份包、逐文件SHA与独立解压均有记录。最终CPU释放、报告远端同步和heartbeat关闭以FINAL_DELIVERY_CLOSURE.json为准，不以本报告生成代替资源收尾。"]
    (out/'REPORT.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    metric_names=['list/ndcg@5','list/hr@5','list/recall@5']+[f'aggregate/{m}@{k}' for k in (10,20,50) for m in ('ndcg','hr','recall')]+['list/completion_rate','list/missing_rate','list/invalid_rate','list/duplicate_rate']
    metric_names += [f'aggregate/coverage@{k}' for k in (10,20,50)]+['aggregate/candidate_count','aggregate/lists_used']
    with (out/'FIXED_TEST_METRICS.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','E','D','A'])
        for k in metric_names:w.writerow([k,*[fixed[n][k] for n in ('E','D','A')]])
    write(out/'REPORT_INPUTS.json',dict(status='generated_from_passed_audits',input_sha256=INPUTS,report_sha256=sha(out/'REPORT.md'),resource_closure_separate=True))
    print(json.dumps(dict(report=str(out/'REPORT.md'),selected_epoch=best,fixed_comparisons=primary),ensure_ascii=False))

if __name__=='__main__':main()
