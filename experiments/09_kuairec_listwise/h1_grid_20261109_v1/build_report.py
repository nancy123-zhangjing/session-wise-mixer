"""Generate the local A/B/C comparison from accepted small JSON artifacts only."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    baseline=json.loads((ROOT/'BASELINE_A.json').read_text())['rows']
    audit=json.loads((ROOT/'FINAL_ARTIFACT_AUDIT.json').read_text())
    if audit['status']!='passed': raise ValueError('Artifact audit has not passed')
    av={r['epoch']:r['ndcg5'] for r in baseline if r['split']=='validation'}
    at={r['epoch']:r for r in baseline if r['split']=='test'}
    curves={'A':av}
    for arm in ('B','C'):
        result=audit['arms'][arm]
        if result['status']!='passed':raise ValueError('Arm audit incomplete')
        curves[arm]={r['epoch']:r['metrics']['list/ndcg@5'] for r in result['validations']}
    for curve in curves.values():curve[5]=audit['shared_R5_validation_ndcg5']
    def number(x):return '—' if x is None else f'{x:.6f}'
    def delta(x,y):return '—' if x is None or y is None else f'{(x/y-1)*100:+.2f}%'
    lines=['# H1 历史 SID 注意力：A/B/C 对照','',
        'A 为已有原 H1 Freeze，B 新增商品内横向注意力，C 在 B 基础上再加同码位跨商品纵向注意力。三组均使用 seed 2024、共同原 H1 R5 起点、R20/T200 冻结 Tokenizer 后继续推荐器至 R35。Grid 属于推荐器，冻结后继续训练。','',
        '新增注意力均为 256 维、4 头、Pre-LN 残差，无新增 FFN，输出投影零初始化。原 MLP、Encoder、Decoder 和训练评价口径保持一致。','',
        '## 逐验证节点','',
        '| R | A NDCG@5 | B NDCG@5 | C NDCG@5 | B 相对 A | C 相对 A |',
        '|---|---:|---:|---:|---:|---:|']
    for epoch in (8,11,14,17,20,23,26,29,32,35):
        a,b,c=(curves[k].get(epoch) for k in ('A','B','C'))
        lines.append(f'| {epoch} | {number(a)} | {number(b)} | {number(c)} | {delta(b,a)} | {delta(c,a)} |')
    lines+=['','## 验证集选定最佳点与固定 R35','',
        '| 组别 | 节点 | R | 验证 NDCG@5 | 测试 NDCG@5 | 测试 HR@5 | 测试 Recall@5 |',
        '|---|---|---:|---:|---:|---:|---:|']
    summary={}
    for arm in ('A','B','C'):
        best_epoch=max(av,key=lambda e:(av[e],-e)) if arm=='A' else audit['arms'][arm]['best_epoch']
        summary[arm]={}
        for label,epoch in [('best',best_epoch),('R35',35)]:
            if arm=='A':
                test=at[epoch];ndcg,hr,recall=test['ndcg5'],test['hr5'],test['recall5']
            else:
                records={r['label']:r for r in audit['arms'][arm]['tests']}
                record=records[label]
                if record.get('reused'):record=records[record['reused']]
                metrics=record['metrics'];ndcg,hr,recall=(metrics['list/'+k] for k in ('ndcg@5','hr@5','recall@5'))
            summary[arm][label]=dict(epoch=epoch,validation_ndcg5=curves[arm].get(epoch),test_ndcg5=ndcg,test_hr5=hr,test_recall5=recall)
            lines.append(f'| {arm} | {label} | {epoch} | {number(curves[arm].get(epoch))} | {number(ndcg)} | {number(hr)} | {number(recall)} |')
    lines+=['','## 观察与边界','']
    for arm in ('B','C'):
        for label in ('best','R35'):
            x,y=summary[arm][label]['test_ndcg5'],summary['A'][label]['test_ndcg5']
            lines.append(f'- {arm} 的 {label} 测试 NDCG@5 相对 A 的对应比较点为 {delta(x,y)}。')
    lines+=['','以上为单 seed 点估计，不能据此断言跨 seed 稳定收益。best 按验证集选择，R35 是固定训练预算对照；两种比较分别解释。C 比 B 还增加了参数量和计算量，差异不能单独归因于纵向注意力拓扑。',
        '', '## 验收范围','',
        '训练产物审计见 FINAL_ARTIFACT_AUDIT.json：验证 R35/T200、R20 后 Tokenizer/SID/T optimizer 不变、Grid 持续学习，检查各验证节点完整 checkpoint 的 CPU 加载与优化器形状，以及 best/R35 测试文件 SHA。CPU 加载检查不等于重新验证 CUDA 下一步数值一致性。平台资源释放另以最终关闭收据为准。','']
    (ROOT/'COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
