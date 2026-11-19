"""Write the agreed comparison only after all four runs have finished."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V=('H0-A','H1-A','H0-B','H1-B')
EPOCHS=(5,8,11,14,17,20)
KEY='list/ndcg@5'

def baseline_rows():
    data=json.loads((HERE/'BASELINES_R20.json').read_text())
    result={}
    r5=json.loads((HERE/'BASELINE_R5_VALIDATION.json').read_text())
    assert r5['split']=='validation' and r5['full_split'] and r5['examples']==71657
    assert r5['r_updates']==3465 and r5['t_updates']==0
    for arm in r5['shared_by']:
        result[arm,5]=r5[KEY]
    for arm,records in data.items():
        for entry in records:
            item=entry['summary']
            if item['split']=='validation':
                epoch=item['r_updates']//693
                if epoch in EPOCHS:result[arm,epoch]=item['metrics'][KEY]
    return result

def main():
    snapshot=json.loads((HERE/'LATEST.json').read_text())
    baselines=baseline_rows()
    table=[]
    for variant in V:
        row=snapshot['variants'][variant]
        assert snapshot['platform'][variant]['status']=='SUCCEEDED',variant
        assert row['TRAINING_COMPLETE.json']['R']==20 and row['TRAINING_COMPLETE.json']['T']==200
        assert row['GPU_COMPLETE.json']['status']=='passed'
        val={r['r_updates']//693:r for r in row['validations']}
        assert set(val)==set(EPOCHS),(variant,list(val))
        checkpoints={r['epoch']:r for r in row['checkpoints']}
        assert set(checkpoints)==set(EPOCHS)
        assert all(checkpoints[e]['T']==max(0,(e-5)//3*40) for e in EPOCHS)
        best=max(EPOCHS,key=lambda e:(val[e]['metrics'][KEY],-e))
        assert best==row['TRAINING_COMPLETE.json']['best_epoch']
        tests={r['path'].split('/')[-2].split('_retry')[0]:r for r in row['tests']}
        assert {'test_best','test_R20'}<=set(tests)
        for epoch in EPOCHS:
            table.append(dict(variant=variant,arm=variant[:2],R=epoch,T=checkpoints[epoch]['T'],
                              validation_ndcg5=val[epoch]['metrics'][KEY],
                              original_validation_ndcg5=baselines.get((variant[:2],epoch)),
                              original_delta=None if (variant[:2],epoch) not in baselines else
                              val[epoch]['metrics'][KEY]-baselines[variant[:2],epoch],
                              selected_best=(epoch==best),
                              test_ndcg5=(tests['test_best']['metrics'][KEY] if epoch==best else
                                          tests['test_R20']['metrics'][KEY] if epoch==20 else None)))
    with (HERE/'COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 四组 Decoder 前缀实验 R20 对照','','同 seed2024；新结构各自重新暖身，不能把与原 H0/H1 的差值解释为单独前缀模块的因果效果。所有 best 只按验证 NDCG@5 选取。','','| 版本 | R5 | R8 | R11 | R14 | R17 | R20 | best (R) | best 测试 | R20 测试 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in V:
        values=[r for r in table if r['variant']==variant];best=next(r for r in values if r['selected_best'])
        tests={r['path'].split('/')[-2].split('_retry')[0]:r for r in snapshot['variants'][variant]['tests']}
        lines.append('| '+variant+' | '+' | '.join(f"{r['validation_ndcg5']:.6f}" for r in values)+' | '+str(best['R'])+' | '+f"{tests['test_best']['metrics'][KEY]:.6f}"+' | '+f"{tests['test_R20']['metrics'][KEY]:.6f}"+' |')
    lines+=['','原 H0/H1 同预算验证指标和差值逐行见 COMPARISON.csv；原模型与新结构的训练起点及容量不同。完整资源和 SHA 验收另见 FINAL_ACCEPTANCE.json。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Comparison generated for four completed runs')

if __name__=='__main__':main()
