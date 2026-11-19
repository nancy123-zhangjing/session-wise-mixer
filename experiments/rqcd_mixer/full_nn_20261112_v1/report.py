"""Report full paired evaluation; optionally compute cross-arm paired comparisons."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
p=argparse.ArgumentParser(); p.add_argument('root',type=Path); args=p.parse_args(); root=args.root
done=json.loads((root/'COMPLETE.json').read_text(encoding='utf-8')); results=done['results']
comparisons={}
for split in ('validation','test'):
    paths={a:root/a/split/'predictions.jsonl.gz' for a in results}
    if not all(p.exists() for p in paths.values()): continue
    arrays={}; identity=None
    for a,path in paths.items():
        sid=[];users=[];score=[]
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);sid.append(row['sample_id']);users.append(row['user_id']);score.append(row['nn_ndcg'])
        if identity is None: identity=(sid,users)
        else: assert identity==(sid,users)
        arrays[a]=np.asarray(score,dtype=np.float64)
    unique,inv=np.unique(identity[1],return_inverse=True); counts=np.bincount(inv)
    comparisons[split]={}
    for other in ('B0','B2'):
        difference=arrays['B1']-arrays[other]; sums=np.bincount(inv,weights=difference)
        rng=np.random.default_rng(20261112); boots=[]
        for _ in range(3000):
            idx=rng.integers(0,len(unique),len(unique));boots.append(sums[idx].sum()/counts[idx].sum())
        comparisons[split]['B1_minus_'+other]={'mean_difference':float(difference.mean()),
            'relative_change':float(arrays['B1'].mean()/arrays[other].mean()-1),
            'paired_user_bootstrap_95pct':np.quantile(boots,[.025,.975]).tolist()}
if comparisons:
    (root/'CROSS_ARM.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
elif (root/'CROSS_ARM.json').exists(): comparisons=json.loads((root/'CROSS_ARM.json').read_text())
lines=['# 三组best：完整验证与测试的同向量最近邻落地对照','',
    '仅推理，不训练、不重新选点。B0/R25、B1/R28、B2/R24均沿用原完整验证选择的best。验证71657条，测试53194条。', '',
    '原20步DDIM、BF16、batch256及固定噪声不变。RQ与NN接收同一批生成向量，NN采用全目录量化前潜向量的平方欧氏距离，各自排除列表内重复商品。NN不反馈改变后续前缀，所以这是固定RQ生成轨迹上的替代读出，不是NN自主滚动生成评测。','']
for split,title in [('validation','完整验证集'),('test','完整测试集')]:
    lines += ['## '+title,'','|组别|RQ NDCG@5|NN NDCG@5|相对变化|RQ/NN Recall@5|RQ/NN命中session数|','|---|---:|---:|---:|---|---:|']
    for arm,v in results.items():
        r=v[split];a,b=r['rq'],r['nn']
        lines.append(f"|{arm}/R{r['R']}|{a['ndcg@5']:.9f}|{b['ndcg@5']:.9f}|{r['relative_ndcg_change']:+.2%}|{a['recall@5']:.9f} / {b['recall@5']:.9f}|{r['rq_hit_sessions']} / {r['nn_hit_sessions']}|")
    lines += ['','|组别|NN-RQ NDCG差的用户配对bootstrap 95%区间|原RQ完整列表复现率|','|---|---|---:|']
    for arm,v in results.items():
        r=v[split];lo,hi=r['paired_user_bootstrap_95pct']
        lines.append(f"|{arm}|[{lo:.9f}, {hi:.9f}]|{r['original_exact_list_agreement']:.2%}|")
    lines += ['']
lines += ['## NN组间比较','','|数据集|比较|B1相对提升|NDCG差95%区间|','|---|---|---:|---|']
for split,items in comparisons.items():
    for name,v in items.items():
        lo,hi=v['paired_user_bootstrap_95pct'];lines.append(f"|{split}|{name}|{v['relative_change']:+.2%}|[{lo:.9f}, {hi:.9f}]|")
lines += ['', '区间按用户聚类配对重采样3000次，未作多重比较校正。验证包含此前查看过的8192条子集，测试未用于重新选点；这些均是单seed诊断。', '',
    '## 比较边界','',
    '若B1在NN读出下优于其他组，可以支持“B1整条路线产生的向量更容易通过最近邻得到有效推荐”。但各组RQ预训练潜空间不同，不能把差异全部归因于生成器本身、某次码本交互或落地方式；当前也不是共用同一潜空间的严格生成器对照。', '',
    '所有大文件保留远端，包括六组生成向量、逐样本预测及哈希；本地仅保存小摘要、报告与资源关闭收据。', '',
    f"程序总耗时{done['seconds']:.1f}秒（不含排队启动）。所有节点检查模型未改变、原RQ逐样本预测完全一致。",'']
(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'cross_arm':comparisons,'results':{a:{s:{'rq':r['rq'],'nn':r['nn'],'change':r['relative_ndcg_change']} for s,r in v.items()} for a,v in results.items()}},ensure_ascii=False))
