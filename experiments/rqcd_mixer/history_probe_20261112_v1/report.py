import json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).parent;p=json.loads((root/'COMPLETE.json').read_text());r=p['results']
lines=['# 真实历史与打乱历史：第一件商品及分噪声诊断','',
       '复用B0/R25、B1/R28和原8192条固定验证子集；只推理，无训练。历史置换来自不同用户，真实与打乱条件共享噪声。只考察第一件，不存在推荐前缀。', '',
       f"历史长度匹配率：{p['shuffle_summary']['same_history_length_fraction']:.4%}；平均长度差：{p['shuffle_summary']['mean_absolute_length_difference']:.4f}；不同用户比例：{p['shuffle_summary']['different_user_fraction']:.2%}。", '',
       '## 从纯噪声自由生成第一件', '',
       '第一目标命中指预测恰好等于列表第一件；任意目标命中指预测属于五件真实目标之一，不是五件推荐列表的HR@5。', '',
       '|组别/历史|NN第一目标命中|NN任意目标命中|RQ任意目标命中|第一目标平均排名|第一目标Recall@10|',
       '|---|---:|---:|---:|---:|---:|']
for arm,v in r.items():
 for mode in ('real','shuffled'):
  s=v['summary']['free/'+mode];lines.append(f"|{arm}/{mode}|{s['nn_first_hit']:.4%}|{s['nn_any5_hit']:.4%}|{s['rq_any5_hit']:.4%}|{s['rank']:.1f}|{s['nn_first_recall10']:.4%}|")
lines+=['','## 单次去噪（真实目标加噪）','','|组别/噪声t|真实历史MSE|打乱历史MSE|真实历史排名|打乱历史排名|真实/打乱Recall@10|','|---|---:|---:|---:|---:|---|']
for arm,v in r.items():
 for level,t in [('low',100),('medium',500),('high',900)]:
  a,b=v['summary'][level+'/real'],v['summary'][level+'/shuffled']
  lines.append(f"|{arm}/{t}|{a['mse']:.6f}|{b['mse']:.6f}|{a['rank']:.1f}|{b['rank']:.1f}|{a['nn_first_recall10']:.4%}/{b['nn_first_recall10']:.4%}|")
lines+=['','时间步不是噪声百分比；实际alpha_bar：'+json.dumps(r['B0']['alpha_bar']), '',
       '## 换历史造成的变化','','|组别/条件|输出向量均方变化|NN商品变化率|RQ商品变化率|真实-打乱MSE配对95%区间|','|---|---:|---:|---:|---|']
for arm,v in r.items():
 for level in ('free','low','medium','high'):
  s=v['summary'][level+'/sensitivity'];pair=v['paired'][level];lo,hi=pair['mse']['user_bootstrap_95pct']
  lines.append(f"|{arm}/{level}|{s['output_mse']:.6f}|{pair['nn_item_changed_fraction']:.2%}|{pair['rq_item_changed_fraction']:.2%}|[{lo:.6f},{hi:.6f}]|")
lines+=['','## 去噪前基线','','|组别/t|去噪前MSE|去噪前目标平均排名|去噪前Recall@10|','|---|---:|---:|---:|']
for arm,v in r.items():
 for level,t in [('low',100),('medium',500),('high',900)]:
  s=v['summary'][level+'/before'];lines.append(f"|{arm}/{t}|{s['mse']:.6f}|{s['rank']:.1f}|{s['nn_first_recall10']:.4%}|")
lines+=['','## 核验与解释范围','',
       '真实历史自由生成的第一件RQ预测与前次诊断逐条完全一致；权重前后不变。完整配对区间、每项均值与向量复现最大差见COMPLETE.json；用户聚类bootstrap 2000次，单seed、单置换，未作多重比较校正。', '',
       '历史影响输出不等于历史有益；高噪声更难本身不等于实现错误；教师加噪输入保留目标信息，不能当作正常推理推荐成绩。排名按全目录潜向量平方距离，等距按商品编号排序。', '',
       f"程序运行{p['seconds']:.1f}秒，不含排队与启动。",'']
(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))
