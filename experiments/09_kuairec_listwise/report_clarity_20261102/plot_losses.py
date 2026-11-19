import json, hashlib, csv
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
paths={
 'H0':'outputs/stage09_restore/goal_20261027_r1/runs/H0/training.jsonl',
 'H1':'outputs/stage09_restore/goal_20261027_r1/runs/H1/training.jsonl',
 'D':'outputs/stage09_spread200_20261031_fresh_r1/runs/H1-Spread200-R35/training.jsonl',
 'E':'outputs/stage09_block40_single_20261101_fresh_r1/runs/H1-Block40-R6-R35/training.jsonl'}
labels={'H0':'H0：无Mixer','H1':'原H1','D':'D：每次编码更新步数减半','E':'E：推荐器连续训练轮数加倍'}
colors={'H0':'#555555','H1':'#d95f02','D':'#2274b8','E':'#22956a'}
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
allrows={}; audit={}; summaries={}; extracted=[]
for name,rel in paths.items():
 p=ROOT/rel; records=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 unique={}
 for r in records:
  key=(r['phase'],r['r_updates'] if r['phase']=='R' else r['t_updates'])
  if key in unique:
   assert abs(unique[key]['loss']-r['loss'])<1e-7, (name,key,'conflicting duplicate')
  unique[key]=r
 rows=list(unique.values()); allrows[name]=rows
 audit[name]={'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':len(records),'unique':len(rows)}
 groups=defaultdict(list)
 for r in rows:
  if r['phase']=='R': groups[r['epoch']].append(r)
 epochs=[]
 for epoch,rr in sorted(groups.items()):
  assert len(rr)==693, (name,epoch,len(rr))
  v=sum(r['loss']*r['lists'] for r in rr)/sum(r['lists'] for r in rr)
  epochs.append((epoch,v)); extracted.append({'model':name,'epoch':epoch,'loss':v})
 summaries[name]={'R_epoch_loss':epochs,'T_first20':{},'T_last20':{}}
 tt=sorted([r for r in rows if r['phase']=='T'],key=lambda r:r['t_updates'])
 assert len(tt)==(400 if name in ['H0','H1'] else 200)
 for field in ['loss','recommendation','reconstruction','balance','anchor']:
  for k,part in [('T_first20',tt[:20]),('T_last20',tt[-20:])]:
   summaries[name][k][field]=sum(r[field] for r in part)/len(part)

fig,ax=plt.subplots(figsize=(11,4.6))
for n,s in summaries.items():
 x,y=zip(*s['R_epoch_loss']); ax.plot(x,y,label=labels[n],color=colors[n],marker='.',lw=1.7)
ax.set(xlabel='推荐器累计训练轮次（epoch）',ylabel='推荐器训练loss（每轮按样本数加权）',title='推荐器训练loss：共同暖身之后的第6—35轮')
ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(OUT/'recommender_loss.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,field,title in zip(axes.flat,['loss','recommendation','reconstruction','balance'],['Tokenizer总loss','推荐损失分量（权重1）','重建损失分量（权重1）','均衡损失分量（图示原值，权重0.01）']):
 for n,rows in allrows.items():
  tt=sorted([r for r in rows if r['phase']=='T'],key=lambda r:r['t_updates'])
  xs=[r['t_updates'] for r in tt]; ys=[r[field] for r in tt]
  ax.plot(xs,ys,color=colors[n],alpha=.13,lw=.65)
  smooth=[sum(ys[max(0,i-9):i+1])/len(ys[max(0,i-9):i+1]) for i in range(len(ys))]
  ax.plot(xs,smooth,color=colors[n],label=labels[n],lw=1.5)
 ax.set(title=title,xlabel='Tokenizer累计优化器更新步数',ylabel='训练loss');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Tokenizer训练loss：浅线为原始记录，实线为过去10步移动平均',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'tokenizer_loss.png',dpi=180);plt.close(fig)
(OUT/'LOSS_AUDIT.json').write_text(json.dumps({'sources':audit,'summaries':summaries},ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'recommender_epoch_loss.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['model','epoch','loss']);w.writeheader();w.writerows(extracted)
for n,s in summaries.items():
 print(n,'R6,R20,R35',[(e,round(v,5)) for e,v in s['R_epoch_loss'] if e in [6,20,35]],'T first/last20', {k:round(s['T_first20'][k],5) for k in s['T_first20']},{k:round(s['T_last20'][k],5) for k in s['T_last20']})
