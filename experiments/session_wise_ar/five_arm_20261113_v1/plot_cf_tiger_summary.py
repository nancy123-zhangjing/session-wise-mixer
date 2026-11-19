"""Plot completed CF/TIGER receipts; no training or inference."""
import json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
O=P/'cf_tiger_summary';O.mkdir(exist_ok=True)
inputs=[]
def read(name):
 p=P/name;inputs.append(p);return json.loads(p.read_text(encoding='utf-8'))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':12})
rows=read('CF_INTERIM_COMPARISON.json')['rows']
recs=[read(p.name)['record'] for p in sorted(P.glob('AR_T0_*_VALIDATION.json'))]
recs=sorted({r['step']:r for r in recs}.values(),key=lambda r:r['step'])
assert len(recs)==15 and all(r['validation']['count']==71657 for r in recs)
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for key,label,color in [('CF_M0_metrics','CF 无 Mixer','#0072B2'),('CF_M1_metrics','CF 有 Mixer','#D55E00')]:
 axes[0].plot([r['R'] for r in rows],[r[key]['ndcg@5'] for r in rows],'-o',ms=3,color=color,label=label)
axes[0].set(title='CF：完整验证曲线',xlabel='推荐器轮次 R',ylabel='验证 NDCG@5')
axes[1].plot([r['session_epochs'] for r in recs],[r['validation']['metrics']['ndcg@5'] for r in recs],'-o',ms=4,color='#009E73',label='TIGER 无 Mixer')
axes[1].set(title='TIGER：完整验证曲线',xlabel='等效 session 轮次',ylabel='验证 NDCG@5')
for ax in axes:ax.grid(alpha=.2);ax.legend();ax.set_ylim(0,.009)
fig.savefig(O/'validation.png',dpi=170);plt.close(fig)
a=read('CF_TEST_COMPLETE.json');b=read('CF_M1_REFERENCE_SUMMARY.json');t=read('AR_T0_TEST_AUDIT.json')
groups=[('CF 无 Mixer',a['test_best'],a['test_final']),('CF 有 Mixer',b['test']['best'],b['test']['final']),('TIGER 无 Mixer',t['best'],t['final'])]
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
for j,(key,label,color) in enumerate([(1,'验证选出的 best','#0072B2'),(2,'实际最终点','#D55E00')]):
 vals=[g[key]['metrics']['ndcg@5'] for g in groups]
 assert all(g[key]['count']==53194 for g in groups)
 bars=ax.bar([i+(j-.5)*.34 for i in range(3)],vals,width=.34,label=label,color=color)
 ax.bar_label(bars,labels=[f'{v:.6f}' for v in vals],padding=4,fontsize=11)
ax.set_xticks(range(3),[g[0] for g in groups]);ax.set(ylabel='测试 NDCG@5',ylim=(0,.0105),title='完整测试集：最佳点与最终点（各方法训练预算不同）');ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(O/'test_best_final.png',dpi=170);plt.close(fig)
(O/'MANIFEST.json').write_text(json.dumps({'scope':'completed CF/T0 results, not five-arm acceptance','inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'limits':['single seed','T1 failed before recommender training; not plotted','method schedules and budgets differ']},indent=2),encoding='utf-8')
print('Created validation.png and test_best_final.png')
