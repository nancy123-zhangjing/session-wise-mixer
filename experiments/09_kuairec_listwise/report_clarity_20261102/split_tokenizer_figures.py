import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
audit=json.loads((OUT/'LOSS_AUDIT.json').read_text(encoding='utf-8'))
data={}
for name,source in audit['sources'].items():
 rows={}
 for line in (ROOT/source['path']).read_text().splitlines():
  r=json.loads(line)
  if r['phase']=='T': rows[r['t_updates']]=r
 data[name]=[rows[k] for k in sorted(rows)]
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':14,'axes.titlesize':20,'axes.labelsize':15,'legend.fontsize':12})
labels={'H0':'H0：无Mixer','H1':'原H1','D':'D：每次编码更新步数减半','E':'E：推荐器连续训练轮数加倍'}
colors={'H0':'#555555','H1':'#d95f02','D':'#0072B2','E':'#B2188B'}
styles={'H0':'-','H1':'-','D':'-','E':'-'}
titles={'loss':'Tokenizer总训练损失','recommendation':'Tokenizer：推荐损失分量','reconstruction':'Tokenizer：重建损失分量','balance':'Tokenizer：均衡损失分量'}
xml=[]
for field,title in titles.items():
 fig,ax=plt.subplots(figsize=(10.8,6.7))
 for n,rows in data.items():
  xs=[r['t_updates'] for r in rows];ys=[r[field] for r in rows]
  smooth=[sum(ys[max(0,i-9):i+1])/len(ys[max(0,i-9):i+1]) for i in range(len(ys))]
  ax.plot(xs,ys,color=colors[n],alpha=.16,lw=.9)
  ax.plot(xs,smooth,label=labels[n],color=colors[n],ls=styles[n],lw=2.6)
 ax.set_title(title,pad=18)
 ax.set_xlabel('Tokenizer累计优化器更新步数',labelpad=12)
 ax.set_ylabel('损失值（未乘权重）' if field!='loss' else '总损失值',labelpad=12)
 ax.set_xlim(0,405);ax.set_xticks(range(0,401,50));ax.yaxis.set_major_locator(MaxNLocator(7))
 ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 ax.grid(alpha=.19);ax.spines[['top','right']].set_visible(False)
 ax.legend(loc='best',framealpha=.94)
 fig.text(.5,.035,'浅线：逐步原始值；深线：过去10步移动平均。H0/原H1为400步，D/E为200步。',ha='center',fontsize=12)
 fig.subplots_adjust(left=.12,right=.97,top=.88,bottom=.19)
 filename=f'tokenizer_{field}_large.png'
 fig.savefig(OUT/filename,dpi=200);plt.close(fig)
 caption=title+'；'+('均衡项在总损失中的权重为0.01。' if field=='balance' else '各图纵轴独立缩放，比较时请留意刻度。')
 xml.append(f'<img path="@./experiments/09_kuairec_listwise/report_clarity_20261102/{filename}" width="1000" caption="{caption}"/>')
(OUT/'split_figures.xml').write_text('\n'.join(xml),encoding='utf-8')
print('Created four separate figures and XML patch.')
