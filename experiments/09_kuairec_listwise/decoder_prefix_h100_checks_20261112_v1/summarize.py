import json,csv
from pathlib import Path
p=Path(__file__).resolve().parent
x=json.loads((p/'COMPLETE.json').read_text())
rows=[]
for name,v in x['variants'].items():
 for phase in ('R','T'):
  a=v['reference']['phases'][phase];b=v['optimized']['phases'][phase]
  rows.append(dict(variant=name,phase=phase,before_s=a['mean_total_s'],after_s=b['mean_total_s'],speedup=a['mean_total_s']/b['mean_total_s'],allocated_gib=b['peak_allocated_gib'],reserved_gib=b['peak_reserved_gib'],exact=v['exact']))
with (p/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
table=[]
for name in x['variants']:
 r,t=[z for z in rows if z['variant']==name]
 table.append(f"|{name}|{r['before_s']:.4f} → {r['after_s']:.4f} ({r['speedup']:.3f}×)|{t['before_s']:.4f} → {t['after_s']:.4f} ({t['speedup']:.3f}×)|{max(r['allocated_gib'],t['allocated_gib']):.3f}|{r['exact']}|")
s='# 第二批执行优化 H100 短测（2026-11-12）\n\n'
s+='两边均启用第一批静态数据与 Tensor 复用，只切换本次三项：固定监督数量直接计算、一次前向入口检查 SID 后内部复用检查结论、OCN 全 MASK 常量缓存。microbatch=256，有效 batch=1024，BF16；训练/损失/随机消耗顺序不变。\n\n'
s+='每个版本 R/T 各8步，首步预热、7步计时；固定真实样本和完整模型，新初始化；另32样本预测及恢复下一步核对。仅短测，没有正式训练。\n\n'
s+='|版本|R 秒/有效更新：前→后|T 秒/有效更新：前→后|优化后活跃张量峰值 GiB|逐位一致|\n|---|---:|---:|---:|---|\n'+'\n'.join(table)+'\n\n'
s+='一致性范围：模型/Tokenizer、梯度、优化器、调度器、每步损失及随机状态、逐样本预测、恢复下一步。短测速度有执行先后与系统噪声，不能直接当作整轮/全程训练收益；不比较模型推荐质量。\n\n'
s+='原始数据 COMPLETE.json；源码 SHA SOURCE_MANIFEST.json；本地测试 CPU_TESTS.json；资源释放以 FINAL_ACCEPTANCE.json 为准。远端根 /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/decoder_prefix_h100_checks_20261112_v1。\n'
(p/'REPORT.md').write_text(s,encoding='utf-8');print('\n'.join(table));print('all_exact',x['all_exact'])
