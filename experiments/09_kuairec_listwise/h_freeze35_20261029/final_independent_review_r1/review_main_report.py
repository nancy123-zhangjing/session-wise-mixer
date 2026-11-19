"""Read-only check of the main manuscript against accepted result receipts."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parents[2]
RUNS = ROOT / 'outputs/stage09_restore/h_freeze35_20261029_final_r1_restored/goal/runs'
EXPECTED_DRAFT_SHA = 'b94242d720673a84da788e3e17d7758a2b0ffaa4c2749942d5925c8d26e40c35'
BOUND = {}
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    assert expected is None or value == expected, str(path)
    assert str(path) not in BOUND or BOUND[str(path)] == value
    BOUND[str(path)] = value
    return path


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text(encoding='utf-8'))


def check(condition, label):
    assert condition, label
    CHECKS.append(label)


def row(cells, label):
    line = '| ' + ' | '.join(str(c) for c in cells) + ' |'
    check(line in text, label + ': exact displayed row')


def pairrow(display, split, value, label):
    row([display, split, f'{value["delta"]:+.8f}', f'{value["relative_delta"]:+.2%}',
         f'[{value["confidence_interval"][0]:+.8f}, {value["confidence_interval"][1]:+.8f}]'], label)


def receipt(report, key):
    spec = report['comparisons'][key]
    result = read(spec['path'], spec['sha256'])
    check(result['bootstrap']['repeats'] == 2000 and result['bootstrap']['seed'] == 2024
          and result['bootstrap']['confidence'] == .95, key + ': original bootstrap')
    return result


draft = bind(BASE / 'FINAL_MAIN_REPORT_DRAFT.md', EXPECTED_DRAFT_SHA)
text = draft.read_text(encoding='utf-8')
reports = {b: read(BASE / f'final_analysis/{b}/analysis_report.json') for b in 'ABC'}
final = read(BASE / 'final_comparison_r1/analysis_report.json', 'ce5f8ecec42a7f73d76d7cf83482cbf8193e2a549e701e06c9097dbca64d8ee2')
models = final['models']
check(all(r['status'] == 'passed' for r in reports.values()) and final['status'] == 'passed', 'accepted source reports')

ref = models['A_R26_validation']['metrics']['list/ndcg@5']
previous = None
for e in (26,29,32,35):
    v = models[f'A_R{e}_validation']['metrics']
    ndcg = v['list/ndcg@5']
    row([e, f'{ndcg:.8f}', f'{v["list/hr@5"]:.8f}', f'{v["aggregate/ndcg@50"]:.8f}',
         f'{ndcg/ref-1:+.2%}', '—' if previous is None else f'{ndcg/previous-1:+.2%}'], f'A validation R{e}')
    previous = ndcg

metric_names = ('list/ndcg@5','list/hr@5','list/recall@5','aggregate/ndcg@50')
fixed_labels = {'A':'A：H1 / R20冻结','B':'B：H0 / R32冻结','C':'C：H0 / R20冻结'}
for b in 'ABC':
    v = models[f'{b}_R35_test']['metrics']
    row([fixed_labels[b], f'R35 / T{360 if b=="B" else 200}', *[f'{v[m]:.8f}' for m in metric_names]], b + ' fixed test metrics')
    for k in (10,20,50):
        row([b,k,*[f'{v[f"aggregate/{m}@{k}"]:.8f}' for m in ('ndcg','hr','recall')],f'{v[f"aggregate/coverage@{k}"]:.2%}'],f'{b} aggregate K{k}')
    check(v['list/completion_rate']==1 and v['list/duplicate_rate']==0 and v['list/invalid_rate']==0, b+' complete unique valid list')
    check(f'{v["aggregate/candidate_count"]:.2f}' in text,b+' displayed mean candidate count')

for label, model, rt in [('A 原共同节点 best','A_R32_test','R32 / T200'),('C 原共同节点 best','C_R23_test','R23 / T200'),
                         ('旧 H0 best / B父模型','B_R32_test','R32 / T360')]:
    row([label,rt,*[f'{models[model]["metrics"][m]:.8f}' for m in metric_names]], model+' selected/old test metrics')

final_labels = {'A_minus_C_R35_test':'A R35 − C R35（主比较）','A_minus_B_R35_test':'A R35 − B R35（补充）',
                'A_minus_C_R35_validation':'A R35 − C R35','A_minus_B_R35_validation':'A R35 − B R35',
                'A_minus_C_validation_best_test':'A best R32 − C best R23','A_minus_C_validation_best_validation':'A best R32 − C best R23'}
final_receipts = {}
for key, label in final_labels.items():
    result = receipt(final,key); final_receipts[key]=result
    pairrow(label,result['split'],result['primary']['list/ndcg@5'],key)
for b in ('B','C'):
    v=final_receipts[f'A_minus_{b}_R35_test']['primary']['aggregate/ndcg@50']
    check(f'{v["delta"]:+.8f}' in text and f'[{v["confidence_interval"][0]:+.8f}, {v["confidence_interval"][1]:+.8f}]' in text,
          f'A-{b} aggregate50 prose difference/CI')
check(f'{final_receipts["A_minus_C_R35_test"]["primary"]["aggregate/ndcg@50"]["relative_delta"]:.2%}' in text,'A-C aggregate50 percentage')

origin_specs = [('A','A_R35_minus_A_R26_test','A R35 − A Freeze R26'),('A','A_R35_minus_A_R20_test','A R35 − 旧 A R20'),
                ('A','A_R35_minus_B_R32_test','A R35 − 旧 B R32'),('A','A_R35_minus_A_R26_validation','A R35 − A Freeze R26'),
                ('B','B_R35_minus_B_R32_test','B R35 − B父 R32'),('B','B_R35_minus_B_R32_validation','B R35 − B父 R32'),
                ('C','C_R35_minus_R20_validation','C R35 − C重放 R20'),('C','C_R35_minus_R5_validation','C R35 − C原 R5')]
for b,key,label in origin_specs:
    r=receipt(reports[b],key); pairrow(label,r['split'],r['primary']['list/ndcg@5'],key)

supp=read(BASE/'final_analysis/A/SUPPLEMENTAL_BEST_COMPARISON_BINDING.json')
check(supp['analysis_report_sha256']==sha(BASE/'final_analysis/A/analysis_report.json'),'supplement binds accepted A branch')
complete=read(supp['source_complete'],supp['source_complete_sha256'])
check(complete['status']=='early_cpu_test_metrics_passed','early CPU complete status')
supp_labels={'A_R32_test_minus_A_R26_test':('A best R32 − A Freeze R26','A_R26_test'),
             'A_R32_test_minus_A_R20_test':('A best R32 − 旧 A R20','A_R20_test'),
             'A_R32_test_minus_B_R32_test':('A best R32 − 旧 B R32','B_R32_test')}
for key,(label,baseline) in supp_labels.items():
    spec=supp['comparisons'][key]; r=read(spec['source'],spec['sha256'])
    check(r['primary']['list/ndcg@5']==spec['primary'] and r['inputs']==spec['inputs'],key+': reused values and inputs')
    check(r['inputs']=={'baseline_sha256':models[baseline]['predictions_sha256'],
                       'treatment_sha256':models['A_R32_test']['predictions_sha256']},key+': endpoint binding')
    check(r['bootstrap']['repeats']==2000 and r['bootstrap']['seed']==2024 and r['bootstrap']['confidence']==.95,key+': bootstrap protocol')
    pairrow(label,'test',spec['primary'],key)
check('没有新增推理或重新计算这三组区间' in text,'supplement correctly labelled reused intervals')

provenance=read(BASE/'final_figures_r1/PLOT_PROVENANCE.json')
for path, expected in provenance['source_sha256'].items(): bind(path,expected)
for name, expected in provenance['output_sha256'].items(): bind(BASE/'final_figures_r1'/name,expected)
with (BASE/'final_figures_r1/TRAINING_CURVES.csv').open(encoding='utf-8-sig',newline='') as f:
    training_csv={(x['branch'],int(x['epoch'])):x for x in csv.DictReader(f)}
with (BASE/'final_figures_r1/VALIDATION_CURVES.csv').open(encoding='utf-8-sig',newline='') as f:
    for x in csv.DictReader(f):
        m=models[f'{x["branch"]}_R{x["epoch"]}_validation']
        check(float(x['ndcg5'])==m['metrics']['list/ndcg@5'] and (x['historical']=='True')==m['reference'],
              f'validation plot {x["branch"]} R{x["epoch"]}')

dirs={'A':'H1-Freeze20-R35','B':'H0-Freeze32-R35','C':'H0-Freeze20-R35'}
rates={}; update_counts={}
for b, directory in dirs.items():
    path=bind(RUNS/directory/'training.jsonl')
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rr=[x for x in rows if x['phase']=='R']; tt=[x for x in rows if x['phase']=='T']
    check(len(rr)=={'A':6237,'B':2079,'C':20790}[b] and len(tt)==(200 if b=='C' else 0),b+' actual new R/T counts')
    check(rr[-1]['r_updates']==24255 and rr[-1]['t_updates']==(360 if b=='B' else 200),b+' actual final counts')
    rates[b]={x['r_updates']:x['learning_rates_used']['R'] for x in rr}
    update_counts[b]={'new_r':len(rr),'new_t':len(tt),'final_r':rr[-1]['r_updates'],'final_t':rr[-1]['t_updates']}
    for e in sorted(set(x['epoch'] for x in rr)):
        group=[x for x in rr if x['epoch']==e]; n=sum(x['lists'] for x in group)
        expected=math.fsum(x['loss']*x['lists'] for x in group)/n
        saved=training_csv[(b,e)]
        check(float(saved['weighted_loss'])==expected and int(saved['lists'])==n==709443 and int(saved['optimizer_updates'])==len(group)==693,
              f'actual R loss plot {b} R{e}')
pairwise={}
for a,b in [('A','C'),('B','C'),('A','B')]:
    common=rates[a].keys() & rates[b].keys();pairwise[a+'-'+b]=len(common)
    check(all(math.isclose(rates[a][k],rates[b][k],rel_tol=1e-12,abs_tol=1e-18) for k in common),a+'-'+b+' logged LR equality')
triple=len(rates['A'].keys() & rates['B'].keys() & rates['C'].keys())
check(pairwise=={'A-C':6237,'B-C':2079,'A-B':2079} and sum(pairwise.values())==10395 and triple==2079,'pairwise versus triple LR counts')
check('三组共同覆盖' in text and '2,079' in text and '10,395' in text and '三对分支' in text,'corrected LR-count wording')

evidence=read(BASE/'C_R20_REPLAY_FREEZE_EVIDENCE.json')
check(evidence['gate']['rule']['max_ndcg5_absolute_delta']==1e-6 and evidence['gate']['rule']['minimum_top_list_agreement']==.999
      and evidence['gate']['rule']['identity_agreement']==1,'locked replay tolerances')
gates=[]
for e in (8,11,14,17,20):
    gate=read(RUNS/dirs['C']/f'replay/R{e:02}.json')
    check(gate['status']=='passed' and gate['examples']==71657 and gate['top_list_agreement']>=.999
          and gate['ndcg5_absolute_delta']<=1e-6,f'C replay R{e}')
    gates.append({'epoch':e,'top_list_agreement':gate['top_list_agreement'],'ndcg5_absolute_delta':gate['ndcg5_absolute_delta']})
check(gates[-1]['top_list_agreement']==1 and gates[-1]['ndcg5_absolute_delta']==0,'C20 exact saved prediction agreement')
traj=evidence['freeze_boundary']['trajectory_reports']
check(len(traj)==20 and all(v['max_absolute_delta']==0 for p in traj for v in p['metrics'].values()),'20 trajectory receipts agree')
read(BASE/'GOAL_OBJECTIVE.md') if False else bind(BASE/'GOAL_OBJECTIVE.md')
bind(BASE/'plot_results.py')
bind(BASE/'cold_popularity_audit_20261030/FINAL_REPORT.md')
for phrase in ['不宣称 bitwise','工程漂移报警','不能把@50解释为一次生成了完整50件列表','辅助比较未做多重检验校正',
               '区间跨零','没有同终点持续更新Tokenizer的反事实','不能把改善全部单独归因于冻结']:
    # The final phrase is present in the linked cold analysis; main text uses an equivalent explicit limitation.
    if phrase=='不能把改善全部单独归因于冻结':
        check('改善不能全归于“冻结”' in text, 'no freeze-only causal claim')
    else: check(phrase in text,'scientific boundary: '+phrase)
check(final['selected_epochs']=={'A':32,'B':35,'C':23},'selection epochs and B fixed endpoint marker')
check(32*693==22176 and 23*693==15939 and (32-23)*693==6237,'best-pair unequal R updates')
for path, expected in BOUND.items(): check(sha(path)==expected,'input unchanged: '+Path(path).name)

review=dict(status='passed',format='freeze35_main_report_independent_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            manuscript=str(draft),manuscript_sha256=EXPECTED_DRAFT_SHA,source_sha256=BOUND,checks_count=len(CHECKS),checks=CHECKS,
            numerical_table_rows_verified=True,supplemental_A32_comparisons_verified=3,plot_sources_and_loss_rows_verified=True,
            learning_rate_checks={'pairwise_counts':pairwise,'total_pairwise_comparisons':sum(pairwise.values()),'triple_common_unique_updates':triple},
            actual_update_counts=update_counts,replay_gates=gates,inputs_unchanged=True,manuscript_modified=False,
            resolved_findings=[{'id':'LR_PAIRWISE_COUNT','original_draft_sha256':'7d51ec049b1f62f3dded15975618f4c7fb20e405bbc6168bde35b74a8c60fa60',
                                'issue':'10395 counts pairwise LR comparisons, not distinct triple-common updates','resolution':'Revised draft explicitly separates 10395 pairwise checks and 2079 triple-common updates'}],
            unresolved_findings=[],manual_scientific_review={
              'question1':'Correct: validation improves through R32 then falls at R35; fixed35 vs R26 test CI crosses zero; best32 vs R26 supplementary CI also crosses zero.',
              'question2':'Correct: fixed35 A-C shares R35/T200 and freeze20; estimates support this one-seed comparison, with H1-informed freeze point and selection limitations retained.',
              'question3':'Correct: B late freeze validation declines and test CI crosses zero; A35 beats new B35 but A35 vs old B32 CI crosses zero; A32 vs old B32 supplemental CI is positive.',
              'cold_diagnostic_summary':'Agrees with linked independently accepted five-model CPU report; not a new evaluation protocol.',
              'figures':'All three PNGs visually inspected; validation curves, hollow references, A32/C23 stars and actual continuation ranges agree with the manuscript.',
              'resource_backup_scope':'Scientific review does not replace the separate lifecycle/restore audit. Main manuscript distinguishes 66 state inspections from 9 real reloads, separate archive dependencies, and unverified platform billing.'},
            finalization_condition='Root may replace draft framing and add delivery/index/git bookkeeping while preserving the reviewed scientific content. Any scientific or numerical changes require a new review binding.')
out=Path(__file__).parent
(out/'MAIN_REPORT_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md=f'''# 主实验报告独立复审

复审通过。绑定修订稿 SHA-256：`{EXPECTED_DRAFT_SHA}`。共执行 {len(CHECKS)} 项收据、数值展示、曲线、计数与输入不变检查；本复审未修改草稿。

主指标、辅助聚合指标、候选覆盖、所有表内差值/相对变化/95%用户区间均与 A/B/C 分支及最终配对原收据逐行一致。A R32−R26、旧 A R20、旧 B R32 三项已经通过补充绑定重新核对原 early CPU 收据 SHA、预测端点 SHA、2000/2024 协议和区间；稿件正确写成复用已有区间。

三个研究问题回答与证据一致：固定 A35−R26 测试区间跨零；固定 A35−C35 在相同 R35/T200、R20 冻结条件下为正；B 晚期冻结未见改善。A35−旧 B32 区间跨零，与 A32−旧 B32 补充区间为正分别陈述，未混淆 checkpoint。单 seed、选模/冻结时点选择、A−B 的 T 时钟差异、best 对照 R 更新量差异及冻结反事实缺失均保留。

训练日志重新核对 A/B/C 新增 R 更新 6237/2079/20790、新增 T 更新 0/0/200、最终 R24255 与 T200/360/200；逐轮训练损失加权值和全部验证曲线点均与来源一致。三张 PNG 已目视检查。

唯一发现的措辞问题已修复：10395 是 A−C 6237、B−C 2079、A−B 2079 次学习率逐点核对之和；三组共同覆盖的不同更新点是2079个。修订稿已明确区分。

C 五个验证重放关口及 R20 的全71657条 Top5一致/指标差0、20个阶段轨迹收据吻合；原阈值1e-6/99.9%/身份100%未变。稿件明确该阈值是工程报警，不是科学等价或旧R20权重 bitwise 恢复证明。完整保存、CPU恢复与资源账务仍以其单独复审为准。

未发现未解决的科学或数值问题。根任务可将开头/结尾的草稿状态改为最终交付并补索引/Git信息；本次签收绑定科学内容不变，任何科学数值修改须重新绑定复核。
'''
(out/'MAIN_REPORT_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':'passed','checks':len(CHECKS),'sources':len(BOUND),'review_sha256':sha(out/'MAIN_REPORT_REVIEW.json'),'md_sha256':sha(out/'MAIN_REPORT_REVIEW.md')},ensure_ascii=False))
