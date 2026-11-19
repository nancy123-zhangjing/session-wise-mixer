"""Present existing audited statistics against original H0 best; no prediction/model run."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCE = ROOT / 'c_supplement_r1'
MODELS = ['H0_best_R32', 'H1_R20', 'H1_freeze_R32', 'C_best_R23']
PAIRS = [('H1_R20', 'H0_best_R32'), ('H1_freeze_R32', 'H0_best_R32'),
         ('H1_freeze_R32', 'C_best_R23')]
SEGMENTS = ['early', 'late', 'full']
GROUPS = ['never_seen', 'history_only', 'target_head', 'target_mid', 'target_tail', 'target_seen']
DISJOINT = GROUPS[:5]
NAMES = {'H0_best_R32': '原 H0 best R32', 'H1_R20': '旧 H1 R20',
         'H1_freeze_R32': '冻结 H1 R32', 'C_best_R23': 'C 自身 best R23'}
GROUP_NAMES = {'never_seen':'训练完全未见', 'history_only':'仅训练历史出现',
               'target_head':'热门暖商品', 'target_mid':'中频暖商品',
               'target_tail':'长尾暖商品', 'target_seen':'全部暖商品'}

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pct(value, baseline):
    return None if baseline == 0 else 100.0 * (value - baseline) / baseline

def fp(value):
    return '不可定义（基线为0）' if value is None else f'{value:+.2f}%'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                     '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                    ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    if (OUT / 'REPORT.md').exists():
        raise SystemExit('Existing presentation preserved; use a new output directory for a rerun.')
    sources = [SOURCE / f'MODEL_{m}.json' for m in MODELS]
    sources += [SOURCE / 'DECOMPOSITION.json', SOURCE / 'DATA_COMPOSITION.json',
                ROOT / 'PROTOCOL.md', ROOT / 'FINAL_REPORT.md', ROOT / 'STATUS.json']
    for m in MODELS[:3]:
        sources.append(ROOT / 'initial_r1' / f'MODEL_{m}.json')
    before = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources}
    models = {m: load(SOURCE / f'MODEL_{m}.json')['segments'] for m in MODELS}
    for m in MODELS[:3]:
        assert load(ROOT / 'initial_r1' / f'MODEL_{m}.json')['segments'] == models[m]
    decomposition = load(SOURCE / 'DECOMPOSITION.json')
    composition = load(SOURCE / 'DATA_COMPOSITION.json')
    rows = []

    def add(model, reference, segment, family, group, metric, k, value, baseline, rate=False):
        rows.append(dict(model=model, reference=reference, segment=segment, family=family,
                         group=group, metric=metric, k=k, model_value=value,
                         reference_value=baseline, absolute_change=value-baseline,
                         relative_change_pct=pct(value, baseline),
                         percentage_point_change=100*(value-baseline) if rate else None,
                         undefined_reason='zero_reference' if baseline == 0 else ''))

    for model, reference in PAIRS:
        for segment in SEGMENTS:
            x, y = models[model][segment], models[reference][segment]
            for metric, value in x['metrics'].items():
                add(model, reference, segment, 'overall', 'all', metric, '', value,
                    y['metrics'][metric], '/hr@' in metric or '/recall@' in metric)
            for group in GROUPS:
                assert x['classes'][group]['target_occurrences'] == y['classes'][group]['target_occurrences']
                for k in ['5', '20', '50']:
                    for metric, value in x['classes'][group]['metrics'][k].items():
                        add(model, reference, segment, 'target', group, metric, k, value,
                            y['classes'][group]['metrics'][k][metric], metric == 'target_hit_rate')
            for protocol in ['list5', 'aggregate50']:
                xx, yy = x['predictions'][protocol], y['predictions'][protocol]
                for metric in ['unique_recommended_items', 'top10_concentration', 'hhi',
                               'mean_log1p_train_target_frequency', 'coverage']:
                    add(model, reference, segment, 'recommendation', protocol, metric, '',
                        xx[metric], yy[metric], metric in ['top10_concentration', 'coverage'])
                for group, value in xx['class_shares'].items():
                    add(model, reference, segment, 'recommendation_share', group, protocol, '',
                        value, yy['class_shares'][group], True)

    with (OUT / 'RELATIVE_CHANGES.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    baseline = models['H0_best_R32']
    def overall_rows(metric):
        result = []
        for model in MODELS[:3]:
            row = [NAMES[model]]
            for segment in SEGMENTS:
                value = models[model][segment]['metrics'][metric]
                change = '基线' if model == 'H0_best_R32' else fp(pct(value, baseline[segment]['metrics'][metric]))
                row += [f'{value:.6f}', change]
            result.append(row)
        return result

    def group_lift_rows(metric):
        result=[]
        for group in ['target_seen', 'target_head', 'target_mid', 'target_tail', 'never_seen', 'history_only']:
            row=[GROUP_NAMES[group]]
            for model in ['H1_R20', 'H1_freeze_R32']:
                for segment in ['early', 'late']:
                    xx=models[model][segment]['classes'][group]['metrics']['5'][metric]
                    yy=baseline[segment]['classes'][group]['metrics']['5'][metric]
                    row.append(fp(pct(xx, yy)))
            result.append(row)
        return result

    parts=['# 相对原 H0 最佳模型：前后段及分组提升百分比',
      '2026-10-30。按用户要求，以原 H0 best R32/T360 为主参照；旧 H1 R20 和冻结 H1 R32 均与同一份 H0 best 比较。C 自身 best R23 只放补充表，不再把 C 固定 R32 作为这份说明的主基线。',
      '这是已验统计的展示补充：从保存的全精度 JSON 计算百分比，没有重读预测、模型推理、训练、重新 bootstrap、选模或改变测试协议。原统计、报告、完成收据均保持不变。',
      '正负变化% =（H1数值 − 同时段H0 best数值）÷ 同时段H0 best数值 × 100%。性能指标称相对提升；推荐份额、集中度和覆盖数只称相对变化，不把增加自动视为改善。基线为0时百分比不可定义；两者都是0也不写0%。',
      '所有数字为 validation。前段38,644条/193,220次目标出现，后段33,013条/165,065次目标出现；沿用旧时间切点。原H0 best训练量T360，H1为T200，因此本表回答相对最佳基线的实际效果，不单独识别Mixer因果作用。',
      '## 1. 整体效果：以原 H0 best 为分母',
      '单份五件列表 NDCG@5：',
      table(['模型','前段NDCG','相对H0','后段NDCG','相对H0','全量NDCG','相对H0'],overall_rows('list/ndcg@5')),
      '列表 HR@5：',
      table(['模型','前段HR','相对H0','后段HR','相对H0','全量HR','相对H0'],overall_rows('list/hr@5')),
      '原多候选列表聚合 NDCG@50（不等于生成一份50件列表）：',
      table(['模型','前段NDCG50','相对H0','后段NDCG50','相对H0','全量NDCG50','相对H0'],overall_rows('aggregate/ndcg@50')),
      '旧H1从前段领先变成后段落后；冻结H1前后段都领先，但后段提升幅度更小。原H0前后段分母不同，不能把两段相对提升率之差当成NDCG绝对分数之差。',
      '## 2. 各类目标：相对 H0 best 的提升',
      '先看逐目标命中率@5。分母为该类目标出现次数，不是整份列表HR。暖商品是曾作为训练目标的商品；按训练目标频次排序，前10%为热门、随后40%为中频、其余50%为长尾。',
      table(['目标类别','旧H1 前段','旧H1 后段','冻结H1 前段','冻结H1 后段'],group_lift_rows('target_hit_rate')),
      '再看各类对整体 NDCG@5 的贡献相对提升。贡献使用原五目标IDCG，按全段样本平均；绝对贡献可相加，下面的相对百分比不可直接相加。',
      table(['贡献来源','旧H1 前段','旧H1 后段','冻结H1 前段','冻结H1 后段'],group_lift_rows('ndcg_contribution')),
      '在@5，三模型的完全未见、仅历史、长尾类别均为0命中、0贡献，因此这些行不能算相对提升。@20/@50的非零和零基线情况均按原值保留在CSV，不能把@5零命中扩展到所有候选深度。',
      '## 3. 哪部分对应优势缩小',
      '下面按可相加的NDCG绝对贡献计算：（后段H1−H0）−（前段H1−H0）。负值表示该组对应优势缩小，正值表示该组抵消一部分缩小；这是描述性分解，不是因果归因。']
    gaps=[]
    gap_rows=[]
    decomp_rows=[]
    for model in ['H1_R20','H1_freeze_R32']:
        total=sum((models[model]['late']['classes'][g]['metrics']['5']['ndcg_contribution']-baseline['late']['classes'][g]['metrics']['5']['ndcg_contribution'])-(models[model]['early']['classes'][g]['metrics']['5']['ndcg_contribution']-baseline['early']['classes'][g]['metrics']['5']['ndcg_contribution']) for g in DISJOINT)
        for group in DISJOINT:
            effects={s:models[model][s]['classes'][group]['metrics']['5']['ndcg_contribution']-baseline[s]['classes'][group]['metrics']['5']['ndcg_contribution'] for s in ['early','late']}
            change=effects['late']-effects['early']; share=100*change/total
            gaps.append(dict(model=model,group=group,**effects,change=change,share_of_advantage_shrink_pct=share))
            if group in ['target_head','target_mid']:
                gap_rows.append([NAMES[model],GROUP_NAMES[group],f"{effects['early']:+.8f}",f"{effects['late']:+.8f}",f'{change:+.8f}',f'{share:.2f}%'])
        d=decomposition[f'{model}_minus_H0_best_R32']['5']
        assert abs(total-d['observed_advantage_change']) < 1e-12
        decomp_rows.append([NAMES[model],f"{d['composition_change']:+.8f}",f"{100*d['composition_change']/total:+.2f}%",f"{d['within_class_change']:+.8f}"])
    write_json(OUT/'ADVANTAGE_CHANGE_BY_GROUP.json',gaps)
    parts += [table(['模型','类别','前段优势','后段优势','优势变化','占整体缩小量'],gap_rows),
       '冻结H1的中频商品对应超过100%的优势缩小，是因为热门商品优势扩大，抵消了其中一部分；不是计算错误。旧H1则同时存在热门商品劣势扩大和中频商品优势减弱。',
       '沿用原五类对称分解，将类别占比变化与类别内部效果变化分开：',
       table(['模型对原H0 best','类别组成变化项','占优势缩小量','类别内部变化项'],decomp_rows),
       '此前0.31%属于冻结H1对C固定R32；换成原H0 best后应为约0.83%，不能沿用旧分母。旧H1组成项为正，反而抵消约7.24%的下降。这里的组成包括冷暖与频次类别，不是冷启动因果贡献。',
       '## 4. 推荐热门与集中度：同时列原值和相对变化',
       '训练热门占比=固定802件热门商品占有效推荐位置比例；前十集中度=每个模型/时段自身推荐最多十件商品的份额。两者都排除PAD；覆盖数在同一时段比较。']
    bias_rows=[]
    for segment in ['early','late']:
        for title,field in [('训练热门占比','head'),('前十件集中度','top10_concentration'),('不同推荐商品数','unique_recommended_items')]:
            def get(model):
                q=models[model][segment]['predictions']['list5']
                return q['class_shares']['target_head'] if field=='head' else q[field]
            base=get('H0_best_R32')
            fmt=lambda v: str(v) if field=='unique_recommended_items' else f'{100*v:.2f}%'
            bias_rows.append(['前段' if segment=='early' else '后段',title,fmt(base),fmt(get('H1_R20')),fp(pct(get('H1_R20'),base)),fmt(get('H1_freeze_R32')),fp(pct(get('H1_freeze_R32'),base))])
    parts += [table(['时段','指标','H0 best','旧H1','相对变化','冻结H1','相对变化'],bias_rows),
      '两份H1的前十件推荐在两段均为训练中频商品。数据支持H1比原H0 best更集中于少数商品；不支持H1比原H0 best更多推训练最高频商品。集中是否造成时间段收益变化尚未证实；H1自身后段集中度低于前段，不能写成越晚越集中。',
      '## 5. 冷目标占比的前后变化',
      '此表分母为该时间段全部目标出现次数，与模型无关。变化%按前段占比作分母；百分点是占比直接相减。']
    comp_rows=[]
    for group in GROUPS:
        e=composition['early']['classes'][group];l=composition['late']['classes'][group]
        comp_rows.append([GROUP_NAMES[group],f"{100*e['target_share']:.2f}%",f"{100*l['target_share']:.2f}%",fp(pct(l['target_share'],e['target_share'])),f"{100*(l['target_share']-e['target_share']):+.2f}"])
    parts += [table(['目标类别','前段占比','后段占比','占比相对变化','变化百分点'],comp_rows),
      '全部暖商品=热门+中频+长尾，汇总行不能和子类重复相加。后段冷目标更多是事实，但本次@5相对收益缩小主要对应暖中频商品的变化，不能从冷占比上升直接推导主要原因。',
      '## 6. C 自身最佳模型的补充比较',
      'C按原共同节点规则选出的best是R23，不是固定R32。下表只作补充；冻结H1 R32与C best R23推荐器轮次不同。']
    c_rows=[]
    for segment in SEGMENTS:
        v=models['H1_freeze_R32'][segment]['metrics']['list/ndcg@5'];b=models['C_best_R23'][segment]['metrics']['list/ndcg@5']
        c_rows.append([{'early':'前段','late':'后段','full':'全量'}[segment],f'{b:.6f}',f'{v:.6f}',fp(pct(v,b))])
    parts += [table(['时段','C best R23','冻结H1 R32','相对提升'],c_rows),
       '## 7. 阅读结论与证据边界',
       '以原H0 best为参照，旧H1后段已落后；冻结后续训把后段恢复到领先，但中频商品的相对增益仍明显收窄。热门商品效果在冻结后反而更好，训练热门推荐占比也低于H0，因此“更多推训练热门”不能解释这组相对差异。少数中频商品集中推荐是需要进一步排查的现象，不是已证明的原因。',
       '主表全部是点估计，不能仅凭每格正负判定统计显著。沿用原用户配对区间，不新建检验；单seed、验证集事后诊断和原H0/H1训练量不同的限制保留。本次没有改变数据划分，也没有执行session留出实验。',
       '全精度原值、绝对差、相对变化%、适用的百分点变化和零基线原因见 RELATIVE_CHANGES.csv；来源SHA见 PROVENANCE.json。原独立审核的完整报告仍保留在上一级 FINAL_REPORT.md。']
    (OUT/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    after={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources}
    assert before==after
    write_json(OUT/'PROVENANCE.json',dict(format='best_h0_reference_presentation_v1',
       generated_utc=datetime.now(timezone.utc).isoformat(), user_requested_primary_reference='H0_best_R32',
       supplementary_reference='C_best_R23', source_sha256=before, source_unchanged=True,
       relative_formula='100 * (model_value - reference_value) / reference_value',
       zero_reference='null / blank CSV field with undefined_reason=zero_reference, including 0/0',
       metric_rows=len(rows), no_model_inference=True, no_raw_prediction_reanalysis=True,
       no_training=True, no_gpu=True, no_bootstrap_rerun=True, original_protocol_unchanged=True,
       outputs={p.name:sha(p) for p in [OUT/'REPORT.md',OUT/'RELATIVE_CHANGES.csv',OUT/'ADVANTAGE_CHANGE_BY_GROUP.json']},
       script_sha256=sha(Path(__file__))))
    print(json.dumps({'output':str(OUT),'metric_rows':len(rows),'source_unchanged':True}))

if __name__=='__main__':
    main()
