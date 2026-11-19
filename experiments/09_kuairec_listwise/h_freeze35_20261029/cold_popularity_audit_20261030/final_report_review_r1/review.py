"""Independent report/table audit. Stdlib only; never imports report builders."""
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'c_supplement_r1'
TABLES = ROOT / 'c_supplement_tables_r1'
EXPECTED_REPORTS = {
    'FINAL_REPORT.md': 'acb366d436260caa459845a6156fc62d1944d83410dcdcfa14e85894bb697225',
    'REPORT_C_SUPPLEMENT.md': '2ba7954d97a68924ce4c5a9acd889dbbac0ed61935836855e8b554f7a4bfabb0',
}
MODELS = {'旧 H0 best R32': 'H0_best_R32', '旧 H1 R20': 'H1_R20',
          '冻结 H1 R32': 'H1_freeze_R32', '冻结 C R32': 'C_freeze_R32', 'C best R23': 'C_best_R23'}
SEGMENTS = {'前段': 'early', '后段': 'late', '全量': 'full'}
PAIR_NAMES = {'旧 H1 − 旧 H0': 'H1_R20_minus_H0_best_R32',
              '冻结 H1 − 旧 H0': 'H1_freeze_R32_minus_H0_best_R32',
              '冻结 H1 − 旧 H1 R20': 'H1_freeze_R32_minus_H1_R20',
              '冻结 H1 R32 − C R32': 'H1_freeze_R32_minus_C_freeze_R32',
              'H1 R32 − C R32': 'H1_freeze_R32_minus_C_freeze_R32',
              '冻结 H1 R32 − C best R23': 'H1_freeze_R32_minus_C_best_R23',
              'H1 R32 − C best R23': 'H1_freeze_R32_minus_C_best_R23'}
FINE = ['never_seen', 'history_only', 'target_head', 'target_mid', 'target_tail']

def check(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def close(a, b, label):
    check(math.isclose(a, b, rel_tol=1e-12, abs_tol=2e-14), label)

def tables(text):
    result, active = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            active.append([c.strip() for c in line.strip('|').split('|')])
        elif active:
            check(len(active) > 2, 'Malformed Markdown table')
            result.append((active[0], active[2:])); active = []
    return result

def main():
    check(not (HERE / 'REVIEW.json').exists() and not (HERE / 'REVIEW.md').exists(), 'Preserve existing audit outputs')
    monitored = {}
    def bind(path, expected=None):
        path = Path(path)
        value = sha(path)
        if expected is not None:
            check(value == expected, 'SHA mismatch: ' + str(path))
        monitored[str(path)] = value
        return value
    for name, digest in EXPECTED_REPORTS.items():
        bind(ROOT / name, digest)
    complete, initial = read(DATA / 'COMPLETE.json'), read(ROOT / 'initial_r1/COMPLETE.json')
    verify = read(ROOT / 'VERIFY_C_SUPPLEMENT.json')
    export = read(TABLES / 'EXPORT_RECEIPT.json')
    build = read(ROOT / 'C_REPORT_BUILD_RECEIPT.json')
    request, result = read(DATA / 'REQUEST.json'), read(DATA / 'RESULTS.json')
    model = {name: read(DATA / ('MODEL_' + name + '.json')) for name in request['models']}
    comp, decomposition, paired = [read(DATA / name) for name in ('DATA_COMPOSITION.json', 'DECOMPOSITION.json', 'PAIRED_BOOTSTRAP.json')]
    for file in ['PROTOCOL.md', 'VERIFY.json', 'REPORT_REVIEW.json', 'C_EXTRA_MODELS.json', 'C_REPORT_BUILD_RECEIPT.json']:
        bind(ROOT / file)
    check(complete['status'] == initial['status'] == verify['status'] == export['status'] == 'passed', 'Accepted evidence statuses')
    check(complete['only_cpu_statistics'] is True and complete['training_or_inference'] is False and complete['full_goal_complete'] is False, 'Analysis scope')
    bind(DATA / 'COMPLETE.json', verify['analysis_complete_sha256'])
    bind(ROOT / 'VERIFY_C_SUPPLEMENT.json', export['verification_sha256'])
    bind(DATA / 'COMPLETE.json', export['complete_sha256'])
    bind(ROOT / 'verify.py', verify['verifier_sha256'])
    bind(ROOT / 'analyze.py', complete['script_sha256'])
    bind(ROOT / 'export_c_supplement_tables.py', export['script_sha256'])
    bind(ROOT / 'build_c_reports.py', build['script_sha256'])
    for directory, record in [(DATA, complete), (ROOT / 'initial_r1', initial)]:
        for name, digest in record['files'].items():
            bind(directory / name, digest)
    for name, digest in build['initial_artifacts_unchanged'].items():
        bind(ROOT / name, digest)
    for name, digest in build['files'].items():
        bind(ROOT / name, digest)
    for path, digest in result['source_sha256'].items():
        bind(path, digest)
    check(len(initial['files']) == build['initial_files_checked'] == 15, 'Initial inventory count')
    # Initial arrays, model summaries and target grouping are unchanged in the five-model run.
    shared_initial = ['DATA_COMPOSITION.json', 'shared_arrays.npz', 'train_item_counts.csv']
    shared_initial += [prefix + name + suffix for name in ['H0_best_R32', 'H1_R20', 'H1_freeze_R32']
                       for prefix, suffix in [('MODEL_', '.json'), ('arrays_', '.npz')]]
    for name in shared_initial:
        check(initial['files'][name] == complete['files'][name], 'Initial reused source differs: ' + name)
    check(request['models'] == list(MODELS.values()), 'Five-model roster/order')
    check(request['cutoff_us'] == 1598745441188500 and request['train_rows'] == 709443, 'Cutoff or training source changed')
    check(request['bootstrap']['repeats'] == 2000 and request['bootstrap']['seed'] == 2024, 'Bootstrap protocol')
    check(request['bootstrap']['temporal_contrast'] == 'same global-user draw for both segments', 'Temporal bootstrap independence error')
    check(result['split'] == 'validation' and result['population'] == {'early': 38644, 'late': 33013, 'full': 71657}, 'Population scope')
    check([comp[s]['users'] for s in ('early', 'late', 'full')] == [4011, 3827, 4733], 'User populations')
    check(verify['catalog_class_counts'] == [1914, 799, 802, 3206, 4007] and verify['train_rows_independently_counted'] == 709443, 'Training classes')
    check(verify['full_source_validation_rows_checked'] == 71657 and verify['max_array_metric_abs_error'] == 0, 'Array verifier scope')
    check(len(request['metric_keys']) == 12, 'Whole metric count')
    for name in request['models']:
        v = verify['models'][name]
        check(v['array_rows_checked'] == v['raw_rows_counted'] == 71657 and v['raw_rows_checked'] == 1027, 'Raw sample/all-array scope')
        check(max(model[name]['baseline_reproduction_absolute_error'].values()) == 0, 'Overall source reproduction')
    check(verify['bootstrap']['all_model_feature_draws_independently_recomputed'] is True and verify['bootstrap']['temporal_contrast_same_global_user_draw'] is True, 'Bootstrap verified scope')
    check(format(verify['max_bootstrap_draw_abs_error'], '.2e') == '1.84e-16', 'Reported maximum bootstrap error')
    check(verify['read_only_source_hashes_and_output_hashes_verified'] is True, 'Source protection receipt')

    # Every row/column in all seven exported tables is compared with locked JSON.
    csv_counts = {}
    for filename, digest in export['files'].items():
        bind(TABLES / filename, digest)
        records = list(csv.DictReader((TABLES / filename).open(encoding='utf-8-sig', newline='')))
        kind = Path(filename).stem; csv_counts[kind] = len(records)
        identities = []
        for row in records:
            if kind == 'METRICS':
                identities.append((row['model'], row['segment']))
                expected = {'model': row['model'], 'segment': row['segment'], **model[row['model']]['segments'][row['segment']]['metrics']}
            elif kind == 'TARGET_CLASSES':
                identities.append((row['model'], row['segment'], row['group'], row['k']))
                group = model[row['model']]['segments'][row['segment']]['classes'][row['group']]
                expected = {k: row[k] for k in ('model', 'segment', 'group', 'k')}
                expected.update(target_occurrences=group['target_occurrences'], **group['metrics'][row['k']])
            elif kind in ('RECOMMENDATIONS', 'TOP10_ITEMS'):
                pred = model[row['model']]['segments'][row['segment']]['predictions'][row['protocol']]
                identities.append((row['model'], row['segment'], row['protocol'], row.get('rank')))
                expected = {k: row[k] for k in ('model', 'segment', 'protocol')}
                if kind == 'TOP10_ITEMS':
                    expected.update(rank=row['rank'], **pred['top10_items'][int(row['rank']) - 1])
                else:
                    expected.update({k: v for k, v in pred.items() if k not in ('top10_items', 'class_shares')})
                    expected.update({'share_' + k: v for k, v in pred['class_shares'].items()})
            elif kind == 'PAIRED_BOOTSTRAP':
                identities.append((row['segment'], row['pair'], row['feature']))
                expected = {k: row[k] for k in ('segment', 'pair', 'feature')}
                expected.update(paired[row['segment']][row['pair']][row['feature']])
            else:
                decomp = decomposition[row['pair']][row['k']]
                identities.append((row['pair'], row['k'], row.get('fine_class')))
                expected = {k: row[k] for k in ('pair', 'k')}
                if kind == 'DECOMPOSITION':
                    expected.update({k: v for k, v in decomp.items() if k != 'classes'})
                else:
                    expected.update(next(v for v in decomp['classes'] if v['fine_class'] == row['fine_class']))
            check(row == {k: str(v) for k, v in expected.items()}, 'Export row mismatch: ' + kind + str(identities[-1]))
        check(len(set(identities)) == len(identities), 'Duplicate export rows')
    check(csv_counts == export['table_rows'] and sum(csv_counts.values()) == 573 and len(csv_counts) == 7, '7 tables/573 rows')

    # Recompute symmetric decomposition directly from grouped counts/gains.
    scale = 5 / sum(1 / math.log2(rank + 1) for rank in range(1, 6))
    for pair, ks in decomposition.items():
        treatment, baseline = pair.split('_minus_')
        for k, saved in ks.items():
            composition_sum, within_sum = 0.0, 0.0
            for group in FINE:
                weights = [comp[s]['classes'][group]['target_share'] for s in ('early', 'late')]
                effects = []
                for s in ('early', 'late'):
                    a, b = [model[m]['segments'][s]['classes'][group] for m in (treatment, baseline)]
                    check(a['target_occurrences'] == b['target_occurrences'] > 0, 'Group denominator')
                    effects.append((a['metrics'][k]['discounted_gain_sum'] - b['metrics'][k]['discounted_gain_sum']) / a['target_occurrences'])
                c = (weights[1] - weights[0]) * (effects[1] + effects[0]) * scale / 2
                w = (weights[1] + weights[0]) * (effects[1] - effects[0]) * scale / 2
                row = next(v for v in saved['classes'] if v['fine_class'] == group)
                close(c, row['composition_ndcg_change'], 'Class composition algebra')
                close(w, row['within_class_ndcg_change'], 'Class within algebra')
                composition_sum += c; within_sum += w
            metric = ('list/' if k == '5' else 'aggregate/') + 'ndcg@' + k
            observed = sum(sign * (model[treatment]['segments'][s]['metrics'][metric] - model[baseline]['segments'][s]['metrics'][metric]) for s, sign in [('early', -1), ('late', 1)])
            close(composition_sum, saved['composition_change'], 'Composition total')
            close(within_sum, saved['within_class_change'], 'Within total')
            close(composition_sum + within_sum, observed, 'Descriptive closure')
            check(saved['causal'] is False, 'Causal decomposition claim')

    # Verify every numerical Markdown row, including source rounding and CIs.
    markdown_checks = {}
    for filename in EXPECTED_REPORTS:
        text = (ROOT / filename).read_text(encoding='utf-8')
        numeric_rows, skipped_tables = 0, []
        bias_table = 0
        for header, tablerows in tables(text):
            if header[0] in ('问题', '模型') and (header[0] == '问题' or header[1] == 'Mixer'):
                skipped_tables.append(header); continue
            for row in tablerows:
                expected = None
                if header[0] == '类别':
                    group = {'训练完全未见': 'never_seen', '仅训练历史出现': 'history_only', '曾作为训练目标': 'target_seen', 'target head': 'target_head', 'target mid': 'target_mid', 'target tail': 'target_tail'}[row[0]]
                    a, b = [comp[s]['classes'][group] for s in ('early', 'late')]
                    expected = [row[0], str(a['target_occurrences']), f"{a['target_share']:.2%}", str(b['target_occurrences']), f"{b['target_share']:.2%}"]
                elif header[0] == '比较':
                    pair = PAIR_NAMES[row[0]]; d = decomposition[pair]['5']; interval = paired['late_minus_early'][pair]['list/ndcg@5/overall']
                    expected = [row[0], f"{d['observed_advantage_change']:+.8f}", f"[{interval['ci_low']:+.6f}, {interval['ci_high']:+.6f}]", f"{d['composition_change']:+.8f}", f"{d['within_class_change']:+.8f}"]
                    # Decomposition tables omit explicit plus for negative CIs only.
                elif header[0].startswith('比较（'):
                    pair = PAIR_NAMES[row[0]]; s = SEGMENTS[row[1]]
                    a, b = [paired[s][pair][k] for k in ('list/ndcg@5/overall', 'aggregate/ndcg@50/overall')]
                    expected = row[:2] + [f"{a['delta']:+.6f}", f"{a['delta']/a['baseline']:+.2%}", f"[{a['ci_low']:+.6f}, {a['ci_high']:+.6f}]", f"{b['delta']:+.6f}", f"[{b['ci_low']:+.6f}, {b['ci_high']:+.6f}]"]
                elif header[0] == '模型' and header[1].startswith('前段：'):
                    name = MODELS[row[0]]
                    expected = [row[0]] + [' / '.join(str(model[name]['segments'][s]['classes'][g]['metrics']['50']['target_hits']) for g in ('never_seen', 'history_only', 'target_tail')) for s in ('early', 'late')]
                elif header[2] == '暖目标命中率@5':
                    segment = model[MODELS[row[0]]]['segments'][SEGMENTS[row[1]]]; classes = segment['classes']
                    expected = row[:2] + [f"{classes[g]['metrics']['5']['target_hit_rate']:.3%}" for g in ('target_seen', 'target_head', 'target_mid')] + [f"{classes[g]['metrics']['5']['ndcg_contribution']:.6f}" for g in ('target_head', 'target_mid')]
                elif header[2] == '训练 head 推荐份额':
                    s = SEGMENTS[row[1]]; p = model[MODELS[row[0]]]['segments'][s]['predictions']['list5' if bias_table == 0 else 'aggregate50']
                    expected = row[:2] + [f"{p['class_shares']['target_head']:.2%}", f"{p['top10_concentration']:.2%}", str(p['unique_recommended_items']), f"{p['hhi']:.5f}", f"{p['real_item_slots']/comp[s]['samples']:.2f}"]
                elif header[2] == 'NDCG@5':
                    m = model[MODELS[row[0]]]['segments'][SEGMENTS[row[1]]]['metrics']
                    expected = row[:2] + [f'{m[k]:.6f}' for k in ('list/ndcg@5', 'list/hr@5', 'aggregate/ndcg@50')]
                check(expected is not None and row == expected, 'Markdown row mismatch: ' + filename + ' ' + str(row) + ' expected ' + str(expected))
                numeric_rows += 1
            if len(header) > 2 and header[2] == '训练 head 推荐份额':
                bias_table += 1
        markdown_checks[filename] = {'numeric_rows_checked': numeric_rows, 'qualitative_tables_manually_reviewed': skipped_tables}
        check('抽查 1,027 条' in text and '全量计数' in text and '全部数组' in text, 'Verifier scope language')
        check('不是因果中介分析' in text and '没有同 R32' in text and '验证集上的事后诊断' in text, 'Causal/selection boundaries')
        check('稳健性检查' in text and 'session 留出' in text and '未做多重检验校正' in text, 'Session/multiple-comparison boundary')
    check(markdown_checks['FINAL_REPORT.md']['numeric_rows_checked'] == 70 and markdown_checks['REPORT_C_SUPPLEMENT.md']['numeric_rows_checked'] == 38, 'Markdown coverage count')
    fair = decomposition['H1_freeze_R32_minus_C_freeze_R32']['5']
    best = decomposition['H1_freeze_R32_minus_C_best_R23']['5']
    fair_fraction = fair['composition_change'] / fair['observed_advantage_change']
    best_fraction = best['composition_change'] / best['observed_advantage_change']
    close(fair_fraction, build['composition_fraction_fair_r32'], '0.31 percent derived ratio')
    close(best_fraction, build['composition_fraction_selected_best'], '1.19 percent derived ratio')
    check(format(fair_fraction, '.2%') == '0.31%' and format(best_fraction, '.2%') == '1.19%', 'Rounded narrative fractions')
    fair_classes = {r['fine_class']: r['within_class_ndcg_change'] for r in fair['classes']}
    check(fair_classes['target_mid'] < fair['within_class_change'] < 0 < fair_classes['target_head'], 'Mid dominance and head offset')
    for name in request['models']:
        for s in ('early', 'late'):
            for g in ('never_seen', 'history_only', 'target_tail'):
                values = model[name]['segments'][s]['classes'][g]['metrics']['5']
                check(values['target_hits'] == values['discounted_gain_sum'] == values['ndcg_contribution'] == 0, 'Five-model cold/tail zeros')
    for h1 in ('H1_R20', 'H1_freeze_R32'):
        for s in ('early', 'late'):
            check({v['fine_class'] for v in model[h1]['segments'][s]['predictions']['list5']['top10_items']} == {'target_mid'}, 'H1 top10 class claim')
    h1, c32, cbest = [model[n]['segments'] for n in ('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23')]
    for s in ('early', 'late'):
        a, b = h1[s]['predictions']['list5'], c32[s]['predictions']['list5']
        check(a['class_shares']['target_head'] < b['class_shares']['target_head'] and a['top10_concentration'] > b['top10_concentration'] and a['hhi'] > b['hhi'], 'Head vs concentration contrast')
    check(h1['early']['predictions']['list5']['top10_concentration'] < cbest['early']['predictions']['list5']['top10_concentration'], 'Cbest early exception')
    check(h1['late']['predictions']['list5']['top10_concentration'] < h1['early']['predictions']['list5']['top10_concentration'], 'No increasing-with-time concentration claim')
    late = paired['late']['H1_freeze_R32_minus_H1_R20']['list/ndcg@5/overall']
    check(f"{late['delta']/late['baseline']:.2%}" == '14.46%', 'Own-start late improvement')
    temporal = paired['late_minus_early']['H1_freeze_R32_minus_H1_R20']['list/ndcg@5/overall']
    check(temporal['ci_low'] < 0 < temporal['ci_high'], 'Improvement not specifically late')
    for name, expected in monitored.items():
        check(sha(name) == expected, 'Source changed during review: ' + name)
    check('torch' not in sys.modules, 'No Torch in review')
    report = {
        'status': 'passed', 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Final Chinese reports, seven presentation tables, source integrity, derived decomposition arithmetic, and wording/evidence boundaries; original independent statistical VERIFY retained unchanged.',
        'reports': EXPECTED_REPORTS, 'script_sha256': sha(__file__),
        'evidence_sha256_before_after_verified': monitored, 'initial_files_checked_unchanged': len(initial['files']),
        'initial_shared_arrays_and_model_summaries_identical': shared_initial,
        'csv_rows': csv_counts, 'csv_total_rows': 573, 'all_csv_cells_match_locked_JSON': True,
        'markdown_checks': markdown_checks, 'all_108_numeric_rows_match_source_rounding': True,
        'independent_algebra': {'five_comparisons_each_at_5_20_50_recomputed': True,
                                'fair_R32_absolute_advantage_change': fair['observed_advantage_change'],
                                'fair_R32_composition_change': fair['composition_change'],
                                'fair_R32_within_change': fair['within_class_change'],
                                'fair_R32_composition_fraction': fair_fraction,
                                'selected_best_composition_fraction': best_fraction,
                                'fair_R32_within_by_class': fair_classes},
        'semantic_findings': [
            {'topic': '0.31%', 'status': 'passed', 'basis': 'Ratio of fixed five-class composition term to absolute advantage change; descriptive and expressly noncausal, not all possible cold-start effects.'},
            {'topic': 'warm mid dominance', 'status': 'passed', 'basis': 'Mid within term -0.001994107150027891; head offsets +0.00012677870027352385. H1 warm/mid conditional hit rates decline while head rises.'},
            {'topic': 'head vs top10', 'status': 'passed', 'basis': 'H1 has lower training-head share but greater list5 concentration/HHI than matched C32 in both segments. H1 list5 top10 are all target_mid; Cbest early exception and lower late-than-early concentration are disclosed.'},
            {'topic': 'session holdout', 'status': 'passed', 'basis': 'Stated as a conditional future robustness question, not tested here or established repair. Reports keep original temporal split and do not infer that changing split resolves temporal generalization.'},
            {'topic': 'freeze causality', 'status': 'passed', 'basis': 'Extra R training accompanies freezing; no equal-R32 continued-tokenizer counterfactual, one seed, posthoc selected validation, unequal R32-vs-R23 best and multiple-comparison limitations stated.'},
            {'topic': 'independent verifier coverage', 'status': 'passed', 'basis': 'All 71,657 saved-array rows/model and bootstrap draws recomputed; raw aggregation independently sampled 1,027/model, all raw rows counted. Reports do not claim full second-implementation aggregation of every original recommendation.'},
        ],
        'source_statistical_verification': {'array_rows_per_model': 71657, 'raw_rows_sampled_per_model': 1027,
                                          'raw_rows_counted_per_model': 71657, 'models': 5,
                                          'max_array_metric_abs_error': verify['max_array_metric_abs_error'],
                                          'max_bootstrap_draw_abs_error': verify['max_bootstrap_draw_abs_error']},
        'review_boundaries': {'raw_prediction_reaggregation_rerun_by_this_review': False,
                             'bootstrap_draw_generation_rerun_by_this_review': False,
                             'source_model_or_training_or_inference_performed': False,
                             'network_or_GPU_performed': False, 'received_reports_modified': False,
                             'status_or_notifications_modified': False, 'full_goal_complete': False},
        'blocking_findings': [], 'nonblocking_findings': []}
    md = '''# 冷启动与热门偏置终稿独立审核

结论：**两份受审报告通过，未发现需修改的问题**。此审核不修改报告、统计来源、STATUS 或 NOTIFICATIONS。

已逐单元格核对 C 补充 7 张 CSV（573 行），逐行核对 `FINAL_REPORT.md` 70 行、`REPORT_C_SUPPLEMENT.md` 38 行数值表，包括相对变化、用户区间及四舍五入。重新核验初版 15 项输出、补充统计全部清单、来源 SHA、VERIFY 绑定和报告生成收据，前后不变。五组比较的 @5/@20/@50 对称分解另按目标次数及 DCG 原值重算，组成项、组内项与总体差值闭合。

- 同 R32/T200 比较：绝对优势变化 −0.0018731801248966376；组成项 −0.000005851675142257205，占比 0.3123925491%，报告写 0.31% 正确。各自 best 的 1.19% 也正确。这是固定粗分层下的描述性占比，不是冷启动因果效应的比例。
- mid 组内项 −0.001994107150027891，head 抵消 +0.00012677870027352385；“mid 主导”有数值支持。暖目标、mid 条件命中率下降，head 条件命中率上升的例外已说明，未把所有暖目标概括为下降。
- H1 的训练 head 推荐份额低于公平 C R32，但 @5 的 top10 集中度与 HHI 更高。H1 @5 top10 均为训练 mid；相对 C best 的前段例外和 H1 自身后段较不集中都已说明，没有将集中度偷换成训练热门偏置。
- 每用户最后两个 session 留出只是未来另一个评估问题的稳健性检查。报告没有说已测试该切分、它一定改善、或冷启动已经被排除；不优先替换原划分属于有边界的研究判断。
- 冻结与额外 R 续训同时发生、单 seed、事后验证选模、best 更新量不等、未校正多重比较等限制均保留，未宣称稳定的 Mixer 或冻结因果收益。

独立统计验收范围表述准确：五模型全部已保存数组各 71,657 行及 2,000 次 bootstrap 值由 `verify.py` 复算；每模型原始候选聚合另抽查 1,027 行，并全量计数 71,657 行。**不是每条原始推荐都经过第二实现聚合核验**。本次文字/数值审核直接读取已锁定统计与收据，没有重跑原始推荐聚合、bootstrap、模型、训练或网络。

完整逐文件 SHA、表格覆盖、描述性分解与语义边界见 `REVIEW.json`。受审报告 SHA：

- FINAL_REPORT.md：`acb366d436260caa459845a6156fc62d1944d83410dcdcfa14e85894bb697225`
- REPORT_C_SUPPLEMENT.md：`2ba7954d97a68924ce4c5a9acd889dbbac0ed61935836855e8b554f7a4bfabb0`

本项仅验收已授权 CPU 冷启动/热门偏置报告，不单独宣告原 Goal 的 Git、完整交付或全部验收完成。
'''
    with (HERE / 'REVIEW.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(md)
    report['review_markdown_sha256'] = sha(HERE / 'REVIEW.md')
    with (HERE / 'REVIEW.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'status': 'passed', 'csv_rows': 573, 'markdown_rows': 108,
                      'fair_composition_fraction': fair_fraction, 'review_sha256': sha(HERE / 'REVIEW.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
