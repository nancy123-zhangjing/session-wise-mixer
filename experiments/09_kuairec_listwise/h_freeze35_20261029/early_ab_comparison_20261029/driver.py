"""Early supplemental A-minus-B fixed-R35 comparison; CPU and local files only.

Reuses accepted branch audits. Only the unchanged paired_comparison function is
run; its internal metric replay is part of pairing, not a new standalone audit.
This does not replace final all-ready analysis or complete the experiment Goal.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
TASK = OUT.parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
CORE = ROOT / 'scripts/listwise_diffgrm/analysis.py'
CORE_SHA = '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
REPORT_SHAS = {
    'A': '66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937',
    'B': '7596546bb014f56b69d0a574d9cfd3a991e267bf667af6b80603cc6f48361a7e',
}
POPULATION = {'test': (53194, 4153), 'validation': (71657, 4733)}
PRIMARY = 'list/ndcg@5'
FEATURED = (PRIMARY, 'list/hr@5', 'list/recall@5', 'aggregate/ndcg@50')
PROTOCOL = dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                target_relevance='unordered_set_of_five_future_distinct_items',
                aggregation='list_score_order_then_item_position_first_occurrence')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    path = OUT / name
    need(path.parent == OUT and not path.exists(), 'Output must be a new direct child')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def authenticate():
    inputs, reports, points = {}, {}, {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        need(path.is_relative_to(ROOT), 'Input is outside the local workspace')
        observed = sha(path)
        need(expected is None or observed == expected, 'Source SHA mismatch: ' + str(path))
        need(str(path) not in inputs or inputs[str(path)] == observed, 'Input changed during authentication')
        inputs[str(path)] = observed
        return observed

    bind(__file__)
    bind(CORE, CORE_SHA)
    bind(TASK / 'GOAL_OBJECTIVE.md')
    for branch, expected in REPORT_SHAS.items():
        path = TASK / 'final_analysis' / branch / 'analysis_report.json'
        bind(path, expected)
        report = reports[branch] = read(path)
        need(report.get('format') == 'freeze35_independent_branch_analysis_v1'
             and report.get('status') == 'passed' and report.get('branch') == branch
             and report.get('fixed_epoch') == 35 and report.get('inputs_unchanged') is True
             and report.get('analysis_sha256') == CORE_SHA
             and report.get('primary_estimand') == 'sample mean', 'Branch report is not accepted')
        for source, digest in report['input_and_implementation_sha256'].items():
            bind(source, digest)
        for split, population in POPULATION.items():
            label = f'{branch}_R35_{split}'
            point = points[label] = report['models'][label]
            need((point['branch'], point['epoch'], point['split'], point['reference'])
                 == (branch, 35, split, False), 'Wrong fixed model endpoint')
            for key in ('summary', 'predictions', 'checkpoint'):
                bind(point[key], point[key + '_sha256'])
            bind(point['audit_path'], point['audit_sha256'])
            audit = read(point['audit_path'])
            need(audit.get('passed') is True and audit.get('mismatches') == 0
                 and audit.get('nonfinite_saved_metric') is False and audit.get('tolerance') == 2e-6
                 and audit.get('source_labels_verified') is True and audit.get('source_cold_groups_verified') is True
                 and (audit.get('examples'), audit.get('users')) == population
                 and audit.get('primary_estimand') == 'sample mean', 'Cached source audit is incomplete')
            need(all(audit[key] == point[key] for key in
                     ('summary_sha256', 'predictions_sha256', 'checkpoint_sha256', 'metrics', 'user_average', 'groups')),
                 'Cached model values differ from their accepted audit')
            meta = read(point['summary'])
            need(meta == point['metadata'] and Path(point['summary']).parent == Path(point['predictions']).parent,
                 'Actual summary differs from the accepted report')
            required = dict(split=split, examples=population[0], full_split=True, subset=None,
                            diagnostic_prefix_limit=None, catalog_size=10728, beam=32, inference_precision='fp32',
                            architecture='H', seed=2024, r_updates=24255, t_updates=200 if branch == 'A' else 360)
            need(all(type(meta.get(k)) is type(v) and meta.get(k) == v for k, v in required.items())
                 and meta['metric_protocol'] == PROTOCOL, 'Evaluation metadata differs from the locked protocol')
            need(meta['dataset_binding']['history_max'] == 50
                 and meta['dataset_binding']['list_length'] == 5 and meta['dataset_binding']['pad_id'] == 0,
                 'Wrong bound history/list/PAD protocol')
            need(audit['train_seen_sha256'] == meta['dataset_binding']['files']['train_seen.json']['sha256'],
                 'Accepted cold-group source differs')
            receipt_path = Path(point['summary']).parent.parent / 'complete.json'
            need(str(receipt_path.resolve()) in inputs, 'Completion receipt missing from accepted source bindings')
            receipt = read(receipt_path)
            need(receipt['attempt'] == Path(point['summary']).parent.name
                 and all(receipt[k] == point[k] for k in ('summary_sha256', 'predictions_sha256')),
                 'Evaluation completion receipt differs')
            binding = receipt['binding']
            need(all(binding.get(k) == v for k, v in dict(branch=branch, split=split, epoch=35,
                     r_updates=24255, t_updates=required['t_updates'], model=point['model_fingerprint'],
                     execution=point['execution_fingerprint']).items()), 'Completed evaluation state binding differs')
    metrics = sorted(points['A_R35_test']['metrics'])
    need(len(metrics) == 37 and all(set(p['metrics']) == set(metrics) for p in points.values()),
         'The four audits must contain the same complete 37 metrics')
    for split in POPULATION:
        a, b = (points[f'{branch}_R35_{split}'] for branch in ('A', 'B'))
        for key in ('split', 'examples', 'full_split', 'subset', 'diagnostic_prefix_limit', 'catalog_size',
                    'metric_protocol', 'dataset_binding', 'beam', 'inference_precision', 'architecture', 'seed'):
            need(a['metadata'].get(key) == b['metadata'].get(key), 'A/B protocol differs: ' + key)
    for branch in ('A', 'B'):
        a, b = (points[f'{branch}_R35_{split}'] for split in ('test', 'validation'))
        need(all(a[k] == b[k] for k in ('checkpoint_sha256', 'model_fingerprint', 'execution_fingerprint')),
             'Test and validation do not use the same fixed-R35 state')
    return inputs, points, metrics


def main():
    need(OUT.name == 'early_ab_comparison_20261029' and TASK.name == 'h_freeze35_20261029', 'Wrong output scope')
    need({p.name for p in OUT.iterdir()} == {'driver.py'}, 'Refuse rerun or a pre-existing output directory')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    inputs, points, metrics = authenticate()
    module = importlib.import_module('scripts.listwise_diffgrm.analysis')
    need(Path(module.__file__).resolve() == CORE.resolve(), 'Imported core is from another checkout')
    import numpy as np
    request = dict(format='freeze35_early_ab_pair_request_v1', created_utc=started,
                   scope='Early supplemental fixed R35 A-minus-B; not all-ready or final Goal acceptance',
                   branch_report_sha256=REPORT_SHAS, input_and_implementation_sha256=inputs,
                   core_sha256=CORE_SHA, driver_sha256=sha(__file__),
                   command_argv=list(sys.orig_argv), command_line=subprocess.list2cmdline(sys.orig_argv),
                   cwd=str(Path.cwd().resolve()), python_executable=sys.executable,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                   comparisons=['A_minus_B_R35_test', 'A_minus_B_R35_validation'],
                   baseline='B_R35', treatment='A_R35', metric_names=metrics, metric_count=37,
                   bootstrap=dict(repeats=2000, seed=2024, confidence=.95, unit='paired user cluster'),
                   primary_estimand='sample mean', independent_branch_audits_reused=True,
                   standalone_audit_recompute_performed=False,
                   internal_metric_replay='Unmodified paired_comparison replays values and matches sample/user/labels/cold flags',
                   gpu_or_platform_access=False, model_restore_performed=False,
                   numerical_agreement_with_cached_audit_absolute_tolerance=1e-10)
    request_path = write('REQUEST.json', request)
    print(json.dumps(dict(event='authenticated', input_files=len(inputs), metrics=37), ensure_ascii=False), flush=True)
    outputs, featured = {}, {}
    for split, population in POPULATION.items():
        name = 'A_minus_B_R35_' + split
        baseline, treatment = points['B_R35_' + split], points['A_R35_' + split]
        begin = time.perf_counter()
        value = module.paired_comparison(baseline['predictions'], treatment['predictions'],
                                        repeats=2000, seed=2024, confidence=.95, metric_names=metrics)
        need((value['examples'], value['users']) == population and set(value['primary']) == set(metrics)
             and set(value['user_average']) == set(metrics), 'Incomplete paired sample/metric coverage')
        need(value['inputs'] == dict(baseline_sha256=baseline['predictions_sha256'],
                                    treatment_sha256=treatment['predictions_sha256']), 'Paired source hash mismatch')
        for metric in metrics:
            result = value['primary'][metric]
            need(math.isclose(result['baseline'], baseline['metrics'][metric], rel_tol=0, abs_tol=1e-10)
                 and math.isclose(result['treatment'], treatment['metrics'][metric], rel_tol=0, abs_tol=1e-10),
                 'Paired point estimates differ from accepted metric audit: ' + metric)
            need(all(math.isfinite(v) for v in result['confidence_interval'])
                 and result['confidence_interval'][0] <= result['confidence_interval'][1], 'Invalid interval')
        value.update(comparison=name, split=split, early_supplemental_pair=True, all_ready=False,
                     standalone_independent_audits_reused=True,
                     accepted_audits={branch: {k: points[f'{branch}_R35_{split}'][k] for k in
                                      ('audit_path', 'audit_sha256', 'summary', 'summary_sha256',
                                       'predictions', 'predictions_sha256', 'checkpoint', 'checkpoint_sha256')}
                                      for branch in ('A', 'B')},
                     source_request_sha256=sha(request_path), core_sha256=CORE_SHA,
                     driver_sha256=request['driver_sha256'], seconds=time.perf_counter() - begin)
        path = write(name + '.json', value)
        outputs[path.name] = sha(path)
        featured[split] = {metric: value['primary'][metric] for metric in FEATURED}
        print(json.dumps(dict(event='comparison_complete', split=split, seconds=value['seconds'],
                              ndcg5=value['primary'][PRIMARY]), ensure_ascii=False), flush=True)
    need(all(sha(path) == expected for path, expected in inputs.items()), 'An original source changed during comparison')
    need('torch' not in sys.modules, 'Unexpected Torch import')
    text = ('# A−B固定R35：提前补充配对比较\n\n'
            '基线为B（H0在R32冻结），处理组为A（H1在R20冻结），差值均为A减B。'
            '复用四份已验收分支审计，未重新执行独立审计或GPU推理；只运行原CPU配对函数。'
            '验证71,657样本/4,733用户，测试53,194样本/4,153用户。全部37项指标均计算2000次、'
            'seed2024、95%用户聚类bootstrap；主指标采用样本平均，辅报等权用户平均。\n\n'
            '| split | 指标 | B R35 | A R35 | A−B绝对差 | 相对B变化 | 95%区间 |\n'
            '|---|---|---:|---:|---:|---:|---|\n')
    for split, values in featured.items():
        for metric, result in values.items():
            lo, hi = result['confidence_interval']
            relative = '未定义' if result['relative_delta'] is None else f"{result['relative_delta']:+.4%}"
            text += (f"| {split} | {metric} | {result['baseline']:.10f} | {result['treatment']:.10f} | "
                     f"{result['delta']:+.10f} | {relative} | [{lo:+.10f}, {hi:+.10f}] |\n")
    text += ('\n这是不同冻结时点、T更新数和续训机会的补充比较，不能把A−B归因于Mixer本身；'
             '主A−C对照仍待完成。单seed用户区间不覆盖训练随机性、冻结点选择，未作多重比较校正。'
             '聚合@50沿用多份五商品候选列表规则，不是一份完整50商品列表。'
             '本结果不替代最终all-ready、模型恢复或Goal验收，也不改变任何训练或选模规则。\n\n'
             '完整来源、原审计SHA、实际运行命令和版本见REQUEST.json；两个结果JSON包含全部37项指标。\n')
    with (OUT / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    outputs['REPORT.md'] = sha(OUT / 'REPORT.md')
    outputs['REQUEST.json'] = sha(request_path)
    write('COMPLETE.json', dict(format='freeze35_early_ab_pair_complete_v1', status='passed',
          scope=request['scope'], started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          request_sha256=sha(request_path), driver_sha256=sha(__file__), core_sha256=CORE_SHA,
          command_argv=request['command_argv'], versions=request['versions'],
          input_and_implementation_sha256=inputs, inputs_unchanged=True, artifact_sha256=outputs,
          comparisons=featured, comparison_count=2, metrics_per_comparison=37,
          primary_estimand='sample mean', bootstrap=request['bootstrap'], accepted_branch_audits_reused=4,
          standalone_audit_recompute_performed=False, gpu_or_platform_access=False, torch_imported=False,
          model_restore_performed=False, all_ready=False, goal_complete=False))
    print(json.dumps(dict(event='complete', status='passed', complete_sha256=sha(OUT / 'COMPLETE.json'))), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (OUT / 'COMPLETE.json').exists() and not (OUT / 'FAILED.json').exists():
            write('FAILED.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                      partial_outputs_not_accepted=True))
        raise
