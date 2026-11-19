"""Independent CPU-only branch delivery and final paired comparisons.

Examples (all outputs must be NEW directories):
  python analyze_results.py --runs-root LOCAL/runs --dataset DATA --references REFS --branch A --output ANALYSIS/A
  python analyze_results.py --runs-root LOCAL/runs --dataset DATA --references REFS --all-ready --branch-reports ANALYSIS --output FINAL

The current, previously accepted independent analysis adds source-verified cold
groups to the old frozen version; ranking/aggregation/bootstrap are unchanged.
Its exact SHA is required. No training/decoding/Torch import occurs here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

ANALYSIS_SHA = '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
SUBSET_SHA = '33f6fb79033ba31d2ec454c19e7b4c2cad38abaf3c926f263c6f6ff2e23aee2a'
DATA_SHA = {'test.parquet': 'f1946e99dfcb1bd1603164a6481fdc3ce7ceacb37bedcd755a025bbf14aa9cee',
            'validation.parquet': '9303e01107e4182a5fb33577cb96d9f256cbc85f737f62f349d8b78a61eaa3d4',
            'train_seen.json': 'f8a0205149fc22d418bf4c28f43037c44c8d7539d1aa9b837f636f6df636efdb'}
NAMES = {'A': 'H1-Freeze20-R35', 'B': 'H0-Freeze32-R35', 'C': 'H0-Freeze20-R35'}
PARENTS = {'A': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
           'B': '8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e',
           'C': 'b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166'}
NEW_VAL = {'A': (29, 32, 35), 'B': (35,), 'C': (8, 11, 14, 17, 20, 23, 26, 29, 32, 35)}
GRID = (20, 23, 26, 29, 32, 35)
PRIMARY = 'list/ndcg@5'
PROTOCOL = dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                target_relevance='unordered_set_of_five_future_distinct_items',
                aggregation='list_score_order_then_item_position_first_occurrence')
POPULATION = {'test': (53194, 4153), 'validation': (71657, 4733)}
PINNED_TEST_REFS = {
    ('A', 20): ('10a2edb7baeb337eba781e26cd9a4d396ff8e6de7cd97bebcc3efd48d3c6b1a2', '2e588690946d14d6c406acc0f00612c9c631c65aaa94adb201eec351afeae1e0'),
    ('A', 26): ('f91a87733db118e39a9e6b13226adabb8e9e3a684e7ebacabcfd4c43f3024f1e', '149350d1253c099131d37c64229365b430ab520e2b4dac716669d5979d7a7cbb'),
    ('B', 32): ('78928d884326139c22a14f7679246ebe6227b02a0452939fec76d9890a693441', 'd0b3b80f3bddb0bc9172e6491eee58e0190aa536b1070076fb04eb7298a1891a')}
PINNED_REFERENCE_MODELS = {('A', 20): 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1',
                           ('A', 26): PARENTS['A'], ('B', 32): PARENTS['B'], ('C', 5): PARENTS['C']}
INTERPRETATION = ('One seed and the original fixed split. User-cluster bootstrap preserves all samples per drawn user; '
    'the primary estimand is sample mean, not equal-user mean. Intervals do not cover training-seed variation, '
    'historical model selection, or the H1-informed choice to freeze at R20. Auxiliary intervals are not multiplicity-adjusted. '
    'A-C compares the common freeze-at-R20 protocol; A-B is a practical baseline comparison with different freeze times/T clocks. '
    'Aggregate @50 is a fixed aggregation of candidate five-item lists, not one complete fifty-item list.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def implementation():
    paths = [ROOT / 'scripts/listwise_diffgrm' / name for name in ('analysis.py', 'subset.py')]
    if [sha(path) for path in paths] != [ANALYSIS_SHA, SUBSET_SHA]:
        raise ValueError('Independent analysis implementation differs from the accepted SHA')
    module = importlib.import_module('scripts.listwise_diffgrm.analysis')
    if Path(module.__file__).resolve() != paths[0].resolve():
        raise ValueError('An analysis module from another checkout is already loaded')
    return module, {str(path): sha(path) for path in [*paths, Path(__file__).resolve()]}


def expected_t(branch, epoch):
    if branch == 'B': return 360
    if branch == 'A': return 200
    if branch == 'C': return min(200, max(0, (epoch - 5) // 3) * 40)
    raise ValueError('Unsupported branch')


def shared_best(validation):
    if set(validation) != set(GRID): raise ValueError('Incomplete common validation grid')
    scores = {epoch: validation[epoch]['metadata']['metrics'][PRIMARY] for epoch in GRID}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite validation selection score')
    return min(GRID, key=lambda epoch: (-scores[epoch], epoch))


def validate_metadata(meta, branch, epoch, split):
    if branch not in NAMES or split not in POPULATION or type(epoch) is not int:
        raise ValueError('Unsupported branch/split/epoch')
    if split == 'test' and branch == 'C' and epoch < 20:
        raise ValueError('C R5 origin is validation-only; no extra origin test is authorized')
    allowed = set(GRID) if branch == 'A' else {32, 35} if branch == 'B' else (set(NEW_VAL['C']) | {5} if split == 'validation' else set(GRID))
    if epoch not in allowed: raise ValueError('Epoch is outside this branch/split protocol')
    expected = dict(split=split, examples=POPULATION[split][0], full_split=True, subset=None,
                    diagnostic_prefix_limit=None, catalog_size=10728, beam=32, inference_precision='fp32',
                    architecture='H', seed=2024, r_updates=epoch * 693, t_updates=expected_t(branch, epoch))
    for key, value in expected.items():
        if type(meta.get(key)) is not type(value) or meta.get(key) != value:
            raise ValueError(f'Wrong {branch}/R{epoch}/{split} protocol: {key}')
    if meta.get('metric_protocol') != PROTOCOL or not isinstance(meta.get('groups'), dict):
        raise ValueError('Wrong item-ranking/aggregation or source-group protocol')
    if any(meta['dataset_binding'].get(key) != value for key, value in dict(history_max=50, list_length=5, pad_id=0).items()):
        raise ValueError('Wrong bound history/list/PAD protocol')
    for name, digest in DATA_SHA.items():
        if meta['dataset_binding']['files'][name]['sha256'] != digest:
            raise ValueError('Evaluation dataset binding changed: ' + name)


def safe_local(branch_dir, saved_path):
    """Relocate only an exact branch-name suffix; never guess a basename."""
    saved = str(saved_path)
    direct = Path(saved)
    if direct.is_file():
        resolved = direct.resolve()
        if not resolved.is_relative_to(branch_dir.resolve()):
            raise ValueError('New artifact path is outside the selected local branch')
        return resolved
    if not direct.is_absolute() and not saved.startswith('/'):
        suffix = saved
    else:
        marker = '/' + branch_dir.name + '/'
        if saved.count(marker) != 1:
            raise ValueError('Cannot uniquely relocate this branch artifact')
        suffix = saved.split(marker, 1)[1]
    parts = PurePosixPath(suffix).parts
    if '..' in parts or PurePosixPath(suffix).is_absolute():
        raise ValueError('Unsafe artifact relocation')
    target = branch_dir.joinpath(*parts).resolve()
    if not target.is_relative_to(branch_dir.resolve()) or not target.is_file():
        raise FileNotFoundError(target)
    return target


class Context:
    def __init__(self, runs_root, dataset, references):
        self.runs_root, self.dataset = Path(runs_root).resolve(), Path(dataset).resolve()
        self.references_path = Path(references).resolve()
        self.inputs = {}; self.refs_sha = self.bind(self.references_path)
        self.refs = read(self.references_path)
        if self.refs.get('format') != 'stage09_h_freeze35_reference_results_v1':
            raise ValueError('Unexpected reference manifest format')
        for name, digest in DATA_SHA.items(): self.bind(self.dataset / name, digest)
        self.module, impl = implementation(); self.inputs.update(impl)

    def bind(self, path, expected=None):
        path = Path(path).resolve(); observed = sha(path)
        if expected is not None and observed != expected:
            raise ValueError('Changed source bytes: ' + str(path))
        if str(path) in self.inputs and self.inputs[str(path)] != observed:
            raise ValueError('Input changed during analysis: ' + str(path))
        self.inputs[str(path)] = observed
        return observed

    def json(self, path, expected=None):
        self.bind(path, expected); return read(path)

    def reference(self, branch, epoch, split):
        if split == 'test' and (branch, epoch) not in PINNED_TEST_REFS:
            raise ValueError('No authorized historical origin test for this model')
        record = dict(self.refs[split][branch][str(epoch)])
        if (branch, epoch) in PINNED_REFERENCE_MODELS and record.get('checkpoint_sha256') != PINNED_REFERENCE_MODELS[branch, epoch]:
            raise ValueError('Historical model identity changed')
        if split == 'test' and (record['predictions_sha256'], record['summary_sha256']) != PINNED_TEST_REFS[branch, epoch]:
            raise ValueError('Historical test reference differs from accepted bytes')
        for key in ('summary', 'predictions', 'checkpoint'):
            if key in record.get('local', {}): record[key] = record['local'][key]
        return self.point(branch, epoch, split, record, reference=True)

    def point(self, branch, epoch, split, record, *, reference=False, model=None, coverage=None):
        directory = self.runs_root / NAMES[branch]
        paths = {key: Path(record[key]).resolve() if reference else safe_local(directory, record[key])
                 for key in ('summary', 'predictions')}
        for key, path in paths.items(): self.bind(path, record[key + '_sha256'])
        if paths['summary'].parent != paths['predictions'].parent:
            raise ValueError('Prediction and summary are from different evaluation directories')
        meta = read(paths['summary']); validate_metadata(meta, branch, epoch, split)
        if model is not None:
            checkpoint = Path(model['checkpoint']); expected_model_sha = model['checkpoint_sha256']
            if record['checkpoint_sha256'] != expected_model_sha:
                raise ValueError('Test checkpoint differs from the validation-locked model')
        else:
            checkpoint = Path(record['checkpoint']).resolve() if reference else safe_local(directory, record['checkpoint'])
            expected_model_sha = record['checkpoint_sha256']
            if not reference and coverage['records'].get(checkpoint.name, {}).get('sha256') != expected_model_sha:
                raise ValueError('Validation model is absent from accepted immutable checkpoint coverage')
        self.bind(checkpoint, expected_model_sha)
        model_fingerprint = execution_fingerprint = None
        if not reference:
            receipt = self.json(paths['summary'].parent.parent / 'complete.json')
            binding = receipt['binding']
            for key, value in dict(branch=branch, split=split, epoch=epoch,
                                   r_updates=meta['r_updates'], t_updates=meta['t_updates']).items():
                if type(binding.get(key)) is not type(value) or binding[key] != value:
                    raise ValueError('Evaluation receipt state/epoch binding mismatch')
            if (receipt['attempt'] != paths['summary'].parent.name
                    or receipt['summary_sha256'] != record['summary_sha256']
                    or receipt['predictions_sha256'] != record['predictions_sha256']):
                raise ValueError('Evaluation completion receipt SHA mismatch')
            model_fingerprint, execution_fingerprint = binding.get('model'), binding.get('execution')
            if not valid_sha(model_fingerprint) or not valid_sha(execution_fingerprint):
                raise ValueError('Evaluation has no model/execution fingerprint binding')
            if model is not None and model.get('model_fingerprint') is not None:
                if (model_fingerprint != model['model_fingerprint']
                        or execution_fingerprint != model['execution_fingerprint']):
                    raise ValueError('Test state differs from the validation-locked model/execution')
        return dict(branch=branch, epoch=epoch, split=split, reference=reference,
                    checkpoint=str(checkpoint), checkpoint_sha256=expected_model_sha,
                    summary=str(paths['summary']), summary_sha256=record['summary_sha256'],
                    predictions=str(paths['predictions']), predictions_sha256=record['predictions_sha256'],
                    metadata=meta, model_fingerprint=model_fingerprint, execution_fingerprint=execution_fingerprint)

    def unchanged(self):
        if any(sha(path) != digest for path, digest in self.inputs.items()):
            raise ValueError('An input or analysis implementation changed during analysis')


def verify_runtime(ctx, branch):
    directory = ctx.runs_root / NAMES[branch]
    # A missing test completion fails before creating an output or any bootstrap.
    tests = ctx.json(directory / 'test_predictions_complete.json')
    complete, request = ctx.json(directory / 'complete.json'), ctx.json(directory / 'request.json')
    selected = ctx.json(directory / 'selection.json', complete['selection_sha256'])
    if (complete.get('status') != 'training_and_validation_complete' or complete.get('branch') != branch
            or (complete.get('phase'), complete.get('r_updates'), complete.get('t_updates')) != (45, 24255, expected_t(branch, 35))
            or tests.get('branch') != branch or tests.get('selection_sha256') != complete['selection_sha256']
            or tests.get('independently_audited') is not False or selected.get('branch') != branch):
        raise ValueError('Training/test completion or selection is incomplete/mixed')
    if (request.get('branch') != branch or request.get('parent_sha256') != PARENTS[branch]
            or request.get('reference_manifest_sha256') != ctx.refs_sha
            or request['protocol'].get('branch') != branch or request['protocol'].get('name') != NAMES[branch]
            or request['protocol'].get('format') != 'stage09_recovery35_v1'
            or request['protocol'].get('parent_sha256') != PARENTS[branch]):
        raise ValueError('Source/branch/reference manifest differs from locked request')
    coverage = ctx.json(directory / 'CHECKPOINT_COVERAGE.json', complete['checkpoint_coverage_sha256'])
    slots = {'A': 13, 'B': 5, 'C': 42}[branch]
    if (coverage.get('status') != 'passed' or coverage.get('expected') != slots
            or coverage.get('present') != slots or len(coverage.get('records', {})) != slots):
        raise ValueError('Required immutable checkpoint coverage is incomplete')
    validation = {}
    for epoch in NEW_VAL[branch]:
        record = ctx.json(directory / 'validation_records' / f'R{epoch:02d}.json')
        if record.get('epoch') != epoch or record.get('kind') != 'new':
            raise ValueError('Unexpected new validation node')
        validation[epoch] = ctx.point(branch, epoch, 'validation', record, coverage=coverage)
    if branch == 'A':
        for epoch in (20, 23, 26): validation[epoch] = ctx.reference(branch, epoch, 'validation')
    if branch in ('A', 'C'):
        if set(selected['candidates']) != set(map(str, GRID)):
            raise ValueError('Shared best-selection grid is incomplete')
        for epoch in GRID:
            candidate = selected['candidates'][str(epoch)]
            actual = validation[epoch]
            if (candidate.get('epoch') != epoch or candidate.get('checkpoint_sha256') != actual['checkpoint_sha256']
                    or candidate.get('summary_sha256') != actual['summary_sha256']
                    or candidate.get('predictions_sha256') != actual['predictions_sha256']
                    or candidate.get('ndcg5') != actual['metadata']['metrics'][PRIMARY]):
                raise ValueError('Shared validation candidate differs from its actual model/summary')
        best = shared_best({epoch: validation[epoch] for epoch in GRID})
    else:
        best = 35
        if selected.get('candidates') != {}: raise ValueError('B cannot use the A/C shared best-selection grid')
    chosen = selected['selected']
    if chosen.get('epoch') != best or chosen.get('checkpoint_sha256') != validation[best]['checkpoint_sha256']:
        raise ValueError('Validation best was not selected by the locked grid/tie rule')
    expected = {35, best} if branch in ('A', 'C') else {35}
    rows = tests['results']
    if len(rows) != len(expected) or {r.get('epoch') for r in rows} != expected:
        raise ValueError('Missing, duplicate or unauthorized additional test endpoints')
    test_points = {}
    for record in rows:
        epoch = record['epoch']
        if record.get('reused') is True:
            point = ctx.reference(branch, epoch, 'test')
            if point['checkpoint_sha256'] != validation[epoch]['checkpoint_sha256']:
                raise ValueError('Reused test belongs to another selected model')
            for key in ('checkpoint_sha256', 'summary_sha256', 'predictions_sha256'):
                if record.get(key) != point[key]: raise ValueError('Cached test completion differs from accepted reference')
        else:
            point = ctx.point(branch, epoch, 'test', record, model=validation[epoch])
        test_points[epoch] = point
    return validation, test_points, best


def audit(ctx, label, point, output):
    value = ctx.module.audit_predictions(point['predictions'], dataset_dir=ctx.dataset)
    if (value.get('passed') is not True or (value['examples'], value['users']) != POPULATION[point['split']]
            or value.get('source_labels_verified') is not True or value.get('source_cold_groups_verified') is not True
            or value.get('mismatches') != 0 or value.get('nonfinite_saved_metric') is not False
            or value['predictions_sha256'] != point['predictions_sha256'] or value['summary_sha256'] != point['summary_sha256']):
        raise ValueError('Independent complete-split source/metric replay failed: ' + label)
    value.update(model=label, branch=point['branch'], epoch=point['epoch'], split=point['split'],
                 checkpoint_sha256=point['checkpoint_sha256'], primary_estimand='sample mean')
    path = Path(output) / 'audits' / (label + '.json'); write(path, value)
    return dict(point, audit_path=str(path), audit_sha256=sha(path), metrics=value['metrics'],
                user_average=value['user_average'], groups=value['groups'])


def pair(ctx, label, baseline, treatment, output):
    if baseline['split'] != treatment['split']: raise ValueError('Cannot compare validation to test')
    value = ctx.module.paired_comparison(baseline['predictions'], treatment['predictions'],
                                         repeats=2000, seed=2024, confidence=.95)
    if (value['examples'], value['users']) != POPULATION[baseline['split']]:
        raise ValueError('Paired complete-split population differs')
    value.update(comparison=label, split=baseline['split'], primary_metric=PRIMARY,
                 primary_result=value['primary'][PRIMARY], interpretation=INTERPRETATION,
                 difference='treatment minus baseline')
    path = Path(output) / 'comparisons' / (label + '.json'); write(path, value)
    return dict(path=str(path), sha256=sha(path), split=baseline['split'], **value['primary_result'])


def branch_comparison_specs(branch):
    # This is the locked INPUT_AUDIT_COVERAGE bootstrap plan, not extra inference.
    if branch == 'A':
        return [(f'A_R35_minus_{b}_R{e}_test', f'{b}_R{e}_test', 'A_R35_test')
                for b, e in [('A', 26), ('A', 20), ('B', 32)]] + [
                ('A_R35_minus_A_R26_validation', 'A_R26_validation', 'A_R35_validation')]
    if branch == 'B':
        return [('B_R35_minus_B_R32_test', 'B_R32_test', 'B_R35_test'),
                ('B_R35_minus_B_R32_validation', 'B_R32_validation', 'B_R35_validation')]
    if branch == 'C':
        return [('C_R35_minus_R5_validation', 'C_R5_validation', 'C_R35_validation'),
                ('C_R35_minus_R20_validation', 'C_R20_validation', 'C_R35_validation')]
    raise ValueError('Unknown branch')


def final_comparison_specs(selected):
    result = [(f'A_minus_{b}_R35_{split}', f'{b}_R35_{split}', f'A_R35_{split}')
              for split in ('test', 'validation') for b in ('C', 'B')]
    if selected['A'] != 35 or selected['C'] != 35:
        result += [(f'A_minus_C_validation_best_{split}', f'C_R{selected["C"]}_{split}',
                    f'A_R{selected["A"]}_{split}') for split in ('test', 'validation')]
    return result


def csv_table(output, models):
    keys = sorted(next(iter(models.values()))['metrics'])
    with (Path(output) / 'METRICS.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'branch', 'epoch', 'split', 'examples', 'users', *keys])
        writer.writeheader()
        for label, point in models.items():
            writer.writerow(dict(model=label, branch=point['branch'], epoch=point['epoch'], split=point['split'],
                                  examples=POPULATION[point['split']][0], users=POPULATION[point['split']][1], **point['metrics']))


def fresh_output(output, ctx):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (ctx.runs_root, ctx.dataset)):
        raise FileExistsError('Analysis needs a new output directory outside protected source directories')
    output.mkdir(parents=True, exist_ok=False)
    return output


def finish(ctx, output, report, text):
    ctx.unchanged()
    if 'torch' in sys.modules: raise RuntimeError('Independent analysis unexpectedly imported Torch')
    report.update(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                  input_and_implementation_sha256=ctx.inputs, inputs_unchanged=True,
                  analysis_sha256=ANALYSIS_SHA, original_frozen_analysis_sha256='487415686596235afe7204168708892d195b203418d1def054bd1355f4f2f29b',
                  analysis_difference='Only adds source-hashed training-seen/cold-group verification; metric and bootstrap formulas unchanged',
                  primary_metric=PRIMARY, primary_estimand='sample mean',
                  bootstrap=dict(repeats=2000, seed=2024, confidence=.95, unit='user cluster'),
                  torch_imported=False, gpu_evaluation_performed=False, interpretation=INTERPRETATION)
    csv_table(output, report['models'])
    (Path(output) / 'REPORT.md').write_text(text, encoding='utf-8')
    report['artifact_sha256'] = {name: sha(Path(output) / name) for name in ('METRICS.csv', 'REPORT.md')}
    write(Path(output) / 'analysis_report.json', report)
    return report


def analyze_branch(*, runs_root, dataset, references, branch, output):
    ctx = Context(runs_root, dataset, references)
    validation, tests, best = verify_runtime(ctx, branch)
    output = fresh_output(output, ctx)
    try:
        models = {}
        for split, points in (('validation', validation), ('test', tests)):
            for epoch, point in sorted(points.items()):
                label = f'{branch}_R{epoch}_{split}'; models[label] = audit(ctx, label, point, output)
        comparisons = {}
        origins = [('A', 26), ('A', 20), ('B', 32)] if branch == 'A' else [('B', 32)] if branch == 'B' else []
        for old_branch, epoch in origins:
            label = f'{old_branch}_R{epoch}_test'
            if label not in models: models[label] = audit(ctx, label, ctx.reference(old_branch, epoch, 'test'), output)
        if branch == 'B':
            label = 'B_R32_validation'; models[label] = audit(ctx, label, ctx.reference('B', 32, 'validation'), output)
        if branch == 'C':
            label = 'C_R5_validation'; models[label] = audit(ctx, label, ctx.reference('C', 5, 'validation'), output)
        for name, baseline, treatment in branch_comparison_specs(branch):
            comparisons[name] = pair(ctx, name, models[baseline], models[treatment], output)
        fixed = models[f'{branch}_R35_test']['metrics']
        text = f'# {NAMES[branch]}：独立结果验收\n\n固定R35完整测试复算通过：NDCG@5={fixed[PRIMARY]:.8f}，HR@5={fixed["list/hr@5"]:.8f}，Recall@5={fixed["list/recall@5"]:.8f}。'
        text += ('本轮B报告固定R35终点，与父R32并列比较，不执行A/C共同验证网格的best选择。测试覆盖按预定规则去重。\n\n'
                 if branch == 'B' else f'验证选出的补充best为R{best}；测试覆盖按预定规则去重。\n\n')
        if best != 35:
            text += f'验证best对应的测试NDCG@5={models[f"{branch}_R{best}_test"]["metrics"][PRIMARY]:.8f}；该模型由验证选择，没有按测试重选轮次。\n\n'
        if branch == 'A':
            change = comparisons['A_R35_minus_A_R26_test']
            direction = '上升' if change['delta'] > 0 else '下降' if change['delta'] < 0 else '持平'
            relative = f'{change["relative_delta"]:+.2%}' if change['relative_delta'] is not None else '起点为零，不计算比例'
            lo, hi = change['confidence_interval']
            interval = '区间在零以上' if lo > 0 else '区间在零以下' if hi < 0 else '区间跨零'
            text += f'相对已有Freeze R26，固定R35的测试点估计{direction}（{relative}），{interval}。验证best与R35{"一致" if best == 35 else "不一致"}；新B/C对照另行汇总，不阻塞本次H1报告。\n\n'
        text += '| 比较 | split | NDCG@5绝对差 | 95%用户聚类区间 |\n|---|---|---:|---|\n'
        for name, value in comparisons.items():
            text += f'| {name} | {value["split"]} | {value["delta"]:+.8f} | [{value["confidence_interval"][0]:+.8f}, {value["confidence_interval"][1]:+.8f}] |\n'
        text += '\n以下为验证曲线，不与测试绝对值混比：\n\n| R | 验证NDCG@5 |\n|---|---:|\n'
        text += ''.join(f'| {epoch} | {models[f"{branch}_R{epoch}_validation"]["metrics"][PRIMARY]:.8f} |\n' for epoch in sorted(validation))
        text += ('\n本报告仅验收该分支，不等待其他组，也不代表Goal完成。A−B属于不同冻结时点的实用比较；C的R5起点仅比较验证，不新增起点测试。冻结仍保留已经学到的编码；单seed及用户区间不能证明跨seed稳定，R20冻结时点也源于之前H1观察。聚合@50是多份五件列表聚合，候选覆盖须同时看CSV。\n')
        return finish(ctx, output, dict(format='freeze35_independent_branch_analysis_v1', branch=branch,
                      fixed_epoch=35, selected_epoch=best, models=models, comparisons=comparisons), text)
    except Exception as exc:
        write(output / 'FAILED.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', partial_outputs_not_accepted=True)); raise


def verify_cached_points(report, branch, validation, tests, best):
    if report.get('selected_epoch') != best or report.get('fixed_epoch') != 35:
        raise ValueError('Cached report selection differs from the current locked branch')
    for split, points in (('validation', validation), ('test', tests)):
        for epoch, actual in points.items():
            cached = report['models'][f'{branch}_R{epoch}_{split}']
            if any(cached[key] != actual[key] for key in ('predictions_sha256', 'summary_sha256', 'checkpoint_sha256')):
                raise ValueError('Cached audited model differs from the downloaded runtime endpoint')


def analyze_all(*, runs_root, dataset, references, branch_reports, output):
    ctx = Context(runs_root, dataset, references)
    models, selected = {}, {}
    for branch in NAMES:
        report = ctx.json(Path(branch_reports) / branch / 'analysis_report.json')
        if (report.get('status') != 'passed' or report.get('branch') != branch
                or report.get('format') != 'freeze35_independent_branch_analysis_v1' or report.get('analysis_sha256') != ANALYSIS_SHA):
            raise ValueError('All-ready requires all three independently accepted branch reports')
        for path, digest in report['input_and_implementation_sha256'].items(): ctx.bind(path, digest)
        for point in report['models'].values():
            cached_audit = ctx.json(point['audit_path'], point['audit_sha256'])
            if (cached_audit.get('passed') is not True
                    or cached_audit['predictions_sha256'] != point['predictions_sha256']
                    or cached_audit['summary_sha256'] != point['summary_sha256']
                    or any(cached_audit[key] != point[key] for key in ('metrics', 'user_average', 'groups'))):
                raise ValueError('Cached report metrics differ from its accepted independent audit')
        for name, point in report['models'].items():
            if name in models and models[name]['predictions_sha256'] != point['predictions_sha256']:
                raise ValueError('Cached branch reports disagree on a shared model')
            models[name] = point
        # Re-authenticate selection and completed test endpoints without repeating audits.
        validation, tests, selected[branch] = verify_runtime(ctx, branch)
        verify_cached_points(report, branch, validation, tests, selected[branch])
    output = fresh_output(output, ctx)
    try:
        comparisons = {}
        for name, baseline, treatment in final_comparison_specs(selected):
            comparisons[name] = pair(ctx, name, models[baseline], models[treatment], output)
        text = '# 三组冻结续训：最终配对比较\n\n各分支独立复算已通过；本步骤复用已验收审计，仅新增配对用户bootstrap。主比较固定R35的A−C；A−B是不同冻结时点的补充基线。\n\n| 固定R35模型 | T更新数 | 测试NDCG@5 | HR@5 | Recall@5 | 聚合NDCG@50 | 验证best R |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for branch in NAMES:
            values = models[f'{branch}_R35_test']['metrics']
            best_label = '不适用（固定R35；父R32另列）' if branch == 'B' else str(selected[branch])
            text += f'| {NAMES[branch]} | {expected_t(branch, 35)} | {values[PRIMARY]:.8f} | {values["list/hr@5"]:.8f} | {values["list/recall@5"]:.8f} | {values["aggregate/ndcg@50"]:.8f} | {best_label} |\n'
        text += '\nB的“35”只标识本轮固定终点；其原R32父模型另列CSV，不与A/C共同网格选best混为一谈。\n\n| 比较 | split | NDCG@5绝对差 | 95%用户聚类区间 |\n|---|---|---:|---|\n'
        text += ''.join(f'| {name} | {v["split"]} | {v["delta"]:+.8f} | [{v["confidence_interval"][0]:+.8f}, {v["confidence_interval"][1]:+.8f}] |\n' for name, v in comparisons.items())
        text += '\n主估计按样本平均，用户等权结果另列；单seed区间不覆盖训练随机性和此前选模/冻结点选择。研究结论按实际正负结果解释，不把A优于B直接归因为交互，也不把测试抽样区间写成稳定收益。完整备份、恢复及GPU释放由另行交付验收覆盖，本分析不宣布整个Goal完成。\n'
        return finish(ctx, output, dict(format='freeze35_independent_final_comparison_v1', selected_epochs=selected,
                      models=models, comparisons=comparisons, reused_branch_audits=True,
                      best_comparison_reuses_fixed_pair=(selected['A'] == 35 and selected['C'] == 35)), text)
    except Exception as exc:
        write(output / 'FAILED.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', partial_outputs_not_accepted=True)); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs-root', 'dataset', 'references', 'output'): parser.add_argument('--' + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--branch', choices=tuple(NAMES)); mode.add_argument('--all-ready', action='store_true')
    parser.add_argument('--branch-reports', help='Root containing A/B/C/analysis_report.json; required for --all-ready')
    args = parser.parse_args(argv)
    common = dict(runs_root=args.runs_root, dataset=args.dataset, references=args.references, output=args.output)
    if args.all_ready:
        if not args.branch_reports: parser.error('--all-ready requires --branch-reports; do not repeat accepted branch audits')
        report = analyze_all(**common, branch_reports=args.branch_reports)
    else: report = analyze_branch(**common, branch=args.branch)
    print(json.dumps(dict(status=report['status'], scope=report['format'], comparisons=report['comparisons']), ensure_ascii=False))


if __name__ == '__main__': main()
