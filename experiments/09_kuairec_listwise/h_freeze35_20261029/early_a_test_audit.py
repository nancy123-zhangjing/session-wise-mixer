"""CPU-only early A test audit using the unchanged, SHA-pinned analysis CLI.

Deploy this file as STAGE/early_a_test_audit.py plus the four STAGED files below.
The stage, original Goal, dataset and NEW output are explicit. No model loading,
training, platform calls, source edits or replacement of formal analysis occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
GOAL = BASE + '/h_freeze35_20261029_r1'
DATASET = BASE + '/goal_20261027_r1/inputs/dataset'
HELPER = 'experiments/09_kuairec_listwise/h_freeze35_20261029/analyze_results.py'
STAGED = {
    HELPER: '47522df77e2eab06e598454ed331e6a939609bf0f63ddb9087d78f7f19aa9677',
    'scripts/listwise_diffgrm/analysis.py': '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd',
    'scripts/listwise_diffgrm/subset.py': '33f6fb79033ba31d2ec454c19e7b4c2cad38abaf3c926f263c6f6ff2e23aee2a',
    'scripts/listwise_diffgrm/__init__.py': '23b895b1c072e3c8a22aa3e8b4d906ad315a35a668cb46fa1e77a21efbdbfb60',
}
EXECUTION_SHA = '9e50d158a68f20232119edee42fae6f5c42a518a27d889b835144aa33b51f28b'
REFS_SHA = '4c0a0e988f9073a89ab6fac37d6d05b394ff4271ccbb553ace9c75ffc5bf7734'
NEW_MODELS = {32: 'b96ab6931b9b44939aae9564cbcaeec266440db3121af2935cc64a7654f53148',
              35: 'df59a9bf8bd3e4d118f0c4082302d15fc5f7642306c71aa71e82abb237bfecfe'}
LEGACY_SELECTION_SUMMARIES = {
    23: '9910012060faae7c4e22c604b326fc3832bc4fbdd6564b95f13e166daef90f91',
    26: 'b375e20ca859f2982019a8d37b882bd9035355bbe88250964fd9706f0036f784',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def scope(stage, goal, dataset, output):
    values = [PurePosixPath(str(v)) for v in (stage, goal, dataset, output)]
    for raw, value in zip((stage, goal, dataset, output), values):
        need(str(value) == str(raw) and value.is_absolute() and '..' not in value.parts
             and '\\' not in str(raw), 'Canonical absolute POSIX paths required')
    s, g, d, o = values
    need(str(g) == GOAL and str(d) == DATASET, 'Only the locked Goal and dataset are allowed')
    need(o.parent == PurePosixPath(BASE) and o.name.startswith('h_freeze35_20261029')
         and not any(o.is_relative_to(p) or p.is_relative_to(o) for p in (s, g, d)),
         'NEW delivery output must be outside and disjoint from stage, Goal and data')
    need(not any(s.is_relative_to(p) or p.is_relative_to(s) for p in (g, d)),
         'Analysis stage must be disjoint from protected Goal and data')


def plain(path):
    need('..' not in Path(path).parts, 'Parent traversal is forbidden')
    path = Path(path).absolute()
    for node in (*reversed(path.parents), path):
        if not node.exists() and not node.is_symlink():
            continue
        info = node.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
             'Linked/reparse path rejected: ' + str(node))
    return path


def sha(path):
    digest = hashlib.sha256()
    with plain(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def legacy_selection_point(helper, ctx, branch, epoch, split, record):
    """Bind two old validation selection records, never audit missing group data."""
    need((branch, split) == ('A', 'validation') and epoch in LEGACY_SELECTION_SUMMARIES
         and ctx.refs_sha == REFS_SHA
         and record == ctx.refs['validation']['A'][str(epoch)]
         and record['summary_sha256'] == LEGACY_SELECTION_SUMMARIES[epoch],
         'Legacy schema compatibility is restricted to two locked selection records')
    paths = {key: plain(record[key]) for key in ('summary', 'predictions', 'checkpoint')}
    for key, path in paths.items():
        ctx.bind(path, record[key + '_sha256'])
    need(paths['summary'].parent == paths['predictions'].parent,
         'Legacy prediction and summary directories differ')
    meta = read(paths['summary'])
    need('groups' not in meta, 'Expected the locked historical missing-group schema')
    # This disposable schema shim invokes all other original metadata checks. It
    # does not invent group statistics: returned metadata remains byte-source true,
    # and these records never enter the five TEST recomputations or comparisons.
    helper.validate_metadata(dict(meta, groups={}), branch, epoch, split)
    return dict(branch=branch, epoch=epoch, split=split, reference=True,
                **{key: str(path) for key, path in paths.items()},
                **{key + '_sha256': record[key + '_sha256'] for key in paths},
                metadata=meta, model_fingerprint=None, execution_fingerprint=None,
                missing_group_summary=True, legacy_validation_selection_only=True,
                saved_groups_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-root', 'goal-root', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    scope(args.stage_root, args.goal_root, args.dataset, args.output)
    need(os.name == 'posix', 'This delivery driver runs only in the authenticated Linux CPU entry')
    stage, goal, dataset, output = [plain(v) for v in
                                   (args.stage_root, args.goal_root, args.dataset, args.output)]
    need(not output.exists(), 'Output already exists; no overwrite or automatic retry')
    need(Path(__file__).resolve() == stage / 'early_a_test_audit.py', 'Driver must be in its isolated stage')
    inputs = {}

    def bind(path, expected=None, size=None):
        path = plain(path)
        need(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'Expected regular input file')
        actual = sha(path)
        need(expected is None or actual == expected, 'Input SHA mismatch: ' + str(path))
        need(size is None or path.stat().st_size == size, 'Input byte count mismatch: ' + str(path))
        need(str(path) not in inputs or inputs[str(path)] == actual, 'Input changed during audit')
        inputs[str(path)] = actual
        return actual

    def bound_json(path, expected=None):
        bind(path, expected)
        return read(path)

    staged_files = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    need(staged_files == {*STAGED, 'early_a_test_audit.py'}, 'Isolated stage file inventory differs')
    for name, digest in STAGED.items():
        bind(stage / name, digest)
    bind(Path(__file__))
    spec = importlib.util.spec_from_file_location('_early_a_pinned_analysis', stage / HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.implementation()  # Verifies the actual imported core and subset source paths/SHA.
    need('torch' not in sys.modules, 'Analysis unexpectedly imported Torch')

    code = goal / 'code'
    execution = bound_json(code / 'FREEZE35_EXECUTION_MANIFEST.json', EXECUTION_SHA)
    need(execution['package_mode'] == 'committed' and execution['gpu_execution_allowed'] is True,
         'Not the committed source package')
    records = {**execution['execution_files'], **execution['metadata']}
    actual_files = {p.relative_to(code).as_posix() for p in code.rglob('*') if p.is_file()}
    need(actual_files == {*records, 'FREEZE35_EXECUTION_MANIFEST.json'}, 'Frozen code inventory differs')
    for name, record in records.items():
        bind(code / name, record['sha256'], record['bytes'])
    parents = bound_json(code / 'FREEZE35_PARENTS.json')
    refs = bound_json(code / 'FREEZE35_REFERENCE_RESULTS.json', REFS_SHA)
    need(str(dataset) == parents['dataset']['remote'], 'Dataset is not the declared remote dependency')
    bind(dataset / 'manifest.json', parents['dataset']['manifest_sha256'])
    for name, record in parents['dataset']['files'].items():
        bind(dataset / name, record['sha256'], record['bytes'])
    class RemoteContext(helper.Context):
        """Remote refs plus a declared selection-only old-validation schema shim."""
        def reference(self, branch, epoch, split):
            if split == 'test':
                need((branch, epoch) in helper.PINNED_TEST_REFS, 'Unauthorized historical test')
            record = dict(self.refs[split][branch][str(epoch)])
            if (branch, epoch) in helper.PINNED_REFERENCE_MODELS:
                need(record['checkpoint_sha256'] == helper.PINNED_REFERENCE_MODELS[branch, epoch],
                     'Historical model differs')
            if split == 'test':
                need((record['predictions_sha256'], record['summary_sha256']) == helper.PINNED_TEST_REFS[branch, epoch],
                     'Historical test bytes differ')
            # Use the SHA-locked original remote fields; never modify refs or use local mappings.
            return self.point(branch, epoch, split, record, reference=True)

        def point(self, branch, epoch, split, record, **kwargs):
            for key in ('summary', 'predictions', 'checkpoint'):
                need(plain(record[key]).is_relative_to(Path(BASE)), 'Artifact escaped owned Stage09 root')
            if (branch, split) == ('A', 'validation') and epoch in LEGACY_SELECTION_SUMMARIES:
                need(kwargs == {'reference': True}, 'Legacy shim cannot validate a new evaluation/model')
                value = legacy_selection_point(helper, self, branch, epoch, split, record)
            else:
                value = super().point(branch, epoch, split, record, **kwargs)
            need(value['metadata']['dataset_binding']['manifest_sha256'] == parents['dataset']['manifest_sha256']
                 and value['metadata']['dataset_binding']['files'] == parents['dataset']['files'],
                 'Dataset metadata differs from locked original')
            return value

    ctx = RemoteContext(goal / 'runs', dataset, code / 'FREEZE35_REFERENCE_RESULTS.json')
    validation, tests, best = helper.verify_runtime(ctx, 'A')
    locked = ctx.json(goal / 'runs' / helper.NAMES['A'] / 'selection.json')
    need(locked['criterion'] == 'validation/list/ndcg@5; tie earlier' and locked['fixed_endpoint'] == 35,
         'Selection criterion/fixed endpoint changed')
    need(best == 32 and set(tests) == {32, 35}, 'Expected locked best R32 and fixed R35 tests')
    for epoch in (32, 35):
        need(tests[epoch]['checkpoint_sha256'] == validation[epoch]['checkpoint_sha256'] == NEW_MODELS[epoch],
             'Wrong locked A checkpoint bytes')
    points = {f'A_R{epoch}_test': value for epoch, value in tests.items()}
    for branch, epoch in (('A', 26), ('A', 20), ('B', 32)):
        points[f'{branch}_R{epoch}_test'] = ctx.reference(branch, epoch, 'test')
    legacy_selection = [{key: validation[epoch][key] for key in
        ('epoch', 'split', 'summary', 'summary_sha256', 'predictions', 'predictions_sha256',
         'checkpoint', 'checkpoint_sha256', 'missing_group_summary',
         'legacy_validation_selection_only', 'saved_groups_verified')}
        for epoch in LEGACY_SELECTION_SUMMARIES]
    points = {label: dict(value, record=value) for label, value in points.items()}
    need(all(path not in inputs or inputs[path] == digest for path, digest in ctx.inputs.items()),
         'Input changed between source and protocol checks')
    inputs.update(ctx.inputs)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'audits').mkdir(); (output / 'comparisons').mkdir(); (output / 'commands').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONPATH=str(stage))
    commands, audits, comparisons = [], {}, {}
    write(output / 'REQUEST.json', dict(arguments=vars(args), python=sys.executable, source_and_input_sha256=inputs,
        selected_epoch=32, fixed_epoch=35, formal_analysis_replaced=False, supplemental_best_comparisons=True,
        legacy_validation_selection_only=legacy_selection))

    def run(label, arguments, destination):
        command = [sys.executable, '-B', '-m', 'scripts.listwise_diffgrm.analysis', *arguments, '--output', str(destination)]
        started = time.monotonic()
        with (output / 'commands' / (label + '.log')).open('xb') as log:
            result = subprocess.run(command, cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        row = dict(label=label, argv=command, cwd=str(stage), returncode=result.returncode,
                   seconds=time.monotonic() - started, stdout_stderr_sha256=sha(output / 'commands' / (label + '.log')))
        commands.append(row); write(output / 'commands' / (label + '.json'), row)
        need(result.returncode == 0, 'Core analysis failed; preserve output: ' + label)
        return read(destination)

    try:
        # ALL five source-bound replays must pass before ANY bootstrap is started.
        for label, value in points.items():
            dest = output / 'audits' / (label + '.json')
            audit = run(label, ['recompute', '--predictions', value['predictions'], '--dataset', str(dataset)], dest)
            need(audit['passed'] is True and (audit['examples'], audit['users']) == (53194, 4153)
                 and audit['mismatches'] == 0 and audit['nonfinite_saved_metric'] is False
                 and audit['source_labels_verified'] is True and audit['source_cold_groups_verified'] is True
                 and audit['tolerance'] == 2e-6
                 and all(audit[k + '_sha256'] == value['record'][k + '_sha256'] for k in ('summary', 'predictions')),
                 'Full test replay failed: ' + label)
            audits[label] = dict(path=str(dest), sha256=sha(dest), metrics=audit['metrics'])
        for epoch in (35, 32):
            treatment = f'A_R{epoch}_test'
            for baseline in ('A_R26_test', 'A_R20_test', 'B_R32_test'):
                label = treatment + '_minus_' + baseline; dest = output / 'comparisons' / (label + '.json')
                result = run(label, ['compare', '--baseline', points[baseline]['predictions'], '--treatment',
                    points[treatment]['predictions'], '--repeats', '2000', '--seed', '2024'], dest)
                need((result['examples'], result['users']) == (53194, 4153)
                     and result['bootstrap']['repeats'] == 2000 and result['bootstrap']['seed'] == 2024
                     and result['bootstrap']['confidence'] == .95
                     and result['inputs'] == dict(baseline_sha256=points[baseline]['record']['predictions_sha256'],
                                                 treatment_sha256=points[treatment]['record']['predictions_sha256']),
                     'Paired source/population/settings mismatch')
                comparisons[label] = dict(path=str(dest), sha256=sha(dest), primary=result['primary']['list/ndcg@5'],
                    supplemental=epoch == 32, difference='treatment minus baseline', split='test')
        need(all(sha(path) == digest for path, digest in inputs.items()), 'Source/input changed during analysis')
        write(output / 'COMPLETE.json', dict(status='early_cpu_test_metrics_passed', source_and_input_sha256=inputs,
            commands=commands, audits=audits, comparisons=comparisons, selected_epoch=32, fixed_epoch=35,
            legacy_validation_selection_only=legacy_selection,
            interpretation=helper.INTERPRETATION, primary_estimand='sample mean',
            gpu_or_training_performed=False, model_restore=False, full_backup_verified=False,
            formal_local_branch_analysis_replaced=False, goal_completion_not_implied=True,
            finished_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(status='early_cpu_test_metrics_passed', audits=5, comparisons=6, output=str(output))))
    except BaseException as exc:
        write(output / 'FAILED.json', dict(status='failed_preserve_outputs_no_retry', error=type(exc).__name__ + ': ' + str(exc),
                                         commands=commands, partial_outputs_not_accepted=True))
        raise


if __name__ == '__main__':
    main()
