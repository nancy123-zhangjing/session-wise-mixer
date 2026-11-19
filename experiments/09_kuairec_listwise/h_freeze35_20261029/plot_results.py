"""Source-bound matplotlib PNG/PDF curves; no training or metric recomputation.

Final mode requires all three independently accepted branch reports. Historical
smoke mode is visibly labelled and cannot produce a formal freeze35 figure set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

NAMES = {'A': 'H1-Freeze20-R35', 'B': 'H0-Freeze32-R35', 'C': 'H0-Freeze20-R35'}
PARENTS = {'A': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
           'B': '8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e',
           'C': 'b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166'}
R_EPOCHS = {'A': tuple(range(27, 36)), 'B': tuple(range(33, 36)), 'C': tuple(range(6, 36))}
VAL_EPOCHS = {'A': (20, 23, 26, 29, 32, 35), 'B': (32, 35), 'C': (5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)}
PRIMARY = 'list/ndcg@5'
OLD_H0_LOG_SHA = 'cacc662296711898aa105d6aab25042e45c2d4583273ab3dcfc945d4ad848dcd'
COLORS = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(path, expected, sources):
    path = Path(path).resolve(); actual = sha(path)
    if expected is not None and expected != actual: raise ValueError('Changed plot input: ' + str(path))
    if str(path) in sources and sources[str(path)] != actual: raise ValueError('Input changed during plotting')
    sources[str(path)] = actual
    return path


def plan():
    result = [('R', e) for e in range(1, 6)]
    for cycle in range(10): result += [('T', 5 + 3 * cycle), *[('R', 6 + 3 * cycle + i) for i in range(3)]]
    return result


def training_rows(path, branch=None):
    """Preserve first occurrence; conflicting replayed updates are never hidden."""
    unique, ordered, duplicates = {}, [], 0
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('preflight_only'): raise ValueError('Preflight data is not formal training')
            for key in ('phase_index', 'epoch', 'update', 'r_updates', 't_updates', 'lists'):
                if type(row.get(key)) is not int: raise ValueError(f'Nonintegral {key} at line {number}')
            phase = row['phase_index']
            if not 0 <= phase < 45 or (row.get('phase'), row['epoch']) != plan()[phase]:
                raise ValueError('Wrong phase/epoch identity')
            if branch is not None and row.get('branch') != branch: raise ValueError('Mixed branch training log')
            key = (phase, row['r_updates'], row['t_updates'])
            scientific = {k: v for k, v in row.items() if k not in ('seconds', 'attempt')}
            if key in unique:
                if unique[key] != scientific: raise ValueError('Conflicting duplicate training update')
                duplicates += 1
            else:
                unique[key] = scientific; ordered.append(row)
    return ordered, duplicates


def epoch_means(rows, branch=None):
    groups, lr = {}, []
    for row in rows:
        if row['phase'] != 'R':
            if branch in ('A', 'B'): raise ValueError('Frozen branch contains a real T update')
            if branch == 'C':
                cycle = (row['phase_index'] - 5) // 4
                if not 0 <= cycle < 5 or row['t_updates'] != cycle * 40 + row['update']:
                    raise ValueError('C T clock differs from the replay protocol')
            continue
        e, u, n = row['epoch'], row['update'], row['lists']
        if not 1 <= u <= 693 or row['r_updates'] != (e - 1) * 693 + u:
            raise ValueError('R update clock differs from original epoch coordinate')
        if n != (835 if u == 693 else 1024): raise ValueError('Unexpected effective list count')
        value = row['loss']
        if type(value) not in (int, float) or not math.isfinite(value): raise ValueError('Nonfinite R loss')
        if branch is not None:
            t = 360 if branch == 'B' else 200 if branch == 'A' or e >= 18 else ((e - 6) // 3 + 1) * 40
            if row['t_updates'] != t: raise ValueError('Wrong frozen/replay T clock')
            used = row.get('learning_rates_used', {}).get('R')
            if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
                raise ValueError('Actual pre-step R LR is missing; never reconstruct/fill it')
            lr.append(dict(epoch_coordinate=row['r_updates'] / 693, r_updates=row['r_updates'], learning_rate=used))
        groups.setdefault(e, []).append(row)
    expected = R_EPOCHS[branch] if branch else tuple(sorted(groups))
    if set(groups) != set(expected): raise ValueError('Training trace is incomplete or starts at another checkpoint')
    means = []
    for epoch in expected:
        values = groups[epoch]
        if [row['update'] for row in values] != list(range(1, 694)):
            raise ValueError('Missing, extra or out-of-order distinct R updates')
        count = sum(row['lists'] for row in values)
        if count != 709443: raise ValueError('R epoch does not cover the full training population')
        means.append(dict(epoch=epoch, weighted_loss=math.fsum(row['loss'] * row['lists'] for row in values) / count,
                          lists=count, optimizer_updates=len(values)))
    if branch:
        expected_r = list(range((expected[0] - 1) * 693 + 1, 24256))
        if [row['r_updates'] for row in lr] != expected_r: raise ValueError('R trace has a clock/order gap')
    return means, lr


def source_suffix(report, branch, relative):
    suffix = '/' + NAMES[branch] + '/' + relative
    values = [digest for path, digest in report['input_and_implementation_sha256'].items()
              if path.replace('\\', '/').endswith(suffix)]
    if len(values) != 1: raise ValueError('Report does not uniquely bind runtime input: ' + relative)
    return values[0]


def load_validation(report_dir, branch, sources):
    report_dir = Path(report_dir); report_path = bound(report_dir / 'analysis_report.json', None, sources)
    report = read(report_path)
    if (report.get('status') != 'passed' or report.get('branch') != branch
            or report.get('format') != 'freeze35_independent_branch_analysis_v1'
            or report.get('primary_metric') != PRIMARY or report.get('primary_estimand') != 'sample mean'
            or report.get('inputs_unchanged') is not True): raise ValueError('Need accepted independent branch analysis')
    table_path = bound(report_dir / 'METRICS.csv', report['artifact_sha256']['METRICS.csv'], sources)
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['branch'] == branch and r['split'] == 'validation']
    if len(rows) != len(VAL_EPOCHS[branch]) or {int(r['epoch']) for r in rows} != set(VAL_EPOCHS[branch]):
        raise ValueError('Validation CSV omits historical/start/replay nodes')
    values = []
    for epoch in VAL_EPOCHS[branch]:
        label = f'{branch}_R{epoch}_validation'; point = report['models'][label]
        audit_path = Path(point['audit_path'])
        if not audit_path.is_file(): audit_path = report_dir / 'audits' / (label + '.json')
        audit = read(bound(audit_path, point['audit_sha256'], sources))
        if (audit.get('passed') is not True or (audit.get('examples'), audit.get('users')) != (71657, 4733)
                or audit.get('source_labels_verified') is not True or audit.get('source_cold_groups_verified') is not True
                or audit['metrics'][PRIMARY] != point['metrics'][PRIMARY]
                or audit['predictions_sha256'] != point['predictions_sha256']
                or audit['summary_sha256'] != point['summary_sha256']):
            raise ValueError('Validation curve is not bound to accepted full-split replay')
        row = next(row for row in rows if int(row['epoch']) == epoch)
        value = float(row[PRIMARY])
        if (int(row['examples']), int(row['users'])) != (71657, 4733) or not math.isfinite(value) or value != point['metrics'][PRIMARY]:
            raise ValueError('Validation CSV differs from its independent audit')
        values.append(dict(epoch=epoch, ndcg5=value, historical=bool(point['reference'])))
    return values, report


def pyplot():
    cache = Path(__file__).resolve().parents[3] / 'tmp/freeze35_matplotlib_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.dpi': 200, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return plt


def annotation(ax):
    ax.axvline(20, color='#666666', linestyle='--', linewidth=1, alpha=.7)
    ax.axvline(32, color='#999999', linestyle=':', linewidth=1, alpha=.8)
    ax.text(20.3, .98, 'A/C freeze after R20', transform=ax.get_xaxis_transform(), va='top', fontsize=8)
    ax.text(31.7, .90, 'B freezes at R32', transform=ax.get_xaxis_transform(), va='top', ha='right', fontsize=8)
    ax.set_xlabel('Original recommender epoch R'); ax.grid(alpha=.18)


def footer(fig, message='seed2024: one run per branch; curves do not establish cross-seed stability.'):
    fig.text(.08, .02, message, fontsize=8, color='#555555')


def save_figure(fig, output, name):
    fig.tight_layout(rect=(0, .055, 1, 1))
    for suffix in ('png', 'pdf'): fig.savefig(Path(output) / (name + '.' + suffix), bbox_inches='tight')


def fresh(output, protected):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) for p in protected):
        raise FileExistsError('Plots need a new directory outside the protected inputs')
    output.mkdir(parents=True, exist_ok=False); return output


def final_plots(runs_root, branch_reports, output):
    runs_root, branch_reports = Path(runs_root).resolve(), Path(branch_reports).resolve()
    sources, training, rates, validation, reports, duplicates = {}, {}, {}, {}, {}, {}
    for branch, name in NAMES.items():
        validation[branch], report = load_validation(branch_reports / branch, branch, sources); reports[branch] = report
        directory = runs_root / name
        complete = read(bound(directory / 'complete.json', source_suffix(report, branch, 'complete.json'), sources))
        request = read(bound(directory / 'request.json', source_suffix(report, branch, 'request.json'), sources))
        if (complete['branch'] != branch or complete['r_updates'] != 24255 or complete['phase'] != 45
                or request['parent_sha256'] != PARENTS[branch]): raise ValueError('Training/analysis runtime identity differs')
        log = bound(directory / 'training.jsonl', None, sources)
        rows, duplicates[branch] = training_rows(log, branch)
        training[branch], rates[branch] = epoch_means(rows, branch)
    overlap = 0
    for left, right in [('A', 'C'), ('B', 'C'), ('A', 'B')]:
        a = {r['r_updates']: r['learning_rate'] for r in rates[left]}
        b = {r['r_updates']: r['learning_rate'] for r in rates[right]}
        for update in a.keys() & b.keys():
            if not math.isclose(a[update], b[update], rel_tol=1e-12, abs_tol=1e-18):
                raise ValueError('Branches did not continue the same original R LR schedule')
            overlap += 1
    output = fresh(output, (runs_root, branch_reports)); plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for branch in NAMES:
        values = training[branch]
        ax.plot([r['epoch'] for r in values], [r['weighted_loss'] for r in values],
                color=COLORS[branch], marker='o', markersize=3, label=f'{branch}: {NAMES[branch]}')
    annotation(ax); ax.set_ylabel('R recommendation loss (weighted mean)')
    ax.set_title('R training loss: only actual new training epochs are shown'); ax.legend(loc='best', fontsize=8)
    footer(fig, 'A: new R27-35 from R26; B: new R33-35 from R32; C: replay R6-20 then frozen R21-35. One seed.')
    save_figure(fig, output, 'r_training_loss'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for branch, style in [('C', '-'), ('A', '--'), ('B', ':')]:
        ax.plot([r['epoch_coordinate'] for r in rates[branch]], [r['learning_rate'] for r in rates[branch]],
                color=COLORS[branch], linestyle=style, linewidth=2 if branch != 'B' else 3,
                label=f'{branch}: actual pre-step R LR')
    annotation(ax); ax.set_ylabel('Learning rate'); ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_title('R learning rate actually used; overlapping schedules coincide'); ax.legend(fontsize=8)
    footer(fig, 'Values come from learning_rates_used.R; no reconstructed LR or synthetic endpoint. seed2024.')
    save_figure(fig, output, 'r_learning_rate'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for branch in NAMES:
        values = validation[branch]
        ax.plot([r['epoch'] for r in values], [r['ndcg5'] for r in values], color=COLORS[branch],
                marker='o', markersize=4, label=f'{branch}: {NAMES[branch]}')
        old = [r for r in values if r['historical']]
        ax.scatter([r['epoch'] for r in old], [r['ndcg5'] for r in old], marker='o', s=65,
                   facecolors='white', edgecolors=COLORS[branch], zorder=4)
        if branch in ('A', 'C'):
            chosen = next(r for r in values if r['epoch'] == reports[branch]['selected_epoch'])
            ax.scatter([chosen['epoch']], [chosen['ndcg5']], marker='*', s=150, color=COLORS[branch], zorder=5)
    annotation(ax); ax.set_ylim(bottom=0); ax.set_ylabel('Single-list validation NDCG@5')
    ax.set_title('Complete validation split: 71,657 samples / 4,733 users'); ax.legend(fontsize=8)
    footer(fig, 'Hollow: historical/start reference. Stars: A/C best on common validation grid. B R32 is its old best. One seed.')
    save_figure(fig, output, 'validation_ndcg5'); plt.close(fig)
    for filename, data, fields in [('TRAINING_CURVES.csv',
        [dict(branch=b, **row) for b in NAMES for row in training[b]], ['branch', 'epoch', 'weighted_loss', 'lists', 'optimizer_updates']),
        ('VALIDATION_CURVES.csv', [dict(branch=b, **row) for b in NAMES for row in validation[b]], ['branch', 'epoch', 'ndcg5', 'historical'])]:
        with (output / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    if any(sha(path) != digest for path, digest in sources.items()): raise ValueError('Plot inputs changed while rendering')
    provenance = dict(status='generated_from_completed_independent_branch_analyses', seed=2024,
        source_sha256=sources, identical_training_duplicates_ignored=duplicates,
        common_original_lr_overlap_points_verified=overlap, missing_values_interpolated=False,
        loss='List-weighted mean of actual R-phase logged loss; no T objective mixed in',
        validation='Single five-item list NDCG@5; no test or aggregate@50 values plotted',
        one_seed_limitation=True, plots_are_not_new_metric_or_model_acceptance=True,
        output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'PLOT_PROVENANCE.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return provenance


def historical_smoke(log, output):
    sources = {}; log = bound(log, OLD_H0_LOG_SHA, sources)
    rows, duplicates = training_rows(log); means, _ = epoch_means(rows)
    output = fresh(output, (log.parent,)); plt = pyplot(); fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([r['epoch'] for r in means], [r['weighted_loss'] for r in means], color='#666666', marker='.')
    ax.set(xlabel='Original H0 epoch R', ylabel='List-weighted R loss', title='HISTORICAL PARSER / RENDER CHECK')
    ax.text(.5, .5, 'NOT FREEZE35 RESULTS', transform=ax.transAxes, ha='center', fontsize=24, alpha=.3)
    footer(fig, 'Only the SHA-locked old H0 trace is used. Old logs lack actual LR fields, so no LR curve is invented.')
    save_figure(fig, output, 'HISTORICAL_ONLY_parser_render_check'); plt.close(fig)
    result = dict(status='historical_parser_render_check_only_not_formal_results', source_sha256=sources,
        original_h0_epochs=[r['epoch'] for r in means], duplicate_rows=duplicates, no_new_model_or_metric_evaluation=True)
    (output / 'HISTORICAL_ONLY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root'); parser.add_argument('--branch-reports'); parser.add_argument('--output', required=True)
    parser.add_argument('--history-smoke', help='Explicit SHA-locked original H0 training.jsonl; parser/render check only')
    args = parser.parse_args(argv)
    if args.history_smoke:
        if args.runs_root or args.branch_reports: parser.error('Historical smoke cannot mix new-run inputs')
        result = historical_smoke(args.history_smoke, args.output)
    else:
        if not args.runs_root or not args.branch_reports: parser.error('Final plots require --runs-root and --branch-reports')
        result = final_plots(args.runs_root, args.branch_reports, args.output)
    print(json.dumps({'status': result['status'], 'output': str(Path(args.output).resolve())}))


if __name__ == '__main__': main()
