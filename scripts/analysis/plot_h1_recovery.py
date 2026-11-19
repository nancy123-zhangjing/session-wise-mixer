"""Export offline PNG/PDF figures from completed H1 recovery artifacts.

Full validation and the fixed diagnostic subset occupy separate panels. The
historical H1 trajectory is authenticated archive context, not a rerun control.
No training, decoder, torch, remote, or platform API is imported.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from scripts.analysis.analyze_h1_recovery import (
    PARENT_SHA256, PRIMARY_METRIC, ROOT, collect_evaluations, expected_evaluations,
    read_json, write_json,
)
from scripts.listwise_diffgrm.subset import file_sha256


HISTORY_MANIFEST_SHA256 = '768ffba5dcb22f6223b3c731cd7292e1b4da26b7b6375b5d5308478a01b2e81c'
DEFAULT_HISTORY_ROOT = ROOT / 'outputs/stage09_restore/goal_20261027_r1'
DEFAULT_HISTORY_MANIFEST = ROOT / 'outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_H1_R1.json'
STAGES = ['initial_R20', 'after_T240', 'R21', 'R22', 'R23', 'after_T280', 'R24', 'R25', 'R26']
STAGE_LABELS = ['R20', 'after\nT240', 'R21', 'R22', 'R23', 'after\nT280', 'R24', 'R25', 'R26']
COLORS = dict(decay='#0072B2', freeze='#D55E00', historical='#727272')


def finite(value, name, *, minimum=0, maximum=None):
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'Invalid finite plotted value: {name}')
    return float(value)


def provenance(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path))


def historical_summaries(root=DEFAULT_HISTORY_ROOT, manifest_path=DEFAULT_HISTORY_MANIFEST):
    """Authenticate only the requested summaries against the accepted H1 archive."""
    root, manifest_path = Path(root), Path(manifest_path)
    manifest = read_json(manifest_path)
    if (file_sha256(manifest_path) != HISTORY_MANIFEST_SHA256
            or any(manifest.get(key) != value for key, value in
                   dict(schema='stage09_completed_run_backup_v1', status='PASS', run='H1').items())):
        raise ValueError('Historical H1 manifest is not the accepted archive manifest')
    result, sources = {}, [provenance(manifest_path)]
    for epoch in (20, 23, 26):
        relative = f'runs/H1/validation_R{epoch}/summary.json'
        matches = [row for row in manifest['files'] if row['path'] == relative]
        path = root / relative
        if len(matches) != 1 or file_sha256(path) != matches[0]['sha256'] or path.stat().st_size != matches[0]['bytes']:
            raise ValueError(f'Historical R{epoch} summary differs from accepted archive bytes')
        value = read_json(path)
        expected = dict(split='validation', architecture='H', seed=2024, beam=32, examples=71657,
                        full_split=True, subset=None, r_updates=693 * epoch, t_updates={20: 200, 23: 240, 26: 280}[epoch])
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError(f'Historical R{epoch} full-validation protocol differs')
        finite(value['metrics'][PRIMARY_METRIC], 'historical NDCG@5', maximum=1)
        result[epoch] = value
        sources.append(provenance(path))
    return result, sources


def training_series(path, policy):
    """Use the LR actually recorded before optimizer.step, not post-step LR.

    Resume replay can append the same logical update. Identical plotted fields
    are deduplicated; conflicting clocks/LRs fail instead of selecting a version.
    Inactive optimizer parameter-group LRs are never plotted as executed steps.
    """
    records, duplicates = {}, 0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                raise ValueError('Blank training record')
            row = json.loads(line)
            keys = ('phase_index', 'update', 'phase', 'epoch', 'r_updates', 't_updates', 'learning_rates_used')
            value = {key: row[key] for key in keys}
            key = (row['phase_index'], row['update'])
            if key in records:
                if records[key] != value:
                    raise ValueError('Conflicting duplicate training update clocks/LRs')
                duplicates += 1
            else:
                records[key] = value
    r, t, result = 13860, 200, dict(R=[], codebook=[], mixer=[], duplicate_replayed_records=duplicates)
    expected_keys = set()
    for phase, epoch, kind in [(25, 20, 'T'), (26, 21, 'R'), (27, 22, 'R'), (28, 23, 'R'),
                               (29, 23, 'T'), (30, 24, 'R'), (31, 25, 'R'), (32, 26, 'R')]:
        count = 0 if kind == 'T' and policy == 'freeze' else (40 if kind == 'T' else 693)
        for update in range(1, count + 1):
            key = (phase, update)
            expected_keys.add(key)
            if key not in records:
                raise ValueError(f'Missing recorded optimizer update {policy}/{key}')
            row = records[key]
            before_r, before_t = r, t
            r += kind == 'R'
            t += kind == 'T'
            if (row['phase'], row['epoch'], row['r_updates'], row['t_updates']) != (kind, epoch, r, t):
                raise ValueError('Training update clocks disagree with the fixed recovery sequence')
            lrs = row['learning_rates_used']
            for group in ('R',) if kind == 'R' else ('codebook', 'mixer'):
                result[group].append([before_r if kind == 'R' else before_t, finite(lrs[group], group + ' LR')])
    if set(records) != expected_keys:
        raise ValueError('Unexpected training updates outside the fixed recovery workload')
    return result


def catalog_series(run_dir, policy, catalog_size):
    root = Path(run_dir) / 'catalog'
    labels = ['initial_R20'] + ([f'T{step:03d}' for step in range(210, 281, 10)] if policy == 'decay' else [])
    rows, sources = [], []
    for index, label in enumerate(labels):
        path = root / (label + '.json')
        row = read_json(path)
        t = 200 if index == 0 else int(label[1:])
        r = 13860 if t <= 240 else 15939
        expected = dict(schema_version=1, scope='full_catalog_excluding_pad', selection='fp32_deterministic_nearest_centroid',
                        num_items=catalog_size, n_digit=4, codebook_size=256, r_updates=r, t_updates=t, event='catalog/' + label)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Catalog diagnostic schema/clock differs: {policy}/{label}')
        if len(row['per_slot']) != 4:
            raise ValueError('Expected four slot diagnostics')
        for slot, value in enumerate(row['per_slot']):
            if value['slot'] != slot or value['used_code_count'] != row['used_codes_per_slot'][slot]:
                raise ValueError('Per-slot diagnostic identity differs')
            finite(value['used_code_count'], 'used codes', minimum=1, maximum=256)
            finite(value['effective_code_count'], 'exp(entropy)', minimum=1, maximum=256 + 1e-9)
        for key in ('collision_item_percent', 'mixer_delta_over_base_global_norm', 'mean_nearest_reconstruction_sse', 'unique_sid_count'):
            finite(row[key], key)
        if not math.isclose(row['collision_item_percent'], 100 * row['collision_item_count'] / catalog_size, abs_tol=1e-10):
            raise ValueError('Collision percentage must count all items in non-singleton SID buckets')
        for key in ('sid_item_change_fraction', 'effective_displacement_over_previous_global_norm'):
            value = row['previous_change'][key]
            if (index == 0) != (value is None):
                raise ValueError('Adjacent-state diagnostic is missing or fabricated at the first snapshot')
            if value is not None:
                finite(value, key, maximum=1 if key == 'sid_item_change_fraction' else None)
        rows.append(row)
        sources.append(provenance(path))
    return rows, sources


def extracted_subset(path, manifest):
    """Verify the saved full-evaluation extraction using its stored row metrics.

    This is a consistency check, not independent prediction-to-metric replay;
    --analysis-report supplies the separately recomputed metric evidence.
    """
    wanted = {row['sample_id'] for row in manifest['rows']}
    found, total = set(), 0.0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['sample_id'] in wanted:
                if row['sample_id'] in found:
                    raise ValueError('Repeated diagnostic subset sample in full predictions')
                found.add(row['sample_id'])
                total += finite(row['metrics'][PRIMARY_METRIC], 'subset NDCG@5', maximum=1)
    value = read_json(Path(path).parent / 'diagnostic_subset.json')
    if (found != wanted or value['examples'] != manifest['count']
            or not math.isclose(value['metrics'][PRIMARY_METRIC], total / len(wanted), abs_tol=2e-6)):
        raise ValueError('Saved diagnostic subset extraction disagrees with full predictions')
    return value


def audited_metrics(report_path, arm, label, artifact, sources):
    report_path = Path(report_path)
    report = read_json(report_path)
    if (report.get('status') != 'passed' or report.get('format') != 'stage09_h1_recovery_analysis_v1'
            or report.get('parent_sha256') != PARENT_SHA256 or report.get('primary_estimand') != 'sample mean'):
        raise ValueError('Independent analysis report is not an accepted recovery analysis')
    record = report['evaluations'][arm][label]
    path = (report_path.parent / record['audit']).resolve()
    if not path.is_relative_to(report_path.parent.resolve()) or file_sha256(path) != record['audit_sha256']:
        raise ValueError('Analysis audit path/hash differs from its report')
    audit = read_json(path)
    if (audit.get('passed') is not True or audit.get('source_labels_verified') is not True
            or audit.get('source_cold_groups_verified') is not True
            or audit['predictions_sha256'] != file_sha256(artifact['path'])
            or audit['summary_sha256'] != file_sha256(artifact['path'].parent / 'summary.json')):
        raise ValueError('Independent analysis is not bound to the plotted accepted predictions')
    sources.append(provenance(path))
    return audit


def load_data(decay_run, freeze_run, subset_path, *, history_root=DEFAULT_HISTORY_ROOT,
              history_manifest=DEFAULT_HISTORY_MANIFEST, analysis_report=None):
    history, sources = historical_summaries(history_root, history_manifest)
    manifest = read_json(subset_path)
    if (manifest.get('count'), manifest.get('population'), manifest.get('selection'), len(manifest.get('rows', []))) != (
            8192, 71657, 'sha256_utf8_sample_id_then_sample_id', 8192):
        raise ValueError('Expected the locked unsalted 8192-row recovery subset')
    if len({row['sample_id'] for row in manifest['rows']}) != 8192:
        raise ValueError('Duplicate locked subset sample IDs')
    sources.append(provenance(subset_path))
    if analysis_report:
        report = read_json(analysis_report)
        if report['subset_file_sha256'] != file_sha256(subset_path):
            raise ValueError('Analysis report and plot use different subset manifest bytes')
        sources.append(provenance(analysis_report))
    data = dict(synthetic=False, history={str(epoch): row['metrics'][PRIMARY_METRIC] for epoch, row in history.items()},
                arms={}, sources=sources, metric_evidence='independent metric replay' if analysis_report else 'receipt-checked saved metrics')
    forks, initial = {}, {}
    for policy, root in [('decay', Path(decay_run)), ('freeze', Path(freeze_run))]:
        fork, artifacts = collect_evaluations(root, policy)
        forks[policy], initial[policy] = fork, artifacts['initial_R20']['binding']
        sources.extend([provenance(root / name) for name in ('fork.json', 'complete.json', 'training.jsonl')])
        rows, catalog_sources = catalog_series(root, policy, history[20]['catalog_size'])
        sources.extend(catalog_sources)
        arm = dict(training=training_series(root / 'training.jsonl', policy), catalog=rows, full=[], subset=[])
        for label, _, _, full in expected_evaluations(policy):
            artifact = artifacts[label]
            path = artifact['path']
            summary = read_json(path.parent / 'summary.json')
            for key in ('dataset_binding', 'catalog_size', 'metric_protocol'):
                if summary[key] != history[20][key]:
                    raise ValueError(f'Historical and recovery evaluation protocols differ: {key}')
            if not full and summary['subset'] != {key: value for key, value in manifest.items() if key != 'rows'}:
                raise ValueError('Plotted subset summary differs from locked manifest')
            sources.extend([provenance(path), provenance(path.parent / 'summary.json'), provenance(path.parent.parent / 'complete.json')])
            audited = audited_metrics(analysis_report, policy, label, artifact, sources) if analysis_report else None
            metric = finite((audited or summary)['metrics'][PRIMARY_METRIC], PRIMARY_METRIC, maximum=1)
            if full:
                subset = audited['diagnostic_subset'] if audited else extracted_subset(path, manifest)
                if subset['examples'] != 8192:
                    raise ValueError('Incorrect extracted subset population')
                arm['full'].append(dict(epoch=int(label[1:]), ndcg5=metric))
                subset_metric = finite(subset['metrics'][PRIMARY_METRIC], 'extracted subset NDCG@5', maximum=1)
                sources.append(provenance(path.parent / 'diagnostic_subset.json'))
            else:
                subset_metric = metric
            arm['subset'].append(dict(stage=label, ndcg5=subset_metric, extracted_from_full=full))
        data['arms'][policy] = arm
    if initial['decay'] != initial['freeze']:
        raise ValueError('Arms do not share initial state/source/subset')
    for key in ('parent_hashes', 'execution_binding'):
        if forks['decay'][key] != forks['freeze'][key]:
            raise ValueError('Arms have different fork/source identity')
    return data


def render(data, output_root):
    """Render four figure families, plus exact plotting data and SHA provenance."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    files = []
    synthetic = data['synthetic']

    def save(fig, name, caption):
        if synthetic:
            fig.suptitle('SYNTHETIC SMOKE - NOT EXPERIMENT RESULTS', color='#B22222', fontsize=13, weight='bold')
        fig.text(.5, .01, caption, ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94 if synthetic else 1))
        for extension in ('png', 'pdf'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', metadata={'Title': name, 'Creator': 'plot_h1_recovery.py'})
            files.append(dict(file=path.name, sha256=file_sha256(path), bytes=path.stat().st_size))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    history = data['history']
    for policy, arm in data['arms'].items():
        full = [dict(epoch=20, ndcg5=history['20'])] + arm['full']
        axes[0].plot([row['epoch'] for row in full], [row['ndcg5'] for row in full],
                     color=COLORS[policy], marker='o', label=policy.capitalize(), linestyle='-' if policy == 'decay' else '--')
        subset = arm['subset']
        axes[1].plot([STAGES.index(row['stage']) for row in subset], [row['ndcg5'] for row in subset],
                     color=COLORS[policy], marker='o', markersize=4, label=policy.capitalize(), linestyle='-' if policy == 'decay' else '--')
        extracted = [row for row in subset if row['extracted_from_full']]
        axes[1].scatter([STAGES.index(row['stage']) for row in extracted], [row['ndcg5'] for row in extracted],
                        s=55, marker='s', facecolors='white', edgecolors=COLORS[policy], zorder=4)
    axes[0].plot([20, 23, 26], [history[str(epoch)] for epoch in (20, 23, 26)], color=COLORS['historical'],
                 linestyle=':', marker='x', label='Historical H1 (archive summary)')
    axes[0].scatter([20], [history['20']], color='black', marker='D', s=40, zorder=5, label='Shared R20 (reused full evaluation)')
    axes[0].set(title='Full validation: 71,657 samples', xlabel='Completed R epoch', ylabel='Sample-mean single-list NDCG@5', xticks=[20, 23, 26])
    axes[1].set(title='Fixed diagnostic subset: 8,192 samples', xlabel='Continuation boundary (T points only measured for Decay)',
                ylabel='Sample-mean single-list NDCG@5', xticks=range(len(STAGES)), xticklabels=STAGE_LABELS)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc='best')
    save(fig, 'validation_ndcg5', 'Full and subset results are separate populations. Squares: subset extracted from R23/R26 full predictions.\n'
         'R26 is the fixed endpoint; R23/subset are process diagnostics. Historical H1 is context, not a rerun control.')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for axis, group, title in zip(axes, ('R', 'codebook', 'mixer'), ('Recommender', 'Base codebooks', 'Full Mixer')):
        for policy, arm in data['arms'].items():
            values = arm['training'][group]
            if values:
                axis.plot([v[0] for v in values], [v[1] for v in values], color=COLORS[policy],
                          linestyle='-' if policy == 'decay' else '--', label=policy.capitalize(), linewidth=1.6)
        axis.set(title=title, xlabel=f'Completed {"R" if group == "R" else "T"} updates before step', ylabel='Actual learning rate used')
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    save(fig, 'learning_rates', 'From learning_rates_used, filtered to the optimizer actually stepped. Freeze has no T steps.\n'
         'No scheduler reconstruction; x is the pre-step count (last T point is u=279).')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    for slot, axis in enumerate(axes.flat):
        for policy, arm in data['arms'].items():
            rows = arm['catalog']
            x = [row['t_updates'] for row in rows]
            for field, style, marker, label in [('used_code_count', '-', 'o', 'used'), ('effective_code_count', '--', 's', 'exp(entropy)')]:
                axis.plot(x, [row['per_slot'][slot][field] for row in rows], color=COLORS[policy],
                          linestyle=style if len(rows) > 1 else 'None', marker=marker, markersize=4,
                          markerfacecolor='white' if field == 'effective_code_count' else COLORS[policy], label=f'{policy.capitalize()}: {label}')
        axis.axhline(256, color='#AAAAAA', linewidth=.6)
        axis.set(title=f'Slot {slot + 1}', xlabel='Completed T updates', ylabel='Number of codes', ylim=(0, 267), xticks=range(200, 281, 20))
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'code_usage', 'Full-catalog temporary FP32 nearest-centroid encoding; formal SIDs refresh only at T-block boundaries.\n'
         'Freeze has one recorded T200 snapshot; no unmeasured flat continuation is drawn.')

    fields = [
        ('Unique SID combinations', 'unique_sid_count', None, 1, 'Count'),
        ('Items sharing a SID', 'collision_item_percent', None, 1, 'Percent of catalog items'),
        ('Mixer correction / base', 'mixer_delta_over_base_global_norm', None, 1, 'Global Frobenius-norm ratio'),
        ('Nearest reconstruction', 'mean_nearest_reconstruction_sse', None, 1, 'SSE per catalog item'),
        ('SID change vs previous snapshot', 'sid_item_change_fraction', 'previous_change', 100, 'Percent of catalog items'),
        ('Effective codebook change vs previous', 'effective_displacement_over_previous_global_norm', 'previous_change', 1, 'Relative global norm'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for axis, (title, field, parent, scale, ylabel) in zip(axes.flat, fields):
        for policy, arm in data['arms'].items():
            values = [(row['t_updates'], (row[parent] if parent else row)[field]) for row in arm['catalog']]
            values = [(x, y * scale) for x, y in values if y is not None]
            if values:
                axis.plot([v[0] for v in values], [v[1] for v in values], color=COLORS[policy],
                          linestyle='-' if len(values) > 1 else 'None', marker='o', markersize=4, label=policy.capitalize())
        axis.set(title=title, xlabel='Completed T updates', ylabel=ylabel, xticks=range(200, 281, 20))
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'catalog_diagnostics', 'Collision counts all items in non-singleton SID buckets, not N minus unique SIDs.\n'
         'Adjacent-state change is undefined at the first snapshot; omitted values are not zero.')
    write_json(output / 'plot_data.json', data)
    manifest = dict(format='stage09_h1_recovery_figures_v1', status='synthetic_smoke_only' if synthetic else 'exported',
        parent_sha256=None if synthetic else PARENT_SHA256, metric_evidence=data['metric_evidence'],
        historical_evidence='Synthetic' if synthetic else 'Archive-authenticated saved summaries; R23/R26 not independently replayed',
        primary_metric=PRIMARY_METRIC, primary_estimand='sample mean',
        source_files=data['sources'], plot_data_sha256=file_sha256(output / 'plot_data.json'),
        script_sha256=file_sha256(__file__), figures=files,
        note='These trajectory figures do not replace the three prespecified paired-bootstrap comparisons.')
    write_json(output / 'FIGURE_MANIFEST.json', manifest)
    return manifest


def synthetic_data():
    """Explicitly fictional display fixture; cannot be presented as experiment data."""
    data = dict(synthetic=True, history={'20': .01, '23': .007, '26': .006}, arms={}, sources=[], metric_evidence='synthetic only')
    for policy in ('decay', 'freeze'):
        training = dict(R=[[step, .0004 - (step - 13860) * .00024 / 4158] for step in range(13860, 18018, 7)],
                        codebook=[], mixer=[], duplicate_replayed_records=0)
        if policy == 'decay':
            for group, peak in [('codebook', 1e-5), ('mixer', 1e-4)]:
                training[group] = [[step, peak * .5 * (1 + math.cos(math.pi * step / 400))] for step in range(200, 280)]
        catalog = []
        for i, t in enumerate(range(200, 281, 10) if policy == 'decay' else [200]):
            catalog.append(dict(t_updates=t, per_slot=[dict(used_code_count=210 - i * 4 + slot,
                effective_code_count=125 - i * 3 + slot) for slot in range(4)], unique_sid_count=9500 - i * 110,
                collision_item_percent=13 + i * 2, mixer_delta_over_base_global_norm=.82 + i * .06,
                mean_nearest_reconstruction_sse=17 - i * .2,
                previous_change=dict(sid_item_change_fraction=.04 + i * .005 if i else None,
                    effective_displacement_over_previous_global_norm=.01 + i * .001 if i else None)))
        labels = [item[0] for item in expected_evaluations(policy)]
        data['arms'][policy] = dict(training=training, catalog=catalog,
            full=[dict(epoch=23, ndcg5=.0102 if policy == 'decay' else .0097), dict(epoch=26, ndcg5=.0105 if policy == 'decay' else .0098)],
            subset=[dict(stage=label, ndcg5=.0095 + index * (.00009 if policy == 'decay' else .00002),
                         extracted_from_full=label in ('R23', 'R26')) for index, label in enumerate(labels)])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decay-run')
    parser.add_argument('--freeze-run')
    parser.add_argument('--subset', help='Locked validation_subset_8192.json')
    parser.add_argument('--analysis-report', help='Optional accepted analyze_h1_recovery.py report; uses independently replayed metrics')
    parser.add_argument('--history-root', default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument('--history-manifest', default=str(DEFAULT_HISTORY_MANIFEST))
    parser.add_argument('--output-root', required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--synthetic-smoke', action='store_true', help='Render explicitly watermarked fictional data without reading run artifacts')
    args = parser.parse_args(argv)
    if args.synthetic_smoke:
        if any((args.decay_run, args.freeze_run, args.subset, args.analysis_report)):
            parser.error('Synthetic smoke cannot be combined with experiment inputs')
        data = synthetic_data()
    else:
        if not all((args.decay_run, args.freeze_run, args.subset)):
            parser.error('--decay-run, --freeze-run, and --subset are required for experiment figures')
        data = load_data(args.decay_run, args.freeze_run, args.subset, history_root=args.history_root,
                         history_manifest=args.history_manifest, analysis_report=args.analysis_report)
    result = render(data, args.output_root)
    print(json.dumps(dict(status=result['status'], figures=len(result['figures']), output=str(Path(args.output_root).resolve()))))


if __name__ == '__main__':
    main()
