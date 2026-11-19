"""Plot six source-verified full-validation summaries; never recompute metrics."""
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
SOURCE = TASK / 'H1_R35_STAGE_EVIDENCE.json'
SOURCE_SHA = '0c8ccbb09c3b0ef69c70f3aca2d37ea59d3e76f24d032b7cae32b4c3107b75f9'
STYLE = TASK / 'plot_results.py'
STYLE_SHA = 'b6c0642d17c3f82cb54ff91d9507471a081b9f14e5065387122f3f9738c69cc4'
NAME = 'H1_full_validation_R20_to_R35_stage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA and sha(STYLE) == STYLE_SHA, 'Plot source/style bytes changed'
    for suffix in ('png', 'pdf', 'csv'):
        assert not (OUT / (NAME + '.' + suffix)).exists(), 'Never overwrite existing figure artifacts'
    assert not (OUT / 'GENERATION_RECORD.json').exists()
    evidence = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert evidence['status'] == 'full_validation_source_sha_and_protocol_verified'
    assert evidence['independent_per_sample_audit'] is False and evidence['selection_pending'] is False
    summary = evidence['summary']
    assert (summary['split'], summary['full_split'], summary['examples'], summary['beam'],
            summary['inference_precision'], summary['seed']) == ('validation', True, 71657, 32, 'fp32', 2024)
    selection = evidence['selection_verification']
    grid = selection['verified_validation_grid']
    epochs = [20, 23, 26, 29, 32, 35]
    assert set(grid) == {str(e) for e in epochs} and selection['selected_epoch'] == 32
    values = [grid[str(e)]['ndcg5'] for e in epochs]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    assert max(epochs, key=lambda e: grid[str(e)]['ndcg5']) == 32
    for e, value in zip(epochs, values):
        assert value == selection['selection']['candidates'][str(e)]['ndcg5']
    # Import only the bound helper; own font cache stays in the new figure dir.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('verified_freeze35_plot_style', STYLE)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.dpi': 200, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 5.7))
    blue = helper.COLORS['A']
    ax.plot(epochs[:3], values[:3], '--o', color=blue, linewidth=1.8, markersize=6,
            label='Earlier full-validation nodes')
    ax.plot(epochs[2:], values[2:], '-o', color=blue, linewidth=2, markersize=6,
            label='R26 to R35 continuation')
    ax.axvline(26, color='#D55E00', linestyle=':', linewidth=1.2, alpha=.8)
    ax.scatter([26], [grid['26']['ndcg5']], marker='s', s=72, color='#D55E00', zorder=5,
               label='This continuation starts at R26')
    ax.scatter([32], [grid['32']['ndcg5']], marker='*', s=180, color='#009E73', edgecolors='white',
               linewidths=.6, zorder=6, label='Validation best: R32')
    for epoch, value in zip(epochs, values):
        ax.annotate(f'{value:.8f}', (epoch, value), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    ax.annotate('R26: start of this continuation', xy=(26, grid['26']['ndcg5']), xytext=(22.5, .0052),
                fontsize=9, color='#9A3D00', arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1))
    ax.annotate('Best on the shared validation grid: R32', xy=(32, grid['32']['ndcg5']), xytext=(28.1, .0070),
                fontsize=9, color='#006C4D', arrowprops=dict(arrowstyle='->', color='#009E73', lw=1))
    ax.set(xlim=(18.5, 36.5), ylim=(0, .0125), xticks=epochs,
           xlabel='Original recommender epoch R', ylabel='Single-list NDCG@5',
           title='H1 full validation through R35 (validation best: R32)')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(alpha=.18)
    ax.legend(loc='lower left', fontsize=8.5, frameon=False, ncol=2)
    fig.text(.08, .027, 'Full validation: 71,657 samples | beam32 | FP32 | seed2024', fontsize=8.5, color='#444444')
    fig.text(.08, .003, 'Verified summary sources; independent per-sample audit pending. Validation only - NOT TEST.',
             fontsize=8.5, color='#7A3300')
    helper.save_figure(fig, OUT, NAME)
    plt.close(fig)
    rows = [dict(epoch=e, ndcg5=grid[str(e)]['ndcg5'], source_kind='historical' if e <= 26 else 'new',
                 continuation_start=e == 26, validation_best=e == 32,
                 summary_sha256=grid[str(e)]['summary_sha256'], checkpoint_sha256=grid[str(e)]['checkpoint_sha256'],
                 evidence_sha256=SOURCE_SHA, examples=71657, beam=32, inference_precision='fp32',
                 independent_per_sample_audit=False, split='validation') for e in epochs]
    with (OUT / (NAME + '.csv')).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assert sha(SOURCE) == SOURCE_SHA and sha(STYLE) == STYLE_SHA, 'Input changed during plotting'
    record = dict(status='stage_summary_plot_generated', scope='Validation summaries only; not test or per-sample audit',
        generated_utc=datetime.now(timezone.utc).isoformat(), source=str(SOURCE), source_sha256=SOURCE_SHA,
        source_checked_utc=evidence['checked_utc'], source_field='selection_verification.verified_validation_grid',
        style_helper=str(STYLE), style_helper_sha256=STYLE_SHA, style='Bound helper.save_figure and matching rcParams',
        script_sha256=sha(__file__), examples=71657, beam=32, inference_precision='fp32', seed=2024,
        epochs=epochs, selected_epoch=32, continuation_start_epoch=26, y_axis_starts_at_zero=True,
        all_six_values_annotated=True, independent_per_sample_audit=False, metrics_recomputed=False,
        training_or_gpu_or_platform_used=False, visual_inspection='pending',
        outputs={NAME + '.' + suffix: dict(bytes=(OUT / (NAME + '.' + suffix)).stat().st_size,
                   sha256=sha(OUT / (NAME + '.' + suffix))) for suffix in ('png', 'pdf', 'csv')},
        environment=dict(python=sys.version, matplotlib=matplotlib.__version__))
    with (OUT / 'GENERATION_RECORD.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=record['status'], selected_epoch=32, rows=len(rows), source_sha256=SOURCE_SHA)))


if __name__ == '__main__':
    main()
