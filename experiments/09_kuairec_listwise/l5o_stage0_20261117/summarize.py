"""Summarize completed short-test receipts; never launch training or inference."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / ('results' if (HERE / 'results/COMPLETE.json').exists() else 'receipts')


def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def main():
    complete = load('COMPLETE.json')
    runs = {f'{arm}_{mode}': load(f'{arm}_{mode}.json')
            for arm in ('H0', 'H1') for mode in ('reference', 'fast')}
    gpu = []
    with (RESULTS / 'gpu.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                gpu.append((t, float(row[' utilization.gpu [%]'].split()[0])))
            except (ValueError, KeyError, IndexError):
                continue  # An incomplete last monitor row is not a measurement.
    rows = []
    for label, run in runs.items():
        assert [len(b['rows']) for b in run['blocks']] == [80, 40, 80]
        for idx, block in enumerate(run['blocks']):
            steady = block['rows'][5:]
            start = block['start_unix'] + sum(r['seconds'] for r in block['rows'][:5])
            values = [v for t, v in gpu if start <= t <= block['end_unix']]
            rows.append(dict(label=label, block=idx, kind=block['kind'],
                seconds_per_update=block['mean_seconds'],
                mean_loss=statistics.mean(r['loss'] for r in steady),
                mean_data_seconds=statistics.mean(r['data_s'] for r in steady),
                allocated_gib=block['peak_allocated_gib'],
                gpu_utilization_mean=statistics.mean(values) if values else None))
    with (HERE / 'TIMINGS.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    candidates = []
    for batch in (128, 256):
        chosen = [next(e for e in runs[arm + '_fast']['evaluations'] if e['batch'] == batch)
                  for arm in ('H0', 'H1')]
        if all(not e.get('oom') and e['identical_lists_fraction'] >= .99 for e in chosen):
            candidates.append((sum(e['seconds'] for e in chosen), batch))
    batch = min(candidates)[1] if candidates else 4
    estimates = {}
    for arm in ('H0', 'H1'):
        run = runs[arm + '_fast']
        r_seconds = statistics.mean(run['blocks'][i]['mean_seconds'] for i in (0, 2))
        t_seconds = run['blocks'][1]['mean_seconds']
        evaluation = next(e for e in run['evaluations'] if e['batch'] == batch)
        estimates[arm] = dict(training_hours=(r_seconds * 24255 + t_seconds * 400) / 3600,
            validation_hours=evaluation['seconds'] * 7176 / 256 * 35 / 3600,
            note='Linear extrapolation only; excludes startup, checkpoint I/O and final test; short run is in LR warmup')
    acceptance = dict(main_shorttest_passed=complete['passed'], common_eval_batch=batch,
        common_execution=dict(microbatch=1024, effective_batch=1024, gpu_static_ids=True,
            fused_adamw=True, compile=False, deterministic_algorithms=True, tf32=False),
        restore={a: runs[a + '_fast']['restore']['exact_next_update'] for a in ('H0', 'H1')},
        compile_probe=next((load(n) for n in ('COMPILE_STATUS.json', 'COMPILE_OUTCOME.json', 'COMPILE_PROBE.json')
                            if (RESULTS / n).exists()), {'pending': True}),
        estimates=estimates, boundary='Execution acceptance only; no recommendation-quality or paper-reproduction claim')
    (HERE / 'EXECUTION_ACCEPTANCE.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for r, arm in enumerate(('H0', 'H1')):
        for c, kind in enumerate(('R', 'T')):
            ax = axes[r, c]
            for mode, color in [('reference', '#64748b'), ('fast', '#d97706')]:
                losses = [v['loss'] for b in runs[arm + '_' + mode]['blocks'] if b['kind'] == kind for v in b['rows']]
                ax.plot(range(1, len(losses) + 1), losses, label=mode, color=color, linewidth=1.5, alpha=.9)
            if kind == 'R':
                ax.axvline(80.5, color='#94a3b8', linestyle=':', linewidth=1)
            ax.set(title=f'{arm}: {kind} updates', xlabel=f'{kind} update', ylabel='Training loss')
            ax.grid(alpha=.2); ax.legend()
    fig.suptitle('H100 execution check | global batch 1024 | history 128 | seed 2024')
    fig.tight_layout(); fig.savefig(HERE / 'loss_comparison.png', dpi=180); plt.close(fig)
    print(json.dumps(acceptance, indent=2))


if __name__ == '__main__':
    main()
