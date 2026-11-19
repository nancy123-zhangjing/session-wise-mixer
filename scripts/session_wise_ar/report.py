"""Combine completed AR and unchanged CF receipts; no training or inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from scripts.rqcd_mixer.common import sha256, write_json

ARMS = ('AR-T0', 'AR-T1', 'AR-D0', 'AR-D1', 'CF-M0', 'CF-M1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_evaluation(result, checkpoint_sha, split):
    if (result['identity']['checkpoint_sha256'] != checkpoint_sha or
            result.get('split', result['identity'].get('split')) != split):
        raise ValueError('Validation/test checkpoint or split identity mismatch')
    for metric in ('ndcg@5', 'recall@5', 'hr@5'):
        value = result['metrics'][metric]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid metric in completed receipt')


def metric_row(arm, point, epoch, step, t_updates, val, test, checksum):
    assert_evaluation(val, checksum, 'validation')
    assert_evaluation(test, checksum, 'test')
    return {'arm': arm, 'point': point, 'session_epochs': epoch, 'r_updates': step, 't_updates': t_updates,
            **{f'{split}_{key}': value['metrics'][key] for split, value in (('validation', val), ('test', test))
               for key in ('ndcg@5', 'recall@5', 'hr@5')}, 'checkpoint_sha256': checksum}


def ar_result(entry):
    root = Path(entry['train']); done = read(root/'complete.json')
    tests = read(Path(entry['test'])/'complete.json')
    arm = entry['arm']
    if done['arm'] != arm or tests['arm'] != arm or not done['progress']['done']:
        raise ValueError('AR run identity/incomplete state')
    rows, curve, seconds = [], [], 0.0
    for node in done['progress']['records']:
        seconds += node['validation']['seconds']
        curve.append({'arm': arm, 'session_epochs': node['session_epochs'], 'r_updates': node['r_updates'],
                      't_updates': node['t_updates'], 'phase_wall_seconds': node['train_seconds']+seconds,
                      **node['validation']['metrics'],
                      **{'loss_'+k: v for k, v in node['loss'].items()}})
    for point in ('best', 'final'):
        node = done[point]
        checkpoint = root/'checkpoints'/f"step_{node['step']:07d}.pt"
        if sha256(checkpoint) != node['checkpoint_sha256']:
            raise ValueError('AR checkpoint changed')
        rows.append(metric_row(arm, point, node['session_epochs'], node['r_updates'], node['t_updates'],
                               node['validation'], tests[point], node['checkpoint_sha256']))
    times = {'arm': arm, 'parameters': done['parameter_count'],
             'recommendation_seconds': done['progress']['train_seconds'], 'validation_seconds': seconds,
             'test_seconds': sum(v['seconds'] for v in {tests[k]['identity']['checkpoint_sha256']: tests[k]
                                                       for k in ('best','final')}.values())}
    if entry.get('pretrain'):
        pre = read(Path(entry['pretrain'])/'complete.json')
        if pre['selected_sha256'] != done['parent_sha256']:
            raise ValueError('AR pretraining result belongs to different parent')
        times['pretraining_seconds'] = pre['progress']['train_seconds']
    return rows, curve, times


def cf_result(entry):
    root = Path(entry['train']); done = read(root/'TRAINING_COMPLETE.json')
    tests, best, cfg = read(root/'TEST_COMPLETE.json'), read(root/'BEST.json'), read(root/'config.json')
    expected = 'B0' if entry['arm']=='CF-M0' else 'B1'
    if done['arm'] != expected or cfg.get('target', {}).get('mode') != 'cf':
        raise ValueError('CF comparison requires original CF B0/B1, not a content or hybrid run')
    events = [json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    times, epoch_seconds = 0.0, {}
    for event in events:
        times += event.get('phase_wall_seconds', 0)
        if event['kind'] == 'validation':
            times += event['seconds']; epoch_seconds[event['R']] = times
    vals = {v['identity']['R']: v for v in [read(p) for p in (root/'validation').glob('R*/summary.json')]}
    updates = math.ceil(cfg['split_counts']['train']/cfg['training']['effective_batch'])
    rows, curve = [], []
    for epoch, val in sorted(vals.items()):
        curve.append({'arm': entry['arm'], 'session_epochs': epoch, 'r_updates': epoch*updates,
                      't_updates': val['identity']['T'], 'phase_wall_seconds': epoch_seconds[epoch], **val['metrics']})
    for point, epoch, path, checksum in (('best',best['R'], best['checkpoint'], best['sha256']),
                                        ('final',done['R'],done['final_checkpoint'],done['final_sha256'])):
        if sha256(root/path) != checksum:
            raise ValueError('CF checkpoint changed')
        rows.append(metric_row(entry['arm'], point, epoch, epoch*updates, vals[epoch]['identity']['T'],
                               vals[epoch], tests[point], checksum))
    return rows, curve, {'arm':entry['arm'], 'recommendation_seconds': sum(e.get('phase_wall_seconds',0) for e in events),
                         'validation_seconds': sum(v['seconds'] for v in vals.values()),
                         'test_seconds': sum(v['seconds'] for v in {tests[k]['identity']['checkpoint_sha256']:tests[k]
                                                                   for k in ('best','final')}.values())}


def report(entries, output, plots=False):
    names = [e['arm'] for e in entries]
    if not names or len(set(names)) != len(names) or any(n not in ARMS for n in names):
        raise ValueError('Provide each explicitly named arm at most once')
    destination = Path(output).resolve()
    for entry in entries:
        for key in ('train', 'test', 'pretrain'):
            if key in entry:
                source = Path(entry[key]).resolve()
                if destination == source or source in destination.parents or destination in source.parents:
                    raise ValueError('Report output must be independent of experiment artifacts')
    rows, curves, times, parents, dataset_identities = [], [], [], {}, []
    for entry in entries:
        if entry['arm'].startswith('AR-'):
            done = read(Path(entry['train'])/'complete.json')
            family = entry['arm'][:4]
            parent = done['prepared_sha256']
            dataset_identities.append(done['input_binding']['dataset'])
        else:
            binding = read(Path(entry['train'])/'binding.json')
            family = 'CF'
            parent = (binding['prepared_sha256'], binding['teacher_sha256'])
            dataset_identities.append(binding['data']['dataset'])
        if family in parents and parents[family] != parent:
            raise ValueError('Within-family comparison requires the same shared parent/teacher')
        parents[family] = parent
        r, c, t = ar_result(entry) if entry['arm'].startswith('AR-') else cf_result(entry)
        rows.extend(r); curves.extend(c); times.append(t)
    if any(d != dataset_identities[0] for d in dataset_identities):
        raise ValueError('Cross-method comparison requires identical dataset/split bindings')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, values in (('comparison', rows), ('validation_nodes', curves), ('timing', times)):
        write_json(output/(name+'.json'), values)
        columns = list(dict.fromkeys(k for row in values for k in row))
        with (output/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(values)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = dict(zip(ARMS, ['#0072B2','#D55E00','#009E73','#CC79A7','#555555','#E69F00']))
        for x, filename, label in (('session_epochs','validation_epochs.png','Session-equivalent epochs'),
                                    ('phase_wall_seconds','validation_time.png','Accounted phase wall seconds')):
            fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
            for arm in names:
                points = [p for p in curves if p['arm']==arm]
                ax.plot([p[x] for p in points], [p['ndcg@5'] for p in points], marker='o', markersize=3,
                        color=colors[arm], label=arm)
            ax.set(xlabel=label, ylabel='Validation NDCG@5'); ax.grid(alpha=.25); ax.legend()
            fig.savefig(output/filename, dpi=180); plt.close(fig)
        # Loss scales differ between methods: draw each arm separately.
        for arm in names:
            points = [p for p in curves if p['arm']==arm]
            keys = sorted({k for p in points for k in p if k.startswith('loss_')})
            if not keys: continue
            fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
            for key in keys:
                selected = [p for p in points if key in p]
                ax.plot([p['session_epochs'] for p in selected], [p[key] for p in selected], label=key[5:])
            ax.set(xlabel='Session-equivalent epochs', ylabel='Training loss', title=arm)
            ax.grid(alpha=.25); ax.legend(); fig.savefig(output/f'{arm}_loss.png',dpi=180); plt.close(fig)
    text = ['# Five-arm comparison', '', 'Only completed receipts are included. Missing arms: '+', '.join(a for a in ARMS if a not in names),
            '', '| Arm | Point | Session epochs | Validation NDCG@5 | Test NDCG@5 |', '|---|---|---:|---:|---:|']
    text += [f"| {r['arm']} | {r['point']} | {r['session_epochs']:.4f} | {r['validation_ndcg@5']:.6f} | {r['test_ndcg@5']:.6f} |" for r in rows]
    text += ['', 'Within-family comparisons isolate the whole-pipeline Mixer intervention. Across-family comparisons are not compute-matched.',
             'A single seed is descriptive evidence, not statistical significance.',
             'Phase wall times exclude job startup, idle periods and unrecorded I/O; they are not GPU billing time. Missing timing/parameter fields are not estimated.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON list of arm/train/test/pretrain paths; CF needs arm/train only')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    report(read(args.manifest), args.output, args.plots)
