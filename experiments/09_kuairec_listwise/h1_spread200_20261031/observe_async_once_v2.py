"""One read-only operational snapshot; never runs a model or starts a monitor."""
import argparse
import json
from pathlib import Path
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def tail_json(path):
    path=Path(path)
    if not path.is_file():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-8192))
        lines=f.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (ValueError,TypeError):pass
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if root!=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'):
        raise ValueError('Wrong owned root')
    jobs=root/'jobs/formal_async_r2';run=root/'runs/H1-Spread200-R35';now=time.time()
    status=read(jobs/'status.json');train=tail_json(root/'runs/H1-Spread200-R35/training.jsonl')
    tasks=[]
    for path in sorted((jobs/'tasks').glob('phase*.json')):
        if path.name.endswith('.done.json'):continue
        task=read(path);done=path.with_suffix('.done.json')
        progress=tail_json(path.with_suffix('.log'))
        entry=dict(phase=int(path.stem[5:]),kind=task['kind'],epoch=task['epoch'],label=task['label'],
            completed_receipt_exists=done.exists())
        if progress and 'processed' in progress:
            entry.update(processed=progress['processed'],total=progress['total'])
        if done.exists():
            receipt=read(done)
            entry.update(receipt_status=receipt['status'],readonly_state_unchanged=receipt['state_before']==receipt['state_after'])
        tasks.append(entry)
    result=read(jobs/'result.json') if (jobs/'result.json').exists() else None
    selection=read(run/'selection.json') if (run/'selection.json').exists() else None
    tests=[]
    for directory in sorted((run/'evaluations').glob('test_R*')):
        entry=dict(epoch=int(directory.name.split('_R')[1]),completed_receipt_exists=(directory/'complete.json').exists())
        for path in sorted(directory.glob('attempt*/summary.json')):
            summary=read(path)
            entry.update(summary_examples=summary['examples'],summary_split=summary['split'])
        tests.append(entry)
    test_progress=tail_json(jobs/'test.log')
    elapsed_upper=now-1793421774
    value=dict(observed_unix=now,heartbeat_age_seconds=now-status['heartbeat_unix'],stage=status['stage'],
        stop_reason=status['stop_reason'],process_result=None if result is None else result['status'],
        training={k:train[k] for k in ('epoch','phase','phase_index','update','r_updates','t_updates')},
        tasks=tasks,last_gpu_sample=tail_json(jobs/'gpu_utilization.jsonl'),
        selected_epoch=None if selection is None else selection['selected']['epoch'],
        selection_candidate_epochs=None if selection is None else [r['epoch'] for r in selection['candidates']],
        tests=tests,test_progress=test_progress,
        cumulative_created_to_now_gpu_hours_upper_bound=(35813+elapsed_upper)/3600,
        limitation='Read-only process/artifact snapshot. Platform status must be queried separately; an absent completion receipt alone does not prove a task is live or dead.')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
