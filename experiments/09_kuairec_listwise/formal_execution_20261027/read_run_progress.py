"""Bounded, read-only file progress; no platform state or model loading."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
RUNS = ('warmup_J', 'warmup_H', 'J0', 'J1', 'H0', 'H1')
LIMIT = 16 * 1024


def metadata(path):
    try:
        value = path.stat()
        return dict(bytes=value.st_size, mtime_utc=datetime.fromtimestamp(
            value.st_mtime, timezone.utc).isoformat())
    except FileNotFoundError:
        return None


def small_json(path):
    info = metadata(path)
    if info is None or info['bytes'] > LIMIT:
        return None
    try:
        with path.open('rb') as stream:
            return json.loads(stream.read(LIMIT).decode('utf-8'))
    except (FileNotFoundError, UnicodeDecodeError, ValueError):
        return None


def tail_rows(path):
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            offset = max(0, size - LIMIT)
            stream.seek(offset)
            lines = stream.read(LIMIT).splitlines(keepends=True)
        if offset:
            lines = lines[1:]  # First retained line may start in the middle.
        for line in reversed(lines):
            if not line.endswith(b'\n'):
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(row, dict):
                yield row
    except FileNotFoundError:
        return


def snapshot(name):
    if name not in RUNS:
        raise ValueError('Unsupported run name')
    run = ROOT / 'runs' / name
    if run.is_symlink() or not run.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Run path escapes the owned Goal root')
    latest = next((r for r in tail_rows(run / 'training.jsonl')
                   if all(k in r for k in ('epoch', 'phase', 'r_updates', 't_updates'))), None)
    directories = [p for p in run.glob('validation_R*') if p.is_dir() and not p.is_symlink()]
    current = max(directories, key=lambda p: p.stat().st_mtime, default=None)
    summary_path = current / 'summary.json' if current else None
    summary = small_json(summary_path) if summary_path else None
    log_name = ('warmup_' + name[-1].lower() if name.startswith('warmup_')
                else 'branch_' + name.lower()) + '_20261027_r1.log'
    log = ROOT / 'logs' / log_name
    evaluation = next((r for r in tail_rows(log) if r.get('event') == 'evaluation_progress'), None)
    complete = run / 'complete.json'
    return dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), run=name,
        training=None if latest is None else {k: latest.get(k) for k in
            ('epoch', 'phase', 'phase_index', 'update', 'r_updates', 't_updates', 'attempt')},
        last_checkpoint=metadata(run / 'last.pt'),
        complete=dict(exists=complete.is_file(), file=metadata(complete), content=small_json(complete)),
        validation=None if current is None else dict(directory=current.name,
            summary_file=metadata(summary_path), summary_complete=isinstance(summary, dict)
            and summary.get('split') == 'validation' and summary.get('full_split') is True,
            examples=summary.get('examples') if isinstance(summary, dict) else None),
        evaluation_progress=evaluation, log_file=metadata(log),
        scope='File observations only; completion does not establish platform success or resource release.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', choices=RUNS)
    print(json.dumps(snapshot(parser.parse_args().run), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
