"""Candidate orchestration only: frozen scientific functions, immutable evaluation tasks.

Not a launcher. A separately verified runtime gate must authorize each invocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def task_kind(row):
    if row['kind'] == 'T':
        return 'diagnostic'
    if row['epoch'] in (8, 11, 14, 17, 20, 23, 26, 29, 32, 35):
        return 'validation'
    if (row['epoch'] - 6) % 3 == 0:
        return 'diagnostic'
    return None


class TaskQueue:
    """At most two read-only subprocesses; a done file alone is not acceptance."""
    def __init__(self, tasks, command, runtime_sha, stopped):
        self.tasks, self.command = Path(tasks), command
        self.runtime_sha, self.stopped = runtime_sha, stopped
        self.active = {}

    def accepted(self, path):
        done = path.with_suffix('.done.json')
        if not done.exists():
            return False
        result, task = read(done), read(path)
        if (result['status'] != 'completed_readonly_task' or result['request_sha256'] != sha(path)
                or result['runtime_sha256'] != self.runtime_sha
                or result['checkpoint_sha256'] != task['checkpoint_sha256']
                or result['state_before'] != result['state_after']):
            raise ValueError('Invalid readonly completion receipt: ' + path.name)
        return True

    def requests(self):
        return sorted(p for p in self.tasks.glob('phase*.json') if not p.name.endswith('.done.json'))

    def pump(self, wait=False):
        while True:
            for path, (proc, handle) in list(self.active.items()):
                if proc.poll() is not None:
                    handle.close()
                    del self.active[path]
                    if proc.returncode != 0 or not self.accepted(path):
                        raise RuntimeError('Readonly task failed: ' + path.name)
            for path in self.requests():
                if path in self.active or self.accepted(path):
                    continue
                if len(self.active) >= 2:
                    break
                if self.stopped():
                    raise SystemExit(124)
                handle = path.with_suffix('.log').open('xb')
                self.active[path] = (subprocess.Popen(self.command(path), stdout=handle, stderr=subprocess.STDOUT), handle)
            pending = [path for path in self.requests() if not self.accepted(path)]
            if not wait or (not pending and not self.active):
                return
            if self.stopped():
                raise SystemExit(124)
            time.sleep(.25)

    def close(self):
        for proc, _handle in self.active.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, handle in self.active.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'gate', 'gate-sha', 'self-sha'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--worker')
    a = p.parse_args()
    root = Path(a.root).resolve()
    if root != Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'):
        raise ValueError('Wrong owned root')
    if sha(__file__) != a.self_sha or sha(a.gate) != a.gate_sha:
        raise ValueError('Runtime authorization bytes changed')
    gate = read(a.gate)
    if gate['status'] != 'accepted_async_resume' or gate['runtime_sha256'] != a.self_sha:
        raise ValueError('Async runtime is not accepted')
    code = root / 'code'
    manifest = code / 'SPREAD200_EXECUTION_MANIFEST.json'
    if sha(manifest) != gate['scientific_manifest_sha256']:
        raise ValueError('Scientific manifest changed')
    for name, value in read(manifest)['files'].items():
        path = code / name
        path.resolve().relative_to(code)
        if path.is_symlink() or sha(path) != value['sha256']:
            raise ValueError('Frozen scientific source changed: ' + name)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
    import torch
    from scripts.listwise_diffgrm.spread200 import build_spread, paired_indices, index_sha, BEST_EPOCHS
    from scripts.listwise_diffgrm.spread200_run import append, last, catalog, diagnostic
    from scripts.listwise_diffgrm.recovery35_run import evaluation
    from scripts.listwise_diffgrm.recovery35_artifacts import save_boundary, write_once
    from scripts.listwise_diffgrm.recovery_evaluation import load_subset
    from scripts.listwise_diffgrm.data import KuaiRecListDataset
    from scripts.listwise_diffgrm.run import microbatches, run_lock
    from scripts.listwise_diffgrm.recovery import fingerprint
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('Only one H100 authorized')
    inputs = read(root / 'INPUTS.json')
    output = root / 'runs/H1-Spread200-R35'
    runtime = root / 'jobs/formal_async_r2'
    tasks = runtime / 'tasks'
    tasks.mkdir(parents=True, exist_ok=True)
    c, assets, data = build_spread(inputs['parent'], gate['parent_sha256'], 'cuda:0', inputs['dataset'], inputs['assets'])
    subset = load_subset(inputs['subset'], c.trainer.config['dataset'])
    validation = KuaiRecListDataset(c.trainer.config['dataset'], 'validation', assets=assets)
    if a.worker:
        request = Path(a.worker).resolve()
        request.relative_to(tasks)
        task = read(request)
        checkpoint = Path(task['checkpoint'])
        checkpoint.resolve().relative_to(output)
        if sha(checkpoint) != task['checkpoint_sha256']:
            raise ValueError('Task checkpoint changed')
        c.restore(torch.load(checkpoint, map_location='cpu', weights_only=False))
        before = fingerprint(c.checkpoint())
        source = None
        if task['kind'] == 'validation':
            record = evaluation(c, assets, output, 'validation', task['epoch'], checkpoint=checkpoint)
            source = record['predictions']
            write_once(output / 'validation_records' / f'R{task["epoch"]:02d}.json', record)
        diagnostic(c, assets, validation, subset, output, task['label'], source=source)
        if fingerprint(c.checkpoint()) != before:
            raise ValueError('Read-only worker changed training state')
        write_once(request.with_suffix('.done.json'), dict(status='completed_readonly_task', request_sha256=sha(request),
            checkpoint_sha256=sha(checkpoint), state_before=before, state_after=before, runtime_sha256=a.self_sha))
        return
    resume = Path(gate['checkpoint'])
    if sha(resume) != gate['checkpoint_sha256'] or sha(output / 'training.jsonl') != gate['journal_sha256']:
        raise ValueError('Preserved continuation point changed')
    stop = {'requested': False}
    def request_stop(_sig, _frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def worker_command(path):
        return [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--root', str(root),
                '--gate', a.gate, '--gate-sha', a.gate_sha, '--self-sha', a.self_sha, '--worker', str(path)]
    queue = TaskQueue(tasks, worker_command, a.self_sha, lambda: stop['requested'])
    try:
        with run_lock(output, resume=True):
            c.restore(torch.load(resume, map_location='cpu', weights_only=False))
            while not c.done:
                queue.pump()
                phase, row = c.progress['phase'], dict(c.row)
                indices = paired_indices(c.trainer.config, len(data), phase, row)
                write_once(output / 'sample_orders' / f'phase{phase:02d}.json', dict(
                    phase=phase, kind=row['kind'], epoch=row['epoch'], lists=len(indices), indices_sha256_int64_le=index_sha(indices),
                    original_t_phase=(5 + 4 * ((phase - 5) // 8)) if row['kind'] == 'T' else None,
                    original_t_half=((phase - 5) // 4) % 2 if row['kind'] == 'T' else None))
                while c.progress['update'] < row['updates']:
                    if stop['requested']:
                        last(c, output)
                        raise SystemExit(124)
                    begin = c.progress['update'] * c.trainer.config['training']['effective_batch_size']
                    selected = indices[begin:begin + c.trainer.config['training']['effective_batch_size']]
                    started = time.perf_counter()
                    values = c.step(microbatches(data, selected, c.trainer.config['training']['microbatch_size']), row['kind'])
                    values.update(epoch=row['epoch'], seconds=time.perf_counter()-started, sample_indices_sha256=index_sha(selected))
                    append(output / 'training.jsonl', values)
                    if c.progress['update'] % (10 if row['kind'] == 'T' else 100) == 0:
                        last(c, output)
                        print(json.dumps(dict(event='spread_progress', **values)), flush=True)
                        queue.pump()
                if not c.progress['boundary_prepared']:
                    c.prepare_boundary()
                    last(c, output)
                label = f'phase{phase:02d}_{row["kind"]}_R{row["epoch"]:02d}_T{c.trainer.t_updates:03d}'
                if row['kind'] == 'T':
                    catalog(c, output, label)
                c.complete_phase()
                snapshot = save_boundary(c, output, label + '.pt')
                kind = task_kind(row)
                if kind:
                    write_once(tasks / f'phase{phase:02d}.json', dict(kind=kind, epoch=row['epoch'], label=label,
                        checkpoint=str(output / (label + '.pt')), checkpoint_sha256=snapshot['sha256']))
                last(c, output)
                print(json.dumps(dict(event='spread_boundary', epoch=row['epoch'], kind=row['kind'],
                    r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates)), flush=True)
            queue.pump(wait=True)
            candidates = [read(output / 'validation_records' / f'R{e:02d}.json') for e in BEST_EPOCHS]
            selected = sorted(candidates, key=lambda r: (-r['ndcg5'], r['epoch']))[0]
            write_once(output / 'selection.json', dict(criterion='validation/list/ndcg@5; tie earlier', candidates=candidates, selected=selected))
            write_once(output / 'complete.json', dict(status='training_and_validation_complete',
                r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates,
                last_sha256=sha(output / 'last.pt'), selection_sha256=sha(output / 'selection.json')))
    finally:
        queue.close()


if __name__ == '__main__':
    main()
