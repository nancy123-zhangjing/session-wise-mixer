"""Run one protocol v2 job: one worker process per GPU, all slots in parallel.

Each GPU job holds whole seed pairs (no-Mixer and Mixer arm of the same seed on
the same node). Quota rows are 1/2/4/8 GPUs, so three seeds split into a
4-GPU job (seeds 2024, 2025) and a 2-GPU job (seed 2026) per group.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PAIRS = {'h': ('H0', 'H1'), 'tiger': ('T0A', 'T1A'), 'cf': ('CF-M0', 'CF-M1')}
JOBS = {}
for family, arms in PAIRS.items():
    JOBS[f'{family}-a'] = [(family, arm, seed) for seed in (2024, 2025) for arm in arms]
    JOBS[f'{family}-b'] = [(family, arm, 2026) for arm in arms]
JOBS['sasrec-a'] = [('sasrec', 'SASREC', 2024), ('sasrec', 'SASREC', 2025)]
JOBS['sasrec-b'] = [('sasrec', 'SASREC', 2026)]
GPU_ROWS = (1, 2, 4, 8)
assert all(len(slots) in GPU_ROWS for slots in JOBS.values())
# Re-runs of single slots whose original job lost them (2026-11-18: H1 seed 2026
# died in the config-lock race; SASRec seed 2025 was recycled by the platform's
# low-GPU-utilisation rule). Workers resume or skip finished stages, so these
# only redo what is missing.
RERUNS = {'h-b-h1': [('h', 'H1', 2026)], 'sasrec-a-2025': [('sasrec', 'SASREC', 2025)]}
JOBS.update(RERUNS)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=sorted(JOBS))
    for name in ('root', 'receipt', 'dataset', 'assets', 'cf-runtime'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tiger-batch', type=int, default=256)
    p.add_argument('--h-rhythm', choices=['pass', 'old693'], default='pass')
    p.add_argument('--h-split', choices=['l5o', 'old'], default='l5o')
    p.add_argument('--smoke', action='store_true', help='pass --smoke to every worker (GPU smoke test, never results)')
    a = p.parse_args(argv)
    slots = JOBS[a.job]
    logs = a.root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started, procs = time.time(), []
    for gpu, (family, arm, seed) in enumerate(slots):
        command = [sys.executable, '-u', '-m', 'scripts.kuairec_data.l5o_v2_worker', family, arm, str(seed),
                   '--root', a.root, '--receipt', a.receipt, '--dataset', a.dataset, '--assets', a.assets,
                   '--cf-runtime', a.cf_runtime, '--tiger-batch', str(a.tiger_batch)]
        if a.h_rhythm != 'pass':
            command += ['--h-rhythm', a.h_rhythm]
        if a.h_split != 'l5o':
            command += ['--h-split', a.h_split]
        if a.smoke:
            command.append('--smoke')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = (logs/f'{a.job}_{family}_{arm}_{seed}.log').open('a', encoding='utf-8')
        procs.append(((family, arm, seed, gpu), subprocess.Popen(list(map(str, command)), env=env,
                                                                  stdout=log, stderr=subprocess.STDOUT), log))
    status = []
    for slot, proc, log in procs:
        code = proc.wait()
        log.close()
        status.append(dict(family=slot[0], arm=slot[1], seed=slot[2], gpu=slot[3], returncode=code))
    record = dict(job=a.job, seconds=time.time() - started, slots=status)
    (a.root/'jobs').mkdir(exist_ok=True)
    (a.root/'jobs'/f'{a.job}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    if any(s['returncode'] for s in status):
        sys.exit(1)


if __name__ == '__main__':
    main()
