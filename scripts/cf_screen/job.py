"""One screening job: one worker per GPU, all four slots in parallel."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .variants import JOBS, RERUNS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=sorted({**JOBS, **RERUNS}))
    for name in ('root', 'wave', 'runtime'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    root = a.root/('smoke' if a.smoke else 'runs')
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started, procs = time.time(), []
    for gpu, name in enumerate({**JOBS, **RERUNS}[a.job]):
        command = [sys.executable, '-u', '-m', 'scripts.cf_screen.worker', name, '--root', a.root,
                   '--wave', a.wave, '--runtime', a.runtime, *(['--smoke'] if a.smoke else [])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = (logs/f'{a.job}_{name}.log').open('a', encoding='utf-8')
        procs.append((name, gpu, subprocess.Popen(list(map(str, command)), env=env,
                                                  stdout=log, stderr=subprocess.STDOUT), log))
    status = []
    for name, gpu, proc, log in procs:
        code = proc.wait()
        log.close()
        status.append(dict(name=name, gpu=gpu, returncode=code))
    (root/'jobs').mkdir(exist_ok=True)
    (root/'jobs'/f'{a.job}.json').write_text(json.dumps(dict(job=a.job, smoke=a.smoke, seconds=time.time()-started,
                                                             slots=status), indent=2), encoding='utf-8')
    if any(s['returncode'] for s in status):
        sys.exit(1)


if __name__ == '__main__':
    main()
