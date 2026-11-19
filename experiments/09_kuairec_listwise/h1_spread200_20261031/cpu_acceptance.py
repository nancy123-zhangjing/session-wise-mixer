"""Audit the actual sealed execution tree, source pairing, and real parent load."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CODE = ROOT / 'outputs/stage09_spread200_20261031/code'
sys.path.insert(0, str(CODE))
os.environ['DIFFGRM_ENGINE_ROOT'] = str(CODE / 'external/DiffGRM')

import numpy as np
import torch
torch.set_num_threads(4)
from scripts.listwise_diffgrm.spread200 import build_spread, paired_indices, index_sha
from scripts.listwise_diffgrm.run import phase_indices
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.recovery35_trajectory import PHASES
from scripts.listwise_diffgrm.training import sha256


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    start = time.monotonic()
    package = json.loads((HERE / 'EXECUTION_PACKAGE.json').read_text(encoding='utf-8'))
    manifest_path = CODE / 'SPREAD200_EXECUTION_MANIFEST.json'
    if sha256(manifest_path) != package['manifest_sha256']:
        raise ValueError('Execution package manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        if sha256(CODE / name) != entry['sha256']:
            raise ValueError('Execution payload changed: ' + name)
    parent = json.loads((HERE / 'PARENT_BINDING.json').read_text(encoding='utf-8'))
    c, assets, data = build_spread(parent['local'], parent['sha256'], 'cpu',
        str(ROOT / 'outputs/kuairec_listwise_v1/run1'),
        str(ROOT / 'outputs/stage09_restore/goal_20261027_r1/assets/pse_v1'), exact_rng=False)
    cfg = c.trainer.config
    old, new, records = [], [], []
    for block in range(5):
        phase = 5 + 4*block
        indices = phase_indices(cfg, len(data), phase, dict(kind='T', updates=40))
        if (len(indices), index_sha(indices)) != (PHASES[phase][4], PHASES[phase][5]):
            raise ValueError('Historical T order receipt mismatch')
        old.append(indices)
    for block in range(10):
        phase = 5 + 4*block
        indices = paired_indices(cfg, len(data), phase, c.trainer.plan[phase])
        new.append(indices)
        records.append(dict(phase=phase, lists=len(indices), sha256=index_sha(indices),
                            original_phase=5+4*(block//2), half=block%2))
    old, new = np.concatenate(old), np.concatenate(new)
    if not np.array_equal(old, new) or len(new) != 204800:
        raise ValueError('Exact first-200T sample stream mismatch')
    r_records = []
    for phase, row in enumerate(c.trainer.plan):
        if phase < 5 or row['kind'] != 'R':
            continue
        indices = paired_indices(cfg, len(data), phase, row)
        digest = index_sha(indices)
        if phase in PHASES and digest != PHASES[phase][5]:
            raise ValueError('Historical R order receipt mismatch')
        r_records.append(dict(phase=phase, epoch=row['epoch'], lists=len(indices), sha256=digest,
                              compared_to_historical_receipt=phase in PHASES))
    state = c.checkpoint()
    preserved = {k: fingerprint(state['inner'][k]) == v for k,v in c.parent_hashes.items()}
    if not all(preserved.values()):
        raise ValueError('Full R5 parent state changed')
    acceptance = dict(status='passed', inspection_only=True, seconds=time.monotonic()-start,
        manifest_sha256=package['manifest_sha256'], parent_sha256=parent['sha256'],
        parent_state_preserved=preserved, train_examples=len(data), catalog_items=assets.n_items,
        r_updates=c.trainer.r_updates, t_updates=c.trainer.t_updates,
        original_t_stream_sha256=index_sha(old), spread_t_stream_sha256=index_sha(new),
        paired_lists=len(new), t_blocks=records, r_phases=r_records,
        noise=c.trainer.alignment.diger_schedule_state(),
        limitation='CPU inspection is not CUDA trajectory acceptance; no formal dataset optimization performed')
    write('CPU_INPUT_ACCEPTANCE.json', acceptance)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    result = subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests',
                             '-p','test_listwise_spread200*.py','-v'], cwd=CODE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    (HERE / 'CPU_TESTS.log').write_text(result.stdout, encoding='utf-8')
    write('CPU_TESTS.json', dict(status='passed' if result.returncode == 0 else 'failed',
        exit_code=result.returncode, log_sha256=sha256(HERE / 'CPU_TESTS.log'),
        manifest_sha256=package['manifest_sha256'], test_source_sha256=sha256(CODE / 'tests/test_listwise_spread200.py')))
    if result.returncode:
        raise RuntimeError('CPU tests failed; inspect CPU_TESTS.log')
    print(json.dumps(dict(status='passed', parent=True, exact_paired_T_lists=len(new),
                         R_phases=len(r_records), tests=9, seconds=time.monotonic()-start)))


if __name__ == '__main__':
    main()
