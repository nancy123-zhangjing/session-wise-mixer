"""Inspect real A/B/C parents in independent CPU processes; never advance them."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def worker(code, branch, destination):
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(code))
    import torch
    from scripts.listwise_diffgrm import recovery35
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.training import sha256

    assert Path(recovery35.__file__).resolve().is_relative_to(code)
    assert torch.version.cuda is None, 'Use the existing CPU-only restore environment'
    torch.set_num_threads(4)
    refs = json.loads((code / 'FREEZE35_PARENTS.json').read_text(encoding='utf-8'))
    parent = refs['parents'][branch]
    start = time.perf_counter()
    controller, assets, data = recovery35.build_recovery35(
        parent['local'], branch, 'cpu', dataset=refs['dataset']['local'], assets=refs['assets']['local'],
        ancestor_path=refs['ancestors']['A']['local'] if branch == 'A' else None, inspection=True)
    checkpoint = controller.checkpoint()
    inner = checkpoint['inner']
    source = torch.load(parent['local'], map_location='cpu', weights_only=False)
    if branch == 'A':
        source = source['inner']
    keys = ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')
    hashes = {key: fingerprint(inner[key]) for key in keys}
    assert hashes == {key: fingerprint(source[key]) for key in keys}
    rng = dict(source['rng'], cuda=None)
    assert fingerprint(inner['rng']) == fingerprint(rng)
    assert (inner['r_updates'], inner['t_updates']) == (parent['r_updates'], parent['t_updates'])
    assert inner['progress'] == dict(phase=parent['phase'], update=0, boundary_prepared=False)
    assert all(p.device.type == 'cpu' and p.grad is None
               for module in (controller.trainer.model, controller.trainer.alignment)
               for p in module.parameters())
    assert checkpoint['inspection_only'] is True
    assert len(data) == 709443
    forbidden_resume_rejected = False
    try:
        controller.restore(checkpoint, exact_rng=False)
    except ValueError as error:
        forbidden_resume_rejected = 'inspection' in str(error).lower()
    assert forbidden_resume_rejected, 'CPU inspection must not become a formal resume'
    # Rejected resume and checkpoint serialization must not change loaded state.
    assert fingerprint(controller.checkpoint()) == fingerprint(checkpoint)
    result = dict(status='passed', branch=branch, parent_sha256=sha256(parent['local']),
        runtime_module=str(Path(recovery35.__file__).resolve()), runtime_module_sha256=sha256(recovery35.__file__),
        mixer_enabled=bool(controller.trainer.alignment.mixer.mixer_enabled),
        r_updates=inner['r_updates'], t_updates=inner['t_updates'], progress=inner['progress'],
        r_learning_rate=controller.trainer.r_optimizer.param_groups[0]['lr'],
        t_learning_rates={group['group_name']: group['lr'] for group in controller.trainer.t_optimizer.param_groups},
        parent_scientific_state_fingerprints=hashes, inner_checkpoint_fingerprint=fingerprint(inner),
        inspection_checkpoint_fingerprint=fingerprint(checkpoint),
        python_numpy_torch_cpu_rng_restored=True, cuda_rng_restored=False,
        inspection_resume_rejected=forbidden_resume_rejected, updates_performed=0,
        train_examples=len(data), elapsed_seconds=time.perf_counter() - start)
    write_new(destination, result)
    print(json.dumps({key: result[key] for key in ('status', 'branch', 'r_updates', 't_updates', 'mixer_enabled', 'elapsed_seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', choices=('A', 'B', 'C'))
    args = parser.parse_args()
    code, output = Path(args.code_root).resolve(), Path(args.output).resolve()
    if args.worker:
        worker(code, args.worker, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', scope='CPU inspection only; two independent real-parent loads per branch',
                  runtime_code=str(code), updates_performed=0, cuda_rng_restored=False, branches={})
    for branch in ('A', 'B', 'C'):
        results = []
        for iteration in (1, 2):
            destination = output / f'{branch}_load{iteration}.json'
            process = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                '--code-root', str(code), '--output', str(destination), '--worker', branch],
                cwd=code, text=True, encoding='utf-8', capture_output=True)
            if process.returncode:
                raise RuntimeError(f'{branch} load {iteration} failed: {process.stderr[-6000:]}')
            print(process.stdout.strip(), flush=True)
            results.append(json.loads(destination.read_text(encoding='utf-8')))
        comparable = lambda result: {key: value for key, value in result.items() if key != 'elapsed_seconds'}
        assert comparable(results[0]) == comparable(results[1]), f'{branch}: independent reconstruction differs'
        report['branches'][branch] = dict(results[0], independent_process_rebuild_identical=True,
                                          two_load_elapsed_seconds=[row['elapsed_seconds'] for row in results])
    report['status'] = 'passed'
    write_new(output / 'CPU_RESTORE_AUDIT.json', report)
    print(json.dumps(dict(status=report['status'], report=str(output / 'CPU_RESTORE_AUDIT.json'))))


if __name__ == '__main__':
    main()
