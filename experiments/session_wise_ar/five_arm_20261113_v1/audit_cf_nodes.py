"""Read-only CPU checkpoint audit. Not a CUDA numerical-resume test.

Run on the shared-storage CPU after CF training, with --run TRAIN_DIRECTORY.
JSON goes to stdout; keep large checkpoints remote. No model forward/update.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(root, expected_epochs=None):
    done = json.loads((root/'TRAINING_COMPLETE.json').read_text())
    config = json.loads((root/'config.json').read_text())
    runtime = json.loads((root/'runtime.json').read_text())
    binding = json.loads((root/'binding.json').read_text())
    rows = []
    for summary in sorted((root/'validation').glob('R*/summary.json')):
        v = json.loads(summary.read_text())
        i = v['identity']
        path = root/f"checkpoints/R{i['R']:02d}_T{i['T']:03d}.pt"
        actual = sha(path)
        require(actual == i['checkpoint_sha256'], str(path)+' SHA mismatch')
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        require(s['format'] == 'rqcd_mixer_joint_v1', 'format')
        require(s['config'] == config and s['runtime'] == runtime, 'configuration')
        require(s['binding'] == binding == i['binding'], 'binding')
        require(s['r_epoch'] == i['R'] and s['t_updates'] == i['T'], 'node clock')
        require(v['count'] == 71657 and i['split'] == 'validation', 'validation scope')
        for key in ('tokenizer', 'model', 'diffusion', 'catalog', 'generators', 'rng'):
            require(bool(s[key]), 'missing '+key)
        require(len(s['rng']['cuda']) == 1, 'single GPU RNG missing')
        for phase in ('r', 't'):
            stream = s[phase+'_stream']
            steps = math.ceil(stream['size']/config['training']['effective_batch'])
            updates = stream['epoch']*steps + math.ceil(stream['cursor']/config['training']['effective_batch'])
            require(updates == s[phase+'_updates'], phase+' stream clock')
            require(torch.equal(stream['order'].sort().values, torch.arange(stream['size'])), 'permutation')
            torch.Generator().set_state(stream['rng'].cpu())
            opt = s['optimizer_'+phase]
            require(bool(opt['param_groups']), 'optimizer groups')
            require((not updates and not opt['state']) or (bool(opt['state']) and all(int(x['step']) == updates for x in opt['state'].values())), 'optimizer clock')
        rows.append({'R': i['R'], 'T': i['T'], 'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual})
        del s
    require([x['R'] for x in rows] == (config['evaluation']['epochs'] if expected_epochs is None else expected_epochs), 'missing validation nodes')
    require(rows[-1]['sha256'] == done['final_sha256'], 'final identity')
    return {'status': 'passed_cpu_structure_and_hash_only', 'gpu_restore': 'not tested by this audit; inspect automatic best/final test receipts', 'nodes': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
