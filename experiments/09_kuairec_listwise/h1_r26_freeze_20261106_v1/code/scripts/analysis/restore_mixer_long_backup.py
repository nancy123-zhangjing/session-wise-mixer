"""Restore real formal states into fresh CPU models using only verified backup assets.

CPU restoration checks CUDA RNG bytes for preservation but does not execute CUDA.
The separate preflight H100 replay receipts document same-backend next-phase replay.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAMPAIGN = 'goal_20261020_long_mixer_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    args = parser.parse_args()
    backup, scratch = args.backup_root.resolve(), args.scratch_root.resolve()
    audit = module('_restore_independent_audit', Path(__file__).with_name('audit_mixer_long_campaign.py'))
    need, read, sha = audit.need, audit.read, audit.sha
    need(backup.name == scratch.name == CAMPAIGN and backup != scratch and not scratch.exists(), 'Fresh campaign scratch required')
    verified = read(backup / 'BYTE_VERIFICATION.json')
    need(verified['status'] == 'FULL_BACKUP_BYTE_VERIFICATION_PASS_NOT_RESTORE'
         and verified['all_member_bytes_verified'] and verified['all_14_locked_inputs_included'], 'Full byte verification required')
    manifest = read(backup / 'MANIFEST.json')
    need(sha(backup / 'MANIFEST.json') == verified['manifest_sha256'], 'Backup manifest identity differs')
    # Recheck every byte before constructing a model; extract/restore checks are distinct.
    for row in manifest['files']:
        path = audit.within(backup, row['member'])
        need(not path.is_symlink() and path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], 'Verified backup changed')
    scratch.mkdir(parents=True)
    (scratch / 'delivery').mkdir()
    shutil.copy2(backup / 'delivery/INPUTS.json', scratch / 'delivery/INPUTS.json')
    input_manifest = read(backup / 'delivery/INPUTS.json')
    for row in input_manifest['records']:
        source = backup / 'inputs' / row['path']; target = scratch / 'inputs' / row['path']
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        need(sha(target) == row['sha256'], 'Restore scratch input copy differs')
    runner = module('_restore_frozen_runner', backup / 'code/scripts/train/run_diger_mixer_long.py')
    config = backup / 'code/configs/05_crosspse_joint/long/mixer_long_seed2024.json'
    context = runner.context_for(SimpleNamespace(config=config,goal_root=scratch,engine_root=backup / 'engine',
        allow_cpu=True,mode='smoke',arm='backup_restore',expected_code_commit=None,expected_engine_commit=None,deadline_unix=None))
    identity = read(backup / 'formal/identity.json')
    cpu_identity = copy.deepcopy(context['identity']); cpu_identity['execution_backend'] = 'cuda'
    need(cpu_identity == identity, 'CPU reconstruction differs beyond declared execution backend')
    import numpy as np
    import torch
    original_initial = torch.load(backup / 'formal/common/initial.pt',map_location='cpu',weights_only=False)
    need(runner.digest(context['common']) == runner.digest(original_initial['model']), 'Rebuilt complete shared start differs')
    context['identity'], context['mode'] = identity, 'formal'
    model = context['pipeline'].model
    protected = {str(backup / 'formal' / arm / name):sha(backup / 'formal' / arm / name)
                 for arm in audit.ARMS for name in ('last.pt','best.pt')}
    records = []
    for arm in audit.ARMS:
        for filename in ('last.pt','best.pt'):
            path = backup / 'formal' / arm / filename
            state = torch.load(path,map_location='cpu',weights_only=False)
            runner.configure(model,arm)
            optimizers, scheduler = runner.optimizers(model,arm,context['protocol'])
            progress = runner.restore(context,arm,state,optimizers,scheduler)
            current_rng = context['runtime'].phase_rng_state(context['device'])
            need(all(runner.digest(current_rng[key]) == runner.digest(state['rng'][key]) for key in ('python','numpy','torch_cpu')),
                 'Restored CPU/Python/NumPy RNG differs')
            cuda_saved = state['rng']['torch_cuda']
            need(torch.is_tensor(cuda_saved) and cuda_saved.dtype == torch.uint8 and cuda_saved.numel() > 0,
                 'CUDA generator state bytes were not retained')
            need(np.array_equal(model.joint_alignment.hard_codes[1:].numpy(),state['model']['joint_alignment.hard_codes'][1:].numpy()),
                 'Restored official SID differs')
            anchor = torch.from_numpy(np.load(scratch / 'inputs/canonical/E0.npy',allow_pickle=False))
            need(torch.equal(model.joint_alignment.mixer.initial_codebooks.cpu(),anchor), 'Nonpersistent anchor was not reconstructed')
            records.append(dict(arm=arm,checkpoint=filename,sha256=sha(path),model_tensor_sha256=runner.digest(model.state_dict()),
                decoder_epochs=progress['decoder_epochs'],decoder_updates=progress['decoder_updates'],tokenizer_updates=progress['tokenizer_updates'],
                strict_model_optimizer_scheduler_restore=True,python_numpy_cpu_rng_restored=True,
                cuda_rng_bytes_preserved_sha256=hashlib.sha256(cuda_saved.numpy().tobytes()).hexdigest(),
                official_SID_restored=True,nonpersistent_E0_anchor_restored=True))
            del state, optimizers, scheduler
    need(all(sha(Path(path)) == digest for path,digest in protected.items()), 'Restore changed a source checkpoint')
    receipt = dict(status='FORMAL_FULL_BACKUP_CPU_RESTORE_PASS',campaign_id=CAMPAIGN,utc=datetime.now(timezone.utc).isoformat(),
        byte_verification_sha256=sha(backup / 'BYTE_VERIFICATION.json'),manifest_sha256=verified['manifest_sha256'],
        restorer_sha256=sha(Path(__file__)),model_source='verified_backup/code',engine_source='verified_backup/engine',
        all_14_inputs_copied_from_verified_backup=True,checkpoints=records,original_checkpoints_unchanged=True,
        execution_backend='cpu',cuda_execution_verified_here=False,goal_complete=False)
    audit.write(scratch / 'RESTORE_PASS.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
