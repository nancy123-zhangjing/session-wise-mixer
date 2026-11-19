"""CPU fixture tests for local backup readback; no real formal evidence created."""
from __future__ import annotations
import argparse
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[2]

def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve an existing selftest receipt')
    verifier = module(Path(__file__).with_name('verify_joint_backup_checkpoints.py'),'_backup_readback')
    packer = module(ROOT/'scripts/analysis/pack_item_local_joint_delivery.py','_byte_fixture_packer')
    fixture = module(ROOT/'tests/test_item_local_joint_training_state.py','_actual_joint_state_fixture')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    checks = []
    with tempfile.TemporaryDirectory(prefix='joint_backup_cpu_fixture_') as temporary, redirect_stdout(io.StringIO()):
        root = Path(temporary); artifacts = root/'offline'; fixture.write_artifacts(artifacts)
        mixer = fixture.Mixer(torch.from_numpy(np.load(artifacts/'base_pse_codebooks.npy')),n_heads=2,
                              train_base_codebooks=True,mixer_enabled=False)
        torch.save({'model':mixer.state_dict()},artifacts/'best.pt')
        original_config = fixture.base.trainer_config
        def refined_config(*values,**kwargs):
            config = original_config(*values,**kwargs)
            config.update(crosspse_mixer_enabled=False,crosspse_train_base_codebooks=True); return config
        boundaries = {}; save = fixture.runtime.save_boundary
        def observed(path,**kwargs):
            payload = save(path,**kwargs); boundaries[kwargs['progress']['next_epoch']] = copy.deepcopy(payload); return payload
        with mock.patch.object(fixture.base,'trainer_config',side_effect=refined_config), mock.patch.object(fixture.runtime,'save_boundary',side_effect=observed):
            trainer, loader = fixture.make_trainer(root/'actual_CPU_run',artifacts)
            trainer.fit(loader,[])
        terminal = boundaries[10]; token_boundary = boundaries[6]
        # Only a serialized byte-slot substitute for the one-GPU RNG schema;
        # these values are not CUDA RNGs and no CUDA restore is performed.
        for payload in (terminal,token_boundary): payload['rng']['torch_cuda_all'] = [torch.get_rng_state().clone()]
        source = root/'fixture_files'; source.mkdir()
        for variant in ('control','item_local'):
            run = source/'runs'/f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'
            for sub in ('training_state','smoke/training_state','smoke/recovery_probe'): (run/sub).mkdir(parents=True)
            for relative in ('training_state/last.pt','smoke/training_state/last.pt'): torch.save(terminal,run/relative)
            for relative in ('smoke/smoke_tokenizer_boundary.pt','smoke/recovery_probe/source_last.pt'): torch.save(token_boundary,run/relative)
            for relative in ('training_state/best_model.bin','smoke/training_state/best_model.bin'):
                (run/relative).write_bytes(terminal['best_checkpoint_bytes'])
            (run/'smoke/recovery_probe/best_model.bin').write_bytes(token_boundary['best_checkpoint_bytes'])
        mutation_target = source/'runs/stage07_item_local_joint_control_seed2024_20261020_r1/training_state/last.pt'
        original = mutation_target.read_bytes(); case_number = 0
        def package_case(change=None,omit=None):
            nonlocal case_number
            case_number += 1; mutation_target.write_bytes(original)
            if change:
                payload = torch.load(mutation_target,map_location='cpu',weights_only=False); change(payload); torch.save(payload,mutation_target)
            for variant in ('control','item_local'):
                run = source/'runs'/f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'
                payload = torch.load(run/'training_state/last.pt',map_location='cpu',weights_only=False)
                best = torch.load(run/'training_state/best_model.bin',map_location='cpu',weights_only=False)
                digest = verifier.sha(run/'training_state/best_model.bin'); sid = verifier.tensor_digest(best['joint_alignment.hard_codes'][1:])
                result = dict(campaign_id=verifier.CAMPAIGN,variant=variant,run_id=run.name,status='formal_completed',
                    checkpoint_sha256=digest,selected_SID_sha256=sid,fixture_only=True,
                    fit=dict(training_state_sha256=verifier.sha(run/'training_state/last.pt'),completed_epochs=payload['progress']['next_epoch'],
                             best_epoch=payload['progress']['best_epoch'],joint_progress=payload['progress']['joint_progress']))
                reload = dict(checkpoint_sha256=digest,strict_all_tensor_match=True,mapping_synchronized_without_requantization=True,
                              best_epoch=payload['progress']['best_epoch'],matching_committed_SID_sha256=sid,fixture_only=True)
                for name,value in (('results.json',result),('checkpoint_reload.json',reload)):
                    (run/name).write_text(json.dumps(value),encoding='utf-8')
                (run/'COMPLETE.json').write_text(json.dumps(dict(campaign_id=verifier.CAMPAIGN,variant=variant,run_id=run.name,
                    mode='formal',status='formal_completed',checkpoint_sha256=digest,result_file='results.json',
                    result_sha256=verifier.sha(run/'results.json'),fixture_only=True)),encoding='utf-8')
            files = {path.relative_to(source).as_posix():path for path in source.rglob('*') if path.is_file() and path.name != omit}
            package = root/f'fixture_package_{case_number}'; package.mkdir()
            packer.archive_files(files,package,dict(fixture_only=True,source_identity={'code':{'commit':verifier.ROOT_COMMIT},
                'engine':{'commit':verifier.ENGINE_COMMIT}}),4096)
            restored = root/f'fixture_restored_{case_number}'; packer.verify(package,restored)
            return package,restored
        package,restored = package_case(); passed = verifier.verify(restored,package)
        assert passed['checkpoint_count'] == 14 and passed['status'] == 'LOCAL_CPU_CHECKPOINT_READBACK_PASS'
        checks.append('actual_CPU_joint_model_boundary_14_checkpoint_pack_verify_load_roundtrip')
        def rejected(label,change=None,omit=None,after=None,expected=''):
            package,restored = package_case(change,omit)
            if after: after(restored)
            try: verifier.verify(restored,package)
            except (ValueError,KeyError) as error:
                assert not expected or expected in str(error),(label,str(error)); checks.append(label); return
            raise AssertionError('Negative fixture accepted: '+label)
        def nonfinite(payload):
            entry = next(iter(payload['optimizer']['state'].values())); entry['exp_avg'].reshape(-1)[0] = float('nan')
        rejected('nonfinite_nested_Adam_tensor_rejected',nonfinite,expected='Nonfinite/non-CPU tensor')
        rejected('nonterminal_formal_last_rejected',lambda p:p['progress'].update(training_complete=False),expected='nonterminal')
        rejected('missing_RNG_key_rejected',lambda p:p['rng'].pop('torch_cpu'),expected='Incomplete RNG')
        rejected('embedded_best_corruption_rejected',lambda p:p.update(best_checkpoint_bytes=b'bad bytes'),expected='Embedded best')
        rejected('missing_necessary_checkpoint_rejected',omit='smoke_tokenizer_boundary.pt',expected='fourteen')
        def changed(root):
            path = root/'runs/stage07_item_local_joint_control_seed2024_20261020_r1/training_state/best_model.bin'
            with path.open('ab') as handle: handle.write(b'changed')
        rejected('post_extraction_checkpoint_byte_change_rejected',after=changed,expected='Backup bytes differ')
    result = dict(status='CPU_BACKUP_READBACK_SELFTEST_PASS',tool_sha256=verifier.sha(Path(verifier.__file__)),
        selftest_sha256=verifier.sha(__file__),checks=checks,real_formal_artifacts_validated=False,GPU_or_platform_used=False,
        fixture_scope='Temporary actual small CPU joint Trainer/model boundaries; bundle/result wrappers and CUDA RNG byte slot are synthetic schema fixtures. No real campaign PASS or CUDA restore is claimed.')
    with args.output.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(result,indent=2)); return 0

if __name__ == '__main__': raise SystemExit(main())
