"""Revalidate the future item-local formal launch on CPU after its real budget gate.

Run only on the existing CPU bridge with the frozen campaign's conda Python.
Reads completed control and paired smoke; never constructs a model or starts
training, evaluation, a platform request, or a GPU process. Missing/not-passed
parent budget evidence is a failure, not permission to manufacture a PASS.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2')
ROOT_COMMIT = '842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', ACCELERATE_USE_CPU='true')
    sys.path.insert(0, str(ROOT / 'code/scripts/train'))
    import item_local_joint_gpu_job as wrapper

    binding = wrapper.source_binding(ROOT, ROOT_COMMIT, ENGINE_COMMIT)
    rows = wrapper.study_rows(ROOT)
    smoke = wrapper.smoke_gate(ROOT / 'delivery/smoke_attempt1/receipt.json', rows, binding)
    # This separate, future parent judgment must already exist and truly pass.
    budget = wrapper.full_job_gate(ROOT / 'delivery/FORMAL_BUDGET_ITEM_LOCAL_LIVE_V2.json',
                                  'item_local', smoke, 10350)
    control = next(row for row in rows if row['variant'] == 'control')
    _, _, control_hashes = wrapper.completion(control, 'formal', binding)
    row = next(row for row in rows if row['variant'] == 'item_local')
    wrapper.need(wrapper.child_guard(row, 'formal', False, binding) is False,
                 'Item-local is not a fresh unstarted formal run')
    work = Path(row['output'])
    config = wrapper.load(work / 'resolved_config.json')
    expected = {'epochs': 45, 'train_batch_size': 64, 'eval_batch_size': 8,
                'gradient_accumulation_steps': 16, 'joint_crosspse_enabled': True,
                'joint_frozen_tokenizer_control': False, 'crosspse_mixer_enabled': False,
                'crosspse_train_base_codebooks': True, 'joint_gradient_mode': 'diger_st'}
    wrapper.need(all(config[key] == value for key, value in expected.items())
                 and config['run_id'] == row['run_id']
                 and config['item_local_attention'] == {'enabled': True, 'num_heads': 4, 'dropout': 0.0},
                 'Wrong formal configuration/local-attention variant')
    wrapper.need(config.get('max_train_batches_per_epoch') is None and config.get('max_eval_batches') is None,
                 'Formal run must not truncate training or evaluation')
    inputs = wrapper.load(work / 'input_identities.json')['artifacts']
    wrapper.need(len(inputs) == 13 and len({item['path'] for item in inputs}) == 13
                 and inputs == smoke['paired_smoke_checks']['variants']['item_local']['input_artifacts'],
                 'Formal thirteen-input identities differ from accepted paired smoke')
    for item in inputs:
        path = work / 'workspace' / item['path']
        wrapper.need(path.stat().st_size == item['size_bytes'] and wrapper.sha(path) == item['sha256'],
                     'Formal workspace input bytes changed: ' + item['path'])
    context_path = ROOT / 'delivery/ENVIRONMENT_IMPORT_CONTEXT_V2.json'
    context = wrapper.load(context_path)
    wrapper.need(context['status'] == 'EXACT_PACKAGES_MATCH_AFTER_PRODUCTION_IMPORT',
                 'Missing proven production package-import context')
    # Match the successful control v2 collector, including setuptools vendor metadata.
    sys.path.insert(0, str(ROOT / 'engine'))
    sys.path.insert(0, str(ROOT / 'code/scripts/repro'))
    importlib.import_module('genrec.pipeline')
    importlib.import_module('genrec.models.DIFF_GRM.trainer')
    current = sorted(f"{entry.metadata['Name']}=={entry.version}" for entry in importlib.metadata.distributions())
    for variant in wrapper.VARIANTS:
        saved = wrapper.load(ROOT / 'runs' / wrapper.run_id(variant) / 'smoke/environment.json')
        wrapper.need(current == saved['packages'], 'Packages differ under the production import context')
        wrapper.need(saved['python'] == sys.version and saved['executable'] == sys.executable,
                     'Python version/executable differs from paired H100 smoke')
    record = {'status': 'FORMAL_ITEM_LOCAL_CPU_REVALIDATION_PASS', 'source_binding': binding,
              'smoke_receipt_sha256': smoke['sha256'], 'budget_sha256': budget['sha256'],
              'completed_control_artifacts': control_hashes,
              'full_formal_configuration_verified': True, 'item_local_attention_enabled': True,
              'all_13_workspace_inputs_rehashed': True,
              'python_packages_exact_match_both_H100_smokes_after_production_import': True,
              'package_import_context_evidence_sha256': wrapper.sha(context_path),
              'explanation': 'Accelerate import exposes setuptools._vendor metadata; compare the exact sorted list after production imports.',
              'no_model_constructed': True, 'no_training': True, 'no_evaluation': True,
              'formal_started': False, 'goal_complete': False,
              'scope': 'CPU launch revalidation only; parent separately verifies completed control Job release, live policy and cumulative allocation.'}
    with (ROOT / 'delivery/FORMAL_ITEM_LOCAL_CPU_REVALIDATION_V2.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
