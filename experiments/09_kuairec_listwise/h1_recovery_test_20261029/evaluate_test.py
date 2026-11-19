"""Evaluate authenticated H1 recovery endpoints; never train or rewrite a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

MANIFEST_SHA = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
PARENT_SHA = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
ENDPOINT_SHA = {
    'decay': '2ccf71e14ade6dbc58848a249f9396cd36d061f64abb65f56152e8afb84585c6',
    'freeze': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def authenticate(root, parent, policy):
    code = root / 'code'
    manifest_path = code / 'RECOVERY_EXECUTION_MANIFEST.json'
    if sha256(manifest_path) != MANIFEST_SHA or sha256(parent) != PARENT_SHA:
        raise ValueError('Recovery source manifest or original R20 parent changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative, expected in manifest['execution_files'].items():
        path = (code / relative).resolve()
        if not path.is_relative_to(code) or not path.is_file():
            raise ValueError(f'Invalid source member: {relative}')
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Frozen source changed: {relative}')
    checkpoint = root / 'runs' / f'H1-{policy}' / 'last.pt'
    if sha256(checkpoint) != ENDPOINT_SHA[policy]:
        raise ValueError('Endpoint is not the independently verified fixed R26 checkpoint')
    return code, checkpoint, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--policy', required=True, choices=tuple(ENDPOINT_SHA))
    parser.add_argument('--output')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    root, parent = Path(args.root).resolve(), Path(args.parent).resolve()
    code, checkpoint, manifest = authenticate(root, parent, args.policy)
    if args.verify_only:
        print(json.dumps(dict(status='authenticated', policy=args.policy,
            source_files=len(manifest['execution_files']), checkpoint_sha256=ENDPOINT_SHA[args.policy])))
        return
    if not args.output:
        parser.error('--output is required for an evaluation')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing test attempt: {output}')
    if output.is_relative_to(code) or output.is_relative_to(root / 'runs'):
        raise ValueError('Test output must be outside the frozen code and training directories')
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    os.chdir(code)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(code / 'external' / 'DiffGRM')
    import torch
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.recovery_run import build_recovery
    from scripts.listwise_diffgrm.recovery_diagnostics import diagnostic_context
    from scripts.listwise_diffgrm.run import evaluate, validate_evaluation_checkpoint

    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This evaluation requires exactly one allocated H100')
    torch.set_num_threads(4)
    started = time.monotonic()
    controller, assets, _ = build_recovery(parent, args.policy, 'cuda:0')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    controller.restore(saved)
    trainer = controller.trainer
    validate_evaluation_checkpoint(saved['inner'], trainer.plan)
    expected_t = 280 if args.policy == 'decay' else 200
    if (trainer.r_updates, trainer.t_updates, controller.progress['phase'], controller.progress['update']) != (18018, expected_t, 33, 0):
        raise ValueError('Expected the completed R26 endpoint')
    cfg = trainer.config
    ev = cfg['evaluation']
    if (cfg['architecture'], cfg['list_length'], cfg['history_max'], ev['test_beam'], ev['batch_size'], ev['candidate_lists']) != ('H', 5, 50, 32, 32, 32):
        raise ValueError('Test protocol differs from original R20')
    if ev['aggregate_ks'] != [10, 20, 50] or ev['filter_history']:
        raise ValueError('Aggregation or history filtering changed')
    if ev['aggregation'] != 'list_score_order_then_item_position_first_occurrence' or ev['collision_rule'] != 'reconstruction_error_then_original_id':
        raise ValueError('Catalog or aggregation rule changed')
    if any(p.is_floating_point() and p.dtype != torch.float32 for module in (trainer.model, trainer.alignment) for p in module.parameters()):
        raise ValueError('FP32 model and alignment parameters are required')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('TF32 must remain disabled')
    before = fingerprint(controller.checkpoint())
    request = dict(policy=args.policy, split='test', checkpoint=str(checkpoint),
        checkpoint_sha256=ENDPOINT_SHA[args.policy], parent_sha256=PARENT_SHA,
        execution_manifest_sha256=MANIFEST_SHA, launcher_sha256=sha256(__file__),
        config=cfg, r_updates=trainer.r_updates, t_updates=trainer.t_updates,
        before_state_sha256=before, inference_precision='fp32', training_performed=False,
        cuda_device=torch.cuda.get_device_name(0), fixed_endpoint='R26',
        dataset_test_sha256=sha256(Path(cfg['dataset']) / 'test.parquet'))
    write_json(output / 'request.json', request)
    with diagnostic_context(trainer), torch.autocast('cuda', enabled=False):
        summary = evaluate(trainer, assets, split='test', destination=output / 'evaluation',
                           beam=32, limit=None, subset_manifest=None)
    after = fingerprint(controller.checkpoint())
    if before != after or sha256(checkpoint) != ENDPOINT_SHA[args.policy] or sha256(parent) != PARENT_SHA:
        raise RuntimeError('Evaluation changed checkpoint state or source bytes')
    if summary['examples'] != 53194 or summary['split'] != 'test' or not summary['full_split']:
        raise RuntimeError('Incomplete or incorrect test split')
    receipt = dict(status='complete', policy=args.policy, examples=summary['examples'],
        checkpoint_sha256=ENDPOINT_SHA[args.policy], parent_sha256=PARENT_SHA,
        execution_manifest_sha256=MANIFEST_SHA, launcher_sha256=request['launcher_sha256'],
        before_state_sha256=before, after_state_sha256=after,
        predictions_sha256=sha256(output / 'evaluation' / 'per_sample.jsonl.gz'),
        summary_sha256=sha256(output / 'evaluation' / 'summary.json'),
        request_sha256=sha256(output / 'request.json'), seconds=time.monotonic() - started,
        training_performed=False, fixed_endpoint='R26')
    write_json(output / 'complete.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
