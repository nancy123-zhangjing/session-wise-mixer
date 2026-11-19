"""One H100, original H1 loop, one allowed model ablation, stop after R20."""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from replay_support import (ORIGINAL, PARENT_SHA, BOUNDARIES, FIELDS,
                            BoundaryPublisher, StopAtR20, differences, sha256, write_json)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'code'
OUTPUT = ROOT / 'runs/H1-NoCodeTag'
FLAG = 'crosspse_code_embedding_enabled'
ENGINE_PREFIX = 'external/DiffGRM/'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage():
    manifest = read(ROOT / 'SOURCE_DELTA.json')
    for path, digests in manifest.items():
        if (sha256(ORIGINAL / 'code' / path) != digests['before']
                or sha256(ROOT / 'overlay' / path) != digests['after']):
            raise ValueError(f'Source delta does not match locked originals: {path}')
    shutil.copytree(ORIGINAL / 'code', SOURCE)
    for path in manifest:
        shutil.copyfile(ROOT / 'overlay' / path, SOURCE / path)


def load_engine():
    os.environ.update(DIFFGRM_ENGINE_ROOT=str(SOURCE / ENGINE_PREFIX),
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0, str(SOURCE))
    from scripts.listwise_diffgrm import run, training
    if Path(run.__file__).resolve() != SOURCE / 'scripts/listwise_diffgrm/run.py':
        raise RuntimeError('Wrong training source imported')
    return run, training


def migrate(run, training):
    import torch
    path = ORIGINAL / 'runs/H1/initial.pt'
    if sha256(path) != PARENT_SHA:
        raise ValueError('Parent SHA256 mismatch')
    parent = torch.load(path, map_location='cpu', weights_only=False)
    if (parent['arm'], parent['precision'], parent['r_updates'], parent['t_updates'],
            parent['progress']['phase'], parent['progress']['update']) != ('H1', 'bf16', 3465, 0, 5, 0):
        raise ValueError('Parent is not the original H1 R5/T0 boundary')
    old = parent['binding']
    if old['config'] != read(ORIGINAL / 'runs/H1/resolved_config.json'):
        raise ValueError('Resolved original configuration mismatch')
    config = copy.deepcopy(old['config'])
    if config['alignment'].get(FLAG, True) is not True:
        raise ValueError('Original must use the code tag')
    config['alignment'][FLAG] = False
    from scripts.listwise_diffgrm.runtime import validate_config
    validate_config(config)
    trainer, assets, data, binding = run.build_context(config, 'cuda:0', 'bf16', True)
    expected = copy.deepcopy(old)
    expected['config'] = config
    manifest = read(ROOT / 'SOURCE_DELTA.json')
    for path, digests in manifest.items():
        section, key = ('engine', path.removeprefix(ENGINE_PREFIX)) if path.startswith(ENGINE_PREFIX) else ('code', path)
        if old[section].get(key) != digests['before']:
            raise ValueError(f'Original checkpoint source hash mismatch: {path}')
        expected[section][key] = digests['after']
    if binding != expected:
        raise ValueError(f'Unexpected binding delta: {differences(expected, binding)}')
    if (len(trainer.plan), sum(p['updates'] for p in trainer.plan if p['kind'] == 'R'),
            sum(p['updates'] for p in trainer.plan if p['kind'] == 'T')) != (45, 24255, 400):
        raise ValueError('Original schedule horizon changed')
    for phase, (epoch, t) in BOUNDARIES.items():
        row = trainer.plan[phase - 1]
        count = training.expected_update_counts(trainer.plan, {'phase': phase, 'update': 0})
        if not row['validate'] or row['epoch'] != epoch or count != {'R': epoch * 693, 'T': t}:
            raise ValueError('Original evaluation schedule changed')
    migrated = dict(parent, binding=binding)
    progress = trainer.restore(migrated, binding, arm='H1')
    restored = trainer.checkpoint(binding, progress, 'H1')
    if differences({k: migrated[k] for k in FIELDS}, {k: restored[k] for k in FIELDS}):
        raise ValueError('Parent state did not restore exactly')
    mixer = trainer.alignment.mixer
    trainer.phase_mode('T')
    if (not mixer.mixer_enabled or mixer.code_embedding_enabled or mixer.code_embedding.requires_grad
            or not mixer.book_embedding.requires_grad or not mixer.base_codebooks.requires_grad
            or torch.count_nonzero(mixer.code_embedding).item()):
        raise ValueError('Incorrect no-code-tag intervention at the branch point')
    with torch.no_grad():
        disabled = mixer.crossed_codebooks()[0]
        mixer.code_embedding_enabled = True
        enabled = mixer.crossed_codebooks()[0]
        mixer.code_embedding_enabled = False
        codes = mixer.squared_distances(trainer.alignment.item_slots[1:].float(), disabled.float()).argmin(-1)
    if not torch.equal(disabled, enabled) or not torch.equal(codes, trainer.alignment.hard_codes[1:]):
        raise ValueError('Initial effective codebook or catalog changed')
    r5 = Path(progress['validations'][-1])
    summary = read(r5 / 'summary.json')
    if (progress['best_epoch'], progress['best_score'], summary['r_updates'], summary['t_updates'],
            summary['examples'], summary['beam'], summary['full_split']) != (
            5, summary['metrics']['list/ndcg@5'], 3465, 0, 71657, 32, True):
        raise ValueError('Parent validation is not the eligible R5 reference')
    # Resume enters the normal strict loader. Only the validated binding changes.
    destination = ROOT / 'parent_R05_T000.pt'
    training.atomic_save(destination, migrated)
    write_json(ROOT / 'MIGRATION.json', dict(status='passed', original_parent=str(path),
        original_sha256=PARENT_SHA, migrated_sha256=sha256(destination), parent_validation=str(r5),
        parent_ndcg5=progress['best_score'], parent_validation_sha256=sha256(r5 / 'summary.json'),
        original_binding=old, new_binding=binding, restore_exact=True,
        same_effective_codebook_and_sid=True, cuda_rng_restored=True, source_delta=manifest))
    return config, destination


def finish(run, config):
    import torch
    trainer, assets, _, binding = run.build_context(config, 'cuda:0', 'bf16', True)
    migration = read(ROOT / 'MIGRATION.json')
    candidates = [dict(epoch=5, t_updates=0, ndcg5=migration['parent_ndcg5'],
                       checkpoint=str(ROOT / 'parent_R05_T000.pt'),
                       checkpoint_sha256=migration['migrated_sha256'],
                       validation=migration['parent_validation'],
                       summary_sha256=migration['parent_validation_sha256'])]
    for epoch, _ in BOUNDARIES.values():
        candidates.append(read(OUTPUT / 'checkpoints' / f'R{epoch:02d}.json'))
    final_state = None
    for entry in candidates:
        path = Path(entry['checkpoint'])
        if sha256(path) != entry['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash mismatch at R{entry["epoch"]}')
        state = torch.load(path, map_location='cpu', weights_only=False)
        run.validate_evaluation_checkpoint(state, trainer.plan)
        progress = trainer.restore(state, binding, arm='H1')
        actual = trainer.checkpoint(binding, progress, 'H1')
        if differences({k: state[k] for k in FIELDS}, {k: actual[k] for k in FIELDS}):
            raise ValueError(f'Strict restore mismatch at R{entry["epoch"]}')
        if torch.count_nonzero(trainer.alignment.mixer.code_embedding).item():
            raise ValueError('Disabled center tag changed')
        entry['restore_exact'] = True
        final_state = progress
        del state, actual
    best = max(candidates, key=lambda row: (row['ndcg5'], -row['epoch']))
    if (final_state['best_epoch'], final_state['best_score']) != (best['epoch'], best['ndcg5']):
        raise ValueError('Training selection disagrees with complete validation grid')
    write_json(ROOT / 'SELECTION.json', dict(best=best, candidates=candidates, metric='list/ndcg@5'))
    print(json.dumps(dict(event='best_locked', epoch=best['epoch'], ndcg5=best['ndcg5'])), flush=True)
    state = torch.load(best['checkpoint'], map_location='cpu', weights_only=False)
    trainer.restore(state, binding, arm='H1')
    dest = OUTPUT / 'test_best'
    test = run.evaluate(trainer, assets, split='test', destination=dest)
    if (test['examples'], test['full_split'], test['beam'], test['r_updates'], test['t_updates']) != (
            53194, True, 32, 693 * best['epoch'], best['t_updates']):
        raise ValueError('Best test protocol mismatch')
    write_json(ROOT / 'TEST_RECEIPT.json', dict(epoch=best['epoch'], checkpoint_sha256=best['checkpoint_sha256'],
        selection_sha256=sha256(ROOT / 'SELECTION.json'), summary_sha256=sha256(dest / 'summary.json'),
        predictions_sha256=sha256(dest / 'per_sample.jsonl.gz'), metrics=test['metrics']))


def main():
    import torch
    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected exactly one H100')
    run, training = load_engine()
    config, parent = migrate(run, training)
    print(json.dumps(dict(event='parent_restored', R=5, T=0)), flush=True)
    publish = run.publish_checkpoint
    run.publish_checkpoint = BoundaryPublisher(publish, ROOT)
    try:
        run.train(SimpleNamespace(command='train', arm='H1', output=str(OUTPUT), resume=str(parent),
                                  shared_warmup=None, device='cuda:0', precision='bf16'), config)
    except StopAtR20:
        write_json(OUTPUT / 'complete.json', dict(status='training_validation_complete', R=20, T=200))
    else:
        raise RuntimeError('Trainer did not stop at R20')
    finally:
        run.publish_checkpoint = publish
    gc.collect()
    torch.cuda.empty_cache()
    finish(run, config)
    write_json(ROOT / 'GPU_COMPLETE.json', dict(status='passed', R=20, T=200, finished_unix=time.time()))


if __name__ == '__main__':
    if sys.argv[1:] == ['stage']:
        stage()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('usage: experiment.py [stage]')
