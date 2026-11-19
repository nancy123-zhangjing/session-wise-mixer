"""Read-only real R5 compatibility check; no optimizer step or dataset evaluation."""
import copy
import json
from pathlib import Path

# Select the same historical source/engine as the toy tests.
from test_grid import ROOT, ORIGINAL, CODE
import torch
from scripts.listwise_diffgrm.assets import load_assets
from scripts.listwise_diffgrm.runtime import build_model
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.training import AlternatingTrainer, sha256
from protocol import GridController, PARENT_SHA256, fingerprint


def main():
    source = ORIGINAL / 'runs/H1/initial.pt'
    if sha256(source) != PARENT_SHA256:
        raise ValueError('Unexpected R5 checkpoint')
    parent = torch.load(source, map_location='cpu', weights_only=False)
    for name, expected in parent['binding']['code'].items():
        if sha256(CODE / name) != expected:
            raise ValueError(f'Historical code changed: {name}')
    for name, expected in parent['binding']['engine'].items():
        if sha256(CODE / 'external/DiffGRM' / name) != expected:
            raise ValueError(f'Historical engine changed: {name}')
    asset_root = ORIGINAL / 'assets/pse_v1'
    if sha256(asset_root / 'manifest.json') != parent['binding']['asset_manifest']:
        raise ValueError('Asset binding changed')
    assets = load_assets(asset_root)
    results = []
    for arm in ('B', 'C'):
        config = json.loads((ROOT / f'config_{arm}.json').read_text())
        cfg = config['base_config']
        if cfg != parent['binding']['config']:
            raise ValueError('Scientific config changed')
        trainer = AlternatingTrainer(build_model(cfg, assets.make_tokenizer()),
                                    build_alignment(assets, True, config_overrides=cfg['alignment']),
                                    cfg, 709443, precision='bf16')
        inspection = copy.deepcopy(parent)
        inspection['rng']['cuda'] = None  # disclosed CPU-only inspection
        trainer.restore(inspection, parent['binding'], arm='H1')
        trainer.model.eval()
        # Synthetic visible history plus PAD, with actual frozen catalog SIDs.
        ids = torch.arange(1, 101).reshape(2, 50)
        ids[0, :20] = 0
        mask = ids.ne(0)
        hs = trainer.alignment.hard_codes[ids]
        partial = torch.full((2, 4), -1)
        with torch.no_grad():
            original_logits = trainer.model._decode(trainer.model._encode(hs, mask), partial, item_index=0)
        spec = {k: v for k, v in config.items() if k != 'base_config'}
        control = GridController(trainer, parent, parent['binding'], spec, exact_rng=False)
        trainer.model.eval()
        with torch.no_grad():
            new_logits = trainer.model._decode(trainer.model._encode(hs, mask), partial, item_index=0)
        unchanged = torch.equal(original_logits, new_logits)
        old_weights = {k: v for k, v in trainer.model.state_dict().items()
                       if not k.startswith('backbone.item_local_attention.')}
        inherited = all((
            fingerprint(old_weights) == fingerprint(parent['model']),
            fingerprint(trainer.alignment.state_dict()) == fingerprint(parent['alignment']),
            fingerprint(trainer.r_optimizer.state_dict()['state']) == fingerprint(parent['r_optimizer']['state']),
            fingerprint(trainer.t_optimizer.state_dict()) == fingerprint(parent['t_optimizer']),
            fingerprint(trainer.scheduler.state_dict()) == fingerprint(parent['scheduler']),
        ))
        if not unchanged or not inherited:
            raise ValueError('Real checkpoint compatibility failed')
        control.restore(control.checkpoint())
        results.append(dict(arm=arm, logits_bitwise_equal=unchanged, inherited_state_equal=inherited,
                            grid_parameters=sum(p.numel() for p in control.grid.parameters()),
                            r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                            r_lr=trainer.r_optimizer.param_groups[0]['lr']))
    report = dict(status='passed', device='cpu', torch_version=torch.__version__,
                  parent_sha256=PARENT_SHA256, source_and_asset_hashes_verified=True,
                  cuda_rng_restored=False, optimizer_updates_performed=0,
                  full_dataset_evaluation_performed=False, results=results)
    (ROOT / 'PARENT_CPU_CHECK.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
