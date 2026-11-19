"""Read-only codebook health of protocol v2 H checkpoints (no training, no writes to run dirs)."""
import gc, json, sys
from pathlib import Path
import torch
from scripts.listwise_diffgrm.run import build_context
from scripts.listwise_diffgrm.recovery_diagnostics import diagnose_catalog

V = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_v2_20261118')
KEYS = ('r_updates', 't_updates', 'used_codes_per_slot', 'unique_sid_count', 'collision_item_fraction',
        'largest_bucket_size', 'mean_nearest_reconstruction_sse', 'mixer_delta_over_base_global_norm',
        'effective_displacement_over_initial_global_norm')
out = {}
SEED, ARM = int(sys.argv[1]), sys.argv[2]
for seed in (SEED,):
    cfg = json.loads((V/'configs'/f'h_{seed}.json').read_text())
    base = V/'h'/f'seed{seed}'
    for arm in (ARM,):
        trainer, assets, data, binding = build_context(cfg, 'cpu', 'bf16', arm == 'H1', prepare_training=False)
        available = sorted(int(p.stem[1:]) for p in (base/arm/'checkpoints').glob('R*.pt'))
        epochs = [e for e in (10, 15, 19, 20, 21, 25, 30, 35) if e in available]
        if available and available[-1] not in epochs:
            epochs.append(available[-1])
        points = ([('warmup', 5, base/'warmup'/'checkpoints'/'R05.pt')] if arm == 'H0' else []) + \
                 [(arm, e, base/arm/'checkpoints'/f'R{e:02d}.pt') for e in epochs]
        previous, rows = None, []
        for label, epoch, path in points:
            state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if state['binding'] != binding:
                raise SystemExit(f'binding mismatch for {path}')
            trainer.alignment.load_state_dict(state['alignment'])
            trainer.r_updates, trainer.t_updates = state['r_updates'], state['t_updates']
            metrics, snapshot = diagnose_catalog(trainer, previous)
            previous = snapshot
            del state
            gc.collect()
            row = {'label': label, 'R': epoch, **{k: metrics.get(k) for k in KEYS},
                   'normalized_entropy': [round(s['normalized_entropy'], 4) for s in metrics['per_slot']],
                   'sid_item_change_vs_previous': metrics['previous_change']['sid_item_change_fraction']}
            rows.append(row)
            print(json.dumps({'seed': seed, 'arm': arm, **row}), flush=True)
        out[f'{seed}_{arm}'] = rows
dest = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_codebook_check_20261118')
dest.mkdir(parents=True, exist_ok=True)
(dest/f'RESULTS_{SEED}_{ARM}.json').write_text(json.dumps(out, indent=2))
print('saved', dest/f'RESULTS_{SEED}_{ARM}.json')
