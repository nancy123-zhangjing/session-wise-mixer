"""TIGER RQ pretraining probe (user-approved, CPU only, isolated output).

Step 1: from the shared T0/T1 parent, take 5 updates with (a) the formal Adagrad lr0.4
and (b) AdamW lr1e-3, recording per-tensor relative weight change, encoder-output
scale, codebook scale and reconstruction loss.
Step 2: 500-epoch pretraining probes, changing ONLY the optimizer:
  A  Adagrad lr0.4 (formal), B1 Adagrad lr0.05, B2 Adagrad lr0.01,
  C  AdamW lr1e-3 wd1e-4, 50-epoch warmup then linear decay over the formal 20000-epoch
     budget, grad-norm clip 1 (ETEGRec/DIGER/RQCD recipe).
Each for no-Mixer (T0) and intra-layer Mixer (T1). Same parent, model, loss, batch size,
data order (same seeded permutation stream) and evaluation code as the formal run.
Health criteria are fixed before running (see HEALTH below); no recommender training,
no formal artifact is read for writing.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HEALTH = {'reconstruction_below': 0.5, 'min_layer_utilization': 0.90, 'max_bucket_at_most': 64,
          'max_mixer_correction_ratio': 3.0,
          'collision_note': 'redundancy compared with the duplicate-feature floor of the content asset'}
CONFIGS = {
    'A_adagrad_0.4': dict(kind='adagrad', lr=0.4),
    'B1_adagrad_0.05': dict(kind='adagrad', lr=0.05),
    'B2_adagrad_0.01': dict(kind='adagrad', lr=0.01),
    'C_adamw_1e-3': dict(kind='adamw', lr=1e-3, wd=1e-4, warmup_epochs=50, clip=1.0),
}
EPOCHS, EVAL_EVERY = 500, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.code_root.resolve(), args.output.resolve()
    assert root not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'ar_deps')); sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    from scripts.session_wise_ar.config import load
    from scripts.session_wise_ar.tokenizer import make_tokenizer
    from scripts.session_wise_ar.catalog import encoding_stats
    from scripts.rqcd_mixer.common import read_state, write_json, sha256, seed_for, seed_all
    from scripts.rqcd_mixer.data import load_content, PermutationStream
    torch.set_num_threads(3)
    data_root = root.parent / 'goal_20261027_r1'
    content_path = data_root / 'assets/pse_v1/content_features.npy'
    prepared = root / 'parents_devicefix/tiger'
    parent = read_state(prepared / 'initial.pt')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    assert manifest['initial_sha256'] == sha256(prepared / 'initial.pt')
    cfg0 = load('AR-T0')
    content = load_content(content_path, cfg0, 'cpu')
    floor = 1 - len(np.unique(content.numpy().round(7), axis=0)) / len(content)
    write_json(out / 'PROTOCOL.json', dict(doc=__doc__, health=HEALTH, configs=CONFIGS, epochs=EPOCHS,
               eval_every=EVAL_EVERY, parent_sha256=manifest['initial_sha256'], script_sha256=sha256(Path(__file__)),
               content_sha256=sha256(content_path), duplicate_feature_redundancy_floor=floor,
               formal_rq=cfg0['rq']))
    per_epoch = math.ceil(len(content) / cfg0['rq']['batch_size'])

    def build(arm, spec):
        cfg = load(arm)
        tok = make_tokenizer(cfg)
        tok.load_state_dict(parent['tokenizers']['intra' if cfg['mixer_enabled'] else 'plain'])
        if spec['kind'] == 'adagrad':
            opt = torch.optim.Adagrad(tok.parameters(), lr=spec['lr'], lr_decay=0, weight_decay=0,
                                      initial_accumulator_value=0, eps=1e-10)
            sched = None
        else:
            opt = torch.optim.AdamW(tok.parameters(), lr=spec['lr'], weight_decay=spec['wd'])
            sched = get_linear_schedule_with_warmup(opt, spec['warmup_epochs'] * per_epoch, cfg['rq']['epochs'] * per_epoch)
        stream = PermutationStream(len(content), seed_for(cfg['seed'], cfg['family'], 'pretrain-data'))
        seed_all(seed_for(cfg['seed'], cfg['family'], 'pretrain-updates'))
        return cfg, tok, opt, sched, stream

    def step(tok, opt, sched, stream, cfg, clip):
        tok.train()
        idx = stream.take(cfg['rq']['batch_size'])
        loss, parts = tok.loss(content[idx])
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(tok.parameters(), float('inf') if clip is None else clip)
        opt.step()
        if sched is not None:
            sched.step()
        return float(loss), {k: float(v) for k, v in parts.items()}, float(gnorm), len(idx)

    @torch.no_grad()
    def scales(tok):
        tok.eval()
        z = tok.encoder(content)
        base = tok.base_books(); eff = tok.effective_books()
        brms = base.square().mean((-2, -1)).sqrt(); erms = eff.square().mean((-2, -1)).sqrt()
        corr = (eff - base).square().mean((-2, -1)).sqrt() / brms.clamp_min(1e-12)
        return dict(z_rms=float(z.square().mean().sqrt()), base_rms=brms.tolist(), effective_rms=erms.tolist(),
                    correction_ratio=corr.tolist())

    @torch.no_grad()
    def evaluate(tok, cfg):
        tok.eval()
        semantic = tok.indices(content)
        sid = torch.cat((semantic.new_full((1, 3), -1), semantic), dim=0)
        stats = encoding_stats(tok, sid)
        stats['max_collision_bucket'] = int(torch.unique(semantic, dim=0, return_counts=True)[1].max())
        z, q, vq, _ = tok.quantize(content)
        recon = (tok.decoder(q) - content).square().sum(-1).mean()
        stats.update(full_reconstruction=float(recon), full_vq=float(vq), **scales(tok))
        return stats

    # ---------------- Step 1: first five updates ----------------
    step1 = {}
    for arm in ('AR-T0', 'AR-T1'):
        for name in ('A_adagrad_0.4', 'C_adamw_1e-3'):
            spec = CONFIGS[name]
            cfg, tok, opt, sched, stream = build(arm, spec)
            rows = [dict(step=0, **scales(tok))]
            for s in range(1, 6):
                before = {n: p.detach().clone() for n, p in tok.named_parameters()}
                loss, parts, gnorm, _ = step(tok, opt, sched, stream, cfg, spec.get('clip'))
                rel = {n: float((p.detach() - before[n]).norm() / before[n].norm().clamp_min(1e-12))
                       for n, p in tok.named_parameters()}
                rows.append(dict(step=s, loss=loss, parts=parts, grad_norm=gnorm,
                                 max_relative_change=max(rel.values()),
                                 relative_change={k: rel[k] for k in rel if k.startswith(('encoder.0', 'encoder.6', 'books', 'mixer.out'))},
                                 **scales(tok)))
            step1[f'{arm}/{name}'] = rows
            print(json.dumps({'step1': f'{arm}/{name}', 'last': rows[-1]['max_relative_change'],
                              'z_rms': [r['z_rms'] for r in rows]}), flush=True)
    write_json(out / 'STEP1.json', step1)

    # ---------------- Step 2: 500-epoch probes ----------------
    step2 = {}
    for name, spec in CONFIGS.items():
        for arm in ('AR-T0', 'AR-T1'):
            key = f'{arm}/{name}'
            cfg, tok, opt, sched, stream = build(arm, spec)
            records, sums, count, status, t0 = [], {}, 0, 'completed', time.time()
            records.append(dict(epoch=0, **evaluate(tok, cfg)))
            try:
                for s in range(1, EPOCHS * per_epoch + 1):
                    loss, parts, gnorm, n = step(tok, opt, sched, stream, cfg, spec.get('clip'))
                    for k, v in dict(total=loss, **parts).items():
                        sums[k] = sums.get(k, 0) + v * n
                    count += n
                    if s % (EVAL_EVERY * per_epoch) == 0:
                        rec = dict(epoch=s // per_epoch, train_loss={k: v / count for k, v in sums.items()}, **evaluate(tok, cfg))
                        records.append(rec); sums, count = {}, 0
                        print(json.dumps({'probe': key, 'epoch': rec['epoch'], 'recon': rec['full_reconstruction'],
                                          'util': rec.get('utilization'), 'coll': rec.get('collision_redundancy'),
                                          'maxb': rec['max_collision_bucket'], 'corr': rec['correction_ratio']}), flush=True)
            except FloatingPointError as e:
                status = f'failed: {e}'
            final = records[-1]
            util = final.get('utilization', [0, 0, 0])
            health = dict(reconstruction=final['full_reconstruction'] < HEALTH['reconstruction_below'],
                          utilization=min(util) >= HEALTH['min_layer_utilization'],
                          max_bucket=final['max_collision_bucket'] <= HEALTH['max_bucket_at_most'],
                          mixer=(max(final['correction_ratio']) <= HEALTH['max_mixer_correction_ratio']) if arm == 'AR-T1' else True)
            step2[key] = dict(status=status, seconds=time.time() - t0, records=records, health=health,
                              healthy=status == 'completed' and all(health.values()))
            write_json(out / 'STEP2.json', step2)
    write_json(out / 'COMPLETE.json', dict(step1_keys=list(step1), step2={k: dict(healthy=v['healthy'], health=v['health'],
               status=v['status'], final=v['records'][-1]) for k, v in step2.items()}, floor=floor))
    print('@@DONE@@', flush=True)


if __name__ == '__main__':
    main()
