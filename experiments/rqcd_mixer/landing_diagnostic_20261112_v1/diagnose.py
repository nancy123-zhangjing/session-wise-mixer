"""Read-only inference diagnosis; imports the immutable formal experiment code."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=8192)
    args = p.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from scripts.rqcd_mixer.common import (read_state, sha256, write_json, configure_runtime,
                                          attention_backend, tensor_digest)
    from scripts.rqcd_mixer.catalog import Catalog, generate
    from scripts.rqcd_mixer.data import PackedSplit
    from scripts.rqcd_mixer.evaluation import item_metrics
    from scripts.rqcd_mixer.model import Recommender, Diffusion
    from scripts.rqcd_mixer.tokenizer import distances

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise RuntimeError('Completed diagnosis already exists; do not overwrite')
    meta = json.loads((args.source/'prepared/PREPARED.json').read_text())
    for path, expected in meta['source'].items():
        assert sha256(args.source/path) == expected, path
    cfg = meta['config']
    runtime = json.loads((args.source/'runtime.locked.json').read_text())
    env = configure_runtime(runtime, 'cuda')
    data = PackedSplit(meta['dataset_path'], 'validation', cfg, device='cpu')
    # Hash selection covers the full time span and is independent of targets/results.
    indices = sorted(sorted(range(len(data)), key=lambda i: hashlib.sha256(
        ('landing-diagnostic-v1:20261112:'+str(data.sample_ids[i])).encode()).digest())[:args.count])
    assert len(indices) == args.count
    ids = [data.sample_ids[i] for i in indices]
    subset = {'method': 'smallest SHA256 of fixed seed prefix plus sample_id; sorted by original row',
              'seed': 20261112, 'split': 'validation', 'indices': indices, 'sample_ids': ids,
              'count': len(indices), 'validation_count': len(data)}
    write_json(out/'SUBSET.json', subset)
    noise_path = args.source/'prepared/validation_noise.npy'
    noise_meta = json.loads(noise_path.with_suffix('.json').read_text())
    assert sha256(noise_path) == noise_meta['sha256']
    noise = np.load(noise_path, mmap_mode='r')
    write_json(out/'PROTOCOL.json', {'source': str(args.source), 'environment': env,
        'subset_sha256': sha256(out/'SUBSET.json'), 'noise_sha256': sha256(noise_path),
        'checkpoint': 'validation-selected best, B0/B1/B2', 'training': False,
        'generation': 'original RQ rollout; original history, 20-step DDIM, BF16, original fixed noise',
        'comparison': 'same captured generated z; RQ vs full-catalog squared Euclidean NN to pre-quantization z_i',
        'prefix': 'both readouts use exactly the RQ rollout generated vectors; NN is post-hoc, never feeds back',
        'exclusion': 'each readout excludes its own earlier selections; history items allowed as original',
        'oracle': 'all catalog z plus normalization roundtrip; validation targets with true earlier-item exclusions'})

    @torch.inference_mode()
    def nearest(cat, z, selected):
        score = distances(z.float(), cat.z)
        if selected.shape[1]:
            score.scatter_(1, selected-1, float('inf'))
        return score.argmin(-1)+1

    def summarize(pred, truth):
        values = item_metrics(pred, truth, cfg['catalog_size'])
        return {k: float(v.double().mean()) for k, v in values.items()}, values

    all_results = {}
    start_all = time.perf_counter()
    with torch.inference_mode(), attention_backend(runtime['attention_backend']):
        for arm in ('B0', 'B1', 'B2'):
            started = time.perf_counter()
            best = json.loads((args.source/arm/'BEST.json').read_text())
            cp = args.source/arm/best['checkpoint']
            cp_hash = sha256(cp)
            assert cp_hash == best['sha256']
            state = read_state(cp)
            assert state['config'] == cfg and state['runtime'] == runtime
            cat = Catalog(state['catalog'], 'cuda')
            model = Recommender(cfg).cuda().eval()
            model.load_state_dict(state['model'])
            diffusion = Diffusion(cfg).cuda().eval()
            diffusion.load_state_dict(state['diffusion'])
            before = tensor_digest(model.state_dict())
            cat_before = cat.version_hash
            # Oracle all-item checks, independent of popularity and subset composition.
            oracle = {}
            for name in ('original_z', 'normalize_inverse_z'):
                rq_ok, nn_ok, sid_ok, raw_ok, errors = 0, 0, 0, 0, []
                for begin in range(0, cat.count, 256):
                    target = torch.arange(begin+1, min(begin+257, cat.count+1), device='cuda')
                    z = cat.z[target-1]
                    if name == 'normalize_inverse_z':
                        z = (z-cat.stats['mean'])/cat.stats['scale']*cat.stats['scale']+cat.stats['mean']
                    empty = torch.empty((len(z), 0), dtype=torch.long, device='cuda')
                    rq, diag = cat.land(z, empty)
                    nn = nearest(cat, z, empty)
                    rq_ok += int((rq == target).sum())
                    nn_ok += int((nn == target).sum())
                    sid_ok += int((cat.sid[rq-1] == cat.sid[target-1]).all(-1).sum())
                    raw_ok += int(diag['raw_valid'].sum())
                    for row in torch.where((rq != target) | (nn != target))[0].tolist():
                        errors.append({'target': int(target[row]), 'rq': int(rq[row]), 'nn': int(nn[row]),
                            'rq_true_squared_distance_fp64': float(((z[row].double()-cat.z[target[row]-1].double())**2).sum()),
                            'rq_selected_squared_distance_fp64': float(((z[row].double()-cat.z[rq[row]-1].double())**2).sum())})
                oracle[name] = {'count': cat.count, 'rq_correct': rq_ok, 'nn_correct': nn_ok,
                    'rq_accuracy': rq_ok/cat.count, 'nn_accuracy': nn_ok/cat.count,
                    'rq_sid_accuracy': sid_ok/cat.count, 'raw_valid': raw_ok/cat.count, 'errors': errors}

            predictions, truths, vectors, diagnostics, oracle_target = [], [], [], [], []
            for begin in range(0, len(indices), runtime['evaluation_batch']):
                row_indices = indices[begin:begin+runtime['evaluation_batch']]
                history, truth = data.batch(torch.tensor(row_indices), 'cuda')
                initial = torch.from_numpy(np.array(noise[row_indices], copy=True)).cuda()
                captured = []
                original_land = cat.land
                def capture(z, selected):
                    captured.append(z.detach().clone())
                    return original_land(z, selected)
                cat.land = capture
                try:
                    rq, diag = generate(model, diffusion, cat, history, initial, runtime)
                finally:
                    cat.land = original_land
                zlist = torch.stack(captured, 1)
                assert torch.isfinite(zlist).all()
                nn = torch.empty((len(history), 0), dtype=torch.long, device='cuda')
                oracle_pair = []
                for j in range(cfg['list_length']):
                    item = nearest(cat, zlist[:, j], nn)
                    nn = torch.cat((nn, item[:, None]), 1)
                    truez = cat.z[truth[:, j]-1]
                    oracle_rq, _ = cat.land(truez, truth[:, :j])
                    oracle_nn = nearest(cat, truez, truth[:, :j])
                    oracle_pair.append(torch.stack((oracle_rq == truth[:, j], oracle_nn == truth[:, j]), -1))
                assert item_metrics(rq, truth, cat.count)['completion_rate'].all()
                assert item_metrics(nn, truth, cat.count)['completion_rate'].all()
                predictions.append(torch.stack((rq, nn), 1).cpu())
                truths.append(truth.cpu())
                vectors.append(zlist.cpu())
                diagnostics.append({k: v.cpu() for k, v in diag.items()})
                oracle_target.append(torch.stack(oracle_pair, 1).cpu())
                print(json.dumps({'arm': arm, 'completed': begin+len(history), 'total': len(indices)}), flush=True)
            pred, truth = torch.cat(predictions), torch.cat(truths)
            rq_summary, rq_values = summarize(pred[:, 0], truth)
            nn_summary, nn_values = summarize(pred[:, 1], truth)
            diff = (nn_values['ndcg@5']-rq_values['ndcg@5']).double().numpy()
            # Paired resampling by user keeps repeated requests from a user together.
            user_ids = [str(data.users[i]) for i in indices]
            unique, inv = np.unique(user_ids, return_inverse=True)
            counts = np.bincount(inv)
            sums = np.bincount(inv, weights=diff)
            rng = np.random.default_rng(20261112)
            boot = np.empty(3000)
            for b in range(len(boot)):
                chosen = rng.integers(0, len(unique), len(unique))
                boot[b] = sums[chosen].sum()/counts[chosen].sum()
            # Check original full-validation predictions on the exact selected sample IDs.
            lookup = {str(s): n for n, s in enumerate(ids)}
            matched = []
            summary_paths = list((args.source/arm/'validation').glob('*/summary.json'))
            original_summary = None
            for path in summary_paths:
                summary = json.loads(path.read_text())
                if summary['identity']['checkpoint_sha256'] == cp_hash:
                    original_summary = path
                    break
            if original_summary is None:
                raise RuntimeError('Cannot locate immutable original validation for chosen checkpoint')
            original_pred = torch.empty_like(pred[:, 0])
            with gzip.open(original_summary.parent/'predictions.jsonl.gz', 'rt') as f:
                for line in f:
                    row = json.loads(line)
                    n = lookup.get(str(row['sample_id']))
                    if n is not None:
                        original_pred[n] = torch.tensor(row['prediction'])
                        assert row['target'] == truth[n].tolist()
                        matched.append(n)
            assert len(set(matched)) == len(ids)
            diag = {k: torch.cat([d[k] for d in diagnostics]) for k in diagnostics[0]}
            target_oracle = torch.cat(oracle_target)
            result = {'arm': arm, 'R': state['r_epoch'], 'T': state['t_updates'],
                'checkpoint': str(cp), 'checkpoint_sha256': cp_hash, 'count': len(indices),
                'oracle': oracle,
                'validation_target_oracle_by_position': target_oracle.double().mean(0).tolist(),
                'rq': rq_summary, 'nn_same_vectors': nn_summary,
                'ndcg_difference_nn_minus_rq': float(diff.mean()),
                'paired_user_bootstrap_95pct': np.quantile(boot, [.025, .975]).tolist(),
                'unique_users': len(unique),
                'rq_hit_sessions': int(rq_values['hr@5'].sum()), 'nn_hit_sessions': int(nn_values['hr@5'].sum()),
                'both_hit_sessions': int(((rq_values['hr@5'] > 0) & (nn_values['hr@5'] > 0)).sum()),
                'prediction_agreement_by_position': (pred[:, 0] == pred[:, 1]).double().mean(0).tolist(),
                'first_position_hit_any_target': {name: float((pred[:, k, 0, None] == truth).any(-1).double().mean())
                    for k, name in enumerate(('rq', 'nn'))},
                'raw_valid_by_position': diag['raw_valid'].double().mean(0).tolist(),
                'original_validation_prediction_agreement': float((original_pred == pred[:, 0]).double().mean()),
                'original_validation_exact_list_agreement': float((original_pred == pred[:, 0]).all(-1).double().mean()),
                'original_subset_metrics': summarize(original_pred, truth)[0],
                'model_unchanged': tensor_digest(model.state_dict()) == before,
                'catalog_unchanged': cat.version_hash == cat_before,
                'seconds': time.perf_counter()-started}
            assert result['model_unchanged'] and result['catalog_unchanged']
            armout = out/arm
            armout.mkdir(exist_ok=True)
            np.savez_compressed(armout/'vectors_and_predictions.npz', indices=np.array(indices),
                z=torch.cat(vectors).numpy(), predictions=pred.numpy(), target=truth.numpy(),
                original_predictions=original_pred.numpy(), oracle_targets=target_oracle.numpy())
            with gzip.open(armout/'predictions.jsonl.gz', 'wt', encoding='utf-8') as f:
                for n, sid in enumerate(ids):
                    f.write(json.dumps({'sample_id': sid, 'user_id': user_ids[n], 'target': truth[n].tolist(),
                        'rq': pred[n, 0].tolist(), 'nn_same_vectors': pred[n, 1].tolist(),
                        'rq_ndcg': float(rq_values['ndcg@5'][n]), 'nn_ndcg': float(nn_values['ndcg@5'][n])})+'\n')
            result['artifact_sha256'] = {f.name: sha256(f) for f in armout.iterdir() if f.is_file()}
            write_json(armout/'SUMMARY.json', result)
            all_results[arm] = result
            print(json.dumps({'arm_complete': arm, 'rq': rq_summary, 'nn': nn_summary}), flush=True)
            del state, model, diffusion, cat
            torch.cuda.empty_cache()
    write_json(out/'COMPLETE.json', {'status': 'DIAGNOSTICS_COMPLETE', 'training_started': False,
        'seconds': time.perf_counter()-start_all, 'results': all_results,
        'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(out/'PROTOCOL.json'),
        'subset_sha256': sha256(out/'SUBSET.json')})


if __name__ == '__main__':
    main()
