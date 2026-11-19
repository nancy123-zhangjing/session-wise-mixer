"""D1 teacher direct retrieval + D2 one-step vs multi-step DDIM. Inference only.

D1: frozen SASRec teacher (the CF target source) produces a 5-item list for each
session, scored with the same list NDCG@5. Variants: (a) one-shot top-5 from one
forward pass; (b) rolling greedy, appending each chosen item to the history
window (the teacher's own training view), mirroring item-by-item generation.
Both exclude only items already in the current list (CF protocol).
D2: CF checkpoints regenerate lists with 1, 2 or 5 DDIM steps (same fixed noise,
same catalog landing, same prefix rule); 20-step results are the stored formal
predictions. One configuration re-runs 20 steps and must reproduce the stored
predictions exactly (environment check). No training, no optimizer, no writes
outside --output.
"""
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

B = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'
FIVE = B + 'session_wise_ar_five_20261113_v1/runs/CF-M0/train/'
M1 = B + 'rqcd_targets_formal_20261113_v1/cf/'
FRZ = B + 'cf_freeze_r9_20261114_v1/runs/'
CHECKPOINTS = {  # name: (checkpoint, {split: stored 20-step prediction dir})
    'M0_best_R9': (FIVE + 'checkpoints/R09_T040.pt', {'validation': FIVE + 'validation/R09', 'test': FIVE + 'test/best'}),
    'M1_best_R9': (M1 + 'checkpoints/R09_T040.pt', {'validation': M1 + 'validation/R09', 'test': M1 + 'test/best'}),
    'M0F_R35': (FRZ + 'M0/train/checkpoints/R35_T040.pt', {'validation': FRZ + 'M0/train/validation/R35', 'test': FRZ + 'M0/train/test/final'}),
    'M1F_R35': (FRZ + 'M1/train/checkpoints/R35_T040.pt', {'validation': FRZ + 'M1/train/validation/R35', 'test': FRZ + 'M1/train/test/final'}),
}
STEPS = (1, 2, 5)
REPRODUCE = ('M1_best_R9', 'test')
SPLITS = {'validation': 71657, 'test': 53194}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from scripts.rqcd_mixer.common import read_state, sha256, write_json, configure_runtime, attention_backend, tensor_digest
    from scripts.rqcd_mixer.catalog import Catalog, generate
    from scripts.rqcd_mixer.data import PackedSplit
    from scripts.rqcd_mixer.evaluation import item_metrics
    from scripts.rqcd_mixer.model import Recommender, Diffusion
    from scripts.rqcd_mixer.teacher import SASRecTeacher

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.source / 'cf_prepared/PREPARED.json').read_text())
    for name, expected in meta['source'].items():
        assert sha256(args.source / name) == expected, name
    runtime = json.loads((args.source / 'runtime.locked.json').read_text())
    environment = configure_runtime(runtime, 'cuda')
    script_sha = sha256(Path(__file__))
    write_json(out / 'PROTOCOL.json', {'doc': __doc__, 'script_sha256': script_sha, 'runtime': runtime,
                                       'environment': environment, 'checkpoints': CHECKPOINTS, 'steps': STEPS,
                                       'reproduce': REPRODUCE, 'training': False})
    started = time.perf_counter()
    results = {'teacher': {}, 'cf': {}}

    def bootstrap(diff, users, reps=2000, seed=2024):
        u, inv = np.unique(np.asarray([str(x) for x in users]), return_inverse=True)
        cnt = np.bincount(inv).astype(float); s = np.bincount(inv, weights=diff)
        rng = np.random.default_rng(seed); boots = np.empty(reps)
        for r in range(reps):
            w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
            boots[r] = (w @ s) / (w @ cnt)
        return np.quantile(boots, [.025, .975]).tolist()

    def summarize(pred, truth, catalog_size):
        m = item_metrics(pred, truth, catalog_size)
        return {k: float(v.double().mean()) for k, v in m.items()}, m['ndcg@5'].double().cpu().numpy()

    data = {s: PackedSplit(meta['dataset_path'], s, meta['config'], device='cpu') for s in SPLITS}
    for s, n in SPLITS.items():
        assert len(data[s]) == n

    # ---------------- D1 teacher ----------------
    tstate = read_state(Path(meta['teacher_path']).parent / 'best.pt')
    teacher = SASRecTeacher(tstate['spec']).cuda().eval()
    teacher.load_state_dict(tstate['model'])
    tdigest = tensor_digest(teacher.state_dict())
    with torch.inference_mode(), attention_backend(runtime['attention_backend']):
        for split in SPLITS:
            d = data[split]; preds = {'oneshot': [], 'rolling': []}; truths = []; first_ndcg = 0.0
            for begin in range(0, len(d), 1024):
                hist, truth = d.batch(torch.arange(begin, min(begin + 1024, len(d))), 'cuda')
                scores = teacher(hist).float()
                t0 = truth[:, 0] - 1
                rank = ((scores > scores.gather(1, t0[:, None])) |
                        ((scores == scores.gather(1, t0[:, None])) & (torch.arange(scores.shape[1], device='cuda')[None] < t0[:, None]))).sum(-1) + 1
                first_ndcg += float(torch.where(rank <= 5, 1 / torch.log2(rank.double() + 1), 0).sum())
                preds['oneshot'].append((scores.topk(5, dim=1).indices + 1).cpu())
                h, chosen = hist.clone(), []
                for j in range(5):
                    sc = teacher(h).float() if j else scores.clone()
                    if chosen:
                        sc.scatter_(1, torch.stack(chosen, 1) - 1, float('-inf'))
                    item = sc.argmax(1) + 1
                    chosen.append(item)
                    h = torch.cat((h[:, 1:], item[:, None]), 1)
                preds['rolling'].append(torch.stack(chosen, 1).cpu())
                truths.append(truth.cpu())
            truth = torch.cat(truths)
            res = {'first_target_ndcg@5_check': first_ndcg / len(d)}
            for variant, chunks in preds.items():
                pred = torch.cat(chunks)
                metrics, per = summarize(pred, truth, meta['config']['catalog_size'])
                np.savez_compressed(out / f'teacher_{variant}_{split}.npz', prediction=pred.numpy(), ndcg=per)
                res[variant] = metrics
                res[variant + '_distinct_items'] = int(pred.unique().numel())
            results['teacher'][split] = res
            print(json.dumps({'teacher': split, **res}), flush=True)
    assert tensor_digest(teacher.state_dict()) == tdigest
    del teacher, tstate
    torch.cuda.empty_cache()

    # ---------------- D2 CF steps ----------------
    def make_sampler(diffusion, k):
        grid = [diffusion.steps] if k == 1 else [1 + math.floor((diffusion.steps - 1) * i / (k - 1) + 0.5) for i in reversed(range(k))]

        def sample(initial, predictor):
            state = initial.float()
            for t, u in zip(grid, grid[1:] + [0]):
                clean = predictor(state, torch.full((len(state),), t, device=state.device, dtype=torch.long)).float()
                if u == 0:
                    state = clean
                else:
                    eps = (state - diffusion.root_alpha[t] * clean) / diffusion.root_noise[t]
                    state = diffusion.root_alpha[u] * clean + diffusion.root_noise[u] * eps
            return state
        return sample, grid

    with torch.inference_mode(), attention_backend(runtime['attention_backend']):
        for name, (cp, stored) in CHECKPOINTS.items():
            state = read_state(Path(cp))
            cfg = state['config']
            cat = Catalog(state['catalog'], 'cuda')
            model = Recommender(cfg).cuda().eval(); model.load_state_dict(state['model'])
            diffusion = Diffusion(cfg).cuda().eval(); diffusion.load_state_dict(state['diffusion'])
            before = tensor_digest(model.state_dict())
            original_sample = diffusion.sample
            results['cf'][name] = {'checkpoint': cp, 'checkpoint_sha256': sha256(Path(cp)), 'R': state['r_epoch'], 'T': state['t_updates']}
            for split in SPLITS:
                d = data[split]
                noise_path = args.source / 'cf_prepared' / f'{split}_noise.npy'
                nmeta = json.loads(noise_path.with_suffix('.json').read_text())
                assert hashlib.sha256(json.dumps(d.sample_ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == nmeta['sample_ids_sha256']
                noise = np.load(noise_path, mmap_mode='r')
                # stored formal 20-step predictions
                stored_pred = np.zeros((len(d), 5), np.int64)
                with gzip.open(Path(stored[split]) / 'predictions.jsonl.gz', 'rt') as f:
                    for i, line in enumerate(f):
                        r = json.loads(line); assert r['sample_id'] == d.sample_ids[i]; stored_pred[i] = r['prediction']
                truth_all = torch.cat([d.batch(torch.arange(b, min(b + 4096, len(d))), 'cpu')[1] for b in range(0, len(d), 4096)])
                m20, per20 = summarize(torch.from_numpy(stored_pred), truth_all, cfg['catalog_size'])
                row = {'stored_20': m20}
                variants = list(STEPS) + ([20] if (name, split) == REPRODUCE else [])
                for k in variants:
                    sampler, grid = make_sampler(diffusion, k)
                    diffusion.sample = sampler
                    preds, raw, chg = [], [], []
                    t_begin = time.perf_counter()
                    for begin in range(0, len(d), runtime['evaluation_batch']):
                        stop = min(begin + runtime['evaluation_batch'], len(d))
                        hist, truth = d.batch(torch.arange(begin, stop), 'cuda')
                        initial = torch.from_numpy(np.array(noise[begin:stop], copy=True)).cuda()
                        pred, diag = generate(model, diffusion, cat, hist, initial, runtime)
                        preds.append(pred.cpu()); raw.append(diag['raw_valid'].float().mean(0).cpu() * len(pred))
                        chg.append(diag['changed_by_level'].float().mean((0, 1)).cpu() * len(pred))
                    diffusion.sample = original_sample
                    pred = torch.cat(preds)
                    metrics, per = summarize(pred, truth_all, cfg['catalog_size'])
                    entry = {'grid': grid, 'metrics': metrics, 'distinct_items': int(pred.unique().numel()),
                             'raw_valid_by_position': (torch.stack(raw).sum(0) / len(d)).tolist(),
                             'changed_by_level': (torch.stack(chg).sum(0) / len(d)).tolist(),
                             'ndcg_minus_stored20': float((per - per20).mean()),
                             'ci95_vs_stored20': bootstrap(per - per20, d.users),
                             'list_agreement_with_stored20': float((pred.numpy() == stored_pred).all(1).mean()),
                             'seconds': time.perf_counter() - t_begin}
                    if k == 20:
                        assert entry['list_agreement_with_stored20'] == 1.0, 'environment does not reproduce stored predictions'
                    np.savez_compressed(out / f'cf_{name}_{split}_steps{k}.npz', prediction=pred.numpy(), ndcg=per)
                    row[f'steps_{k}'] = entry
                    print(json.dumps({'cf': name, 'split': split, 'steps': k, **metrics, 'diff': entry['ndcg_minus_stored20']}), flush=True)
                results['cf'][name][split] = row
            assert tensor_digest(model.state_dict()) == before
            del state, model, diffusion, cat
            torch.cuda.empty_cache()
    results['seconds'] = time.perf_counter() - started
    results['script_sha256'] = script_sha
    write_json(out / 'COMPLETE.json', results)
    print('@@DONE@@', flush=True)


if __name__ == '__main__':
    main()
