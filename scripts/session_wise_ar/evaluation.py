"""Full list metrics, source-bound reusable predictions and encoding diagnostics."""
import gzip
import json
from pathlib import Path
import time
import torch
from scripts.rqcd_mixer.common import preserve_rng, write_json, sha256
from scripts.rqcd_mixer.evaluation import item_metrics, packed_l5o_metrics
from .data import user_buckets


@torch.inference_mode()
def evaluate(rec, catalog, data, cfg, output, identity):
    output = Path(output)
    summary = output/'summary.json'
    if summary.exists():
        result = json.loads(summary.read_text(encoding='utf-8'))
        if result['identity'] != identity or result['prediction_sha256'] != sha256(output/'predictions.jsonl.gz'):
            raise ValueError('Evaluation identity or prediction SHA mismatch')
        return result
    output.mkdir(parents=True, exist_ok=True)
    device = next(rec.parameters()).device
    batch_size = cfg['evaluation']['batch_size']
    users = user_buckets(data.users)
    was_training = rec.training
    rec.eval()
    started = time.perf_counter()
    total, hits = {}, torch.zeros(cfg['list_length'], device=device)
    temporary = output/'predictions.jsonl.gz.partial'
    try:
        with preserve_rng(), gzip.open(temporary, 'wt', encoding='utf-8') as stream:
            for start in range(0, len(data), batch_size):
                stop = min(start+batch_size, len(data))
                history, truth = data.batch(torch.arange(start, stop), device)
                pred = catalog.generate(rec, history, users[start:stop].to(device),
                                        cfg['list_length'], cfg['evaluation']['beam'],
                                        filter_history=cfg['evaluation'].get('filter_history', False))
                metrics = item_metrics(pred, truth, cfg['catalog_size'])
                for key, values in metrics.items():
                    total[key] = total.get(key, torch.zeros((), device=device))+values.double().sum()
                hits += (pred[:, :, None] == truth[:, None, :]).any(-1).sum(0)
                rows = zip(data.sample_ids[start:stop], pred.cpu().tolist(), truth.cpu().tolist())
                stream.write(''.join(json.dumps({'sample_id': s, 'prediction': p, 'target': t})+'\n'
                                     for s, p, t in rows))
        final = output/'predictions.jsonl.gz'
        temporary.replace(final)
        result = {'identity': identity, 'count': len(data), 'split': data.split,
                  'metrics': {k: (v/len(data)).item() for k, v in total.items()},
                  'position_hit_rate': (hits/len(data)).tolist(),
                  'sid_legal_rate': 1.0, 'legality_scope': 'catalog-constrained decoding, not unconstrained accuracy',
                  'seconds': time.perf_counter()-started, 'prediction_sha256': sha256(final)}
        if hasattr(data, 'watch_ratios'):
            result['legacy_metrics'] = result['metrics']
            result.update(packed_l5o_metrics(data, final))
        write_json(summary, result)
        return result
    finally:
        rec.train(was_training)
