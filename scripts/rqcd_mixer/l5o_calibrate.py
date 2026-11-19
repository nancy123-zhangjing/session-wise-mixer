"""Calibrate the existing SASRec teacher on the verified KuaiRec L5O split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.kuairec_data.l5o_metrics import slate_metrics
from scripts.listwise_diffgrm.data import KuaiRecListDataset
from scripts.listwise_diffgrm.execution import TrainingTensorCache
from .common import atomic_save, read_state, output_lock, write_json, seed_all, seed_for, append_log, sha256
from .data import dataset_binding, catalog_original_ids
from .teacher import TeacherTrainer
from .targets import teacher_spec, teacher_sources


class Split:
    def __init__(self, root, name, device, expected_count):
        data = KuaiRecListDataset(root, name)
        if data.config.get('protocol_id') != 'kuairec_l5o_records_v1' or len(data) != expected_count:
            raise ValueError('Expected verified record-preserving L5O data')
        cache = TrainingTensorCache(data, vectorized=True, device=device)
        self.history = cache.tensors['history_item_ids']
        self.target = cache.tensors['target_item_ids']
        if name != 'train':
            self.ratios = np.asarray(pq.read_table(root / f'{name}.parquet',
                columns=['target_watch_ratio'])[0].to_pylist())

    def __len__(self):
        return len(self.history)

    def batch(self, ids, device):
        ids = ids.to(self.history.device)
        return tuple(x[ids].to(device=device, dtype=torch.long) for x in (self.history, self.target))


@torch.inference_mode()
def evaluate(model, split, device, batch_size, filters, thresholds=(1.0,)):
    model.eval()
    predictions = {f: [] for f in filters}
    for h in split.history.split(batch_size):
        h = h.to(device=device, dtype=torch.long)
        scores = model(h).float()
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite SASRec scores')
        for f in filters:
            # PAD has no score column. Repeated indices all receive the same mask.
            s = scores.clone() if f else scores
            if f:
                s.scatter_(1, (h - 1).clamp_min(0), -torch.inf)
                s[:, 0] = torch.where((h == 1).any(1), -torch.inf, scores[:, 0])
            predictions[f].append((s.argsort(dim=1, descending=True, stable=True)[:, :5] + 1).cpu().numpy())
    y = split.target.cpu().numpy()
    return {str(f): {str(t): slate_metrics(np.concatenate(predictions[f]), y, split.ratios, t)
                     for t in thresholds} for f in filters}


def passes(metrics):
    hit, ndcg = metrics['imp_hit@5'], metrics['ev_ndcg@5']
    return hit is not None and ndcg is not None and .25 <= hit <= .45 and .05 <= ndcg <= .12


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output', 'config', 'runtime'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    from .run import _runtime
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    runtime, env = _runtime(args, cfg, require_batches=False)
    data_id = dataset_binding(args.dataset, verify=True)
    ids = torch.from_numpy(catalog_original_ids(args.dataset))
    v2 = 'protocol_v2' in cfg
    if v2:
        from scripts.kuairec_data.l5o_v2 import PROTOCOL, SEEDS
        if (cfg['protocol_v2'] != PROTOCOL or cfg['seed'] not in SEEDS
                or cfg['evaluation'] != {'filter_history': True, 'selection_rule': 'smoothed3'}):
            raise ValueError('Protocol v2 SASRec requires a v2 seed, fixed history filter and smoothed selection')
    allowed_seeds = SEEDS if v2 else (2024,)
    if (cfg['history_max'], cfg['list_length'], cfg['catalog_size']) != (128, 5, len(ids)) or cfg['seed'] not in allowed_seeds:
        raise ValueError('Calibration geometry or seed differs from the locked protocol')
    # First wave chose the filter on validation; protocol v2 fixes it for every method.
    filters = (True,) if v2 else (False, True)
    binding = dict(data_binding=data_id, environment=env, source=teacher_sources(), calibration=sha256(Path(__file__)))
    with output_lock(args.output, args.resume) as root:
        if (root/'COMPLETE.json').exists():
            raise ValueError('Calibration already completed; read that receipt')
        def split(name):
            return Split(args.dataset, name, args.device, cfg['split_counts'][name])
        train = split('train')
        val = split('validation')
        seed_all(seed_for(cfg['seed'], 'teacher_init'))
        trainer = TeacherTrainer(cfg, runtime, train, args.device, binding)
        if args.resume:
            trainer.restore(read_state(root/'last.pt'))
        while True:
            if trainer.epochs and trainer.epochs not in trainer.validated_epochs:
                metrics = evaluate(trainer.model, val, args.device, 256, filters)
                selected = max(filters, key=lambda f: metrics[str(f)]['1.0']['ev_ndcg@5'])
                score = metrics[str(selected)]['1.0']['ev_ndcg@5']
                improved = score > trainer.best_score
                trainer.bad_epochs = 0 if improved else trainer.bad_epochs + 1
                trainer.validated_epochs.append(trainer.epochs)
                if v2:
                    # Every validated epoch is kept (weights only) for the smoothed selection.
                    atomic_save(root/'checkpoints'/f'E{trainer.epochs:03d}.pt',
                                dict(model=trainer.model.state_dict(), epochs=trainer.epochs, validation_score=score))
                if improved:
                    trainer.best_score = score
                    atomic_save(root/'best.pt', dict(trainer.state(), filter_history=selected, validation=metrics))
                row = dict(kind='validation', epoch=trainer.epochs, updates=trainer.updates, metrics=metrics)
                append_log(root, row)
                atomic_save(root/'last.pt', trainer.state())
            if trainer.epochs >= cfg['teacher']['epochs'] or trainer.bad_epochs >= cfg['teacher']['patience']:
                break
            event = trainer.advance()
            if event['kind'] == 'teacher_epoch' or trainer.updates % runtime['checkpoint_every_updates'] == 0:
                atomic_save(root/'last.pt', trainer.state())
            if trainer.updates % 1000 == 0:
                print(json.dumps(event), flush=True)
        best = read_state(root/'best.pt')
        freeze = dict(best_sha256=sha256(root/'best.pt'), epoch=best['epochs'], filter_history=best['filter_history'],
                      validation=best['validation'], binding=binding)
        write_json(root/'SELECTION.json', freeze)
        trainer.model.load_state_dict(best['model'])
        test = evaluate(trainer.model, split('test'), args.device,
                        256, (best['filter_history'],), (1.0, 2.0))[str(best['filter_history'])]
        supported = torch.zeros(len(ids)+1, dtype=torch.bool, device=train.history.device)
        for items in (train.history, train.target):
            for block in items.split(4096):
                supported[block.long()] = True
        atomic_save(root/'items.pt', dict(format='rqcd_teacher_items_v1', spec=teacher_spec(cfg),
            data_binding=data_id, source=teacher_sources(), original_ids=ids, train_supported=supported[1:].cpu(),
            embedding=best['model']['items.weight'][1:], selected_epoch=best['epochs'],
            validation_ev_ndcg_at_5=best['best_score'], calibration_source=binding['calibration']))
        result = dict(passed=passes(test['1.0']), test=test, selection=freeze,
                      items_sha256=sha256(root/'items.pt'), completed_epochs=trainer.epochs)
        if v2:
            from scripts.kuairec_data.selection import smoothed_choice
            nodes = sorted(root.joinpath('checkpoints').glob('E*.pt'))
            points = [(path, read_state(path)['validation_score']) for path in nodes]
            if [read_state(path)['epochs'] for path in nodes] != sorted(trainer.validated_epochs):
                raise ValueError('Smoothed selection needs every validated epoch checkpoint')
            path, smoothed, raw, _ = smoothed_choice(points)
            node = read_state(path)
            trainer.model.load_state_dict(node['model'])
            result['smoothed'] = dict(epoch=node['epochs'], checkpoint_sha256=sha256(path),
                                      smoothed_validation=smoothed, raw_validation=raw,
                                      test=evaluate(trainer.model, split('test'), args.device, 256, (True,),
                                                    (1.0, 2.0))['True'])
        write_json(root/'COMPLETE.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
