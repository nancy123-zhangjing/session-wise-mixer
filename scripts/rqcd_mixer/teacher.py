"""Explicit SASRec-style teacher command. Importing or planning never trains."""
import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import (atomic_save, read_state, output_lock, write_json, seed_all, seed_for,
                     cpu_copy, rng_state, restore_rng, autocast, attention_backend, append_log)
from .data import PermutationStream, PackedSplit, dataset_binding, catalog_original_ids
from .targets import teacher_spec, teacher_sources


class TeacherBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, heads, drop = cfg['width'], cfg['heads'], cfg['dropout']
        self.heads, self.dropout = heads, drop
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, cfg['ffn']), nn.ReLU(), nn.Dropout(drop), nn.Linear(cfg['ffn'], d))

    def forward(self, x, legal, valid):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=legal[:, None],
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.out(y.transpose(1, 2).reshape(b, n, d)), self.dropout, self.training)
        x = x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)
        return x * valid[..., None]


class SASRecTeacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        t = cfg['teacher']
        self.width, self.dropout = t['width'], t['dropout']
        self.items = nn.Embedding(cfg['catalog_size']+1, self.width, padding_idx=0)
        self.positions = nn.Embedding(cfg['history_max'], self.width)
        self.blocks = nn.ModuleList([TeacherBlock(t) for _ in range(t['layers'])])
        self.norm = nn.LayerNorm(self.width)
        nn.init.normal_(self.items.weight, std=.02)
        nn.init.normal_(self.positions.weight, std=.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def forward(self, history):
        n = history.shape[1]
        valid = history != 0
        x = self.items(history)*math.sqrt(self.width) + self.positions.weight[:n]
        x = F.dropout(x, self.dropout, self.training) * valid[..., None]
        causal = torch.ones(n, n, dtype=torch.bool, device=history.device).tril()
        legal = causal[None] & valid[:, None, :]
        # Left PAD queries may otherwise have no keys. Give only those queries
        # their own dummy key, then zero their output after every block.
        legal = legal | ((~valid)[:, :, None] & torch.eye(n, dtype=torch.bool, device=history.device))
        for block in self.blocks:
            x = block(x, legal, valid)
        return self.norm(x[:, -1]) @ self.items.weight[1:].T


def next_item_batch(data, indices, cfg, device):
    """View j sees history and true targets <j, never target j or later."""
    length, h = cfg['list_length'], cfg['history_max']
    rows, j = indices//length, indices % length
    history, target = data.batch(rows, device)
    j = j.to(device)
    sequence = torch.cat((history, target), 1)
    positions = j[:, None]+torch.arange(h, device=device)
    return sequence.gather(1, positions), target.gather(1, j[:, None]).squeeze(1)


class TeacherTrainer:
    def __init__(self, cfg, runtime, data, device, binding):
        self.cfg, self.runtime, self.data, self.device, self.binding = cfg, runtime, data, torch.device(device), binding
        self.model = SASRecTeacher(cfg).to(device)
        t = cfg['teacher']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'],
                                          betas=tuple(cfg['training']['betas']), eps=cfg['training']['eps'],
                                          fused=runtime['fused_adamw'] and self.device.type == 'cuda')
        self.stream = PermutationStream(len(data)*cfg['list_length'], seed_for(cfg['seed'], 'teacher_order'))
        self.epochs = self.updates = self.bad_epochs = 0
        self.validated_epochs, self.best_score = [], -1.0

    def advance(self):
        ids = self.stream.take(self.cfg['teacher']['batch_size'])
        history, labels = next_item_batch(self.data, ids, self.cfg, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with attention_backend(self.runtime['attention_backend']), autocast(self.runtime, self.device):
            logits = self.model(history)
        loss = F.cross_entropy(logits.float(), labels-1)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['max_grad_norm'], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.stream.at_end:
            self.epochs += 1
        return {'kind': 'teacher_epoch' if self.stream.at_end else 'teacher_update',
                'epoch': self.epochs, 'updates': self.updates, 'batch_ce': float(loss.detach())}

    @torch.no_grad()
    def evaluate(self, validation):
        self.model.eval()
        score = torch.zeros((), device=self.device, dtype=torch.float64)
        for start in range(0, len(validation), self.cfg['teacher']['batch_size']):
            ids = torch.arange(start, min(start+self.cfg['teacher']['batch_size'], len(validation)))
            history, targets = validation.batch(ids, self.device)
            with attention_backend(self.runtime['attention_backend']), autocast(self.runtime, self.device):
                logits = self.model(history).float()
            # Full-catalog rank with a deterministic smaller-item-ID tie break.
            truth = targets[:, 0]-1
            true_score = logits.gather(1, truth[:, None])
            rank = ((logits > true_score) | ((logits == true_score) &
                     (torch.arange(logits.shape[1], device=self.device)[None] < truth[:, None]))).sum(-1)+1
            score += torch.where(rank <= 5, 1/torch.log2(rank.double()+1), 0).sum()
        return float(score/len(validation))

    def state(self):
        return cpu_copy({'format': 'rqcd_teacher_checkpoint_v1', 'spec': teacher_spec(self.cfg), 'runtime': self.runtime,
                         'binding': self.binding, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                         'stream': self.stream.state(), 'epochs': self.epochs, 'updates': self.updates,
                         'bad_epochs': self.bad_epochs, 'validated_epochs': self.validated_epochs,
                         'best_score': self.best_score, 'rng': rng_state()})

    def restore(self, state):
        if (state['format'] != 'rqcd_teacher_checkpoint_v1' or state['spec'] != teacher_spec(self.cfg) or
                state['runtime'] != self.runtime or state['binding'] != self.binding):
            raise ValueError('Teacher resume identity mismatch')
        s = state['stream']; steps = math.ceil(self.stream.size/self.cfg['teacher']['batch_size'])
        updates = s['epoch']*steps+math.ceil(s['cursor']/self.cfg['teacher']['batch_size'])
        if updates != state['updates'] or s['epoch']+int(s['cursor']==self.stream.size) != state['epochs']:
            raise ValueError('Teacher data/update clocks disagree')
        optim_steps = [int(v['step']) for v in state['optimizer']['state'].values()]
        if updates and (not optim_steps or any(v != updates for v in optim_steps)):
            raise ValueError('Teacher optimizer clock mismatch')
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.stream.restore(s)
        for key in ('epochs', 'updates', 'bad_epochs', 'validated_epochs', 'best_score'):
            setattr(self, key, state[key])
        restore_rng(state['rng'])


def train_teacher(args, cfg):
    from .run import _runtime
    runtime, environment = _runtime(args, cfg, require_batches=False)
    data_id = dataset_binding(Path(args.dataset), verify=True)
    originals = torch.from_numpy(catalog_original_ids(Path(args.dataset)))
    if (len(originals) != cfg['catalog_size'] or data_id['history_max'] != cfg['history_max'] or
            data_id['list_length'] != cfg['list_length']):
        raise ValueError('Teacher dataset protocol mismatch')
    place = args.device if runtime['data_device']=='cuda' else 'cpu'
    train = PackedSplit(args.dataset, 'train', cfg, device=place)
    validation = PackedSplit(args.dataset, 'validation', cfg, device=place)
    binding = {'data_binding': data_id, 'source': teacher_sources(), 'environment': environment}
    seed_all(seed_for(cfg['seed'], 'teacher_init'))
    trainer = TeacherTrainer(cfg, runtime, train, args.device, binding)
    with output_lock(args.output, args.resume) as root:
        if args.resume:
            trainer.restore(read_state(root/'last.pt'))
        while True:
            if trainer.epochs and trainer.epochs not in trainer.validated_epochs:
                score = trainer.evaluate(validation)
                improved = score > trainer.best_score
                trainer.bad_epochs = 0 if improved else trainer.bad_epochs+1
                if improved:
                    trainer.best_score = score
                trainer.validated_epochs.append(trainer.epochs)
                if improved:
                    atomic_save(root/'best.pt', trainer.state())
                append_log(root, {'kind': 'teacher_validation', 'epoch': trainer.epochs, 'first_item_ndcg@5': score})
                atomic_save(root/'last.pt', trainer.state())
            if trainer.epochs >= cfg['teacher']['epochs'] or trainer.bad_epochs >= cfg['teacher']['patience']:
                break
            event = trainer.advance()
            if event['kind']=='teacher_epoch' or trainer.updates % runtime['checkpoint_every_updates']==0:
                atomic_save(root/'last.pt', trainer.state())
            if event['kind']=='teacher_epoch':
                append_log(root, event)
        best = read_state(root/'best.pt')
        support = torch.zeros(cfg['catalog_size']+1, dtype=torch.bool, device=train.history.device)
        support[train.history.flatten()] = True; support[train.target.flatten()] = True
        asset = {'format': 'rqcd_teacher_items_v1', 'spec': teacher_spec(cfg), 'data_binding': data_id,
                 'source': teacher_sources(), 'original_ids': originals, 'train_supported': support[1:],
                 'embedding': best['model']['items.weight'][1:], 'selected_epoch': best['epochs'],
                 'validation_first_item_ndcg@5': best['best_score']}
        atomic_save(root/'items.pt', asset)
        from .common import sha256
        result = {'selected_epoch': best['epochs'], 'completed_epochs': trainer.epochs,
                  'train_unsupported_items': int((~support[1:]).sum()), 'items_sha256': sha256(root/'items.pt'),
                  'validation_first_item_ndcg@5': best['best_score'], 'test_used': False}
        write_json(root/'COMPLETE.json', result)
    return result


def main():
    import json
    from .config import TARGET_CONFIGS, RUNTIME_TEMPLATE, load_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=TARGET_CONFIGS['cf'])
    p.add_argument('--runtime', type=Path, default=RUNTIME_TEMPLATE)
    p.add_argument('--device', default='cuda'); p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg = load_config(args.config)
    if 'teacher' not in cfg:
        p.error('Teacher requires a target-v2 protocol')
    print(json.dumps(train_teacher(args, cfg), indent=2))


if __name__ == '__main__':
    main()
