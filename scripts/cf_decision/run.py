"""Run the inference-only decision-rule check on first-wave L5O CF checkpoints.

Validation split only. Per checkpoint, in order:
  1. Rule A with the recorded noise, the recorded batching and the unchanged generate().
     On CUDA it must reproduce the recorded validation predictions exactly, or the run
     stops (gate).
  2. Consistency: rules.LandingScorer against losses.landing_loss on the slot-1 vectors.
  3. B (C with K = 1), C for each larger K, the reference top five (from the largest K,
     not session-wise), and D (minimum Bayes risk over the lists of rule A).
Noise set 0 is the recorded table. Set k >= 1 uses the same generator with the noise
version suffixed by "+decision_rules_kXX". Every finished rule writes its predictions
and a summary, and is skipped on a rerun.

Screening runs (scripts/cf_screen): --run-dir/--prepared evaluate one run's own
validation checkpoints; --guidance adds guided variants (rule names suffixed _wX)
and --history-head adds the encoder-only HEAD diagnostic.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.kuairec_data.l5o_metrics import load_predictions, slate_metrics
from scripts.rqcd_mixer.catalog import Catalog, evaluation_noise, generate
from scripts.rqcd_mixer.common import (attention_backend, configure_runtime, preserve_rng, read_state,
                                       sha256, write_json)
from scripts.rqcd_mixer.data import PackedSplit
from scripts.rqcd_mixer.model import Diffusion, Recommender
from scripts.rqcd_mixer.screen import generation_dim, switches

from .rules import (LandingScorer, first_slot_vectors, generate_lists, generate_mode, history_head_top_five,
                    landing_consistency, mbr_select)

RUNS = 'cf_repeat_v2/runs'
PREPARED = 'parents/cf'
EXTRA = '+decision_rules_k{:02d}'


def log(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def noise_path(a, run, cfg, split):
    """The table the run was validated with: prepared, or run-local when the generation width differs."""
    if generation_dim(cfg) != cfg['rq']['latent_dim']:
        return run/'noise'/f'{split}_noise.npy'
    return (a.prepared or a.wave/PREPARED)/f'{split}_noise.npy'


def recorded_noise(path, cfg, data, split):
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    ids = hashlib.sha256(json.dumps(data.sample_ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if (meta['split'] != split or meta['sha256'] != sha256(path) or meta['sample_ids_sha256'] != ids or
            meta['seed'] != cfg['evaluation']['seed'] or meta['noise_version'] != cfg['evaluation']['noise_version']):
        raise ValueError('Recorded evaluation noise identity/order/hash mismatch')
    return np.load(path, mmap_mode='r', allow_pickle=False), meta['sha256']


def extra_noise(cfg, sample_ids, split, sets, cache):
    """Noise sets 1..sets-1 for the given rows, [sets-1, rows, list, dim], cached by identity."""
    identity = [cfg['evaluation']['noise_version'], cfg['evaluation']['seed'], split, sets - 1,
                hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False).encode()).hexdigest()]
    if generation_dim(cfg) != cfg['rq']['latent_dim']:
        identity.append(generation_dim(cfg))  # older (latent-width) cache names stay valid
    path = cache/f"{split}_{hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]}.npy"
    if not path.is_file():
        out = np.empty((sets - 1, len(sample_ids), cfg['list_length'], generation_dim(cfg)), dtype=np.float32)
        for k in range(1, sets):
            shifted = {**cfg, 'evaluation': {**cfg['evaluation'],
                                             'noise_version': cfg['evaluation']['noise_version'] + EXTRA.format(k)}}
            out[k - 1] = evaluation_noise(shifted, sample_ids, split)
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.partial.npy')
        np.save(tmp, out)
        tmp.replace(path)
        write_json(path.with_suffix('.json'), {'identity': identity, 'suffix': EXTRA, 'sha256': sha256(path)})
    return np.load(path, mmap_mode='r', allow_pickle=False), sha256(path)


def noise_sets(official, extra, rows, sets):
    """[rows, sets, list, dim]; set 0 is the recorded noise. Rows are split rows 0..n-1."""
    parts = [np.asarray(official[rows])[:, None]]
    if sets > 1:
        parts.append(np.asarray(extra[:sets - 1, rows]).transpose(1, 0, 2, 3))
    return torch.from_numpy(np.ascontiguousarray(np.concatenate(parts, 1)))


def write_predictions(path, sample_ids, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for sample_id, row in zip(sample_ids, prediction.tolist()):
            stream.write(json.dumps({'sample_id': sample_id, 'prediction': row}, ensure_ascii=False) + '\n')
    tmp.replace(path)


def score(prediction, target, ratios):
    flat = prediction.reshape(-1)
    return {'metrics': slate_metrics(prediction, target, ratios, 1.0),
            'sensitivity_ev2': slate_metrics(prediction, target, ratios, 2.0),
            'distinct_items': int(len(np.unique(flat[flat > 0]))),
            'valid_unique_lists': bool(all(len(set(r)) == len(r) and min(r) > 0 for r in prediction.tolist()))}


class Node:
    """One first-wave checkpoint; every rule's output goes to OUTPUT/arm/Rxx/rule/."""

    def __init__(self, a, arm, epoch, device, runtime):
        self.a, self.arm, self.epoch, self.device = a, arm, epoch, device
        # The NGC image enables TF32 by default; the recorded runtime turns it off. It must be
        # applied before any GPU work, or the catalog's own SID check fails (2026-11-18 first try).
        if device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32 or
                                      torch.get_float32_matmul_precision() != 'highest'):
            raise RuntimeError('Configure the recorded runtime (no TF32) before loading a checkpoint')
        run = self.run = a.run_dir if a.run_dir is not None else a.wave/RUNS/arm
        [self.checkpoint] = sorted((run/'checkpoints').glob(f'R{epoch:02d}_T*.pt'))
        recorded = run/a.split/f'R{epoch:02d}'
        self.recorded = json.loads((recorded/'summary.json').read_text(encoding='utf-8'))
        self.recorded_predictions = recorded/'predictions.jsonl.gz'
        if sha256(self.checkpoint) != self.recorded['identity']['checkpoint_sha256']:
            raise ValueError(f'{self.checkpoint} differs from the checkpoint its record was made with')
        state = read_state(self.checkpoint)
        if state['format'] != 'rqcd_mixer_joint_v1':
            raise ValueError('Not a joint CF checkpoint')
        self.cfg, self.runtime = state['config'], dict(state['runtime'])
        if a.cpu_fp32:
            self.runtime['precision'] = 'fp32'
        if self.runtime != runtime:
            raise ValueError('Checkpoint runtime differs from the configured runtime')
        self.model = Recommender(self.cfg).to(device)
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.diffusion = Diffusion(self.cfg).to(device)
        self.diffusion.load_state_dict(state['diffusion'])
        self.catalog = Catalog(state['catalog'], device)
        if self.catalog.version_hash != state['catalog']['version_hash']:
            raise ValueError('Catalog state hash mismatch')
        self.scorer = LandingScorer(self.catalog, self.cfg)
        self.filter = self.cfg['evaluation'].get('filter_history', False)
        self.out = a.output/arm/f'R{epoch:02d}'

    def bind(self, shared):
        if self.cfg != shared['cfg']:
            raise ValueError('Checkpoints were trained with different configurations')
        self.data, self.rows = shared['data'], shared['rows']
        self.official, self.extra = shared['official'], shared['extra']
        self.sample_ids = self.data.sample_ids[:len(self.rows)]
        self.target = self.data.target[:len(self.rows)].cpu().numpy()
        self.ratios = self.data.watch_ratios[:len(self.rows)]

    def batches(self, size):
        for start in range(0, len(self.rows), size):
            rows = self.rows[start:start + size]
            history, _ = self.data.batch(torch.as_tensor(rows), self.catalog.device)
            yield rows, history

    def done(self, name):
        path = self.out/name/'summary.json'
        if not path.is_file():
            return None
        return (load_predictions(self.out/name/'predictions.jsonl.gz', self.sample_ids),
                json.loads(path.read_text(encoding='utf-8')))

    def finish(self, name, prediction, seconds, **extra):
        write_predictions(self.out/name/'predictions.jsonl.gz', self.sample_ids, prediction)
        summary = {'arm': self.arm, 'R': self.epoch, 'rule': name, 'samples': len(self.rows),
                   'checkpoint': str(self.checkpoint),
                   'checkpoint_sha256': self.recorded['identity']['checkpoint_sha256'],
                   'seconds': seconds, **score(prediction, self.target, self.ratios), **extra}
        if not summary['valid_unique_lists']:
            raise RuntimeError(f'{name} produced an invalid or repeated list')
        write_json(self.out/name/'summary.json', summary)
        log(event='rule_done', arm=self.arm, R=self.epoch, rule=name, seconds=round(seconds, 1),
            ev_ndcg5=summary['metrics']['ev_ndcg@5'], distinct=summary['distinct_items'])
        return prediction, summary

    def rule_a(self, guidance=1.0):
        """Recorded batching (evaluation_batch rows from row 0), recorded noise, unchanged generate()."""
        name = 'A' + suffix(guidance)
        cached = self.done(name)
        if cached:
            return cached
        start, out = time.perf_counter(), []
        with preserve_rng(), attention_backend(self.runtime['attention_backend']):
            for rows, history in self.batches(self.runtime['evaluation_batch']):
                noise = torch.from_numpy(np.ascontiguousarray(self.official[rows])).to(self.catalog.device)
                items, _ = generate(self.model, self.diffusion, self.catalog, history, noise, self.runtime,
                                    filter_history=self.filter, guidance=guidance)
                out.append(items.cpu().numpy())
        return self.finish(name, np.concatenate(out), time.perf_counter() - start, guidance=guidance)

    def rule_head(self, item_vectors):
        """Encoder-only top five of the history head (diagnostic, not a recommender)."""
        cached = self.done('HEAD')
        if cached:
            return cached
        head = switches(self.cfg)['history_head']
        start, out = time.perf_counter(), []
        with preserve_rng(), attention_backend(self.runtime['attention_backend']):
            for rows, history in self.batches(self.runtime['evaluation_batch']):
                out.append(history_head_top_five(self.model, self.catalog, history, item_vectors, head['temperature'],
                                                 self.runtime, filter_history=self.filter).cpu().numpy())
        return self.finish('HEAD', np.concatenate(out), time.perf_counter() - start, session_wise=False,
                           note='encoder-only history head; diagnostic only')

    def gate(self, prediction):
        recorded = load_predictions(self.recorded_predictions, self.data.sample_ids)[:len(self.rows)]
        same = (prediction == recorded).all(1)
        result = {'rows_identical': float(same.mean()), 'rows': int(len(same)),
                  'recorded_ev_ndcg@5_full_split': self.recorded['metrics']['ev_ndcg@5'],
                  'recorded_ev_ndcg@5_rows': slate_metrics(recorded, self.target, self.ratios, 1.0)['ev_ndcg@5'],
                  'reproduced_ev_ndcg@5_rows': slate_metrics(prediction, self.target, self.ratios, 1.0)['ev_ndcg@5']}
        write_json(self.out/'GATE.json', result)
        log(event='gate', arm=self.arm, R=self.epoch, **result)
        strict = self.device.type == 'cuda' and not self.a.cpu_fp32
        if strict and not same.all():
            raise SystemExit(f'Reproduction gate failed for {self.arm} R{self.epoch}: {result}')
        return result

    def consistency(self):
        rows = self.rows[:256]
        history, target = self.data.batch(torch.as_tensor(rows), self.catalog.device)
        noise = torch.from_numpy(np.ascontiguousarray(self.official[rows][:, 0])).to(self.catalog.device)
        with preserve_rng(), attention_backend(self.runtime['attention_backend']):
            z = first_slot_vectors(self.model, self.diffusion, self.catalog, history, noise, self.runtime)
        ours, reference = landing_consistency(self.scorer, z, target[:, 0], self.cfg)
        result = {'scorer_nll': ours, 'landing_loss': reference, 'rows': int(len(rows))}
        write_json(self.out/'CONSISTENCY.json', result)
        log(event='consistency', arm=self.arm, R=self.epoch, **result)
        if not abs(ours - reference) <= 1e-3 + 1e-3 * abs(reference):
            raise SystemExit(f'Landing scorer disagrees with landing_loss: {result}')
        return result

    def rule_mode(self, k, reference=None, guidance=1.0):
        name = ('B' if k == 1 else f'C{k}') + suffix(guidance)
        reference = None if reference is None else reference + suffix(guidance)
        cached = self.done(name)
        if cached and (reference is None or self.done(reference)):
            return cached
        start, out, top = time.perf_counter(), [], []
        with preserve_rng(), attention_backend(self.runtime['attention_backend']):
            for rows, history in self.batches(self.a.users_per_batch):
                noise = noise_sets(self.official, self.extra, rows, k).to(self.catalog.device)
                items, five = generate_mode(self.model, self.diffusion, self.catalog, self.scorer, history, noise,
                                            self.runtime, filter_history=self.filter, reference=reference is not None,
                                            guidance=guidance)
                out.append(items.cpu().numpy())
                if five is not None:
                    top.append(five.cpu().numpy())
        seconds = time.perf_counter() - start
        result = self.finish(name, np.concatenate(out), seconds, noise_sets=k, guidance=guidance)
        if reference is not None:
            self.finish(reference, np.concatenate(top), seconds, noise_sets=k, session_wise=False, guidance=guidance,
                        note='top five of the slot-1 mean landing distribution; diagnostic only')
        return result

    def rule_lists(self, sets, rule_a):
        name = f'D{sets}'
        cached = self.done(name)
        if cached:
            return cached
        start, out = time.perf_counter(), []
        with preserve_rng(), attention_backend(self.runtime['attention_backend']):
            for rows, history in self.batches(self.a.users_per_batch):
                noise = noise_sets(self.official, self.extra, rows, sets)[:, 1:].to(self.catalog.device)
                out.append(generate_lists(self.model, self.diffusion, self.catalog, history, noise, self.runtime,
                                          filter_history=self.filter).cpu().numpy())
        lists = np.concatenate((rule_a[:, None], np.concatenate(out)), 1)
        chosen = mbr_select(lists)
        np.save(self.out/f'{name}_lists.npy', lists)
        return self.finish(name, lists[np.arange(len(lists)), chosen], time.perf_counter() - start,
                           noise_sets=sets, chosen_index_counts=np.bincount(chosen, minlength=sets).tolist())


def suffix(guidance):
    return '' if guidance == 1.0 else f'_w{guidance:g}'


def configure(a, device):
    """Apply the recorded runtime before anything touches the GPU, as run.py's _runtime() does."""
    run = a.run_dir if a.run_dir is not None else a.wave/RUNS/a.arms[0]
    runtime = json.loads((run/'runtime.json').read_text(encoding='utf-8'))
    if a.cpu_fp32:
        runtime['precision'] = 'fp32'
    environment = configure_runtime(runtime, device)
    # Deterministic kernels stay selected (rule A is unchanged); an op without one warns
    # instead of stopping the job. Only rule A is held to bit-exact reproduction.
    torch.use_deterministic_algorithms(runtime['deterministic'], warn_only=True)
    return runtime, environment


def setup(a, node, sets, environment):
    data_device = a.device if node.runtime['data_device'] == 'cuda' else 'cpu'
    data = PackedSplit(a.dataset, a.split, node.cfg, device=data_device)
    rows = np.arange(len(data) if a.limit is None else min(a.limit, len(data)))
    official, official_sha = recorded_noise(noise_path(a, node.run, node.cfg, a.split), node.cfg, data, a.split)
    extra, extra_sha = extra_noise(node.cfg, data.sample_ids[:len(rows)], a.split, sets, a.output/'noise')
    write_json(a.output/'RUN.json', {'environment': environment, 'args': {k: str(v) for k, v in vars(a).items()},
                                     'official_noise_sha256': official_sha, 'extra_noise_sha256': extra_sha,
                                     'extra_noise_suffix': EXTRA, 'code_commit': os.environ.get('CODE_COMMIT')})
    return {'cfg': node.cfg, 'data': data, 'rows': rows, 'official': official, 'extra': extra}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', type=Path, help='first-wave L5O root (l5o_wave_20261117_v1)')
    p.add_argument('--run-dir', type=Path, help='one screening run directory instead of first-wave arms')
    p.add_argument('--prepared', type=Path, help='prepared parent of --run-dir (noise, item vectors)')
    p.add_argument('--guidance', nargs='+', type=float, default=[1.0], help='guidance weights; 1 = none')
    p.add_argument('--history-head', action='store_true', help='also score the encoder-only history head')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['CF-M0', 'CF-M1'])
    p.add_argument('--epochs', nargs='+', type=int, default=[10, 19, 30, 35])
    p.add_argument('--split', default='validation', choices=['validation'])
    p.add_argument('--ks', nargs='+', type=int, default=[1, 4, 16], help='K for rule B (K=1) and rule C')
    p.add_argument('--lists', type=int, default=16, help='lists for rule D; list 0 is rule A; 0 skips rule D')
    p.add_argument('--users-per-batch', type=int, default=64)
    p.add_argument('--limit', type=int, help='first N validation rows only (self-test)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-fp32', action='store_true', help='self-test on CPU in FP32 instead of the recorded BF16')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.limit is not None and not a.output.name.startswith('selftest'):
        raise ValueError('A --limit run must write to a directory named selftest*')
    if min(a.ks) < 1 or a.lists == 1 or a.lists < 0:
        raise ValueError('K must be >= 1; rule D needs at least two lists (0 skips it)')
    if (a.run_dir is None) == (a.wave is None) or (a.run_dir is not None and a.prepared is None):
        raise ValueError('Give either --wave, or --run-dir with --prepared')
    if 1.0 not in a.guidance:
        raise ValueError('Guidance 1 (no guidance) is the reference and must be evaluated')
    device = torch.device(a.device)
    sets = max(max(a.ks), a.lists)
    a.output.mkdir(parents=True, exist_ok=True)
    started, results, shared = time.perf_counter(), [], None
    runtime, environment = configure(a, device)
    item_vectors = None
    if a.history_head:
        item_vectors = read_state(a.prepared/'target_inputs.pt')['features'].float().to(device)
    for arm in ([a.run_dir.name] if a.run_dir is not None else a.arms):
        for epoch in a.epochs:
            node = Node(a, arm, epoch, device, runtime)
            shared = shared or setup(a, node, sets, environment)
            node.bind(shared)
            log(event='node_start', arm=arm, R=epoch, checkpoint=str(node.checkpoint))
            prediction_a, _ = node.rule_a()
            gate = node.gate(prediction_a)
            consistency = node.consistency()
            for k in sorted(a.ks):
                node.rule_mode(k, reference=f'REF{k}' if k == max(a.ks) else None)
            if a.lists:
                node.rule_lists(a.lists, prediction_a)
            for guidance in sorted(set(a.guidance) - {1.0}):
                node.rule_a(guidance)
                for k in sorted(a.ks):
                    node.rule_mode(k, guidance=guidance)
            if a.history_head:
                node.rule_head(item_vectors)
            results.append({'arm': arm, 'R': epoch, 'gate': gate, 'consistency': consistency})
            del node
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_json(a.output/'COMPLETE.json', {'nodes': results, 'seconds': time.perf_counter() - started})
    log(event='complete', seconds=round(time.perf_counter() - started, 1))


if __name__ == '__main__':
    main()
