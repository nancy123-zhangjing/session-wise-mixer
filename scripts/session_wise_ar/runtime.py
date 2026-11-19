"""Resumable explicit stages. Importing this module never starts an experiment."""
from __future__ import annotations

import copy
import json
import math
import importlib.metadata
from contextlib import contextmanager
from pathlib import Path
import time
import torch
from transformers import Adafactor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from scripts.rqcd_mixer.common import (atomic_save, write_json, read_state, sha256, digest,
                                      seed_all, seed_for, rng_state, restore_rng, preserve_rng, output_lock)
from scripts.rqcd_mixer.data import PermutationStream, bind_inputs, load_content
from scripts.rqcd.data import PackedSplit
from scripts.listwise_diffgrm.assets import catalog_original_ids
from .config import ROOT
from .model import ARRecommender
from .data import SessionTasks
from .tokenizer import make_tokenizer, joint_loss
from .catalog import Catalog, add_suffix, encoding_stats
from .evaluation import evaluate
from .upstream import verify


def family_identity(cfg):
    return {k: v for k, v in cfg.items() if k not in ('arm', 'mixer_enabled')}


def source_identity():
    verify()
    roots = ['scripts/session_wise_ar', 'configs/session_wise_ar']
    paths = [p for folder in roots for p in (ROOT/folder).rglob('*') if p.suffix in ('.py', '.json', '.txt')]
    paths += [ROOT/p for p in ('scripts/rqcd_mixer/tokenizer.py', 'scripts/rqcd_mixer/common.py',
               'scripts/rqcd_mixer/data.py', 'scripts/rqcd_mixer/evaluation.py', 'scripts/rqcd/data.py',
               'scripts/listwise_diffgrm/assets.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def configure_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable')
    # AR initial reference is FP32; do not silently substitute TF32/BF16.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return device


def training_precision(cfg):
    value = cfg.get('execution', {}).get('precision', 'fp32')
    if value not in ('fp32', 'tf32', 'bf16'):
        raise ValueError('execution.precision must be fp32, tf32 or bf16')
    return value


def train_autocast(cfg, device):
    """Opt-in BF16 autocast for the recommendation training forward/loss only.

    Master weights, optimizer states, SID refresh and every evaluation stay FP32.
    FP32 configs (all historical presets) run exactly the original kernels."""
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                          enabled=training_precision(cfg) == 'bf16')


@contextmanager
def train_kernels(cfg, device):
    """Kernel policy for one recommendation TRAINING update only.

    precision=tf32 enables TF32 matmuls for the update and restores the strict
    FP32 flags afterwards, so SID refresh, validation and test stay exact FP32.
    model.sdpa_backend=efficient pins the fused memory-efficient attention kernel
    on CUDA (an unsupported shape raises instead of silently falling back)."""
    cuda = torch.device(device).type == 'cuda'
    tf32 = training_precision(cfg) == 'tf32' and cuda
    backend = cfg.get('model', {}).get('sdpa_backend', 'auto')
    if backend not in ('auto', 'efficient'):
        raise ValueError('model.sdpa_backend must be auto or efficient')
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32 or previous[0]
    try:
        if cuda and backend == 'efficient':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                yield
        else:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


class DeviceLossMeter:
    """Per-update loss sums kept on the device; one host copy per checkpoint/validation.

    Each running sum starts from the stored float64 value and adds the per-update
    terms in the original order, so flushed values equal the former per-step
    Python accumulation bit for bit (also across resume)."""
    def __init__(self):
        self.sums, self.norms = {}, None

    def add(self, progress, parts, count, norms):
        for key, value in parts.items():
            term = value.detach().double()*count
            if key not in self.sums:
                self.sums[key] = torch.tensor(float(progress['loss_sum'].get(key, 0)),
                                              dtype=torch.float64, device=term.device)
            self.sums[key] = self.sums[key]+term
        self.norms = norms

    def flush(self, progress):
        for key, value in self.sums.items():
            progress['loss_sum'][key] = value.item()
        if self.norms is not None:
            progress['last_gradient_norms'] = self.norms.cpu().tolist()
        self.sums, self.norms = {}, None


def environment(device):
    result = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'scikit-learn')}
    result.update(device_type=torch.device(device).type, cuda=torch.version.cuda)
    if torch.device(device).type == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    return result


def independent_output(output, *inputs):
    destination = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError('Output must be independent of data, parents and other run stages')


def copy_base(dst, src):
    if src.cfg['family'] == 'tiger':
        dst.encoder.load_state_dict(src.encoder.state_dict())
        dst.decoder.load_state_dict(src.decoder.state_dict())
        with torch.no_grad(): dst.books.copy_(src.books)
    else:
        dst.native.encoder.load_state_dict(src.native.encoder.state_dict())
        dst.native.decoder.load_state_dict(src.native.decoder.state_dict())
        with torch.no_grad():
            for target, source in zip(dst.original_weights(), src.original_weights()): target.copy_(source)
    dst.initialized.fill_(True)


def prepare(cfg, dataset, content_path, output, device):
    independent_output(output, dataset, content_path)
    device = configure_device(device)
    binding = bind_inputs(dataset, content_path, cfg)
    with output_lock(output):
        output = Path(output)
        content = load_content(content_path, cfg, device)
        seed_all(seed_for(cfg['seed'], cfg['family'], 'rq-init'))
        plain_cfg = dict(cfg, mixer_enabled=False)
        base = make_tokenizer(plain_cfg).to(device)
        stream = PermutationStream(len(content), seed_for(cfg['seed'], cfg['family'], 'pretrain-data'))
        first = stream.take(cfg['rq']['batch_size']).to(device)
        base.initialize(content[first])
        plus = make_tokenizer(dict(cfg, mixer_enabled=True)).to(device)
        copy_base(plus, base)
        seed_all(seed_for(cfg['seed'], cfg['family'], 'recommender-init'))
        rec = ARRecommender(cfg).to(device)
        if cfg['family'] == 'diger':
            with torch.no_grad():
                rec.native.semantic_embedding.weight[0].zero_()
                rec.native.semantic_embedding.weight[1:].copy_(content)
        record = {'family_config': family_identity(cfg), 'binding': binding, 'environment': environment(device),
                  'sources': source_identity(), 'original_ids': list(map(int, catalog_original_ids(Path(dataset)))),
                  'tokenizers': {'plain': base.state_dict(), 'intra': plus.state_dict()},
                  'recommender': rec.state_dict()}
        atomic_save(output/'initial.pt', record)
        write_json(output/'manifest.json', {'family_config': family_identity(cfg), 'binding': binding,
                   'initial_sha256': sha256(output/'initial.pt'), 'sources': record['sources']})


def load_parent(cfg, prepared):
    prepared = Path(prepared)
    meta = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if meta['family_config'] != family_identity(cfg) or meta['sources'] != source_identity():
        raise ValueError('Prepared parent belongs to different configuration/source')
    if meta['initial_sha256'] != sha256(prepared/'initial.pt'):
        raise ValueError('Prepared parent checksum mismatch')
    return read_state(prepared/'initial.pt'), meta


def ensure_inputs(cfg, meta, dataset, content_path):
    if bind_inputs(dataset, content_path, cfg) != meta['binding']:
        raise ValueError('Dataset/content differs from prepared parent')


def pretrain_optimizer(model, cfg, steps_per_epoch):
    c = cfg['rq']
    # Default (absent field) keeps the original paper-configured Adagrad exactly.
    if cfg['family'] == 'tiger' and c.get('optimizer', 'adagrad') == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=c['lr'], lr_decay=0, weight_decay=0,
                                    initial_accumulator_value=0, eps=1e-10), None
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    schedule = get_linear_schedule_with_warmup(optimizer, c['warmup_epochs']*steps_per_epoch,
                                               c['epochs']*steps_per_epoch)
    return optimizer, schedule


def recommendation_optimizers(rec, tokenizer, cfg, updates_per_epoch):
    c = cfg['training']
    if cfg['family'] == 'tiger':
        optimizer = Adafactor(rec.parameters(), lr=c['r_lr'], eps=(1e-30, 1e-3), clip_threshold=1.0,
                              decay_rate=-0.8, beta1=None, weight_decay=0.0,
                              scale_parameter=True, relative_step=False, warmup_init=False)
        # Historical presets hard-coded 10000 (their warmup_updates is 10000 too and
        # is ignored here); a protocol may set inverse_sqrt_warmup explicitly.
        warmup = c.get('inverse_sqrt_warmup', 10000)
        if type(warmup) is not int or warmup < 1:
            raise ValueError('TIGER inverse_sqrt_warmup must be a positive integer')
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(1.0, math.sqrt(warmup/max(1, n))))
        return [optimizer], [schedule]
    rec_params = [p for p in rec.parameters() if p.requires_grad]
    id_params = [p for p in tokenizer.parameters() if p.requires_grad]
    optimizers = [torch.optim.AdamW(params, lr=lr, weight_decay=c['weight_decay'], betas=(.9, .999), eps=1e-8)
                  for params, lr in ((rec_params, c['r_lr']), (id_params, c['id_lr']))]
    schedulers = [get_cosine_schedule_with_warmup(o, c['warmup_updates'], c['max_epochs']*updates_per_epoch)
                  for o in optimizers]
    return optimizers, schedulers


def finite_step(loss, modules, optimizers, schedulers, clip=None):
    """One update; loss and gradient finiteness share a single host synchronization.

    A nonfinite loss or gradient norm still rejects the update before any optimizer
    step. Finite updates are numerically identical to the former two-gate version."""
    for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norms = torch.stack([torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf') if clip is None else clip,
                                                        error_if_nonfinite=False).detach()
                         for model in modules])
    if not bool(torch.isfinite(loss.detach()) & torch.isfinite(norms).all()):
        raise FloatingPointError('Nonfinite loss or gradient; update rejected')
    for optimizer in optimizers: optimizer.step()
    for scheduler in schedulers:
        if scheduler is not None: scheduler.step()
    return norms


def _save(path, cfg, parent_sha, model, optimizers, schedulers, stream, progress, rec=None, sid=None):
    atomic_save(path, {'config': cfg, 'sources': source_identity(), 'parent_sha256': parent_sha,
                'environment': environment(next(model.parameters()).device),
                'tokenizer': model.state_dict(), 'recommender': None if rec is None else rec.state_dict(),
                'optimizers': [o.state_dict() for o in optimizers],
                'schedulers': [None if s is None else s.state_dict() for s in schedulers],
                'stream': stream.state(), 'progress': progress, 'rng': rng_state(), 'sid': sid})


def _restore(path, cfg, parent_sha, model, optimizers, schedulers, stream, rec=None):
    state = read_state(path)
    if state['config'] != cfg or state['sources'] != source_identity() or state['parent_sha256'] != parent_sha:
        raise ValueError('Resume source/config/parent mismatch')
    if state['environment'] != environment(next(model.parameters()).device):
        raise ValueError('Resume environment changed; numerical equivalence has not been established')
    model.load_state_dict(state['tokenizer'])
    if rec is not None: rec.load_state_dict(state['recommender'])
    for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
    for scheduler, saved in zip(schedulers, state['schedulers']):
        if scheduler is not None: scheduler.load_state_dict(saved)
    stream.restore(state['stream'])
    restore_rng(state['rng'])
    return state


def pretrain(cfg, prepared, dataset, content_path, output, device, resume=False, *, _stop_after=None):
    independent_output(output, prepared, dataset, content_path)
    device = configure_device(device)
    parent, meta = load_parent(cfg, prepared)
    ensure_inputs(cfg, meta, dataset, content_path)
    content = load_content(content_path, cfg, device)
    tok = make_tokenizer(cfg).to(device)
    tok.load_state_dict(parent['tokenizers']['intra' if cfg['mixer_enabled'] else 'plain'])
    c = cfg['rq']; per_epoch = math.ceil(len(content)/c['batch_size'])
    optimizer, scheduler = pretrain_optimizer(tok, cfg, per_epoch)
    stream = PermutationStream(len(content), seed_for(cfg['seed'], cfg['family'], 'pretrain-data'))
    seed_all(seed_for(cfg['seed'], cfg['family'], 'pretrain-updates'))
    progress = {'step': 0, 'evaluated': 0, 'best_collision': None, 'best_step': None, 'records': [],
                'loss_sum': {}, 'loss_count': 0, 'train_seconds': 0.0, 'done': False}
    with output_lock(output, resume=resume):
        output = Path(output)
        if resume:
            saved = _restore(output/'last.pt', cfg, meta['initial_sha256'], tok, [optimizer], [scheduler], stream)
            progress = saved['progress']
        def save(path): _save(path, cfg, meta['initial_sha256'], tok, [optimizer], [scheduler], stream, progress)
        while progress['step'] < per_epoch*c['epochs'] or progress['evaluated'] < progress['step']:
            epoch = progress['step']//per_epoch
            due = stream.at_end and (epoch % c['eval_interval'] == 0 or epoch == c['epochs'])
            if due and progress['evaluated'] < progress['step']:
                tok.eval()
                with preserve_rng(), torch.no_grad():
                    # RQ pretraining needs semantic collision statistics, not a
                    # deployable recommender catalog at every intermediate epoch.
                    semantic = tok.indices(content)
                    sid = torch.cat((semantic.new_full((1, 3), -1), semantic), dim=0)
                    stats = encoding_stats(tok, sid)
                    maximum = torch.unique(semantic, dim=0, return_counts=True)[1].max().item()
                    stats.update(max_collision_bucket=maximum,
                                 suffix_capacity_ok=maximum <= c['codes'])
                row = {'epoch': epoch, 'step': progress['step'], 'encoding': stats,
                       'loss': {k: v/progress['loss_count'] for k, v in progress['loss_sum'].items()}}
                progress['records'].append(row); progress['evaluated'] = progress['step']
                progress['loss_sum'], progress['loss_count'] = {}, 0
                score = stats['collision_redundancy']
                improved = progress['best_collision'] is None or score < progress['best_collision']
                if improved:
                    progress['best_collision'], progress['best_step'] = score, progress['step']
                    if c['selection'] == 'best_collision': save(output/'selected.pt')
                save(output/'last.pt'); write_json(output/'progress.json', progress)
            if progress['step'] >= per_epoch*c['epochs']: break
            tok.train()
            start = time.perf_counter()
            idx = stream.take(c['batch_size']).to(device)
            loss, parts = tok.loss(content[idx])
            # Original DIGER pretraining trainer clips at norm 1 as well; TIGER clips only
            # when its (versioned) config explicitly sets rq.clip_norm.
            finite_step(loss, [tok], [optimizer], [scheduler], 1.0 if cfg['family']=='diger' else c.get('clip_norm'))
            progress['step'] += 1
            for k, v in dict(total=loss, **parts).items():
                progress['loss_sum'][k] = progress['loss_sum'].get(k, 0)+v.detach().item()*len(idx)
            progress['loss_count'] += len(idx)
            progress['train_seconds'] += time.perf_counter()-start
            if progress['step'] % cfg['execution']['checkpoint_every_updates'] == 0 or progress['step'] == per_epoch*c['epochs']:
                save(output/'last.pt')
            if _stop_after is not None and progress['step'] >= _stop_after:
                save(output/'last.pt'); return progress
        progress['done'] = True
        save(output/'last.pt')
        if c['selection'] == 'final': save(output/'selected.pt')
        write_json(output/'complete.json', {'arm': cfg['arm'], 'selected_sha256': sha256(output/'selected.pt'),
                   'last_sha256': sha256(output/'last.pt'), 'parent_sha256': meta['initial_sha256'],
                   'selection': c['selection'], 'progress': progress})
        return progress


def initialize_recommendation(cfg, parent, pretrain_path, device):
    checkpoint = read_state(Path(pretrain_path)/'selected.pt')
    receipt = json.loads((Path(pretrain_path)/'complete.json').read_text(encoding='utf-8'))
    if checkpoint['config'] != cfg or checkpoint['sources'] != source_identity():
        raise ValueError('Tokenizer checkpoint configuration/source mismatch')
    if receipt['selected_sha256'] != sha256(Path(pretrain_path)/'selected.pt'):
        raise ValueError('Tokenizer checkpoint checksum mismatch')
    pre = make_tokenizer(cfg).to(device); pre.load_state_dict(checkpoint['tokenizer'])
    if cfg['family'] == 'diger':
        tok = make_tokenizer(cfg, joint=True).to(device); tok.load_pretrained(pre)
    else:
        tok = pre; tok.requires_grad_(False); tok.eval()
    rec = ARRecommender(cfg).to(device); rec.load_state_dict(parent['recommender'])
    return rec, tok, receipt['selected_sha256']


def train(cfg, prepared, pretrain_path, dataset, content_path, output, device, resume=False, *, _stop_after=None):
    independent_output(output, prepared, pretrain_path, dataset, content_path)
    device = configure_device(device)
    parent, meta = load_parent(cfg, prepared)
    ensure_inputs(cfg, meta, dataset, content_path)
    pre_receipt = json.loads((Path(pretrain_path)/'complete.json').read_text(encoding='utf-8'))
    if pre_receipt['parent_sha256'] != meta['initial_sha256']:
        raise ValueError('Pretraining belongs to a different common parent')
    content = load_content(content_path, cfg, device)
    rec, tok, parent_sha = initialize_recommendation(cfg, parent, pretrain_path, device)
    tasks = SessionTasks(PackedSplit(dataset, 'train', cfg))
    validation = PackedSplit(dataset, 'validation', cfg)
    c = cfg['training']; per_epoch = math.ceil(len(tasks)/c['batch_size'])
    validation_interval = c.get('validation_interval_updates')
    if validation_interval is not None and (type(validation_interval) is not int or validation_interval < 1):
        raise ValueError('validation_interval_updates must be a positive integer')
    total = c['max_steps'] or c['max_epochs']*per_epoch
    training_precision(cfg)
    meter = DeviceLossMeter()
    optimizers, schedulers = recommendation_optimizers(rec, tok, cfg, per_epoch)
    stream = PermutationStream(len(tasks), seed_for(cfg['seed'], 'ar', 'recommendation-data'))
    seed_all(seed_for(cfg['seed'], cfg['family'], 'recommendation-updates'))
    progress = {'step': 0, 'evaluated': 0, 'best_score': None, 'best_step': None, 'bad_epochs': 0,
                'records': [], 'loss_sum': {}, 'loss_count': 0, 'train_seconds': 0.0, 'done': False}
    tok.eval()
    with preserve_rng(): sid = add_suffix(tok.indices(content), cfg['rq']['codes'])
    previous_books = tok.effective_books().detach().clone()
    with output_lock(output, resume=resume):
        output = Path(output)
        if resume:
            saved = _restore(output/'last.pt', cfg, parent_sha, tok, optimizers, schedulers, stream, rec)
            progress, sid = saved['progress'], saved['sid'].to(device)
            previous_books = torch.tensor(progress['previous_books'], device=device)
        else:
            progress['previous_books'] = previous_books.cpu().tolist()
        def save(path):
            meter.flush(progress)
            _save(path, cfg, parent_sha, tok, optimizers, schedulers, stream, progress, rec, sid)
        while not progress['done']:
            boundary = (stream.at_end if validation_interval is None
                        else progress['step'] % validation_interval == 0)
            due = (boundary or progress['step'] == total) and progress['step'] > progress['evaluated']
            if due:
                tok.eval()
                checkpoint = output/'checkpoints'/f"step_{progress['step']:07d}.pt"
                if 'pending_validation' not in progress:
                    old_sid = sid
                    if cfg['family'] == 'diger':
                        with preserve_rng(): sid = add_suffix(tok.indices(content), cfg['rq']['codes'])
                    stats = encoding_stats(tok, sid, old_sid, previous_books)
                    previous_books = tok.effective_books().detach().clone()
                    progress['previous_books'] = previous_books.cpu().tolist()
                    # Immutable model snapshot followed by a resumable pending
                    # receipt. Never rerun SID refresh after evaluation started.
                    save(checkpoint)
                    progress['pending_validation'] = {'encoding': stats, 'checkpoint_sha256': sha256(checkpoint)}
                    save(output/'last.pt')
                pending = progress['pending_validation']
                checkpoint_sha = pending['checkpoint_sha256']
                if sha256(checkpoint) != checkpoint_sha:
                    raise ValueError('Pending validation checkpoint was modified')
                stats = pending['encoding']
                catalog = Catalog(sid, cfg['rq']['codes'], parent['original_ids'])
                result = evaluate(rec, catalog, validation, cfg, output/'validation'/f"step_{progress['step']:07d}",
                                  {'checkpoint_sha256': checkpoint_sha, 'config_sha256': digest(cfg)})
                metric = 'ev_ndcg@5' if hasattr(validation, 'watch_ratios') else 'ndcg@5'
                score = result['metrics'][metric]
                row = {'step': progress['step'], 'session_epochs': stream.epoch+stream.cursor/len(tasks),
                       'r_updates': progress['step'], 't_updates': progress['step'] if cfg['family']=='diger' else 0,
                       'validation': result, 'encoding': stats, 'train_seconds': progress['train_seconds'],
                       'loss': {k: v/progress['loss_count'] for k, v in progress['loss_sum'].items()},
                       'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha}
                progress['records'].append(row); progress['evaluated'] = progress['step']
                del progress['pending_validation']
                progress['loss_sum'], progress['loss_count'] = {}, 0
                improved = progress['best_score'] is None or score > progress['best_score']
                progress['bad_epochs'] = 0 if improved else progress['bad_epochs']+1
                if improved:
                    progress['best_score'], progress['best_step'] = score, progress['step']
                progress['done'] = progress['step'] >= total or (c['patience'] is not None and progress['bad_epochs'] >= c['patience'])
                save(output/'last.pt')
                write_json(output/'progress.json', progress)
            if progress['done']: break
            rec.train(); tok.train(cfg['family']=='diger')
            start = time.perf_counter()
            batch = tasks.batch(stream.take(c['batch_size']), device)
            with train_kernels(cfg, device):
                with train_autocast(cfg, device):
                    if cfg['family'] == 'diger':
                        loss, parts = joint_loss(rec, tok, batch, sid, stream.epoch, progress['step'])
                        models = [rec, tok]
                    else:
                        loss = rec.hard_loss(batch, sid); parts = {'code': loss}; models = [rec]
                norms = finite_step(loss, models, optimizers, schedulers, c['clip_norm'])
            progress['step'] += 1
            meter.add(progress, dict(total=loss, **parts), len(batch['target']), norms)
            progress['loss_count'] += len(batch['target'])
            progress['train_seconds'] += time.perf_counter()-start
            if progress['step'] % cfg['execution']['checkpoint_every_updates'] == 0 or stream.at_end or progress['step'] == total:
                save(output/'last.pt')
            if _stop_after is not None and progress['step'] >= _stop_after:
                save(output/'last.pt'); return progress
        receipt = {'arm': cfg['arm'], 'parent_sha256': parent_sha, 'prepared_sha256': meta['initial_sha256'],
                   'input_binding': meta['binding'], 'environment': environment(device),
                   'config_sha256': digest(cfg), 'sources': source_identity(), 'progress': progress,
                   'parameter_count': sum(p.numel() for p in rec.parameters())+sum(p.numel() for p in tok.parameters()),
                   'best': progress['records'][next(i for i,r in enumerate(progress['records']) if r['step']==progress['best_step'])],
                   'final': progress['records'][-1]}
        write_json(output/'complete.json', receipt)
        return progress


def test(cfg, prepared, pretrain_path, dataset, content_path, training, output, device):
    independent_output(output, prepared, pretrain_path, dataset, content_path, training)
    device = configure_device(device)
    parent, meta = load_parent(cfg, prepared)
    ensure_inputs(cfg, meta, dataset, content_path)
    training = Path(training)
    receipt = json.loads((training/'complete.json').read_text(encoding='utf-8'))
    if receipt['sources'] != source_identity() or receipt['config_sha256'] != digest(cfg):
        raise ValueError('Training receipt belongs to different sources/config')
    data = PackedSplit(dataset, 'test', cfg)
    rec, tok, _ = initialize_recommendation(cfg, parent, pretrain_path, device)
    if receipt['prepared_sha256'] != meta['initial_sha256']:
        raise ValueError('Training result belongs to a different prepared parent')
    results = {}
    nodes = {'best': receipt['best'], 'final': receipt['final']}
    if cfg['evaluation'].get('selection_rule') == 'smoothed3':
        from scripts.kuairec_data.selection import smoothed_choice
        metric = 'ev_ndcg@5' if hasattr(data, 'watch_ratios') else 'ndcg@5'
        records = receipt['progress']['records']
        chosen, smoothed, raw, _ = smoothed_choice([(r, r['validation']['metrics'][metric]) for r in records])
        nodes['smoothed'] = dict(chosen, smoothed_validation=smoothed)
    for name, node in nodes.items():
        checkpoint = training/'checkpoints'/f"step_{node['step']:07d}.pt"
        if sha256(checkpoint) != node['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint SHA mismatch')
        state = read_state(checkpoint)
        if state['config'] != cfg or state['sources'] != source_identity() or state['parent_sha256'] != receipt['parent_sha256']:
            raise ValueError('Test snapshot configuration/source/parent mismatch')
        rec.load_state_dict(state['recommender'])
        sid = state['sid'].to(device)
        catalog = Catalog(sid, cfg['rq']['codes'], parent['original_ids'])
        results[name] = evaluate(rec, catalog, data, cfg, Path(output)/f"step_{node['step']:07d}",
                                {'checkpoint_sha256': node['checkpoint_sha256'], 'config_sha256': digest(cfg)})
    summary = {'arm': cfg['arm'], **results}
    if 'smoothed' in nodes:
        summary['smoothed_selection'] = {'step': nodes['smoothed']['step'],
                                         'smoothed_validation': nodes['smoothed']['smoothed_validation']}
    write_json(Path(output)/'complete.json', summary)
    return results
