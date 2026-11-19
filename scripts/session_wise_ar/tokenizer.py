"""Paper-configured TIGER RQ and an unmodified DIGER RQ with effective weights."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from scripts.rqcd_mixer.tokenizer import CodebookMixer, make_mlp, distances
from .model import diger_config
from .upstream import modules


def new_mixer(cfg):
    # Adding Mixer must not consume the base model/data random stream.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 901)
        return CodebookMixer(cfg, 'intra')


class TigerRQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg['rq']
        self.encoder = make_mlp([c['input_dim'], *c['hidden_dims'], c['latent_dim']])
        self.decoder = make_mlp([c['latent_dim'], *reversed(c['hidden_dims']), c['input_dim']])
        self.books = nn.Parameter(torch.zeros(c['levels'], c['codes'], c['latent_dim']))
        self.mixer = new_mixer(cfg) if cfg['mixer_enabled'] else None
        self.register_buffer('initialized', torch.tensor(False))

    def base_books(self):
        return self.books

    def effective_books(self):
        return self.books if self.mixer is None else self.books + self.mixer(self.books)

    @torch.no_grad()
    def initialize(self, x):
        from sklearn.cluster import KMeans
        residual = self.encoder(x)
        for i in range(len(self.books)):
            fit = KMeans(n_clusters=self.books.shape[1], n_init=10,
                         max_iter=self.cfg['rq']['kmeans_iterations'], random_state=42)
            fit.fit(residual.cpu().numpy())
            centers = torch.as_tensor(fit.cluster_centers_, device=x.device)
            self.books[i].copy_(centers)
            residual = residual - centers[distances(residual, centers).argmin(-1)]
        self.initialized.fill_(True)

    def quantize(self, x):
        if not self.initialized:
            raise RuntimeError('Initialize the shared RQ parent before training')
        z = self.encoder(x)
        residual, quantized = z, torch.zeros_like(z)
        losses, ids = [], []
        for book in self.effective_books():
            index = distances(residual, book).argmin(-1)
            selected = book[index]
            # TIGER uses dimension-summed squared norms, not mean over dimensions.
            losses.append(((selected-residual.detach()).square().sum(-1)
                           + self.cfg['rq']['beta']*(selected.detach()-residual).square().sum(-1)).mean())
            quantized = quantized + selected
            residual = residual - selected.detach()
            ids.append(index)
        return z, quantized, torch.stack(losses).sum(), torch.stack(ids, -1)

    def loss(self, x):
        z, q, vq, _ = self.quantize(x)
        reconstruction = self.decoder(z + (q-z).detach())
        recon = (reconstruction-x).square().sum(-1).mean()
        return recon+vq, {'reconstruction': recon, 'vq': vq}

    @torch.no_grad()
    def indices(self, x):
        return self.quantize(x)[-1]


class _EffectiveWeight(nn.Module):
    def __init__(self, owner, level):
        super().__init__()
        object.__setattr__(self, 'owner', weakref.ref(owner))
        self.level = level

    def forward(self, original):
        owner = self.owner()
        # During parametrization registration, return an identity for shape checks.
        if owner._effective is None:
            return original
        return owner._effective[self.level]


class DigerRQ(nn.Module):
    def __init__(self, cfg, *, joint=False):
        super().__init__()
        self.cfg, self.joint = cfg, joint
        native_cfg = diger_config(cfg)
        self._effective = None
        if joint:
            self.native = modules().vq.RQVAE(native_cfg, in_dim=cfg['rq']['input_dim'])
        else:
            native_cfg.update(kmeans_init=False, quant_loss_weight=1.0)
            self.native = modules().pre.RQVAE(SimpleNamespace(**native_cfg), in_dim=cfg['rq']['input_dim'])
        self.mixer = new_mixer(cfg) if cfg['mixer_enabled'] else None
        self.register_buffer('initialized', torch.tensor(False))
        if self.mixer is not None:
            for i, layer in enumerate(self.native.rq.vq_layers):
                parametrize.register_parametrization(layer.embedding, 'weight', _EffectiveWeight(self, i))

    def original_weights(self):
        for layer in self.native.rq.vq_layers:
            embedding = layer.embedding
            yield (embedding.parametrizations.weight.original if self.mixer is not None else embedding.weight)

    def base_books(self):
        return torch.stack(list(self.original_weights()))

    def effective_books(self):
        base = self.base_books()
        return base if self.mixer is None else base + self.mixer(base)

    @contextmanager
    def effective(self):
        if self._effective is not None:
            raise RuntimeError('Nested effective-codebook contexts are unsupported')
        self._effective = self.effective_books()
        try:
            with parametrize.cached():
                yield
        finally:
            self._effective = None

    @torch.no_grad()
    def initialize(self, x):
        # Execute upstream sequential kmeans initialization on a plain temporary
        # model. Writing embedding.weight.data through parametrization is unsafe.
        native_cfg = diger_config(self.cfg)
        native_cfg.update(kmeans_init=True, quant_loss_weight=1.0,
                          kmeans_iters=self.cfg['rq']['kmeans_iterations'])
        with torch.random.fork_rng(devices=[]):
            plain = modules().pre.RQVAE(SimpleNamespace(**native_cfg), in_dim=x.shape[-1]).to(x.device)
        plain.encoder.load_state_dict(self.native.encoder.state_dict())
        plain.decoder.load_state_dict(self.native.decoder.state_dict())
        plain.train()(x)
        for target, layer in zip(self.original_weights(), plain.rq.vq_layers):
            target.copy_(layer.embedding.weight)
        self.initialized.fill_(True)

    def loss(self, x):
        if not self.initialized:
            raise RuntimeError('Missing initialized RQ parent')
        with self.effective():
            out, vq, _ = self.native(x)
            total, recon = self.native.compute_loss(out, vq, xs=x)
        return total, {'reconstruction': recon, 'vq': vq}

    @torch.no_grad()
    def indices(self, x):
        with self.effective():
            if self.joint:
                # Preserve original full-catalog Sinkhorn assignment batch.
                return self.native.get_indices(x, use_sinkhorn=True)
            return self.native.get_indices(x)

    def structured(self, x, epoch=0, step=0):
        if not self.joint:
            raise RuntimeError('Structured recommendation path requires joint RQ')
        with self.effective():
            return self.native(x, current_epoch=epoch, global_step=step,
                               return_structured=True, deterministic_st=False,
                               assignment_temperature=2.0, assignment_mode='true_e2e_frqud',
                               assignment_forward='hard_st')

    def load_pretrained(self, pre):
        if not self.joint or pre.joint:
            raise ValueError('Expected pretraining -> joint conversion')
        self.native.encoder.load_state_dict(pre.native.encoder.state_dict())
        with torch.no_grad():
            for dst, src in zip(self.original_weights(), pre.original_weights()):
                dst.copy_(src)
        if self.mixer is not None:
            self.mixer.load_state_dict(pre.mixer.state_dict())
        self.initialized.fill_(True)
        self.native.encoder.requires_grad_(False)


def make_tokenizer(cfg, joint=False):
    return TigerRQ(cfg) if cfg['family'] == 'tiger' else DigerRQ(cfg, joint=joint)


def joint_loss(rec, tokenizer, batch, sid, epoch=0, step=0):
    """Native DIGER true_e2e_frqud losses; only session context assembly changes."""
    context, targets = batch['context'], batch['target']
    valid = context.ne(0)
    flat = torch.cat((context[valid], targets))
    unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    hist_inverse = torch.zeros_like(context)
    hist_inverse[valid] = inverse[:-len(targets)]
    target_inverse = inverse[-len(targets):]
    output = tokenizer.structured(rec.native.semantic_embedding(unique), epoch, step)
    soft = torch.stack([level.soft_probabilities for level in output.levels], 1)
    hard = F.one_hot(sid[unique, :3], rec.codes).to(soft)
    probabilities = hard - soft.detach() + soft
    target_p = probabilities[target_inverse]
    inputs, mask = rec.native.get_mixture_input_embeddings(
        probabilities[hist_inverse], sid[context, -1].clamp_min(0), valid)
    decoder = rec.native.get_differentiable_decoder_inputs(target_p)
    prediction = rec.native(inputs_embeds=inputs, attention_mask=mask, decoder_inputs_embeds=decoder)
    semantic, suffix = rec.native.compute_differentiable_code_losses(
        prediction.logits, target_p, sid[targets, -1], suffix_logits=prediction.suffix_logits)
    recon = (output.quantized-output.latent).square().mean(-1)[target_inverse].mean()
    vq = output.vq_loss_per_item[target_inverse].mean()
    parts = {'semantic': semantic, 'suffix': suffix, 'reconstruction': recon, 'vq': vq}
    return sum(parts.values()), parts
