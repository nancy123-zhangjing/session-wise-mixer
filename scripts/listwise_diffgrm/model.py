"""List adapters over the real DiffGRM backbone; no replacement attention layers.

J denoises K*4 slots jointly. H reuses four-slot denoising for each true prefix
during training (generated prefixes are supplied by the separate beam decoder).
H defaults to historical re-encoding; opt-in modes move four-SID item prefixes
to asymmetric self-attention or an independent prefix cross-attention branch.
The historical OCN teacher uses raw encoder hidden states as cross K/V, whereas
the student and beam inference use projected K/V. This baseline distinction is
deliberately retained, including in the K=1 compatibility tests.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .prefix_decoder import MODES, PrefixCrossAttention, asymmetric_mask, decoder_layer


class ListwiseDiffGRM(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self.config = dict(config)
        # Ephemeral constants only: no parameters, RNG, gradients or checkpoint keys.
        self._execution_constants = {}
        self._sid_inputs_validated = False
        self.backbone = backbone
        self.architecture = config.get('architecture', 'J')
        self.list_length = int(config.get('list_length', 5))
        self.n_digit = int(backbone.n_digit)
        self.codebook_size = int(backbone.codebook_size)
        self.n_embd = int(backbone.n_embd)
        self.prefix_mode = config.get('h_prefix_mode', 'history')
        if self.prefix_mode not in MODES or (self.prefix_mode != 'history' and self.architecture != 'H'):
            raise ValueError('h_prefix_mode requires history/asymmetric/cross_attention; new modes are H-only')
        self.uses_decoder_prefix = self.prefix_mode != 'history'
        if self.architecture not in ('J', 'H') or self.list_length < 1:
            raise ValueError('architecture must be J/H and list_length positive')
        if self.n_digit != 4:
            raise ValueError('The list protocol keeps four shared SID codebooks')
        if getattr(backbone, 'item_local_attention_enabled', False):
            raise ValueError('Item-local attention is not part of this experiment')
        if (backbone.masking_strategy != 'guided' or backbone.guided_select != 'least'
                or backbone.guided_conf_metric != 'msp' or backbone.guided_steps != 4
                or config.get('guided_refresh_each_step', False)
                or config.get('diffusion_loss_reduction', 'per_view') != 'per_view'):
            raise ValueError('List protocol requires static guided-least/MSP, four views, per_view')
        # The runner normally constructs both routes with this capacity already.
        required = int(config.get('observed_history_max', 50)) + self.list_length - 1
        if backbone.pos_emb_enc.num_embeddings < required:
            old = backbone.pos_emb_enc
            expanded = nn.Embedding(required, self.n_embd, device=old.weight.device,
                                    dtype=old.weight.dtype)
            nn.init.normal_(expanded.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                expanded.weight[:old.num_embeddings].copy_(old.weight)
            backbone.pos_emb_enc = expanded
            backbone.max_history_len = required
        self.item_position_embedding = nn.Embedding(self.list_length, self.n_embd)
        nn.init.normal_(self.item_position_embedding.weight, mean=0.0, std=0.02)
        if self.prefix_mode == 'cross_attention':
            # Extra parameters must not shift downstream alignment initialization
            # merely by consuming additional CPU RNG draws during construction.
            with torch.random.fork_rng(devices=[]):
                self.prefix_cross_layers = nn.ModuleList([
                    PrefixCrossAttention(config, type(block.self_attn), self.n_embd)
                    for block in backbone.decoder_blocks])

    def _validate_history(self, history_sid, history_mask):
        if history_sid.ndim != 3 or history_sid.shape[-1] != 4:
            raise ValueError('history_sid must have shape [B,L,4]')
        if history_mask.shape != history_sid.shape[:2]:
            raise ValueError('history_mask must have shape [B,L]')
        if history_sid.shape[1] > self.backbone.pos_emb_enc.num_embeddings:
            raise ValueError('History exceeds the configured positional capacity')
        if self.uses_decoder_prefix and history_sid.shape[1] > int(self.config.get('observed_history_max', 50)):
            raise ValueError('Decoder-prefix H accepts observed history only; pass generated items as prefix_sid')
        if self._sid_inputs_validated:
            return
        valid = history_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Each list needs at least one visible historical item')
        if bool(((history_sid[valid] < 0) | (history_sid[valid] >= self.codebook_size)).any()):
            raise ValueError('Visible history must contain local SID values 0..C-1')
        if bool(history_sid[~valid].ne(-1).any()):
            raise ValueError('Padded history SID slots must be -1')

    def _check_assignments(self, assignments, sid, name):
        if assignments is not None and assignments.shape != (*sid.shape, self.codebook_size):
            raise ValueError(f'{name} must be SID assignments [...,4,C], not embeddings')

    def _encode(self, history_sid, history_mask, history_st=None):
        self._validate_history(history_sid, history_mask)
        self._check_assignments(history_st, history_sid, 'history_st')
        batch = {'history_sid': history_sid, 'history_mask': history_mask.bool(),
                 '_joint_prepared': True}
        if history_st is not None:
            batch['history_sid_st'] = history_st
            # The original forward uses these only to recognize padding after
            # preparation has been bypassed; no catalog lookup is performed.
            batch['history_item_ids'] = history_mask.long()
        return self.backbone(batch, return_loss=False).hidden_states

    def _embed_prefix(self, prefix_sid, item_index, prefix_st=None):
        if prefix_sid.ndim != 3 or prefix_sid.shape[1:] != (int(item_index), 4):
            raise ValueError('prefix_sid must be [B,item_index,4], with completed earlier items only')
        if not self._sid_inputs_validated and bool(((prefix_sid < 0) | (prefix_sid >= self.codebook_size)).any()):
            raise ValueError('Decoder prefix must contain completed local SID values')
        self._check_assignments(prefix_st, prefix_sid, 'prefix_st')
        parts = []
        for digit in range(4):
            start = self.backbone.tokenizer.sid_offset + digit * self.codebook_size
            weight = self.backbone.embedding.weight[start:start + self.codebook_size]
            emb = F.embedding(prefix_sid[:, :, digit], weight) if prefix_st is None else prefix_st[:, :, digit] @ weight
            parts.append(emb)
        hidden = torch.stack(parts, dim=2)
        hidden = hidden + self.item_position_embedding.weight[:int(item_index)][None, :, None, :]
        return hidden.reshape(prefix_sid.shape[0], int(item_index) * 4, self.n_embd)

    def _decode(self, encoded, partial_sid, item_index=None, target_st=None, teacher=False,
                prefix_sid=None, prefix_st=None):
        core = self.backbone
        batch_size, positions = partial_sid.shape
        parts = []
        for p in range(positions):
            digit = p % 4
            start = core.tokenizer.sid_offset + digit * self.codebook_size
            if target_st is None:
                emb = core.embedding(partial_sid[:, p].clamp_min(0) + start)
            else:
                emb = target_st[:, p] @ core.embedding.weight[start:start + self.codebook_size]
            emb = torch.where(partial_sid[:, p, None].eq(-1),
                              core.mask_emb_table.weight[digit][None], emb)
            if self.list_length > 1:  # K=1 is an exact original-model oracle.
                idx = p // 4 if item_index is None else int(item_index)
                emb = emb + self.item_position_embedding.weight[idx]
            parts.append(emb)
        hidden = torch.stack(parts, dim=1)
        prefix = None
        mask = None
        if self.uses_decoder_prefix:
            if item_index is None or not 0 <= int(item_index) < self.list_length or positions != 4:
                raise ValueError('Decoder prefix requires a valid item_index and four current slots')
            if prefix_sid is None:
                if item_index != 0:
                    raise ValueError('Completed prefix SID is required after the first item')
                prefix_sid = partial_sid.new_empty((batch_size, 0, 4))
            if prefix_sid.shape[0] != batch_size:
                raise ValueError('Prefix and current batch sizes differ')
            prefix = self._embed_prefix(prefix_sid, item_index, prefix_st)
            if self.prefix_mode == 'asymmetric':
                key = ('prefix_mask', prefix.shape[1], hidden.device)
                if self.config.get('reuse_static_tensors', False):
                    if key not in self._execution_constants:
                        self._execution_constants[key] = asymmetric_mask(prefix.shape[1], 4, hidden.device)
                    mask = self._execution_constants[key]
                else:
                    mask = asymmetric_mask(prefix.shape[1], 4, hidden.device)
                hidden = torch.cat((prefix, hidden), dim=1)
        hidden = core.drop(hidden)
        for layer_index, block in enumerate(core.decoder_blocks):
            cross_kv = None
            if not teacher:
                projected = block.cross_attn.qkv(encoded)
                cross_kv = projected[..., self.n_embd:]
            if self.uses_decoder_prefix:
                prefix_layer = self.prefix_cross_layers[layer_index] if self.prefix_mode == 'cross_attention' else None
                hidden = decoder_layer(block, hidden, encoded, cross_kv, mask, prefix_layer, prefix)
            else:
                hidden = block(hidden, encoder_hidden=encoded, past_key_value=None,
                               use_cache=False, cross_key_value=cross_kv)['hidden_states']
        if self.prefix_mode == 'asymmetric':
            hidden = hidden[:, -4:]
        hidden = core.ln_f(hidden)
        return torch.stack([core._compute_digit_logits(hidden[:, p], digit=p % 4)
                            for p in range(positions)], dim=1)

    def _ocn_order(self, encoded, positions, item_index, prefix_sid=None):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        device = encoded.device
        try:
            # Disabling the teacher's cast cache prevents detached BF16 weights
            # from being reused by the subsequent differentiable student pass.
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.get_autocast_dtype(device.type),
                enabled=torch.is_autocast_enabled(device.type), cache_enabled=False,
            ):
                key = ('ocn_full_mask', encoded.shape[0], positions, device)
                if self.config.get('optimized_model_checks', False):
                    if key not in self._execution_constants:
                        self._execution_constants[key] = torch.full(
                            (encoded.shape[0], positions), -1, dtype=torch.long, device=device)
                    full_mask = self._execution_constants[key]
                else:
                    full_mask = torch.full((encoded.shape[0], positions), -1,
                                           dtype=torch.long, device=device)
                kwargs = {'prefix_sid': prefix_sid} if self.uses_decoder_prefix else {}
                logits = self._decode(encoded, full_mask, item_index, teacher=True, **kwargs)
                confidence = logits.softmax(dim=-1).max(dim=-1).values
                return confidence.argsort(dim=1).detach()
        finally:
            for module, training in modes:
                module.training = training

    def _depths(self, batch_size, positions, device, supplied=None):
        if supplied is None:
            if positions == 4:
                if self.config.get('reuse_static_tensors', False):
                    key = ('depths', device)
                    if key not in self._execution_constants:
                        self._execution_constants[key] = torch.arange(1, 5, device=device)
                    return self._execution_constants[key].expand(batch_size, -1)
                return torch.arange(1, 5, device=device).expand(batch_size, -1)
            # IID random keys give uniform sampling without replacement.
            shallow = torch.rand(batch_size, positions - 1, device=device).argsort(1)[:, :3] + 1
            return torch.cat((shallow.sort(1).values,
                              torch.full((batch_size, 1), positions, device=device)), dim=1)
        depths = torch.as_tensor(supplied, dtype=torch.long, device=device)
        if depths.ndim == 1:
            depths = depths.expand(batch_size, -1)
        if (depths.shape != (batch_size, 4) or bool((depths[:, 0] < 1).any())
                or bool((depths[:, -1] != positions).any())
                or not bool((depths[:, 1:] > depths[:, :-1]).all())):
            raise ValueError('Four strictly increasing depths ending in full mask are required')
        return depths

    def _condition_loss(self, encoded, labels, target_st=None, item_index=None, mask_depths=None,
                        prefix_sid=None, prefix_st=None):
        n, positions = labels.shape
        order = self._ocn_order(encoded, positions, item_index, prefix_sid) if self.uses_decoder_prefix else self._ocn_order(encoded, positions, item_index)
        depths = self._depths(n, positions, labels.device, mask_depths)
        # [N,V,P], then view-major to match the original four-view construction.
        ranks = order.argsort(dim=1)
        masks = ranks[:, None, :] < depths[:, :, None]
        flat_masks = masks.transpose(0, 1).reshape(4 * n, positions)
        repeated_labels = labels.repeat(4, 1)
        partial = repeated_labels.masked_fill(flat_masks, -1)
        assignments = None if target_st is None else target_st.repeat(4, 1, 1)
        kwargs = {}
        if self.uses_decoder_prefix:
            kwargs = dict(prefix_sid=prefix_sid.repeat(4, 1, 1),
                          prefix_st=None if prefix_st is None else prefix_st.repeat(4, 1, 1, 1))
        logits = self._decode(encoded.repeat(4, 1, 1), partial, item_index, assignments, **kwargs)
        smoothing = float(self.config.get('label_smoothing', 0.1))
        if assignments is not None:
            log_probs = logits.float().log_softmax(dim=-1)
            nll = -(assignments.float() * log_probs).sum(dim=-1)
            losses = (1.0 - smoothing) * nll - smoothing * log_probs.mean(dim=-1)
        else:
            losses = F.cross_entropy(logits.transpose(1, 2), repeated_labels,
                                     reduction='none', label_smoothing=smoothing)
        per_view = (losses * flat_masks).sum(-1) / flat_masks.sum(-1)
        per_condition = per_view.reshape(4, n).mean(0)
        diagnostics = {'mask_depths': depths.detach(), 'ocn_order': order,
                       'supervised_positions': (n * 10 if positions == 4 and
                           self.config.get('optimized_model_checks', False)
                           else int(flat_masks.sum().item())),
                       'masked_positions': masks.detach()}
        return per_condition, diagnostics

    def forward_loss(self, history_sid, history_mask, target_sid, history_st=None,
                     target_st=None, mask_depths=None):
        # Validate current SID values once per forward, including after any T
        # refresh. Only slices/repeats of these inputs bypass redundant checks.
        if not (self.uses_decoder_prefix and self.config.get('optimized_model_checks', False)):
            return self._forward_loss_impl(history_sid, history_mask, target_sid,
                                           history_st, target_st, mask_depths)
        if target_sid.shape != (history_sid.shape[0], self.list_length, 4):
            raise ValueError('target_sid must have shape [B,K,4]')
        if bool(((target_sid < 0) | (target_sid >= self.codebook_size)).any()):
            raise ValueError('Targets must contain local SID values 0..C-1')
        self._validate_history(history_sid, history_mask)
        previous = self._sid_inputs_validated
        self._sid_inputs_validated = True
        try:
            return self._forward_loss_impl(history_sid, history_mask, target_sid,
                                           history_st, target_st, mask_depths)
        finally:
            self._sid_inputs_validated = previous

    def _forward_loss_impl(self, history_sid, history_mask, target_sid, history_st=None,
                           target_st=None, mask_depths=None):
        """Return scalar rec loss and list-normalized losses; ST inputs are assignments.

        Diagnostics use [B,4,P] masks for J and item-major [K*B,4,4] for H.
        Auxiliary PSE losses and optimizer updates belong to the external runner.
        """
        batch_size = history_sid.shape[0]
        if target_sid.shape != (batch_size, self.list_length, 4):
            raise ValueError('target_sid must have shape [B,K,4]')
        if not self._sid_inputs_validated and bool(((target_sid < 0) | (target_sid >= self.codebook_size)).any()):
            raise ValueError('Targets must contain local SID values 0..C-1')
        self._check_assignments(target_st, target_sid, 'target_st')
        if self.architecture == 'J':
            encoded = self._encode(history_sid, history_mask, history_st)
            labels = target_sid.reshape(batch_size, -1)
            assignments = None if target_st is None else target_st.reshape(batch_size, -1, self.codebook_size)
            per_list, diagnostics = self._condition_loss(encoded, labels, assignments,
                                                        mask_depths=mask_depths)
        elif self.uses_decoder_prefix:
            if mask_depths is not None:
                raise ValueError('H uses the original fixed depths 1,2,3,4')
            # History is encoded once per forward; keep its autograd graph.
            encoded = self._encode(history_sid, history_mask, history_st)
            losses, reports = [], []
            for j in range(self.list_length):
                per_item, report = self._condition_loss(
                    encoded, target_sid[:, j], None if target_st is None else target_st[:, j],
                    item_index=j, prefix_sid=target_sid[:, :j],
                    prefix_st=None if target_st is None else target_st[:, :j])
                losses.append(per_item)
                reports.append(report)
            per_list = torch.stack(losses, dim=1).mean(dim=1)
            diagnostics = {key: torch.cat([report[key] for report in reports], dim=0)
                           for key in ('mask_depths', 'ocn_order', 'masked_positions')}
            diagnostics['supervised_positions'] = sum(report['supervised_positions'] for report in reports)
            diagnostics['per_item_loss'] = torch.stack(losses, dim=1).detach()
        else:
            if mask_depths is not None:
                raise ValueError('H uses the original fixed depths 1,2,3,4')
            losses, reports = [], []
            use_st = history_st is not None or target_st is not None
            if use_st:
                hs = history_st if history_st is not None else F.one_hot(
                    history_sid.clamp_min(0), self.codebook_size).to(self.backbone.embedding.weight.dtype)
                ts = target_st if target_st is not None else F.one_hot(
                    target_sid, self.codebook_size).to(self.backbone.embedding.weight.dtype)
            for j in range(self.list_length):
                # Only earlier items enter the condition. Reuse the same ST
                # sample (and its gradient) for every occurrence of a prefix.
                hist = torch.cat((history_sid, target_sid[:, :j]), dim=1)
                mask = torch.cat((history_mask.bool(), torch.ones(
                    batch_size, j, device=history_mask.device, dtype=torch.bool)), dim=1)
                st = torch.cat((hs, ts[:, :j]), dim=1) if use_st else None
                encoded = self._encode(hist, mask, st)
                per_item, report = self._condition_loss(
                    encoded, target_sid[:, j], None if target_st is None else target_st[:, j],
                    item_index=j)
                losses.append(per_item)
                reports.append(report)
            per_list = torch.stack(losses, dim=1).mean(dim=1)
            diagnostics = {key: torch.cat([report[key] for report in reports], dim=0)
                           for key in ('mask_depths', 'ocn_order', 'masked_positions')}
            diagnostics['supervised_positions'] = sum(report['supervised_positions'] for report in reports)
            diagnostics['per_item_loss'] = torch.stack(losses, dim=1).detach()
        return {'loss': per_list.mean(), 'per_list_loss': per_list,
                'diagnostics': diagnostics}

    def forward(self, *args, **kwargs):
        return self.forward_loss(*args, **kwargs)

    def encode_context(self, history_sid, history_mask):
        """Encode a request/outer-item condition for explicit local beam reuse.

        The caller owns this tensor and reorders it with beam parent indices.
        There is no persistent model cache to survive optimizer/phase changes.
        Legacy H re-encodes each prefix. Decoder-prefix H encodes observed history
        once and passes each beam's completed SID prefix separately to decoding.
        """
        return self._encode(history_sid, history_mask)

    def decode_with_context(self, encoded, partial_sid, item_index=None, prefix_sid=None):
        """Decode against an already encoded condition using projected cross K/V."""
        if encoded.ndim != 3 or encoded.shape[-1] != self.n_embd:
            raise ValueError('encoded context must have shape [B,L,D]')
        positions = 4 * self.list_length if self.architecture == 'J' else 4
        if partial_sid.shape != (encoded.shape[0], positions):
            raise ValueError('partial_sid has the wrong target length for this route')
        if bool(((partial_sid < -1) | (partial_sid >= self.codebook_size)).any()):
            raise ValueError('partial_sid uses -1 for MASK and local values 0..C-1')
        if self.architecture == 'H':
            if item_index is None or not 0 <= int(item_index) < self.list_length:
                raise ValueError('H decoding requires item_index within the list')
        elif item_index is not None:
            raise ValueError('J decoding has no outer item_index')
        return self._decode(encoded, partial_sid, item_index=item_index, prefix_sid=prefix_sid)

    def decode_logits(self, history_sid, history_mask, partial_sid, item_index=None, prefix_sid=None):
        """Convenience uncached entry; no true target is accepted.

        Legacy H histories include the generated prefix. In decoder-prefix H,
        histories are observed-only and prefix_sid supplies completed items.
        Production beam search caches history with encode_context/decode_with_context.
        """
        encoded = self.encode_context(history_sid, history_mask)
        return self.decode_with_context(encoded, partial_sid, item_index=item_index, prefix_sid=prefix_sid)
