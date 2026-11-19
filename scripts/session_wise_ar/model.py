"""Native DIGER recommender and paper-configured TIGER T5 wrapper."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Attention
from .upstream import modules


class ARRecommender(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.family = cfg, cfg["family"]
        m, k = cfg["model"], cfg["rq"]["codes"]
        self.codes = k
        self.user_offset = 1 + 4*k
        vocab = self.user_offset + cfg["model"]["user_buckets"] if self.family == "tiger" else 1
        config = T5Config(d_model=m["width"], d_kv=m["d_kv"], d_ff=m["ffn"],
                          num_layers=m["encoder_layers"], num_decoder_layers=m["decoder_layers"],
                          num_heads=m["heads"], dropout_rate=m["dropout"], vocab_size=vocab,
                          feed_forward_proj="relu", decoder_start_token_id=0, pad_token_id=0,
                          eos_token_id=None, relative_attention_num_buckets=32,
                          relative_attention_max_distance=128, layer_norm_epsilon=1e-6,
                          tie_word_embeddings=True, use_cache=False)
        config._attn_implementation = "eager"
        backbone = T5ForConditionalGeneration(config)
        attention = m.get("attention", "eager")
        if attention not in ("eager", "sdpa"):
            raise ValueError("model.attention must be eager or sdpa")
        if attention == "sdpa":
            use_sdpa_attention(backbone)
        if self.family == "diger":
            native_config = diger_config(cfg)
            self.native = modules().model.Model(native_config, backbone,
                                                cfg["catalog_size"]+1, code_length=4, code_number=k)
        else:
            self.native = backbone

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Upstream caches this plain attribute at construction on CPU.
        # Keep its temporary allocations on the device of its actual weights.
        if self.family == "diger":
            self.native.device = self.native.model.shared.weight.device
        return result

    def global_tokens(self, sid):
        return sid + torch.arange(4, device=sid.device)*self.codes + 1

    def history_embeddings(self, context, sid, users):
        mask = context.ne(0)
        local = sid[context].clamp_min(0)
        if self.family == "diger":
            flat = local.reshape(len(context), -1)
            expanded = mask.repeat_interleave(4, dim=1)
            return self.native.get_input_embeddings(flat.clone(), expanded), expanded
        ids = self.global_tokens(local).reshape(len(context), -1)
        expanded = mask.repeat_interleave(4, dim=1)
        ids = ids.masked_fill(~expanded, 0)
        ids = torch.cat((users[:, None] + self.user_offset, ids), dim=1)
        expanded = torch.cat((torch.ones_like(users[:, None], dtype=torch.bool), expanded), dim=1)
        return self.native.shared(ids), expanded

    def hard_loss(self, batch, sid):
        context, target, users = batch["context"], batch["target"], batch["users"]
        embeddings, mask = self.history_embeddings(context, sid, users)
        labels = sid[target]
        if self.family == "tiger":
            return self.native(inputs_embeds=embeddings, attention_mask=mask,
                               labels=self.global_tokens(labels), use_cache=False).loss
        outputs = self.native(inputs_embeds=embeddings, attention_mask=mask, labels=labels)
        return F.cross_entropy(outputs.logits.reshape(-1, self.codes), labels.reshape(-1))

    def encode(self, context, sid, users):
        embeddings, mask = self.history_embeddings(context, sid, users)
        encoder = self.native.get_encoder()
        encoded = encoder(inputs_embeds=embeddings, attention_mask=mask, return_dict=True)
        return encoded.last_hidden_state, mask

    def next_logprobs(self, hidden, mask, local_prefix):
        start = torch.zeros((len(hidden), 1), dtype=torch.long, device=hidden.device)
        encoded = BaseModelOutput(last_hidden_state=hidden)
        if self.family == "diger":
            decoder = torch.cat((start, local_prefix), dim=1)
            logits = self.native(encoder_outputs=encoded, attention_mask=mask,
                                 decoder_input_ids=decoder).position_logits[-1]
            return F.log_softmax(logits.float(), dim=-1)
        prefix = local_prefix + torch.arange(local_prefix.shape[1], device=hidden.device)*self.codes + 1
        decoder = torch.cat((start, prefix), dim=1)
        logits = self.native(encoder_outputs=encoded, attention_mask=mask,
                             decoder_input_ids=decoder, use_cache=False).logits[:, -1].float()
        # Normalize over the ORIGINAL global vocabulary, then filter this digit.
        logp = F.log_softmax(logits, dim=-1)
        offset = 1 + local_prefix.shape[1]*self.codes
        return logp[:, offset:offset+self.codes]


def diger_config(cfg):
    import yaml
    from .upstream import verify
    native = yaml.safe_load((verify()/"config/beauty_gradient_fix.yaml").read_text())
    native.update(seed=cfg["seed"], use_constrained_kmeans=False, semantic_hidden_size=cfg["rq"]["input_dim"],
                  e_dim=cfg["rq"]["latent_dim"], layers=cfg["rq"]["hidden_dims"],
                  num_emb_list=[cfg["rq"]["codes"]]*3, code_num=cfg["rq"]["codes"],
                  sk_epsilons=[cfg["rq"]["sinkhorn_epsilon"]]*3,
                  sk_iters=cfg["rq"]["sinkhorn_iterations"], num_beams=cfg["evaluation"]["beam"],
                  max_length=cfg["model"]["input_max"], valid_metric="ndcg@5")
    return native


class SdpaT5Attention(T5Attention):
    """T5 attention whose TRAINING path uses fused scaled_dot_product_attention.

    Same parameters and the same mathematics as eager T5 (no 1/sqrt(d) scaling,
    learned relative position bias plus the additive padding mask, both added to
    the logits before softmax), but without materializing the attention matrix.
    Evaluation (module.eval()), caches, head masks and attention outputs fall
    back to the unmodified eager implementation, so validation/test decoding
    keeps exactly the eager kernels."""

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None,
                past_key_values=None, layer_head_mask=None, query_length=None, use_cache=False,
                output_attentions=False, cache_position=None):
        if (not self.training or past_key_values is not None or use_cache or output_attentions
                or layer_head_mask is not None or self.pruned_heads):
            return super().forward(hidden_states, mask=mask, key_value_states=key_value_states,
                                   position_bias=position_bias, past_key_values=past_key_values,
                                   layer_head_mask=layer_head_mask, query_length=query_length,
                                   use_cache=use_cache, output_attentions=output_attentions,
                                   cache_position=cache_position)
        batch_size, seq_length = hidden_states.shape[:2]
        source = hidden_states if key_value_states is None else key_value_states

        def heads(x):
            return x.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        query, key, value = heads(self.q(hidden_states)), heads(self.k(source)), heads(self.v(source))
        if position_bias is None:
            key_length = key.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros((1, self.n_heads, seq_length, key_length),
                                            device=query.device, dtype=query.dtype)
            else:
                real_length = (query_length if query_length is not None else
                               cache_position[-1] + 1 if cache_position is not None else seq_length)
                position_bias = self.compute_bias(real_length, key_length, device=query.device,
                                                  cache_position=cache_position)
                position_bias = position_bias[:, :, -seq_length:, :]
            if mask is not None:
                position_bias = position_bias + mask[:, :, :, :key_length]
        # Fused kernels need a unit stride on the key axis. T5's bias comes from a
        # permute (heads innermost), and broadcasting with the mask keeps that layout.
        bias = position_bias.to(query.dtype).contiguous().expand(batch_size, -1, seq_length, -1)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=bias,
                                                dropout_p=self.dropout if self.training else 0.0, scale=1.0)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        return (self.o(output), position_bias)


def use_sdpa_attention(backbone):
    """Swap every T5Attention to the SDPA training path in place (state_dict unchanged)."""
    count = 0
    for module in backbone.modules():
        if type(module) is T5Attention:
            module.__class__ = SdpaT5Attention
            count += 1
    if not count:
        raise ValueError("No T5 attention modules found")
    return count
