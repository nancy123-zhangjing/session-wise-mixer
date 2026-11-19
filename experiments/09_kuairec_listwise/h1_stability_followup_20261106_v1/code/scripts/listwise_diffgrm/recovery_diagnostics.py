"""Read-only, deterministic process diagnostics for the R20 recovery experiment.

Call at an optimizer-update boundary. ``diagnose_catalog`` never refreshes the
training SID buffer; its CPU snapshot can be saved alongside the run checkpoint
and supplied as ``previous`` after a resume. The caller controls the 10T cadence.
No optimizer, parameter, requires-grad flag, or gradient is modified here.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import random

import numpy as np
import torch


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _restore_value(original, saved):
    """Keep top-level cache/buffer identity, including external tensor references."""
    if torch.is_tensor(original) and torch.is_tensor(saved):
        if original.shape != saved.shape:
            original.resize_(saved.shape)
        if not torch.equal(original, saved):
            original.copy_(saved)
        return original
    if isinstance(original, dict) and isinstance(saved, dict):
        for key in list(original):
            if key not in saved:
                del original[key]
        for key, value in saved.items():
            original[key] = (_restore_value(original[key], value)
                             if key in original else _clone(value))
        return original
    if isinstance(original, list) and isinstance(saved, list):
        for index, value in enumerate(saved):
            if index < len(original):
                original[index] = _restore_value(original[index], value)
            else:
                original.append(_clone(value))
        del original[len(saved):]
        return original
    if isinstance(original, tuple) and isinstance(saved, tuple):
        restored = tuple(_restore_value(old, value) for old, value in zip(original, saved))
        if len(original) == len(saved) and all(left is right for left, right in zip(original, restored)):
            return original
        return _clone(saved)
    return _clone(saved)


def _cache_attribute(name):
    # JointCrossPSE's codebook cache is a registered buffer, handled separately.
    # The real backbone additionally stores attention captures as plain attrs.
    return ('cache' in name.lower() or name in {
        'last_attention', 'capture_attention', 'past_key_values', 'past_key_value',
    })


@contextmanager
def diagnostic_context(trainer):
    """Temporarily use eval/no-grad and restore modes, buffers, caches and RNG.

    The roots are ``trainer.model`` and ``trainer.alignment``. Every child mode
    is saved individually, so mixed train/eval trees survive. Registered buffers
    (including nonpersistent caches and formal hard_codes) and ordinary cache
    attributes are restored on success or exception. This is an inference-state
    guard, not a transaction that permits parameter/optimizer updates inside it.
    """
    roots = (trainer.model, trainer.alignment)
    modules = list({id(module): module for root in roots
                    for module in root.modules()}.values())
    modes = [(module, module.training) for module in modules]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else None)
    buffers = [(module, dict(module._buffers),
                {key: _clone(value) for key, value in module._buffers.items()},
                set(module._non_persistent_buffers_set)) for module in modules]
    caches = [(module, {key: (value, _clone(value))
                        for key, value in vars(module).items()
                        if _cache_attribute(key)}) for module in modules]
    try:
        for root in roots:
            root.eval()
        with torch.no_grad():
            yield
    finally:
        # Do not call phase_mode: it changes requires_grad and clears the cache.
        try:
            with torch.no_grad():
                for module, originals, saved, nonpersistent in buffers:
                    module._buffers.clear()
                    for key, original in originals.items():
                        module._buffers[key] = _restore_value(original, saved[key])
                    module._non_persistent_buffers_set.clear()
                    module._non_persistent_buffers_set.update(nonpersistent)
                for module, saved in caches:
                    for key in list(vars(module)):
                        if _cache_attribute(key) and key not in saved:
                            delattr(module, key)
                    for key, (original, value) in saved.items():
                        setattr(module, key, _restore_value(original, value))
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def _norm(tensor):
    return float(torch.linalg.vector_norm(tensor.double()).item())


def _ratio(numerator, denominator):
    # A zero reference norm has no relative-displacement meaning. JSON null is
    # preferable to silently claiming zero or writing nonstandard NaN/Infinity.
    return numerator / denominator if denominator > 0 else None


def diagnose_catalog(trainer, previous=None, *, batch_size=2048):
    """Return ``(json_metrics, cpu_snapshot)`` without publishing temporary SIDs.

    ``snapshot`` has independent CPU ``hard_codes`` [N,M] (PAD excluded) and
    ``effective_codebooks`` [M,K,D] tensors. Compare only snapshots of the same
    fixed catalog in the same item order. Entropy is in nats; effective usage is
    exp(entropy). Collision fraction counts *all items in non-singleton buckets*,
    not N minus the number of distinct SIDs. First-snapshot change fields are null.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    with diagnostic_context(trainer):
        alignment = trainer.alignment
        mixer = alignment.mixer
        device = mixer.base_codebooks.device
        # Do not inherit BF16 autocast from the training callsite.
        with torch.autocast(device_type=device.type, enabled=False, cache_enabled=False):
            books, _ = mixer.crossed_codebooks(return_attention=False)
            books = books.detach().float()
            if books.ndim != 3 or any(size < 1 for size in books.shape):
                raise ValueError('effective codebooks must be nonempty [digits,codes,dim]')
            if not bool(torch.isfinite(books).all()):
                raise ValueError('effective codebooks contain nonfinite values')
            digits, size, dimension = books.shape
            slots = alignment.item_slots
            if slots.ndim != 3 or tuple(slots.shape[1:]) != (digits, dimension):
                raise ValueError('item_slots must have matching [PAD+items,digits,dim] shape')
            count = len(slots) - 1
            if count < 1:
                raise ValueError('catalog must contain at least one real item after PAD')
            codes, total_sse = [], 0.0
            for start in range(1, len(slots), batch_size):
                batch = slots[start:start + batch_size].to(device=device, dtype=torch.float32)
                if not bool(torch.isfinite(batch).all()):
                    raise ValueError('catalog item slots contain nonfinite values')
                distances = mixer.squared_distances(batch, books)
                if not bool(torch.isfinite(distances).all()):
                    raise ValueError('catalog distances contain nonfinite values')
                chosen = distances.argmin(dim=-1)
                reconstructed = mixer.gather_codes(books, chosen)
                # Sum the FP32-coordinate errors in FP64 for stable aggregation.
                total_sse += float((batch.double() - reconstructed.double()).square().sum().item())
                codes.append(chosen.detach().cpu())
            hard = torch.cat(codes).clone()
            effective = books.cpu().clone()
            base = mixer.base_codebooks.detach().float().cpu()
            initial = mixer.initial_codebooks.detach().float().cpu()

        per_slot = []
        for digit in range(digits):
            frequency = torch.bincount(hard[:, digit], minlength=size)
            probability = frequency.double() / count
            positive = probability[frequency > 0]
            entropy = float(-(positive * positive.log()).sum().item())
            per_slot.append(dict(slot=digit, used_code_count=int((frequency > 0).sum()),
                counts=frequency.tolist(), frequencies=probability.tolist(), entropy_nats=entropy,
                normalized_entropy=entropy / math.log(size) if size > 1 else 0.0,
                effective_code_count=math.exp(entropy)))
        _, bucket_sizes = torch.unique(hard, dim=0, return_counts=True)
        collision_items = int(bucket_sizes[bucket_sizes > 1].sum().item())
        previous_change = dict(sid_item_change_fraction=None, sid_digit_change_fraction=None,
            sid_change_fraction_per_slot=None, effective_displacement_norm=None,
            effective_displacement_over_previous_global_norm=None)
        if previous is not None:
            old_codes = previous['hard_codes'].detach().cpu()
            old_books = previous['effective_codebooks'].detach().float().cpu()
            if old_codes.shape != hard.shape or old_books.shape != effective.shape:
                raise ValueError('previous snapshot shape does not match this catalog/codebook')
            if old_codes.dtype != hard.dtype or bool(((old_codes < 0) | (old_codes >= size)).any()):
                raise ValueError('previous snapshot has invalid hard code indices')
            if not bool(torch.isfinite(old_books).all()):
                raise ValueError('previous snapshot contains nonfinite effective codebooks')
            changed = hard.ne(old_codes)
            displacement = _norm(effective.double() - old_books.double())
            previous_change = dict(
                sid_item_change_fraction=float(changed.any(dim=-1).double().mean()),
                sid_digit_change_fraction=float(changed.double().mean()),
                sid_change_fraction_per_slot=changed.double().mean(dim=0).tolist(),
                effective_displacement_norm=displacement,
                effective_displacement_over_previous_global_norm=_ratio(displacement, _norm(old_books)))
        base_norm, initial_norm = _norm(base), _norm(initial)
        metrics = dict(schema_version=1, scope='full_catalog_excluding_pad',
            selection='fp32_deterministic_nearest_centroid', num_items=count,
            n_digit=digits, codebook_size=size,
            r_updates=int(trainer.r_updates), t_updates=int(trainer.t_updates),
            per_slot=per_slot, used_codes_per_slot=[row['used_code_count'] for row in per_slot],
            unique_sid_count=int(len(bucket_sizes)), collision_item_count=collision_items,
            collision_item_fraction=collision_items / count,
            collision_item_percent=100.0 * collision_items / count,
            duplicate_row_count=count - len(bucket_sizes), largest_bucket_size=int(bucket_sizes.max()),
            mean_nearest_reconstruction_sse=total_sse / count,
            effective_codebook_norm=_norm(effective), base_codebook_norm=base_norm,
            mixer_delta_over_base_global_norm=_ratio(_norm(effective.double() - base.double()), base_norm),
            base_displacement_over_initial_global_norm=_ratio(_norm(base.double() - initial.double()), initial_norm),
            effective_displacement_over_initial_global_norm=_ratio(
                _norm(effective.double() - initial.double()), initial_norm),
            previous_change=previous_change)
        snapshot = dict(hard_codes=hard, effective_codebooks=effective)
        return metrics, snapshot
