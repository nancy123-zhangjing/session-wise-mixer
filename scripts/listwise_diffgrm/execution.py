"""Opt-in execution-only acceleration; no model math, sampling or phase changes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch


def enabled(config):
    return config.get('execution', {}).get('optimized_io', False)


class ValidatedTrainingBatch(dict):
    """Internal batch produced only from a fully validated static item-ID cache."""


class TrainingTensorCache:
    fields = ('history_item_ids', 'history_mask', 'target_item_ids')

    def __init__(self, data, chunk_size=2048, *, vectorized=False, device=None):
        self.count = len(data)
        self.tensors = {}
        if vectorized:
            self._build_vectorized(data)
        else:
            collate = data.collator()
            # Use the original collator, including ALL timestamp/identity checks,
            # once at startup. Never cache SID, assignments or learned embeddings.
            for start in range(0, self.count, chunk_size):
                batch = collate(data[i] for i in range(start, min(start + chunk_size, self.count)))
                for key in self.fields:
                    value = batch[key]
                    if key not in self.tensors:
                        self.tensors[key] = torch.empty((self.count, *value.shape[1:]), dtype=value.dtype)
                    self.tensors[key][start:start + len(value)].copy_(value)
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        if self.device.type != 'cpu':
            self.tensors = {k: v.to(self.device) for k, v in self.tensors.items()}

    def _build_vectorized(self, data):
        from types import SimpleNamespace
        total, combined = self.count, {}
        for start in range(0, total, 4096):
            self.count = min(4096, total - start)
            self._build_vectorized_block(SimpleNamespace(table=data.table.slice(start, self.count),
                config=data.config, n_items=data.n_items))
            for key, value in self.tensors.items():
                if key not in combined:
                    combined[key] = torch.empty((total, *value.shape[1:]), dtype=value.dtype)
                combined[key][start:start + self.count].copy_(value)
        self.count, self.tensors = total, combined

    def _build_vectorized_block(self, data):
        """Arrow offsets -> dense tensors with the collator's checks done as array ops."""
        import numpy as np
        table = data.table.combine_chunks()
        h, length, n_items = int(data.config['history_max']), int(data.config['list_length']), data.n_items
        n = self.count
        dense = {}
        lengths = None
        for name in ('history_item_ids', 'history_timestamp_us'):
            col = table[name].chunk(0)
            offsets = col.offsets.to_numpy()
            lens = np.diff(offsets)
            if np.any(lens < 1) or np.any(lens > h):
                raise ValueError('History exceeds its locked length or is empty; never silently truncate')
            if lengths is not None and not np.array_equal(lens, lengths):
                raise ValueError('Item/timestamp length mismatch')
            lengths = lens
            out = np.zeros((n, h), dtype=np.int64)
            rows = np.repeat(np.arange(n), lens)
            cols = np.arange(lens.sum()) - np.repeat(offsets[:-1] - offsets[0], lens) + np.repeat(h - lens, lens)
            out[rows, cols] = col.values.to_numpy()[offsets[0]:offsets[-1]]
            dense[name] = out
        targets = {}
        for name in ('target_item_ids', 'target_timestamp_us', 'target_mask'):
            col = table[name].chunk(0)
            offsets = col.offsets.to_numpy()
            if not np.all(np.diff(offsets) == length):
                raise ValueError('Targets must be the locked number of distinct items')
            values = col.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
            targets[name] = np.asarray(values).reshape(n, length)
        target, tt = targets['target_item_ids'], targets['target_timestamp_us']
        mask = np.arange(h)[None, :] >= (h - lengths)[:, None]
        hist, ht = dense['history_item_ids'], dense['history_timestamp_us']
        if not targets['target_mask'].all():
            raise ValueError('Prepared lists must contain every target; PAD targets are unsupported')
        if not data.config.get('allow_repeated_targets', False) and np.any(np.diff(np.sort(target, axis=1), axis=1) == 0):
            raise ValueError('Targets must be the locked number of distinct items')
        if (np.any(target <= 0) or np.any(target > n_items) or np.any(hist[mask] <= 0)
                or np.any(hist > n_items)):
            raise ValueError('Real item IDs must be in 1..N; PAD is not an item')
        if (np.any(np.diff(tt, axis=1) <= 0) or np.any(ht[:, -1] >= tt[:, 0])
                or np.any((np.diff(ht, axis=1) <= 0) & mask[:, :-1])):
            raise ValueError('History must be strictly earlier than the ordered target list')
        # Compact immutable IDs; convert only selected batches to embedding-index dtype.
        self.tensors = {'history_item_ids': torch.from_numpy(hist.astype(np.int32)),
                        'history_mask': torch.from_numpy(mask),
                        'target_item_ids': torch.from_numpy(np.ascontiguousarray(target, dtype=np.int32))}

    def select(self, indices, *, pin=False):
        index = torch.as_tensor(indices, dtype=torch.long)
        if index.ndim != 1 or (len(index) and (index.min() < 0 or index.max() >= self.count)):
            raise ValueError('Training indices outside static data')
        result = ValidatedTrainingBatch()
        if self.device.type != 'cpu':
            index = index.to(self.device, non_blocking=True)
            for key, source in self.tensors.items():
                result[key] = source.index_select(0, index)
                if result[key].dtype == torch.int32:
                    result[key] = result[key].long()
            return result
        for key, source in self.tensors.items():
            out = torch.empty((len(index), *source.shape[1:]), dtype=source.dtype, pin_memory=pin)
            torch.index_select(source, 0, index, out=out)
            result[key] = out.long() if out.dtype == torch.int32 else out
        return result


@contextmanager
def prefetched_updates(cache, indices, effective_size, micro_size, start_update, device):
    """One ordered producer, at most one future effective batch; no RNG use.

    Producer owns CPU packing and H2D copies only. All model/Tokenizer operations
    stay on the caller thread/stream in their original order. CUDA stream events
    and record_stream protect both transfer dependencies and tensor lifetime.
    A device-resident cache needs no producer: batches are gathered on the GPU.
    """
    device = torch.device(device)
    if cache.device.type != 'cpu':
        def resident():
            for update in range(start_update, (len(indices) + effective_size - 1) // effective_size):
                values = cache.select(indices[update * effective_size:(update + 1) * effective_size])
                count = len(values['target_item_ids'])
                yield update, [ValidatedTrainingBatch({k: v[s:s + micro_size] for k, v in values.items()})
                               for s in range(0, count, micro_size)]
        items = resident()
        try:
            yield items
        finally:
            items.close()
        return
    cuda = device.type == 'cuda'
    stream = torch.cuda.Stream(device=device) if cuda else None
    positions = iter(range(start_update, (len(indices) + effective_size - 1) // effective_size))

    def prepare(update):
        selected = indices[update * effective_size:(update + 1) * effective_size]
        host = cache.select(selected, pin=cuda)
        if not cuda:
            return update, host, host, None
        with torch.cuda.device(device), torch.cuda.stream(stream):
            values = ValidatedTrainingBatch({k: v.to(device, non_blocking=True) for k, v in host.items()})
            done = torch.cuda.Event()
            done.record(stream)
        return update, host, values, done

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='item-id-prefetch')

    def iterator():
        first = next(positions, None)
        future = executor.submit(prepare, first) if first is not None else None
        while future is not None:
            update, host, values, done = future.result()
            if cuda:
                consumer = torch.cuda.current_stream(device)
                consumer.wait_event(done)
                for value in values.values():
                    value.record_stream(consumer)
            following = next(positions, None)
            future = executor.submit(prepare, following) if following is not None else None
            count = len(values['target_item_ids'])
            batches = [ValidatedTrainingBatch({k: v[s:s + micro_size] for k, v in values.items()})
                       for s in range(0, count, micro_size)]
            # Keep pinned sources alive until the asynchronous H2D transfer ends.
            yield update, batches
            if done is not None:
                done.synchronize()

    items = iterator()
    try:
        yield items
    finally:
        items.close()
        executor.shutdown(wait=True, cancel_futures=True)
        if stream is not None:
            stream.synchronize()


class TrainingLog:
    def __init__(self, path, buffered):
        self.path, self.buffered, self.stream = path, buffered, None

    def __enter__(self):
        if self.buffered:
            self.stream = self.path.open('a', encoding='utf-8', buffering=65536)
        return self

    def write(self, text):
        if self.stream is None:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(text)
        else:
            self.stream.write(text)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()
