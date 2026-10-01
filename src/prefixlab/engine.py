"""Prefix-reuse generation on a real HF causal LM.

The KV tensors for every cached block live in a `KVPool` indexed by the same
block ids that `RadixCache` hands out, so the tree, the eviction policy and the
real KV memory stay in lockstep. A request only runs prefill on the uncached
suffix; the cached prefix is gathered from the pool into a `DynamicCache`.

Requires the optional `torch` and `transformers` dependencies.
"""

import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from .radix_cache import RadixCache


class KVPool:
    """K and V tensors of shape [layers, num_blocks, kv_heads, block_size, head_dim]."""

    def __init__(self, config, num_blocks: int, block_size: int, dtype, device):
        layers = config.num_hidden_layers
        kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (layers, num_blocks, kv_heads, block_size, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size

    def gather(self, blocks: list[int]) -> DynamicCache:
        """Build a batch-1 cache holding the given blocks, concatenated in order."""
        cache = DynamicCache()
        idx = torch.tensor(blocks, device=self.k.device)
        for layer in range(self.k.shape[0]):
            # [n, H, bs, D] -> [1, H, n*bs, D]
            k = self.k[layer, idx].permute(1, 0, 2, 3).flatten(1, 2).unsqueeze(0)
            v = self.v[layer, idx].permute(1, 0, 2, 3).flatten(1, 2).unsqueeze(0)
            cache.update(k, v, layer)
        return cache

    def scatter(self, cache: DynamicCache, start_block: int, blocks: list[int]) -> None:
        """Copy token range [start_block*bs, ...) of `cache` into the given pool blocks."""
        bs = self.block_size
        for layer, cl in enumerate(cache.layers):
            for i, b in enumerate(blocks):
                lo = (start_block + i) * bs
                self.k[layer, b] = cl.keys[0, :, lo : lo + bs]
                self.v[layer, b] = cl.values[0, :, lo : lo + bs]


@dataclass
class GenResult:
    tokens: list[int]          # generated token ids
    prompt_len: int
    cached_tokens: int         # prompt tokens served from the pool
    prefill_seconds: float     # wall time of the suffix forward pass
    gather_seconds: float = 0.0  # wall time copying the cached prefix out of the pool

    @property
    def ttft_seconds(self) -> float:
        return self.gather_seconds + self.prefill_seconds


class PrefixEngine:
    def __init__(self, model, cache: RadixCache):
        self.model = model.eval()
        self.cache = cache
        p = next(model.parameters())
        self.pool = KVPool(model.config, cache.allocator.num_blocks, cache.block_size,
                           p.dtype, p.device)
        self.device = p.device

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new_tokens: int = 16) -> GenResult:
        bs = self.cache.block_size
        matched = self.cache.match_prefix(prompt)
        # Keep at least one prompt token uncached so the forward pass yields logits.
        matched = matched[: (len(prompt) - 1) // bs]
        self.cache.lock(matched)
        try:
            n_cached = len(matched) * bs
            self._sync()
            tg = time.perf_counter()
            past = self.pool.gather([n.block for n in matched]) if matched else DynamicCache()
            self._sync()
            t0 = time.perf_counter()
            gather_s = t0 - tg
            suffix = torch.tensor([prompt[n_cached:]], device=self.device)
            out = self.model(input_ids=suffix, past_key_values=past, use_cache=True)
            self._sync()
            prefill_s = time.perf_counter() - t0

            # Store the prompt's new full blocks before decode grows the cache further.
            fresh = self.cache.insert_nodes(prompt, matched)
            if fresh:
                self.pool.scatter(out.past_key_values, len(matched), [n.block for n in fresh])

            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            generated = [nxt]
            for _ in range(max_new_tokens - 1):
                out = self.model(input_ids=torch.tensor([[nxt]], device=self.device),
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                generated.append(nxt)
        finally:
            self.cache.unlock(matched)
        return GenResult(generated, len(prompt), n_cached, prefill_s, gather_s)
