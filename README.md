# prefix-cache-lab

Study of **prefix-cache eviction policies** for a paged KV cache under
multi-turn agent workloads. Prefix caching lets requests that share a prompt
prefix (system prompt, earlier turns) skip re-running prefill for it, which
lowers time-to-first-token (TTFT) and GPU compute. It does not change the number
of tokens in a request. When the block pool is full, which cached block to evict
decides how much of that saving survives.

Status: early. Everything here is a CPU-side simulator; no GPU or real model yet.

## What exists

- `BlockAllocator`: fixed-size block pool.
- `RadixCache`: block-granular prefix tree with ref-counted pinning; only
  unpinned leaves are evictable.
- Policies: `LRU`, `LFU`, `CostAware` (pluggable, lowest priority evicted first).
- `generate_trace`: synthetic multi-turn agent traces (shared system prompts,
  growing per-session history, interleaved sessions).
- `simulate`: replays a trace and reports token hit rate, prefill tokens
  (a TTFT proxy) and evictions.

## Run

```
pip install -e .[dev]
pytest
python benchmarks/compare_policies.py
```

## Findings so far

1. **Bug found via the benchmark.** The first run showed LFU and cost-aware far
   below LRU (e.g. 0.28 / 0.09 vs 0.57 hit rate at 128 blocks). Cause: a request's
   newly inserted blocks were not pinned, so frequency-based policies evicted
   them (0 hits) while the same request was still inserting its remaining blocks,
   leaving stale high-hit blocks in the pool. Pinning fresh blocks during insert
   fixed it (regression test in `tests/test_radix_cache.py`).
2. **After the fix** the picture is mixed: LFU / cost-aware are slightly ahead at
   64-128 blocks (0.50 vs 0.40 at 64), LRU is ahead at 256+ blocks (0.81 vs 0.76
   at 512). The current cost-aware score is not yet better than LFU.
   Single trace, single seed; no claims until multiple seeds and trace shapes.

## Roadmap

- [x] Diagnose LFU / cost-aware underperformance (unpinned fresh blocks)
- [ ] Multiple seeds with confidence intervals; add aging or LRU-2 style variants
- [ ] More trace shapes (RAG with shared documents, branching agents, Zipf popularity)
- [ ] Compare against SGLang RadixAttention / vLLM prefix caching behavior
- [ ] Real paged-KV engine with HF greedy parity check
- [ ] Measured TTFT on GPU
