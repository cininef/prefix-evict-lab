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

## First observation (not a conclusion)

On the default trace (240 requests, block size 16) LRU beats LFU and the
initial cost-aware score at every cache size where eviction happens. Likely
cause: a freshly inserted tail block has `hit_count == 0`, so frequency-based
policies evict it immediately and the working set of the active session thrashes.
The cost-aware score inherits this. Next step is to confirm with per-request
diagnostics and try recency-aware or aging variants before drawing conclusions.

## Roadmap

- [ ] Diagnose LFU / cost-aware underperformance; add aging or LRU-2 style variants
- [ ] More trace shapes (RAG with shared documents, branching agents, Zipf popularity)
- [ ] Compare against SGLang RadixAttention / vLLM prefix caching behavior
- [ ] Real paged-KV engine with HF greedy parity check
- [ ] Measured TTFT on GPU
