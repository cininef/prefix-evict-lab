# prefix-evict-lab

[![tests](https://github.com/cininef/prefix-evict-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cininef/prefix-evict-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-research%20prototype-orange)

**How much of prefix caching's prefill saving survives memory pressure, and which
eviction policy keeps the most of it, for multi-turn agent and RAG workloads?**

Prefix caching lets requests that share a prompt prefix (system prompt, tool
schemas, retrieved documents, earlier turns) reuse the KV cache of that prefix
instead of recomputing it in prefill. This lowers time-to-first-token (TTFT) and
GPU compute. It does not reduce the number of tokens in a request.

The KV block pool is finite. When it is full, something must be evicted, and that
choice decides how much of the saving is kept. This repo isolates that choice: a
block-level prefix tree over a paged KV pool, pluggable eviction policies, an
offline Belady oracle as an upper bound, and multi-seed evaluation on synthetic
agent traces.

## Key findings

Token hit rate vs. cache size on a synthetic multi-turn agent trace (10 seeds,
error bars are one standard deviation; block size 16 tokens; working set is about
1000 blocks).

![hit rate vs cache size](docs/hit_rate_vs_cache.png)

| Blocks | LRU | LFU | CostAware (v1) | Belady oracle |
|---:|---:|---:|---:|---:|
| 64  | 0.407 | 0.508 | 0.506 | 0.538 |
| 128 | 0.575 | 0.578 | 0.576 | 0.705 |
| 256 | 0.698 | 0.642 | 0.634 | 0.807 |
| 512 | 0.814 | 0.755 | 0.741 | 0.859 |

1. **LRU is not always the best policy.** LFU is ahead when the cache is small
   (64 blocks); LRU is ahead once it holds a quarter of the working set or more.
   The crossover is around 128 blocks. Shared heads (system prompts) reward
   frequency; private tails (the latest turn of a live session) reward recency.
   No single signal serves both.
2. **There is headroom.** The oracle beats the best online policy by up to about
   13 points (128 blocks: 0.705 vs 0.578).
3. **A guessed cost term does not help.** The first cost-aware score
   (`1 + 0.1 * depth`) is indistinguishable from LFU. A useful cost model has to
   come from measured recompute time.

**A bug the benchmark exposed.** An earlier version did not pin a request's newly
inserted blocks, so frequency-based policies evicted them mid-insert. LFU and
cost-aware looked far worse than LRU (0.28 / 0.09 vs 0.57 at 128 blocks) until
this was fixed. The regression test is
`test_insert_does_not_evict_its_own_new_blocks`.

## Architecture

```
                 ┌────────────────────────────────────────────┐
  workloads      │ trace.py: synthetic multi-turn agent trace  │
  (B)            │ (planned: RAG-Zipf, branching, real logs)   │
                 └───────────────────┬────────────────────────┘
                                     │ list of token-id prompts
                                     ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ simulator.py (C): for each request                              │
 │   match_prefix -> lock -> insert uncached blocks -> unlock       │
 │   records hit tokens, prefill tokens, evictions, eviction age    │
 └───────────────┬──────────────────────────────┬─────────────────┘
                 ▼                              ▼
   ┌───────────────────────────┐    ┌──────────────────────────────┐
   │ radix_cache.py (A)        │    │ policies.py / oracle.py      │
   │ block-level prefix tree   │◄───│ LRU, LFU, CostAware, Belady  │
   │ ref-count pinning         │    │ priority(node, now); lowest   │
   │ leaf-only eviction        │    │ priority is evicted first     │
   └─────────────┬─────────────┘    └──────────────────────────────┘
                 ▼
   ┌───────────────────────────┐
   │ block_allocator.py (A)    │   fixed pool of KV blocks
   └───────────────────────────┘

 evaluate.py: sweep policies x cache sizes x seeds -> mean and std
 (D, planned) real-model engine: the same cache holds actual KV tensors
```

Design notes:

- The tree is **block-granular**: each node owns one block of `block_size` tokens
  keyed by those token ids, and a root-to-node path is a cached prefix. Only full
  blocks are cached. This is simpler than SGLang's compressed radix tree and
  closer to vLLM's block hashing. It is not the same data structure.
- Only **unpinned leaves** are evictable, so a cached prefix is never left with a
  hole in the middle. A request pins its matched path and its own new blocks
  while it is being inserted.
- A policy is one method, `priority(node, now)`. The lowest priority is evicted
  first, so a new policy is a small class.

## Quickstart

```bash
git clone https://github.com/cininef/prefix-evict-lab
cd prefix-evict-lab
pip install -e ".[dev]"
pytest                                   # 19 tests
python benchmarks/compare_policies.py    # one trace, several cache sizes
python benchmarks/multi_seed.py          # 10 seeds + oracle, writes the figure
```

Add a policy:

```python
class MyPolicy:
    name = "mine"
    def priority(self, node, now):        # node has last_access, hit_count, depth, created
        return (node.hit_count, node.last_access)
```

## Roadmap and future investigations

The simulator is complete for offline evaluation. Future work bridges the gap
between logical cache behavior and serving latency:

- **End-to-end latency.** Put a real causal LM (for example Qwen2.5-0.5B) behind
  the cache, check greedy-decoding parity with plain generation, and measure real
  TTFT so hit rate can be converted into latency.
- **Broader workloads.** RAG with Zipf-popular documents, branching and
  multi-agent trees, long-context coding agents, and tool-calling agents whose
  sessions pause while waiting for a tool. The pause case is an open hypothesis:
  a policy that models expected resume time may keep blocks that LRU evicts.
- **Structure-aware eviction.** Use tree topology (depth, sharing degree, subtree
  size) and measured recompute cost as signals; try segmented (SLRU/2Q-style)
  and adaptive recency/frequency policies, and measure how much of the gap to the
  oracle they close.

Tiered KV offload, non-prefix KV reuse for RAG chunks, prefix-aware routing and
prefill/decode disaggregation are related directions that change what "evict"
means; they are out of scope here.

## Limitations

- All numbers come from a **simulator on a synthetic trace**. Hit rate is not TTFT.
- The trace generator is my own; real agent traffic may differ (no Zipf
  popularity, no tool-call pauses yet).
- The tree is block-granular, not SGLang's compressed radix tree.
- The oracle is an upper-bound reference. Because eviction is leaf-only, it is a
  strong heuristic bound and not a proven optimum.
- No comparison against vLLM or SGLang yet.

## Related work

- PagedAttention / vLLM (Kwon et al., 2023): paged KV blocks, automatic prefix caching.
- RadixAttention / SGLang (Zheng et al., 2023): radix-tree prefix reuse with LRU eviction.
- Belady's algorithm (1966): the offline optimal replacement policy, used here as the oracle.

Later work on KV reuse for RAG, KV tiering and prefix-aware scheduling exists. I
list only the references I am sure of and do not cite the rest from memory.

## Repository layout

```
src/prefixlab/   block_allocator, radix_cache, policies, oracle, trace, simulator, evaluate
tests/           19 unit tests
benchmarks/      compare_policies.py, multi_seed.py
docs/            hit_rate_vs_cache.png
```

## License

MIT
