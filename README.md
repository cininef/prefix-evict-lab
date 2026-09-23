# prefix-cache-lab

**How much of prefix caching's prefill saving survives memory pressure, and which
eviction policy keeps the most of it, for multi-turn agent and RAG workloads?**

![status](https://img.shields.io/badge/status-early%20research%20prototype-orange)

Prefix caching lets requests that share a prompt prefix (system prompt, tool
schemas, retrieved documents, earlier turns) reuse the KV cache of that prefix
instead of recomputing it in prefill. This lowers time-to-first-token (TTFT) and
GPU compute. It does **not** reduce the number of tokens in a request.

The KV block pool is finite. When it is full, something must be evicted, and
that choice decides how much of the saving is kept. This repo studies that choice
in a controlled setting: a block-level prefix tree over a paged KV pool,
pluggable eviction policies, synthetic and (later) real workloads, an offline
oracle upper bound, and finally a real model to measure TTFT.

> **Status.** The simulator, policies, oracle and multi-seed evaluation are done
> (milestones M0-M1). Real-model integration, latency measurement, more workloads
> and a new policy are planned (M2-M7). Nothing here is a benchmark against
> SGLang or vLLM yet. See [Limitations](#limitations).

## Results so far

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

What this shows:

1. **No single policy wins.** LFU is ahead when the cache is small (64 blocks);
   LRU is ahead once it holds a quarter of the working set or more. The crossover
   is around 128 blocks.
2. **There is headroom.** The oracle beats the best online policy by up to about
   13 points (128 blocks: 0.705 vs 0.578). A better policy is not ruled out.
3. **The first cost-aware score adds nothing over LFU.** Its cost term
   (`1 + 0.1 * depth`) was a guess. Cost-aware v2 (milestone M5) will use measured
   recompute time instead.

A bug worth knowing about, found through this benchmark: an earlier version did
not pin a request's newly inserted blocks, so frequency-based policies evicted
them mid-insert. LFU and cost-aware looked far worse than LRU (0.28 / 0.09 vs
0.57 at 128 blocks) until this was fixed. The regression test is
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
git clone https://github.com/cininef/prefix-cache-lab
cd prefix-cache-lab
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

## Plan

The full module map is in [docs/ROADMAP.md](docs/ROADMAP.md). Summary:

| # | Milestone | Done when | Status |
|---|---|---|---|
| M0 | Cache core and simulator | tests pass, benchmark runs | done |
| M1 | Rigorous simulation | multi-seed error bars, oracle bound, figure | done |
| M2 | Real model, correctness | paged prefix-reuse generation matches HuggingFace greedy output token for token | next |
| M3 | Latency ground truth | measured prefill time vs. cached-prefix length; fitted latency model | planned |
| M4 | More workloads | RAG and agent traces below; check whether the M1 ranking holds | planned |
| M5 | Better policy | a policy that beats LRU and LFU across sizes, or a documented failure | planned |
| M6 | End-to-end TTFT | policies compared on the real model under a request stream | planned |
| M7 | Write-up | results, limits, related work, negative results included | planned |

### M2-M3: real model

Wrap a small HuggingFace causal LM (for example Qwen2.5-0.5B) that runs on CPU or
Apple MPS. Store per-block KV tensors in the pool, prefill only the uncached
suffix of a prompt, and require token-for-token agreement with plain
`generate()` under greedy decoding. Then measure prefill time as a function of
cached-prefix length and total length, and fit a latency model so the simulator
can report estimated TTFT and not only hit rate.

### M4: workloads beyond a single chat

Hit rate depends heavily on the workload, so the ranking above must be re-tested:

- **RAG with popular documents.** A few documents are retrieved by many requests
  (Zipf popularity). The shared prefix is the document, placed before the
  question.
- **Tool-calling agent loops (ReAct style).** A long system prompt plus tool
  schemas, then many short turns. The KV cache of a paused agent sits idle while
  it waits for a tool result. **Hypothesis to test:** eviction should use the
  expected resume time of a session, not only past access, so a policy aware of
  tool-call pauses should keep the right blocks. This is the main new idea in
  the plan and it may turn out to be wrong.
- **Branching agents and multi-agent systems.** One prefix forks into several
  sub-tasks or several agents that share a large context and then diverge.
- **Long-context coding agents.** A very long, slowly growing context per
  session. Recompute cost is high, so cost-aware eviction should matter most here.
- **Reasoning models.** Long decode phases hold KV blocks for a long time, which
  shrinks the pool available to prefix caching.
- **Real logs** (for example public chat or agent datasets) where usable, to
  check that conclusions are not an artifact of the synthetic generator.

### M5: policies to try

Aging and LRU-2 style variants, ARC-style adaptive recency/frequency, cost-aware
v2 (weight by measured recompute time and estimated reuse probability), and the
pause-aware policy above. Ablate each signal: recency, frequency, depth, cost,
expected resume time.

### Related directions, not in scope yet

Tiered KV storage (offloading evicted blocks to CPU or disk instead of dropping
them), non-prefix KV reuse for RAG chunks, prefix-aware request routing across
replicas, and prefill/decode disaggregation. These change what "evict" means, so
they are natural follow-ups.

## Related work

- PagedAttention / vLLM (Kwon et al., 2023): paged KV blocks, automatic prefix caching.
- RadixAttention / SGLang (Zheng et al., 2023): radix-tree prefix reuse with LRU eviction.
- Belady's algorithm (1966): the offline optimal replacement policy, used here as the oracle.
- Follow-up systems on KV reuse for RAG, KV tiering and prefix-aware scheduling exist;
  the reference list will be added and checked in M7.

## Limitations

- All numbers are from a **simulator on a synthetic trace**. Hit rate is not TTFT
  until M3 and M6.
- The trace generator is my own; real agent traffic may differ (no Zipf
  popularity, no tool-call pauses yet).
- The tree is block-granular, not SGLang's compressed radix tree.
- The oracle is an upper-bound reference. Because eviction is leaf-only, it is a
  strong heuristic bound and not a proven optimum.
- No comparison against vLLM or SGLang yet.

## Repository layout

```
src/prefixlab/   block_allocator, radix_cache, policies, oracle, trace, simulator, evaluate
tests/           19 unit tests
benchmarks/      compare_policies.py, multi_seed.py
docs/            ROADMAP.md, hit_rate_vs_cache.png
```

## License

MIT
