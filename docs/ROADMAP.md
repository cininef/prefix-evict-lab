# Project overview and milestones

**Question.** Under multi-turn agent / RAG workloads, how much prefill work does
prefix caching save, and how much of that survives when the KV block pool is
under memory pressure, as a function of the eviction policy?

**Claim to earn** (not yet earned): a workload-aware policy closes part of the gap
between LRU and the offline oracle, and the saving shows up as lower TTFT on a
real model, not just as a higher simulated hit rate.

## Modules

```
prefixlab/
├── A. Cache core            (done)
│   ├── block_allocator      fixed-size block pool
│   ├── radix_cache          block-level prefix tree, ref-count pinning, leaf eviction
│   └── policies             LRU, LFU, CostAware, pluggable interface
├── B. Workloads             (started)
│   ├── trace                synthetic multi-turn agent trace            [done]
│   ├── rag trace            shared documents with Zipf popularity      [todo]
│   ├── branching trace      agent that forks sub-tasks from one prefix [todo]
│   └── real trace import    ShareGPT / public agent logs (if usable)   [todo]
├── C. Simulator + evaluation (started)
│   ├── simulator            hit rate, prefill tokens, eviction age      [done]
│   ├── oracle               Belady upper bound                          [done]
│   ├── evaluate / sweep     policies x sizes x seeds, error bars        [done]
│   └── latency model        prefill time = f(tokens) fitted from D      [todo]
├── D. Real-model engine     (todo)
│   ├── model wrapper        small HF causal LM (e.g. Qwen2.5-0.5B), greedy
│   ├── paged KV store       KV tensors kept in the block pool
│   ├── prefix-reuse prefill compute only the uncached suffix
│   └── parity test          token-for-token match with plain HF generate
├── E. Policy research       (todo)
│   ├── aging / LRU-2 / ARC-style variants
│   ├── cost-aware v2        weight by measured recompute time, from D
│   └── ablations            which signal (recency, frequency, depth, cost) matters
└── F. Reporting             (todo)
    ├── measured TTFT vs policy on the real model
    ├── comparison notes vs SGLang RadixAttention / vLLM prefix caching
    └── write-up with negative results included
```

## Milestones

| # | Milestone | Done when | Status |
|---|---|---|---|
| M0 | Cache core + simulator | tests pass, benchmark runs | done |
| M1 | Rigorous simulation | multi-seed error bars, oracle bound, figure in README | done |
| M2 | Real model, correctness | paged prefix-reuse generation matches HF greedy output | todo |
| M3 | Latency ground truth | measured prefill time vs cached-prefix length, fitted latency model | todo |
| M4 | More workloads | RAG (Zipf) and branching traces; check whether M1 ranking holds | todo |
| M5 | Better policy | a policy that beats LRU and LFU across sizes, or a documented failure | todo |
| M6 | End-to-end TTFT | policies compared on the real model under a request stream | todo |
| M7 | Write-up | README/report with results, limits, related work | todo |

## Things not to overclaim

- The tree is block-granular, not SGLang's compressed radix tree.
- Hit rate is not TTFT until M3 and M6 are done.
- Traces are synthetic until M4 says otherwise.
- The oracle is an upper-bound reference, not an implementable policy.
