"""Sweep policies x cache sizes x seeds and aggregate token hit rate."""

import statistics as st
from dataclasses import replace

from .oracle import BeladyOracle
from .policies import POLICIES
from .radix_cache import RadixCache
from .simulator import simulate
from .trace import TraceConfig, generate_trace


def sweep(cache_sizes, seeds, cfg: TraceConfig | None = None, block_size: int = 16,
          include_oracle: bool = True):
    """Return {policy: {num_blocks: (mean, stdev)}} of token hit rate."""
    cfg = cfg or TraceConfig()
    names = list(POLICIES) + (["belady"] if include_oracle else [])
    samples = {n: {b: [] for b in cache_sizes} for n in names}
    for seed in seeds:
        trace = generate_trace(replace(cfg, seed=seed))
        for b in cache_sizes:
            for n in names:
                policy = BeladyOracle(trace, block_size) if n == "belady" else POLICIES[n]()
                samples[n][b].append(simulate(trace, RadixCache(b, block_size, policy)).token_hit_rate)
    return {n: {b: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for b, v in per.items()}
            for n, per in samples.items()}
