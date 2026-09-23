"""Compare eviction policies on a synthetic agent trace across cache sizes.

    python benchmarks/compare_policies.py
"""

from prefixlab.policies import POLICIES
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate
from prefixlab.trace import TraceConfig, generate_trace

BLOCK_SIZE = 16


def main():
    trace = generate_trace(TraceConfig())
    print(f"{len(trace)} requests, block_size={BLOCK_SIZE}")
    print(f"{'blocks':>7} {'policy':>11} {'hit_rate':>9} {'prefill_tok':>12} {'evictions':>10} {'med_age':>8}")
    for num_blocks in (64, 128, 256, 512, 1024):
        for name, cls in POLICIES.items():
            res = simulate(trace, RadixCache(num_blocks, BLOCK_SIZE, cls()))
            print(f"{num_blocks:>7} {name:>11} {res.token_hit_rate:>9.3f} "
                  f"{res.prefill_tokens:>12} {res.evictions:>10} {res.median_eviction_age:>8.0f}")


if __name__ == "__main__":
    main()
