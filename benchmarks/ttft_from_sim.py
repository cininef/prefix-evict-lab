"""Translate simulated hit rates into estimated TTFT with a measured LatencyModel.

For every policy x cache size x seed, replays the synthetic agent trace and
prices each request with the fitted model: TTFT = gather(C) + prefill(C, N).
Reports mean TTFT relative to a no-cache baseline, so a hit-rate gap between
policies can be read as a latency gap.

    python benchmarks/ttft_from_sim.py docs/latency_mps.json
Writes docs/ttft_vs_cache_<device>.png.
"""

import argparse
import json
import statistics as st
from dataclasses import replace

from prefixlab.latency import LatencyModel, estimated_ttft
from prefixlab.oracle import BeladyOracle
from prefixlab.policies import POLICIES
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate
from prefixlab.trace import TraceConfig, generate_trace

SIZES = [32, 64, 128, 192, 256, 384, 512, 768]
SEEDS = range(10)
BS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latency_json")
    args = ap.parse_args()
    with open(args.latency_json) as f:
        d = json.load(f)
    lm = LatencyModel(**d["model"])
    device = d.get("device", "")

    names = list(POLICIES) + ["belady"]
    hit = {n: {b: [] for b in SIZES} for n in names}
    ttft = {n: {b: [] for b in SIZES} for n in names}
    base, best = [], []
    for seed in SEEDS:
        trace = generate_trace(replace(TraceConfig(), seed=seed))
        nocache = st.mean(lm.ttft(0, len(t)) for t in trace)
        base.append(nocache)
        # Every request served with all but its last block from cache (engine cap).
        full = [(len(t) - 1) // BS * BS for t in trace]
        best.append(st.mean(lm.ttft(c, len(t) - c) for t, c in zip(trace, full)) / nocache)
        for b in SIZES:
            for n in names:
                pol = BeladyOracle(trace, BS) if n == "belady" else POLICIES[n]()
                r = simulate(trace, RadixCache(b, BS, pol))
                hit[n][b].append(r.token_hit_rate)
                ttft[n][b].append(st.mean(estimated_ttft(trace, r.per_request_hit, lm)) / nocache)

    print(f"no-cache mean TTFT: {st.mean(base) * 1e3:.1f} ms   ({lm.label})")
    print(f"best case, every request all-but-one-block cached (fixed cost / floor bound): "
          f"{st.mean(best):.0%} of no-cache\n")
    print(f"{'blocks':>6}  " + "  ".join(f"{n:>22}" for n in names))
    print(f"{'':>6}  " + "  ".join(f"{'hit / TTFT vs no-cache':>22}" for _ in names))
    for b in SIZES:
        cells = [f"{st.mean(hit[n][b]):.3f} / {st.mean(ttft[n][b]):.3f}±{st.stdev(ttft[n][b]):.3f}"
                 for n in names]
        print(f"{b:>6}  " + "  ".join(f"{c:>22}" for c in cells))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for n in names:
        ls = "--" if n == "belady" else "-"
        ax1.errorbar(SIZES, [st.mean(hit[n][b]) for b in SIZES],
                     yerr=[st.stdev(hit[n][b]) for b in SIZES], marker="o", ms=3, capsize=2,
                     ls=ls, label=n)
        ax2.errorbar(SIZES, [st.mean(ttft[n][b]) for b in SIZES],
                     yerr=[st.stdev(ttft[n][b]) for b in SIZES], marker="o", ms=3, capsize=2,
                     ls=ls, label=n)
    ax1.set_ylabel("token hit rate")
    ax2.set_ylabel("mean TTFT / no-cache TTFT")
    for ax in (ax1, ax2):
        ax.set_xlabel(f"cache size (blocks of {BS} tokens)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic agent trace, 10 seeds; TTFT from fitted latency model ({lm.label})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"docs/ttft_vs_cache_{device}.png", dpi=150)


if __name__ == "__main__":
    main()
