"""M4: how much does document order limit prefix caching on a RAG workload?

For each document order, replays the synthetic RAG trace with an unbounded
cache (structural ceiling: no eviction) and with policies x cache sizes, over
several seeds. Hit rate is priced as TTFT with a measured LatencyModel.

    python benchmarks/rag_order.py docs/latency_mps.json [--k 3] [--zipf 1.0]
Writes docs/rag_order_k<k>_s<zipf>.{png,json}.
"""

import argparse
import json
import statistics as st
from dataclasses import replace

from prefixlab.latency import LatencyModel, estimated_ttft
from prefixlab.oracle import BeladyOracle
from prefixlab.policies import POLICIES
from prefixlab.rag_trace import ORDERS, RagTraceConfig, generate_rag_trace
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate

BS = 16
SIZES = [128, 256, 512, 1024, 2048]
UNBOUNDED = 10**7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latency_json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()
    with open(args.latency_json) as f:
        lm = LatencyModel(**json.load(f)["model"])

    names = list(POLICIES) + ([] if args.no_oracle else ["belady"])
    # res[order][policy][size] -> list over seeds of (hit, ttft_ratio)
    res = {o: {n: {b: [] for b in SIZES + [UNBOUNDED]} for n in names} for o in ORDERS}
    for seed in range(args.seeds):
        for order in ORDERS:
            cfg = RagTraceConfig(k=args.k, zipf_s=args.zipf, order=order, seed=seed)
            trace = generate_rag_trace(cfg)
            nocache = st.mean(lm.ttft(0, len(t)) for t in trace)
            for b in SIZES + [UNBOUNDED]:
                for n in names:
                    if b == UNBOUNDED and n != "lru":
                        continue  # no eviction: every policy is identical
                    pol = BeladyOracle(trace, BS) if n == "belady" else POLICIES[n]()
                    r = simulate(trace, RadixCache(b, BS, pol))
                    t = st.mean(estimated_ttft(trace, r.per_request_hit, lm)) / nocache
                    # Hits beyond the shared system prompt: the part order can affect.
                    sys_blk = cfg.system_len // BS * BS
                    doc_hit = sum(max(0, h - sys_blk) for h in r.per_request_hit)
                    doc_tot = sum(len(p) - sys_blk for p in trace)
                    res[order][n][b].append((r.token_hit_rate, t, doc_hit / doc_tot))
        print(f"seed {seed} done", flush=True)

    tag = f"k{args.k}_s{args.zipf:g}"
    with open(f"docs/rag_order_{tag}.json", "w") as f:
        json.dump(dict(k=args.k, zipf=args.zipf, seeds=args.seeds, sizes=SIZES,
                       latency=lm.label, fields=["hit", "ttft_ratio", "doc_hit"],
                       results={o: {n: {("unbounded" if b == UNBOUNDED else str(b)): v
                                        for b, v in per.items() if v}
                                    for n, per in res[o].items()} for o in res}), f)

    def cell(v):
        return "/".join(f"{st.mean(x[i] for x in v):.3f}" for i in range(3))

    print(f"\nRAG k={args.k} zipf={args.zipf}, {args.seeds} seeds; cells are "
          f"hit/TTFT-ratio/doc-hit ({lm.label})")
    for order in ORDERS:
        ceil = res[order]["lru"][UNBOUNDED]
        print(f"\n[{order}] unbounded cache: {cell(ceil)}")
        print(f"{'blocks':>7} " + " ".join(f"{n:>19}" for n in names))
        for b in SIZES:
            print(f"{b:>7} " + " ".join(f"{cell(res[order][n][b]):>19}" for n in names))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ORDERS), figsize=(13, 4), sharey=True)
    for ax, order in zip(axes, ORDERS):
        for n in names:
            m = [st.mean(v[0] for v in res[order][n][b]) for b in SIZES]
            s = [st.stdev(v[0] for v in res[order][n][b]) if args.seeds > 1 else 0.0
                 for b in SIZES]
            ax.errorbar(SIZES, m, yerr=s, marker="o", ms=3, capsize=2, label=n,
                        ls="--" if n == "belady" else "-")
        ceil = st.mean(v[0] for v in res[order]["lru"][UNBOUNDED])
        ax.axhline(ceil, color="gray", lw=1, ls=":", label="unbounded cache")
        ax.set_xscale("log", base=2)
        ax.set_title(f"order = {order}")
        ax.set_xlabel(f"cache size (blocks of {BS} tokens)")
    axes[0].set_ylabel("token hit rate")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Synthetic RAG, k={args.k}, Zipf s={args.zipf}, {args.seeds} seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"docs/rag_order_{tag}.png", dpi=150)


if __name__ == "__main__":
    main()
