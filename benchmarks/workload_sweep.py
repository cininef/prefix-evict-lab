"""M4: run the policy sweep on several agent workloads and check whether the M1
ranking holds.

Workloads:
  agent         the M1 multi-turn trace (40 sessions x 6 turns)
  branch        30 roots, each forks 3 sub-agents (deep shared prefixes)
  branch_pause  branch + tool-call pauses (p=0.6, mean 300 time units)
  branch_pause_matched  branch_pause with 60 roots, so its cache load matches branch

Why the matched variant: in this closed system, sessions waiting on a tool
are not issuing requests, so at equal root count the pause trace has fewer
sessions active at once and less cache pressure. Comparing branch with
branch_pause alone would credit pauses with a hit-rate gain that is really
lower load. Load is reported as distinct KV blocks touched per 50-request
window (the short-term working set).

For each: unbounded-cache ceiling, then policies (incl. Belady) x cache sizes x
seeds, with hit rate priced as TTFT by a fitted LatencyModel.

    python benchmarks/workload_sweep.py docs/latency_mps.json [--workloads agent,branch]
Writes docs/workloads_<name>.json and docs/workloads.png.
"""

import argparse
import json
import statistics as st
from dataclasses import replace

from prefixlab.branch_trace import BranchTraceConfig, generate_branch_trace
from prefixlab.latency import LatencyModel, estimated_ttft
from prefixlab.oracle import BeladyOracle
from prefixlab.policies import POLICIES
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate
from prefixlab.trace import TraceConfig, generate_trace

BS = 16
SIZES = [64, 128, 256, 512, 1024]
UNBOUNDED = 10**7

WORKLOADS = {
    "agent": lambda seed: generate_trace(replace(TraceConfig(), seed=seed)),
    "branch": lambda seed: generate_branch_trace(
        BranchTraceConfig(num_roots=30, seed=seed)),
    "branch_pause": lambda seed: generate_branch_trace(
        BranchTraceConfig(num_roots=30, pause_prob=0.6, pause_mean=300, seed=seed)),
    "branch_pause_matched": lambda seed: generate_branch_trace(
        BranchTraceConfig(num_roots=60, pause_prob=0.6, pause_mean=300, seed=seed)),
}


def window_working_set(trace, window=50):
    """Mean number of distinct full blocks (as prefix chains) touched per window."""
    ids = []
    for p in trace:
        chain, blocks = (), set()
        for i in range(0, len(p) - BS + 1, BS):
            chain = hash((chain, tuple(p[i : i + BS])))
            blocks.add(chain)
        ids.append(blocks)
    sizes = [len(set().union(*ids[i : i + window])) for i in range(0, len(ids) - window + 1)]
    return st.mean(sizes)


def run(name, lm, seeds, names):
    res = {n: {b: [] for b in SIZES + [UNBOUNDED]} for n in names}
    loads = []
    for seed in range(seeds):
        trace = WORKLOADS[name](seed)
        loads.append(window_working_set(trace))
        nocache = st.mean(lm.ttft(0, len(t)) for t in trace)
        for b in SIZES + [UNBOUNDED]:
            for n in names:
                if b == UNBOUNDED and n != "lru":
                    continue
                pol = BeladyOracle(trace, BS) if n == "belady" else POLICIES[n]()
                r = simulate(trace, RadixCache(b, BS, pol))
                t = st.mean(estimated_ttft(trace, r.per_request_hit, lm)) / nocache
                res[n][b].append((r.token_hit_rate, t))
    return res, len(trace), st.mean(len(t) for t in trace), st.mean(loads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latency_json")
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    with open(args.latency_json) as f:
        lm = LatencyModel(**json.load(f)["model"])
    names = list(POLICIES) + ["belady"]
    wls = args.workloads.split(",")

    all_res = {}
    for w in wls:
        res, nreq, mean_len, load = run(w, lm, args.seeds, names)
        all_res[w] = res
        with open(f"docs/workloads_{w}.json", "w") as f:
            json.dump(dict(workload=w, seeds=args.seeds, requests=nreq, mean_prompt=mean_len,
                           window_working_set=load,
                           latency=lm.label, fields=["hit", "ttft_ratio"],
                           results={n: {("unbounded" if b == UNBOUNDED else str(b)): v
                                        for b, v in per.items() if v}
                                    for n, per in res.items()}), f)
        ceil = res["lru"][UNBOUNDED]
        print(f"\n[{w}] {nreq} requests, mean prompt {mean_len:.0f} tokens, "
              f"{load:.0f} blocks touched per 50 requests; "
              f"unbounded: hit {st.mean(h for h, _ in ceil):.3f} "
              f"TTFT {st.mean(t for _, t in ceil):.3f}   (cells: hit / TTFT ratio)")
        print(f"{'blocks':>7} " + " ".join(f"{n:>13}" for n in names) + "   best online")
        for b in SIZES:
            means = {n: (st.mean(h for h, _ in res[n][b]), st.mean(t for _, t in res[n][b]))
                     for n in names}
            best = max((n for n in names if n != "belady"), key=lambda n: means[n][0])
            print(f"{b:>7} " + " ".join(f"{means[n][0]:.3f}/{means[n][1]:.3f}".rjust(13)
                                        for n in names) + f"   {best}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(wls), figsize=(4.3 * len(wls), 4), sharey=True, squeeze=False)
    for ax, w in zip(axes[0], wls):
        for n in names:
            m = [st.mean(h for h, _ in all_res[w][n][b]) for b in SIZES]
            s = [st.stdev(h for h, _ in all_res[w][n][b]) if args.seeds > 1 else 0 for b in SIZES]
            ax.errorbar(SIZES, m, yerr=s, marker="o", ms=3, capsize=2, label=n,
                        ls="--" if n == "belady" else "-")
        ax.axhline(st.mean(h for h, _ in all_res[w]["lru"][UNBOUNDED]), color="gray", lw=1,
                   ls=":", label="unbounded cache")
        ax.set_xscale("log", base=2)
        ax.set_title(w)
        ax.set_xlabel(f"cache size (blocks of {BS} tokens)")
    axes[0][0].set_ylabel("token hit rate")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Agent workloads, {args.seeds} seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig("docs/workloads.png", dpi=150)


if __name__ == "__main__":
    main()
