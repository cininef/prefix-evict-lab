"""M5: paired comparison of policies across all saved sweep results.

Every policy in a sweep json was run on the same trace per seed, so policies
are compared per seed (paired) instead of by overlapping error bars. For each
workload x cache size the reference is the best fixed policy in that cell,
chosen in hindsight (an online policy could not know it in advance), so the
comparison is strict for an adaptive policy.

Per policy it reports, over all cells: mean and worst shortfall ("regret") to
the per-cell best fixed policy, and for the candidate (default arc) the paired
difference to that best, with standard error and seeds won.

    python benchmarks/paired.py docs/workloads_*.json docs/rag_order_*.json
"""

import argparse
import json
import statistics as st

FIXED = ["lru", "lfu", "cost_aware"]


def cells(path):
    """Yield (label, size, {policy: [hit per seed]}) for every cache size in a json."""
    with open(path) as f:
        d = json.load(f)
    if "workload" in d:  # workload_sweep.py
        groups = {d["workload"]: d["results"]}
    else:  # rag_order.py: one group per document order
        src = f"real-{d['real'].split('real_rag_')[-1].removesuffix('.json')}" if d.get("real") \
            else f"zipf{d['zipf']:g}"
        groups = {f"rag-{src}-k{d['k']}-{o}": per for o, per in d["results"].items()}
    for label, per in groups.items():
        sizes = [s for s in per["lru"] if s != "unbounded"]
        for s in sorted(sizes, key=int):
            yield label, int(s), {p: [v[0] for v in per[p][s]] for p in per if s in per[p]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--candidate", default="arc")
    args = ap.parse_args()

    rows = [c for path in args.jsons for c in cells(path)]
    policies = [p for p in FIXED + [args.candidate] if all(p in r[2] for r in rows)]
    regret = {p: [] for p in policies}

    print(f"{'workload':<34}{'blocks':>7}  {'best fixed':>10}  "
          f"{args.candidate + ' - best (paired)':>22}  {'won':>5}")
    for label, size, hits in rows:
        means = {p: st.mean(hits[p]) for p in policies}
        best = max(FIXED, key=lambda p: means[p])
        for p in policies:
            regret[p].append(means[best] - means[p])
        if args.candidate in hits:
            diff = [a - b for a, b in zip(hits[args.candidate], hits[best])]
            se = st.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else float("nan")
            won = sum(x > 0 for x in diff)
            print(f"{label:<34}{size:>7}  {best:>10}  {st.mean(diff):+9.4f} ± {se:.4f}  "
                  f"{won:>2}/{len(diff)}")

    print(f"\nShortfall to the per-cell best fixed policy over {len(rows)} cells "
          f"(hit-rate points; lower is better):")
    for p in policies:
        r = regret[p]
        print(f"  {p:<11} mean {100 * st.mean(r):5.2f}   worst {100 * max(r):5.2f}")


if __name__ == "__main__":
    main()
