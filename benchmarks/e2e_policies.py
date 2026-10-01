"""M6: measured TTFT per eviction policy on the real engine.

Runs a request stream through PrefixEngine (real model, real KV pool, one
generated token per request) for each policy x cache size x seed, and reports
mean measured TTFT next to the simulator + latency-model prediction for the
same run. All runs are executed in one shuffled order so slow drift on the
host cannot favour one policy (see the M3 measurement notes), and host load is
logged.

    python benchmarks/e2e_policies.py docs/latency_mps.json --device mps
Writes docs/e2e_policies_<device>.json.
"""

import argparse
import json
import os
import random
import statistics as st
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM

from prefixlab.engine import PrefixEngine
from prefixlab.latency import LatencyModel, estimated_ttft
from prefixlab.policies import POLICIES
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate
from prefixlab.trace import TraceConfig, generate_trace

BS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latency_json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--policies", default="lru,lfu,arc_aged")
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=20)
    args = ap.parse_args()

    with open(args.latency_json) as f:
        lm = LatencyModel(**json.load(f)["model"])
    policies = args.policies.split(",")
    sizes = [int(x) for x in args.sizes.split(",")]
    traces = {s: generate_trace(replace(TraceConfig(), seed=s, num_sessions=args.sessions,
                                        vocab=30000)) for s in range(args.seeds)}

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    warm = PrefixEngine(model, RadixCache(64, BS, POLICIES["lru"]()))
    for t in traces[0][:5]:
        warm.generate(t, 1)
    del warm

    runs = [(p, b, s) for p in policies for b in sizes for s in range(args.seeds)]
    random.Random(0).shuffle(runs)
    load_before = os.getloadavg()[0]
    rows = []
    for i, (p, b, s) in enumerate(runs):
        trace = traces[s]
        sim = simulate(trace, RadixCache(b, BS, POLICIES[p]()))
        pred = st.mean(estimated_ttft(trace, sim.per_request_hit, lm))
        eng = PrefixEngine(model, RadixCache(b, BS, POLICIES[p]()))
        ttft, cached = [], 0
        for tokens in trace:
            r = eng.generate(tokens, 1)
            ttft.append(r.ttft_seconds)
            cached += r.cached_tokens
        rows.append(dict(policy=p, blocks=b, seed=s, measured=st.mean(ttft), predicted=pred,
                         hit=cached / sum(len(t) for t in trace), sim_hit=sim.token_hit_rate))
        print(f"[{i + 1}/{len(runs)}] {p:9s} {b:4d} seed {s}: measured {st.mean(ttft) * 1e3:6.1f} ms"
              f"  predicted {pred * 1e3:6.1f} ms  hit {rows[-1]['hit']:.3f}", flush=True)
    load_after = os.getloadavg()[0]

    print(f"\nhost 1-min load: {load_before:.2f} before, {load_after:.2f} after")
    print(f"{'blocks':>6} " + " ".join(f"{p:>22}" for p in policies)
          + "   (mean TTFT ms measured / predicted)")
    for b in sizes:
        cells = []
        for p in policies:
            r = [x for x in rows if x["policy"] == p and x["blocks"] == b]
            cells.append(f"{st.mean(x['measured'] for x in r) * 1e3:6.1f} / "
                         f"{st.mean(x['predicted'] for x in r) * 1e3:6.1f}")
        print(f"{b:>6} " + " ".join(f"{c:>22}" for c in cells))

    # Paired: each policy vs LRU on the same trace and size.
    print("\npaired vs lru, measured (predicted), % TTFT change:")
    for p in policies:
        if p == "lru":
            continue
        for b in sizes:
            dm, dp = [], []
            for s in range(args.seeds):
                a = next(x for x in rows if x["policy"] == p and x["blocks"] == b and x["seed"] == s)
                l = next(x for x in rows if x["policy"] == "lru" and x["blocks"] == b and x["seed"] == s)
                dm.append(a["measured"] / l["measured"] - 1)
                dp.append(a["predicted"] / l["predicted"] - 1)
            print(f"  {p:9s} {b:4d}: {100 * st.mean(dm):+5.1f}% ({100 * st.mean(dp):+5.1f}%)")

    with open(f"docs/e2e_policies_{args.device}.json", "w") as f:
        json.dump(dict(model=args.model, device=args.device, sessions=args.sessions,
                       loadavg_1m_before=load_before, loadavg_1m_after=load_after,
                       latency=lm.label, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
