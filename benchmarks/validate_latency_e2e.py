"""Check the latency model and the simulator against the real engine on a request stream.

Runs the synthetic agent trace through PrefixEngine (real model, real KV pool,
one generated token per request so TTFT is all that is timed) and compares
  - per-request cached tokens with the simulator's (must agree, up to the
    engine's keep-one-token cap), and
  - measured TTFT (gather + prefill) with LatencyModel.ttft.

    python benchmarks/validate_latency_e2e.py docs/latency_mps.json --device mps
"""

import argparse
import json
import os
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=128)
    ap.add_argument("--policy", default="lru", choices=list(POLICIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--dump", help="write per-request (cached, new, measured, predicted) json")
    args = ap.parse_args()

    with open(args.latency_json) as f:
        lm = LatencyModel(**json.load(f)["model"])
    # Token ids drawn from a range every Qwen tokenizer id covers.
    cfg = replace(TraceConfig(), seed=args.seed, num_sessions=args.sessions, vocab=30000)
    trace = generate_trace(cfg)

    sim = simulate(trace, RadixCache(args.blocks, BS, POLICIES[args.policy]()))
    predicted = estimated_ttft(trace, sim.per_request_hit, lm)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    # Warm up kernels on a throwaway engine so the first requests are not penalised.
    warm = PrefixEngine(model, RadixCache(64, BS, POLICIES[args.policy]()))
    for t in trace[:3]:
        warm.generate(t, 1)
    del warm

    eng = PrefixEngine(model, RadixCache(args.blocks, BS, POLICIES[args.policy]()))
    measured, mismatched, rows = [], 0, []
    for i, (tokens, sim_hit) in enumerate(zip(trace, sim.per_request_hit)):
        r = eng.generate(tokens, 1)
        measured.append(r.ttft_seconds)
        rows.append(dict(cached=r.cached_tokens, new=len(tokens) - r.cached_tokens,
                         prefill=r.prefill_seconds, gather=r.gather_seconds,
                         predicted=predicted[i]))
        mismatched += r.cached_tokens != min(sim_hit, (len(tokens) - 1) // BS * BS)
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(trace)} requests", flush=True)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(rows, f, indent=1)
    rel = [(p - m) / m for p, m in zip(predicted, measured)]
    print(f"\n{len(trace)} requests, policy={args.policy}, blocks={args.blocks}, "
          f"hit rate {sim.token_hit_rate:.3f}")
    print(f"host 1-min load average: {os.getloadavg()[0]:.2f}")
    print(f"cached-token mismatches engine vs simulator: {mismatched}")
    print(f"mean TTFT measured {st.mean(measured) * 1e3:.1f} ms, "
          f"predicted {st.mean(predicted) * 1e3:.1f} ms "
          f"({(st.mean(predicted) - st.mean(measured)) / st.mean(measured):+.1%})")
    print(f"per-request error: median {st.median(rel):+.1%}, "
          f"median |err| {st.median(abs(x) for x in rel):.1%}, "
          f"p90 |err| {st.quantiles([abs(x) for x in rel], n=10)[-1]:.1%}")


if __name__ == "__main__":
    main()
