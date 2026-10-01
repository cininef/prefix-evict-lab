"""M3: measure prefill time vs cached-prefix length on a real model and fit LatencyModel.

For each (total, cached) point the full prompt's KV is first written to a KVPool,
then every repeat gathers the cached blocks and runs the forward pass on the
suffix only, exactly as PrefixEngine does. Points are visited in shuffled
rounds and summarised by the median across rounds.
A grid is used for fitting; random held-out points check the fit.

    python benchmarks/prefill_latency.py --device mps
    python benchmarks/prefill_latency.py --refit docs/latency_mps.json
Writes docs/latency_<device>.json (fitted model + raw points) and
docs/prefill_latency_<device>.png.
"""

import argparse
import json
import os
import random
import statistics as st
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from prefixlab.engine import KVPool
from prefixlab.latency import LatencyModel

BS = 16
TOTALS = [128, 256, 512, 1024, 2048]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]  # 1.0 -> all but the last block


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def sample(model, pool, prompt, cached):
    """One timed request: gather the cached blocks, prefill the suffix."""
    device = pool.k.device
    blocks = list(range(cached // BS))
    ids = torch.tensor([prompt[cached:]], device=device)
    sync(device)
    t0 = time.perf_counter()
    past = pool.gather(blocks) if blocks else DynamicCache()
    sync(device)
    t1 = time.perf_counter()
    model(input_ids=ids, past_key_values=past, use_cache=True)
    sync(device)
    t2 = time.perf_counter()
    return t2 - t1, (t1 - t0 if blocks else 0.0)


def run_rounds(model, pool, prompt, points, rounds, warmup_rounds, rng):
    """Visit every point once per round in shuffled order, so slow drift (thermal,
    clock changes) spreads across all points as noise instead of biasing the
    points that happened to run during a slow spell."""
    pre = {p: [] for p in points}
    gat = {p: [] for p in points}
    for r in range(warmup_rounds + rounds):
        order = list(points)
        rng.shuffle(order)
        for total, cached in order:
            pt, gt = sample(model, pool, prompt[:total], cached)
            if r >= warmup_rounds:
                pre[(total, cached)].append(pt)
                gat[(total, cached)].append(gt)
        print(f"round {r + 1}/{warmup_rounds + rounds} done", flush=True)
    out = []
    for total, cached in points:
        ps = pre[(total, cached)]
        q = st.quantiles(ps, n=4)
        out.append(dict(cached=cached, new=total - cached, prefill=st.median(ps),
                        prefill_iqr=q[2] - q[0], gather=st.median(gat[(total, cached)])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--warmup", type=int, default=1, help="discarded rounds")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--floor", choices=["auto", "on", "off"], default="auto",
                    help="saturation floor term; auto = on for accelerators, off for cpu")
    ap.add_argument("--refit", help="re-fit and re-plot an existing docs/latency_*.json")
    args = ap.parse_args()
    if args.refit:
        with open(args.refit) as f:
            d = json.load(f)
        keep = ("label", "device", "block_size", "loadavg_1m_before", "loadavg_1m_after")
        meta = {k: d[k] for k in keep if k in d}
        meta.setdefault("label", d["model"].get("label", ""))
        meta.setdefault("device", args.refit.split("_")[-1].removesuffix(".json"))
        meta.update(warmup_rounds=d.get("warmup_rounds", d.get("warmup")),
                    rounds=d.get("rounds", d.get("repeats")))
        meta["floor"] = use_floor(args.floor, meta["device"])
        report(d["grid"], d["heldout"], meta)
        return

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    max_total = max(TOTALS)
    pool = KVPool(model.config, max_total // BS, BS, dtype, device)
    rng = random.Random(0)
    prompt = [rng.randrange(1000, 30000) for _ in range(max_total)]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt], device=device), use_cache=True)
    pool.scatter(out.past_key_values, 0, list(range(max_total // BS)))
    del out

    grid_pts = []
    for total in TOTALS:
        for f in FRACS:
            grid_pts.append((total, min(int(total * f) // BS * BS, total - BS)))
    held_pts = []
    while len(held_pts) < args.holdout:
        total = rng.randrange(BS * 4, max_total + 1)
        p = (total, rng.randrange(0, (total - 1) // BS + 1) * BS)
        if p not in grid_pts and p not in held_pts:
            held_pts.append(p)
    # Grid and held-out points share rounds so both see the same drift.
    # Short forwards are bound by host-side kernel dispatch: with every core busy,
    # a 32-token suffix on MPS took +20% while a 1024-token one took +3%. Record
    # host load so a contaminated run can be spotted.
    load_before = os.getloadavg()
    res = run_rounds(model, pool, prompt, grid_pts + held_pts, args.rounds, args.warmup, rng)
    grid, held = res[: len(grid_pts)], res[len(grid_pts):]
    for name, rows in (("grid", grid), ("held-out", held)):
        print(f"-- {name} --")
        for r in rows:
            print(f"cached={r['cached']:5d} new={r['new']:5d}  prefill={r['prefill'] * 1e3:8.1f} ms "
                  f"(IQR {r['prefill_iqr'] * 1e3:5.1f})  gather={r['gather'] * 1e3:6.2f} ms")

    label = f"{args.model} {args.device} {args.dtype}"
    meta = dict(label=label, device=args.device, floor=use_floor(args.floor, args.device), warmup_rounds=args.warmup,
                rounds=args.rounds, block_size=BS,
                loadavg_1m_before=load_before[0], loadavg_1m_after=os.getloadavg()[0])
    print(f"host 1-min load average: {load_before[0]:.2f} before, "
          f"{meta['loadavg_1m_after']:.2f} after")
    report(grid, held, meta)


def use_floor(choice, device):
    """On an accelerator a short forward is bound by host-side kernel dispatch, so
    time is flat until the device saturates (MPS: floor fit 5.5% vs plain 8.1% error
    on N<=128). On CPU compute and dispatch share the cores, small-N time keeps
    growing with C, and the floor over-predicts (11.3% vs 8.9%)."""
    if choice == "auto":
        return not device.startswith("cpu")
    return choice == "on"


def report(grid, held, meta):
    """Fit on the grid, check on held-out points, write json + figure."""
    kw = dict(floor_max_new=2 * BS, linear_min_new=4 * BS) if meta["floor"] else {}
    m = LatencyModel.fit([(r["cached"], r["new"], r["prefill"], r["gather"]) for r in grid],
                         meta["label"], **kw)

    def err(r):
        t = r["prefill"] + r["gather"]
        return abs(m.ttft(r["cached"], r["new"]) - t) / t

    errs = [err(r) for r in held]
    short = [err(r) for r in grid + held if r["new"] <= 128]
    long_ = [err(r) for r in grid + held if r["new"] > 128]
    print(f"\nfit: floor={m.floor * 1e3:.1f} ms  a={m.a * 1e3:.2f} ms  b={m.b * 1e6:.1f} us/token  "
          f"c={m.c * 1e9:.3f} ns/token^2  R2={m.r2_prefill:.4f}")
    print(f"gather: g0={m.g0 * 1e3:.3f} ms  g1={m.g1 * 1e6:.3f} us/token  R2={m.r2_gather:.4f}")
    print(f"held-out TTFT error: mean {st.mean(errs):.1%}, max {max(errs):.1%}")
    print(f"all points, new<=128: mean {st.mean(short):.1%}   new>128: mean {st.mean(long_):.1%}")
    cold = max((r for r in grid if r["cached"] == 0), key=lambda r: r["new"])
    hot = max((r for r in grid if r["new"] <= BS), key=lambda r: r["cached"])
    print(f"longest prompt cold vs all-but-one-block cached: "
          f"{cold['prefill'] * 1e3:.0f} ms -> {(hot['prefill'] + hot['gather']) * 1e3:.0f} ms "
          f"({cold['prefill'] / (hot['prefill'] + hot['gather']):.1f}x)")

    tag = meta["device"]
    with open(f"docs/latency_{tag}.json", "w") as f:
        json.dump(dict(model=m.__dict__, heldout_mean_err=st.mean(errs),
                       heldout_max_err=max(errs), grid=grid, heldout=held, **meta), f, indent=2)
    plot(m, grid, held, f"docs/prefill_latency_{tag}.png", meta["label"])


def plot(m, grid, held, path, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for total in TOTALS:
        pts = [r for r in grid if r["cached"] + r["new"] == total]
        xs = [r["cached"] / total for r in pts]
        line, = ax1.plot(xs, [r["prefill"] * 1e3 for r in pts], "o", ms=4, label=f"{total} tok")
        fx = [i / 50 for i in range(51)]
        ax1.plot(fx, [m.prefill(int(total * x), total - int(total * x)) * 1e3 for x in fx],
                 "-", lw=1, color=line.get_color())
    ax1.set_xlabel("fraction of prompt cached")
    ax1.set_ylabel("prefill time (ms)")
    ax1.set_yscale("log")
    ax1.set_title("measured (dots) vs fit (lines)")
    ax1.legend(fontsize=8)

    meas = [(r["prefill"] + r["gather"]) * 1e3 for r in held]
    pred = [m.ttft(r["cached"], r["new"]) * 1e3 for r in held]
    ax2.plot(meas, pred, "o", ms=4)
    lim = [0, max(meas + pred) * 1.05]
    ax2.plot(lim, lim, "--", lw=1, color="gray")
    ax2.set_xlabel("measured TTFT (ms), held-out")
    ax2.set_ylabel("predicted TTFT (ms)")
    ax2.set_title("held-out check")
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
