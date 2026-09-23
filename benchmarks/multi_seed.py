"""Multi-seed policy sweep with an oracle upper bound; writes docs/hit_rate_vs_cache.png.

    python benchmarks/multi_seed.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prefixlab.evaluate import sweep

SIZES = [32, 64, 128, 192, 256, 384, 512, 768]
SEEDS = range(10)


def main():
    res = sweep(SIZES, SEEDS)
    print(f"{'blocks':>7} " + " ".join(f"{n:>17}" for n in res))
    for b in SIZES:
        print(f"{b:>7} " + " ".join(f"{res[n][b][0]:>9.3f}±{res[n][b][1]:.3f}" for n in res))
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, per in res.items():
        m = [per[b][0] for b in SIZES]
        s = [per[b][1] for b in SIZES]
        ax.errorbar(SIZES, m, yerr=s, marker="o", ms=3, capsize=2, label=n,
                    ls="--" if n == "belady" else "-")
    ax.set_xlabel("cache size (blocks of 16 tokens)")
    ax.set_ylabel("token hit rate")
    ax.set_title("Prefix-cache hit rate, 10 seeds (synthetic agent trace)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/hit_rate_vs_cache.png", dpi=150)


if __name__ == "__main__":
    main()
