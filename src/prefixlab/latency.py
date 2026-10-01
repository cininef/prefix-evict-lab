"""Prefill latency as a function of cached and new tokens, fitted from measurements.

    prefill(C, N) = max(floor, a + b*N + c*N*(C + N/2))   # forward over N new tokens
    gather(C)     = g0 + g1*C                             # copying C cached tokens out

b*N covers the per-token work (projections, MLP); c*N*(C + N/2) is attention,
where each new token attends to every cached token and, on average, half of
the other new ones. `floor` is the latency of a forward pass too small to
saturate the device (kernel launch overhead per layer dominates); without it
the linear fit, dominated by long prompts, underestimates short suffixes,
which is exactly the regime of a multi-turn hit.

TTFT estimate = gather(C) + prefill(C, N); the gather term is the price of a
hit and is zero when C == 0.
"""

import json
from dataclasses import asdict, dataclass


def _lstsq(rows: list[list[float]], ys: list[float]) -> list[float]:
    """Ordinary least squares via the normal equations (small k, no numpy needed)."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    # Gauss-Jordan with partial pivoting.
    m = [ata[i] + [aty[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if m[col][col] == 0:
            raise ValueError("singular design: vary both cached and new token counts")
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][k] / m[i][i] for i in range(k)]


def _r2(pred: list[float], ys: list[float]) -> float:
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    return 1 - ss_res / ss_tot if ss_tot else 1.0


def _features(cached: int, new: int) -> list[float]:
    return [1.0, float(new), new * (cached + new / 2)]


@dataclass
class LatencyModel:
    a: float
    b: float
    c: float
    g0: float = 0.0
    g1: float = 0.0
    floor: float = 0.0
    r2_prefill: float = float("nan")
    r2_gather: float = float("nan")
    label: str = ""

    def prefill(self, cached: int, new: int) -> float:
        lin = sum(w * x for w, x in zip((self.a, self.b, self.c), _features(cached, new)))
        return max(self.floor, lin)

    def gather(self, cached: int) -> float:
        return self.g0 + self.g1 * cached if cached else 0.0

    def ttft(self, cached: int, new: int) -> float:
        return self.gather(cached) + self.prefill(cached, new)

    @classmethod
    def fit(cls, samples, label: str = "", floor_max_new: int | None = None,
            linear_min_new: int = 0) -> "LatencyModel":
        """samples: iterable of (cached, new, prefill_s, gather_s).

        With `floor_max_new`, the floor is the median prefill of samples with
        new <= floor_max_new, and the linear part is fitted on samples with
        new >= linear_min_new only.
        """
        samples = list(samples)
        lin = [s for s in samples if s[1] >= linear_min_new]
        x = [_features(c, n) for c, n, _, _ in lin]
        a, b, c = _lstsq(x, [p for _, _, p, _ in lin])
        floor = 0.0
        if floor_max_new is not None:
            small = sorted(p for _, n, p, _ in samples if n <= floor_max_new)
            if small:
                floor = small[len(small) // 2]
        m = cls(a, b, c, floor=floor, label=label)
        r2p = _r2([m.prefill(c_, n) for c_, n, _, _ in samples], [p for _, _, p, _ in samples])

        hit = [(c, g) for c, _, _, g in samples if c > 0]
        g0 = g1 = 0.0
        r2g = float("nan")
        if len({c for c, _ in hit}) >= 2:
            g0, g1 = _lstsq([[1.0, float(c)] for c, _ in hit], [g for _, g in hit])
            r2g = _r2([g0 + g1 * c for c, _ in hit], [g for _, g in hit])
        m.g0, m.g1, m.r2_prefill, m.r2_gather = g0, g1, r2p, r2g
        return m

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path) -> "LatencyModel":
        with open(path) as f:
            return cls(**json.load(f))


def estimated_ttft(trace, per_request_hit, model: LatencyModel) -> list[float]:
    """Per-request TTFT estimate for a simulated run (see SimResult.per_request_hit).

    The engine always recomputes at least one prompt token, so a hit is capped at
    len(prompt) - 1 to match what a real run would do.
    """
    out = []
    for tokens, hit in zip(trace, per_request_hit):
        cached = min(hit, len(tokens) - 1)
        out.append(model.ttft(cached, len(tokens) - cached))
    return out
