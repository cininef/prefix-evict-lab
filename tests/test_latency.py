import random

import pytest

from prefixlab.latency import LatencyModel, estimated_ttft


def synthetic(a, b, c, g0, g1, noise=0.0, seed=0):
    rng = random.Random(seed)
    out = []
    for total in (128, 256, 512, 1024):
        for cached in (0, total // 4, total // 2, 3 * total // 4, total - 16):
            new = total - cached
            p = a + b * new + c * new * (cached + new / 2)
            g = g0 + g1 * cached if cached else 0.0
            out.append((cached, new, p * (1 + rng.gauss(0, noise)), g))
    return out


def test_fit_recovers_coefficients():
    m = LatencyModel.fit(synthetic(0.01, 2e-4, 3e-8, 1e-3, 5e-6))
    assert m.a == pytest.approx(0.01, rel=1e-6)
    assert m.b == pytest.approx(2e-4, rel=1e-6)
    assert m.c == pytest.approx(3e-8, rel=1e-6)
    assert m.g1 == pytest.approx(5e-6, rel=1e-6)
    assert m.r2_prefill == pytest.approx(1.0)


def test_fit_is_robust_to_noise():
    m = LatencyModel.fit(synthetic(0.01, 2e-4, 3e-8, 0, 0, noise=0.03))
    assert m.b == pytest.approx(2e-4, rel=0.1)
    assert m.r2_prefill > 0.98


def test_hits_reduce_ttft_and_gather_only_on_hit(tmp_path):
    m = LatencyModel(a=0.01, b=2e-4, c=3e-8, g0=1e-3, g1=5e-6, r2_prefill=1.0, r2_gather=1.0)
    assert m.gather(0) == 0.0
    assert m.ttft(768, 256) < m.ttft(0, 1024)
    m.save(tmp_path / "m.json")
    assert LatencyModel.load(tmp_path / "m.json") == m


def test_estimated_ttft_caps_full_hit():
    m = LatencyModel(a=0.0, b=1.0, c=0.0)
    trace = [[0] * 32, [0] * 32]
    assert estimated_ttft(trace, [0, 32], m) == [32.0, 1.0]
