import random

from prefixlab.policies import ARC, LFU, LRU
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate

BS = 4


def hot_set_with_scans(seed=0, hot=4, hot_len=16, scan_len=40, n=400):
    """A few hot prompts reused often, interleaved with one-off prompts whose
    total size far exceeds the cache (a scan)."""
    rng = random.Random(seed)
    hots = [[rng.randrange(10**6) for _ in range(hot_len)] for _ in range(hot)]
    trace = []
    for _ in range(n):
        if rng.random() < 0.4:
            trace.append(list(rng.choice(hots)))
        else:
            trace.append([rng.randrange(10**6) for _ in range(scan_len)])
    return trace


def hit(trace, policy, blocks=24):
    return simulate(trace, RadixCache(blocks, BS, policy)).token_hit_rate


def test_arc_resists_scans_like_lfu():
    t = hot_set_with_scans()
    lru, lfu, arc = hit(t, LRU()), hit(t, LFU()), hit(t, ARC())
    assert lfu > lru  # the scan flushes LRU's hot set
    assert arc > lru + 0.5 * (lfu - lru)


def test_arc_grows_t1_target_on_recency_workload():
    # Many sessions each re-requested soon after creation: evicted T1 blocks
    # come back quickly (B1 ghost hits), so p should grow.
    rng = random.Random(1)
    sessions = [[rng.randrange(10**6) for _ in range(8)] for _ in range(30)]
    trace = []
    for _ in range(300):
        s = rng.randrange(30)
        sessions[s] = sessions[s] + [rng.randrange(10**6) for _ in range(4)]
        trace.append(list(sessions[s]))
    arc = ARC()
    simulate(trace, RadixCache(40, BS, arc))
    assert arc.p > 0
    assert arc.c == 40


def test_arc_aged_protects_a_hot_prefix_that_plain_arc_drops():
    # Agent trace at 64 blocks: 3 system prompts take 48 blocks. A prompt block
    # is briefly a leaf whenever its sessions' tails are evicted, and recency
    # alone lets it go despite its many hits.
    from dataclasses import replace

    from prefixlab.policies import ARCAged
    from prefixlab.trace import TraceConfig, generate_trace

    t = generate_trace(replace(TraceConfig(), seed=0))
    arc = simulate(t, RadixCache(64, 16, ARC())).token_hit_rate
    aged = simulate(t, RadixCache(64, 16, ARCAged())).token_hit_rate
    assert aged > arc + 0.01


def test_arc_aged_score_decays():
    from prefixlab.policies import ARCAged
    from prefixlab.radix_cache import Node

    pol, n = ARCAged(half_life=10), Node(key=(), parent=None)
    pol.on_hit(n, 0)
    pol.on_hit(n, 0)
    assert pol._decayed(n, 0) == 2.0
    assert abs(pol._decayed(n, 10) - 1.0) < 1e-9
