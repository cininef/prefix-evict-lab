from prefixlab.policies import LRU
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate
from prefixlab.trace import TraceConfig, generate_trace


def small_cfg(**kw):
    base = dict(num_system_prompts=2, system_len=32, num_sessions=5,
                turns_per_session=3, turn_len=16, seed=1)
    base.update(kw)
    return TraceConfig(**base)


def test_trace_is_deterministic_and_sized():
    a, b = generate_trace(small_cfg()), generate_trace(small_cfg())
    assert a == b
    assert len(a) == 5 * 3


def test_turns_extend_previous_prompt_of_same_session():
    trace = generate_trace(small_cfg(num_sessions=1, num_system_prompts=1))
    for prev, nxt in zip(trace, trace[1:]):
        assert nxt[: len(prev)] == prev


def test_large_cache_hits_all_but_first_touch():
    trace = generate_trace(small_cfg(num_sessions=1, num_system_prompts=1))
    res = simulate(trace, RadixCache(1000, 16, LRU()))
    assert res.evictions == 0
    # first request misses fully; later ones reuse all previously cached full blocks
    assert res.hit_tokens > 0
    assert res.token_hit_rate > 0.5


def test_tiny_cache_forces_evictions_and_lowers_hits():
    trace = generate_trace(small_cfg(num_sessions=10))
    big = simulate(trace, RadixCache(1000, 16, LRU()))
    tiny = simulate(trace, RadixCache(4, 16, LRU()))
    assert tiny.evictions > 0
    assert tiny.token_hit_rate < big.token_hit_rate
