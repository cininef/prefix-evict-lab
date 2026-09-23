from prefixlab.evaluate import sweep
from prefixlab.trace import TraceConfig


def small():
    return TraceConfig(num_system_prompts=2, system_len=32, num_sessions=8,
                       turns_per_session=4, turn_len=16)


def test_oracle_matches_or_beats_every_online_policy():
    res = sweep([8, 16], seeds=range(3), cfg=small())
    for b in (8, 16):
        best_online = max(res[n][b][0] for n in ("lru", "lfu", "cost_aware"))
        assert res["belady"][b][0] >= best_online - 1e-9


def test_sweep_shape_and_monotone_in_cache_size():
    res = sweep([8, 64], seeds=range(3), cfg=small(), include_oracle=False)
    assert set(res) == {"lru", "lfu", "cost_aware"}
    for n in res:
        assert res[n][64][0] >= res[n][8][0]
