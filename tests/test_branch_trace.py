from dataclasses import replace

from prefixlab.branch_trace import BranchTraceConfig, generate_branch_trace

SMALL = BranchTraceConfig(num_system_prompts=1, system_len=32, num_roots=4, root_turns=2,
                          fanout=2, child_turns=2, turn_len=16, seed=5)


def test_deterministic_and_counts_every_turn():
    a, b = generate_branch_trace(SMALL), generate_branch_trace(SMALL)
    assert a == b
    # per root: root turns + aggregation turn + fanout * child turns
    assert len(a) == 4 * (2 + 1 + 2 * 2)


def test_every_request_extends_some_earlier_request():
    trace = generate_branch_trace(SMALL)
    for i, p in enumerate(trace[1:], 1):
        assert len(p) == 32 + 16 or any(p[: len(q)] == q and len(p) == len(q) + 16
                                         for q in trace[:i])


def test_children_share_the_parent_prefix_and_diverge():
    trace = generate_branch_trace(replace(SMALL, num_roots=1))
    fork_len = 32 + 2 * 16
    after = [p for p in trace if len(p) == fork_len + 16]
    # two children's first turns plus the root's aggregation turn
    assert len(after) == 3
    assert len({tuple(p[:fork_len]) for p in after}) == 1
    assert len({tuple(p) for p in after}) == 3


def reuse_gaps(trace, turn_len):
    """Requests between each turn and the request it extends (its previous turn)."""
    last, gaps = {}, []
    for i, p in enumerate(trace):
        j = last.get(tuple(p[: len(p) - turn_len]))
        if j is not None:
            gaps.append(i - j)
        last[tuple(p)] = i
    return sorted(gaps)


def test_pauses_widen_the_reuse_distance_distribution():
    # Closed system: the mean gap is pinned near the number of live sessions, so
    # pauses show up as dispersion (p90 / median), not as a longer mean.
    cfg = BranchTraceConfig(seed=0)
    base = reuse_gaps(generate_branch_trace(cfg), cfg.turn_len)
    paused = reuse_gaps(generate_branch_trace(replace(cfg, pause_prob=0.6, pause_mean=300)),
                        cfg.turn_len)
    assert len(paused) == len(base)

    def spread(g):
        return g[int(0.9 * len(g))] / g[len(g) // 2]

    assert spread(paused) > 1.5 * spread(base)
