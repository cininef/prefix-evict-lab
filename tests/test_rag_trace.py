from dataclasses import replace

import pytest

from prefixlab.policies import LRU
from prefixlab.rag_trace import ORDERS, RagTraceConfig, generate_rag_trace
from prefixlab.radix_cache import RadixCache
from prefixlab.simulator import simulate

SMALL = RagTraceConfig(num_docs=20, doc_len_min=32, doc_len_max=64, num_requests=60,
                       system_len=32, seed=3)


def test_deterministic_and_shares_system_prompt():
    a, b = generate_rag_trace(SMALL), generate_rag_trace(SMALL)
    assert a == b and len(a) == 60
    assert all(p[:32] == a[0][:32] for p in a)


def test_orders_contain_the_same_tokens_per_request():
    traces = [generate_rag_trace(replace(SMALL, order=o)) for o in ORDERS]
    for reqs in zip(*traces):
        assert len({len(r) for r in reqs}) == 1
        assert len({tuple(sorted(r)) for r in reqs}) == 1


def test_unknown_order_rejected():
    with pytest.raises(ValueError):
        generate_rag_trace(replace(SMALL, order="nope"))


def test_fixed_order_beats_retrieval_order_with_unbounded_cache():
    # Not a theorem, but expected for Zipf popularity: a fixed order makes two
    # requests that retrieved the same set share the whole document prefix.
    def hit(order):
        t = generate_rag_trace(replace(SMALL, order=order))
        return simulate(t, RadixCache(100_000, 16, LRU())).token_hit_rate

    assert hit("canonical") >= hit("retrieval")
    assert hit("popular_first") >= hit("retrieval")


def test_trace_from_retrievals_reuses_document_tokens():
    from prefixlab.rag_trace import generate_rag_trace_from_retrievals

    retrievals = [[0, 1], [1, 0], [2, 0]]
    doc_lens, query_lens = [20, 30, 40], [3, 4, 5]
    t = generate_rag_trace_from_retrievals(retrievals, doc_lens, query_lens,
                                           order="canonical", system_len=8, seed=1)
    assert len(t) == 3
    by_len = {len(p): p for p in t}
    # queries 0 and 1 retrieved the same set; canonical order makes the docs identical
    assert by_len[8 + 50 + 3][: 8 + 50] == by_len[8 + 50 + 4][: 8 + 50]
    pf = generate_rag_trace_from_retrievals(retrievals, doc_lens, query_lens,
                                            order="popular_first", system_len=8, seed=1)
    # doc 0 is in every retrieval, so it is the most frequent and leads every prompt
    assert len({tuple(p[8:28]) for p in pf}) == 1
    assert len({tuple(p[28:48]) for p in pf}) > 1  # what follows it differs
