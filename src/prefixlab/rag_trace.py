"""Synthetic RAG traces: system prompt + top-k retrieved documents + query.

Document popularity is Zipf(s) over a fixed corpus. Each request retrieves k
distinct documents, sampled by popularity without replacement; the sampling
order stands in for the retriever's ranking (popular documents tend to rank
high, but not always).

Prefix caching can only reuse a document's KV if everything before it in the
prompt is identical, so the order in which documents are placed matters:

- "retrieval":     as ranked by the retriever
- "canonical":     sorted by document id (a fixed order unrelated to popularity)
- "popular_first": sorted by global popularity, most popular first

Document ids are a random permutation of popularity ranks, so "canonical" does
not accidentally coincide with "popular_first".
"""

import random
from dataclasses import dataclass

ORDERS = ("retrieval", "canonical", "popular_first")


@dataclass
class RagTraceConfig:
    num_docs: int = 200
    doc_len_min: int = 128
    doc_len_max: int = 512
    zipf_s: float = 1.0
    k: int = 3
    num_requests: int = 400
    system_len: int = 256
    query_len_min: int = 16
    query_len_max: int = 48
    order: str = "retrieval"
    vocab: int = 32000
    seed: int = 0


def generate_rag_trace(cfg: RagTraceConfig) -> list[list[int]]:
    """Return a list of prompts (token ids). Same seed -> same corpus and retrievals
    for every `order`, so orders can be compared request by request."""
    if cfg.order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    rng = random.Random(cfg.seed)

    def rand_tokens(n):
        return [rng.randrange(cfg.vocab) for _ in range(n)]

    system = rand_tokens(cfg.system_len)
    docs = [rand_tokens(rng.randint(cfg.doc_len_min, cfg.doc_len_max))
            for _ in range(cfg.num_docs)]
    # rank_of[doc_id] = popularity rank (0 = most popular).
    rank_of = list(range(cfg.num_docs))
    rng.shuffle(rank_of)
    weight = [1.0 / (rank_of[d] + 1) ** cfg.zipf_s for d in range(cfg.num_docs)]

    trace = []
    for _ in range(cfg.num_requests):
        picked: list[int] = []
        w = list(weight)
        for _ in range(cfg.k):
            d = rng.choices(range(cfg.num_docs), weights=w)[0]
            picked.append(d)
            w[d] = 0.0
        query = rand_tokens(rng.randint(cfg.query_len_min, cfg.query_len_max))
        if cfg.order == "canonical":
            picked.sort()
        elif cfg.order == "popular_first":
            picked.sort(key=lambda d: rank_of[d])
        prompt = list(system)
        for d in picked:
            prompt += docs[d]
        trace.append(prompt + query)
    return trace


def generate_rag_trace_from_retrievals(retrievals, doc_lens, query_lens, order="retrieval",
                                       system_len=256, num_requests=None, vocab=32000,
                                       seed=0) -> list[list[int]]:
    """RAG trace from real retrieval results (see benchmarks/build_real_rag.py).

    retrievals[q] is the ranked list of document ids for query q. Token contents
    are random but stable per document, so only identity and length carry over.
    Queries are issued in a random order (one pass, or `num_requests` draws
    without replacement); real query arrival order is not available.
    "popular_first" ranks documents by how often they are retrieved over the
    whole query set, which a live system would have to estimate online.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    rng = random.Random(seed)

    def rand_tokens(n):
        return [rng.randrange(vocab) for _ in range(n)]

    system = rand_tokens(system_len)
    doc_tokens: dict[int, list[int]] = {}
    freq: dict[int, int] = {}
    for r in retrievals:
        for d in r:
            freq[d] = freq.get(d, 0) + 1

    qs = list(range(len(retrievals)))
    rng.shuffle(qs)
    if num_requests is not None:
        qs = qs[:num_requests]
    trace = []
    for q in qs:
        picked = list(retrievals[q])
        if order == "canonical":
            picked.sort()
        elif order == "popular_first":
            picked.sort(key=lambda d: (-freq[d], d))
        prompt = list(system)
        for d in picked:
            if d not in doc_tokens:
                doc_tokens[d] = rand_tokens(doc_lens[d])
            prompt += doc_tokens[d]
        trace.append(prompt + rand_tokens(max(1, query_lens[q])))
    return trace
