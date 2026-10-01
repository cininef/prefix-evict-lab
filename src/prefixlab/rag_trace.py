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
