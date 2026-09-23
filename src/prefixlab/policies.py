"""Eviction policies. A policy maps a node to a priority; the lowest priority is evicted first."""

from typing import Protocol


class EvictionPolicy(Protocol):
    name: str

    def priority(self, node, now: int) -> tuple: ...


class LRU:
    name = "lru"

    def priority(self, node, now: int) -> tuple:
        return (node.last_access,)


class LFU:
    name = "lfu"

    def priority(self, node, now: int) -> tuple:
        return (node.hit_count, node.last_access)


class CostAware:
    """Keep blocks that are expensive to recompute and likely to be reused.

    Recompute cost of a block grows with its depth (attention over a longer
    context), and reuse likelihood is approximated by hit count. A leaf's
    score is (hits + 1) * cost(depth); ties fall back to recency.
    """

    name = "cost_aware"

    def __init__(self, depth_weight: float = 0.1):
        self.depth_weight = depth_weight

    def priority(self, node, now: int) -> tuple:
        cost = 1.0 + self.depth_weight * node.depth
        return ((node.hit_count + 1) * cost, node.last_access)


POLICIES = {"lru": LRU, "lfu": LFU, "cost_aware": CostAware}
