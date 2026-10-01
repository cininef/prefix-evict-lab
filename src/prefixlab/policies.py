"""Eviction policies. A policy maps a node to a priority; the lowest priority is evicted first."""

from typing import Protocol


class EvictionPolicy(Protocol):
    """priority(node, now): the evictable leaf with the lowest priority goes first.

    Stateful policies may also define any of these optional hooks, which the
    cache calls when present:
      on_insert(node, now)    a new block was cached
      on_hit(node, now)       a cached block was matched by a request
      on_evict(node, now)     a block was evicted
      choose(leaves, now)     pick the victim among evictable leaves directly,
                              replacing the min-priority rule
    """

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


class ARC:
    """Adaptive Replacement Cache (Megiddo & Modha, 2003) adapted to a prefix tree.

    T1 holds blocks cached but not yet reused, T2 blocks reused at least once.
    Ghost lists B1/B2 remember prefix ids recently evicted from T1/T2. A miss on
    a B1 ghost means T1 was too small: grow the target p for T1; a B2 ghost
    shrinks it. A re-inserted ghost goes straight to T2. On eviction, take the
    least recently used evictable leaf of T1 if |T1| > p, else of T2, falling
    back to the other list when the preferred one has no evictable leaf (only
    leaves can be evicted, so ARC's choice is not always available).

    Capacity c is learned at the first eviction, when the pool is full.
    """

    name = "arc"

    def __init__(self):
        from collections import OrderedDict

        self.p = 0.0
        self.c = 0
        self.t2: set = set()  # nodes in T2; every other cached node is in T1
        self.live = 0
        self.b1: "OrderedDict[int, None]" = OrderedDict()
        self.b2: "OrderedDict[int, None]" = OrderedDict()
        self.choices = 0
        self.fallbacks = 0  # preferred list had no evictable leaf

    def priority(self, node, now: int) -> tuple:
        return (node.last_access,)

    def on_insert(self, node, now: int) -> None:
        self.live += 1
        if node.pid in self.b1:
            self.p = min(self.c, self.p + max(1.0, len(self.b2) / len(self.b1)))
            del self.b1[node.pid]
            self.t2.add(node)
        elif node.pid in self.b2:
            self.p = max(0.0, self.p - max(1.0, len(self.b1) / len(self.b2)))
            del self.b2[node.pid]
            self.t2.add(node)

    def on_hit(self, node, now: int) -> None:
        self.t2.add(node)

    def on_evict(self, node, now: int) -> None:
        self.live -= 1
        ghosts = self.b2 if node in self.t2 else self.b1
        self.t2.discard(node)
        ghosts[node.pid] = None
        while len(ghosts) > max(self.c, 1):
            ghosts.popitem(last=False)

    def choose(self, leaves, now: int):
        if self.c == 0:
            self.c = self.live
        t1 = [n for n in leaves if n not in self.t2]
        t2 = [n for n in leaves if n in self.t2]
        t1_size = self.live - len(self.t2)
        prefer_t1 = t1_size > self.p
        preferred = t1 if prefer_t1 else t2
        self.choices += 1
        self.fallbacks += not preferred
        pool = preferred or t1 or t2
        return min(pool, key=lambda n: (n.last_access, n.seq))


class ARCAged(ARC):
    """ARC whose T2 victim is chosen by recent popularity, then recency.

    Plain ARC orders T2 by recency only, so a system-prompt block reused ~100
    times goes as easily as one reused once whenever it is briefly a leaf; at
    64 blocks on the agent trace that cost 4.6 points against LFU. Ordering T2
    by raw hit count fixes small caches but brings back LFU's staleness at
    large ones. Here each block keeps an exponentially decayed hit score with
    half-life `half_life` requests, and T2 is ordered by (log2 bucket of the
    score, last access). Half-life 20 was picked on the agent, branch and
    Zipf-RAG (retrieval order) traces; longer half-lives lose at large caches
    (agent 512 blocks: -0.3 / -0.7 / -1.6 points at 20 / 50 / 200).
    """

    name = "arc_aged"

    def __init__(self, half_life: float = 20.0):
        super().__init__()
        self.half_life = half_life
        self.score: dict = {}  # node -> (score at time t, t)

    def _decayed(self, node, now: int) -> float:
        sc, t = self.score.get(node, (0.0, now))
        return sc * 2.0 ** (-(now - t) / self.half_life)

    def on_hit(self, node, now: int) -> None:
        super().on_hit(node, now)
        self.score[node] = (self._decayed(node, now) + 1.0, now)

    def on_evict(self, node, now: int) -> None:
        self.score.pop(node, None)
        super().on_evict(node, now)

    def choose(self, leaves, now: int):
        if self.c == 0:
            self.c = self.live
        t1 = [n for n in leaves if n not in self.t2]
        t2 = [n for n in leaves if n in self.t2]
        preferred = t1 if self.live - len(self.t2) > self.p else t2
        self.choices += 1
        self.fallbacks += not preferred
        pool = preferred or t1 or t2
        if pool is t2:
            return min(pool, key=lambda n: (int(self._decayed(n, now) + 1).bit_length(),
                                            n.last_access, n.seq))
        return min(pool, key=lambda n: (n.last_access, n.seq))


POLICIES = {"lru": LRU, "lfu": LFU, "cost_aware": CostAware, "arc": ARC, "arc_aged": ARCAged}
