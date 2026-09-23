"""Belady-style oracle: evict the leaf whose prefix is reused farthest in the future.

Needs the whole trace up front, so it is an upper-bound reference, not a policy
that could run online. The leaf-only eviction constraint of the prefix tree means
this is a strong heuristic bound rather than a proven optimum.
"""

from bisect import bisect_left

NEVER = 1 << 60


def _prefix_ids(tokens, block_size):
    ids, chain = [], ()
    for i in range(0, len(tokens) - block_size + 1, block_size):
        chain = chain + (tuple(tokens[i : i + block_size]),)
        ids.append(hash(chain))
    return ids


class BeladyOracle:
    name = "belady"

    def __init__(self, trace, block_size: int):
        self._uses: dict[int, list[int]] = {}
        for idx, tokens in enumerate(trace):
            for pid in _prefix_ids(tokens, block_size):
                self._uses.setdefault(pid, []).append(idx)

    @staticmethod
    def _node_id(node) -> int:
        chain = []
        while node.parent is not None:
            chain.append(node.key)
            node = node.parent
        return hash(tuple(reversed(chain)))

    def priority(self, node, now: int) -> tuple:
        # `now` is the cache clock: request index + 1, so future requests are idx >= now.
        uses = self._uses.get(self._node_id(node), [])
        pos = bisect_left(uses, now)
        next_use = uses[pos] if pos < len(uses) else NEVER
        return (-next_use,)
