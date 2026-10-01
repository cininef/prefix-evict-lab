"""Belady-style oracle: evict the leaf whose prefix is reused farthest in the future.

Needs the whole trace up front, so it is an upper-bound reference, not a policy
that could run online. The leaf-only eviction constraint of the prefix tree means
this is a strong heuristic bound rather than a proven optimum.
"""

from bisect import bisect_left

NEVER = 1 << 60
ROOT_ID = 0


def _chain(parent_id: int, key: tuple) -> int:
    """Id of a prefix = hash of (id of its parent prefix, its last block's tokens).
    O(1) per block, where hashing the whole path was O(depth * block_size)."""
    return hash((parent_id, key))


def _prefix_ids(tokens, block_size):
    ids, chain = [], ROOT_ID
    for i in range(0, len(tokens) - block_size + 1, block_size):
        chain = _chain(chain, tuple(tokens[i : i + block_size]))
        ids.append(chain)
    return ids


class BeladyOracle:
    name = "belady"

    def __init__(self, trace, block_size: int):
        self._uses: dict[int, list[int]] = {}
        self._ids: dict = {}  # node -> prefix id; a node's path never changes
        for idx, tokens in enumerate(trace):
            for pid in _prefix_ids(tokens, block_size):
                self._uses.setdefault(pid, []).append(idx)

    def _node_id(self, node) -> int:
        if node.parent is None:
            return ROOT_ID
        nid = self._ids.get(node)
        if nid is None:
            nid = self._ids[node] = _chain(self._node_id(node.parent), node.key)
        return nid

    def priority(self, node, now: int) -> tuple:
        # `now` is the cache clock: request index + 1, so future requests are idx >= now.
        uses = self._uses.get(self._node_id(node), [])
        pos = bisect_left(uses, now)
        next_use = uses[pos] if pos < len(uses) else NEVER
        return (-next_use,)
