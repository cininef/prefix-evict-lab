"""Block-granular prefix tree over a paged KV pool.

Each node owns exactly one KV block covering `block_size` tokens, keyed by that
chunk of token ids. A root-to-node path is a cached prefix. Only full blocks
are cached; a trailing partial block is never shared.
"""

from dataclasses import dataclass, field

from .block_allocator import BlockAllocator, OutOfBlocks
from .policies import EvictionPolicy


@dataclass(eq=False)
class Node:
    key: tuple
    parent: "Node | None"
    block: int = -1
    depth: int = 0
    children: dict = field(default_factory=dict)
    ref_count: int = 0
    last_access: int = 0
    hit_count: int = 0


class RadixCache:
    def __init__(self, num_blocks: int, block_size: int, policy: EvictionPolicy):
        self.block_size = block_size
        self.policy = policy
        self.allocator = BlockAllocator(num_blocks)
        self.root = Node(key=(), parent=None)
        self._clock = 0
        self.num_evictions = 0

    def _chunks(self, tokens):
        bs = self.block_size
        return [tuple(tokens[i : i + bs]) for i in range(0, len(tokens) - bs + 1, bs)]

    def match_prefix(self, tokens) -> list[Node]:
        """Return the nodes along the longest cached prefix, updating access stats."""
        self._clock += 1
        node, path = self.root, []
        for chunk in self._chunks(tokens):
            child = node.children.get(chunk)
            if child is None:
                break
            child.last_access = self._clock
            child.hit_count += 1
            path.append(child)
            node = child
        return path

    def lock(self, path: list[Node]) -> None:
        for n in path:
            n.ref_count += 1

    def unlock(self, path: list[Node]) -> None:
        for n in path:
            assert n.ref_count > 0
            n.ref_count -= 1

    def insert(self, tokens, matched: list[Node]) -> int:
        """Cache blocks beyond `matched`. Returns the number of new blocks stored.

        Stops early if the pool is full and nothing is evictable.
        """
        chunks = self._chunks(tokens)
        node = matched[-1] if matched else self.root
        added = 0
        for chunk in chunks[len(matched) :]:
            block = self._allocate_block()
            if block is None:
                break
            child = Node(key=chunk, parent=node, block=block, depth=node.depth + 1,
                         last_access=self._clock)
            node.children[chunk] = child
            node = child
            added += 1
        return added

    def _allocate_block(self) -> int | None:
        try:
            return self.allocator.allocate()
        except OutOfBlocks:
            if not self.evict_one():
                return None
            return self.allocator.allocate()

    def _evictable_leaves(self) -> list[Node]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children.values())
            elif n is not self.root and n.ref_count == 0:
                out.append(n)
        return out

    def evict_one(self) -> bool:
        leaves = self._evictable_leaves()
        if not leaves:
            return False
        victim = min(leaves, key=lambda n: self.policy.priority(n, self._clock))
        del victim.parent.children[victim.key]
        self.allocator.free(victim.block)
        self.num_evictions += 1
        return True

    @property
    def num_cached_blocks(self) -> int:
        return self.allocator.num_blocks - self.allocator.num_free
