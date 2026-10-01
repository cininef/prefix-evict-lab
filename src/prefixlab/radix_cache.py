"""Block-granular prefix tree over a paged KV pool.

Each node owns exactly one KV block covering `block_size` tokens, keyed by that
chunk of token ids. A root-to-node path is a cached prefix. Only full blocks
are cached; a trailing partial block is never shared.
"""

from dataclasses import dataclass, field

from .block_allocator import BlockAllocator, OutOfBlocks
from .policies import EvictionPolicy

ROOT_PID = 0


def chain_pid(parent_pid: int, key: tuple) -> int:
    """Stable id of a cached prefix: hash of (parent prefix id, last block's tokens).
    Two nodes with the same token path get the same pid, even across evictions."""
    return hash((parent_pid, key))


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
    created: int = 0
    seq: int = 0  # unique insertion order; final tie-break between equal priorities
    pid: int = ROOT_PID  # prefix id, see chain_pid


class RadixCache:
    def __init__(self, num_blocks: int, block_size: int, policy: EvictionPolicy):
        self.block_size = block_size
        self.policy = policy
        self.allocator = BlockAllocator(num_blocks)
        self.root = Node(key=(), parent=None)
        self._clock = 0
        self.num_evictions = 0
        self.eviction_ages: list[int] = []  # requests between insert and eviction
        # Leaves are tracked incrementally: scanning the whole tree on every
        # eviction was 99% of simulation time on long RAG prompts.
        self._leaves: set[Node] = set()
        self._seq = 0

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
        return len(self.insert_nodes(tokens, matched))

    def insert_nodes(self, tokens, matched: list[Node]) -> list[Node]:
        """Like `insert`, but return the new nodes so a caller can fill their blocks."""
        chunks = self._chunks(tokens)
        node = matched[-1] if matched else self.root
        fresh: list[Node] = []  # pinned so this request cannot evict its own new blocks
        walked: list[Node] = []  # existing nodes past `matched`, pinned while we extend them
        for chunk in chunks[len(matched) :]:
            existing = node.children.get(chunk)
            if existing is not None:
                existing.ref_count += 1
                walked.append(existing)
                node = existing
                continue
            block = self._allocate_block()
            if block is None:
                break
            self._seq += 1
            child = Node(key=chunk, parent=node, block=block, depth=node.depth + 1,
                         last_access=self._clock, created=self._clock, seq=self._seq,
                         pid=chain_pid(node.pid, chunk))
            node.children[chunk] = child
            self._leaves.discard(node)
            self._leaves.add(child)
            if hasattr(self.policy, "on_insert"):
                self.policy.on_insert(child, self._clock)
            child.ref_count += 1
            fresh.append(child)
            node = child
        self.unlock(fresh)
        self.unlock(walked)
        return fresh

    def _allocate_block(self) -> int | None:
        try:
            return self.allocator.allocate()
        except OutOfBlocks:
            if not self.evict_one():
                return None
            return self.allocator.allocate()

    def _evictable_leaves(self) -> list[Node]:
        return [n for n in self._leaves if n.ref_count == 0]

    def evict_one(self) -> bool:
        leaves = self._evictable_leaves()
        if not leaves:
            return False
        if hasattr(self.policy, "choose"):
            victim = self.policy.choose(leaves, self._clock)
        else:
            victim = min(leaves, key=lambda n: (self.policy.priority(n, self._clock), n.seq))
        parent = victim.parent
        del parent.children[victim.key]
        self._leaves.discard(victim)
        if not parent.children and parent is not self.root:
            self._leaves.add(parent)
        self.allocator.free(victim.block)
        if hasattr(self.policy, "on_evict"):
            self.policy.on_evict(victim, self._clock)
        self.num_evictions += 1
        self.eviction_ages.append(self._clock - victim.created)
        return True

    @property
    def num_cached_blocks(self) -> int:
        return self.allocator.num_blocks - self.allocator.num_free
