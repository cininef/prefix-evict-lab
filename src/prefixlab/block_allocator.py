"""Fixed-size block pool, the GPU-memory analogue of a physical page frame table."""


class OutOfBlocks(Exception):
    """Raised when the pool has no free block."""


class BlockAllocator:
    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks
        block = self._free.pop()
        self._used.add(block)
        return block

    def free(self, block: int) -> None:
        if block not in self._used:
            raise ValueError(f"block {block} is not allocated")
        self._used.remove(block)
        self._free.append(block)
