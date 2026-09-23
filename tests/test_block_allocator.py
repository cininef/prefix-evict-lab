import pytest

from prefixlab.block_allocator import BlockAllocator, OutOfBlocks


def test_allocates_distinct_blocks_until_exhausted():
    alloc = BlockAllocator(3)
    blocks = {alloc.allocate() for _ in range(3)}
    assert blocks == {0, 1, 2}
    assert alloc.num_free == 0
    with pytest.raises(OutOfBlocks):
        alloc.allocate()


def test_free_makes_block_reusable():
    alloc = BlockAllocator(1)
    b = alloc.allocate()
    alloc.free(b)
    assert alloc.num_free == 1
    assert alloc.allocate() == b


def test_double_free_rejected():
    alloc = BlockAllocator(2)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(ValueError):
        alloc.free(b)


def test_invalid_size():
    with pytest.raises(ValueError):
        BlockAllocator(0)
