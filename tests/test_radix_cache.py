from prefixlab.policies import LRU, LFU
from prefixlab.radix_cache import RadixCache


def make(num_blocks=8, block_size=2, policy=None):
    return RadixCache(num_blocks, block_size, policy or LRU())


def store(cache, tokens):
    m = cache.match_prefix(tokens)
    cache.lock(m)
    cache.insert(tokens, m)
    cache.unlock(m)
    return len(m)


def test_miss_then_hit():
    c = make()
    assert store(c, [1, 2, 3, 4]) == 0
    assert store(c, [1, 2, 3, 4]) == 2


def test_shared_prefix_reuses_blocks():
    c = make()
    store(c, [1, 2, 3, 4])
    store(c, [1, 2, 5, 6])
    assert c.num_cached_blocks == 3  # [1,2] shared, [3,4] and [5,6] distinct


def test_partial_trailing_block_not_cached():
    c = make()
    store(c, [1, 2, 3])
    assert c.num_cached_blocks == 1
    assert len(c.match_prefix([1, 2, 3])) == 1


def test_eviction_frees_space_and_drops_lru_leaf():
    c = make(num_blocks=2)
    store(c, [1, 2])
    store(c, [3, 4])
    store(c, [5, 6])  # forces eviction of [1,2]
    assert c.num_evictions == 1
    assert len(c.match_prefix([1, 2])) == 0
    assert len(c.match_prefix([3, 4])) == 1


def test_only_leaves_are_evicted():
    c = make(num_blocks=3)
    store(c, [1, 2, 3, 4])  # chain of 2
    store(c, [7, 8])
    store(c, [9, 10])  # must evict a leaf, never the interior [1,2]
    assert len(c.match_prefix([1, 2])) == 1


def test_locked_nodes_are_not_evicted():
    c = make(num_blocks=1)
    m_tokens = [1, 2]
    store(c, m_tokens)
    m = c.match_prefix(m_tokens)
    c.lock(m)
    assert c.insert([3, 4], []) == 0  # pool full, only block is pinned
    c.unlock(m)
    assert c.insert([3, 4], []) == 1


def test_lfu_keeps_frequently_hit_block():
    c = make(num_blocks=2, policy=LFU())
    store(c, [1, 2])
    for _ in range(3):
        store(c, [1, 2])
    store(c, [3, 4])
    store(c, [5, 6])  # evicts [3,4] (fewer hits) even though [1,2] is older
    assert len(c.match_prefix([1, 2])) == 1
    assert len(c.match_prefix([3, 4])) == 0
