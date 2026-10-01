import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from prefixlab.engine import PrefixEngine
from prefixlab.policies import LRU
from prefixlab.radix_cache import RadixCache

BS = 8
NEW = 8


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=512)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def reference(model, prompt):
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=NEW, do_sample=False,
                             pad_token_id=0)
    return out[0, len(prompt):].tolist()


def make_engine(model, num_blocks):
    return PrefixEngine(model, RadixCache(num_blocks, BS, LRU()))


def toks(rng, n):
    return [rng.randrange(1, 256) for _ in range(n)]


def test_cold_request_matches_hf(model):
    prompt = toks(random.Random(0), 37)
    r = make_engine(model, 64).generate(prompt, NEW)
    assert r.cached_tokens == 0
    assert r.tokens == reference(model, prompt)


def test_shared_prefix_reuse_matches_hf(model):
    rng = random.Random(1)
    system = toks(rng, 40)
    eng = make_engine(model, 64)
    eng.generate(system + toks(rng, 11), NEW)
    p2 = system + toks(rng, 13)
    r = eng.generate(p2, NEW)
    assert r.cached_tokens == 40  # 5 full blocks of the shared system prompt
    assert r.tokens == reference(model, p2)


def test_fully_cached_prompt_keeps_one_token_uncached(model):
    prompt = toks(random.Random(2), 32)  # exactly 4 blocks
    eng = make_engine(model, 64)
    eng.generate(prompt, NEW)
    used = eng.cache.num_cached_blocks
    r = eng.generate(prompt, NEW)
    assert r.cached_tokens == 24
    assert r.tokens == reference(model, prompt)
    assert eng.cache.num_cached_blocks == used  # no duplicate blocks for the last chunk


def test_parity_under_eviction_pressure(model):
    rng = random.Random(3)
    systems = [toks(rng, 24) for _ in range(3)]
    eng = make_engine(model, 10)  # far smaller than the working set
    hits = 0
    for _ in range(12):
        p = rng.choice(systems) + toks(rng, rng.randrange(5, 30))
        r = eng.generate(p, NEW)
        hits += r.cached_tokens
        assert r.tokens == reference(model, p)
    assert eng.cache.num_evictions > 0 and hits > 0
