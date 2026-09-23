"""Replay a trace through a RadixCache and report cache metrics."""

from dataclasses import dataclass

from .radix_cache import RadixCache


@dataclass
class SimResult:
    policy: str
    requests: int
    total_tokens: int
    hit_tokens: int
    evictions: int

    @property
    def token_hit_rate(self) -> float:
        return self.hit_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def prefill_tokens(self) -> int:
        """Tokens that must be recomputed; a proxy for TTFT."""
        return self.total_tokens - self.hit_tokens


def simulate(trace, cache: RadixCache) -> SimResult:
    total = hit = 0
    for tokens in trace:
        matched = cache.match_prefix(tokens)
        cache.lock(matched)
        cache.insert(tokens, matched)
        cache.unlock(matched)
        total += len(tokens)
        hit += len(matched) * cache.block_size
    return SimResult(cache.policy.name, len(trace), total, hit, cache.num_evictions)
