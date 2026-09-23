"""Synthetic multi-turn agent traces.

Each session starts from one of a few shared system prompts, then grows turn by
turn (user message + tool output appended to the history). Sessions are
interleaved at random, so cache pressure comes from many live conversations.
"""

import random
from dataclasses import dataclass


@dataclass
class TraceConfig:
    num_system_prompts: int = 3
    system_len: int = 256
    num_sessions: int = 40
    turns_per_session: int = 6
    turn_len: int = 64
    vocab: int = 32000
    seed: int = 0


def generate_trace(cfg: TraceConfig) -> list[list[int]]:
    """Return a list of requests; each request is the full prompt (token ids)."""
    rng = random.Random(cfg.seed)

    def rand_tokens(n):
        return [rng.randrange(cfg.vocab) for _ in range(n)]

    systems = [rand_tokens(cfg.system_len) for _ in range(cfg.num_system_prompts)]
    histories = [list(rng.choice(systems)) for _ in range(cfg.num_sessions)]
    remaining = [cfg.turns_per_session] * cfg.num_sessions

    trace = []
    live = list(range(cfg.num_sessions))
    while live:
        s = rng.choice(live)
        histories[s] += rand_tokens(cfg.turn_len)
        trace.append(list(histories[s]))
        remaining[s] -= 1
        if remaining[s] == 0:
            live.remove(s)
    return trace
