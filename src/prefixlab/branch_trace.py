"""Synthetic agent traces with sub-agent forks and tool-call pauses.

Each root agent starts from a shared system prompt and runs `root_turns` turns.
It then forks `fanout` sub-agents that each continue from a copy of the root's
history for `child_turns` turns (a deep shared prefix). When all its children
are done, the root runs one final aggregation turn on its own history, which
by then has been idle for a long time.

After any turn, a session waits on a tool with probability `pause_prob`; the
wait is exponential with mean `pause_mean` time units, where each request
takes one unit. A session cannot issue its next turn while waiting. If every
live session is waiting, time jumps ahead with no requests (an idle server
puts no pressure on the cache).

What pauses do and do not change: this is a closed system, so every request
goes to some live session and the mean reuse distance stays near the number of
live sessions. Pauses widen the distribution instead: sessions that are not
waiting cycle faster (shorter median gap) while a waiting tail sits idle longer
(the p90/median ratio grows from ~4x to ~10x on the default config with
pause_prob=0.6, pause_mean=300). Recency alone cannot tell a tail that is
waiting on a tool from one whose session has ended.
"""

import random
from dataclasses import dataclass, field


@dataclass
class BranchTraceConfig:
    num_system_prompts: int = 2
    system_len: int = 256
    num_roots: int = 10
    root_turns: int = 3
    fanout: int = 3
    child_turns: int = 3
    turn_len: int = 64
    pause_prob: float = 0.0
    pause_mean: float = 20.0
    vocab: int = 32000
    seed: int = 0


@dataclass
class _Session:
    history: list
    turns_left: int
    parent: "int | None" = None
    children_left: int = 0
    ready_at: float = 0.0
    is_root: bool = False
    forked: bool = False
    children: list = field(default_factory=list)


def generate_branch_trace(cfg: BranchTraceConfig) -> list[list[int]]:
    rng = random.Random(cfg.seed)

    def rand_tokens(n):
        return [rng.randrange(cfg.vocab) for _ in range(n)]

    systems = [rand_tokens(cfg.system_len) for _ in range(cfg.num_system_prompts)]
    sessions = [_Session(list(rng.choice(systems)), cfg.root_turns, is_root=True)
                for _ in range(cfg.num_roots)]
    live = set(range(len(sessions)))
    trace: list[list[int]] = []
    now = 0.0

    while live:
        ready = sorted(s for s in live if sessions[s].ready_at <= now
                       and sessions[s].children_left == 0)
        if not ready:
            # Everyone is waiting on a tool or on children: jump to the next wake-up.
            now = min(sessions[s].ready_at for s in live if sessions[s].children_left == 0)
            continue
        sid = rng.choice(ready)
        s = sessions[sid]
        s.history += rand_tokens(cfg.turn_len)
        trace.append(list(s.history))
        now += 1
        s.turns_left -= 1
        s.ready_at = now + (rng.expovariate(1 / cfg.pause_mean)
                            if rng.random() < cfg.pause_prob else 0.0)

        if s.turns_left > 0:
            continue
        if s.is_root and not s.forked and cfg.fanout > 0:
            s.forked = True
            s.children_left = cfg.fanout
            s.turns_left = 1  # the aggregation turn after the children finish
            for _ in range(cfg.fanout):
                sessions.append(_Session(list(s.history), cfg.child_turns, parent=sid,
                                         ready_at=s.ready_at))
                live.add(len(sessions) - 1)
            continue
        live.discard(sid)
        if s.parent is not None:
            p = sessions[s.parent]
            p.children_left -= 1
            if p.children_left == 0:
                p.ready_at = max(p.ready_at, now)
    return trace
