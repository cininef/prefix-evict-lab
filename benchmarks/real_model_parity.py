"""M2 check on a real model: prefix-reuse greedy output must equal HF generate.

Runs a few multi-turn conversations that share a system prompt through one
PrefixEngine, and compares every reply token-for-token with plain HF greedy
decoding of the same prompt.

    python benchmarks/real_model_parity.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--device cpu]
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefixlab.engine import PrefixEngine
from prefixlab.policies import LRU
from prefixlab.radix_cache import RadixCache

SYSTEM = ("You are a coding agent. You can call tools: read_file, grep, run_tests. "
          "Think step by step, keep answers short, and cite file paths. ") * 4
TURNS = [
    ["Where is the eviction policy defined?", "And how is LRU priority computed?"],
    ["Run the tests and summarise failures.", "Fix the first failure.", "Re-run."],
    ["What does the block allocator do?"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--new", type=int, default=24)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    eng = PrefixEngine(model, RadixCache(args.blocks, args.block_size, LRU()))

    # Pure argmax. The model's generation_config sets repetition_penalty=1.1 and
    # sampling knobs, and generate() merges them in even when given a fresh
    # GenerationConfig, so each one is overridden explicitly.
    greedy = dict(do_sample=False, max_new_tokens=args.new, repetition_penalty=1.0,
                  top_k=None, top_p=None, temperature=None)
    histories = [[{"role": "system", "content": SYSTEM}] for _ in TURNS]
    # Interleave sessions turn by turn, as a server would see them.
    mismatches = total = 0
    for t in range(max(len(s) for s in TURNS)):
        for s, turns in enumerate(TURNS):
            if t >= len(turns):
                continue
            histories[s].append({"role": "user", "content": turns[t]})
            text = tok.apply_chat_template(histories[s], add_generation_prompt=True, tokenize=False)
            prompt = tok.encode(text)
            r = eng.generate(prompt, args.new)
            with torch.no_grad():
                ref = model.generate(torch.tensor([prompt], device=args.device),
                                     **greedy)[0, len(prompt):]
            ref = ref.tolist()[: len(r.tokens)]
            # HF stops at EOS; compare up to the shorter of the two.
            ok = r.tokens[: len(ref)] == ref
            total += 1
            mismatches += not ok
            print(f"session {s} turn {t}: prompt={r.prompt_len:4d} cached={r.cached_tokens:4d} "
                  f"prefill={r.prefill_seconds * 1e3:7.1f} ms  match={ok}")
            histories[s].append({"role": "assistant",
                                 "content": tok.decode(r.tokens, skip_special_tokens=True)})
    print(f"\n{total - mismatches}/{total} replies identical to HF greedy")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
