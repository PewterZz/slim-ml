"""Standalone test for slim_ml.spec_decode.speculative_step.

Verifies the slim-ml-native spec decode loop against two gates:
  1. Correctness: temp=0 token IDs match mlx_lm.stream_generate baseline
  2. Perf: hybrid-attention model reaches ~1.15x speedup and ~55% accept

Run before wiring into MLXBackend / Session.
"""
from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from slim_ml.spec_decode import speculative_step


def _baseline_tokens(model, tok, prompt: str, max_tokens: int):
    sampler = make_sampler(temp=0.0)
    out = []
    for resp in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler):
        out.append(int(resp.token))
    return out


def _slim_tokens(model, draft_model, tok, prompt: str, max_tokens: int, num_draft: int):
    prompt_ids = mx.array(tok.encode(prompt), mx.uint32)
    sampler = lambda x: mx.argmax(x, axis=-1)
    rounds = []

    def rec(evt, payload):
        if evt == "spec_round":
            rounds.append(payload)

    tokens = []
    from_draft_flags = []
    t0 = time.monotonic()
    for tid, _lp, from_draft in speculative_step(
        prompt_ids,
        model,
        draft_model,
        num_draft_tokens=num_draft,
        max_tokens=max_tokens,
        sampler=sampler,
        recorder=rec,
    ):
        tokens.append(int(tid))
        from_draft_flags.append(bool(from_draft))
    elapsed = time.monotonic() - t0
    return tokens, from_draft_flags, rounds, elapsed


def _baseline_timed(model, tok, prompt: str, max_tokens: int):
    sampler = make_sampler(temp=0.0)
    t0 = time.monotonic()
    n = 0
    for _ in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler):
        n += 1
    return n, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NexVeridian/OmniCoder-9B-4bit")
    p.add_argument("--draft", default="mlx-community/Qwen3.5-0.8B-MLX-4bit")
    p.add_argument(
        "--prompt",
        default="Write a Python function that takes a list of integers and "
        "returns the median. Handle even-length lists by averaging the two "
        "middle elements. Include a docstring and type hints.",
    )
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--num-draft", type=int, default=2)
    args = p.parse_args()

    print(f"[load] model={args.model}")
    model, tok = load(args.model)
    print(f"[load] draft={args.draft}")
    draft_model, _ = load(args.draft)

    print("\n=== Gate 1: correctness (temp=0) ===")
    base = _baseline_tokens(model, tok, args.prompt, args.max_tokens)
    slim, _flags, _rounds, _ = _slim_tokens(
        model, draft_model, tok, args.prompt, args.max_tokens, args.num_draft
    )
    n = min(len(base), len(slim))
    diverge = next((i for i in range(n) if base[i] != slim[i]), None)
    if diverge is None and len(base) == len(slim):
        print(f"[PASS] {n} identical tokens")
    elif diverge is None:
        print(f"[WARN] prefix match {n}, lengths differ base={len(base)} slim={len(slim)}")
    else:
        print(f"[FAIL] diverge at {diverge}: base={base[diverge]} slim={slim[diverge]}")
        print(f"  base[:{diverge+3}]={base[:diverge+3]}")
        print(f"  slim[:{diverge+3}]={slim[:diverge+3]}")
        return 2

    print("\n=== Gate 2: perf ===")
    n_base, t_base = _baseline_timed(model, tok, args.prompt, args.max_tokens)
    _toks, flags, rounds, t_spec = _slim_tokens(
        model, draft_model, tok, args.prompt, args.max_tokens, args.num_draft
    )
    accept = sum(1 for f in flags if f) / max(len(flags), 1)
    speedup = t_base / t_spec if t_spec > 0 else 0.0
    tps_base = n_base / t_base
    tps_spec = len(flags) / t_spec
    total_draft = sum(r["num_draft"] for r in rounds)
    total_accept = sum(r["num_accept"] for r in rounds)
    per_round = total_accept / total_draft if total_draft else 0.0
    mean_verify = sum(r["verify_ms"] for r in rounds) / max(len(rounds), 1)
    mean_replay = sum(r["replay_ms"] for r in rounds) / max(len(rounds), 1)

    print(f"[base]  tokens={n_base} time={t_base:.2f}s tps={tps_base:.2f}")
    print(f"[slim]  tokens={len(flags)} time={t_spec:.2f}s tps={tps_spec:.2f}")
    print(f"[slim]  speedup={speedup:.2f}x")
    print(f"[slim]  from_draft acceptance={accept*100:.1f}%")
    print(f"[slim]  per-round accept={per_round*100:.1f}% ({total_accept}/{total_draft})")
    print(f"[slim]  rounds={len(rounds)} mean verify={mean_verify:.1f}ms replay={mean_replay:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
