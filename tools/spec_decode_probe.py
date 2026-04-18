"""Speculative decoding diagnostic.

Measures acceptance rate (fraction of tokens from draft model) and end-to-end
speedup for a big/draft pair on MLX. Produces the two numbers that discriminate
whether spec decode pencils out on this hardware/model combo:

  - acceptance%: if ~0%, the draft distribution disagrees with the big model;
                 cache surgery can't save you — the draft is wrong.
  - speedup×:   if acceptance is 60%+ but speedup is ~1×, you're bandwidth-bound;
                spec decode doesn't buy anything on this hardware.

Usage:
  python tools/spec_decode_probe.py --model X --draft Y [--num-draft 2] [--max-tokens 64]
"""
from __future__ import annotations

import argparse
import sys
import time

from mlx_lm import load, stream_generate


def run_baseline(model_path: str, prompt: str, max_tokens: int) -> tuple[float, float]:
    model, tok = load(model_path)
    t0 = time.perf_counter()
    n = 0
    ttft = None
    for resp in stream_generate(model, tok, prompt, max_tokens=max_tokens):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    elapsed = time.perf_counter() - t0
    decode_tps = (n - 1) / (elapsed - ttft) if n > 1 and ttft else 0.0
    print(f"[baseline] tokens={n} ttft={ttft*1000:.0f}ms total={elapsed:.2f}s decode_tps={decode_tps:.2f}")
    return decode_tps, elapsed


def run_spec(
    model_path: str, draft_path: str, prompt: str, max_tokens: int, num_draft: int
) -> tuple[float, float, float]:
    model, tok = load(model_path)
    draft_model, _ = load(draft_path)
    t0 = time.perf_counter()
    n = 0
    n_from_draft = 0
    ttft = None
    for resp in stream_generate(
        model,
        tok,
        prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft,
    ):
        if ttft is None:
            ttft = time.perf_counter() - t0
        if resp.from_draft:
            n_from_draft += 1
        n += 1
    elapsed = time.perf_counter() - t0
    decode_tps = (n - 1) / (elapsed - ttft) if n > 1 and ttft else 0.0
    acceptance = n_from_draft / n if n else 0.0
    print(
        f"[spec num_draft={num_draft}] tokens={n} from_draft={n_from_draft} "
        f"accept={acceptance*100:.1f}% ttft={ttft*1000:.0f}ms total={elapsed:.2f}s "
        f"decode_tps={decode_tps:.2f}"
    )
    return decode_tps, elapsed, acceptance


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument(
        "--prompt",
        default="Write a Python function that takes a list of integers and "
        "returns the median. Handle even-length lists by averaging the two "
        "middle elements. Include a docstring and type hints.",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-draft", type=int, default=2)
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()

    if not args.skip_baseline:
        print("=== baseline (no draft) ===")
        base_tps, _ = run_baseline(args.model, args.prompt, args.max_tokens)
    else:
        base_tps = 0.0

    print(f"\n=== spec decode num_draft={args.num_draft} ===")
    try:
        spec_tps, _, accept = run_spec(
            args.model, args.draft, args.prompt, args.max_tokens, args.num_draft
        )
    except ValueError as e:
        print(f"[spec] FAILED: {e}")
        return 2

    if base_tps > 0:
        speedup = spec_tps / base_tps if base_tps else 0.0
        print(f"\n[result] baseline={base_tps:.2f} t/s  spec={spec_tps:.2f} t/s  "
              f"speedup={speedup:.2f}×  accept={accept*100:.1f}%")

        # Diagnostic interpretation
        if accept < 0.10:
            print("[diagnosis] Very low acceptance — draft distribution disagrees "
                  "with big model. Fix the draft, not the cache.")
        elif accept >= 0.60 and speedup < 1.15:
            print("[diagnosis] High acceptance but flat speedup — hardware "
                  "bandwidth-bound. Spec decode can't help on this setup.")
        elif speedup >= 1.3:
            print("[diagnosis] Real payoff. Worth investing in cache surgery "
                  "for hybrid-attention models.")
        else:
            print("[diagnosis] Modest payoff. Borderline worth the engineering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
