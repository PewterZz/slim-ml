"""Spec decode correctness check.

Runs baseline (no draft) vs spec (with draft) at temperature=0 on the same
prompt, then diffs the token sequences. If they diverge, the snapshot/restore
implementation has a bug — `from_draft=True` alone doesn't prove correctness.

Usage:
  python tools/spec_decode_correctness.py --model X --draft Y
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def collect_tokens(model, tok, prompt: str, max_tokens: int, draft_model=None, num_draft: int = 2):
    sampler = make_sampler(temp=0.0)
    tokens = []
    texts = []
    kwargs = {"max_tokens": max_tokens, "sampler": sampler}
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = num_draft
    for resp in stream_generate(model, tok, prompt, **kwargs):
        tokens.append(int(resp.token))
        texts.append(resp.text)
    return tokens, "".join(texts)


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
    args = p.parse_args()

    model, tok = load(args.model)
    draft_model, _ = load(args.draft)

    print("=== baseline (no draft, temp=0) ===")
    base_tokens, base_text = collect_tokens(model, tok, args.prompt, args.max_tokens)
    print(f"[baseline] tokens={len(base_tokens)}")
    print(f"[baseline] first 20: {base_tokens[:20]}")

    print(f"\n=== spec decode num_draft={args.num_draft} (temp=0) ===")
    spec_tokens, spec_text = collect_tokens(
        model, tok, args.prompt, args.max_tokens, draft_model=draft_model, num_draft=args.num_draft
    )
    print(f"[spec] tokens={len(spec_tokens)}")
    print(f"[spec] first 20: {spec_tokens[:20]}")

    n = min(len(base_tokens), len(spec_tokens))
    diverge_idx = None
    for i in range(n):
        if base_tokens[i] != spec_tokens[i]:
            diverge_idx = i
            break

    if diverge_idx is None and len(base_tokens) == len(spec_tokens):
        print(f"\n[PASS] token sequences identical across {n} tokens")
        return 0
    elif diverge_idx is None:
        print(f"\n[WARN] identical prefix {n} tokens but lengths differ: "
              f"baseline={len(base_tokens)} spec={len(spec_tokens)}")
        return 0
    else:
        print(f"\n[FAIL] diverged at token index {diverge_idx}")
        print(f"       baseline[{diverge_idx}] = {base_tokens[diverge_idx]}")
        print(f"       spec[{diverge_idx}]     = {spec_tokens[diverge_idx]}")
        print(f"\n[baseline text] {base_text[:400]}")
        print(f"\n[spec text]     {spec_text[:400]}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
