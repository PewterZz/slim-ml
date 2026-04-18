"""Measure marginal impact of kv_bits on OmniCoder at long context.

Runs N=256 tokens at temp=0 three times: no quant, kv-bits=8, kv-bits=4.
Reports tps, peak memory, and token identity vs baseline.
"""
from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from slim_ml.backend import GenerationSettings, MLXBackend
from slim_ml.budget import StaticBudget, auto_detect_limits
from slim_ml.runtime import Session


def run(session, prompt, max_tokens, kv_bits=None, kv_start=0):
    settings = GenerationSettings(
        max_tokens=max_tokens,
        temperature=0.0,
        kv_bits=kv_bits,
        quantized_kv_start=kv_start,
    )
    mx.reset_peak_memory()
    tokens = []
    t0 = time.monotonic()
    for tok in session.generate(prompt, settings):
        tokens.append(tok.token_id)
    dt = time.monotonic() - t0
    peak = mx.get_peak_memory() / 1e9
    return tokens, dt, peak


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NexVeridian/OmniCoder-9B-4bit")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--kv-start", type=int, default=0,
                   help="Keep first N steps FP16 before quantizing")
    args = p.parse_args()

    prompt = (
        "Write a detailed essay on the history of Python, covering its "
        "creation by Guido van Rossum, major version milestones, design "
        "philosophy, and its rise to dominance in data science and "
        "machine learning. Include specific years and version numbers."
    )

    be = MLXBackend()
    be.load(args.model, None, StaticBudget(auto_detect_limits()))
    session = Session(backend=be, budget=StaticBudget(auto_detect_limits()))

    try:
        print(f"[info] model={args.model} max_tokens={args.max_tokens}")

        print("\n=== baseline (FP16 KV) ===")
        base_tok, base_dt, base_mem = run(session, prompt, args.max_tokens)
        base_tps = len(base_tok) / base_dt
        print(f"tokens={len(base_tok)} time={base_dt:.2f}s tps={base_tps:.1f} peak={base_mem:.2f}GB")

        print("\n=== kv-bits=8 ===")
        tok8, dt8, mem8 = run(session, prompt, args.max_tokens, kv_bits=8, kv_start=args.kv_start)
        tps8 = len(tok8) / dt8
        diverge8 = next((i for i in range(min(len(base_tok), len(tok8))) if base_tok[i] != tok8[i]), None)
        print(f"tokens={len(tok8)} time={dt8:.2f}s tps={tps8:.1f} peak={mem8:.2f}GB "
              f"speedup={base_tps/tps8:.2f}×⁻¹  mem_save={base_mem-mem8:.2f}GB "
              f"diverge_at={'match' if diverge8 is None else diverge8}")

        print("\n=== kv-bits=4 ===")
        tok4, dt4, mem4 = run(session, prompt, args.max_tokens, kv_bits=4, kv_start=args.kv_start)
        tps4 = len(tok4) / dt4
        diverge4 = next((i for i in range(min(len(base_tok), len(tok4))) if base_tok[i] != tok4[i]), None)
        print(f"tokens={len(tok4)} time={dt4:.2f}s tps={tps4:.1f} peak={mem4:.2f}GB "
              f"speedup={base_tps/tps4:.2f}×⁻¹  mem_save={base_mem-mem4:.2f}GB "
              f"diverge_at={'match' if diverge4 is None else diverge4}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
