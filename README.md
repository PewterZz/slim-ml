# slim-ml

An adaptive LLM runtime for running big models on constrained hardware.

Targets the cases current tools do poorly:
- 30-80B MoE models (A3B-style) on a laptop with 6-8GB VRAM + DDR4 RAM
- Dense models on M-series Macs where unified memory pressure fluctuates with your workflow

## Approach

A scheduling/orchestration layer on top of proven kernels (MLX on Mac, llama.cpp on Linux). Value-add is not new matmul — it's:

1. **Adaptive memory budgeting** across VRAM / RAM / NVMe with live pressure awareness
2. **Composable system techniques** — expert caching, speculative decoding, KV cache quantization, layer streaming — each pluggable against a stable backend interface
3. **Telemetry first** — tok/s, bandwidth utilization, cache hit rate, expert routing distribution — can't optimize what you can't measure

## Status

Early scaffold. What works end-to-end today:
- `slim-ml probe` — system probe
- `slim-ml run MODEL` — MLX backend generation with t/s telemetry (OmniCoder 9B etc.)
- `slim-ml bench MODEL` — prompt cache reuse benchmark (~6.4× TTFT on turn 2)
- `slim-ml spec MODEL --draft DRAFT` — speculative decoding with correctness gate and per-round telemetry

What's interface-only (NotImplementedError):
- Expert caching (primary v0 technique)
- KV quantization, layer streaming

## Speculative decoding

A small draft model proposes N tokens; the target model verifies them in one
batched forward pass and accepts the prefix that matches. Works on hybrid
linear+full-attention models (Qwen3.5 family, OmniCoder) via slim-ml's own
`ArraysCache`+`KVCache` snapshot/restore — no mlx-lm fork required.

```bash
slim-ml spec NexVeridian/OmniCoder-9B-4bit \
  --draft mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --num-draft 2 \
  --max-tokens 96 \
  --temperature 0.0    # 0.0 enables built-in correctness gate vs baseline
```

At `--temperature 0.0` the CLI runs baseline + spec back-to-back and asserts
identical token sequences before reporting speedup.

Measured on M3 Air 16GB (April 2026, `num_draft=2`, temp=0):
| Target | Draft | Speedup | Accept |
|---|---|---|---|
| OmniCoder 9B 4bit (hybrid) | Qwen3.5-0.8B 4bit | 1.18× | 55.2% |
| Qwen2.5-Coder-7B 4bit | Qwen2.5-Coder-1.5B 4bit | 1.55× | 62.5% |

The hybrid ceiling (~1.15-1.2×) is lower because 24/32 layers are
linear-attention and can't batch-verify; only the 8 full-attention layers
benefit from parallel verification.

Programmatic use:

```python
from slim_ml.backend import GenerationSettings, MLXBackend
from slim_ml.budget import StaticBudget, auto_detect_limits
from slim_ml.runtime import Session

be = MLXBackend()
be.load("NexVeridian/OmniCoder-9B-4bit", None, StaticBudget(auto_detect_limits()))
be.load_draft("mlx-community/Qwen3.5-0.8B-MLX-4bit")

session = Session(backend=be, budget=StaticBudget(auto_detect_limits()))
for tok in session.generate_speculative(
    "Write a Python bubble sort with type hints.",
    GenerationSettings(max_tokens=128, temperature=0.7),
    num_draft=2,
):
    print(tok.text, end="", flush=True)
```

With `log=<path>.jsonl` on the `Session`, each verify round emits a
`spec_round` event with `{num_draft, num_accept, verify_ms, replay_ms}` for
per-round profiling.

## Install

```bash
pip install -e ".[mlx]"          # Mac
pip install -e ".[llama]"        # Linux/CUDA
pip install -e ".[mlx,llama,dev]" # both + tests
```

## Known limitations (scaffold stage)

- **Mac unified memory and `Tier.VRAM`.** `auto_detect_limits` returns VRAM=0 on
  Apple Silicon (no discrete GPU). For MoE expert caching on M-series the "fast
  tier" is really the Metal device's recommended working-set size, not a separate
  VRAM pool. A future `Tier.UNIFIED` (or platform-aware mapping in `StaticBudget`)
  will address this. Until then, configure budgets manually on Mac.
- **End-to-end MLX path is written but unverified on a real model.** CLI and
  modules import cleanly; `slim-ml run <mlx-model>` has not been executed against
  a live mlx-lm install yet — first-run may surface API signature mismatches.
- **Expert cache + all other techniques raise `NotImplementedError` on attach.**
  The scaffold defines the interfaces; the implementations are the roadmap.
- **macOS memory-pressure signal is deliberately *not* wired.** `StaticBudget` is
  a placeholder; a live-pressure Budget is blocked on measuring whether
  `memory_pressure` / `recommendedMaxWorkingSetSize` signals are stable enough
  at the cadence we'd poll them.
