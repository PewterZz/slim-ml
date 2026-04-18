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
- `slim-ml run MODEL` — MLX backend generation with t/s telemetry (omni-coder 9B etc.)

What's interface-only (NotImplementedError):
- Expert caching (primary v0 technique)
- Speculative decoding, KV quantization, layer streaming

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
