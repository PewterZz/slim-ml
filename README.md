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
- Expert caching migration (hot-set selection + tier moves; observation hook below is shipped)
- KV quantization Technique (runtime knob shipped as `--kv-bits`), layer streaming

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

Measured on M3 Air 16GB (April 2026, temp=0):

| Target | Draft | num_draft | Speedup | Accept |
|---|---|---|---|---|
| OmniCoder 9B 4bit (hybrid) | Qwen3.5-0.8B 4bit | 1 | **1.43×** | 44.8% |
| OmniCoder 9B 4bit (hybrid) | Qwen3.5-0.8B 4bit | 2 | 1.25× | 55.2% |
| OmniCoder 9B 4bit (hybrid) | Qwen3.5-0.8B 4bit | 4 | 0.62× | 46.9% |
| Qwen2.5-Coder-7B 4bit | Qwen2.5-Coder-1.5B 4bit | 2 | 1.55× | 62.5% |

Use `slim-ml spec-sweep` to pick the best `num_draft` for your pair — on
OmniCoder `num_draft=1` wins because drafts cost less to run and rejections
carry a heavy replay tax (24/32 layers are linear-attention and replay
sequentially). The hybrid ceiling (~1.4× at best) stays below the pure-KV
ceiling (1.55×+) for that same reason.

```bash
slim-ml spec-sweep NexVeridian/OmniCoder-9B-4bit \
  --draft mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --num-drafts "1,2,4"
```

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

## KV cache quantization

`slim-ml run --kv-bits {4,8}` wires mlx-lm's quantized KV cache through
`GenerationSettings`. Use `--kv-start N` to keep the first N decoded steps
FP16 before quantizing (reduces early-token drift).

Honest measurement (April 2026, OmniCoder 9B 4bit, 256 tokens, M3 Air):

| setting | tps | peak mem | note |
|---|---|---|---|
| baseline (FP16 KV) | 6.6 | 5.22 GB | |
| `--kv-bits 8` | 4.3 | 5.22 GB | 34% slower, no mem savings |
| `--kv-bits 4` | 4.9 | 5.22 GB | 26% slower, no mem savings |

The knob is shipped but **not a win on this config**: hybrid attention means
only 8/32 layers have a quantizable KV cache, and at 256-token context the
peak is dominated by model weights, not KV. Expected win cases: pure-KV
models (Qwen2.5-Coder), long context (>4k tokens), or tight-memory regimes
where trading tps for headroom unblocks a bigger model.

## Expert caching (Stage 0: observation)

Expert-cache migration is the primary v0 technique — for MoE models where
only ~5% of parameters are active per token, placing hot experts in fast
memory and cold experts in slow memory can unlock models that don't fit in
VRAM. Before committing to the migration design, Stage 0 just profiles
routing to verify the premise is true on the target model.

`MLXBackend.set_route_callback(cb)` patches MLX's `SwitchGLU`/`SwitchMLP`
to emit `(layer_idx, expert_ids, weights)` into `cb` on each MoE forward.
The patch is a class-level monkey-patch with a per-instance sentinel, so
unrelated models loaded in the same process are unaffected. There's a
device-sync cost (`indices.tolist()`) — this is a profiling hook, not a
production hot-path.

Run the profiler on any MoE model:

```bash
python tools/routing_observe.py \
  --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit \
  --max-tokens 128
```

Reports total routing events and global top-{5,10,20,50}% expert capture.
If routing is Zipfian, top-10% should capture >30% of routes — the
threshold we'd need for migration to pay off on a laptop-tier model.

**What's not shipped yet:** the `ExpertCache` technique still raises on
`attach()` for migration. Per-expert weight relocation in MLX is non-trivial
— `SwitchLinear` packs all N experts into one `(num_experts, out, in)`
tensor accessed via `mx.gather_mm`, so "move expert 47 to VRAM" is tensor
surgery, not a pointer swap. And on M3-class unified memory there's no
VRAM/RAM bandwidth gap to exploit anyway — tiered placement is a
gaming-laptop-with-dGPU payoff, not an Apple Silicon one. The observation
scaffold exists on both; migration is gated on a target with the memory
topology to make it worthwhile.

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
