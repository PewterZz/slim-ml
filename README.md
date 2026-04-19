# slim-ml

An adaptive LLM runtime for running big models on constrained hardware.

Targets the cases current tools do poorly:
- 30-80B MoE models (A3B-style) on a laptop with 6-8GB VRAM + DDR4 RAM
- Dense models on M-series Macs where unified memory pressure fluctuates with your workflow

## Quickstart

Pick the path that matches your machine. Both use the same `slim-ml` CLI.

### Mac (MLX)

```bash
pip install -e ".[mlx]"

# 1. Fastest decoding on coding/edit workloads (no draft model needed)
slim-ml pld mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

# 2. Best speedup if you have a small draft model of the same family
slim-ml spec mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit

# 3. Plain generation with good defaults
slim-ml run <mlx-model>
```

Don't know which draft pair works for your target? `slim-ml spec-sweep` tries
`num_draft ∈ {1,2,4,6,8}` and prints a ranked table.

### Linux + NVIDIA GPU (llama.cpp)

```bash
pip install -e ".[llama]"

# Find the highest-tps llama-server config that fits your GPU.
# Prints a ready-to-paste llama-server command with the winning settings.
slim-ml lc-sweep models/YOUR_MODEL.gguf \
  --server-bin ./build-linux/bin/llama-server

# Dry-run: inspect candidate configs without launching anything.
slim-ml lc-probe models/YOUR_MODEL.gguf
```

Measured wins on a RTX 3060 Laptop 6 GB: **+35%** on OmniCoder 9B (24 → 32.3 t/s),
**+22%** on Qwen3.6-35B-A3B MoE (26 → 31.6 t/s). Detail + tables in the
[Linux/CUDA autoconfig](#linux--cuda-autoconfig-lc-probe-lc-sweep) section.

### Which technique for which workload?

| your situation | reach for |
|---|---|
| Mac, writing new code | `spec` (target + small draft of same family) |
| Mac, editing / refactoring existing code | `pld` (draftless; wins from prompt repetition) |
| Mac, long-context edit + wants correctness gate | `pld --temperature 0.0` (auto baseline diff) |
| Linux laptop GPU, just want best t/s | `lc-sweep` then paste the winning command |
| Linux laptop GPU, MoE model (A3B-style) | `lc-sweep` — it's MoE-aware, sweeps `--n-cpu-moe` |
| Investigating an MoE's routing skew | `tools/routing_observe.py` |

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
- `slim-ml pld MODEL` — draftless Prompt Lookup Decoding (n-gram match against context) with correctness gate
- `slim-ml hybrid MODEL --draft DRAFT` — PLD first, draft model on no-match fallback (investigated; see results below)

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

## Prompt Lookup Decoding (PLD)

Same speculative verifier, but drafts come from n-gram matching the
prompt + already-generated tokens instead of a draft model. Zero extra
parameters, zero draft-model load time. Win condition is context-local
repetition (field names, repeated method structure, refactors) — exactly
the shape of most coding and editing workloads.

```bash
slim-ml pld mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "Implement to_dict, from_dict, __repr__, __eq__ for this dataclass..." \
  --num-draft 8 \
  --max-tokens 192 \
  --temperature 0.0
```

Measured on M3 Air 16GB (April 2026, Qwen2.5-Coder-7B 4bit, temp=0, 192
tokens, num_draft=8):

| mode | tps | speedup | correctness |
|---|---|---|---|
| baseline | 17.7 | 1.00× | — |
| pld      | 24.4 | **1.38×** | 192/192 tokens identical |

Per-round telemetry for that run: 87 rounds, 63 of which found no n-gram
match and degraded to a plain step; of the 24 rounds that did propose
drafts, 63.5% of drafts were accepted. Ceiling rises on edit-shaped
workloads (rename refactor, docstring pass) where the model output
heavily mirrors the prompt. This is the first PLD port to MLX that I'm
aware of — PROMTEC's 4.23× figure (HumanEval edit tasks) is the theoretical
ceiling, not what generate-from-scratch on a laptop will hit.

**Edit-workload measurement (April 2026, ~1500-token prompt asking for
type-annotation pass on existing code, temp=0, num_draft=8):**

| output tokens | baseline tps | PLD tps | speedup | from_draft |
|---|---|---|---|---|
| 192 | 2.6 | 4.2 | **1.60×** | 70.3% |
| 384 | 4.4 | 4.7 | 1.07× | 68.8% |

The 1.60× at 192 tokens confirms PLD's ceiling is higher on edit-shaped
workloads (70% of output tokens come from prompt n-grams). But the
speedup fades as output extends past the mirrored portion — at 384 tokens
the model is generating novel code and PLD's hit rate drops. Per-round
telemetry at 192 tokens: 58 rounds, 33 with draft, 25 no match. Bimodal
distribution — 14 rounds accepted all 8 drafts, 36 accepted 0.

**Load sensitivity:** under heavy system contention (baseline ~6 tps
instead of ~18 tps), PLD degrades to 0.76-0.80× vs baseline — the
speedup flips sign. On a laptop with browser/other apps running, PLD may
perform worse than baseline. The 1.38× and 1.60× numbers are clean-room
upper bounds, not floors.

The implementation (`src/slim_ml/prompt_lookup.py` + `speculative_step_pld`
in `spec_decode.py`) shares the same snapshot/restore machinery as the
draft-model path, so it works on hybrid-attention models too.

## Hybrid PLD + draft-model (investigated, negative on unified memory)

Hypothesis: PLD telemetry showed 72% of rounds (63/87) found no n-gram
match and degraded to a plain step. Stacking PLD first, falling back to
a draft model on no-match, should fill that bottleneck using the same
verifier — expected aggregate 1.5-1.8×.

```bash
slim-ml hybrid mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --num-draft 8 --max-tokens 192 --temperature 0.0
```

Measured on M3 Air 16GB (April 2026, Qwen2.5-Coder-7B 4bit target +
1.5B 4bit draft, temp=0, 192 tokens, num_draft=8, back-to-back baseline
vs hybrid in a single command):

| workload | baseline | hybrid | speedup | correctness |
|---|---|---|---|---|
| dataclass (open gen) | 19.1 tps | 19.3 tps | 1.01× | 192/192 identical |
| rename refactor | 13.4 tps | 10.0 tps | 0.74× | 165/165 prefix match, hybrid stops 1 token before baseline's `<|endoftext|>` (same EOS-boundary behavior as `spec` alone) |

Compare against PLD-alone on the same workload: **1.38×** on dataclass
and 1.01× on rename. Hybrid is **neutral-to-negative vs PLD alone** on
MLX unified memory.

Per-round telemetry on the dataclass run: 36 rounds total (vs PLD-alone's
87), of which 25 were draft-model fallback at 75% accept rate. Each
draft-model round costs ~190ms verify + ~55ms draft gen = ~245ms for
~6 accepted tokens. PLD-alone's cheap "plain step" fallback at ~50ms
beats that on unified memory, where the draft model has no bandwidth
advantage over the target. Draft-model speculation amortizes when draft
forward passes are ~10× cheaper than target passes (dGPU with VRAM/RAM
gap); on M3 unified memory they're the same speed, so the draft work
is mostly overhead.

Hybrid's correctness holds (gate passes on both runs). The 165/166 rename
length difference matches `spec`-alone's EOS-boundary behavior — both
stop one token before baseline emits `<|endoftext|>`. Content prefix is
identical.

Shipped as an opt-in CLI command to keep the negative result
reproducible. Do not reach for it; reach for `pld` (no draft model) or
`spec` (with draft, copy-from-prompt workloads excluded).

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

**Long-context correctness gate (April 2026, Qwen2.5-Coder-7B 4bit,
~2800-token prompt, 256 output tokens, temp=0):**

| setting | prefill (s) | mean tps | correctness |
|---|---|---|---|
| baseline (FP16 KV) | 52.9 | 4.71 | — |
| `--kv-bits 8` | 41.3 | 3.55 | **FAIL** — diverges from baseline at ~token 9 |
| `--kv-bits 4` | 40.5 | 4.43 | **FAIL** — diverges from baseline at token 1 |

KV-quant saves ~12s on prefill (smaller cache = less bandwidth) but
**corrupts output at temp=0** on a 2800-token context. The quantization
error accumulates enough over 2800 prefill tokens to shift the model's
first decode choice. On short context (256 tokens, the earlier OmniCoder
test) the drift was tolerable; on long context it's not. KV-quant on
MLX is currently a memory-headroom trade, not a speed win, and it's
unsafe for correctness-sensitive workloads beyond ~1K context.

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

Measured on OLMoE-1B-7B-0125-Instruct-4bit (16 MoE layers × 64 experts,
128-token multi-topic prompt, April 2026):

| top-k% experts | routes captured |
|---|---|
| 5% | 16.4% |
| 10% | 26.2% |
| 20% | 41.9% |
| 50% | 75.8% |

Skew is real (top-10% = 2.6× uniform) but below the 30% threshold.
Per-layer top-10% ranges 17–35%, so migration might still pay off on the
deeper layers (layer 14 alone hits 35%). This is the whole point of Stage 0
— quantify the premise before building the migration. Next step is to
repeat on a longer, code-heavy sample (where routing may concentrate
more) and on Qwen3-MoE before committing to migration.

**What's not shipped yet:** the `ExpertCache` technique still raises on
`attach()` for migration. Per-expert weight relocation in MLX is non-trivial
— `SwitchLinear` packs all N experts into one `(num_experts, out, in)`
tensor accessed via `mx.gather_mm`, so "move expert 47 to VRAM" is tensor
surgery, not a pointer swap. And on M3-class unified memory there's no
VRAM/RAM bandwidth gap to exploit anyway — tiered placement is a
gaming-laptop-with-dGPU payoff, not an Apple Silicon one. The observation
scaffold exists on both; migration is gated on a target with the memory
topology to make it worthwhile.

The **llama.cpp/CUDA path** is a different story — the RTX 3060 laptop has a
~336 GB/s VRAM vs ~4 GB/s PCIe Gen3 x4 bandwidth gap that tiered placement can
exploit. RFC drafted at [`docs/expert-cache-rfc.md`](docs/expert-cache-rfc.md):
LRU cache of hot experts in GPU VRAM in front of the current `--n-cpu-moe`
CPU-resident tensors, intercepted at `ggml_cuda_mul_mat_id`. Gated on a
premise-check showing Qwen3.6-35B-A3B routing skew beats the 30% top-10%
threshold on a coding workload — OLMoE came in at 26.2% so the threshold is
real, not ornamental.

## Linux / CUDA autoconfig (`lc-probe`, `lc-sweep`)

The MLX path has the interesting kernels; the Linux path has a different
problem — `llama-server` on a 6-8 GB consumer card is a dance of
`--n-gpu-layers`, `--batch-size`, `--ctx-size`, and `-ctk/-ctv` flags where
the wrong combo fails to load, loads but OOMs mid-generation, or loads but
leaves 2 GB of VRAM unused. The sweep tool turns that from restart-and-guess
into probe-and-measure.

```bash
# 1. Inspect a GGUF and see what configs would fit in VRAM
slim-ml lc-probe models/OmniCoder-9B-Q4_K_M.gguf --ctx-target 32000

# 2. Actually launch each candidate, time 96 tokens, kill, report
slim-ml lc-sweep models/OmniCoder-9B-Q4_K_M.gguf \
  --server-bin ./build-linux/bin/llama-server \
  --ctx-target 32000 --max-configs 6 \
  --log-dir /tmp/sweep-logs
```

`lc-probe` reads the GGUF header (block count, hidden size, head count) and
queries `nvidia-smi` for total VRAM, then bisects `--n-gpu-layers` against a
coarse VRAM estimator at three (batch, ctx) points. The estimator is
intentionally approximate — its only job is narrowing the search from 40
values to 6.

`lc-sweep` runs each candidate end-to-end: fresh `llama-server` process,
wait for `/health`, warmup completion, timed completion via `/completion`
(reads `timings.predicted_per_second`). Kill the server (SIGTERM then
SIGKILL) before moving on so CUDA contexts don't stack. Each run costs
~30-90 s, mostly model load. Per-run stderr lands in `--log-dir` so OOM
crashes can be inspected.

The tool is deliberately blunt — no binary-search across gpu-layers to find
the exact maximum. Candidates come pre-bisected from the VRAM estimator,
sweep just validates which actually launch and which deliver the most t/s.

**Measured on this laptop** (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, DDR4-3200):

_OmniCoder 9B Q4_K_M (dense, hybrid attention), 96 tokens, `--spec-type ngram-mod`_:

| ngl | ctx | batch | tps | prefill tps | vram peak |
|-----|-----|-------|-----|-------------|-----------|
| 34 (all) | 16000 | 256 | **32.3** | 386 | 5878 |
| 34 (all) | 16000 | 128 | 31.7 | 381 | 5752 |
| 34 (all) | 8000 | 256 | 31.5 | 382 | 5808 |
| 33 | 32000 | 256 | 30.9 | 381 | 6014 |

User's prior hand-tuned config (ngl=28, ctx=32000, batch=512) was 24 t/s.
Sweep-picked winner is **32.3 t/s — +35%**. The lever the sweep found:
drop ctx 32000→16000 to free ~140 MiB of KV cache, which buys enough
headroom to land all 34 layers on GPU. With no CPU-resident layers, the
bottleneck moves from DDR4 (~40 GB/s) to VRAM (~336 GB/s).

_Qwen3.6-35B-A3B UD-Q2_K_XL (MoE, 256 experts / 8 active), 64 tokens,
`--spec-type ngram-mod`_:

| n_cpu_moe | ctx | batch | tps | prefill tps | vram peak |
|-----------|-----|-------|-----|-------------|-----------|
| 26 (14 on GPU) | 16000 | 256 | **31.6** | 73 | 5926 |
| 27 (13 on GPU) | 32000 | 256 | 29.5 | 64 | 5766 |
| 28 (12 on GPU) | 32000 | 512 | 28.3 | 60 | 5768 |
| 28 (12 on GPU) | 16000 | 512 | 27.3 | 59 | 5682 |
| 31 (9 on GPU)  | 16000 | 1024 | 25.9 | 54 | 5434 |

User's prior config pinned all 40 MoE expert layers to CPU
(`--n-cpu-moe 33` on a 40-layer model effectively means "all experts on
CPU"). The sweep says landing 14 of 40 expert layers on GPU is worth
**+22% t/s** at the cost of dropping ctx to 16K. Prefill stays modest at
MoE (~70 t/s) because only 8/256 experts activate per token — compute
is sparse, memory bandwidth for the active expert weights is still the
binding constraint.

The sweep also encodes some honest tradeoffs. Bigger batch (`--batch-size
1024`) shaves 20% off generation tps on MoE because the compute-buffer
scratch displaces expert weights off GPU. The prior hand-tuned config ran
128K context — worth it if you need long context, but the sweep shows
what you trade for it.

### Why not just binary-search n-gpu-layers?

Because the variable you actually want to pick is closer to "which (ngl,
batch, ctx, kv-quant) combination maximizes t/s without crashing on the
second request." Load succeeds is necessary but not sufficient — the
user hit a case earlier tonight where load + first request succeeded at
`--batch-size 1024 --n-gpu-layers 32`, then the second request OOM'd in
flash attention scratch. The sweep catches that by requiring two
completions (warmup + measured).

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
