"""Linux/CUDA autoconfig for llama.cpp.

Probes a GGUF file + current GPU state and produces a set of candidate
`llama-server` argument combinations that are expected to fit in VRAM.

The value-add vs guess-and-restart: we read the model's layer count and
file size from the GGUF header, query free VRAM via nvidia-smi, and
estimate per-layer memory cost to pick a starting point for
`--n-gpu-layers`. A sweep can then climb from there until OOM.

This is deliberately coarse. llama.cpp's actual allocator has its own
padding, compute-buffer scaling with batch size, and KV-cache overhead
that the model can't predict perfectly from outside. The job here is to
narrow the search from "binary search across 40 values from scratch" to
"try N, N-2, N-4 starting near the ceiling."
"""
from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# GGUF metadata value type tags (from gguf spec).
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


@dataclass
class GGUFInfo:
    path: Path
    file_size_bytes: int
    architecture: str
    block_count: int  # number of repeating transformer blocks
    context_length: int
    embedding_length: int
    head_count: int
    head_count_kv: int
    # MoE-only (0 / absent for dense models).
    expert_count: int = 0
    expert_used_count: int = 0
    # Raw metadata for debugging / future knobs.
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == _GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _GGUF_TYPE_STRING:
        return _read_string(f)
    if vtype == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if vtype == _GGUF_TYPE_ARRAY:
        (inner_type,) = struct.unpack("<I", f.read(4))
        (length,) = struct.unpack("<Q", f.read(8))
        # Don't materialize huge arrays (tokens, merges) — sample one and skip rest.
        if length > 1024:
            # Skip remainder by reading into void via iterating type size.
            # We take a conservative path: read and discard.
            out = []
            for _ in range(length):
                out.append(_read_value(f, inner_type))
            return out  # caller can drop; kept for completeness
        return [_read_value(f, inner_type) for _ in range(length)]
    raise ValueError(f"unknown GGUF value type: {vtype}")


def read_gguf_info(path: os.PathLike | str) -> GGUFInfo:
    """Parse a GGUF file's metadata header. Does not read tensor data."""
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {p} (magic={magic!r})")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"unsupported GGUF version: {version}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        meta: dict[str, object] = {}
        arch: Optional[str] = None
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = _read_value(f, vtype)
            # Stash scalars; skip huge arrays from the returned dict.
            if not (isinstance(val, list) and len(val) > 64):
                meta[key] = val
            if key == "general.architecture":
                arch = str(val)

        if arch is None:
            raise ValueError(f"GGUF metadata missing general.architecture: {p}")

        def m(key: str, default=None):
            return meta.get(f"{arch}.{key}", default)

        return GGUFInfo(
            path=p,
            file_size_bytes=size,
            architecture=arch,
            block_count=int(m("block_count", 0) or 0),
            context_length=int(m("context_length", 0) or 0),
            embedding_length=int(m("embedding_length", 0) or 0),
            head_count=int(m("attention.head_count", 0) or 0),
            head_count_kv=int(m("attention.head_count_kv", 0) or 0),
            expert_count=int(m("expert_count", 0) or 0),
            expert_used_count=int(m("expert_used_count", 0) or 0),
            metadata=meta,
        )


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int

    @property
    def used_mib(self) -> int:
        return self.total_mib - self.free_mib


def query_nvidia_gpus() -> list[GPUInfo]:
    """Return a list of visible NVIDIA GPUs. Empty list if none / nvidia-smi missing."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=False, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    gpus: list[GPUInfo] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                total_mib=int(parts[2]),
                free_mib=int(parts[3]),
            ))
        except ValueError:
            continue
    return gpus


@dataclass
class ServerConfig:
    """A candidate llama-server launch config. Ordered by expected quality."""
    n_gpu_layers: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    kv_type_k: str = "q4_0"
    kv_type_v: str = "q4_0"
    flash_attn: bool = True
    spec_type: Optional[str] = "ngram-mod"
    # MoE-specific: keep expert weights on CPU for the first N layers.
    # None = flag not passed (dense models); int = passed as --n-cpu-moe.
    n_cpu_moe: Optional[int] = None
    # Expected VRAM usage (MiB) — for sorting / filtering, approximate.
    est_vram_mib: int = 0
    notes: str = ""

    def to_args(self, model_path: str, port: int = 8080, host: str = "127.0.0.1",
                parallel: int = 1, cache_reuse: int = 256,
                threads: Optional[int] = None) -> list[str]:
        args = [
            "-m", model_path,
            "--host", host, "--port", str(port),
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "--parallel", str(parallel),
            "--cache-reuse", str(cache_reuse),
            "-ctk", self.kv_type_k,
            "-ctv", self.kv_type_v,
        ]
        if self.flash_attn:
            args += ["--flash-attn", "on"]
        if self.spec_type:
            args += ["--spec-type", self.spec_type]
        if self.n_cpu_moe is not None:
            args += ["--n-cpu-moe", str(self.n_cpu_moe)]
        if threads is not None:
            args += ["--threads", str(threads)]
        return args


# Rough per-MiB cost estimates. These are empirical defaults; sweep confirms.
# KV-cache q4_0 per-layer per-token for typical 8-head GQA: ~4 KiB/tok.
# For hybrid attention (Qwen3.5 family), only ~25% of layers keep full KV.
_KV_BYTES_PER_LAYER_PER_TOK_Q4 = 4 * 1024
_KV_BYTES_PER_LAYER_PER_TOK_Q8 = 8 * 1024
_KV_BYTES_PER_LAYER_PER_TOK_F16 = 16 * 1024

# Compute buffer grows roughly linearly with batch size; this constant gets
# tuned by the sweep. Mib per (batch=512 step, layer) typical on CUDA.
_COMPUTE_MIB_PER_512_BATCH_PER_LAYER = 14


def _kv_cache_mib(ctx_size: int, n_layers_on_gpu: int,
                  kv_type: str, head_count_kv: int,
                  arch: str) -> int:
    # Hybrid (Qwen3.5/Qwen3.6): only `full_attention_interval` layers keep full KV.
    # For other archs, every layer has KV.
    if kv_type.startswith("q4"):
        per_tok = _KV_BYTES_PER_LAYER_PER_TOK_Q4
    elif kv_type.startswith("q8"):
        per_tok = _KV_BYTES_PER_LAYER_PER_TOK_Q8
    else:
        per_tok = _KV_BYTES_PER_LAYER_PER_TOK_F16
    # Adjust for GQA: fewer KV heads -> smaller cache.
    if head_count_kv > 0:
        # Rough normalization to GQA-8 baseline used in the constant.
        per_tok = int(per_tok * max(1, head_count_kv) / 8)
    hybrid_fraction = 0.25 if arch.startswith("qwen3") else 1.0
    effective_layers = max(1, int(n_layers_on_gpu * hybrid_fraction))
    return (per_tok * ctx_size * effective_layers) // (1024 * 1024)


def estimate_vram_mib(info: GGUFInfo, cfg: ServerConfig) -> int:
    """Approximate VRAM usage for a given config. Coarse but monotonic."""
    if info.block_count <= 0:
        return 0
    model_mib = info.file_size_bytes // (1024 * 1024)
    # Heuristic: +2 for output/embedding layers beyond block_count.
    layer_share = min(1.0, cfg.n_gpu_layers / (info.block_count + 2))
    weights_mib = int(model_mib * layer_share)
    kv_mib = _kv_cache_mib(
        cfg.ctx_size, cfg.n_gpu_layers, cfg.kv_type_k,
        info.head_count_kv or 8, info.architecture,
    )
    compute_mib = int(
        _COMPUTE_MIB_PER_512_BATCH_PER_LAYER
        * (cfg.batch_size / 512)
        * cfg.n_gpu_layers
    )
    return weights_mib + kv_mib + compute_mib


def suggest_configs(info: GGUFInfo, free_vram_mib: int,
                    vram_safety_mib: int = 200,
                    ctx_target: int = 32000) -> list[ServerConfig]:
    """Produce a ranked list of configs expected to fit within free VRAM.

    Dispatches to dense or MoE strategy based on `info.is_moe`. Safety margin
    (default 200 MiB) covers CUDA context, driver overhead, small kernels.
    """
    if info.is_moe:
        return _suggest_moe_configs(info, free_vram_mib, vram_safety_mib, ctx_target)
    return _suggest_dense_configs(info, free_vram_mib, vram_safety_mib, ctx_target)


def _suggest_dense_configs(info: GGUFInfo, free_vram_mib: int,
                           vram_safety_mib: int, ctx_target: int
                           ) -> list[ServerConfig]:
    budget = max(0, free_vram_mib - vram_safety_mib)
    max_layers = info.block_count + 2
    candidates: list[ServerConfig] = []

    batch_pairs = [(512, 512), (256, 256), (128, 128)]
    ctx_options = [ctx_target, min(ctx_target, 16000), min(ctx_target, 8000)]

    for ctx in ctx_options:
        for batch, ubatch in batch_pairs:
            lo, hi = 0, max_layers
            best_fit = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                trial = ServerConfig(
                    n_gpu_layers=mid, ctx_size=ctx,
                    batch_size=batch, ubatch_size=ubatch,
                )
                est = estimate_vram_mib(info, trial)
                if est <= budget:
                    best_fit = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_fit == 0:
                continue
            cfg = ServerConfig(
                n_gpu_layers=best_fit, ctx_size=ctx,
                batch_size=batch, ubatch_size=ubatch,
                est_vram_mib=estimate_vram_mib(
                    info, ServerConfig(
                        n_gpu_layers=best_fit, ctx_size=ctx,
                        batch_size=batch, ubatch_size=ubatch,
                    )
                ),
                notes=f"bisect fit @ ctx={ctx} batch={batch}",
            )
            candidates.append(cfg)

    seen: set[tuple[int, int, int]] = set()
    unique: list[ServerConfig] = []
    for c in candidates:
        key = (c.n_gpu_layers, c.ctx_size, c.batch_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    unique.sort(key=lambda c: (c.n_gpu_layers, c.ctx_size, c.batch_size), reverse=True)
    return unique


def _suggest_moe_configs(info: GGUFInfo, free_vram_mib: int,
                         vram_safety_mib: int, ctx_target: int
                         ) -> list[ServerConfig]:
    """For MoE models the knob is --n-cpu-moe (higher = more experts on CPU).

    Strategy: keep all attention on GPU (`-ngl 999`). Sweep n_cpu_moe from
    `block_count` (all experts on CPU) down toward 0 (all on GPU), picking
    the lowest value whose VRAM estimate still fits the budget. Smaller
    n_cpu_moe = more expert weights on GPU = faster generation.
    """
    budget = max(0, free_vram_mib - vram_safety_mib)
    n_layers = info.block_count

    # Estimate bytes per layer's expert weights (model_bytes minus attention/embedding).
    # Rough heuristic: ~85% of an A3B MoE's weights are expert FFN tensors.
    expert_share = 0.85
    model_mib = info.file_size_bytes // (1024 * 1024)
    non_expert_mib = int(model_mib * (1 - expert_share))
    per_layer_expert_mib = int((model_mib * expert_share) / max(1, n_layers))

    batch_pairs = [(1024, 1024), (512, 512), (256, 256)]
    ctx_options = [ctx_target, min(ctx_target, 32000), min(ctx_target, 16000)]

    candidates: list[ServerConfig] = []
    for ctx in ctx_options:
        for batch, ubatch in batch_pairs:
            # Fixed VRAM cost: attention/embeddings + KV + compute buffer.
            kv_mib = _kv_cache_mib(ctx, n_layers, "q4_0",
                                    info.head_count_kv or 8, info.architecture)
            compute_mib = int(
                _COMPUTE_MIB_PER_512_BATCH_PER_LAYER
                * (batch / 512) * n_layers * 1.5
            )
            fixed = non_expert_mib + kv_mib + compute_mib
            if fixed > budget:
                continue
            # How many expert layers can we pull onto GPU?
            remaining = budget - fixed
            experts_on_gpu = min(n_layers, remaining // max(1, per_layer_expert_mib))
            n_cpu_moe = max(0, n_layers - experts_on_gpu)
            est = fixed + experts_on_gpu * per_layer_expert_mib
            cfg = ServerConfig(
                n_gpu_layers=999,
                ctx_size=ctx, batch_size=batch, ubatch_size=ubatch,
                n_cpu_moe=n_cpu_moe,
                est_vram_mib=est,
                notes=f"moe: {experts_on_gpu}/{n_layers} expert-layers on GPU "
                      f"@ ctx={ctx} batch={batch}",
            )
            candidates.append(cfg)

    # Dedupe on (n_cpu_moe, ctx, batch).
    seen: set[tuple[int, int, int]] = set()
    unique: list[ServerConfig] = []
    for c in candidates:
        key = (c.n_cpu_moe or 0, c.ctx_size, c.batch_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    # Rank: fewer n_cpu_moe = more experts on GPU = faster gen.
    unique.sort(key=lambda c: (-(n_layers - (c.n_cpu_moe or 0)),
                               -c.ctx_size, -c.batch_size))
    return unique
