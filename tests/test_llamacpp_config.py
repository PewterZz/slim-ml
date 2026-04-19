"""Smoke tests for the Linux/CUDA autoconfig scaffold.

These are pure unit tests — they don't launch llama-server. Real
end-to-end validation lives in `slim-ml lc-sweep` (run manually).
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import pytest

from slim_ml.llamacpp_config import (
    GGUFInfo,
    ServerConfig,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT32,
    estimate_vram_mib,
    read_gguf_info,
    suggest_configs,
)


def _write_kv_string(f, key: str, val: str) -> None:
    f.write(struct.pack("<Q", len(key)) + key.encode())
    f.write(struct.pack("<I", _GGUF_TYPE_STRING))
    f.write(struct.pack("<Q", len(val)) + val.encode())


def _write_kv_u32(f, key: str, val: int) -> None:
    f.write(struct.pack("<Q", len(key)) + key.encode())
    f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", val))


def _make_fake_gguf(path: Path, *, arch: str, block_count: int,
                    ctx: int = 4096, embed: int = 2048,
                    heads: int = 16, heads_kv: int = 4) -> None:
    kv = [
        ("general.architecture", "string", arch),
        (f"{arch}.block_count", "u32", block_count),
        (f"{arch}.context_length", "u32", ctx),
        (f"{arch}.embedding_length", "u32", embed),
        (f"{arch}.attention.head_count", "u32", heads),
        (f"{arch}.attention.head_count_kv", "u32", heads_kv),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))      # version
        f.write(struct.pack("<Q", 0))      # tensor_count
        f.write(struct.pack("<Q", len(kv)))  # metadata_kv_count
        for key, typ, val in kv:
            if typ == "string":
                _write_kv_string(f, key, val)  # type: ignore[arg-type]
            elif typ == "u32":
                _write_kv_u32(f, key, int(val))
            else:
                raise AssertionError(typ)
        # Add some bulk so file_size_bytes is non-trivial (fakes model weight size).
        f.write(b"\x00" * (8 * 1024 * 1024))


def test_gguf_parser_reads_block_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf"
        _make_fake_gguf(p, arch="qwen35", block_count=32)
        info = read_gguf_info(p)
    assert info.architecture == "qwen35"
    assert info.block_count == 32
    assert info.context_length == 4096
    assert info.head_count == 16
    assert info.head_count_kv == 4
    assert info.file_size_bytes > 8 * 1024 * 1024


def test_gguf_parser_rejects_non_gguf(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOTA" + b"\x00" * 32)
    with pytest.raises(ValueError):
        read_gguf_info(p)


def test_estimate_vram_monotonic_in_layers(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _make_fake_gguf(p, arch="qwen35", block_count=32, ctx=32000)
    info = read_gguf_info(p)
    e1 = estimate_vram_mib(info, ServerConfig(n_gpu_layers=8, ctx_size=4096,
                                               batch_size=512, ubatch_size=512))
    e2 = estimate_vram_mib(info, ServerConfig(n_gpu_layers=24, ctx_size=4096,
                                               batch_size=512, ubatch_size=512))
    e3 = estimate_vram_mib(info, ServerConfig(n_gpu_layers=32, ctx_size=4096,
                                               batch_size=512, ubatch_size=512))
    assert e1 < e2 < e3


def test_suggest_configs_respects_budget(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _make_fake_gguf(p, arch="qwen35", block_count=32)
    info = read_gguf_info(p)
    cfgs = suggest_configs(info, free_vram_mib=6144, vram_safety_mib=200,
                           ctx_target=32000)
    assert cfgs, "should produce at least one candidate"
    for c in cfgs:
        assert c.est_vram_mib <= 6144 - 200
    # Ranking invariant: higher ngl first.
    for a, b in zip(cfgs, cfgs[1:]):
        assert a.n_gpu_layers >= b.n_gpu_layers


def test_suggest_configs_tiny_budget_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _make_fake_gguf(p, arch="qwen35", block_count=32)
    info = read_gguf_info(p)
    cfgs = suggest_configs(info, free_vram_mib=50, vram_safety_mib=200,
                           ctx_target=32000)
    assert cfgs == []


def test_server_config_to_args_roundtrip() -> None:
    c = ServerConfig(n_gpu_layers=30, ctx_size=16000, batch_size=256, ubatch_size=256)
    args = c.to_args(model_path="/tmp/m.gguf", port=8090, threads=8)
    assert "--n-gpu-layers" in args
    assert args[args.index("--n-gpu-layers") + 1] == "30"
    assert args[args.index("--batch-size") + 1] == "256"
    assert args[args.index("--ctx-size") + 1] == "16000"
    assert "--flash-attn" in args and args[args.index("--flash-attn") + 1] == "on"
    assert "-ctk" in args and args[args.index("-ctk") + 1] == "q4_0"
    # Dense config omits --n-cpu-moe.
    assert "--n-cpu-moe" not in args


def _make_fake_moe_gguf(path: Path, *, block_count: int = 40,
                        expert_count: int = 256) -> None:
    arch = "qwen35moe"
    kv = [
        ("general.architecture", "string", arch),
        (f"{arch}.block_count", "u32", block_count),
        (f"{arch}.context_length", "u32", 262144),
        (f"{arch}.embedding_length", "u32", 2048),
        (f"{arch}.attention.head_count", "u32", 16),
        (f"{arch}.attention.head_count_kv", "u32", 2),
        (f"{arch}.expert_count", "u32", expert_count),
        (f"{arch}.expert_used_count", "u32", 8),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", len(kv)))
        for key, typ, val in kv:
            if typ == "string":
                _write_kv_string(f, key, val)  # type: ignore[arg-type]
            else:
                _write_kv_u32(f, key, int(val))
        # Simulate large MoE file size so the expert-share split is meaningful.
        f.write(b"\x00" * (1 * 1024 * 1024 * 1024))  # ~1 GiB placeholder


def test_moe_detection(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _make_fake_moe_gguf(p, block_count=40, expert_count=256)
    info = read_gguf_info(p)
    assert info.is_moe
    assert info.expert_count == 256
    assert info.expert_used_count == 8


def test_moe_suggest_configs_sets_n_cpu_moe(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _make_fake_moe_gguf(p, block_count=40, expert_count=256)
    info = read_gguf_info(p)
    cfgs = suggest_configs(info, free_vram_mib=6144, vram_safety_mib=200,
                           ctx_target=32000)
    assert cfgs, "MoE path should yield candidates"
    for c in cfgs:
        # MoE strategy pins ngl to 999 and uses --n-cpu-moe for placement.
        assert c.n_gpu_layers == 999
        assert c.n_cpu_moe is not None
        assert 0 <= c.n_cpu_moe <= info.block_count
        # Serialized args should carry --n-cpu-moe.
        args = c.to_args(model_path="/tmp/x.gguf")
        assert "--n-cpu-moe" in args
    # Rank invariant: fewer n_cpu_moe (more experts on GPU) listed first.
    for a, b in zip(cfgs, cfgs[1:]):
        assert (a.n_cpu_moe or 0) <= (b.n_cpu_moe or 0) or a.ctx_size >= b.ctx_size
