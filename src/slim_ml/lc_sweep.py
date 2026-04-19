"""Sweep llama-server configurations to find the max-tps config that doesn't OOM.

Strategy:
  - Generate candidate ServerConfigs from `suggest_configs` (GGUF-derived).
  - For each candidate, launch `llama-server` on a sweep port, wait for /health,
    send a warmup completion, then a measured completion.
  - Record tokens/sec, OOM, load-time. Kill server between runs.
  - Report a ranked table; return the winner.

Each run is isolated: a fresh server process means fresh CUDA context, no
cross-run VRAM leakage. Slow (load time dominates) but reliable.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .llamacpp_config import (
    GGUFInfo,
    ServerConfig,
    query_nvidia_gpus,
    read_gguf_info,
    suggest_configs,
)


DEFAULT_PROMPT = (
    "Write a Python function that takes a list of integers and returns the "
    "median. Handle even-length lists by averaging the two middle elements. "
    "Include a docstring and type hints."
)


@dataclass
class SweepResult:
    config: ServerConfig
    ok: bool
    tps: float = 0.0
    prefill_tps: float = 0.0
    load_s: float = 0.0
    first_token_s: float = 0.0
    error: str = ""
    tokens_predicted: int = 0
    tokens_prompt: int = 0
    vram_peak_mib: int = 0


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Match llama-server's SO_REUSEADDR so TIME_WAIT sockets don't block us.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_port_free(port: int, timeout_s: float = 20.0) -> bool:
    """Poll until the port is bindable or timeout. Handles TIME_WAIT cleanup."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_free(port):
            return True
        time.sleep(0.5)
    return False


def _wait_health(port: int, timeout_s: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError):
            pass
        time.sleep(0.5)
    return False


def _post_completion(port: int, prompt: str, n_predict: int,
                     temperature: float = 0.0, timeout_s: float = 300.0) -> dict:
    url = f"http://127.0.0.1:{port}/completion"
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "cache_prompt": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def _vram_used_mib(gpu_index: int = 0) -> int:
    gpus = query_nvidia_gpus()
    for g in gpus:
        if g.index == gpu_index:
            return g.used_mib
    return 0


def _wait_vram_released(gpu_index: int = 0, idle_threshold_mib: int = 500,
                         timeout_s: float = 30.0) -> bool:
    """Poll nvidia-smi until VRAM usage drops below threshold, or timeout.

    Used between sweep runs to ensure the previous CUDA context has torn down
    before we launch the next server (avoids spurious OOMs from stacked contexts).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = _vram_used_mib(gpu_index)
        if used <= idle_threshold_mib:
            return True
        time.sleep(0.5)
    return False


def run_one(
    server_bin: str,
    model_path: str,
    cfg: ServerConfig,
    port: int = 8090,
    prompt: str = DEFAULT_PROMPT,
    n_predict: int = 96,
    threads: Optional[int] = None,
    log_dir: Optional[Path] = None,
    startup_timeout_s: float = 180.0,
) -> SweepResult:
    """Launch one server, measure, kill. Returns a SweepResult."""
    if not _wait_port_free(port, timeout_s=20.0):
        return SweepResult(config=cfg, ok=False,
                           error=f"port {port} busy after 20s — stop existing llama-server")

    args = [server_bin] + cfg.to_args(
        model_path=model_path, port=port, host="127.0.0.1",
        threads=threads,
    )
    log_f = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ngl{cfg.n_gpu_layers}_ctx{cfg.ctx_size}_b{cfg.batch_size}"
        log_f = open(log_dir / f"{stem}.log", "w")

    t_launch = time.monotonic()
    # Use a fresh process group so we can kill the whole tree cleanly.
    proc = subprocess.Popen(
        args,
        stdout=log_f or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        if not _wait_health(port, timeout_s=startup_timeout_s):
            # Server died or never came up — OOM or other fatal.
            rc = proc.poll()
            err_msg = "server failed to come up"
            if rc is not None:
                err_msg = f"server exited early rc={rc}"
            if log_f:
                log_f.flush()
                log_path = log_dir / f"ngl{cfg.n_gpu_layers}_ctx{cfg.ctx_size}_b{cfg.batch_size}.log"
                tail = _read_tail(log_path, 4000)
                if "out of memory" in tail.lower() or "cudaMalloc failed" in tail:
                    err_msg = "CUDA OOM during load"
                elif "failed to allocate" in tail.lower():
                    err_msg = "allocation failure during load"
            return SweepResult(config=cfg, ok=False, error=err_msg,
                               load_s=time.monotonic() - t_launch)

        load_s = time.monotonic() - t_launch
        # Warmup — not measured. Ensures caches, compiled kernels, etc.
        try:
            _post_completion(port, prompt, n_predict=8, temperature=0.0, timeout_s=120)
        except Exception as e:
            return SweepResult(config=cfg, ok=False,
                               error=f"warmup failed: {e}",
                               load_s=load_s)

        vram_before = _vram_used_mib()
        t_req = time.monotonic()
        try:
            resp = _post_completion(port, prompt, n_predict=n_predict, temperature=0.0)
        except Exception as e:
            return SweepResult(config=cfg, ok=False,
                               error=f"completion failed: {e}",
                               load_s=load_s)
        wall = time.monotonic() - t_req
        vram_peak = max(vram_before, _vram_used_mib())

        timings = resp.get("timings", {}) or {}
        tps = float(timings.get("predicted_per_second")
                    or (resp.get("tokens_predicted", 0) / wall if wall > 0 else 0.0))
        prefill_tps = float(timings.get("prompt_per_second", 0.0) or 0.0)
        n_pred = int(resp.get("tokens_predicted", 0) or timings.get("predicted_n", 0))
        n_prompt = int(resp.get("tokens_evaluated", 0) or timings.get("prompt_n", 0))
        first_token_s = float(timings.get("prompt_ms", 0.0)) / 1000.0

        return SweepResult(
            config=cfg, ok=True, tps=tps, prefill_tps=prefill_tps,
            load_s=load_s, first_token_s=first_token_s,
            tokens_predicted=n_pred, tokens_prompt=n_prompt,
            vram_peak_mib=vram_peak,
        )
    finally:
        # Graceful shutdown; SIGTERM then SIGKILL.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        if log_f is not None:
            log_f.close()
        # Wait for CUDA context teardown AND port to free up before next launch.
        _wait_vram_released(timeout_s=30.0)
        _wait_port_free(port, timeout_s=20.0)


def _read_tail(path: Path, n_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


@dataclass
class SweepReport:
    model_path: str
    results: list[SweepResult] = field(default_factory=list)

    @property
    def winner(self) -> Optional[SweepResult]:
        ok = [r for r in self.results if r.ok]
        if not ok:
            return None
        return max(ok, key=lambda r: r.tps)

    def to_jsonl(self, path: Path) -> None:
        with open(path, "w") as f:
            for r in self.results:
                d = {
                    "ok": r.ok, "error": r.error,
                    "tps": r.tps, "prefill_tps": r.prefill_tps,
                    "load_s": r.load_s, "first_token_s": r.first_token_s,
                    "vram_peak_mib": r.vram_peak_mib,
                    "n_gpu_layers": r.config.n_gpu_layers,
                    "ctx_size": r.config.ctx_size,
                    "batch_size": r.config.batch_size,
                    "ubatch_size": r.config.ubatch_size,
                    "kv_type": r.config.kv_type_k,
                    "est_vram_mib": r.config.est_vram_mib,
                    "notes": r.config.notes,
                }
                f.write(json.dumps(d) + "\n")


def run_sweep(
    server_bin: str,
    model_path: str,
    configs: list[ServerConfig],
    port: int = 8090,
    threads: Optional[int] = None,
    prompt: str = DEFAULT_PROMPT,
    n_predict: int = 96,
    log_dir: Optional[Path] = None,
) -> SweepReport:
    rep = SweepReport(model_path=model_path)
    for cfg in configs:
        r = run_one(
            server_bin=server_bin, model_path=model_path, cfg=cfg,
            port=port, prompt=prompt, n_predict=n_predict,
            threads=threads, log_dir=log_dir,
        )
        rep.results.append(r)
    return rep


def auto_sweep(
    model_path: str,
    server_bin: str,
    ctx_target: int = 32000,
    vram_override_mib: Optional[int] = None,
    vram_safety_mib: int = 200,
    max_configs: int = 6,
    port: int = 8090,
    threads: Optional[int] = None,
    n_predict: int = 96,
    prompt: str = DEFAULT_PROMPT,
    log_dir: Optional[Path] = None,
) -> SweepReport:
    info = read_gguf_info(model_path)
    if vram_override_mib is not None:
        free_vram = vram_override_mib
    else:
        gpus = query_nvidia_gpus()
        if not gpus:
            raise RuntimeError("no nvidia-smi GPU visible; pass vram_override_mib")
        free_vram = gpus[0].total_mib  # assume fresh run, use total
    cfgs = suggest_configs(
        info, free_vram_mib=free_vram,
        vram_safety_mib=vram_safety_mib, ctx_target=ctx_target,
    )[:max_configs]
    return run_sweep(
        server_bin=server_bin, model_path=model_path, configs=cfgs,
        port=port, threads=threads, n_predict=n_predict,
        prompt=prompt, log_dir=log_dir,
    )
