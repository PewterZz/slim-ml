from __future__ import annotations

import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .backend import GenerationSettings, LlamaCppBackend, MLXBackend
from .budget import StaticBudget, Tier, auto_detect_limits
from .model import ARCH_REGISTRY
from .runtime import Session
from .telemetry import JsonlRecorder, NullRecorder

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def probe(headroom_gb: float = typer.Option(4.0, help="RAM bytes to leave for your other work")):
    """Show detected hardware tiers and what the budget would be."""
    headroom_bytes = int(headroom_gb * 1024**3)
    limits = auto_detect_limits(headroom_bytes=headroom_bytes)
    table = Table(title="Hardware tiers")
    table.add_column("Tier"); table.add_column("Available", justify="right")
    for tier, nbytes in limits.items():
        table.add_row(tier.name, f"{nbytes / 1024**3:.1f} GiB")
    console.print(table)
    console.print(f"[dim]headroom reserved: {headroom_gb} GiB RAM[/dim]")


@app.command()
def run(
    model: str = typer.Argument(..., help="HF repo id, local path, or GGUF file"),
    backend: str = typer.Option("auto", help="mlx | llama_cpp | auto"),
    arch: Optional[str] = typer.Option(None, help=f"Known arch: {sorted(ARCH_REGISTRY)}"),
    prompt: str = typer.Option("Hello, how are you?"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.7),
    kv_bits: Optional[int] = typer.Option(
        None, help="Quantize KV cache to N bits (MLX: 4 or 8). None = off."
    ),
    kv_start: int = typer.Option(0, help="Step to begin quantizing KV (keep first N FP16)."),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Run a model end-to-end through the slim-ml runtime."""
    spec = ARCH_REGISTRY.get(arch) if arch else None

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "llama_cpp"
    be = MLXBackend() if backend == "mlx" else LlamaCppBackend()

    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model} via {backend}")
    be.load(model, spec, budget)

    session = Session(backend=be, budget=budget, spec=spec, recorder=recorder)
    settings = GenerationSettings(
        max_tokens=max_tokens,
        temperature=temperature,
        kv_bits=kv_bits,
        quantized_kv_start=kv_start,
    )

    try:
        for tok in session.generate(prompt, settings):
            console.print(tok.text, end="")
        console.print()
    finally:
        session.close()


@app.command()
def bench(
    model: str = typer.Argument(..., help="HF repo id, local path, or GGUF file"),
    backend: str = typer.Option("auto", help="mlx | llama_cpp | auto"),
    system_prompt: str = typer.Option(
        "You are an expert Python programmer. Respond only with idiomatic, well-typed code. "
        * 20,
        help="Long shared prefix. Dominates prefill time.",
    ),
    user_a: str = typer.Option("Write a bubble sort.", help="First user turn"),
    user_b: str = typer.Option("Write a merge sort.", help="Second user turn"),
    max_tokens: int = typer.Option(40),
    temperature: float = typer.Option(0.7),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """
    Benchmark prompt-cache reuse.

    Runs the SAME conversation twice:
    - Cold: fresh cache, full prompt prefilled every turn.
    - Warm: shared cache, turn 2 submits only the new user tokens.

    Reports TTFT delta — how much prefill you save on turn 2 by keeping state.
    """
    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "llama_cpp"
    be = MLXBackend() if backend == "mlx" else LlamaCppBackend()

    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model} via {backend}")
    be.load(model, None, budget)
    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    def turn(label: str, prompt: str, cache):
        t0 = time.monotonic()
        ttft = None
        out_parts = []
        count = 0
        for tok in session.generate(prompt, settings, cache=cache):
            if ttft is None:
                ttft = time.monotonic() - t0
            out_parts.append(tok.text)
            count += 1
        elapsed = time.monotonic() - t0
        decode_tps = (count - 1) / (elapsed - ttft) if count > 1 and ttft else 0.0
        out = "".join(out_parts)
        console.print(f"[bold cyan]{label}[/bold cyan] ttft={ttft*1000:.0f}ms  total={elapsed:.2f}s  decode={decode_tps:.1f} t/s  tokens={count}")
        console.print(f"  [dim]{out[:120]}...[/dim]" if len(out) > 120 else f"  [dim]{out}[/dim]")
        return ttft or 0.0, out

    try:
        # --- Cold path: no cache reuse, each turn sends full prompt ---
        console.print("\n[bold]Cold (no cache reuse)[/bold]")
        ttft_c1, out_c1 = turn("t1 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant:", None)
        ttft_c2, _ = turn("t2 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant: " + out_c1 + "\nUser: " + user_b + "\nAssistant:", None)

        # --- Warm path: shared cache, t2 submits only the new tokens ---
        console.print("\n[bold]Warm (shared cache)[/bold]")
        cache = session.new_cache()
        if cache is None:
            console.print("[yellow]backend does not support prompt cache — skipping warm path[/yellow]")
            return
        ttft_w1, _ = turn("t1 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant:", cache)
        # Generated tokens from t1 are already in the cache. t2 submits only new user turn.
        ttft_w2, _ = turn("t2 (delta only)   ", "\nUser: " + user_b + "\nAssistant:", cache)

        console.print()
        console.print(f"[green]T2 TTFT  cold={ttft_c2*1000:.0f}ms  warm={ttft_w2*1000:.0f}ms  savings={(ttft_c2-ttft_w2)*1000:.0f}ms  speedup={(ttft_c2/ttft_w2 if ttft_w2 else float('inf')):.1f}×[/green]")
    finally:
        session.close()


@app.command()
def spec(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    draft: str = typer.Option(..., help="Draft model for speculation"),
    prompt: str = typer.Option("Write a Python bubble sort with type hints."),
    num_draft: int = typer.Option(2, help="Draft tokens per verify round"),
    max_tokens: int = typer.Option(96),
    temperature: float = typer.Option(0.0, help="0.0 for greedy (correctness-comparable)"),
    compare_baseline: bool = typer.Option(
        True, help="Also run non-speculative baseline and report speedup"
    ),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Speculative decoding with per-round telemetry.

    MLX-only. Handles hybrid-attention models (Qwen3.5 family) via slim-ml's
    own snapshot/restore of ArraysCache + KVCache state.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] target={model}  draft={draft}")
    be.load(model, None, budget)
    be.load_draft(draft)

    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        if compare_baseline:
            t0 = time.monotonic()
            base_tokens: list[int] = []
            base_text = []
            for tok in session.generate(prompt, settings):
                base_tokens.append(tok.token_id)
                base_text.append(tok.text)
            t_base = time.monotonic() - t0
            n_base = len(base_tokens)
            console.print(
                f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s "
                f"tps={n_base / t_base:.1f}"
            )

        t0 = time.monotonic()
        spec_tokens: list[int] = []
        spec_text = []
        accepted = 0
        n = 0
        for tok in session.generate_speculative(
            prompt, settings, num_draft=num_draft
        ):
            spec_tokens.append(tok.token_id)
            spec_text.append(tok.text)
            if tok.from_draft:
                accepted += 1
            if tok.token_id != -1:
                n += 1
        t_spec = time.monotonic() - t0
        tps = n / t_spec if t_spec > 0 else 0.0
        accept_rate = (accepted / n) * 100 if n else 0.0
        console.print(
            f"[bold]spec[/bold]     tokens={n} time={t_spec:.2f}s "
            f"tps={tps:.1f}  from_draft={accept_rate:.1f}%"
        )

        if compare_baseline:
            speedup = t_base / t_spec if t_spec > 0 else 0.0
            console.print(f"[green]speedup {speedup:.2f}×[/green]")
            if temperature == 0.0:
                m = min(len(base_tokens), len(spec_tokens))
                diverge = next((i for i in range(m) if base_tokens[i] != spec_tokens[i]), None)
                if diverge is None and len(base_tokens) == len(spec_tokens):
                    console.print(f"[green]correctness: {m} tokens identical[/green]")
                elif diverge is None:
                    console.print(
                        f"[yellow]correctness: prefix {m} match, lengths differ "
                        f"base={len(base_tokens)} spec={len(spec_tokens)}[/yellow]"
                    )
                else:
                    console.print(
                        f"[red]correctness FAIL at idx {diverge}: "
                        f"base={base_tokens[diverge]} spec={spec_tokens[diverge]}[/red]"
                    )

        console.print(f"\n[dim]{''.join(spec_text)[:200]}[/dim]")
    finally:
        session.close()


@app.command("spec-sweep")
def spec_sweep(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    draft: str = typer.Option(..., help="Draft model for speculation"),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints."
    ),
    num_drafts: str = typer.Option("1,2,4,6,8", help="Comma-separated num_draft values to sweep"),
    max_tokens: int = typer.Option(96),
    temperature: float = typer.Option(0.0),
    headroom_gb: float = typer.Option(4.0),
):
    """Sweep num_draft values, report which maximizes speedup for this model pair.

    Loads models once, reuses across runs. Prints a table keyed by num_draft.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)

    console.print(f"[cyan]loading[/cyan] target={model}  draft={draft}")
    be.load(model, None, budget)
    be.load_draft(draft)
    session = Session(backend=be, budget=budget, recorder=NullRecorder())
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        t0 = time.monotonic()
        n_base = 0
        for _ in session.generate(prompt, settings):
            n_base += 1
        t_base = time.monotonic() - t0
        tps_base = n_base / t_base if t_base > 0 else 0.0
        console.print(f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s tps={tps_base:.1f}")

        table = Table(title="num_draft sweep")
        table.add_column("n_draft", justify="right")
        table.add_column("tps", justify="right")
        table.add_column("speedup", justify="right")
        table.add_column("from_draft", justify="right")

        values = [int(x) for x in num_drafts.split(",")]
        for n in values:
            t0 = time.monotonic()
            n_tok = 0
            accepted = 0
            for tok in session.generate_speculative(prompt, settings, num_draft=n):
                if tok.token_id == -1:
                    continue
                n_tok += 1
                if tok.from_draft:
                    accepted += 1
            dt = time.monotonic() - t0
            tps = n_tok / dt if dt > 0 else 0.0
            speedup = tps / tps_base if tps_base > 0 else 0.0
            rate = (accepted / n_tok * 100) if n_tok else 0.0
            table.add_row(str(n), f"{tps:.1f}", f"{speedup:.2f}×", f"{rate:.1f}%")
        console.print(table)
    finally:
        session.close()


@app.command()
def pld(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints."
    ),
    num_draft: int = typer.Option(4, help="Max draft tokens per verify round (variable)"),
    max_ngram: int = typer.Option(3, help="Max n-gram size to match against history"),
    min_ngram: int = typer.Option(2, help="Min n-gram size before giving up"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.0, help="0.0 enables correctness gate vs baseline"),
    compare_baseline: bool = typer.Option(
        True, help="Also run non-speculative baseline and report speedup"
    ),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Prompt Lookup Decoding — draftless speculation via n-gram match.

    MLX-only. No draft model. Drafts come from searching the prompt + what's
    been generated for the last n-gram's earlier occurrence and taking what
    followed. Fast path is code / refactor workloads with context-local repeats.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model}")
    be.load(model, None, budget)

    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        if compare_baseline:
            t0 = time.monotonic()
            base_tokens: list[int] = []
            for tok in session.generate(prompt, settings):
                base_tokens.append(tok.token_id)
            t_base = time.monotonic() - t0
            n_base = len(base_tokens)
            console.print(
                f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s "
                f"tps={n_base / t_base:.1f}"
            )

        t0 = time.monotonic()
        pld_tokens: list[int] = []
        pld_text: list[str] = []
        accepted = 0
        n = 0
        for tok in session.generate_pld_speculative(
            prompt,
            settings,
            num_draft=num_draft,
            max_ngram_size=max_ngram,
            min_ngram_size=min_ngram,
        ):
            pld_tokens.append(tok.token_id)
            pld_text.append(tok.text)
            if tok.from_draft:
                accepted += 1
            if tok.token_id != -1:
                n += 1
        t_pld = time.monotonic() - t0
        tps = n / t_pld if t_pld > 0 else 0.0
        accept_rate = (accepted / n) * 100 if n else 0.0
        console.print(
            f"[bold]pld[/bold]      tokens={n} time={t_pld:.2f}s "
            f"tps={tps:.1f}  from_draft={accept_rate:.1f}%"
        )

        if compare_baseline:
            speedup = t_base / t_pld if t_pld > 0 else 0.0
            console.print(f"[green]speedup {speedup:.2f}×[/green]")
            if temperature == 0.0:
                m = min(len(base_tokens), len(pld_tokens))
                diverge = next((i for i in range(m) if base_tokens[i] != pld_tokens[i]), None)
                if diverge is None and len(base_tokens) == len(pld_tokens):
                    console.print(f"[green]correctness: {m} tokens identical[/green]")
                elif diverge is None:
                    console.print(
                        f"[yellow]correctness: prefix {m} match, lengths differ "
                        f"base={len(base_tokens)} pld={len(pld_tokens)}[/yellow]"
                    )
                else:
                    console.print(
                        f"[red]correctness FAIL at idx {diverge}: "
                        f"base={base_tokens[diverge]} pld={pld_tokens[diverge]}[/red]"
                    )

        console.print(f"\n[dim]{''.join(pld_text)[:200]}[/dim]")
    finally:
        session.close()


@app.command()
def techniques():
    """List registered techniques and their implementation status."""
    table = Table(title="Techniques")
    table.add_column("Name"); table.add_column("Target"); table.add_column("Status")
    table.add_row("expert_cache", "v0", "interface sketched")
    table.add_row("spec_decode", "v1", "shipped (MLX, hybrid-attention supported)")
    table.add_row("kv_quant", "v2", "runtime knob shipped (--kv-bits); Technique interface TBD")
    table.add_row("layer_stream", "v2", "stub")
    console.print(table)


if __name__ == "__main__":
    app()
