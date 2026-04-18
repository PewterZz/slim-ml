from __future__ import annotations

import sys
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
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        for tok in session.generate(prompt, settings):
            console.print(tok.text, end="")
        console.print()
    finally:
        session.close()


@app.command()
def techniques():
    """List registered techniques and their implementation status."""
    table = Table(title="Techniques")
    table.add_column("Name"); table.add_column("Target"); table.add_column("Status")
    table.add_row("expert_cache", "v0", "interface sketched")
    table.add_row("spec_decode", "v1", "stub")
    table.add_row("kv_quant", "v2", "stub")
    table.add_row("layer_stream", "v2", "stub")
    console.print(table)


if __name__ == "__main__":
    app()
