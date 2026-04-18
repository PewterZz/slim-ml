"""Pluggable system techniques.

The Technique ABC defines lifecycle hooks; ExpertCache is fully sketched because its
requirements shape the Backend interface. Other techniques are stubbed but their
interfaces must match what they'd need from the runtime.

A Backend that can host ExpertCache can host the rest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .budget import Tier

if TYPE_CHECKING:
    from .runtime import RuntimeContext


@dataclass
class StepState:
    layer_idx: Optional[int] = None
    token_idx: Optional[int] = None
    routed_expert_ids: Optional[list[int]] = None
    routed_expert_weights: Optional[list[float]] = None


class Technique(ABC):
    name: str

    @abstractmethod
    def attach(self, ctx: "RuntimeContext") -> None: ...

    def detach(self, ctx: "RuntimeContext") -> None:
        return None

    def on_generation_start(self, ctx: "RuntimeContext") -> None:
        return None

    def on_generation_end(self, ctx: "RuntimeContext") -> None:
        return None

    def on_route(self, ctx: "RuntimeContext", state: StepState) -> None:
        return None

    def before_step(self, ctx: "RuntimeContext", state: StepState) -> None:
        return None

    def after_step(self, ctx: "RuntimeContext", state: StepState) -> None:
        return None


@dataclass
class ExpertStats:
    hits: int = 0
    last_seen_token: int = -1
    ema_weight: float = 0.0


class ExpertCache(Technique):
    """Keeps frequently-routed experts in a fast tier (typically VRAM) and cold experts
    in slower tiers (RAM / NVMe).

    This class is the scaffold's load-bearing technique: the Backend must be able to
    surface routing decisions and execute tier migrations for this to work. The
    interface here pins what we need from any Backend that wants to host expert caching.

    Sizing math (Qwen3-Next-80B-A3B as example):
        80B total / 512 experts / 48 layers = ~3.3M params per expert slot
        @ Q4 → ~1.6 MB per expert → 6GB VRAM fits ~3800 experts (16% of total)
        If routing is Zipfian, a 16% hot set can capture 40-60% of routes.
    """

    name = "expert_cache"

    def __init__(
        self,
        fast_tier: Tier = Tier.VRAM,
        slow_tier: Tier = Tier.RAM,
        ema_alpha: float = 0.05,
        reselect_every_n_tokens: int = 256,
    ):
        self.fast_tier = fast_tier
        self.slow_tier = slow_tier
        self.ema_alpha = ema_alpha
        self.reselect_every_n_tokens = reselect_every_n_tokens
        self._stats: dict[tuple[int, int], ExpertStats] = defaultdict(ExpertStats)
        self._hot_set: set[tuple[int, int]] = set()
        self._tokens_since_reselect: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def attach(self, ctx: "RuntimeContext") -> None:
        if not ctx.backend.supports_routing_hooks():
            raise NotImplementedError(
                f"{ctx.backend.__class__.__name__} does not expose routing hooks; "
                "expert caching cannot be attached to this backend yet."
            )
        if not ctx.backend.supports_expert_migration():
            raise NotImplementedError(
                f"{ctx.backend.__class__.__name__} does not support runtime expert "
                "migration between memory tiers."
            )
        self._compute_initial_hot_set(ctx)
        raise NotImplementedError("ExpertCache.attach: initial migration not wired yet")

    def on_route(self, ctx: "RuntimeContext", state: StepState) -> None:
        if state.layer_idx is None or state.routed_expert_ids is None:
            return
        weights = state.routed_expert_weights or [1.0] * len(state.routed_expert_ids)
        for eid, w in zip(state.routed_expert_ids, weights):
            key = (state.layer_idx, eid)
            s = self._stats[key]
            s.hits += 1
            s.last_seen_token = state.token_idx or 0
            s.ema_weight = (1 - self.ema_alpha) * s.ema_weight + self.ema_alpha * w
            if key in self._hot_set:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def after_step(self, ctx: "RuntimeContext", state: StepState) -> None:
        self._tokens_since_reselect += 1
        if self._tokens_since_reselect >= self.reselect_every_n_tokens:
            self._tokens_since_reselect = 0
            self._reselect_hot_set(ctx)

    def on_generation_end(self, ctx: "RuntimeContext") -> None:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total else 0.0
        ctx.recorder.record("expert_cache_summary",
                            hits=self._cache_hits, misses=self._cache_misses,
                            hit_rate=hit_rate, hot_set_size=len(self._hot_set))

    def _compute_initial_hot_set(self, ctx: "RuntimeContext") -> None:
        raise NotImplementedError("Initial hot-set selection: cold-start policy TBD")

    def _reselect_hot_set(self, ctx: "RuntimeContext") -> None:
        raise NotImplementedError("Hot-set reselection + migration loop")


class SpecDecode(Technique):
    """Speculative decoding: small draft model proposes, big model verifies in parallel.

    Requires backend to support: batched forward with accepted-prefix reuse, and
    ability to run two models sharing tokenizer. See docs/plans/spec_decode.md (TODO).
    """
    name = "spec_decode"

    def __init__(self, draft_model: str, gamma: int = 4):
        self.draft_model = draft_model
        self.gamma = gamma

    def attach(self, ctx: "RuntimeContext") -> None:
        raise NotImplementedError("SpecDecode: v1 target, not yet implemented")


class KVQuant(Technique):
    """Runtime KV cache quantization + optional eviction (H2O / attention-sinks)."""
    name = "kv_quant"

    def __init__(self, bits: int = 4, evict: str | None = None):
        self.bits = bits
        self.evict = evict

    def attach(self, ctx: "RuntimeContext") -> None:
        raise NotImplementedError("KVQuant: v2 target, not yet implemented")


class LayerStream(Technique):
    """Stream model layers from NVMe, overlapping I/O with compute of current layer."""
    name = "layer_stream"

    def __init__(self, prefetch_depth: int = 2):
        self.prefetch_depth = prefetch_depth

    def attach(self, ctx: "RuntimeContext") -> None:
        raise NotImplementedError("LayerStream: v2 target, not yet implemented")
