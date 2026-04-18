from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .backend import Backend, GenerationSettings, PromptCache, Token
from .budget import Budget
from .model import ModelSpec
from .technique import StepState, Technique
from .telemetry import JsonlRecorder, NullRecorder, TokRateMeter


@dataclass
class RuntimeContext:
    backend: Backend
    budget: Budget
    spec: Optional[ModelSpec]
    techniques: list[Technique] = field(default_factory=list)
    recorder: "JsonlRecorder | NullRecorder" = field(default_factory=NullRecorder)


class Session:
    def __init__(
        self,
        backend: Backend,
        budget: Budget,
        spec: Optional[ModelSpec] = None,
        techniques: Optional[list[Technique]] = None,
        recorder: Optional["JsonlRecorder | NullRecorder"] = None,
    ):
        self.ctx = RuntimeContext(
            backend=backend,
            budget=budget,
            spec=spec,
            techniques=techniques or [],
            recorder=recorder or NullRecorder(),
        )
        for t in self.ctx.techniques:
            t.attach(self.ctx)

    def new_cache(self) -> Optional[PromptCache]:
        """Create a reusable prompt cache. None if backend doesn't support it."""
        if not self.ctx.backend.supports_prompt_cache():
            return None
        return self.ctx.backend.make_prompt_cache()

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        cache: Optional[PromptCache] = None,
    ) -> Iterator[Token]:
        meter = TokRateMeter()
        for t in self.ctx.techniques:
            t.on_generation_start(self.ctx)
        self.ctx.recorder.record(
            "generation_start",
            prompt_len=len(prompt),
            cache_reused=cache is not None,
        )
        t_start = time.monotonic()
        first_token_seen = False
        try:
            for tok in self.ctx.backend.generate(prompt, settings, prompt_cache=cache):
                if not first_token_seen:
                    self.ctx.recorder.record("prefill_done", prefill_s=time.monotonic() - t_start)
                    first_token_seen = True
                meter.tick()
                state = StepState(token_idx=meter.count)
                # on_route is invoked by the backend via set_route_callback once
                # supports_routing_hooks() is True; Session drives only per-step hooks.
                for t in self.ctx.techniques:
                    t.after_step(self.ctx, state)
                if meter.count % 16 == 0:
                    self.ctx.recorder.record("tps", rolling=meter.rolling_tps(), n=meter.count)
                yield tok
        finally:
            self.ctx.recorder.record(
                "generation_end", tokens=meter.count,
                mean_tps=meter.mean_tps(), rolling_tps=meter.rolling_tps(),
            )
            for t in self.ctx.techniques:
                t.on_generation_end(self.ctx)

    def load_draft(self, draft_ref: str) -> None:
        """Load a draft model for speculative decoding on supporting backends."""
        self.ctx.backend.load_draft(draft_ref)

    def generate_speculative(
        self,
        prompt: str,
        settings: GenerationSettings,
        cache: Optional[PromptCache] = None,
        num_draft: int = 2,
    ) -> Iterator[Token]:
        """Speculative decoding via backend.generate_speculative.

        Emits `spec_round` telemetry per verify round (num_draft, num_accept,
        verify_ms, replay_ms) and aggregate counters at generation_end.
        """
        meter = TokRateMeter()
        rounds: list[dict] = []

        def rec_spec(evt: str, payload: dict) -> None:
            if evt == "spec_round":
                rounds.append(payload)
                self.ctx.recorder.record(evt, **payload)

        for t in self.ctx.techniques:
            t.on_generation_start(self.ctx)
        self.ctx.recorder.record(
            "generation_start",
            prompt_len=len(prompt),
            cache_reused=cache is not None,
            mode="speculative",
            num_draft=num_draft,
        )
        t_start = time.monotonic()
        first_token_seen = False
        accepted_from_draft = 0
        try:
            for tok in self.ctx.backend.generate_speculative(
                prompt, settings, num_draft, prompt_cache=cache, recorder=rec_spec,
            ):
                if tok.token_id == -1:
                    yield tok
                    continue
                if not first_token_seen:
                    self.ctx.recorder.record("prefill_done", prefill_s=time.monotonic() - t_start)
                    first_token_seen = True
                meter.tick()
                if tok.from_draft:
                    accepted_from_draft += 1
                state = StepState(token_idx=meter.count)
                for t in self.ctx.techniques:
                    t.after_step(self.ctx, state)
                if meter.count % 16 == 0:
                    self.ctx.recorder.record("tps", rolling=meter.rolling_tps(), n=meter.count)
                yield tok
        finally:
            total_draft = sum(r["num_draft"] for r in rounds)
            total_accept = sum(r["num_accept"] for r in rounds)
            self.ctx.recorder.record(
                "generation_end",
                tokens=meter.count,
                mean_tps=meter.mean_tps(),
                rolling_tps=meter.rolling_tps(),
                mode="speculative",
                rounds=len(rounds),
                from_draft_rate=(accepted_from_draft / meter.count) if meter.count else 0.0,
                per_round_accept=(total_accept / total_draft) if total_draft else 0.0,
                mean_accept_per_round=(total_accept / len(rounds)) if rounds else 0.0,
            )
            for t in self.ctx.techniques:
                t.on_generation_end(self.ctx)

    def close(self) -> None:
        for t in self.ctx.techniques:
            t.detach(self.ctx)
        self.ctx.backend.unload()
        self.ctx.recorder.close()
