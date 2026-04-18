from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .backend import Backend, GenerationSettings, Token
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

    def generate(self, prompt: str, settings: GenerationSettings) -> Iterator[Token]:
        meter = TokRateMeter()
        for t in self.ctx.techniques:
            t.on_generation_start(self.ctx)
        self.ctx.recorder.record("generation_start", prompt_len=len(prompt))
        try:
            for tok in self.ctx.backend.generate(prompt, settings):
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

    def close(self) -> None:
        for t in self.ctx.techniques:
            t.detach(self.ctx)
        self.ctx.backend.unload()
        self.ctx.recorder.close()
