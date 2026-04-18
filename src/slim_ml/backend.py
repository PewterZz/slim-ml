"""Backend interface + initial implementations.

The Backend ABC exposes two tiers of capability:
1. Base generation: load a model, stream tokens, stop. All backends support this.
2. Hooks for advanced techniques: routing observation, expert migration, layer access.
   Backends advertise support via `supports_*` methods so Technique.attach can fail
   cleanly on backends that can't host it.

Initial implementations:
- MLXBackend: generation works; hook points stubbed (require deeper MLX integration)
- LlamaCppBackend: generation works; hooks all NotImpl (requires ctypes/fork)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from .budget import Budget, Tier
from .model import ModelSpec


@dataclass
class GenerationSettings:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    seed: Optional[int] = None


@dataclass
class Token:
    text: str
    token_id: int
    logprob: Optional[float] = None


PromptCache = Any


class Backend(ABC):
    name: str

    @abstractmethod
    def load(self, model_ref: str, spec: Optional[ModelSpec], budget: Budget) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        prompt_cache: Optional[PromptCache] = None,
    ) -> Iterator[Token]: ...

    @abstractmethod
    def unload(self) -> None: ...

    def make_prompt_cache(self) -> Optional[PromptCache]:
        return None

    def supports_prompt_cache(self) -> bool:
        return False

    def supports_routing_hooks(self) -> bool:
        return False

    def supports_expert_migration(self) -> bool:
        return False

    def supports_layer_hooks(self) -> bool:
        return False

    def supports_kv_hooks(self) -> bool:
        return False

    def set_route_callback(self, cb: Callable[[int, list[int], list[float]], None]) -> None:
        raise NotImplementedError(f"{self.name}: routing hooks not implemented")

    def migrate_expert(self, layer_idx: int, expert_id: int, to_tier: Tier) -> None:
        raise NotImplementedError(f"{self.name}: expert migration not implemented")

    def expert_placement(self, layer_idx: int, expert_id: int) -> Tier:
        raise NotImplementedError(f"{self.name}: placement introspection not implemented")


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_ref: Optional[str] = None
        self._spec: Optional[ModelSpec] = None

    def load(self, model_ref: str, spec: Optional[ModelSpec], budget: Budget) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError("mlx-lm not installed. `pip install -e '.[mlx]'`") from e
        self._model, self._tokenizer = load(model_ref)
        self._model_ref = model_ref
        self._spec = spec

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        prompt_cache: Optional[PromptCache] = None,
    ) -> Iterator[Token]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=settings.temperature, top_p=settings.top_p)
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            max_tokens=settings.max_tokens,
            sampler=sampler,
        )
        if prompt_cache is not None:
            kwargs["prompt_cache"] = prompt_cache
        for resp in stream_generate(self._model, self._tokenizer, **kwargs):
            yield Token(text=resp.text, token_id=resp.token, logprob=None)

    def make_prompt_cache(self) -> Optional[PromptCache]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self._model)

    def supports_prompt_cache(self) -> bool:
        return True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None


class LlamaCppBackend(Backend):
    name = "llama_cpp"

    def __init__(self):
        self._llm = None
        self._spec: Optional[ModelSpec] = None

    def load(self, model_ref: str, spec: Optional[ModelSpec], budget: Budget) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python not installed. `pip install -e '.[llama]'`") from e
        vram_cap = budget.capacity(Tier.VRAM).available_bytes
        n_gpu_layers = -1 if vram_cap > 0 else 0
        self._llm = Llama(model_path=model_ref, n_gpu_layers=n_gpu_layers, n_ctx=4096, verbose=False)
        self._spec = spec

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        prompt_cache: Optional[PromptCache] = None,
    ) -> Iterator[Token]:
        if self._llm is None:
            raise RuntimeError("model not loaded")
        if prompt_cache is not None:
            raise NotImplementedError(
                "llama_cpp: prompt cache reuse not wired — use llama.cpp's own "
                "state save/load via ctypes or handle at a higher layer"
            )
        stream = self._llm(
            prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            stream=True,
        )
        for chunk in stream:
            choice = chunk["choices"][0]
            text = choice.get("text", "")
            if text:
                yield Token(text=text, token_id=-1)

    def unload(self) -> None:
        self._llm = None
