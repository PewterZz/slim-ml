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
    kv_bits: Optional[int] = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0


@dataclass
class Token:
    text: str
    token_id: int
    logprob: Optional[float] = None
    from_draft: Optional[bool] = None


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

    def supports_speculative(self) -> bool:
        return False

    def load_draft(self, draft_ref: str) -> None:
        raise NotImplementedError(f"{self.name}: draft model loading not implemented")

    def set_route_callback(
        self,
        cb: Optional[Callable[[int, list[int], Optional[list[float]]], None]],
    ) -> int:
        raise NotImplementedError(f"{self.name}: routing hooks not implemented")

    def generate_speculative(
        self,
        prompt: str,
        settings: "GenerationSettings",
        num_draft: int,
        prompt_cache: Optional["PromptCache"] = None,
        recorder: Optional[Callable[[str, dict], None]] = None,
    ) -> Iterator["Token"]:
        raise NotImplementedError(f"{self.name}: speculative decoding not implemented")

    def supports_pld(self) -> bool:
        return False

    def generate_pld_speculative(
        self,
        prompt: str,
        settings: "GenerationSettings",
        num_draft: int,
        prompt_cache: Optional["PromptCache"] = None,
        recorder: Optional[Callable[[str, dict], None]] = None,
        max_ngram_size: int = 3,
        min_ngram_size: int = 2,
    ) -> Iterator["Token"]:
        raise NotImplementedError(f"{self.name}: PLD decoding not implemented")

    def migrate_expert(self, layer_idx: int, expert_id: int, to_tier: Tier) -> None:
        raise NotImplementedError(f"{self.name}: expert migration not implemented")

    def expert_placement(self, layer_idx: int, expert_id: int) -> Tier:
        raise NotImplementedError(f"{self.name}: placement introspection not implemented")


# Route observation scaffold (Stage 0 of expert caching).
# We patch MLX's SwitchGLU/SwitchMLP __call__ at class level on first use and
# route observations only for instances that have registered a callback via
# _ROUTE_STATE[id(instance)]. Classes with no observed instances see two
# getattr-equivalent dict misses per forward — negligible on the MoE critical
# path. `indices.reshape(-1).tolist()` forces a device sync, so this is
# deliberately a profiling hook, not something to leave wired in hot-path.
_ROUTE_ORIG_CALLS: dict[type, Any] = {}
_ROUTE_STATE: dict[int, tuple[int, Callable[..., None]]] = {}
_ROUTE_CB_WARNED: set[int] = set()


def _patch_switch_class_once(cls: type) -> None:
    if cls in _ROUTE_ORIG_CALLS:
        return
    orig = cls.__call__
    _ROUTE_ORIG_CALLS[cls] = orig

    def patched(self, x, indices):
        st = _ROUTE_STATE.get(id(self))
        if st is not None:
            layer_idx, cb = st
            try:
                flat = indices.reshape(-1).tolist()
                cb(layer_idx, flat, None)
            except Exception:
                if id(cb) not in _ROUTE_CB_WARNED:
                    _ROUTE_CB_WARNED.add(id(cb))
                    import traceback
                    traceback.print_exc()
        return _ROUTE_ORIG_CALLS[type(self)](self, x, indices)

    cls.__call__ = patched  # type: ignore[method-assign]


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_ref: Optional[str] = None
        self._spec: Optional[ModelSpec] = None
        self._draft_model = None
        self._draft_ref: Optional[str] = None
        self._route_instance_ids: list[int] = []

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
        if settings.kv_bits is not None:
            kwargs["kv_bits"] = settings.kv_bits
            kwargs["kv_group_size"] = settings.kv_group_size
            kwargs["quantized_kv_start"] = settings.quantized_kv_start
        for resp in stream_generate(self._model, self._tokenizer, **kwargs):
            yield Token(text=resp.text, token_id=resp.token, logprob=None)

    def make_prompt_cache(self) -> Optional[PromptCache]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self._model)

    def supports_prompt_cache(self) -> bool:
        return True

    def supports_speculative(self) -> bool:
        return self._model is not None and self._draft_model is not None

    def supports_pld(self) -> bool:
        return self._model is not None

    def _iter_moe_switch_layers(self):
        if self._model is None:
            return
        inner = getattr(self._model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if not layers:
            return
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            switch = getattr(mlp, "switch_mlp", None) if mlp is not None else None
            if switch is not None:
                yield idx, switch

    def supports_routing_hooks(self) -> bool:
        return any(True for _ in self._iter_moe_switch_layers())

    def set_route_callback(
        self,
        cb: Optional[Callable[[int, list[int], Optional[list[float]]], None]],
    ) -> int:
        if self._model is None:
            raise RuntimeError("model not loaded")
        for iid in self._route_instance_ids:
            _ROUTE_STATE.pop(iid, None)
        self._route_instance_ids = []
        n = 0
        for layer_idx, switch in self._iter_moe_switch_layers():
            if cb is None:
                continue
            _patch_switch_class_once(type(switch))
            _ROUTE_STATE[id(switch)] = (layer_idx, cb)
            self._route_instance_ids.append(id(switch))
            n += 1
        return n

    def load_draft(self, draft_ref: str) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError("mlx-lm not installed. `pip install -e '.[mlx]'`") from e
        self._draft_model, _ = load(draft_ref)
        self._draft_ref = draft_ref

    def generate_speculative(
        self,
        prompt: str,
        settings: GenerationSettings,
        num_draft: int,
        prompt_cache: Optional[PromptCache] = None,
        recorder: Optional[Callable[[str, dict], None]] = None,
    ) -> Iterator[Token]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        if self._draft_model is None:
            raise RuntimeError("draft model not loaded — call load_draft() first")
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        from .spec_decode import speculative_step

        sampler = make_sampler(temp=settings.temperature, top_p=settings.top_p)
        prompt_ids = mx.array(self._tokenizer.encode(prompt), mx.uint32)
        detok = self._tokenizer.detokenizer
        detok.reset()
        eos_ids = self._tokenizer.eos_token_ids

        gen = speculative_step(
            prompt_ids,
            self._model,
            self._draft_model,
            num_draft_tokens=num_draft,
            max_tokens=settings.max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            recorder=recorder,
        )
        for token_id, _lp, from_draft in gen:
            tid = int(token_id)
            if tid in eos_ids:
                break
            detok.add_token(tid)
            yield Token(
                text=detok.last_segment,
                token_id=tid,
                logprob=None,
                from_draft=bool(from_draft),
            )
        detok.finalize()
        tail = detok.last_segment
        if tail:
            yield Token(text=tail, token_id=-1, logprob=None, from_draft=None)

    def generate_pld_speculative(
        self,
        prompt: str,
        settings: GenerationSettings,
        num_draft: int,
        prompt_cache: Optional[PromptCache] = None,
        recorder: Optional[Callable[[str, dict], None]] = None,
        max_ngram_size: int = 3,
        min_ngram_size: int = 2,
    ) -> Iterator[Token]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        from .spec_decode import speculative_step_pld

        sampler = make_sampler(temp=settings.temperature, top_p=settings.top_p)
        prompt_ids = mx.array(self._tokenizer.encode(prompt), mx.uint32)
        detok = self._tokenizer.detokenizer
        detok.reset()
        eos_ids = self._tokenizer.eos_token_ids

        gen = speculative_step_pld(
            prompt_ids,
            self._model,
            num_draft_tokens=num_draft,
            max_tokens=settings.max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            max_ngram_size=max_ngram_size,
            min_ngram_size=min_ngram_size,
            recorder=recorder,
        )
        for token_id, _lp, from_draft in gen:
            tid = int(token_id)
            if tid in eos_ids:
                break
            detok.add_token(tid)
            yield Token(
                text=detok.last_segment,
                token_id=tid,
                logprob=None,
                from_draft=bool(from_draft),
            )
        detok.finalize()
        tail = detok.last_segment
        if tail:
            yield Token(text=tail, token_id=-1, logprob=None, from_draft=None)

    def unload(self) -> None:
        for iid in self._route_instance_ids:
            _ROUTE_STATE.pop(iid, None)
        self._route_instance_ids = []
        self._model = None
        self._tokenizer = None
        self._draft_model = None


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
