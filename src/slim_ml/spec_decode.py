"""Speculative decoding for MLX models, including hybrid-attention (Qwen3.5).

The core algorithm is a direct port of mlx_lm.generate.speculative_generate_step,
extended with ArraysCache snapshot/restore so hybrid models (linear + full
attention) can speculate correctly. The upstream fix lives in mlx-lm PR #1111;
slim-ml carries its own copy here so users don't have to pin a fork.

slim-ml value-add over upstream: per-verify-round telemetry via the recorder.
Each round emits `spec_round` with `{round, num_draft, num_accept, verify_ms,
replay_ms}`, and generation_end includes aggregate acceptance + mean accept/round.

`speculative_step_pld` is the same verifier with a prompt-lookup draft: drafts
come from an n-gram match against what we've already produced, not a draft
model. Cheap win on workloads with context-local repetition (code, refactors).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from .prompt_lookup import prompt_lookup_draft

_STREAM = mx.new_stream(mx.default_device())


def speculative_step(
    prompt: mx.array,
    model,
    draft_model,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 512,
    recorder: Optional[Callable[[str, dict], None]] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Port of mlx_lm speculative_generate_step + PR #1111 hybrid snapshot/restore.

    Yields (token_id, logprobs, from_draft) per token. The caller is responsible
    for detokenization and EOS handling.
    """
    y = prompt.astype(mx.uint32)

    if prompt_cache is None:
        model_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    has_arrays = any(isinstance(c, ArraysCache) for c in model_cache)
    if not has_arrays and not can_trim_prompt_cache(model_cache):
        bad = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Non-hybrid prompt cache is not trimmable: {bad}. "
            "slim-ml spec decode handles ArraysCache but not arbitrary non-trimmable caches."
        )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs), logprobs

    def _step(m, c, tokens, n_predict=1):
        with mx.stream(_STREAM):
            logits = m(tokens[None], cache=c)[:, -n_predict:, :]
            if n_predict == 1:
                return _sample(logits.squeeze(0))
            outs, lps = [], []
            for i in range(n_predict):
                tok, lp = _sample(logits[:, i, :])
                outs.append(tok)
                lps.append(lp)
            return mx.concatenate(outs, axis=0), mx.concatenate(lps, axis=0)

    def _prefill(m, c, tokens):
        while tokens.size > 1:
            n = min(prefill_step_size, tokens.size - 1)
            m(tokens[:n][None], cache=c)
            mx.eval([ch.state for ch in c])
            tokens = tokens[n:]
            mx.clear_cache()
        return tokens

    def _draft_generate(tok, n_draft):
        if n_draft == 0:
            return mx.array([], mx.uint32)
        out = []
        for _ in range(n_draft):
            tok, _ = _step(draft_model, draft_cache, tok)
            mx.async_eval(tok)
            out.append(tok)
        return mx.concatenate(out)

    hybrid_snap: dict = {"arrays": {}, "kv_offsets": {}, "verify_input": None}

    def _snapshot(verify_input: mx.array) -> None:
        hybrid_snap["verify_input"] = verify_input
        hybrid_snap["arrays"] = {
            i: list(c.cache) for i, c in enumerate(model_cache) if isinstance(c, ArraysCache)
        }
        hybrid_snap["kv_offsets"] = {
            i: c.offset for i, c in enumerate(model_cache) if isinstance(c, KVCache)
        }

    def _rewind(n_draft: int, n_accept: int):
        # Trim KVCache like the upstream non-hybrid path.
        trim_prompt_cache(model_cache, n_draft - n_accept)
        trim_prompt_cache(draft_cache, max(n_draft - n_accept - 1, 0))
        if not (has_arrays and n_draft > n_accept):
            return 0.0
        t0 = time.monotonic()
        saved_arrays = hybrid_snap["arrays"]
        saved_kv_offsets = hybrid_snap["kv_offsets"]
        verify_input = hybrid_snap["verify_input"]
        if not saved_arrays or verify_input is None:
            return 0.0
        for i, c in enumerate(model_cache):
            if isinstance(c, KVCache) and i in saved_kv_offsets:
                c.offset = saved_kv_offsets[i]
            if isinstance(c, ArraysCache) and i in saved_arrays:
                c.cache = list(saved_arrays[i])
        rejected = n_draft - n_accept
        accepted = verify_input[:-rejected]
        with mx.stream(_STREAM):
            model(accepted[None], cache=model_cache)
            mx.eval([c.state for c in model_cache if isinstance(c, ArraysCache)])
        return (time.monotonic() - t0) * 1000.0

    with mx.stream(_STREAM):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    ntoks = 0
    num_draft = 0
    n_accept = 0
    round_idx = 0
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            draft_tokens = _draft_generate(draft_y, num_draft)
            y = mx.concatenate([y, draft_tokens])
            if has_arrays:
                _snapshot(y)

            t_verify = time.monotonic()
            tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
            mx.eval(tokens, draft_tokens)
            verify_ms = (time.monotonic() - t_verify) * 1000.0

            draft_list = draft_tokens.tolist()
            tokens_list = tokens.tolist()

            n_accept = 0
            while n_accept < num_draft:
                if tokens_list[n_accept] != draft_list[n_accept]:
                    break
                tn, lpn = tokens_list[n_accept], logprobs[n_accept]
                n_accept += 1
                ntoks += 1
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break

            if ntoks < max_tokens:
                ntoks += 1
                yield tokens_list[n_accept], logprobs[n_accept], False

            replay_ms = _rewind(num_draft, n_accept)
            if recorder is not None:
                recorder(
                    "spec_round",
                    {
                        "round": round_idx,
                        "num_draft": num_draft,
                        "num_accept": n_accept,
                        "verify_ms": verify_ms,
                        "replay_ms": replay_ms,
                    },
                )
            round_idx += 1

            if ntoks >= max_tokens:
                break

            y = mx.array([tokens_list[n_accept]], mx.uint32)
            draft_y = y
            if n_accept == num_draft:
                draft_y = mx.concatenate(
                    [mx.array(draft_list[-1:], mx.uint32), draft_y]
                )
    finally:
        _rewind(num_draft, n_accept)


def speculative_step_pld(
    prompt: mx.array,
    model,
    num_draft_tokens: int = 4,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 512,
    max_ngram_size: int = 3,
    min_ngram_size: int = 2,
    recorder: Optional[Callable[[str, dict], None]] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """PLD variant: drafts come from prompt n-gram lookup, not a draft model.

    Same verifier path and snapshot/restore as `speculative_step`, but no
    draft model and no draft cache. Per-round draft length varies with how
    long a match we find — rounds with no match degrade to a plain 1-token
    step. Emits `pld_round` events on the recorder.
    """
    y = prompt.astype(mx.uint32)
    history: list[int] = y.tolist()

    if prompt_cache is None:
        model_cache = make_prompt_cache(model)
    else:
        model_cache = prompt_cache

    has_arrays = any(isinstance(c, ArraysCache) for c in model_cache)
    if not has_arrays and not can_trim_prompt_cache(model_cache):
        bad = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Non-hybrid prompt cache is not trimmable: {bad}. "
            "slim-ml PLD handles ArraysCache but not arbitrary non-trimmable caches."
        )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs), logprobs

    def _step(m, c, tokens, n_predict=1):
        with mx.stream(_STREAM):
            logits = m(tokens[None], cache=c)[:, -n_predict:, :]
            if n_predict == 1:
                return _sample(logits.squeeze(0))
            outs, lps = [], []
            for i in range(n_predict):
                tok, lp = _sample(logits[:, i, :])
                outs.append(tok)
                lps.append(lp)
            return mx.concatenate(outs, axis=0), mx.concatenate(lps, axis=0)

    def _prefill(m, c, tokens):
        while tokens.size > 1:
            n = min(prefill_step_size, tokens.size - 1)
            m(tokens[:n][None], cache=c)
            mx.eval([ch.state for ch in c])
            tokens = tokens[n:]
            mx.clear_cache()
        return tokens

    hybrid_snap: dict = {"arrays": {}, "kv_offsets": {}, "verify_input": None}

    def _snapshot(verify_input: mx.array) -> None:
        hybrid_snap["verify_input"] = verify_input
        hybrid_snap["arrays"] = {
            i: list(c.cache) for i, c in enumerate(model_cache) if isinstance(c, ArraysCache)
        }
        hybrid_snap["kv_offsets"] = {
            i: c.offset for i, c in enumerate(model_cache) if isinstance(c, KVCache)
        }

    def _rewind(n_draft: int, n_accept: int):
        trim_prompt_cache(model_cache, n_draft - n_accept)
        if not (has_arrays and n_draft > n_accept):
            return 0.0
        t0 = time.monotonic()
        saved_arrays = hybrid_snap["arrays"]
        saved_kv_offsets = hybrid_snap["kv_offsets"]
        verify_input = hybrid_snap["verify_input"]
        if not saved_arrays or verify_input is None:
            return 0.0
        for i, c in enumerate(model_cache):
            if isinstance(c, KVCache) and i in saved_kv_offsets:
                c.offset = saved_kv_offsets[i]
            if isinstance(c, ArraysCache) and i in saved_arrays:
                c.cache = list(saved_arrays[i])
        rejected = n_draft - n_accept
        accepted = verify_input[:-rejected]
        with mx.stream(_STREAM):
            model(accepted[None], cache=model_cache)
            mx.eval([c.state for c in model_cache if isinstance(c, ArraysCache)])
        return (time.monotonic() - t0) * 1000.0

    with mx.stream(_STREAM):
        y = _prefill(model, model_cache, y)

    ntoks = 0
    num_draft = 0
    n_accept = 0
    round_idx = 0
    try:
        while True:
            remaining = max_tokens - ntoks
            draft_list = prompt_lookup_draft(
                history,
                min(remaining, num_draft_tokens),
                max_ngram_size=max_ngram_size,
                min_ngram_size=min_ngram_size,
            )
            num_draft = len(draft_list)

            if num_draft > 0:
                draft_tokens = mx.array(draft_list, mx.uint32)
                y_verify = mx.concatenate([y, draft_tokens])
                if has_arrays:
                    _snapshot(y_verify)
            else:
                y_verify = y

            t_verify = time.monotonic()
            tokens, logprobs = _step(model, model_cache, y_verify, num_draft + 1)
            mx.eval(tokens)
            verify_ms = (time.monotonic() - t_verify) * 1000.0

            tokens_list = tokens.tolist()

            n_accept = 0
            while n_accept < num_draft:
                if tokens_list[n_accept] != draft_list[n_accept]:
                    break
                tn, lpn = tokens_list[n_accept], logprobs[n_accept]
                n_accept += 1
                ntoks += 1
                history.append(tn)
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break

            if ntoks < max_tokens:
                fresh = tokens_list[n_accept]
                ntoks += 1
                history.append(fresh)
                yield fresh, logprobs[n_accept], False

            replay_ms = _rewind(num_draft, n_accept)
            if recorder is not None:
                recorder(
                    "pld_round",
                    {
                        "round": round_idx,
                        "num_draft": num_draft,
                        "num_accept": n_accept,
                        "verify_ms": verify_ms,
                        "replay_ms": replay_ms,
                    },
                )
            round_idx += 1

            if ntoks >= max_tokens:
                break

            y = mx.array([tokens_list[n_accept]], mx.uint32)
    finally:
        _rewind(num_draft, n_accept)
