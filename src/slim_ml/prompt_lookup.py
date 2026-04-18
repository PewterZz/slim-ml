"""Prompt Lookup Decoding (PLD) draft generator.

Replaces a draft model with an n-gram match against the tokens we've already
seen (prompt + generated). Cheap, zero extra params — the win is on workloads
where the target's next tokens already appear verbatim in context (code,
refactors, docs with repeated identifiers).

Algorithm: take the last `n` tokens as a needle, scan left-to-right for the
first earlier occurrence, and return up to `num_draft` tokens that followed
it. Try n = max_ngram_size..min_ngram_size; longer needles first so we
prefer specific matches. Returns [] if nothing matches — caller should treat
that as "no draft this round" and fall through to a single target step.
"""
from __future__ import annotations


def prompt_lookup_draft(
    tokens: list[int],
    num_draft: int,
    max_ngram_size: int = 3,
    min_ngram_size: int = 2,
) -> list[int]:
    if num_draft <= 0 or not tokens:
        return []
    L = len(tokens)
    for n in range(max_ngram_size, min_ngram_size - 1, -1):
        if L < n + 1:
            continue
        needle = tuple(tokens[-n:])
        for i in range(L - n):
            if tuple(tokens[i : i + n]) == needle:
                start = i + n
                return tokens[start : start + num_draft]
    return []
