"""Unit tests for prompt_lookup_draft."""
from __future__ import annotations

from slim_ml.prompt_lookup import prompt_lookup_draft


def test_empty_and_zero_budget():
    assert prompt_lookup_draft([], 4) == []
    assert prompt_lookup_draft([1, 2, 3], 0) == []


def test_no_match_returns_empty():
    # needle [9, 10] never appears earlier in the sequence
    assert prompt_lookup_draft([1, 2, 3, 4, 9, 10], 4) == []


def test_basic_match():
    # needle [4, 5] appears at index 3; return up to 4 tokens after that
    tokens = [1, 4, 5, 99, 4, 5]
    # Wait — this has the needle itself at the end. Let's be precise:
    # last 2 = (4, 5); search in tokens[:-0]? No — we scan i in range(L - n)
    # = range(6 - 2) = 0..3. tokens[0:2]=(1,4), tokens[1:3]=(4,5) ✓
    # start = 1 + 2 = 3, return tokens[3:7] = [99, 4, 5]
    assert prompt_lookup_draft(tokens, 4) == [99, 4, 5]


def test_longest_ngram_wins():
    # Both n=3 match (last 3 = A,B,C at idx 0) and n=2 match (B,C at idx 1)
    # exist. Longer needle should be tried first and return its followers.
    tokens = [10, 20, 30, 99, 20, 30, 10, 20, 30]
    # last 3 = (10, 20, 30), found at i=0, start=3, next 4 = [99, 20, 30, 10]
    assert prompt_lookup_draft(tokens, 4, max_ngram_size=3) == [99, 20, 30, 10]


def test_falls_back_to_smaller_ngram():
    # No 3-gram match for needle (2,3,4); does match on 2-gram (3,4)
    tokens = [1, 3, 4, 7, 8, 2, 3, 4]
    # n=3: needle=(2,3,4); tokens[0:3]=(1,3,4), tokens[1:3+1]=(3,4,7)... no match
    # n=2: needle=(3,4); tokens[1:3]=(3,4) ✓; start=3, return [7, 8, 2]
    assert prompt_lookup_draft(tokens, 3) == [7, 8, 2]


def test_match_near_end_returns_short_draft():
    # needle = (5, 6); match at i=0, start=2, sequence only has 3 tokens after
    # that point so draft is shorter than num_draft.
    tokens = [5, 6, 7, 5, 6]
    assert prompt_lookup_draft(tokens, 8) == [7, 5, 6]


def test_first_match_wins():
    # Needle (1, 2) appears at i=0; follower slice is [100, 5, 1, 2].
    tokens = [1, 2, 100, 5, 1, 2]
    assert prompt_lookup_draft(tokens, 4) == [100, 5, 1, 2]


def test_respects_min_ngram_size():
    # With min=3 and no 3-gram match, should return [] even if 2-gram matches.
    tokens = [1, 3, 4, 7, 8, 2, 3, 4]
    assert prompt_lookup_draft(tokens, 3, min_ngram_size=3) == []
