"""Tests targeting uncovered lines in verifiers/rewards.py.

Covers:
- ExactMatchReward._normalize on empty text (L136)
- TokenF1Reward with zero overlap → 0.0 (L286)
- TokenF1Reward tokenize="word" and "character" (L304-307)
- TokenF1Reward unknown tokenize method → ValueError (L309)
- IoUReward._extract_bbox with unclosed brace (L459)
- CombinedReward with empty rewards → ValueError (L496)
- CombinedReward weights count mismatch → ValueError (L503)
"""

from __future__ import annotations

import pytest

from gaze.verifiers.rewards import CombinedReward
from gaze.verifiers.rewards import ExactMatchReward
from gaze.verifiers.rewards import IoUReward
from gaze.verifiers.rewards import TokenF1Reward

# ---------------------------------------------------------------------------
# ExactMatchReward._normalize
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExactMatchNormalize:
    def test_normalize_empty_string_returns_empty(self) -> None:
        reward = ExactMatchReward(normalize=True, strip_braces=True)
        result = reward._normalize("")
        assert result == ""

    def test_normalize_strips_braces_and_collapses_whitespace(self) -> None:
        reward = ExactMatchReward(normalize=True, strip_braces=True)
        result = reward._normalize("{  hello   world  }")
        assert result == "hello world"

    def test_normalize_disabled_returns_unchanged(self) -> None:
        reward = ExactMatchReward(normalize=False)
        score = reward("", "  hello  ", {"gold": "  hello  "})
        assert score == 1.0


# ---------------------------------------------------------------------------
# TokenF1Reward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTokenF1Coverage:
    def test_zero_overlap_returns_zero(self) -> None:
        """When pred and ref share no tokens, precision+recall==0 → 0.0 (L286)."""
        reward = TokenF1Reward(
            normalize=True,
            filter_stopwords=True,
        )
        # Use words that are not stopwords but have zero overlap
        score = reward("", "alpha beta gamma", {"gold": "delta epsilon zeta"})
        assert score == 0.0

    def test_word_tokenizer(self) -> None:
        """tokenize='word' splits on whitespace (L304-305)."""
        reward = TokenF1Reward(tokenize="word", normalize=False)
        # Exact same text → F1 = 1.0
        score = reward("", "hello world", {"gold": "hello world"})
        assert score == 1.0

    def test_word_tokenizer_partial_overlap(self) -> None:
        reward = TokenF1Reward(tokenize="word", normalize=False, filter_stopwords=False)
        score = reward("", "a b c", {"gold": "a b d"})
        # pred_tokens = ["a", "b", "c"], ref_tokens = ["a", "b", "d"]
        # intersection = 2, precision = 2/3, recall = 2/3
        # F1 = 2 * (2/3) * (2/3) / ((2/3) + (2/3)) = 2/3
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_character_tokenizer(self) -> None:
        """tokenize='character' converts text to list of chars (L306-307)."""
        reward = TokenF1Reward(tokenize="character", normalize=False)
        score = reward("", "abc", {"gold": "abc"})
        assert score == 1.0

    def test_character_tokenizer_partial(self) -> None:
        reward = TokenF1Reward(tokenize="character", normalize=False)
        score = reward("", "ab", {"gold": "abc"})
        # pred = ['a','b'], ref = ['a','b','c']
        # intersection = 2, precision = 1.0, recall = 2/3
        # F1 = 2 * 1.0 * (2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 4/5
        assert abs(score - 0.8) < 1e-9

    def test_unknown_tokenizer_raises_valueerror(self) -> None:
        """Unknown tokenize method → ValueError (L309)."""
        reward = TokenF1Reward(tokenize="bpe")
        with pytest.raises(ValueError, match="Unknown tokenize method: bpe"):
            reward("", "hello", {"gold": "hello"})


# ---------------------------------------------------------------------------
# IoUReward._extract_bbox — unclosed brace (L459)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIoUBboxExtraction:
    def test_unclosed_brace_falls_through_to_regex(self) -> None:
        """Unclosed brace stops JSON search; regex fallback extracts bbox (L459)."""
        reward = IoUReward()
        # JSON has unclosed brace, but a valid [x1, y1, x2, y2] array for regex fallback
        text = '{"bbox": blah... and separately [10, 20, 30, 40]'
        bbox = reward._extract_bbox(text)
        assert bbox == [10.0, 20.0, 30.0, 40.0]

    def test_unclosed_brace_no_regex_returns_empty(self) -> None:
        """Unclosed brace with no regex-extractable coords → empty list."""
        reward = IoUReward()
        text = '{"bbox": broken data here'
        bbox = reward._extract_bbox(text)
        assert bbox == []

    def test_multiple_json_objects_takes_last_bbox(self) -> None:
        """Multiple JSON objects — last bbox wins."""
        reward = IoUReward()
        import json

        first = json.dumps({"bbox": [0, 0, 10, 10]})
        second = json.dumps({"bbox": [50, 50, 100, 100]})
        text = f"reasoning: {first} final answer: {second}"
        bbox = reward._extract_bbox(text)
        assert bbox == [50, 50, 100, 100]


# ---------------------------------------------------------------------------
# CombinedReward validation (L496, L503)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedRewardValidation:
    def test_empty_rewards_raises_valueerror(self) -> None:
        """Empty rewards list → ValueError (L496)."""
        with pytest.raises(ValueError, match="At least one reward function required"):
            CombinedReward(rewards=[])

    def test_weights_count_mismatch_raises_valueerror(self) -> None:
        """Weights count != rewards count → ValueError (L503)."""
        r1 = ExactMatchReward()
        with pytest.raises(ValueError, match="Number of weights must match"):
            CombinedReward(rewards=[r1], weights=[0.5, 0.5])

    def test_valid_combined_reward_computes_weighted_sum(self) -> None:
        r1 = ExactMatchReward()
        r2 = ExactMatchReward()
        combined = CombinedReward(rewards=[r1, r2], weights=[0.6, 0.4])
        # Exact match on both → 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        score = combined("", "hello", {"gold": "hello"})
        assert abs(score - 1.0) < 1e-9

    def test_combined_reward_partial_match(self) -> None:
        r_exact = ExactMatchReward()
        r_token = TokenF1Reward(tokenize="word", normalize=False, filter_stopwords=False)
        combined = CombinedReward(
            rewards=[r_exact, r_token],
            weights=[0.5, 0.5],
            names=["exact", "f1"],
        )
        # "hello" vs "hello world": exact=0.0, token_f1= 2*(1.0)*(0.5)/(1.5)=2/3
        score = combined("", "hello", {"gold": "hello world"})
        expected = 0.5 * 0.0 + 0.5 * (2.0 / 3.0)
        assert abs(score - expected) < 1e-9
