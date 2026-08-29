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

    def test_normalize_disabled_keeps_whitespace_significant(self) -> None:
        """With normalization off, padding must actually change the verdict.

        The previous version compared two byte-identical strings, so it passed
        whether or not `normalize` was honoured.
        """
        reward = ExactMatchReward(normalize=False)
        assert reward("", "  hello  ", {"gold": "  hello  "}) == 1.0
        assert reward("", "  hello  ", {"gold": "hello"}) == 0.0


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


class TestIoURewardPixelCoordinatesFailClosed:
    """The default IoUReward path with pixel coords was entirely untested.

    Every other test passes ``area_penalty_start=1.0`` to disable the penalty,
    so nothing pinned what plain ``IoUReward()`` does when a model emits pixel
    coordinates. It fails closed at 0.0 — deliberately, to stop a model dodging
    the area penalty by switching coordinate space — which means a *perfect*
    box scores zero unless ``info["image_area"]`` is supplied.
    """

    _PERFECT_PIXEL_BOX = '{"bbox": [10, 10, 50, 50]}'

    def test_perfect_pixel_box_scores_full_credit_without_image_area(self) -> None:
        """Regression: this used to fail closed at 0.0 for a correct box."""
        reward = IoUReward()("", self._PERFECT_PIXEL_BOX, {"bbox": [10, 10, 50, 50]})
        assert reward == pytest.approx(1.0)

    def test_degenerate_box_is_still_penalized_when_area_is_known(self) -> None:
        """Skipping the penalty must not become a way to game full-image boxes."""
        reward = IoUReward()(
            "", '{"bbox": [0, 0, 512, 512]}', {"bbox": [0, 0, 512, 512], "image_area": 512 * 512}
        )
        assert reward == 0.0

    def test_image_width_and_height_are_accepted_in_place_of_image_area(self) -> None:
        reward = IoUReward()(
            "",
            '{"bbox": [0, 0, 512, 512]}',
            {"bbox": [0, 0, 512, 512], "image_width": 512, "image_height": 512},
        )
        assert reward == 0.0

    def test_supplying_image_area_restores_the_score(self) -> None:
        reward = IoUReward()(
            "", self._PERFECT_PIXEL_BOX, {"bbox": [10, 10, 50, 50], "image_area": 65536}
        )
        assert reward == pytest.approx(1.0)

    def test_normalized_coordinates_need_no_image_area(self) -> None:
        reward = IoUReward()("", '{"bbox": [0.1, 0.1, 0.3, 0.3]}', {"bbox": [0.1, 0.1, 0.3, 0.3]})
        assert reward == pytest.approx(1.0)


class TestExactMatchReferenceHandling:
    """Gold answers are not always bare strings, and `normalize` must normalize."""

    def test_list_gold_uses_the_first_string_instead_of_crashing(self) -> None:
        """A list gold used to reach ``.strip()`` and raise AttributeError."""
        assert ExactMatchReward()("", "cat", {"gold": ["cat"]}) == 1.0
        assert ExactMatchReward()("", "dog", {"gold": ["cat"]}) == 0.0

    def test_normalize_strips_surrounding_whitespace(self) -> None:
        """Whitespace collapsing used to happen only when strip_braces was on."""
        reward = ExactMatchReward(normalize=True, strip_braces=False)
        assert reward("", "  Cat  ", {"gold": "cat"}) == 1.0

    def test_normalize_disabled_stays_strict(self) -> None:
        reward = ExactMatchReward(normalize=False, case_sensitive=True)
        assert reward("", "  Cat  ", {"gold": "cat"}) == 0.0

    def test_non_string_gold_degrades_instead_of_raising(self) -> None:
        assert ExactMatchReward()("", "42", {"gold": 42}) == 1.0
        assert ExactMatchReward()("", "cat", {"gold": None}) == 0.0
