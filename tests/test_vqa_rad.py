"""Tests for VQA-RAD question handling, reward/evaluation parity, and edge cases.

Covers:
- Reward key dispatch (answer_type vs question_type)
- Closed reward binary matching
- Open reward token F1
- Answer type inference from dataset
- Evaluation/reward parity for closed and open questions
- Malformed JSON zero reward
- normalize_binary edge cases
"""

from __future__ import annotations

import json

import pytest

from examples.vqa_rad.src.evaluation import compute_token_f1
from examples.vqa_rad.src.evaluation import evaluate_closed_only
from examples.vqa_rad.src.evaluation import evaluate_vqa_rad
from examples.vqa_rad.src.evaluation import normalize_binary
from examples.vqa_rad.src.processor import VQARadVerifiersReward
from examples.vqa_rad.src.schemas import validate_vqa_rad_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(answer_dict: dict) -> str:
    """Wrap a dict as a JSON string completion."""
    return json.dumps(answer_dict)


def _make_info(
    answer: str,
    answer_type: str = "open",
) -> dict:
    """Build an info dict mimicking what the verifiers base class provides."""
    return {"answer": answer, "answer_type": answer_type}


# ---------------------------------------------------------------------------
# 1. test_reward_key_dispatch — Finding 1
# ---------------------------------------------------------------------------


class TestRewardKeyDispatch:
    """Verify VQARadVerifiersReward reads answer_type (not question_type)."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    def test_answer_type_closed_dispatches_binary(self) -> None:
        """Closed question should use binary match, not token F1."""
        completion = _make_completion({"answer": "yes"})
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", completion, info) == 1.0

    def test_answer_type_closed_verbose_still_matches(self) -> None:
        """Verbose 'yes ...' should match 'yes' under binary matching."""
        completion = _make_completion({"answer": "yes, the image shows a fracture"})
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", completion, info) == 1.0

    def test_answer_type_closed_wrong_answer(self) -> None:
        completion = _make_completion({"answer": "no"})
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", completion, info) == 0.0

    def test_answer_type_open_uses_f1(self) -> None:
        """Open question should use token F1, not binary match."""
        completion = _make_completion({"answer": "left lung opacity"})
        info = _make_info("opacity in left lung", answer_type="open")
        score = self.reward("", completion, info)
        # Token overlap is partial, should be between 0 and 1
        assert 0.0 < score < 1.0

    def test_fallback_to_question_type_key(self) -> None:
        """If only question_type is present, still works."""
        completion = _make_completion({"answer": "yes"})
        info = {"answer": "yes", "question_type": "closed"}
        assert self.reward("", completion, info) == 1.0

    def test_defaults_to_open_when_no_type(self) -> None:
        """With no type key at all, defaults to open (token F1)."""
        completion = _make_completion({"answer": "yes"})
        info = {"answer": "yes"}
        score = self.reward("", completion, info)
        # "yes" vs "yes" under token F1 = 1.0
        assert score == 1.0


# ---------------------------------------------------------------------------
# 2. test_closed_reward_binary_match — Finding 2
# ---------------------------------------------------------------------------


class TestClosedRewardBinaryMatch:
    """Verify closed reward uses normalize_binary (first-token extraction)."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    @pytest.mark.parametrize(
        "pred,ref,expected",
        [
            ("yes", "yes", 1.0),
            ("no", "no", 1.0),
            ("yes", "no", 0.0),
            ("no", "yes", 0.0),
            ("y", "yes", 1.0),
            ("n", "no", 1.0),
            ("Yes", "yes", 1.0),
            ("NO", "yes", 0.0),
        ],
    )
    def test_binary_basics(self, pred: str, ref: str, expected: float) -> None:
        completion = _make_completion({"answer": pred})
        info = _make_info(ref, answer_type="closed")
        assert self.reward("", completion, info) == expected

    def test_unrecognized_prediction_is_zero(self) -> None:
        """'possibly' is not yes/no, so closed reward is 0.0."""
        completion = _make_completion({"answer": "possibly"})
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", completion, info) == 0.0

    def test_verbose_yes_matches(self) -> None:
        completion = _make_completion({"answer": "yes the image is normal"})
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", completion, info) == 1.0

    def test_verbose_no_matches(self) -> None:
        completion = _make_completion({"answer": "no, there is no fracture"})
        info = _make_info("no", answer_type="closed")
        assert self.reward("", completion, info) == 1.0


# ---------------------------------------------------------------------------
# 3. test_open_reward_token_f1 — Finding 4
# ---------------------------------------------------------------------------


class TestOpenRewardTokenF1:
    """Verify open reward computes correct F1."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    def test_exact_match_gives_one(self) -> None:
        completion = _make_completion({"answer": "left lung opacity"})
        info = _make_info("left lung opacity", answer_type="open")
        assert self.reward("", completion, info) == 1.0

    def test_partial_overlap(self) -> None:
        completion = _make_completion({"answer": "fracture"})
        info = _make_info("no fracture", answer_type="open")
        score = self.reward("", completion, info)
        assert 0.0 < score < 1.0

    def test_no_overlap_gives_zero(self) -> None:
        completion = _make_completion({"answer": "normal"})
        info = _make_info("fracture", answer_type="open")
        assert self.reward("", completion, info) == 0.0

    def test_empty_prediction_gives_zero(self) -> None:
        completion = _make_completion({"answer": ""})
        info = _make_info("fracture", answer_type="open")
        assert self.reward("", completion, info) == 0.0

    def test_both_empty_gives_zero(self) -> None:
        """Both empty should be 0.0 (Finding 7 fix)."""
        completion = _make_completion({"answer": ""})
        info = _make_info("", answer_type="open")
        assert self.reward("", completion, info) == 0.0

    def test_articles_only_gives_zero(self) -> None:
        """Answers that normalize to empty (just articles) should be 0.0."""
        completion = _make_completion({"answer": "the"})
        info = _make_info("the", answer_type="open")
        assert self.reward("", completion, info) == 0.0


# ---------------------------------------------------------------------------
# 4. test_answer_type_inference — Finding 3
# ---------------------------------------------------------------------------


class TestAnswerTypeInference:
    """Verify dataset answer_type classification."""

    @pytest.mark.parametrize(
        "answer,expected_type",
        [
            ("yes", "closed"),
            ("no", "closed"),
            ("Yes", "closed"),
            ("No", "closed"),
            ("y", "closed"),
            ("n", "closed"),
            # These should NOT be classified as closed
            ("1", "open"),
            ("0", "open"),
            ("true", "open"),
            ("false", "open"),
            ("left lung", "open"),
            ("fracture", "open"),
            ("ct", "open"),
        ],
    )
    def test_answer_type_classification(self, answer: str, expected_type: str) -> None:
        # Replicate the dataset logic
        _closed_answers = frozenset({"yes", "no", "y", "n"})
        is_closed = answer.strip().lower() in _closed_answers
        result_type = "closed" if is_closed else "open"
        assert result_type == expected_type, f"answer={answer!r} classified as {result_type}"


# ---------------------------------------------------------------------------
# 5. test_eval_reward_parity_closed — Finding 2
# ---------------------------------------------------------------------------


class TestEvalRewardParityClosed:
    """Verify evaluation and reward agree on closed questions."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    @pytest.mark.parametrize(
        "pred,ref",
        [
            ("yes", "yes"),
            ("no", "no"),
            ("yes", "no"),
            ("no", "yes"),
            ("yes the image is normal", "yes"),
            ("no, there is no fracture", "no"),
        ],
    )
    def test_closed_parity(self, pred: str, ref: str) -> None:
        # Reward
        completion = _make_completion({"answer": pred})
        info = _make_info(ref, answer_type="closed")
        reward_score = self.reward("", completion, info)

        # Evaluation (evaluate_vqa_rad with closed type)
        metrics = evaluate_vqa_rad([pred], [ref], answer_types=["closed"])
        eval_score = metrics["closed_accuracy"]

        assert reward_score == eval_score, (
            f"pred={pred!r}, ref={ref!r}: reward={reward_score}, eval={eval_score}"
        )


# ---------------------------------------------------------------------------
# 6. test_eval_reward_parity_open
# ---------------------------------------------------------------------------


class TestEvalRewardParityOpen:
    """Verify evaluation and reward produce identical F1 for open questions."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    @pytest.mark.parametrize(
        "pred,ref",
        [
            ("left lung opacity", "left lung opacity"),
            ("fracture", "no fracture"),
            ("opacity left lung", "opacity in left lung"),
            ("normal", "abnormal"),
            ("", "fracture"),
        ],
    )
    def test_open_f1_parity(self, pred: str, ref: str) -> None:
        # Reward
        completion = _make_completion({"answer": pred})
        info = _make_info(ref, answer_type="open")
        reward_score = self.reward("", completion, info)

        # Evaluation
        eval_score = compute_token_f1(pred, ref)

        assert abs(reward_score - eval_score) < 1e-9, (
            f"pred={pred!r}, ref={ref!r}: reward={reward_score}, eval={eval_score}"
        )


# ---------------------------------------------------------------------------
# 7. test_malformed_json_zero_reward
# ---------------------------------------------------------------------------


class TestMalformedJsonZeroReward:
    """Verify malformed completions receive 0.0 reward."""

    def setup_method(self) -> None:
        self.reward = VQARadVerifiersReward()

    def test_plain_text_no_json(self) -> None:
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", "I think the answer is yes", info) == 0.0

    def test_empty_completion(self) -> None:
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", "", info) == 0.0

    def test_json_without_answer_key(self) -> None:
        completion = json.dumps({"reasoning": "looks normal"})
        info = _make_info("yes", answer_type="closed")
        # answer defaults to "" → normalize_binary("") → None → 0.0
        assert self.reward("", completion, info) == 0.0

    def test_incomplete_json(self) -> None:
        info = _make_info("fracture", answer_type="open")
        assert self.reward("", '{"answer": "frac', info) == 0.0

    def test_message_list_completion(self) -> None:
        """Completion as message list with valid JSON in assistant content."""
        messages = [
            {"role": "assistant", "content": json.dumps({"answer": "yes"})},
        ]
        info = _make_info("yes", answer_type="closed")
        assert self.reward("", messages, info) == 1.0


# ---------------------------------------------------------------------------
# 8. test_normalize_binary_edge_cases
# ---------------------------------------------------------------------------


class TestNormalizeBinaryEdgeCases:
    """Verify normalize_binary behavior on edge cases."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("yes", "yes"),
            ("Yes", "yes"),
            ("YES", "yes"),
            ("y", "yes"),
            ("Y", "yes"),
            ("no", "no"),
            ("No", "no"),
            ("NO", "no"),
            ("n", "no"),
            ("N", "no"),
            # These are recognized by normalize_binary but NOT used for
            # dataset classification (Finding 3 fix restricts dataset inference)
            ("true", "yes"),
            ("false", "no"),
            ("1", "yes"),
            ("0", "no"),
            # Unrecognized
            ("maybe", None),
            ("possibly", None),
            ("", None),
            ("  ", None),
            ("fracture", None),
        ],
    )
    def test_normalize_binary(self, input_str: str, expected: str | None) -> None:
        assert normalize_binary(input_str) == expected

    def test_verbose_yes_extracts_first_token(self) -> None:
        assert normalize_binary("yes, the image is normal") == "yes"

    def test_verbose_no_extracts_first_token(self) -> None:
        assert normalize_binary("no there is no fracture") == "no"


# ---------------------------------------------------------------------------
# 9. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Verify schema validation catches invalid responses."""

    def test_valid_response(self) -> None:
        response = {
            "answer": "yes",
            "answer_type": "closed",
            "confidence": 0.9,
            "reasoning": "The image shows...",
            "image_observations": ["opacity visible"],
            "region_of_interest": {
                "description": "Left lung",
                "location": "Lower lobe",
            },
            "continue": False,
        }
        assert validate_vqa_rad_response(response) is True

    def test_missing_answer_fails(self) -> None:
        response = {
            "answer_type": "closed",
            "confidence": 0.9,
            "reasoning": "The image shows...",
            "image_observations": [],
            "region_of_interest": {"description": "x", "location": "y"},
            "continue": False,
        }
        assert validate_vqa_rad_response(response) is False

    def test_empty_answer_fails(self) -> None:
        response = {
            "answer": "",
            "answer_type": "closed",
            "confidence": 0.9,
            "reasoning": "x",
            "image_observations": [],
            "region_of_interest": {"description": "x", "location": "y"},
            "continue": False,
        }
        assert validate_vqa_rad_response(response) is False

    def test_invalid_answer_type_fails(self) -> None:
        response = {
            "answer": "yes",
            "answer_type": "multiple_choice",
            "confidence": 0.9,
            "reasoning": "x",
            "image_observations": [],
            "region_of_interest": {"description": "x", "location": "y"},
            "continue": False,
        }
        assert validate_vqa_rad_response(response) is False

    def test_confidence_out_of_range_clamped(self) -> None:
        """Out-of-range confidence is clamped to 1.0 (not rejected)."""
        response = {
            "answer": "yes",
            "answer_type": "closed",
            "confidence": 1.5,
            "reasoning": "x",
            "image_observations": [],
            "region_of_interest": {"description": "x", "location": "y"},
            "continue": False,
        }
        assert validate_vqa_rad_response(response) is True
        assert response["confidence"] == 1.0


# ---------------------------------------------------------------------------
# 10. evaluate_vqa_rad integration
# ---------------------------------------------------------------------------


class TestEvaluateVqaRad:
    """Integration tests for the evaluate_vqa_rad function."""

    def test_all_correct_closed(self) -> None:
        preds = ["yes", "no", "yes"]
        refs = ["yes", "no", "yes"]
        types = ["closed", "closed", "closed"]
        metrics = evaluate_vqa_rad(preds, refs, types)
        assert metrics["closed_accuracy"] == 1.0

    def test_all_wrong_closed(self) -> None:
        preds = ["no", "yes"]
        refs = ["yes", "no"]
        types = ["closed", "closed"]
        metrics = evaluate_vqa_rad(preds, refs, types)
        assert metrics["closed_accuracy"] == 0.0

    def test_mixed_types(self) -> None:
        preds = ["yes", "left lung opacity"]
        refs = ["yes", "left lung opacity"]
        types = ["closed", "open"]
        metrics = evaluate_vqa_rad(preds, refs, types)
        assert metrics["closed_accuracy"] == 1.0
        assert metrics["open_accuracy"] == 1.0
        assert metrics["open_f1"] == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_vqa_rad(["yes"], ["yes", "no"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot evaluate empty"):
            evaluate_vqa_rad([], [])


# ---------------------------------------------------------------------------
# 11. evaluate_closed_only edge cases
# ---------------------------------------------------------------------------


class TestEvaluateClosedOnly:
    """Test evaluate_closed_only handles edge cases."""

    def test_unrecognized_pred_counted_wrong(self) -> None:
        """Prediction 'possibly' should be counted as incorrect."""
        metrics = evaluate_closed_only(["possibly"], ["yes"])
        assert metrics["accuracy"] == 0.0
        assert metrics["num_samples"] == 1.0

    def test_unrecognized_ref_excluded(self) -> None:
        """Reference 'maybe' is not a valid binary → sample excluded."""
        metrics = evaluate_closed_only(["yes"], ["maybe"])
        assert metrics["num_samples"] == 0.0

    def test_per_class_accuracy(self) -> None:
        # preds: yes, no, yes, no
        # refs:  yes, no, no,  yes
        # idx 0: pred=yes ref=yes → correct (yes class)
        # idx 1: pred=no  ref=no  → correct (no class)
        # idx 2: pred=yes ref=no  → wrong   (no class)
        # idx 3: pred=no  ref=yes → wrong   (yes class)
        preds = ["yes", "no", "yes", "no"]
        refs = ["yes", "no", "no", "yes"]
        metrics = evaluate_closed_only(preds, refs)
        assert metrics["accuracy"] == 0.5
        assert metrics["yes_accuracy"] == 0.5  # 1 correct / 2 yes refs
        assert metrics["no_accuracy"] == 0.5  # 1 correct / 2 no refs
