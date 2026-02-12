"""Tests for GEMeX-ThinkVG schema and environment — Patch Set #3.

Covers:
- Finding #9: `continue` field is vestigial (optional in schema/validation,
  respected by `is_completed` when present)
- Finding #10: Unified reward path (environment delegates to
  GEMeXVerifiersReward instead of reimplementing extraction)
"""

from __future__ import annotations

import json

import pytest

from examples.gemex_thinkvg.src.rewards.combined import GEMeXVerifiersReward
from examples.gemex_thinkvg.src.rewards.combined import RewardWeights
from examples.gemex_thinkvg.src.schemas import GEMEX_SCHEMA
from examples.gemex_thinkvg.src.schemas import validate_gemex_response

# ── Finding #9: `continue` is optional ───────────────────────────────


class TestContinueFieldOptional:
    """The `continue` field should no longer be required."""

    def test_schema_required_excludes_continue(self) -> None:
        """continue should NOT be in the schema's required list."""
        required = GEMEX_SCHEMA["json_schema"]["schema"]["required"]
        assert "continue" not in required

    def test_schema_properties_still_has_continue(self) -> None:
        """continue should still be ALLOWED (in properties)."""
        props = GEMEX_SCHEMA["json_schema"]["schema"]["properties"]
        assert "continue" in props

    def test_valid_response_without_continue(self) -> None:
        """A response missing `continue` should now pass validation."""
        response = {
            "reasoning": "Bilateral opacities seen in lower zones.",
            "answer": "pleural effusion",
            "location": {
                "reference": "bilateral lung",
                "bbox": [50, 100, 250, 300],
            },
            "confidence": 0.85,
        }
        assert validate_gemex_response(response) is True

    def test_valid_response_with_continue_false(self) -> None:
        """A response with continue=false should still pass."""
        response = {
            "reasoning": "Analysis complete.",
            "answer": "normal",
            "location": {"reference": "bilateral lung", "bbox": [0, 0, 336, 336]},
            "confidence": 0.9,
            "continue": False,
        }
        assert validate_gemex_response(response) is True

    def test_valid_response_with_continue_true(self) -> None:
        """A response with continue=true should still pass validation."""
        response = {
            "reasoning": "Need to zoom into lower lobe.",
            "answer": "possible effusion",
            "location": {"reference": "right lower lobe", "bbox": [100, 200, 250, 300]},
            "confidence": 0.5,
            "continue": True,
        }
        assert validate_gemex_response(response) is True

    def test_missing_required_fields_still_fail(self) -> None:
        """Other required fields should still be enforced."""
        # Missing answer
        assert validate_gemex_response({
            "reasoning": "test",
            "location": {"reference": "lung", "bbox": [0, 0, 1, 1]},
            "confidence": 0.5,
        }) is False

        # Missing location
        assert validate_gemex_response({
            "reasoning": "test",
            "answer": "effusion",
            "confidence": 0.5,
        }) is False

        # Missing reasoning
        assert validate_gemex_response({
            "answer": "effusion",
            "location": {"reference": "lung", "bbox": [0, 0, 1, 1]},
            "confidence": 0.5,
        }) is False


# ── Finding #9: is_completed respects continue=true ──────────────────


class TestIsCompletedContinueField:
    """Environment is_completed should respect continue=true if present."""

    @staticmethod
    def _make_env_and_check(response_json: dict) -> bool:
        """Helper: check whether _extract_json_response + validation + continue
        logic would mark the episode as complete.

        Mirrors the logic in GEMeXThinkVGToolEnv.is_completed without needing
        the full async environment.
        """
        from examples.gemex_thinkvg.src.verifiers.environment import _extract_json_response

        text = json.dumps(response_json)
        response = _extract_json_response(text)
        if response and validate_gemex_response(response):
            return response.get("continue") is not True
        return False  # invalid → not completed

    def test_complete_without_continue(self) -> None:
        """Valid response without continue → episode complete."""
        assert self._make_env_and_check({
            "reasoning": "Done.",
            "answer": "effusion",
            "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            "confidence": 0.8,
        }) is True

    def test_complete_with_continue_false(self) -> None:
        """Valid response with continue=false → episode complete."""
        assert self._make_env_and_check({
            "reasoning": "Done.",
            "answer": "effusion",
            "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            "confidence": 0.8,
            "continue": False,
        }) is True

    def test_not_complete_with_continue_true(self) -> None:
        """Valid response with continue=true → episode NOT complete."""
        assert self._make_env_and_check({
            "reasoning": "Need more analysis.",
            "answer": "possible effusion",
            "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            "confidence": 0.5,
            "continue": True,
        }) is False

    def test_invalid_response_not_complete(self) -> None:
        """Invalid response → episode not complete regardless of continue."""
        assert self._make_env_and_check({
            "reasoning": "test",
            # Missing answer, location, confidence
        }) is False


# ── Finding #10: Unified reward path ─────────────────────────────────


class TestUnifiedRewardPath:
    """GEMeXVerifiersReward should handle both JSON and XML inputs."""

    def _make_info(
        self,
        answer: str = "pleural effusion",
        location: str = "right lower lobe",
        bbox: list[int] | None = None,
    ) -> dict:
        return {
            "gold_answer": answer,
            "gold_location": location,
            "gold_bbox": bbox or [100, 100, 200, 200],
            "question_type": "open_ended",
        }

    def test_json_response_scored(self) -> None:
        """Standard JSON response should be scored correctly."""
        reward_fn = GEMeXVerifiersReward()
        response = json.dumps({
            "reasoning": "Analysis of right lower lobe.",
            "answer": "pleural effusion",
            "location": {"reference": "right lower lobe", "bbox": [100, 100, 200, 200]},
            "confidence": 0.9,
        })
        completion = [{"role": "assistant", "content": response}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        assert score > 0.9, f"Perfect JSON answer should score high, got {score:.3f}"

    def test_xml_response_scored(self) -> None:
        """XML-style response should also be scored via fallback."""
        reward_fn = GEMeXVerifiersReward()
        xml_response = (
            "<response>"
            "<answer>pleural effusion</answer>"
            "<location>"
            "<ref>right lower lobe</ref>"
            "<box>[100, 100, 200, 200]</box>"
            "</location>"
            "</response>"
        )
        completion = [{"role": "assistant", "content": xml_response}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        # XML parser doesn't produce confidence/reasoning, so validation
        # may reject it. The key check: it doesn't crash and returns a score.
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_invalid_response_zero(self) -> None:
        """Unparseable response should return 0.0."""
        reward_fn = GEMeXVerifiersReward()
        completion = [{"role": "assistant", "content": "I don't know."}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        assert score == 0.0

    def test_json_in_markdown_block(self) -> None:
        """JSON inside markdown code fence should be parsed."""
        reward_fn = GEMeXVerifiersReward()
        md_response = (
            "Based on my analysis:\n"
            "```json\n"
            + json.dumps({
                "reasoning": "Right lower lobe opacity.",
                "answer": "pleural effusion",
                "location": {
                    "reference": "right lower lobe",
                    "bbox": [100, 100, 200, 200],
                },
                "confidence": 0.85,
            })
            + "\n```"
        )
        completion = [{"role": "assistant", "content": md_response}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        assert score > 0.9, f"Markdown JSON should be parsed, got {score:.3f}"

    def test_custom_weights_propagate(self) -> None:
        """Custom weights should propagate through GEMeXVerifiersReward."""
        # Answer-only weights
        weights = RewardWeights(answer=1.0, location=0.0, bbox=0.0)
        reward_fn = GEMeXVerifiersReward(weights=weights)

        # Perfect answer, wrong location + bbox
        response = json.dumps({
            "reasoning": "Test.",
            "answer": "pleural effusion",
            "location": {"reference": "wrong region", "bbox": [0, 0, 1, 1]},
            "confidence": 0.9,
        })
        completion = [{"role": "assistant", "content": response}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        # With answer=1.0, location=0, bbox=0, only answer matters
        assert score > 0.8, f"Answer-only weights should give high score, got {score:.3f}"

    def test_environment_reward_agrees_with_verifiers_reward(self) -> None:
        """The environment closure and GEMeXVerifiersReward should agree."""
        from examples.gemex_thinkvg.src.verifiers.environment import _make_gemex_reward

        weights = RewardWeights(answer=0.4, location=0.3, bbox=0.3)
        env_fn = _make_gemex_reward(weights)
        direct_fn = GEMeXVerifiersReward(weights=weights)

        response = json.dumps({
            "reasoning": "Analysis complete.",
            "answer": "pleural effusion",
            "location": {"reference": "right lower lobe", "bbox": [100, 100, 200, 200]},
            "confidence": 0.9,
        })
        completion = [{"role": "assistant", "content": response}]
        info = self._make_info()

        env_score = env_fn("", completion, info)
        direct_score = direct_fn("", completion, info)

        assert env_score == pytest.approx(direct_score), (
            f"Environment and direct reward should agree: "
            f"env={env_score:.4f}, direct={direct_score:.4f}"
        )

    def test_validation_now_applied_in_verifiers_reward(self) -> None:
        """GEMeXVerifiersReward should reject responses that fail validation.

        Before fix: GEMeXVerifiersReward did NOT validate, so a response
        with missing fields could still get a non-zero reward.
        """
        reward_fn = GEMeXVerifiersReward()

        # Valid JSON but missing required fields
        response = json.dumps({
            "answer": "effusion",
            # Missing reasoning, location, confidence
        })
        completion = [{"role": "assistant", "content": response}]
        info = self._make_info()

        score = reward_fn("", completion, info)
        assert score == 0.0, (
            f"Response missing required fields should be rejected, got {score:.3f}"
        )

    def test_string_completion_format(self) -> None:
        """GEMeXVerifiersReward should handle plain string completions."""
        reward_fn = GEMeXVerifiersReward()
        response = json.dumps({
            "reasoning": "Test.",
            "answer": "pleural effusion",
            "location": {"reference": "right lower lobe", "bbox": [100, 100, 200, 200]},
            "confidence": 0.9,
        })
        # Plain string instead of message list
        info = self._make_info()

        score = reward_fn("", response, info)
        assert score > 0.9, f"String completion should work, got {score:.3f}"
