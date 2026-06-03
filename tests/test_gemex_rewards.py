"""Tests for GEMeX-ThinkVG reward functions.

Focuses on reward-hacking vectors and correctness:
- Full-image bbox hack (area penalty)
- Contains-match answer hack (length penalty)
- Default bbox consistency
- Custom weights propagation
"""

from __future__ import annotations

import pytest

from examples.gemex_thinkvg.src.rewards.answer import compute_answer_reward
from examples.gemex_thinkvg.src.rewards.answer import compute_contains_match
from examples.gemex_thinkvg.src.rewards.bbox import IMAGE_SIZE
from examples.gemex_thinkvg.src.rewards.bbox import compute_bbox_reward
from examples.gemex_thinkvg.src.rewards.combined import RewardWeights
from examples.gemex_thinkvg.src.rewards.combined import compute_combined_reward

# ── BBox: full-image hack (Finding #1) ──────────────────────────────


class TestFullImageBboxPenalty:
    """A model predicting [0, 0, 336, 336] must NOT get a free ride."""

    def test_fullimage_small_gt_near_zero(self) -> None:
        """Full-image pred vs small GT should yield near-zero reward."""
        result = compute_bbox_reward(
            prediction=[0, 0, IMAGE_SIZE, IMAGE_SIZE],
            reference=[100, 100, 200, 200],
        )
        # Before fix: ~0.19 (proximity) or IoU-based. After: should be ≈0.
        assert result["reward"] < 0.05, (
            f"Full-image bbox should yield near-zero reward, got {result['reward']:.4f}"
        )

    def test_fullimage_large_gt_penalised(self) -> None:
        """Full-image pred vs large GT (IoU>0.5) should still be penalised."""
        result = compute_bbox_reward(
            prediction=[0, 0, IMAGE_SIZE, IMAGE_SIZE],
            reference=[50, 50, 300, 300],
        )
        # Without penalty this gives IoU ≈ 0.55.
        # With penalty (area_ratio=1.0): penalty = 0 → reward = 0.
        assert result["reward"] < 0.05, (
            f"Full-image bbox vs large GT should be penalised, got {result['reward']:.4f}"
        )

    def test_reasonable_pred_unaffected(self) -> None:
        """A correctly-sized prediction should NOT be penalised."""
        result = compute_bbox_reward(
            prediction=[100, 100, 200, 200],
            reference=[100, 100, 200, 200],
        )
        assert result["reward"] == pytest.approx(1.0), (
            f"Perfect prediction should get 1.0 reward, got {result['reward']:.4f}"
        )

    def test_half_image_pred_moderate_penalty(self) -> None:
        """A prediction covering ~50% should get little/no penalty."""
        half = IMAGE_SIZE // 2
        result = compute_bbox_reward(
            prediction=[0, 0, half, IMAGE_SIZE],
            reference=[0, 0, half, IMAGE_SIZE],
        )
        # area_ratio ≈ 0.5, at the boundary → penalty = 1.0 (no penalty)
        assert result["reward"] > 0.8, (
            f"Half-image perfect match should still score high, got {result['reward']:.4f}"
        )

    def test_70pct_image_pred_partial_penalty(self) -> None:
        """A prediction covering ~70% of the image should get some penalty."""
        w = int(IMAGE_SIZE * 0.837)  # sqrt(0.7) ≈ 0.837 → area ratio ≈ 0.7
        result = compute_bbox_reward(
            prediction=[0, 0, w, w],
            reference=[0, 0, w, w],
        )
        # area_ratio ≈ 0.7, penalty = (1 - 0.7) / 0.5 = 0.6
        # IoU = 1.0, so reward ≈ 0.6
        assert 0.3 < result["reward"] < 0.9, (
            f"70%-area perfect IoU should get partial penalty, got {result['reward']:.4f}"
        )

    def test_invalid_bbox_still_zero(self) -> None:
        """Invalid prediction bbox should get 0 reward."""
        result = compute_bbox_reward(
            prediction=[0, 0, 0, 0],  # x2 == x1
            reference=[100, 100, 200, 200],
        )
        assert result["reward"] == 0.0
        assert result["valid_prediction"] is False


# ── Answer: contains-match hack (Finding #2) ─────────────────────────


class TestContainsMatchLengthPenalty:
    """Contains-match should penalise verbose predictions."""

    def test_exact_containment_full_score(self) -> None:
        """If pred == ref, contains_match should be 1.0."""
        score = compute_contains_match("pleural effusion", "pleural effusion")
        assert score == pytest.approx(1.0)

    def test_slight_elaboration_high_score(self) -> None:
        """If pred is slightly longer, score should still be high."""
        score = compute_contains_match("small pleural effusion", "pleural effusion")
        # len("pleural effusion")=16, len("small pleural effusion")=22
        # After normalization. Score = 16/22 ≈ 0.73
        assert score > 0.6, f"Slight elaboration should score well: {score:.3f}"

    def test_kitchen_sink_low_score(self) -> None:
        """Dumping many diagnoses should get a low contains score."""
        kitchen_sink = (
            "pneumothorax pleural effusion consolidation atelectasis "
            "cardiomegaly nodule mass opacity fibrosis edema"
        )
        score = compute_contains_match(kitchen_sink, "effusion")
        # "effusion" in kitchen_sink → True, but length ratio is tiny
        assert score < 0.2, f"Kitchen-sink answer should be penalised, got {score:.3f}"

    def test_ref_in_pred_penalised_proportionally(self) -> None:
        """Penalty should scale with length disparity."""
        ref = "yes"
        short_pred = "yes, likely"
        long_pred = "yes, this is likely due to extensive bilateral involvement"

        short_score = compute_contains_match(short_pred, ref)
        long_score = compute_contains_match(long_pred, ref)

        assert short_score > long_score, (
            f"Shorter prediction should score higher: short={short_score}, long={long_score}"
        )

    def test_no_containment_zero(self) -> None:
        """No containment should return 0."""
        score = compute_contains_match("pneumothorax", "effusion")
        assert score == 0.0

    def test_both_empty_zero_score(self) -> None:
        """Both empty = no content to match → 0.0 (hardened against gaming)."""
        score = compute_contains_match("", "")
        assert score == pytest.approx(0.0)


class TestAnswerRewardIntegration:
    """Verify the complete answer reward pipeline is hack-resistant."""

    def test_kitchen_sink_open_ended(self) -> None:
        """Kitchen-sink answer for open_ended should score poorly."""
        kitchen_sink = (
            "pneumothorax pleural effusion consolidation atelectasis "
            "cardiomegaly nodule mass opacity fibrosis edema"
        )
        result = compute_answer_reward(
            prediction=kitchen_sink,
            reference="pleural effusion",
            question_type="open_ended",
        )
        # Should be much less than 1.0 due to length penalty + low token precision
        assert result["reward"] < 0.4, (
            f"Kitchen-sink open_ended should score low, got {result['reward']:.3f}"
        )

    def test_exact_match_full_reward(self) -> None:
        """Exact match should still get full reward."""
        result = compute_answer_reward(
            prediction="pleural effusion",
            reference="pleural effusion",
            question_type="open_ended",
        )
        assert result["reward"] > 0.9

    def test_closed_ended_exact_dominates(self) -> None:
        """For closed_ended, exact match should dominate."""
        result = compute_answer_reward(
            prediction="Yes",
            reference="Yes",
            question_type="closed_ended",
        )
        assert result["reward"] > 0.9


# ── Default bbox consistency (Finding #8) ─────────────────────────────


class TestDefaultBboxConsistency:
    """Missing bbox should yield 0 reward, not partial credit."""

    def test_missing_pred_bbox_zero_reward(self) -> None:
        """Default missing bbox [0,0,0,0] should be invalid → 0 reward."""
        result = compute_combined_reward(
            prediction={"answer": "effusion", "location": {"reference": "right lung"}},
            reference={
                "answer": "effusion",
                "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            },
        )
        # pred bbox defaults to [0,0,0,0] (invalid) → bbox_reward = 0
        assert result["bbox_reward"] == 0.0

    def test_missing_ref_bbox_zero_reward(self) -> None:
        """Missing reference bbox should yield 0 bbox reward."""
        result = compute_combined_reward(
            prediction={
                "answer": "effusion",
                "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            },
            reference={"answer": "effusion", "location": {"reference": "right lung"}},
        )
        # ref bbox defaults to [0,0,0,0] (invalid) → bbox_reward = 0
        assert result["bbox_reward"] == 0.0

    def test_missing_gt_fullimage_not_rewarded(self) -> None:
        """Full-image prediction with missing GT should NOT get IoU=1.0."""
        result = compute_combined_reward(
            prediction={
                "answer": "effusion",
                "location": {
                    "reference": "right lung",
                    "bbox": [0, 0, IMAGE_SIZE, IMAGE_SIZE],
                },
            },
            reference={
                "answer": "effusion",
                "location": {"reference": "right lung"},
                # bbox missing → defaults to [0,0,0,0]
            },
        )
        # Before fix: full-image vs full-image → IoU=1.0!
        # After fix: ref is [0,0,0,0] (invalid) → bbox_reward = 0
        assert result["bbox_reward"] == 0.0


# ── Custom weights propagation (Finding #3) ──────────────────────────


class TestCustomWeightsPropagation:
    """Verify that custom weights are actually used."""

    def test_answer_only_weights(self) -> None:
        """With answer=1.0, location=0, bbox=0, only answer should matter."""
        weights = RewardWeights(answer=1.0, location=0.0, bbox=0.0)
        result = compute_combined_reward(
            prediction={
                "answer": "effusion",
                "location": {"reference": "wrong", "bbox": [0, 0, 1, 1]},
            },
            reference={
                "answer": "effusion",
                "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            },
            weights=weights,
        )
        # Answer should be high, location/bbox components are zeroed out by weights
        assert result["reward"] == pytest.approx(result["answer_reward"], abs=1e-6)

    def test_bbox_only_weights(self) -> None:
        """With answer=0, location=0, bbox=1.0, only bbox should matter."""
        weights = RewardWeights(answer=0.0, location=0.0, bbox=1.0)
        result = compute_combined_reward(
            prediction={
                "answer": "wrong",
                "location": {"reference": "wrong", "bbox": [100, 100, 200, 200]},
            },
            reference={
                "answer": "effusion",
                "location": {"reference": "right lung", "bbox": [100, 100, 200, 200]},
            },
            weights=weights,
        )
        assert result["reward"] == pytest.approx(result["bbox_reward"], abs=1e-6)

    def test_environment_uses_custom_weights(self) -> None:
        """The environment reward closure should capture custom weights."""
        from examples.gemex_thinkvg.src.verifiers.environment import _make_gemex_reward

        # Default weights
        default_fn = _make_gemex_reward(None)

        # Custom weights emphasising answer heavily
        custom_weights = RewardWeights(answer=0.8, location=0.1, bbox=0.1)
        custom_fn = _make_gemex_reward(custom_weights)

        # A case with perfect answer but wrong bbox/location
        import json

        response = json.dumps(
            {
                "reasoning": "Test reasoning",
                "answer": "pleural effusion",
                "location": {"reference": "wrong region", "bbox": [0, 0, 1, 1]},
                "confidence": 0.9,
                "continue": False,
            }
        )
        completion = [{"role": "assistant", "content": response}]

        info = {
            "gold_answer": "pleural effusion",
            "gold_location": "right lower lobe",
            "gold_bbox": [100, 100, 200, 200],
            "question_type": "open_ended",
        }

        default_reward = default_fn("", completion, info)
        custom_reward = custom_fn("", completion, info)

        # Custom should give higher reward since answer weight is 0.8 vs 0.4
        assert custom_reward > default_reward, (
            f"Custom (answer=0.8) should reward higher than default: "
            f"custom={custom_reward:.3f}, default={default_reward:.3f}"
        )


# ── Combined adversarial scenarios ────────────────────────────────────


class TestCombinedAdversarialInputs:
    """End-to-end tests with adversarial inputs that a model might learn."""

    def test_all_hacks_combined(self) -> None:
        """A model using all hacks at once should score poorly."""
        result = compute_combined_reward(
            prediction={
                # Kitchen-sink answer
                "answer": (
                    "pneumothorax pleural effusion consolidation atelectasis "
                    "cardiomegaly nodule mass opacity fibrosis edema"
                ),
                "location": {
                    # Vague location
                    "reference": "chest",
                    # Full-image bbox
                    "bbox": [0, 0, IMAGE_SIZE, IMAGE_SIZE],
                },
            },
            reference={
                "answer": "pleural effusion",
                "location": {
                    "reference": "right lower lobe",
                    "bbox": [150, 200, 250, 300],
                },
                "question_type": "open_ended",
            },
        )
        # This should score much below a legitimate answer
        assert result["reward"] < 0.25, (
            f"All-hacks-combined should score low, got {result['reward']:.3f}"
        )

    def test_repeated_tokens_no_boost(self) -> None:
        """Repeating the reference answer N times should NOT boost score."""
        single = compute_answer_reward(
            prediction="pleural effusion",
            reference="pleural effusion",
        )
        repeated = compute_answer_reward(
            prediction="pleural effusion " * 10,
            reference="pleural effusion",
        )
        # Token F1 uses sets, so repetition doesn't help precision.
        # Contains-match has length penalty.
        assert repeated["reward"] < single["reward"], (
            f"Repeated answer should not beat single: "
            f"single={single['reward']:.3f}, repeated={repeated['reward']:.3f}"
        )

    def test_legitimate_answer_high_reward(self) -> None:
        """A legitimate answer should still score well."""
        result = compute_combined_reward(
            prediction={
                "answer": "pleural effusion",
                "location": {
                    "reference": "right lower lobe",
                    "bbox": [150, 200, 250, 300],
                },
            },
            reference={
                "answer": "pleural effusion",
                "location": {
                    "reference": "right lower lobe",
                    "bbox": [150, 200, 250, 300],
                },
                "question_type": "open_ended",
            },
        )
        assert result["reward"] > 0.9, (
            f"Legitimate answer should score high, got {result['reward']:.3f}"
        )
