"""Evaluation integrity tests for Patch Set #2.

Tests cover:
1. Caption reward — multiset token F1 (frequency-aware)
2. Detection evaluation — recall50 and precision50 metrics
3. IoUReward — continuous vs step-function modes
"""

from __future__ import annotations

import pytest

from examples.nova.src.rewards import compute_caption_reward

# IoUReward from the core harness
from radiant_harness.verifiers.rewards import IoUReward

# Optional torch imports for detection tests
try:
    from examples.nova.src.evaluation.detection import evaluate_detection

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =====================================================================
# 1. Caption reward — multiset token F1
# =====================================================================


class TestCaptionRewardMultiset:
    """Verify that token frequency is respected in F1 computation."""

    def test_repeated_tokens_penalized(self) -> None:
        """Repeating a token should not inflate precision.

        'the the the dog' has 4 tokens, 3 of which are 'the'.
        Against 'the dog', multiset intersection is {'the':1, 'dog':1} = 2.
        Precision = 2/4 = 0.5, Recall = 2/2 = 1.0, F1 = 2/3 ≈ 0.667.
        With set-based (old bug), it would be 1.0.
        """
        score = compute_caption_reward("the the the dog", "the dog")
        assert score < 0.9, f"Expected < 0.9 for inflated repetition, got {score}"
        assert abs(score - 2 / 3) < 1e-6

    def test_identical_strings(self) -> None:
        """Identical strings should score 1.0."""
        assert compute_caption_reward("the quick brown fox", "the quick brown fox") == 1.0

    def test_empty_prediction(self) -> None:
        assert compute_caption_reward("", "some reference") == 0.0

    def test_empty_reference(self) -> None:
        assert compute_caption_reward("some prediction", "") == 0.0

    def test_both_empty(self) -> None:
        assert compute_caption_reward("", "") == 0.0

    def test_no_overlap(self) -> None:
        assert compute_caption_reward("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self) -> None:
        """One shared token out of two on each side."""
        score = compute_caption_reward("alpha beta", "alpha gamma")
        # intersection=1, pred_total=2, ref_total=2
        # P=0.5, R=0.5, F1=0.5
        assert abs(score - 0.5) < 1e-6

    def test_frequency_matters_both_sides(self) -> None:
        """Both pred and ref have repeated tokens — Counter intersection uses min."""
        # pred: a a b (counts: a=2, b=1)
        # ref:  a b b (counts: a=1, b=2)
        # intersection: min(a)=1, min(b)=1 => 2
        # P = 2/3, R = 2/3, F1 = 2/3
        score = compute_caption_reward("a a b", "a b b")
        assert abs(score - 2 / 3) < 1e-6

    def test_superset_prediction(self) -> None:
        """Prediction has all ref tokens plus extras — recall=1 but precision<1."""
        score = compute_caption_reward("a b c d", "a b")
        # intersection=2, P=2/4=0.5, R=2/2=1.0, F1=2/3
        assert abs(score - 2 / 3) < 1e-6

    def test_subset_prediction(self) -> None:
        """Prediction is a subset of reference — precision=1 but recall<1."""
        score = compute_caption_reward("a b", "a b c d")
        # intersection=2, P=2/2=1.0, R=2/4=0.5, F1=2/3
        assert abs(score - 2 / 3) < 1e-6


# =====================================================================
# 2. Detection evaluation — recall50 and precision50
# =====================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestDetectionRecallPrecision:
    """Verify that recall50/precision50 are computed and correct."""

    def test_recall_precision_in_results(self) -> None:
        """New metrics should appear in output."""
        preds = [{"boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [0]}]
        refs = [{"boxes": [[10, 10, 20, 20]], "scores": [1.0], "labels": [0]}]
        results = evaluate_detection(preds, refs)
        assert "recall50" in results
        assert "precision50" in results

    def test_perfect_detection(self) -> None:
        """Perfect match: recall=1.0, precision=1.0."""
        preds = [{"boxes": [[0, 0, 50, 50]], "scores": [0.9], "labels": [0]}]
        refs = [{"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [0]}]
        results = evaluate_detection(preds, refs)
        assert results["recall50"] == 1.0
        assert results["precision50"] == 1.0

    def test_missed_lesion_penalizes_recall(self) -> None:
        """Two GT boxes, one prediction matches — recall should be 0.5."""
        preds = [
            {
                "boxes": [[0, 0, 50, 50]],
                "scores": [0.9],
                "labels": [0],
            }
        ]
        refs = [
            {
                "boxes": [[0, 0, 50, 50], [100, 100, 200, 200]],
                "scores": [1.0, 1.0],
                "labels": [0, 0],
            }
        ]
        results = evaluate_detection(preds, refs)
        assert results["recall50"] == 0.5
        assert results["precision50"] == 1.0  # The one prediction was correct

    def test_false_positive_penalizes_precision(self) -> None:
        """One GT box, two predictions — one matches, one spurious."""
        preds = [
            {
                "boxes": [[0, 0, 50, 50], [200, 200, 300, 300]],
                "scores": [0.9, 0.8],
                "labels": [0, 0],
            }
        ]
        refs = [
            {
                "boxes": [[0, 0, 50, 50]],
                "scores": [1.0],
                "labels": [0],
            }
        ]
        results = evaluate_detection(preds, refs)
        assert results["recall50"] == 1.0  # The one GT box was found
        assert results["precision50"] == 0.5  # One of two predictions was correct

    def test_no_predictions_zero_recall(self) -> None:
        """No predictions with GT boxes — recall=0."""
        preds = [{"boxes": [], "scores": [], "labels": []}]
        refs = [{"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [0]}]
        results = evaluate_detection(preds, refs)
        assert results["recall50"] == 0.0

    def test_no_gt_no_preds(self) -> None:
        """No GT and no predictions — both should be 0.0 (no positives to recall/precision)."""
        preds = [{"boxes": [], "scores": [], "labels": []}]
        refs = [{"boxes": [], "scores": [], "labels": []}]
        results = evaluate_detection(preds, refs)
        # No GT boxes and no predictions: (tp=0, fp=0, fn=0)
        assert results["recall50"] == 0.0
        assert results["precision50"] == 0.0

    def test_acc50_still_works(self) -> None:
        """ACC50 should still function correctly alongside new metrics."""
        # Two samples: first has a match, second doesn't
        preds = [
            {"boxes": [[0, 0, 50, 50]], "scores": [0.9], "labels": [0]},
            {"boxes": [[200, 200, 300, 300]], "scores": [0.9], "labels": [0]},
        ]
        refs = [
            {"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [0]},
            {"boxes": [[0, 0, 50, 50]], "scores": [1.0], "labels": [0]},
        ]
        results = evaluate_detection(preds, refs)
        assert results["acc50"] == 0.5  # 1 of 2 samples has a hit


# =====================================================================
# 3. IoUReward — continuous vs step modes
# =====================================================================


class TestIoURewardContinuous:
    """Verify IoUReward continuous mode provides smooth gradient signal."""

    def _make_info(self, bbox: list[float]) -> dict:
        return {"bbox": bbox}

    def _make_completion(self, bbox: list[float]) -> str:
        import json

        return json.dumps({"bbox": bbox})

    def test_continuous_default(self) -> None:
        """Default mode should be continuous."""
        reward = IoUReward()
        assert reward.continuous is True

    def test_continuous_returns_raw_iou(self) -> None:
        """In continuous mode, should return the raw IoU value."""
        reward = IoUReward(continuous=True)
        # Perfect overlap
        score = reward("", self._make_completion([0, 0, 100, 100]), self._make_info([0, 0, 100, 100]))
        assert score == 1.0

    def test_continuous_partial_overlap(self) -> None:
        """Partial overlap returns raw IoU, not 1.0 or 0.0."""
        reward = IoUReward(continuous=True, iou_threshold=0.5)
        # Two 100x100 boxes offset by 50 pixels
        # Box1: (0,0)-(100,100), Box2: (50,50)-(150,150)
        # Intersection: 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429
        score = reward(
            "", self._make_completion([0, 0, 100, 100]), self._make_info([50, 50, 150, 150])
        )
        assert abs(score - 2500 / 17500) < 1e-4
        # This IoU is below 0.5 threshold — old code returned this too,
        # but above threshold it would have jumped to 1.0

    def test_continuous_above_threshold_not_clipped(self) -> None:
        """IoU above threshold should NOT be clipped to 1.0 in continuous mode."""
        reward = IoUReward(continuous=True, iou_threshold=0.3)
        # Two 100x100 boxes offset by 20 pixels
        # Box1: (0,0)-(100,100), Box2: (20,20)-(120,120)
        # Intersection: 80*80 = 6400
        # Union: 10000 + 10000 - 6400 = 13600
        # IoU = 6400/13600 ≈ 0.4706
        score = reward(
            "", self._make_completion([0, 0, 100, 100]), self._make_info([20, 20, 120, 120])
        )
        expected_iou = 6400 / 13600
        assert abs(score - expected_iou) < 1e-4
        # Old step function would return 1.0 here (iou >= 0.3)
        assert score < 1.0

    def test_step_mode_binary(self) -> None:
        """Step mode returns 1.0 above threshold, 0.0 below."""
        reward = IoUReward(continuous=False, iou_threshold=0.5)

        # Perfect overlap → 1.0
        score_perfect = reward(
            "", self._make_completion([0, 0, 100, 100]), self._make_info([0, 0, 100, 100])
        )
        assert score_perfect == 1.0

        # No overlap → 0.0
        score_none = reward(
            "", self._make_completion([0, 0, 10, 10]), self._make_info([200, 200, 300, 300])
        )
        assert score_none == 0.0

    def test_step_mode_below_threshold_is_zero(self) -> None:
        """In step mode, IoU below threshold should be 0.0 (not the raw IoU).

        The old buggy code returned raw IoU below threshold, which was
        inconsistent with the step function above threshold.
        """
        reward = IoUReward(continuous=False, iou_threshold=0.5)
        # Slight overlap (IoU ≈ 0.14) → should be 0.0 in step mode
        score = reward(
            "", self._make_completion([0, 0, 100, 100]), self._make_info([50, 50, 150, 150])
        )
        assert score == 0.0

    def test_no_prediction_returns_zero(self) -> None:
        """Missing bbox should return 0.0 regardless of mode."""
        for continuous in (True, False):
            reward = IoUReward(continuous=continuous)
            score = reward("", "no bounding box here", self._make_info([0, 0, 100, 100]))
            assert score == 0.0
