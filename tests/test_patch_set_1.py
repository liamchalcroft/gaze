"""Tests for Patch Set 1: Evaluation integrity fixes.

Covers:
1. Area penalty for localization reward (NOVA + core IoUReward)
2. Uniform rescale_and_clamp_box (aspect-ratio preservation)
3. BERTScore baseline rescaling change is verified via integration
"""

from __future__ import annotations

import json

import pytest

from examples.nova.src.rewards import _area_penalty
from examples.nova.src.rewards import compute_localization_reward
from radiant_harness.verifiers.rewards import IoUReward

try:
    from examples.nova.src.evaluation.detection import rescale_and_clamp_box

    DETECTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DETECTION_AVAILABLE = False


# =====================================================================
# 1. Area penalty helper
# =====================================================================


class TestAreaPenalty:
    """Verify _area_penalty returns correct multiplier for various coverage."""

    def test_small_box_no_penalty(self) -> None:
        """A box covering 10% of the image should get full credit (1.0)."""
        # 100x100 box inside 1000x1000 image
        box = [0.0, 0.0, 100.0, 100.0]
        image_area = 1000.0 * 1000.0
        assert _area_penalty(box, image_area) == 1.0

    def test_half_image_no_penalty(self) -> None:
        """Exactly at penalty_start (50%) — should still be 1.0."""
        box = [0.0, 0.0, 500.0, 1000.0]
        image_area = 1000.0 * 1000.0
        assert _area_penalty(box, image_area) == 1.0

    def test_full_image_zero_reward(self) -> None:
        """A box covering the entire image should get 0.0."""
        box = [0.0, 0.0, 1000.0, 1000.0]
        image_area = 1000.0 * 1000.0
        assert _area_penalty(box, image_area) == 0.0

    def test_75_percent_coverage(self) -> None:
        """75% coverage → linear interpolation between 50% and 100%.

        area_ratio = 0.75, penalty_start = 0.5
        penalty = (1.0 - 0.75) / (1.0 - 0.5) = 0.25 / 0.5 = 0.5
        """
        box = [0.0, 0.0, 750.0, 1000.0]
        image_area = 1000.0 * 1000.0
        penalty = _area_penalty(box, image_area)
        assert abs(penalty - 0.5) < 1e-6

    def test_zero_image_area(self) -> None:
        """Zero image area → should return 1.0 (no penalty)."""
        box = [0.0, 0.0, 100.0, 100.0]
        assert _area_penalty(box, 0.0) == 1.0

    def test_custom_penalty_start(self) -> None:
        """Custom penalty_start=0.3: 60% coverage should be penalized."""
        box = [0.0, 0.0, 600.0, 1000.0]
        image_area = 1000.0 * 1000.0
        # area_ratio=0.6, penalty_start=0.3
        # penalty = (1.0 - 0.6) / (1.0 - 0.3) = 0.4/0.7 ≈ 0.5714
        penalty = _area_penalty(box, image_area, penalty_start=0.3)
        assert abs(penalty - 0.4 / 0.7) < 1e-6


# =====================================================================
# 2. compute_localization_reward with area penalty
# =====================================================================


class TestLocalizationRewardAreaPenalty:
    """Verify that full-image boxes are penalized in NOVA localization reward."""

    def test_full_image_box_penalized(self) -> None:
        """Full-image prediction should get zero reward due to area penalty.

        GT covers 80% of the image so the full-image pred has high IoU,
        but the area penalty for 100% coverage should drive reward to 0.
        """
        # Ground truth: large box covering most of 100x100 image
        gt = [[5, 5, 95, 95]]
        # Prediction: entire image
        pred = [[0, 0, 100, 100]]
        image_area = 100.0 * 100.0

        reward_with_penalty = compute_localization_reward(
            pred, gt, image_area=image_area
        )
        reward_without = compute_localization_reward(pred, gt, image_area=0.0)

        # Without penalty, high IoU → nonzero reward
        assert reward_without > 0.0
        # With penalty, full-image box (100% coverage) → penalty=0.0 → reward=0.0
        assert reward_with_penalty == 0.0

    def test_tight_box_not_penalized(self) -> None:
        """Tight prediction matching GT should not be penalized."""
        gt = [[40, 40, 60, 60]]
        pred = [[38, 38, 62, 62]]
        image_area = 100.0 * 100.0

        reward = compute_localization_reward(pred, gt, image_area=image_area)
        reward_no_area = compute_localization_reward(pred, gt, image_area=0.0)

        # Small box (<50% coverage) should not be penalized
        assert reward == reward_no_area
        assert reward > 0.5

    def test_no_image_area_backward_compatible(self) -> None:
        """Without image_area, behavior should be identical to old code."""
        gt = [[0, 0, 100, 100]]
        pred = [[0, 0, 100, 100]]

        reward = compute_localization_reward(pred, gt)
        assert reward == 1.0

    def test_half_image_box_moderate_penalty(self) -> None:
        """A box covering ~75% of the image should be moderately penalized."""
        gt = [[10, 10, 90, 90]]
        # Pred covers 75% of 100x100 image
        pred = [[0, 0, 75, 100]]
        image_area = 100.0 * 100.0

        reward = compute_localization_reward(pred, gt, image_area=image_area)
        reward_no_penalty = compute_localization_reward(pred, gt, image_area=0.0)

        # Should be penalized but not zero
        assert reward < reward_no_penalty
        assert reward >= 0.0


# =====================================================================
# 3. Core IoUReward area penalty
# =====================================================================


class TestIoURewardAreaPenalty:
    """Verify core IoUReward applies area penalty for normalized coords."""

    def _make_completion(self, bbox: list[float]) -> str:
        return json.dumps({"bbox": bbox})

    def _make_info(self, bbox: list[float]) -> dict:
        return {"bbox": bbox}

    def test_full_normalized_box_penalized(self) -> None:
        """[0,0,1,1] covers 100% of normalized space → should be penalized."""
        reward_fn = IoUReward(
            continuous=True,
            normalized=True,
            area_penalty_start=0.5,
        )
        # GT: small box. Pred: full image.
        score = reward_fn(
            "",
            self._make_completion([0.0, 0.0, 1.0, 1.0]),
            self._make_info([0.3, 0.3, 0.7, 0.7]),
        )
        # Should be heavily penalized (area_ratio=1.0 → penalty=0.0)
        assert score == 0.0

    def test_small_box_no_penalty(self) -> None:
        """Small box should not be penalized."""
        reward_fn = IoUReward(
            continuous=True,
            normalized=True,
            area_penalty_start=0.5,
        )
        score = reward_fn(
            "",
            self._make_completion([0.3, 0.3, 0.7, 0.7]),
            self._make_info([0.3, 0.3, 0.7, 0.7]),
        )
        # Perfect match, small box → IoU=1.0, no penalty
        assert score == 1.0

    def test_penalty_disabled(self) -> None:
        """area_penalty_start=1.0 should disable the penalty."""
        reward_fn = IoUReward(
            continuous=True,
            normalized=True,
            area_penalty_start=1.0,
        )
        score = reward_fn(
            "",
            self._make_completion([0.0, 0.0, 1.0, 1.0]),
            self._make_info([0.3, 0.3, 0.7, 0.7]),
        )
        # No penalty → raw IoU
        assert score > 0.0

    def test_step_mode_with_penalty(self) -> None:
        """Step mode should also apply area penalty."""
        reward_fn = IoUReward(
            continuous=False,
            normalized=True,
            iou_threshold=0.1,
            area_penalty_start=0.5,
        )
        # Full-image box overlaps everything above threshold
        score = reward_fn(
            "",
            self._make_completion([0.0, 0.0, 1.0, 1.0]),
            self._make_info([0.3, 0.3, 0.7, 0.7]),
        )
        # Step mode gives 1.0, but penalty makes it 0.0
        assert score == 0.0


# =====================================================================
# 4. rescale_and_clamp_box — uniform scale factor
# =====================================================================


@pytest.mark.skipif(not DETECTION_AVAILABLE, reason="detection module not available")
class TestRescaleAndClampBoxUniform:
    """Verify rescale_and_clamp_box preserves aspect ratio."""

    def test_within_bounds_unchanged(self) -> None:
        """Box within bounds should pass through unchanged."""
        box = [10, 20, 100, 200]
        result = rescale_and_clamp_box(box, 480, 480)
        assert result == [10.0, 20.0, 100.0, 200.0]

    def test_single_axis_overflow_uniform_scale(self) -> None:
        """When only x-axis overflows, both axes should be scaled uniformly.

        Old behavior: per-axis scaling would leave y untouched and only
        scale x, distorting the aspect ratio.
        New behavior: min(scale_x, scale_y) is used for both axes.
        """
        # Box: [0, 0, 960, 240] in a 480x480 image
        # max_x=960 > 480 → scale_x = 480/960 = 0.5
        # max_y=240 < 480 → scale_y = 1.0
        # Uniform: scale = min(0.5, 1.0) = 0.5
        # Expected output: [0, 0, 480, 120]  # noqa: ERA001
        result = rescale_and_clamp_box([0, 0, 960, 240], 480, 480)
        assert result == [0.0, 0.0, 480.0, 120.0]

    def test_both_axes_overflow(self) -> None:
        """Both axes overflow — min scale should be used."""
        # Box: [0, 0, 960, 720] in 480x480
        # scale_x = 480/960 = 0.5, scale_y = 480/720 = 0.667
        # Uniform: scale = min(0.5, 0.667) = 0.5
        # Expected output: [0, 0, 480, 360]  # noqa: ERA001
        result = rescale_and_clamp_box([0, 0, 960, 720], 480, 480)
        assert result == [0.0, 0.0, 480.0, 360.0]

    def test_aspect_ratio_preserved(self) -> None:
        """The width:height ratio of the box should be preserved after rescale."""
        box = [100, 50, 1000, 300]
        w, h = 480, 480
        result = rescale_and_clamp_box(box, w, h)

        original_width = box[2] - box[0]
        original_height = box[3] - box[1]
        original_ratio = original_width / original_height

        result_width = result[2] - result[0]
        result_height = result[3] - result[1]
        result_ratio = result_width / result_height

        assert abs(original_ratio - result_ratio) < 1e-4, (
            f"Aspect ratio changed: {original_ratio:.4f} → {result_ratio:.4f}"
        )

    def test_swapped_coordinates_handled(self) -> None:
        """Swapped coordinates (x1 > x2) should be fixed before scaling."""
        result = rescale_and_clamp_box([960, 240, 0, 0], 480, 480)
        assert result[0] <= result[2]
        assert result[1] <= result[3]
