"""Tests for visual tools: image operations, tool executors, and factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools.registry import ToolRegistry
from radiant_harness.tools.visual import WINDOW_PRESETS
from radiant_harness.tools.visual import adaptive_equalize
from radiant_harness.tools.visual import adjust_brightness
from radiant_harness.tools.visual import adjust_contrast
from radiant_harness.tools.visual import adjust_sharpness
from radiant_harness.tools.visual import annotate_region
from radiant_harness.tools.visual import apply_intensity_threshold
from radiant_harness.tools.visual import apply_window_level
from radiant_harness.tools.visual import compute_intensity_profile
from radiant_harness.tools.visual import compute_symmetry_diff
from radiant_harness.tools.visual import create_visual_tools
from radiant_harness.tools.visual import crop_image
from radiant_harness.tools.visual import denoise_gaussian
from radiant_harness.tools.visual import detect_edges
from radiant_harness.tools.visual import draw_grid_overlay
from radiant_harness.tools.visual import equalize_histogram
from radiant_harness.tools.visual import flip_horizontal
from radiant_harness.tools.visual import flip_vertical
from radiant_harness.tools.visual import get_intensity_stats
from radiant_harness.tools.visual import invert_image
from radiant_harness.tools.visual import measure_distance
from radiant_harness.tools.visual import morphological_op
from radiant_harness.tools.visual import rotate_90
from radiant_harness.tools.visual import zoom_image


def _make_image(
    width: int = 100, height: int = 100, color: tuple[int, ...] = (128, 128, 128)
) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _save_image(tmp_path: Path, width: int = 100, height: int = 100) -> Path:
    path = tmp_path / "test.png"
    _make_image(width, height).save(path)
    return path


# ── Default config used across pure-function tests ──────────────────
_CFG = ImageProcessingConfig()


# =====================================================================
# Pure image operations
# =====================================================================


class TestZoomImage:
    def test_zoom_2x_doubles_dimensions(self) -> None:
        img = _make_image(50, 50)
        result = zoom_image(img, 2.0, config=_CFG)
        assert result.size == (100, 100)

    def test_zoom_half_halves_dimensions(self) -> None:
        img = _make_image(100, 100)
        result = zoom_image(img, 0.5, config=_CFG)
        # max(10, 50) = 50
        assert result.size == (50, 50)

    def test_zoom_enforces_min_size(self) -> None:
        img = _make_image(10, 10)
        result = zoom_image(img, 0.5, config=_CFG)
        # int(10*0.5)=5, but min_image_size=10 → (10, 10)
        assert result.size[0] >= _CFG.min_image_size
        assert result.size[1] >= _CFG.min_image_size

    def test_zoom_preserves_aspect_ratio_at_min_size(self) -> None:
        """Zoom must preserve aspect ratio even when clamping to min_image_size."""
        img = _make_image(10, 20)
        result = zoom_image(img, 0.5, config=_CFG)
        # Both dims must meet min_image_size, and ratio must be preserved
        assert result.size[0] >= _CFG.min_image_size
        assert result.size[1] >= _CFG.min_image_size
        # Original aspect ratio is 1:2, result must maintain it
        original_ratio = 10 / 20
        result_ratio = result.size[0] / result.size[1]
        assert abs(original_ratio - result_ratio) < 0.15, (
            f"Aspect ratio distorted: original={original_ratio:.2f}, result={result_ratio:.2f}"
        )

    def test_zoom_nonsquare_preserves_ratio(self) -> None:
        """Non-square image zoomed down should keep proportions."""
        img = _make_image(15, 30)
        result = zoom_image(img, 0.5, config=_CFG)
        assert result.size[0] >= _CFG.min_image_size
        assert result.size[1] >= _CFG.min_image_size
        original_ratio = 15 / 30
        result_ratio = result.size[0] / result.size[1]
        assert abs(original_ratio - result_ratio) < 0.15

    def test_zoom_factor_below_min_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="factor must be in range"):
            zoom_image(img, 0.1, config=_CFG)

    def test_zoom_factor_above_max_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="factor must be in range"):
            zoom_image(img, 10.0, config=_CFG)


class TestCropImage:
    def test_crop_full_image(self) -> None:
        img = _make_image(100, 100)
        result = crop_image(img, (0.0, 0.0, 1.0, 1.0), config=_CFG)
        assert result.size == (100, 100)

    def test_crop_quarter(self) -> None:
        img = _make_image(100, 100)
        result = crop_image(img, (0.0, 0.0, 0.5, 0.5), config=_CFG)
        assert result.size == (50, 50)

    def test_crop_coords_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="must be in range"):
            crop_image(img, (-0.1, 0.0, 1.0, 1.0), config=_CFG)

    def test_crop_inverted_coords_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="x2 must be > x1"):
            crop_image(img, (0.8, 0.0, 0.2, 1.0), config=_CFG)

    def test_crop_too_small_image_raises(self) -> None:
        img = _make_image(5, 5)
        with pytest.raises(ValueError, match="Image too small"):
            crop_image(img, (0.0, 0.0, 1.0, 1.0), config=_CFG)

    def test_crop_result_too_small_raises(self) -> None:
        img = _make_image(100, 100)
        # Crop to a 1x100 pixel region → width < min_image_size
        with pytest.raises(ValueError, match="too small"):
            crop_image(img, (0.0, 0.0, 0.05, 1.0), config=_CFG)


class TestAdjustContrast:
    def test_factor_1_preserves_image(self) -> None:
        img = _make_image()
        result = adjust_contrast(img, 1.0, config=_CFG)
        assert result.size == img.size

    def test_high_contrast(self) -> None:
        img = _make_image()
        result = adjust_contrast(img, 2.0, config=_CFG)
        assert result.size == img.size

    def test_invalid_factor_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_contrast(img, 0.1, config=_CFG)

        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_contrast(img, 5.0, config=_CFG)


class TestApplyIntensityThreshold:
    def test_basic_threshold(self) -> None:
        img = _make_image()
        result = apply_intensity_threshold(img, 50, 200)
        assert result.mode == "L"
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_full_range_preserves_pixels(self) -> None:
        # Uniform gray 128 → after threshold [0,255] and rescale should stay 128
        img = Image.new("L", (10, 10), color=128)
        result = apply_intensity_threshold(img, 0, 255)
        arr = np.array(result)
        assert np.all(arr == 128)

    def test_lower_negative_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="lower must be >= 0"):
            apply_intensity_threshold(img, -1, 200)

    def test_upper_exceeds_255_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="upper must be <= 255"):
            apply_intensity_threshold(img, 0, 256)

    def test_upper_leq_lower_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="upper must be > lower"):
            apply_intensity_threshold(img, 100, 100)


class TestFlipAndRotate:
    def test_flip_horizontal_mirrors_pixels(self) -> None:
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        # Set top-left to red
        img.putpixel((0, 0), (255, 0, 0))
        result = flip_horizontal(img)
        # Red pixel should now be at top-right
        assert result.getpixel((3, 0)) == (255, 0, 0)
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_flip_vertical_mirrors_pixels(self) -> None:
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))
        result = flip_vertical(img)
        # Red pixel should now be at bottom-left
        assert result.getpixel((0, 3)) == (255, 0, 0)
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_rotate_clockwise_swaps_dimensions(self) -> None:
        img = _make_image(100, 50)
        result = rotate_90(img, clockwise=True)
        assert result.size == (50, 100)

    def test_rotate_counter_clockwise(self) -> None:
        img = _make_image(100, 50)
        result = rotate_90(img, clockwise=False)
        assert result.size == (50, 100)

    def test_double_flip_horizontal_is_identity(self) -> None:
        img = _make_image(20, 20)
        img.putpixel((0, 0), (255, 0, 0))
        result = flip_horizontal(flip_horizontal(img))
        assert result.getpixel((0, 0)) == (255, 0, 0)

    def test_four_rotations_is_identity(self) -> None:
        img = _make_image(20, 20)
        img.putpixel((0, 0), (255, 0, 0))
        result = img
        for _ in range(4):
            result = rotate_90(result, clockwise=True)
        assert result.getpixel((0, 0)) == (255, 0, 0)


class TestAdjustBrightness:
    def test_factor_1_preserves_image(self) -> None:
        img = _make_image()
        result = adjust_brightness(img, 1.0, config=_CFG)
        assert result.size == img.size

    def test_increase_brightness(self) -> None:
        img = _make_image(50, 50, color=(100, 100, 100))
        result = adjust_brightness(img, 2.0, config=_CFG)
        arr_orig = np.array(img)
        arr_result = np.array(result)
        assert arr_result.mean() > arr_orig.mean()

    def test_decrease_brightness(self) -> None:
        img = _make_image(50, 50, color=(200, 200, 200))
        result = adjust_brightness(img, 0.5, config=_CFG)
        arr_orig = np.array(img)
        arr_result = np.array(result)
        assert arr_result.mean() < arr_orig.mean()

    def test_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_brightness(img, 0.1, config=_CFG)
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_brightness(img, 5.0, config=_CFG)


class TestAdjustSharpness:
    def test_factor_1_preserves_image(self) -> None:
        img = _make_image()
        result = adjust_sharpness(img, 1.0, config=_CFG)
        assert result.size == img.size

    def test_sharpen(self) -> None:
        img = _make_image()
        result = adjust_sharpness(img, 2.0, config=_CFG)
        assert result.size == img.size

    def test_blur(self) -> None:
        img = _make_image()
        result = adjust_sharpness(img, 0.0, config=_CFG)
        assert result.size == img.size

    def test_negative_raises(self) -> None:
        img = _make_image()
        cfg = ImageProcessingConfig(min_sharpness_factor=0.0, max_sharpness_factor=3.0)
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_sharpness(img, -0.5, config=cfg)


class TestEqualizeHistogram:
    def test_returns_grayscale(self) -> None:
        img = _make_image()
        result = equalize_histogram(img)
        assert result.mode == "L"

    def test_preserves_size(self) -> None:
        img = _make_image(80, 60)
        result = equalize_histogram(img)
        assert result.size == (80, 60)

    def test_spreads_narrow_range(self) -> None:
        # Image with narrow intensity range should be spread after equalization
        arr = np.full((50, 50), 128, dtype=np.uint8)
        arr[10:20, 10:20] = 130
        img = Image.fromarray(arr, mode="L")
        result = equalize_histogram(img)
        result_arr = np.array(result)
        # After equalization, the range should be wider
        assert result_arr.max() - result_arr.min() >= arr.max() - arr.min()


class TestGetIntensityStats:
    def test_uniform_image_stats(self) -> None:
        img = Image.new("L", (50, 50), color=128)
        stats = get_intensity_stats(img)
        assert stats["mean"] == pytest.approx(128.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["min"] == 128
        assert stats["max"] == 128
        assert stats["median"] == pytest.approx(128.0)

    def test_rgb_conversion(self) -> None:
        img = _make_image(50, 50, color=(100, 150, 200))
        stats = get_intensity_stats(img)
        assert "mean" in stats
        assert isinstance(stats["mean"], float)

    def test_sub_region_box(self) -> None:
        img = Image.new("L", (100, 100), color=0)
        # Fill top-left quadrant with white
        arr = np.array(img)
        arr[:50, :50] = 255
        img = Image.fromarray(arr)
        stats_full = get_intensity_stats(img)
        stats_tl = get_intensity_stats(img, box=(0.0, 0.0, 0.5, 0.5))
        # Top-left is all white, full image is mixed
        assert stats_tl["mean"] > stats_full["mean"]  # type: ignore[operator]

    def test_invalid_box_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="box coordinates must be in"):
            get_intensity_stats(img, box=(-0.1, 0.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="x2 must be > x1"):
            get_intensity_stats(img, box=(0.8, 0.0, 0.2, 1.0))

    def test_histogram_has_10_bins(self) -> None:
        img = _make_image()
        stats = get_intensity_stats(img)
        assert len(stats["histogram"]) == 10  # type: ignore[arg-type]


class TestMeasureDistance:
    def test_horizontal_distance(self) -> None:
        img = _make_image(100, 100)
        result = measure_distance(img, (0.0, 0.5), (1.0, 0.5))
        assert result["distance_pixels"] == pytest.approx(100.0)

    def test_vertical_distance(self) -> None:
        img = _make_image(100, 200)
        result = measure_distance(img, (0.5, 0.0), (0.5, 1.0))
        assert result["distance_pixels"] == pytest.approx(200.0)

    def test_diagonal_distance(self) -> None:
        img = _make_image(100, 100)
        result = measure_distance(img, (0.0, 0.0), (1.0, 1.0))
        expected = (100**2 + 100**2) ** 0.5
        assert result["distance_pixels"] == pytest.approx(expected)

    def test_same_point_zero(self) -> None:
        img = _make_image()
        result = measure_distance(img, (0.5, 0.5), (0.5, 0.5))
        assert result["distance_pixels"] == pytest.approx(0.0)

    def test_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="coordinates must be in"):
            measure_distance(img, (-0.1, 0.5), (1.0, 0.5))
        with pytest.raises(ValueError, match="coordinates must be in"):
            measure_distance(img, (0.5, 0.5), (0.5, 1.1))

    def test_returns_pixel_coords(self) -> None:
        img = _make_image(200, 100)
        result = measure_distance(img, (0.5, 0.5), (1.0, 1.0))
        assert result["point1_pixels"] == (100.0, 50.0)
        assert result["point2_pixels"] == (200.0, 100.0)
        assert result["image_size"] == (200, 100)


class TestDrawGridOverlay:
    def test_preserves_size(self) -> None:
        img = _make_image(100, 100)
        result = draw_grid_overlay(img, 4, config=_CFG)
        assert result.size == (100, 100)

    def test_converts_grayscale_to_rgb(self) -> None:
        img = Image.new("L", (100, 100), color=128)
        result = draw_grid_overlay(img, 3, config=_CFG)
        assert result.mode == "RGB"

    def test_does_not_mutate_input(self) -> None:
        img = _make_image(100, 100)
        original_data = list(img.getdata())
        draw_grid_overlay(img, 4, config=_CFG)
        assert list(img.getdata()) == original_data

    def test_divisions_below_2_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="divisions must be >= 2"):
            draw_grid_overlay(img, 1, config=_CFG)

    def test_divisions_above_max_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="divisions must be <="):
            draw_grid_overlay(img, 100, config=_CFG)

    def test_has_visible_lines(self) -> None:
        # Black image + green grid lines should introduce green pixels
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        result = draw_grid_overlay(img, 4, config=_CFG)
        arr = np.array(result)
        # Green channel should have some non-zero pixels (grid lines)
        assert arr[:, :, 1].max() > 0


class TestDetectEdges:
    def test_sobel_returns_grayscale(self) -> None:
        img = _make_image()
        result = detect_edges(img, method="sobel")
        assert result.mode == "L"
        assert result.size == img.size

    def test_laplacian_returns_grayscale(self) -> None:
        img = _make_image()
        result = detect_edges(img, method="laplacian")
        assert result.mode == "L"

    def test_detects_sharp_boundary(self) -> None:
        # Image with left half black, right half white → edge in middle
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:, 50:] = 255
        img = Image.fromarray(arr)
        result = detect_edges(img, method="sobel")
        result_arr = np.array(result)
        # Middle column should have high edge values
        assert result_arr[:, 50].mean() > result_arr[:, 0].mean()

    def test_invalid_method_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="method must be"):
            detect_edges(img, method="invalid")


class TestComputeSymmetryDiff:
    def test_symmetric_image_low_diff(self) -> None:
        # Horizontally symmetric image should have near-zero diff
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:, :50] = 128
        arr[:, 50:] = 128
        img = Image.fromarray(arr)
        result = compute_symmetry_diff(img)
        assert np.array(result).max() == 0

    def test_asymmetric_image_high_diff(self) -> None:
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:, 80:] = 255  # only right side bright
        img = Image.fromarray(arr)
        result = compute_symmetry_diff(img)
        assert np.array(result).max() > 0

    def test_returns_grayscale(self) -> None:
        img = _make_image()
        result = compute_symmetry_diff(img)
        assert result.mode == "L"
        assert result.size == img.size


class TestAnnotateRegion:
    def test_draws_box_returns_rgb(self) -> None:
        img = Image.new("L", (100, 100), color=0)
        result = annotate_region(img, (0.1, 0.1, 0.9, 0.9))
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_does_not_mutate_input(self) -> None:
        img = _make_image(100, 100)
        original_data = list(img.getdata())
        annotate_region(img, (0.1, 0.1, 0.5, 0.5))
        assert list(img.getdata()) == original_data

    def test_with_label(self) -> None:
        img = _make_image()
        result = annotate_region(img, (0.1, 0.1, 0.5, 0.5), color="green", label="lesion")
        assert result.size == img.size

    def test_invalid_box_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="box coordinates must be in"):
            annotate_region(img, (-0.1, 0.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="x2 must be > x1"):
            annotate_region(img, (0.8, 0.0, 0.2, 1.0))


class TestInvertImage:
    def test_invert_returns_grayscale(self) -> None:
        img = _make_image()
        result = invert_image(img)
        assert result.mode == "L"

    def test_double_invert_identity(self) -> None:
        img = Image.new("L", (50, 50), color=100)
        result = invert_image(invert_image(img))
        assert np.array_equal(np.array(img), np.array(result))

    def test_black_becomes_white(self) -> None:
        img = Image.new("L", (10, 10), color=0)
        result = invert_image(img)
        assert np.array(result).min() == 255


class TestApplyWindowLevel:
    def test_brain_preset(self) -> None:
        # Use gradient image with range covering the brain window [0, 80]
        arr = np.linspace(0, 80, 100 * 100, dtype=np.uint8).reshape(100, 100)
        img = Image.fromarray(arr, mode="L")
        result = apply_window_level(img, preset="brain")
        assert result.mode == "L"
        assert result.size == img.size

    def test_custom_center_width(self) -> None:
        img = Image.new("L", (50, 50), color=128)
        result = apply_window_level(img, center=128, width=256)
        assert result.mode == "L"

    def test_unknown_preset_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="Unknown preset"):
            apply_window_level(img, preset="nonexistent")

    def test_missing_both_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="Must provide"):
            apply_window_level(img)

    def test_all_presets_work(self) -> None:
        # Full 0-255 gradient ensures all presets overlap with image range
        arr = np.linspace(0, 255, 100 * 100, dtype=np.uint8).reshape(100, 100)
        img = Image.fromarray(arr, mode="L")
        for preset_name in WINDOW_PRESETS:
            result = apply_window_level(img, preset=preset_name)
            assert result.mode == "L"

    def test_degenerate_window_raises(self) -> None:
        """Window that doesn't overlap image range must raise."""
        arr = np.arange(100, 200, dtype=np.uint8).reshape(10, 10)
        img = Image.fromarray(arr, mode="L")
        with pytest.raises(ValueError, match="does not overlap"):
            apply_window_level(img, center=25, width=50)


class TestAdaptiveEqualize:
    def test_returns_grayscale(self) -> None:
        img = _make_image(64, 64)
        result = adaptive_equalize(img, clip_limit=2.0, tile_size=4, config=_CFG)
        assert result.mode == "L"
        assert result.size == (64, 64)

    def test_clip_limit_out_of_range_raises(self) -> None:
        img = _make_image(64, 64)
        with pytest.raises(ValueError, match="clip_limit must be in"):
            adaptive_equalize(img, clip_limit=0.5, config=_CFG)

    def test_tile_size_below_2_raises(self) -> None:
        img = _make_image(64, 64)
        with pytest.raises(ValueError, match="tile_size must be in"):
            adaptive_equalize(img, tile_size=1, config=_CFG)

    def test_gradient_image_equalization(self) -> None:
        """Verify CLAHE produces valid output on a gradient image."""
        arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        img = Image.fromarray(arr)
        result = adaptive_equalize(img, clip_limit=2.0, tile_size=8, config=_CFG)
        result_arr = np.array(result)
        assert result_arr.shape == (256, 256)
        assert result_arr.min() >= 0
        assert result_arr.max() <= 255

    def test_non_square_image(self) -> None:
        """Verify CLAHE handles non-square images where dimensions aren't tile multiples."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(100, 200), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = adaptive_equalize(img, clip_limit=3.0, tile_size=6, config=_CFG)
        assert result.size == (200, 100)
        assert result.mode == "L"

    def test_larger_image_completes_quickly(self) -> None:
        """Vectorized CLAHE should handle 512x512 in well under 1 second."""
        import time

        rng = np.random.default_rng(99)
        arr = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
        img = Image.fromarray(arr)
        start = time.perf_counter()
        result = adaptive_equalize(img, clip_limit=2.0, tile_size=8, config=_CFG)
        elapsed = time.perf_counter() - start
        assert result.size == (512, 512)
        # Vectorized: ~5-20ms. Old loop: ~8s on 512x512. Use 1s as generous bound.
        assert elapsed < 1.0, f"adaptive_equalize took {elapsed:.2f}s on 512×512 (expected <1s)"

    def test_uniform_image_unchanged(self) -> None:
        """A uniform image should remain roughly uniform after CLAHE."""
        arr = np.full((64, 64), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        result = adaptive_equalize(img, clip_limit=2.0, tile_size=4, config=_CFG)
        result_arr = np.array(result)
        # All pixels start at same value, so CDF maps them all identically
        assert result_arr.std() < 1.0


class TestComputeIntensityProfile:
    def test_horizontal_profile(self) -> None:
        # Gradient image: left=0, right=255
        arr = np.zeros((100, 100), dtype=np.uint8)
        for x in range(100):
            arr[:, x] = int(x * 255 / 99)
        img = Image.fromarray(arr)
        result = compute_intensity_profile(img, (0.0, 0.5), (1.0, 0.5))
        profile = result["profile"]
        assert len(profile) > 0  # type: ignore[arg-type]
        # First value should be low, last should be high
        assert profile[0] < profile[-1]  # type: ignore[index]

    def test_returns_stats(self) -> None:
        img = _make_image()
        result = compute_intensity_profile(img, (0.0, 0.0), (1.0, 1.0))
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="coordinates must be in"):
            compute_intensity_profile(img, (-0.1, 0.5), (1.0, 0.5))


class TestDenoiseGaussian:
    def test_preserves_size(self) -> None:
        img = _make_image()
        result = denoise_gaussian(img, sigma=1.0, config=_CFG)
        assert result.size == img.size

    def test_reduces_noise(self) -> None:
        # Noisy image should become smoother
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = denoise_gaussian(img, sigma=2.0, config=_CFG)
        # Standard deviation should decrease after smoothing
        assert np.std(np.array(result)) < np.std(arr)

    def test_sigma_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="sigma must be in"):
            denoise_gaussian(img, sigma=0.1, config=_CFG)
        with pytest.raises(ValueError, match="sigma must be in"):
            denoise_gaussian(img, sigma=10.0, config=_CFG)


class TestMorphologicalOp:
    def test_erode_shrinks(self) -> None:
        # White square on black → erosion should shrink it
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[30:70, 30:70] = 255
        img = Image.fromarray(arr)
        result = morphological_op(img, "erode", iterations=1, config=_CFG)
        # White area should be smaller
        assert np.sum(np.array(result) > 128) < np.sum(arr > 128)

    def test_dilate_expands(self) -> None:
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[40:60, 40:60] = 255
        img = Image.fromarray(arr)
        result = morphological_op(img, "dilate", iterations=1, config=_CFG)
        assert np.sum(np.array(result) > 128) > np.sum(arr > 128)

    def test_open_and_close(self) -> None:
        img = _make_image()
        for op in ("open", "close"):
            result = morphological_op(img, op, config=_CFG)
            assert result.mode == "L"

    def test_with_threshold(self) -> None:
        img = _make_image()
        result = morphological_op(img, "erode", threshold_value=100, config=_CFG)
        assert result.mode == "L"

    def test_invalid_operation_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="operation must be one of"):
            morphological_op(img, "invalid", config=_CFG)

    def test_iterations_out_of_range_raises(self) -> None:
        img = _make_image()
        with pytest.raises(ValueError, match="iterations must be in"):
            morphological_op(img, "erode", iterations=0, config=_CFG)
        with pytest.raises(ValueError, match="iterations must be in"):
            morphological_op(img, "erode", iterations=100, config=_CFG)


# =====================================================================
# Tool executors via ToolRegistry
# =====================================================================


class TestToolExecutors:
    @pytest.mark.asyncio
    async def test_zoom_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("zoom", factor=2.0)
        assert result.success
        assert result.image_base64 is not None
        assert result.metadata["factor"] == 2.0

    @pytest.mark.asyncio
    async def test_crop_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("crop", box=[0.0, 0.0, 0.5, 0.5])
        assert result.success
        assert result.metadata["area_percentage"] == pytest.approx(25.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_contrast_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("adjust_contrast", factor=1.5)
        assert result.success

    @pytest.mark.asyncio
    async def test_threshold_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("threshold", lower=50, upper=200)
        assert result.success

    @pytest.mark.asyncio
    async def test_flip_horizontal_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("flip_horizontal")
        assert result.success

    @pytest.mark.asyncio
    async def test_flip_vertical_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("flip_vertical")
        assert result.success

    @pytest.mark.asyncio
    async def test_rotate_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("rotate", clockwise=True)
        assert result.success
        assert result.metadata["clockwise"] is True

    @pytest.mark.asyncio
    async def test_reset_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        # First modify, then reset
        await registry.execute("zoom", factor=2.0)
        result = await registry.execute("reset")
        assert result.success
        # After reset, size should match original
        assert result.metadata["size"] == (100, 100)

    @pytest.mark.asyncio
    async def test_brightness_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("adjust_brightness", factor=1.5)
        assert result.success
        assert result.image_base64 is not None
        assert result.metadata["factor"] == 1.5

    @pytest.mark.asyncio
    async def test_sharpness_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("adjust_sharpness", factor=2.0)
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.metadata["factor"] == 2.0

    @pytest.mark.asyncio
    async def test_equalize_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("equalize_histogram")
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.metadata.get("image_replaced") is True

    @pytest.mark.asyncio
    async def test_intensity_stats_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("get_intensity_stats")
        assert result.success
        assert result.image_base64 is None
        assert "mean" in result.metadata
        assert "histogram" in result.metadata

    @pytest.mark.asyncio
    async def test_intensity_stats_with_box(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("get_intensity_stats", box=[0.0, 0.0, 0.5, 0.5])
        assert result.success

    @pytest.mark.asyncio
    async def test_intensity_stats_invalid_box_raises(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="box requires"):
            await registry.execute("get_intensity_stats", box=[0.0, 0.0])

    @pytest.mark.asyncio
    async def test_measure_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("measure", point1=[0.0, 0.0], point2=[1.0, 1.0])
        assert result.success
        assert result.image_base64 is None
        assert "distance_pixels" in result.metadata

    @pytest.mark.asyncio
    async def test_measure_wrong_length_raises(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must have 2 values"):
            await registry.execute("measure", point1=[0.0], point2=[1.0, 1.0])

    @pytest.mark.asyncio
    async def test_show_grid_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("show_grid", divisions=4)
        assert result.success
        assert result.image_base64 is not None
        assert result.metadata["divisions"] == 4
        assert len(result.metadata["cell_labels"]) == 16

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_detect_edges_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("detect_edges", method="sobel")
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.metadata["method"] == "sobel"

    @pytest.mark.asyncio
    async def test_symmetry_diff_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("symmetry_diff")
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert "symmetry" in result.description.lower() or "diff" in result.description.lower()

    @pytest.mark.asyncio
    async def test_annotate_region_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute(
            "annotate_region", box=[0.1, 0.1, 0.9, 0.9], color="red", label="test"
        )
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert "annotated" in result.description.lower()

    @pytest.mark.asyncio
    async def test_invert_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("invert")
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert "invert" in result.description.lower()

    @pytest.mark.asyncio
    async def test_window_level_tool_preset(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("window_level", preset="brain")
        assert result.success
        assert result.metadata["preset"] == "brain"

    @pytest.mark.asyncio
    async def test_window_level_tool_custom(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("window_level", center=40, width=80)
        assert result.success

    @pytest.mark.asyncio
    async def test_adaptive_equalize_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("adaptive_equalize", clip_limit=2.0, tile_size=4)
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.metadata["clip_limit"] == 2.0
        assert result.metadata["tile_size"] == 4

    @pytest.mark.asyncio
    async def test_intensity_profile_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("intensity_profile", point1=[0.0, 0.5], point2=[1.0, 0.5])
        assert result.success
        assert result.image_base64 is None
        assert "profile" in result.metadata

    @pytest.mark.asyncio
    async def test_denoise_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("denoise", sigma=1.0)
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert "denoise" in result.description.lower() or "noise" in result.description.lower()

    @pytest.mark.asyncio
    async def test_morphological_tool(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("morphological", operation="erode", iterations=1)
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.metadata["operation"] == "erode"

    @pytest.mark.asyncio
    async def test_morphological_invalid_op_raises(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="Invalid morphological"):
            await registry.execute("morphological", operation="invalid")

    async def test_tool_without_image_raises(self) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
        with pytest.raises(ToolExecutionError, match="No image path set"):
            await registry.execute("zoom", factor=2.0)

    @pytest.mark.asyncio
    async def test_zoom_invalid_factor_raises(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="Invalid zoom factor"):
            await registry.execute("zoom", factor=99.0)

    @pytest.mark.asyncio
    async def test_crop_wrong_length_raises(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="requires \\[x1, y1, x2, y2\\]"):
            await registry.execute("crop", box=[0.0, 0.0, 1.0])

    @pytest.mark.asyncio
    async def test_crop_non_numeric_box_raises(self, tmp_path: Path) -> None:
        """Non-numeric box values must raise ToolExecutionError, not TypeError."""
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must be numbers"):
            await registry.execute("crop", box=["a", "b", "c", "d"])

    @pytest.mark.asyncio
    async def test_reset_without_path_raises(self) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
        with pytest.raises(ToolExecutionError):
            await registry.execute("reset")


# =====================================================================
# Factory
# =====================================================================


class TestCreateVisualTools:
    def test_default_creates_all_tools(self) -> None:
        tools = create_visual_tools()
        names = {t.name for t in tools}
        expected = {
            "zoom",
            "crop",
            "adjust_contrast",
            "adjust_brightness",
            "adjust_sharpness",
            "threshold",
            "window_level",
            "equalize_histogram",
            "adaptive_equalize",
            "detect_edges",
            "denoise",
            "morphological",
            "get_intensity_stats",
            "intensity_profile",
            "symmetry_diff",
            "invert",
            "annotate_region",
            "flip_horizontal",
            "flip_vertical",
            "rotate",
            "show_grid",
            "measure",
            "reset",
        }
        assert names == expected

    def test_disable_single_tool(self) -> None:
        tools = create_visual_tools(disabled_tools={"zoom"})
        names = {t.name for t in tools}
        assert "zoom" not in names
        assert len(tools) == 22

    def test_disable_multiple_tools(self) -> None:
        tools = create_visual_tools(disabled_tools={"zoom", "crop", "rotate"})
        names = {t.name for t in tools}
        assert "zoom" not in names
        assert "crop" not in names
        assert "rotate" not in names
        assert len(tools) == 20

    def test_disable_all_returns_empty(self) -> None:
        all_names = {
            "zoom",
            "crop",
            "adjust_contrast",
            "adjust_brightness",
            "adjust_sharpness",
            "threshold",
            "window_level",
            "equalize_histogram",
            "adaptive_equalize",
            "detect_edges",
            "denoise",
            "morphological",
            "get_intensity_stats",
            "intensity_profile",
            "symmetry_diff",
            "invert",
            "annotate_region",
            "flip_horizontal",
            "flip_vertical",
            "rotate",
            "show_grid",
            "measure",
            "reset",
        }
        tools = create_visual_tools(disabled_tools=all_names)
        assert tools == []

    def test_all_tools_have_category_visual(self) -> None:
        tools = create_visual_tools()
        for tool in tools:
            assert tool.category == "visual"

    def test_all_tools_require_image(self) -> None:
        tools = create_visual_tools()
        for tool in tools:
            assert tool.requires_image is True

    def test_schema_ranges_derived_from_config(self) -> None:
        """Schema min/max must reflect the config, not hardcoded values."""
        custom_cfg = ImageProcessingConfig(
            min_zoom_factor=1.0,
            max_zoom_factor=2.0,
            min_contrast_factor=0.8,
            max_contrast_factor=1.5,
        )
        tools = create_visual_tools(config=custom_cfg)
        tool_map = {t.name: t for t in tools}

        zoom_params = tool_map["zoom"].parameters["factor"]
        assert zoom_params["minimum"] == 1.0
        assert zoom_params["maximum"] == 2.0

        contrast_params = tool_map["adjust_contrast"].parameters["factor"]
        assert contrast_params["minimum"] == 0.8
        assert contrast_params["maximum"] == 1.5

    def test_schema_ranges_for_new_tools(self) -> None:
        """Brightness, sharpness, grid, denoise, CLAHE, morph schema ranges reflect config."""
        custom_cfg = ImageProcessingConfig(
            min_brightness_factor=0.8,
            max_brightness_factor=2.0,
            min_sharpness_factor=0.5,
            max_sharpness_factor=2.5,
            max_grid_divisions=6,
            min_gaussian_sigma=1.0,
            max_gaussian_sigma=3.0,
            max_morphological_iterations=3,
            min_clahe_clip_limit=2.0,
            max_clahe_clip_limit=8.0,
        )
        tools = create_visual_tools(config=custom_cfg)
        tool_map = {t.name: t for t in tools}

        brightness_params = tool_map["adjust_brightness"].parameters["factor"]
        assert brightness_params["minimum"] == 0.8
        assert brightness_params["maximum"] == 2.0

        sharpness_params = tool_map["adjust_sharpness"].parameters["factor"]
        assert sharpness_params["minimum"] == 0.5
        assert sharpness_params["maximum"] == 2.5

        grid_params = tool_map["show_grid"].parameters["divisions"]
        assert grid_params["minimum"] == 2
        assert grid_params["maximum"] == 6

        denoise_params = tool_map["denoise"].parameters["sigma"]
        assert denoise_params["minimum"] == 1.0
        assert denoise_params["maximum"] == 3.0

        morph_params = tool_map["morphological"].parameters["iterations"]
        assert morph_params["maximum"] == 3

        clahe_params = tool_map["adaptive_equalize"].parameters["clip_limit"]
        assert clahe_params["minimum"] == 2.0
        assert clahe_params["maximum"] == 8.0
