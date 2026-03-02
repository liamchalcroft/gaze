"""Tests for tool system audit findings.

Covers:
- Finding 1: _execute_intensity_stats string box handling
- Finding 2: _maybe_normalize_box borderline value clamping
- Finding 3: _maybe_normalize_box / _maybe_normalize_point test coverage
- Finding 4: window_level prompt documentation clarity
- Finding 5: Visual tool boundary value tests
- Finding 6: window_level preset override warning
- Finding 7: window_level metadata reports actual values
- Finding 9: annotate_region box type validation
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools.visual import _PIXEL_COORD_THRESHOLD
from radiant_harness.tools.visual import WINDOW_PRESETS
from radiant_harness.tools.visual import _maybe_normalize_box
from radiant_harness.tools.visual import _maybe_normalize_point
from radiant_harness.tools.visual import adjust_brightness
from radiant_harness.tools.visual import adjust_contrast
from radiant_harness.tools.visual import adjust_sharpness
from radiant_harness.tools.visual import apply_intensity_threshold
from radiant_harness.tools.visual import apply_window_level
from radiant_harness.tools.visual import create_visual_tools
from radiant_harness.tools.visual import crop_image
from radiant_harness.tools.visual import zoom_image


def _create_test_image(tmp_path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    path = tmp_path / "test.png"
    Image.new("L", size, color=128).save(path)
    return path


# ── Finding 1: _execute_intensity_stats string box handling ───────


class TestIntensityStatsStringBox:
    """String box must be rejected (not silently converted to None)."""

    @pytest.mark.asyncio
    async def test_string_null_becomes_none(self, tmp_path: Path) -> None:
        """The literal string 'null' should be treated as None (full image)."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("get_intensity_stats", box="null")
        assert result.success
        assert "for region" not in result.description

    @pytest.mark.asyncio
    async def test_string_none_becomes_none(self, tmp_path: Path) -> None:
        """The literal string 'none' should be treated as None."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("get_intensity_stats", box="none")
        assert result.success

    @pytest.mark.asyncio
    async def test_empty_string_becomes_none(self, tmp_path: Path) -> None:
        """Empty string should be treated as None."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("get_intensity_stats", box="")
        assert result.success

    @pytest.mark.asyncio
    async def test_json_string_box_raises(self, tmp_path: Path) -> None:
        """A JSON-like string box must raise ToolExecutionError, not silently
        return full-image stats."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must be an array"):
            await registry.execute("get_intensity_stats", box="[0.1,0.2,0.9,0.8]")

    @pytest.mark.asyncio
    async def test_arbitrary_string_box_raises(self, tmp_path: Path) -> None:
        """Arbitrary strings must raise, not be silently ignored."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must be an array"):
            await registry.execute("get_intensity_stats", box="center")


# ── Finding 2: _maybe_normalize_box borderline clamping ──────────


class TestMaybeNormalizeBoxClamping:
    """Values slightly above 1.0 should be clamped, not pixel-normalized."""

    def test_borderline_1_001_clamped_not_normalized(self) -> None:
        """[0.5, 0.5, 1.001, 1.001] should clamp to [0.5, 0.5, 1.0, 1.0]."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_box([0.5, 0.5, 1.001, 1.001], img)
        assert result == [0.5, 0.5, 1.0, 1.0]

    def test_borderline_1_5_clamped(self) -> None:
        """Values up to (but below) threshold should be clamped."""
        img = Image.new("L", (100, 100))
        result = _maybe_normalize_box([0.0, 0.0, 1.5, 1.5], img)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_threshold_constant_is_2(self) -> None:
        """Verify the threshold constant value."""
        assert _PIXEL_COORD_THRESHOLD == 2.0

    def test_at_threshold_triggers_normalization(self) -> None:
        """Values at threshold (2.0) should trigger pixel normalization."""
        img = Image.new("L", (100, 100))
        result = _maybe_normalize_box([0.0, 0.0, 2.0, 2.0], img)
        # 2.0/100 = 0.02, so this is now pixel-normalized
        assert result == [0.0, 0.0, 0.02, 0.02]

    def test_negative_values_clamped_to_zero(self) -> None:
        """Negative values in the borderline range should clamp to 0."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_box([-0.1, 0.0, 1.1, 1.0], img)
        # max is 1.1 < 2.0, so clamped
        assert result[0] == 0.0
        assert result[2] == 1.0


# ── Finding 3: _maybe_normalize_box/_point comprehensive tests ───


class TestMaybeNormalizeBox:
    """Comprehensive tests for the auto-normalization heuristic."""

    def test_valid_normalized_passes_through(self) -> None:
        """Coordinates already in [0, 1] pass through unchanged."""
        img = Image.new("L", (256, 256))
        box = [0.1, 0.2, 0.8, 0.9]
        result = _maybe_normalize_box(box, img)
        assert result == box

    def test_zero_zero_one_one_passes_through(self) -> None:
        """Full image box [0, 0, 1, 1] passes through."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_box([0.0, 0.0, 1.0, 1.0], img)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_pixel_coords_normalized(self) -> None:
        """Large pixel coords are divided by image dimensions."""
        img = Image.new("L", (200, 100))
        result = _maybe_normalize_box([20.0, 10.0, 180.0, 90.0], img)
        assert result == pytest.approx([0.1, 0.1, 0.9, 0.9])

    def test_pixel_coords_with_zero_origin(self) -> None:
        """Pixel coords starting at (0, 0) are still detected."""
        img = Image.new("L", (100, 100))
        result = _maybe_normalize_box([0.0, 0.0, 50.0, 50.0], img)
        assert result == [0.0, 0.0, 0.5, 0.5]

    def test_all_zeros_passes_through(self) -> None:
        """[0, 0, 0, 0] passes through (will fail validation later)."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_box([0.0, 0.0, 0.0, 0.0], img)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_exactly_one_passes_through(self) -> None:
        """Exactly 1.0 in coords doesn't trigger normalization."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_box([0.0, 0.0, 1.0, 0.5], img)
        assert result == [0.0, 0.0, 1.0, 0.5]


class TestMaybeNormalizePoint:
    """Tests for _maybe_normalize_point auto-normalization."""

    def test_valid_normalized_passes_through(self) -> None:
        img = Image.new("L", (100, 100))
        result = _maybe_normalize_point([0.5, 0.7], img)
        assert result == [0.5, 0.7]

    def test_pixel_coords_normalized(self) -> None:
        img = Image.new("L", (200, 100))
        result = _maybe_normalize_point([100.0, 50.0], img)
        assert result == [0.5, 0.5]

    def test_borderline_clamped(self) -> None:
        """Point like [0.5, 1.001] should clamp, not normalize."""
        img = Image.new("L", (64, 64))
        result = _maybe_normalize_point([0.5, 1.001], img)
        assert result == [0.5, 1.0]

    def test_at_threshold_normalizes(self) -> None:
        """Point at threshold [0.5, 2.0] should pixel-normalize."""
        img = Image.new("L", (100, 100))
        result = _maybe_normalize_point([0.5, 2.0], img)
        assert result == [0.005, 0.02]


# ── Finding 4: window_level prompt doc ───────────────────────────


class TestWindowLevelPromptDoc:
    """Window level tool should clearly document conditional requirements."""

    def test_prompt_doc_states_either_or(self) -> None:
        """Prompt documentation must mention 'EITHER preset OR center+width'."""
        tools = create_visual_tools()
        wl_tool = next(t for t in tools if t.name == "window_level")
        doc = wl_tool.get_prompt_documentation()
        assert "Must provide EITHER" in doc or "must provide" in doc.lower()
        assert "preset" in doc
        assert "center" in doc
        assert "width" in doc


# ── Finding 5: Visual tool boundary value tests ──────────────────


class TestZoomBoundaries:
    """Zoom at exact min/max factor boundaries."""

    def test_zoom_at_min_factor(self) -> None:
        img = Image.new("L", (64, 64))
        result = zoom_image(img, 0.5)  # min_zoom_factor default
        assert result.size == (32, 32)

    def test_zoom_at_max_factor(self) -> None:
        img = Image.new("L", (64, 64))
        result = zoom_image(img, 4.0)  # max_zoom_factor default
        assert result.size == (256, 256)

    def test_zoom_below_min_raises(self) -> None:
        img = Image.new("L", (64, 64))
        with pytest.raises(ValueError, match="factor must be in range"):
            zoom_image(img, 0.49)

    def test_zoom_above_max_raises(self) -> None:
        img = Image.new("L", (64, 64))
        with pytest.raises(ValueError, match="factor must be in range"):
            zoom_image(img, 4.01)


class TestContrastBoundaries:
    """Contrast at exact min/max factor boundaries."""

    def test_contrast_at_min_factor(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_contrast(img, 0.5)
        assert result.size == (32, 32)

    def test_contrast_at_max_factor(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_contrast(img, 3.0)
        assert result.size == (32, 32)

    def test_contrast_below_min_raises(self) -> None:
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_contrast(img, 0.49)

    def test_contrast_above_max_raises(self) -> None:
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="factor must be in range"):
            adjust_contrast(img, 3.01)


class TestBrightnessBoundaries:
    """Brightness at exact min/max factor boundaries."""

    def test_brightness_at_min(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_brightness(img, 0.5)
        assert result.size == (32, 32)

    def test_brightness_at_max(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_brightness(img, 3.0)
        assert result.size == (32, 32)


class TestSharpnessBoundaries:
    """Sharpness at exact min/max factor boundaries."""

    def test_sharpness_at_min(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_sharpness(img, 0.0)  # full blur
        assert result.size == (32, 32)

    def test_sharpness_at_max(self) -> None:
        img = Image.new("L", (32, 32), color=128)
        result = adjust_sharpness(img, 3.0)
        assert result.size == (32, 32)


class TestCropBoundaries:
    """Crop at exact edge coordinates."""

    def test_crop_full_image(self) -> None:
        """Crop with [0, 0, 1, 1] should return the full image."""
        img = Image.new("L", (64, 64), color=128)
        result = crop_image(img, (0.0, 0.0, 1.0, 1.0))
        assert result.size == (64, 64)

    def test_crop_exactly_at_min_size(self) -> None:
        """Crop resulting in exactly min_image_size should succeed."""
        cfg = ImageProcessingConfig(min_image_size=10)
        img = Image.new("L", (100, 100), color=128)
        # 10/100 = 0.1 width → exactly 10 pixels
        result = crop_image(img, (0.0, 0.0, 0.1, 0.1), config=cfg)
        assert result.size[0] >= 10
        assert result.size[1] >= 10

    def test_crop_below_min_size_raises(self) -> None:
        """Crop below min_image_size should raise."""
        cfg = ImageProcessingConfig(min_image_size=10)
        img = Image.new("L", (100, 100))
        with pytest.raises(ValueError, match="too small"):
            crop_image(img, (0.0, 0.0, 0.05, 0.05), config=cfg)  # 5x5


class TestThresholdBoundaries:
    """Threshold at exact min_threshold_window boundary."""

    def test_threshold_at_exact_min_window(self) -> None:
        """Window width exactly at min_threshold_window should succeed."""
        img = Image.new("L", (32, 32), color=128)
        # Default min_threshold_window = 50, so width = upper - lower = 50
        result = apply_intensity_threshold(img, lower=100, upper=150)
        assert result.size == (32, 32)

    def test_threshold_below_min_window_raises(self) -> None:
        """Window width below min_threshold_window should raise."""
        img = Image.new("L", (32, 32), color=128)
        with pytest.raises(ValueError, match="below minimum"):
            apply_intensity_threshold(img, lower=100, upper=149)  # width=49 < 50

    def test_threshold_extreme_valid_bounds(self) -> None:
        """lower=0, upper=255 (full range) should succeed."""
        img = Image.new("L", (32, 32), color=128)
        result = apply_intensity_threshold(img, lower=0, upper=255)
        assert result.size == (32, 32)

    def test_threshold_lower_negative_raises(self) -> None:
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="lower must be >= 0"):
            apply_intensity_threshold(img, lower=-1, upper=255)

    def test_threshold_upper_256_raises(self) -> None:
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="upper must be <= 255"):
            apply_intensity_threshold(img, lower=0, upper=256)

    def test_threshold_equal_bounds_raises(self) -> None:
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="upper must be > lower"):
            apply_intensity_threshold(img, lower=128, upper=128)


# ── End-to-end: crop with pixel coords via registry ──────────────


class TestCropPixelCoordsE2E:
    """End-to-end test: model sends pixel coords, auto-normalization kicks in."""

    @pytest.mark.asyncio
    async def test_pixel_coords_crop(self, tmp_path: Path) -> None:
        """Pixel coordinates should be auto-normalized and produce correct crop."""
        path = tmp_path / "test.png"
        Image.new("L", (200, 100), color=128).save(path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=path, tools=tools)

        # Model sends pixel coords [20, 10, 180, 90]
        result = await registry.execute("crop", box=[20.0, 10.0, 180.0, 90.0])
        assert result.success
        # Normalized to [0.1, 0.1, 0.9, 0.9] on a 200x100 image
        # Crop size: 0.8 * 200 = 160, 0.8 * 100 = 80
        assert result.metadata["new_size"] == (160, 80)


# ── Finding 6: window_level preset override warning ──────────────


class TestWindowLevelPresetOverride:
    """When preset AND center/width are given, preset wins with a warning."""

    def test_preset_overrides_center_width(self) -> None:
        """apply_window_level with preset+center+width uses preset values."""
        img = Image.new("L", (64, 64), color=128)
        # "brain" preset is (40, 80).  Passing center=100, width=200 should be ignored.
        result = apply_window_level(img, center=100, width=200, preset="brain")
        assert result.size == (64, 64)
        # Verify it actually applied brain preset values, not center=100/width=200.
        # With brain (center=40, width=80): lower=0, upper=80.
        # pixel 128 > upper 80 => clamped to 80 => (80-0)/(80-0)*255 = 255
        import numpy as np

        arr = np.array(result)
        assert arr[0, 0] == 255  # 128 is above brain window upper bound

    def test_preset_only_no_warning(self) -> None:
        """apply_window_level with only preset should not produce issues."""
        img = Image.new("L", (32, 32), color=40)
        result = apply_window_level(img, preset="brain")
        assert result.size == (32, 32)

    def test_no_args_raises(self) -> None:
        """No preset and no center/width must raise ValueError."""
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="Must provide either preset or both center and width"):
            apply_window_level(img)

    def test_center_only_raises(self) -> None:
        """center without width must raise ValueError."""
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="Must provide either preset or both center and width"):
            apply_window_level(img, center=40)

    def test_width_only_raises(self) -> None:
        """width without center must raise ValueError."""
        img = Image.new("L", (32, 32))
        with pytest.raises(ValueError, match="Must provide either preset or both center and width"):
            apply_window_level(img, width=80)


# ── Finding 7: window_level metadata reports actual values ───────


class TestWindowLevelMetadata:
    """Result metadata should report actual center/width values, not kwargs."""

    @pytest.mark.asyncio
    async def test_preset_metadata_has_actual_values(self, tmp_path: Path) -> None:
        """When preset is used, metadata center/width must be the preset values."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)

        result = await registry.execute("window_level", preset="brain")
        assert result.success
        expected_center, expected_width = WINDOW_PRESETS["brain"]
        assert result.metadata["center"] == expected_center
        assert result.metadata["width"] == expected_width
        assert result.metadata["preset"] == "brain"

    @pytest.mark.asyncio
    async def test_preset_override_metadata(self, tmp_path: Path) -> None:
        """When preset overrides kwargs, metadata should reflect preset values."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)

        # Pass center=999, width=999 but preset="subdural"
        result = await registry.execute("window_level", center=999, width=999, preset="subdural")
        assert result.success
        expected_center, expected_width = WINDOW_PRESETS["subdural"]
        assert result.metadata["center"] == expected_center
        assert result.metadata["width"] == expected_width
        # NOT 999

    @pytest.mark.asyncio
    async def test_custom_center_width_metadata(self, tmp_path: Path) -> None:
        """When using custom center/width (no preset), metadata matches kwargs."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)

        result = await registry.execute("window_level", center=40, width=80)
        assert result.success
        assert result.metadata["center"] == 40
        assert result.metadata["width"] == 80
        assert result.metadata["preset"] is None

    @pytest.mark.asyncio
    async def test_preset_description_includes_values(self, tmp_path: Path) -> None:
        """Description for preset should include the actual center/width."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)

        result = await registry.execute("window_level", preset="bone")
        assert "bone" in result.description
        expected_center, expected_width = WINDOW_PRESETS["bone"]
        assert str(expected_center) in result.description
        assert str(expected_width) in result.description


# ── Finding 9: annotate_region box type validation ───────────────


class TestAnnotateRegionBoxValidation:
    """annotate_region must validate box element types like crop does."""

    @pytest.mark.asyncio
    async def test_valid_box(self, tmp_path: Path) -> None:
        """Normal float box should work."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        result = await registry.execute("annotate_region", box=[0.1, 0.1, 0.9, 0.9])
        assert result.success

    @pytest.mark.asyncio
    async def test_bool_in_box_raises(self, tmp_path: Path) -> None:
        """Boolean values in box must be rejected."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must be numbers"):
            await registry.execute("annotate_region", box=[True, 0.1, 0.9, 0.9])

    @pytest.mark.asyncio
    async def test_string_in_box_raises(self, tmp_path: Path) -> None:
        """String values in box must be rejected."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="must be numbers"):
            await registry.execute("annotate_region", box=["0.1", 0.1, 0.9, 0.9])

    @pytest.mark.asyncio
    async def test_wrong_length_raises(self, tmp_path: Path) -> None:
        """Box with wrong number of elements must be rejected."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="got 3 values"):
            await registry.execute("annotate_region", box=[0.1, 0.1, 0.9])

    @pytest.mark.asyncio
    async def test_pixel_coords_auto_normalize(self, tmp_path: Path) -> None:
        """Pixel coords should be auto-normalized for annotate_region too."""
        path = tmp_path / "test.png"
        Image.new("L", (200, 100), color=128).save(path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=path, tools=tools)
        result = await registry.execute("annotate_region", box=[20.0, 10.0, 180.0, 90.0])
        assert result.success
        # Verify normalized box values in metadata
        box = result.metadata["box"]
        assert pytest.approx(box[0], abs=0.01) == 0.1
        assert pytest.approx(box[1], abs=0.01) == 0.1
        assert pytest.approx(box[2], abs=0.01) == 0.9
        assert pytest.approx(box[3], abs=0.01) == 0.9
