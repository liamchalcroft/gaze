"""Tests for visual tools: image operations, tool executors, and factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools.registry import ToolRegistry
from radiant_harness.tools.visual import adjust_contrast
from radiant_harness.tools.visual import apply_intensity_threshold
from radiant_harness.tools.visual import create_visual_tools
from radiant_harness.tools.visual import crop_image
from radiant_harness.tools.visual import flip_horizontal
from radiant_harness.tools.visual import flip_vertical
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
    async def test_reset_without_path_raises(self) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
        with pytest.raises(ToolExecutionError):
            await registry.execute("reset")


# =====================================================================
# Factory
# =====================================================================


class TestCreateVisualTools:
    def test_default_creates_all_eight(self) -> None:
        tools = create_visual_tools()
        names = {t.name for t in tools}
        expected = {
            "zoom",
            "crop",
            "adjust_contrast",
            "threshold",
            "flip_horizontal",
            "flip_vertical",
            "rotate",
            "reset",
        }
        assert names == expected

    def test_disable_single_tool(self) -> None:
        tools = create_visual_tools(disabled_tools={"zoom"})
        names = {t.name for t in tools}
        assert "zoom" not in names
        assert len(tools) == 7

    def test_disable_multiple_tools(self) -> None:
        tools = create_visual_tools(disabled_tools={"zoom", "crop", "rotate"})
        names = {t.name for t in tools}
        assert names == {
            "adjust_contrast",
            "threshold",
            "flip_horizontal",
            "flip_vertical",
            "reset",
        }

    def test_disable_all_returns_empty(self) -> None:
        all_names = {
            "zoom",
            "crop",
            "adjust_contrast",
            "threshold",
            "flip_horizontal",
            "flip_vertical",
            "rotate",
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
