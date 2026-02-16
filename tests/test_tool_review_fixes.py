"""Tests for tool system review findings (Patch Set #1).

Each test is tagged with the finding number it validates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools.registry import ToolDocumenter
from radiant_harness.tools.visual import (
    adaptive_equalize,
    apply_window_level,
    create_visual_tools,
    WINDOW_PRESETS,
)


def _create_test_image(tmp_path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    path = tmp_path / "test.png"
    Image.new("L", size, color=128).save(path)
    return path


# ── Finding 1: CLAHE tile_size OOM prevention ────────────────────


class TestCLAHETileSizeBound:
    """Finding 1: tile_size must have an upper bound to prevent OOM."""

    def test_tile_size_exceeding_max_raises(self) -> None:
        """tile_size above max_clahe_tile_size must be rejected."""
        cfg = ImageProcessingConfig(max_clahe_tile_size=16)
        img = Image.new("L", (256, 256), color=128)
        with pytest.raises(ValueError, match="tile_size must be in"):
            adaptive_equalize(img, tile_size=32, config=cfg)

    def test_tile_size_at_max_succeeds(self) -> None:
        """tile_size exactly at max_clahe_tile_size must work."""
        cfg = ImageProcessingConfig(max_clahe_tile_size=16)
        img = Image.new("L", (256, 256), color=128)
        result = adaptive_equalize(img, tile_size=16, config=cfg)
        assert result.size == (256, 256)

    def test_default_tile_size_succeeds(self) -> None:
        """Default tile_size=8 must work with default config."""
        img = Image.new("L", (256, 256), color=128)
        result = adaptive_equalize(img, tile_size=8)
        assert result.size == (256, 256)

    def test_config_rejects_extreme_max_tile_size(self) -> None:
        """max_clahe_tile_size > 64 must be rejected by config validation."""
        with pytest.raises(ValueError, match="max_clahe_tile_size"):
            ImageProcessingConfig(max_clahe_tile_size=100)


# ── Finding 2: detect_edges enum constraint ──────────────────────


class TestDetectEdgesEnum:
    """Finding 2: detect_edges method must have enum in schema."""

    def test_schema_has_method_enum(self) -> None:
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        edges_schema = next(s for s in schemas if s["function"]["name"] == "detect_edges")
        method_prop = edges_schema["function"]["parameters"]["properties"]["method"]
        assert "enum" in method_prop, "detect_edges.method must have enum constraint"
        assert set(method_prop["enum"]) == {"sobel", "laplacian"}


# ── Finding 3: morphological enum constraint ─────────────────────


class TestMorphologicalEnum:
    """Finding 3: morphological operation must have enum in schema."""

    def test_schema_has_operation_enum(self) -> None:
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        morph_schema = next(s for s in schemas if s["function"]["name"] == "morphological")
        op_prop = morph_schema["function"]["parameters"]["properties"]["operation"]
        assert "enum" in op_prop, "morphological.operation must have enum constraint"
        assert set(op_prop["enum"]) == {"erode", "dilate", "open", "close"}


# ── Finding 4: window_level preset enum constraint ───────────────


class TestWindowLevelPresetEnum:
    """Finding 4: window_level preset must have enum in schema."""

    def test_schema_has_preset_enum(self) -> None:
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        wl_schema = next(s for s in schemas if s["function"]["name"] == "window_level")
        preset_prop = wl_schema["function"]["parameters"]["properties"]["preset"]
        assert "enum" in preset_prop, "window_level.preset must have enum constraint"
        assert set(preset_prop["enum"]) == set(WINDOW_PRESETS.keys())


# ── Finding 5: window_level width=1 produces all-black ───────────


class TestWindowLevelMinWidth:
    """Finding 5: width=1 silently destroys the image; must be rejected."""

    def test_width_1_rejected(self) -> None:
        """width=1 must raise ValueError, not silently produce all-black."""
        img = Image.new("L", (32, 32), color=128)
        with pytest.raises(ValueError, match="width must be >= "):
            apply_window_level(img, center=128, width=1)

    def test_width_2_succeeds(self) -> None:
        """width=2 (the new default minimum) must work."""
        img = Image.new("L", (32, 32), color=128)
        result = apply_window_level(img, center=128, width=2)
        assert result.size == (32, 32)

    def test_preset_stroke_still_works(self) -> None:
        """The 'stroke' preset has width=8, must still work."""
        img = Image.new("L", (32, 32), color=128)
        result = apply_window_level(img, preset="stroke")
        assert result.size == (32, 32)

    @pytest.mark.asyncio
    async def test_window_level_tool_width_1_error(self, tmp_path: Path) -> None:
        """End-to-end: window_level tool call with width=1 should fail gracefully."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="width must be"):
            await registry.execute("window_level", center=128, width=1)


# ── Finding 6: adaptive_equalize schema tile_size maximum ────────


class TestCLAHESchemaMaximum:
    """Finding 6: tile_size schema must include maximum from config."""

    def test_schema_has_tile_size_maximum(self) -> None:
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        clahe_schema = next(s for s in schemas if s["function"]["name"] == "adaptive_equalize")
        tile_prop = clahe_schema["function"]["parameters"]["properties"]["tile_size"]
        assert "maximum" in tile_prop, "tile_size must have maximum in schema"
        assert tile_prop["maximum"] == 32  # default max_clahe_tile_size

    @pytest.mark.asyncio
    async def test_tool_rejects_oversized_tile(self, tmp_path: Path) -> None:
        """End-to-end: adaptive_equalize with tile_size > max must fail."""
        image_path = _create_test_image(tmp_path)
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=image_path, tools=tools)
        with pytest.raises(ToolExecutionError, match="tile_size"):
            await registry.execute("adaptive_equalize", tile_size=64)


# ── Finding 7: annotate_region color enum ────────────────────────


class TestAnnotateRegionColorEnum:
    """Finding 7: annotate_region color should have enum constraint."""

    def test_schema_has_color_enum(self) -> None:
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        annotate_schema = next(s for s in schemas if s["function"]["name"] == "annotate_region")
        color_prop = annotate_schema["function"]["parameters"]["properties"]["color"]
        assert "enum" in color_prop, "annotate_region.color must have enum constraint"
        assert "red" in color_prop["enum"]
        assert "green" in color_prop["enum"]


# ── Cross-cutting: all enum schemas stay in sync with validators ─


class TestEnumSchemaSync:
    """Verify all enum-constrained string params match their runtime validators."""

    def test_all_string_params_with_known_values_have_enums(self) -> None:
        """Every string param whose runtime validator uses a fixed set
        must have a matching enum in the schema."""
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()

        # These are the params we know must have enums
        expected_enums = {
            ("detect_edges", "method"): {"sobel", "laplacian"},
            ("morphological", "operation"): {"erode", "dilate", "open", "close"},
            ("window_level", "preset"): set(WINDOW_PRESETS.keys()),
        }

        for schema in schemas:
            fn = schema["function"]
            name = fn["name"]
            for pname, pdef in fn["parameters"]["properties"].items():
                key = (name, pname)
                if key in expected_enums:
                    assert "enum" in pdef, f"{name}.{pname} missing enum"
                    assert set(pdef["enum"]) == expected_enums[key], (
                        f"{name}.{pname} enum mismatch: "
                        f"schema={set(pdef['enum'])} vs expected={expected_enums[key]}"
                    )
