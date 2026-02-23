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

    def test_width_10_succeeds(self) -> None:
        """width=10 (the default minimum) must work."""
        img = Image.new("L", (32, 32), color=128)
        result = apply_window_level(img, center=128, width=10)
        assert result.size == (32, 32)

    def test_width_below_minimum_fails(self) -> None:
        """width below min_window_width=10 must be rejected."""
        img = Image.new("L", (32, 32), color=128)
        with pytest.raises(ValueError, match="width must be >= "):
            apply_window_level(img, center=128, width=2)

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


# ── Patch Set #2: window_level float division ────────────────────


class TestWindowLevelFloatDivision:
    """Verify apply_window_level uses float (not integer) division for bounds."""

    def test_odd_width_not_lossy(self) -> None:
        """With center=128, width=11, effective window should be exactly 11, not 10."""
        import numpy as np

        img = Image.new("L", (32, 32), color=128)
        result = apply_window_level(img, center=128, width=11)
        arr = np.array(result)
        # center=128, width=11 → lower=122.5, upper=133.5
        # pixel=128 is within [122.5, 133.5] and should map to ~128
        # With old integer division: lower=123, upper=133 (window=10, asymmetric)
        # With float division: lower=122.5, upper=133.5 (window=11, symmetric)
        assert arr.max() > 0, "Odd-width window should not produce all-black"
        # The center pixel (128) should map to approximately the midpoint
        expected_mid = int((128 - 122.5) / 11 * 255)
        assert abs(int(arr[0, 0]) - expected_mid) <= 1

    def test_symmetric_windowing(self) -> None:
        """Window bounds should be symmetric around center for any width."""
        import numpy as np

        # Pixel at center-5 and center+5 should map to symmetric output
        img_data = np.full((1, 11), 0, dtype=np.uint8)
        for i in range(11):
            img_data[0, i] = 95 + i  # 95..105
        img = Image.fromarray(img_data, mode="L")
        result = apply_window_level(img, center=100, width=11)
        arr = np.array(result, dtype=np.float64)
        # lower=94.5, upper=105.5 → all pixels within range
        # pixel 95 → (95-94.5)/11*255 ≈ 11.6
        # pixel 105 → (105-94.5)/11*255 ≈ 243.4
        # Values at equal distance from center should be symmetric
        mid = 5  # center pixel (value=100)
        for offset in range(1, 5):
            lo_val = arr[0, mid - offset]
            hi_val = arr[0, mid + offset]
            assert abs(lo_val + hi_val - 255) <= 2, (
                f"offset={offset}: lo={lo_val}, hi={hi_val}, sum={lo_val + hi_val} != ~255"
            )


# ── Patch Set #3: config min_window_width >= 2 ──────────────────


class TestMinWindowWidthConfig:
    """Verify config rejects min_window_width < 2."""

    def test_min_window_width_1_rejected(self) -> None:
        """min_window_width=1 would produce degenerate output; config must reject it."""
        with pytest.raises(ValueError, match="min_window_width must be >= 2"):
            ImageProcessingConfig(min_window_width=1)

    def test_min_window_width_0_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_window_width must be >= 2"):
            ImageProcessingConfig(min_window_width=0)

    def test_min_window_width_2_accepted(self) -> None:
        cfg = ImageProcessingConfig(min_window_width=2)
        assert cfg.min_window_width == 2


# ── Patch Set #4: schema "default": null not emitted ─────────────


class TestSchemaNoNullDefault:
    """Verify that 'default': None parameters don't emit 'default': null in schemas."""

    def test_no_null_defaults_in_any_schema(self) -> None:
        """No generated schema property should contain 'default': null."""
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()

        violations: list[str] = []
        for schema in schemas:
            fn = schema["function"]
            name = fn["name"]
            for pname, pdef in fn["parameters"]["properties"].items():
                if "default" in pdef and pdef["default"] is None:
                    violations.append(f"{name}.{pname}")

        assert not violations, (
            f"Schema properties with 'default': null (invalid for typed fields): {violations}"
        )

    def test_window_level_center_not_required(self) -> None:
        """center param should be optional (not in required) but without null default."""
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        wl = next(s for s in schemas if s["function"]["name"] == "window_level")
        params = wl["function"]["parameters"]
        assert "center" not in params["required"]
        assert "default" not in params["properties"]["center"]

    def test_morphological_threshold_value_not_required(self) -> None:
        """threshold_value param should be optional without null default."""
        tools = create_visual_tools()
        doc = ToolDocumenter(tools)
        schemas = doc.get_tool_schemas()
        morph = next(s for s in schemas if s["function"]["name"] == "morphological")
        params = morph["function"]["parameters"]
        assert "threshold_value" not in params["required"]
        assert "default" not in params["properties"]["threshold_value"]
