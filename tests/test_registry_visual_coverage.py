"""Tests targeting uncovered lines in tools/registry.py and tools/visual.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_visual_tools
from radiant_harness.tools.registry import ToolDocumenter
from radiant_harness.types import ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_execute(registry: ToolRegistry, **kwargs: Any) -> ToolResult:
    return ToolResult(tool_name="noop", description="noop", metadata=kwargs)


def _save_image(tmp_path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    p = tmp_path / "test.png"
    Image.new("RGB", size, "gray").save(p)
    return p


# ---------------------------------------------------------------------------
# ToolDocumenter.get_tool_schemas validation  (lines 194, 198)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolDocumenterValidation:
    def test_missing_param_type_raises(self) -> None:
        doc = ToolDocumenter()
        tool = Tool(
            name="bad",
            description="bad tool",
            parameters={"x": {"description": "no type"}},
            execute=_noop_execute,
            requires_image=False,
        )
        doc.register(tool)
        with pytest.raises(ValueError, match="missing required 'type'"):
            doc.get_tool_schemas()

    def test_invalid_param_type_raises(self) -> None:
        doc = ToolDocumenter()
        tool = Tool(
            name="bad",
            description="bad tool",
            parameters={"x": {"type": "invalid_type", "description": "wrong"}},
            execute=_noop_execute,
            requires_image=False,
        )
        doc.register(tool)
        with pytest.raises(ValueError, match="invalid type"):
            doc.get_tool_schemas()

    def test_default_none_not_emitted(self) -> None:
        """Parameter with default=None: 'default' omitted from schema, not in required."""
        doc = ToolDocumenter()
        tool = Tool(
            name="opt",
            description="optional param",
            parameters={"x": {"type": "integer", "description": "optional", "default": None}},
            execute=_noop_execute,
            requires_image=False,
        )
        doc.register(tool)
        schemas = doc.get_tool_schemas()
        assert len(schemas) == 1
        params = schemas[0]["function"]["parameters"]
        # "x" should NOT be in required (has a default)
        assert "x" not in params["required"]
        # "default" should NOT appear in the property (None is skipped)
        assert "default" not in params["properties"]["x"]


# ---------------------------------------------------------------------------
# ToolDocumenter.generate_prompt_documentation  (lines 304-338)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolDocumenterFiltering:
    def _make_doc_with_categories(self) -> ToolDocumenter:
        doc = ToolDocumenter()
        for name, cat in [("zoom", "visual"), ("crop", "visual"), ("search_web", "search")]:
            doc.register(
                Tool(
                    name=name,
                    description=f"{name} tool",
                    parameters={},
                    execute=_noop_execute,
                    requires_image=False,
                    category=cat,
                )
            )
        return doc

    def test_include_categories_grouped(self) -> None:
        doc = self._make_doc_with_categories()
        result = doc.generate_prompt_documentation(
            group_by_category=True, include_categories={"visual"}
        )
        assert "zoom" in result
        assert "crop" in result
        assert "search_web" not in result

    def test_exclude_categories_grouped(self) -> None:
        doc = self._make_doc_with_categories()
        result = doc.generate_prompt_documentation(
            group_by_category=True, exclude_categories={"search"}
        )
        assert "zoom" in result
        assert "search_web" not in result

    def test_include_categories_flat(self) -> None:
        doc = self._make_doc_with_categories()
        result = doc.generate_prompt_documentation(
            group_by_category=False, include_categories={"search"}
        )
        assert "search_web" in result
        assert "zoom" not in result

    def test_exclude_categories_flat(self) -> None:
        doc = self._make_doc_with_categories()
        result = doc.generate_prompt_documentation(
            group_by_category=False, exclude_categories={"visual"}
        )
        assert "search_web" in result
        assert "zoom" not in result

    def test_compact_mode(self) -> None:
        doc = self._make_doc_with_categories()
        full = doc.generate_prompt_documentation(compact=False)
        compact = doc.generate_prompt_documentation(compact=True)
        assert len(compact) < len(full)
        assert "zoom" in compact


# ---------------------------------------------------------------------------
# ToolRegistry: lazy managers  (lines 425-427, 433-435)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryLazyManagers:
    def test_get_web_search_manager(self) -> None:
        registry = ToolRegistry(image_path=None, tools=[])
        mgr1 = registry.get_web_search_manager()
        mgr2 = registry.get_web_search_manager()
        assert mgr1 is mgr2  # cached
        from radiant_harness.retrieval.web_search import WebSearchManager

        assert isinstance(mgr1, WebSearchManager)

    def test_get_image_search_manager(self) -> None:
        registry = ToolRegistry(image_path=None, tools=[])
        mgr1 = registry.get_image_search_manager()
        mgr2 = registry.get_image_search_manager()
        assert mgr1 is mgr2
        from radiant_harness.retrieval.image_search import MedicalImageSearchManager

        assert isinstance(mgr1, MedicalImageSearchManager)


# ---------------------------------------------------------------------------
# ToolRegistry.execute: array element coercion  (lines 496-497)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_array_element_coercion_float_to_int() -> None:
    """Array items of type 'integer' coerce float → int."""
    tool = Tool(
        name="ints",
        description="takes ints",
        parameters={
            "values": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "int list",
            }
        },
        execute=_noop_execute,
        requires_image=False,
    )
    registry = ToolRegistry(image_path=None, tools=[tool])
    result = await registry.execute("ints", values=[1.0, 2.0, 3.0])
    assert result.success
    assert list(result.metadata["values"]) == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_array_element_coercion_int_to_float() -> None:
    """Array items of type 'number' coerce int → float."""
    tool = Tool(
        name="floats",
        description="takes floats",
        parameters={
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "float list",
            }
        },
        execute=_noop_execute,
        requires_image=False,
    )
    registry = ToolRegistry(image_path=None, tools=[tool])
    result = await registry.execute("floats", values=[1, 2, 3])
    assert result.success
    assert list(result.metadata["values"]) == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in result.metadata["values"])


# ---------------------------------------------------------------------------
# Visual tool error wrapping  (uncovered ValueError → ToolExecutionError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
class TestVisualToolErrors:
    async def test_crop_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch("radiant_harness.tools.visual.crop_image", side_effect=ValueError("bad crop")),
            pytest.raises(ToolExecutionError, match="Invalid crop region"),
        ):
            await registry.execute("crop", box=[0.1, 0.1, 0.9, 0.9])

    async def test_contrast_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch("radiant_harness.tools.visual.adjust_contrast", side_effect=ValueError("bad")),
            pytest.raises(ToolExecutionError, match="Invalid contrast factor"),
        ):
            await registry.execute("adjust_contrast", factor=1.5)

    async def test_threshold_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.apply_intensity_threshold",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid threshold"),
        ):
            await registry.execute("threshold", lower=0, upper=255)

    async def test_brightness_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch("radiant_harness.tools.visual.adjust_brightness", side_effect=ValueError("bad")),
            pytest.raises(ToolExecutionError, match="Invalid brightness factor"),
        ):
            await registry.execute("adjust_brightness", factor=1.5)

    async def test_sharpness_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch("radiant_harness.tools.visual.adjust_sharpness", side_effect=ValueError("bad")),
            pytest.raises(ToolExecutionError, match="Invalid sharpness factor"),
        ):
            await registry.execute("adjust_sharpness", factor=1.5)

    async def test_intensity_stats_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.get_intensity_stats",
                side_effect=ValueError("bad box"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid box"),
        ):
            await registry.execute("get_intensity_stats", box=[0.1, 0.1, 0.9, 0.9])

    async def test_measure_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.measure_distance", side_effect=ValueError("bad pts")
            ),
            pytest.raises(ToolExecutionError, match="Invalid measurement"),
        ):
            await registry.execute("measure", point1=[0.1, 0.1], point2=[0.9, 0.9])

    async def test_show_grid_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.draw_grid_overlay",
                side_effect=ValueError("bad grid"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid grid"),
        ):
            await registry.execute("show_grid", divisions=4)

    async def test_detect_edges_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.detect_edges", side_effect=ValueError("bad method")
            ),
            pytest.raises(ToolExecutionError, match="Invalid edge detection"),
        ):
            await registry.execute("detect_edges", method="sobel")

    async def test_annotate_region_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.annotate_region",
                side_effect=ValueError("bad annot"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid annotation"),
        ):
            await registry.execute("annotate_region", box=[0.1, 0.1, 0.9, 0.9], color="red")

    async def test_intensity_profile_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.compute_intensity_profile",
                side_effect=ValueError("bad pts"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid profile"),
        ):
            await registry.execute("intensity_profile", point1=[0.1, 0.1], point2=[0.9, 0.9])

    async def test_denoise_value_error(self, tmp_path: Path) -> None:
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=_save_image(tmp_path), tools=tools)
        with (
            patch(
                "radiant_harness.tools.visual.denoise_gaussian",
                side_effect=ValueError("bad sigma"),
            ),
            pytest.raises(ToolExecutionError, match="Invalid denoise"),
        ):
            await registry.execute("denoise", sigma=1.0)

    async def test_no_image_raises(self) -> None:
        """Executing a tool that requires an image on empty registry raises."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=None, tools=tools)
        with pytest.raises(ToolExecutionError, match="No image path set"):
            await registry.execute("zoom", factor=2.0)

    async def test_reset_no_image_path(self, tmp_path: Path) -> None:
        """Reset when no original image path stored raises ToolExecutionError."""
        tools = create_visual_tools()
        p = _save_image(tmp_path)
        registry = ToolRegistry(image_path=p, tools=tools)
        mgr = registry.get_image_manager()
        # Clear the image_path to simulate no original stored
        object.__setattr__(mgr, "_image_path", None)
        with pytest.raises(ToolExecutionError, match="Cannot reset"):
            await registry.execute("reset")
