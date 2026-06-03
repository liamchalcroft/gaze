"""Tests for coordinate space tracking in the agentic loop.

Verifies that tools which modify the image coordinate space (crop, zoom,
rotate, flip_horizontal, flip_vertical) trigger the final-turn coordinate
invalidation warning.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image as PILImage

from gaze.base import _COORD_MODIFYING_TOOLS
from gaze.base import AgenticProcessorBase
from gaze.base import ImageInput
from gaze.models import AdapterProtocol
from gaze.models import GenerationLog
from gaze.tools import Tool
from gaze.tools import ToolRegistry
from gaze.types import ToolResult

# ---------------------------------------------------------------------------
# Constant membership
# ---------------------------------------------------------------------------


class TestCoordModifyingToolsConstant:
    """Verify _COORD_MODIFYING_TOOLS contains all coordinate-modifying tools."""

    def test_contains_crop(self) -> None:
        assert "crop" in _COORD_MODIFYING_TOOLS

    def test_contains_zoom(self) -> None:
        assert "zoom" in _COORD_MODIFYING_TOOLS

    def test_contains_rotate(self) -> None:
        assert "rotate" in _COORD_MODIFYING_TOOLS

    def test_contains_flip_horizontal(self) -> None:
        assert "flip_horizontal" in _COORD_MODIFYING_TOOLS

    def test_contains_flip_vertical(self) -> None:
        assert "flip_vertical" in _COORD_MODIFYING_TOOLS

    def test_is_frozenset(self) -> None:
        assert isinstance(_COORD_MODIFYING_TOOLS, frozenset)


# ---------------------------------------------------------------------------
# Helpers: fake tools and adapters
# ---------------------------------------------------------------------------


async def _fake_rotate(registry: ToolRegistry, angle: float = 90.0) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="rotate",
        description=f"Rotated by {angle} degrees",
        metadata={"angle": angle},
    )


async def _fake_flip_horizontal(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="flip_horizontal",
        description="Flipped horizontal",
    )


async def _fake_flip_vertical(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="flip_vertical",
        description="Flipped vertical",
    )


async def _fake_crop(
    registry: ToolRegistry, x1: float = 0, y1: float = 0, x2: float = 50, y2: float = 50
) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="crop",
        description="Cropped region",
        metadata={"box": [x1, y1, x2, y2]},
    )


async def _fake_contrast(registry: ToolRegistry, factor: float = 1.5) -> ToolResult:  # noqa: ARG001
    """Non-coordinate-modifying tool for negative tests."""
    return ToolResult(
        tool_name="contrast",
        description=f"Contrast factor {factor}",
        metadata={"factor": factor},
    )


def _make_coord_tools() -> list[Tool]:
    """Create a set of fake tools for testing coordinate tracking."""
    return [
        Tool(
            name="rotate",
            description="Rotate image",
            parameters={"angle": {"type": "number", "description": "degrees", "default": 90.0}},
            execute=_fake_rotate,
            requires_image=False,  # False to avoid needing real image loading
        ),
        Tool(
            name="flip_horizontal",
            description="Flip image horizontally",
            parameters={},
            execute=_fake_flip_horizontal,
            requires_image=False,
        ),
        Tool(
            name="flip_vertical",
            description="Flip image vertically",
            parameters={},
            execute=_fake_flip_vertical,
            requires_image=False,
        ),
        Tool(
            name="crop",
            description="Crop image",
            parameters={
                "x1": {"type": "number", "description": "x1", "default": 0},
                "y1": {"type": "number", "description": "y1", "default": 0},
                "x2": {"type": "number", "description": "x2", "default": 50},
                "y2": {"type": "number", "description": "y2", "default": 50},
            },
            execute=_fake_crop,
            requires_image=False,
        ),
        Tool(
            name="contrast",
            description="Adjust contrast",
            parameters={"factor": {"type": "number", "description": "factor", "default": 1.5}},
            execute=_fake_contrast,
            requires_image=False,
        ),
    ]


class CoordTrackingAdapter(AdapterProtocol):
    """Adapter that calls a specified tool on turn 1, then finalizes on the last turn.

    With max_turns=2: turn 0 calls the tool, turn 1 is the forced last turn
    where the coordinate warning is injected (tools stripped, schema enforced).
    Records the messages sent on each call so tests can inspect the
    final-turn coordinate warning.
    """

    supports_multipart_tool_content: bool = True

    def __init__(self, tool_name: str, tool_args: dict[str, Any] | None = None) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.calls = 0
        self.messages_history: list[list[dict[str, Any]]] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format, stream, kwargs
        self.messages_history.append(list(messages or []))
        self.calls += 1

        if self.calls == 1:
            import json

            return (
                "",
                [
                    {
                        "id": f"call-{self.calls}",
                        "name": self.tool_name,
                        "arguments": json.dumps(self.tool_args),
                    }
                ],
                GenerationLog(prompt_tokens=10, completion_tokens=10, finish_reason="tool_call"),
            )
        # Turn 2+: finalize
        return (
            '{"continue": false, "result": "analysis complete", "bounding_box": [10, 20, 50, 60]}',
            None,
            GenerationLog(prompt_tokens=10, completion_tokens=10, finish_reason="stop"),
        )


class CoordTrackingProcessor(AgenticProcessorBase):
    """Processor wired to a CoordTrackingAdapter for testing coordinate tracking."""

    def __init__(self, adapter: CoordTrackingAdapter, with_images: bool = True) -> None:
        self._with_images = with_images
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=2,
            adapter_factory=lambda: adapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "You are a radiology analysis system."

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "Analyze this brain MRI."

    def get_response_schema(self) -> dict[str, Any] | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "continue": {"type": "boolean"},
                        "result": {"type": "string"},
                        "bounding_box": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["continue", "result"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "result" in response

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        return ToolRegistry(image_path=None, tools=_make_coord_tools())


def _extract_final_turn_text(messages: list[dict[str, Any]]) -> str:
    """Extract all text content from user messages in a message list."""
    texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.get("text", "")  # type: ignore[union-attr]
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Integration tests: coordinate warning on final turn
# ---------------------------------------------------------------------------


class TestCoordSpaceWarning:
    """Verify the final-turn coordinate warning fires for coord-modifying tools."""

    @pytest.mark.asyncio
    async def test_rotate_triggers_coord_warning(self) -> None:
        """Using rotate should trigger the coordinate space warning on the final turn."""
        adapter = CoordTrackingAdapter("rotate", {"angle": 90.0})
        processor = CoordTrackingProcessor(adapter)

        # Use a real PIL image to trigger the image-aware final turn path
        img = PILImage.new("RGB", (100, 100), color="red")
        result = await processor.analyze(images=img, metadata={})

        assert result.final_response["result"] == "analysis complete"

        # The final call (turn 1, the forced last turn) should include coord warning
        final_messages = adapter.messages_history[-1]
        final_text = _extract_final_turn_text(final_messages)
        assert "coordinate" in final_text.lower() or "invalid" in final_text.lower(), (
            f"Expected coordinate warning in final turn messages. Got: {final_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_flip_horizontal_triggers_coord_warning(self) -> None:
        """Using flip_horizontal should trigger the coordinate space warning."""
        adapter = CoordTrackingAdapter("flip_horizontal")
        processor = CoordTrackingProcessor(adapter)

        img = PILImage.new("RGB", (100, 100), color="blue")
        result = await processor.analyze(images=img, metadata={})

        assert result.final_response["result"] == "analysis complete"

        final_messages = adapter.messages_history[-1]
        final_text = _extract_final_turn_text(final_messages)
        assert "coordinate" in final_text.lower() or "invalid" in final_text.lower(), (
            f"Expected coordinate warning in final turn messages. Got: {final_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_flip_vertical_triggers_coord_warning(self) -> None:
        """Using flip_vertical should trigger the coordinate space warning."""
        adapter = CoordTrackingAdapter("flip_vertical")
        processor = CoordTrackingProcessor(adapter)

        img = PILImage.new("RGB", (100, 100), color="blue")
        result = await processor.analyze(images=img, metadata={})

        assert result.final_response["result"] == "analysis complete"

        final_messages = adapter.messages_history[-1]
        final_text = _extract_final_turn_text(final_messages)
        assert "coordinate" in final_text.lower() or "invalid" in final_text.lower(), (
            f"Expected coordinate warning in final turn messages. Got: {final_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_crop_triggers_coord_warning(self) -> None:
        """Using crop should trigger the coordinate space warning (existing behavior)."""
        adapter = CoordTrackingAdapter("crop", {"x1": 0, "y1": 0, "x2": 50, "y2": 50})
        processor = CoordTrackingProcessor(adapter)

        img = PILImage.new("RGB", (100, 100), color="green")
        result = await processor.analyze(images=img, metadata={})

        assert result.final_response["result"] == "analysis complete"

        final_messages = adapter.messages_history[-1]
        final_text = _extract_final_turn_text(final_messages)
        assert "coordinate" in final_text.lower() or "invalid" in final_text.lower(), (
            f"Expected coordinate warning in final turn messages. Got: {final_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_contrast_does_not_trigger_coord_warning(self) -> None:
        """Non-coordinate-modifying tools should NOT trigger the coord warning."""
        adapter = CoordTrackingAdapter("contrast", {"factor": 1.5})
        processor = CoordTrackingProcessor(adapter)

        img = PILImage.new("RGB", (100, 100), color="yellow")
        result = await processor.analyze(images=img, metadata={})

        assert result.final_response["result"] == "analysis complete"

        # The final turn should still have a message about original image
        # but NOT the "previously used crop/zoom which changed the coordinate space" warning
        final_messages = adapter.messages_history[-1]
        final_text = _extract_final_turn_text(final_messages)
        assert "changed the coordinate space" not in final_text.lower(), (
            f"Non-coord tool should NOT trigger coord change warning. Got: {final_text[:500]}"
        )
