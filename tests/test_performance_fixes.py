"""Tests for performance optimizations in Patch Set #1.

Covers:
1. _strip_stale_tool_images — message payload pruning
2. _transform_and_encode — offloaded PIL transforms
3. ImageInput.from_pil — skip disk I/O for in-memory images
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import ImageInput
from radiant_harness.tools.registry import ToolRegistry
from radiant_harness.tools.visual import create_visual_tools


# =====================================================================
# 1. _strip_stale_tool_images tests
# =====================================================================


def _make_data_url(label: str = "img") -> str:
    """Create a fake base64 data URL for testing."""
    return f"data:image/jpeg;base64,fake-{label}"


def _build_messages_with_tool_images(
    num_tool_rounds: int,
) -> list[dict[str, Any]]:
    """Build a realistic messages list with multiple tool image rounds.

    Each round consists of:
      - assistant message (with tool_calls)
      - tool message (with image_url content)
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]

    for i in range(num_tool_rounds):
        # Assistant requests a tool call
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call-{i}",
                        "type": "function",
                        "function": {"name": "zoom", "arguments": '{"factor": 2.0}'},
                    }
                ],
            }
        )
        # Tool returns an image
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{i}",
                "content": [
                    {"type": "text", "text": f"Zoomed round {i}"},
                    {"type": "image_url", "image_url": {"url": _make_data_url(f"r{i}")}},
                ],
            }
        )

    return messages


class TestStripStaleToolImages:
    def test_no_messages_is_noop(self) -> None:
        messages: list[dict[str, Any]] = []
        AgenticProcessorBase._strip_stale_tool_images(messages)
        assert messages == []

    def test_single_tool_round_preserved(self) -> None:
        """With only one round, images after the last assistant msg are kept."""
        messages = _build_messages_with_tool_images(1)
        original_len = len(messages)
        AgenticProcessorBase._strip_stale_tool_images(messages)
        assert len(messages) == original_len
        # The tool message should still have its image_url
        tool_msg = messages[-1]
        assert any(p["type"] == "image_url" for p in tool_msg["content"])

    def test_older_tool_images_stripped(self) -> None:
        """With multiple rounds, only the latest round's images survive."""
        messages = _build_messages_with_tool_images(3)
        AgenticProcessorBase._strip_stale_tool_images(messages)

        # The last tool message (round 2) should still have image_url
        last_tool = messages[-1]
        assert last_tool["role"] == "tool"
        has_image = any(
            p.get("type") == "image_url" for p in last_tool["content"]
        )
        assert has_image, "Latest tool image should be preserved"

        # Earlier tool messages (round 0, 1) should have placeholders
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            if msg is last_tool:
                continue
            for part in msg["content"]:
                assert part["type"] != "image_url", (
                    f"Stale image_url should be stripped: {part}"
                )
                if "previous tool image omitted" in part.get("text", ""):
                    break
            else:
                pytest.fail("Expected placeholder text in stripped tool message")

    def test_text_only_tool_messages_untouched(self) -> None:
        """Tool messages with string content (no images) are not modified."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Found 5 results"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "Found 3 results"},
        ]
        original = [m.get("content") for m in messages]
        AgenticProcessorBase._strip_stale_tool_images(messages)
        current = [m.get("content") for m in messages]
        assert original == current

    def test_preserves_text_parts_in_stripped_messages(self) -> None:
        """Text parts in tool messages with images should be kept."""
        messages = _build_messages_with_tool_images(2)
        AgenticProcessorBase._strip_stale_tool_images(messages)

        # First tool message (index 3) should have text preserved
        first_tool = messages[3]
        text_parts = [p for p in first_tool["content"] if p["type"] == "text"]
        assert len(text_parts) >= 1
        # Should have original text AND placeholder
        texts = [p["text"] for p in text_parts]
        assert any("Zoomed round 0" in t for t in texts)


# =====================================================================
# 2. _transform_and_encode tests
# =====================================================================


def _save_image(tmp_path: Path, width: int = 100, height: int = 100) -> Path:
    path = tmp_path / "test.png"
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(path)
    return path


class TestTransformAndEncode:
    @pytest.mark.asyncio
    async def test_zoom_produces_encoded_image(self, tmp_path: Path) -> None:
        """Zoom via _transform_and_encode returns valid encoded image."""
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)

        result = await registry.execute("zoom", factor=2.0)
        assert result.success
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0
        assert result.image_mime_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_sequential_transforms_work(self, tmp_path: Path) -> None:
        """Multiple sequential transforms produce correct final state."""
        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)

        # Zoom then crop
        r1 = await registry.execute("zoom", factor=2.0)
        assert r1.success
        assert r1.metadata["new_size"] == (200, 200)

        r2 = await registry.execute("crop", box=[0.0, 0.0, 0.5, 0.5])
        assert r2.success
        assert r2.metadata["new_size"] == (100, 100)

    @pytest.mark.asyncio
    async def test_error_propagation(self, tmp_path: Path) -> None:
        """ValueError from transform is re-raised as ToolExecutionError."""
        from radiant_harness.exceptions import ToolExecutionError

        tools = create_visual_tools()
        image_path = _save_image(tmp_path)
        registry = ToolRegistry(image_path=image_path, tools=tools)

        with pytest.raises(ToolExecutionError, match="Invalid zoom factor"):
            await registry.execute("zoom", factor=99.0)

    @pytest.mark.asyncio
    async def test_flip_returns_same_size(self, tmp_path: Path) -> None:
        """Flip operations return same-size encoded image."""
        tools = create_visual_tools()
        image_path = _save_image(tmp_path, 50, 80)
        registry = ToolRegistry(image_path=image_path, tools=tools)

        result = await registry.execute("flip_horizontal")
        assert result.success
        assert result.metadata["size"] == (50, 80)


# =====================================================================
# 3. ImageInput.from_pil tests (skip temp-file round-trip)
# =====================================================================


class TestImageInputFromPil:
    def test_from_pil_populates_all_fields(self) -> None:
        """from_pil sets width, height, encoded, and pil_image."""
        img = Image.new("RGB", (64, 48), color=(255, 0, 0))
        inp = ImageInput.from_pil(img)

        assert inp.width == 64
        assert inp.height == 48
        assert inp.encoded is not None
        assert inp.encoded.mime_type == "image/jpeg"
        assert len(inp.encoded.data) > 0
        assert inp.pil_image is img
        assert inp.path == Path("<in-memory>")

    def test_from_pil_custom_label_and_path(self) -> None:
        """from_pil accepts optional label and path."""
        img = Image.new("RGB", (32, 32))
        custom_path = Path("/fake/test.png")
        inp = ImageInput.from_pil(img, label="T1-weighted", path=custom_path)

        assert inp.label == "T1-weighted"
        assert inp.path == custom_path

    def test_load_is_noop_when_already_loaded(self) -> None:
        """Calling load() on a from_pil input does not re-read from disk."""
        img = Image.new("RGB", (32, 32))
        inp = ImageInput.from_pil(img)
        original_encoded = inp.encoded

        # load() should be a no-op — no disk path to read from
        inp.load()

        assert inp.encoded is original_encoded
        assert inp.pil_image is img

    def test_from_pil_rejects_oversized_image(self) -> None:
        """from_pil validates max dimension just like load()."""
        from radiant_harness.config import get_config

        max_dim = get_config().image.max_image_dimension
        oversized = Image.new("RGB", (max_dim + 1, 10))

        with pytest.raises(ValueError, match="exceed"):
            ImageInput.from_pil(oversized)
