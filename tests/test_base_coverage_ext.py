"""Extended tests targeting uncovered lines in base.py.

Covers:
- _try_wrap_inner_schema with "object" type default fill (L291)
- _build_schema_skeleton with field descriptions → field_hints (L381)
- _normalize_images single PIL/Path with wrong label count (L765, L778)
- _normalize_images list of Paths: traversal, missing, bad extension (L800, L802, L804)
- _parse_tool_args with non-Mapping arguments (L1604-1605)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from gaze.base import AgenticProcessorBase
from gaze.base import _build_schema_skeleton
from gaze.base import _try_wrap_inner_schema
from gaze.exceptions import ToolExecutionError
from gaze.types import ToolCall

# ---------------------------------------------------------------------------
# _try_wrap_inner_schema — "object" type default fill (L291)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTryWrapInnerSchemaDefaults:
    def _schema_with_types(self) -> dict[str, Any]:
        """Schema with inner object + various other top-level types."""
        return {
            "json_schema": {
                "schema": {
                    "properties": {
                        "caption": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "score": {"type": "number"},
                            },
                        },
                        "metadata": {"type": "object"},
                        "tags": {"type": "array"},
                        "count": {"type": "integer"},
                        "verified": {"type": "boolean"},
                        "label": {"type": "string"},
                    }
                }
            }
        }

    def test_wraps_inner_keys_and_fills_object_default(self) -> None:
        """When salvaged keys match inner schema, remaining object props get {} (L291)."""
        salvaged = {"text": "hello", "score": 0.9}
        schema = self._schema_with_types()
        result = _try_wrap_inner_schema(salvaged, schema)
        assert result["caption"] == {"text": "hello", "score": 0.9}
        assert result["metadata"] == {}
        assert result["tags"] == []
        assert result["count"] == 0
        assert result["verified"] is False
        assert result["label"] == ""

    def test_no_match_returns_original(self) -> None:
        salvaged = {"unrelated_key": "value"}
        schema = self._schema_with_types()
        result = _try_wrap_inner_schema(salvaged, schema)
        assert result == salvaged


# ---------------------------------------------------------------------------
# _build_schema_skeleton — field descriptions (L381)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSchemaSkeleton:
    def test_field_descriptions_produce_hints(self) -> None:
        """Fields with 'description' populate field_hints list (L381)."""
        schema = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "diagnosis": {
                            "type": "string",
                            "description": "Primary diagnosis",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score 0-1",
                        },
                        "details": {"type": "object"},
                    }
                }
            }
        }
        skeleton, field_hints = _build_schema_skeleton(schema)
        assert skeleton["diagnosis"] == "..."
        assert skeleton["confidence"] == 0
        assert skeleton["details"] == {}  # nested object now recursed
        assert skeleton["continue"] is False
        assert "- diagnosis: Primary diagnosis" in field_hints
        assert "- confidence: Confidence score 0-1" in field_hints
        assert len(field_hints) >= 2  # details fields may also appear

    def test_enum_values_in_skeleton(self) -> None:
        schema = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["normal", "abnormal"],
                        },
                    }
                }
            }
        }
        skeleton, _ = _build_schema_skeleton(schema)
        assert skeleton["status"] == "normal|abnormal"

    def test_none_schema_returns_minimal(self) -> None:
        skeleton, hints = _build_schema_skeleton(None)
        assert skeleton == {"continue": False}
        assert hints == []


# ---------------------------------------------------------------------------
# _normalize_images — single PIL/Path with wrong label count (L765, L778)
# ---------------------------------------------------------------------------


def _make_processor():
    """Create a concrete processor subclass for testing."""

    class TestProcessor(AgenticProcessorBase):
        def get_system_prompt(self, images, metadata):
            return "test"

        def get_user_message(self, images, metadata):
            return "test"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return True

    return TestProcessor(model_name="test")


@pytest.mark.unit
class TestNormalizeImagesValidation:
    def test_single_pil_wrong_label_count_raises(self) -> None:
        """Single PIL Image with 2 labels → ValueError (L765)."""
        proc = _make_processor()
        img = Image.new("RGB", (10, 10))
        with pytest.raises(ValueError, match="Number of labels.*must match.*1"):
            proc._normalize_image_inputs(img, labels=["a", "b"])

    def test_single_path_wrong_label_count_raises(self, tmp_path: Path) -> None:
        """Single Path with 2 labels → ValueError (L778)."""
        proc = _make_processor()
        p = tmp_path / "img.png"
        Image.new("RGB", (10, 10)).save(p)
        with pytest.raises(ValueError, match="Number of labels.*must match.*1"):
            proc._normalize_image_inputs(p, labels=["a", "b"])


# ---------------------------------------------------------------------------
# _normalize_images — list of Paths: traversal, missing, bad extension (L800-804)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeImagesListPaths:
    def test_path_traversal_in_list_raises(self, tmp_path: Path) -> None:
        """Path with '..' in list raises ValueError (L800)."""
        proc = _make_processor()
        traversal_path = tmp_path / ".." / "etc" / "passwd.png"
        with pytest.raises(ValueError, match="Path traversal"):
            proc._normalize_image_inputs([traversal_path], labels=None)

    def test_missing_file_in_list_raises(self, tmp_path: Path) -> None:
        """Non-existent file in list raises FileNotFoundError (L802)."""
        proc = _make_processor()
        missing = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            proc._normalize_image_inputs([missing], labels=None)

    def test_bad_extension_in_list_raises(self, tmp_path: Path) -> None:
        """Unsupported extension in list raises ValueError (L804)."""
        proc = _make_processor()
        bad = tmp_path / "data.txt"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported image format"):
            proc._normalize_image_inputs([bad], labels=None)

    def test_valid_list_of_paths(self, tmp_path: Path) -> None:
        proc = _make_processor()
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.jpg"
        Image.new("RGB", (5, 5)).save(p1)
        Image.new("RGB", (5, 5)).save(p2)
        result = proc._normalize_image_inputs([p1, p2], labels=["first", "second"])
        assert len(result) == 2
        assert result[0].path == p1
        assert result[0].label == "first"
        assert result[1].path == p2
        assert result[1].label == "second"


# ---------------------------------------------------------------------------
# _parse_tool_args — non-Mapping arguments (L1604-1605)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseToolArgsNonMapping:
    def test_string_json_array_raises_tool_execution_error(self) -> None:
        """String arguments that parse to a non-dict raise ToolExecutionError."""
        proc = _make_processor()
        tc = ToolCall(id="1", name="zoom", arguments="[1, 2, 3]")
        with pytest.raises(ToolExecutionError, match="must be a JSON object.*list"):
            proc._parse_tool_args(tc)

    def test_string_invalid_json_raises_tool_execution_error(self) -> None:
        proc = _make_processor()
        tc = ToolCall(id="1", name="zoom", arguments="not json")
        with pytest.raises(ToolExecutionError, match="Malformed JSON"):
            proc._parse_tool_args(tc)

    def test_valid_string_json_object_succeeds(self) -> None:
        proc = _make_processor()
        tc = ToolCall(id="1", name="zoom", arguments='{"x": 10}')
        result = proc._parse_tool_args(tc)
        assert result == {"x": 10}

    def test_valid_frozen_dict_succeeds(self) -> None:
        from gaze._frozen import deep_freeze

        proc = _make_processor()
        frozen = deep_freeze({"x": 10})
        tc = ToolCall(id="1", name="zoom", arguments=frozen)
        result = proc._parse_tool_args(tc)
        assert result == {"x": 10}
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _strip_stale_images tests
# ---------------------------------------------------------------------------


def _make_data_url(label: str = "img") -> str:
    return f"data:image/jpeg;base64,fake-{label}"


def _build_messages_with_tool_images(
    num_tool_rounds: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    for i in range(num_tool_rounds):
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


class TestStripStaleImages:
    def test_no_messages_is_noop(self) -> None:
        messages: list[dict[str, Any]] = []
        AgenticProcessorBase._strip_stale_images(messages)
        assert messages == []

    def test_single_tool_round_preserved(self) -> None:
        messages = _build_messages_with_tool_images(1)
        original_len = len(messages)
        AgenticProcessorBase._strip_stale_images(messages)
        assert len(messages) == original_len
        tool_msg = messages[-1]
        assert any(p["type"] == "image_url" for p in tool_msg["content"])

    def test_older_tool_images_stripped(self) -> None:
        messages = _build_messages_with_tool_images(3)
        AgenticProcessorBase._strip_stale_images(messages)

        last_tool = messages[-1]
        assert last_tool["role"] == "tool"
        has_image = any(p.get("type") == "image_url" for p in last_tool["content"])
        assert has_image, "Latest tool image should be preserved"

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            if msg is last_tool:
                continue
            for part in msg["content"]:
                assert part["type"] != "image_url", f"Stale image_url should be stripped: {part}"
                if "previous tool image omitted" in part.get("text", ""):
                    break
            else:
                pytest.fail("Expected placeholder text in stripped tool message")

    def test_text_only_tool_messages_untouched(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Found 5 results"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "Found 3 results"},
        ]
        original = [m.get("content") for m in messages]
        AgenticProcessorBase._strip_stale_images(messages)
        current = [m.get("content") for m in messages]
        assert original == current

    def test_preserves_text_parts_in_stripped_messages(self) -> None:
        messages = _build_messages_with_tool_images(2)
        AgenticProcessorBase._strip_stale_images(messages)

        first_tool = messages[3]
        text_parts = [p for p in first_tool["content"] if p["type"] == "text"]
        assert len(text_parts) >= 1
        texts = [p["text"] for p in text_parts]
        assert any("Zoomed round 0" in t for t in texts)

    def test_user_message_images_stripped(self) -> None:
        messages = [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this brain MRI"},
                    {"type": "image_url", "image_url": {"url": _make_data_url("input1")}},
                    {"type": "image_url", "image_url": {"url": _make_data_url("input2")}},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "zoom", "arguments": '{"factor": 2.0}'},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {"type": "text", "text": "Zoomed image"},
                    {"type": "image_url", "image_url": {"url": _make_data_url("zoomed")}},
                ],
            },
        ]

        AgenticProcessorBase._strip_stale_images(messages)

        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0] == {"type": "text", "text": "Analyze this brain MRI"}
        assert user_content[1] == {"type": "text", "text": "[original image omitted]"}
        assert user_content[2] == {"type": "text", "text": "[original image omitted]"}

        tool_content = messages[3]["content"]
        assert any(p.get("type") == "image_url" for p in tool_content)

    def test_user_message_string_content_untouched(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plain text question"},
            {"role": "assistant", "content": '{"continue": false, "answer": "ok"}'},
        ]
        AgenticProcessorBase._strip_stale_images(messages)
        assert messages[1]["content"] == "plain text question"
