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

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import _build_schema_skeleton
from radiant_harness.base import _try_wrap_inner_schema
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.types import ToolCall

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
        assert skeleton["confidence"] == "0"
        assert skeleton["details"] == "{...}"
        assert skeleton["continue"] == "false"
        assert "- diagnosis: Primary diagnosis" in field_hints
        assert "- confidence: Confidence score 0-1" in field_hints
        assert len(field_hints) == 2  # details has no description

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
        assert skeleton == {"continue": "false"}
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
        from radiant_harness._frozen import deep_freeze

        proc = _make_processor()
        frozen = deep_freeze({"x": 10})
        tc = ToolCall(id="1", name="zoom", arguments=frozen)
        result = proc._parse_tool_args(tc)
        assert result == {"x": 10}
        assert isinstance(result, dict)
