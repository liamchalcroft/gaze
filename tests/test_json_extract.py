"""Tests for JSON extraction utility."""

from __future__ import annotations

import pytest

from radiant_harness.utils import extract_json_from_text


class TestExtractJsonFromText:
    """Test JSON extraction from model output text."""

    def test_empty_text(self) -> None:
        assert extract_json_from_text("") is None
        assert extract_json_from_text("   ") is None

    def test_raw_json(self) -> None:
        result = extract_json_from_text('{"answer": "yes"}')
        assert result == {"answer": "yes"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is my answer: {"answer": "no"} I hope that helps.'
        result = extract_json_from_text(text)
        assert result == {"answer": "no"}

    def test_markdown_json_block(self) -> None:
        text = """Here is my response:
```json
{"diagnosis": "pneumonia", "confidence": 0.85}
```
That's my analysis."""
        result = extract_json_from_text(text)
        assert result == {"diagnosis": "pneumonia", "confidence": 0.85}

    def test_markdown_block_no_language(self) -> None:
        text = """```
{"value": 42}
```"""
        result = extract_json_from_text(text)
        assert result == {"value": 42}

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": {"deep": true}}}'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": {"deep": True}}}

    def test_json_with_array(self) -> None:
        text = '{"items": [1, 2, 3], "count": 3}'
        result = extract_json_from_text(text)
        assert result == {"items": [1, 2, 3], "count": 3}

    def test_non_dict_json(self) -> None:
        # Arrays should return None (we only want dicts)
        assert extract_json_from_text("[1, 2, 3]") is None
        assert extract_json_from_text('"just a string"') is None

    def test_invalid_json(self) -> None:
        assert extract_json_from_text("{invalid json}") is None
        assert extract_json_from_text("not json at all") is None

    def test_incomplete_markdown_block(self) -> None:
        # No closing fence
        assert extract_json_from_text('```json\n{"key": "value"}') is None
        # No opening fence content
        assert extract_json_from_text("```") is None

    def test_multiple_json_objects(self) -> None:
        # Should return the first valid JSON object
        text = '{"first": 1} {"second": 2}'
        result = extract_json_from_text(text)
        assert result == {"first": 1}

    def test_json_with_unicode(self) -> None:
        text = '{"message": "Hello 世界", "emoji": "🎉"}'
        result = extract_json_from_text(text)
        assert result == {"message": "Hello 世界", "emoji": "🎉"}
