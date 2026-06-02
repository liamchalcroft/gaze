"""Tests for JSON extraction utility."""

from __future__ import annotations

from gaze.utils import extract_json_from_text
from gaze.utils.json_extract import _try_repair_truncated


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

    def test_json_with_braces_in_string(self) -> None:
        # Test that braces inside string values are handled correctly
        text = '{"code": "function() { return {}; }", "valid": true}'
        result = extract_json_from_text(text)
        assert result == {"code": "function() { return {}; }", "valid": True}

    def test_json_with_escaped_quotes_and_braces(self) -> None:
        # Complex case with escaped quotes and nested braces in strings
        text = (
            'Some prefix: {"description": "Use {name} in template",'
            ' "example": "{\\"key\\": \\"value\\"}"}'
        )
        result = extract_json_from_text(text)
        assert result is not None, "Expected JSON extraction from escaped text, got None"
        assert result["description"] == "Use {name} in template"

    def test_json_embedded_after_prose(self) -> None:
        # Model output often has explanation before JSON
        text = """Based on my analysis of the image, I found the following:

The lesion appears to be located in the frontal lobe.

{"finding": "lesion", "location": "frontal lobe", "confidence": 0.92, "continue": false}"""
        result = extract_json_from_text(text)
        assert result == {
            "finding": "lesion",
            "location": "frontal lobe",
            "confidence": 0.92,
            "continue": False,
        }


class TestTruncatedJsonRepair:
    """Test truncated JSON repair for local model outputs."""

    def test_missing_closing_brace(self) -> None:
        text = '{"diagnosis": "tumor", "confidence": 0.85'
        result = extract_json_from_text(text)
        assert result == {"diagnosis": "tumor", "confidence": 0.85}

    def test_missing_nested_closing_braces(self) -> None:
        # raw_decode finds the complete inner object first; repair handles
        # the case where no inner object is complete either.
        text = '{"caption": {"description": "mass in frontal lobe", "confidence": 0.9}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["description"] == "mass in frontal lobe"

    def test_both_braces_missing(self) -> None:
        text = '{"caption": {"description": "mass in frontal lobe"'
        result = extract_json_from_text(text)
        assert result is not None

    def test_truncated_mid_string_value(self) -> None:
        text = '{"diagnosis": "glioblastoma multiforme grade IV'
        result = extract_json_from_text(text)
        assert result is not None
        assert "glioblastoma" in result["diagnosis"]

    def test_truncated_after_colon(self) -> None:
        text = '{"diagnosis": "tumor", "confidence": '
        result = extract_json_from_text(text)
        assert result is not None
        assert result["diagnosis"] == "tumor"

    def test_truncated_mid_key(self) -> None:
        text = '{"diagnosis": "tumor", "confid'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["diagnosis"] == "tumor"

    def test_too_short_fragment_returns_none(self) -> None:
        result = _try_repair_truncated('{"a": 1')
        assert result is None  # < 20 chars

    def test_no_brace_returns_none(self) -> None:
        result = _try_repair_truncated("no json here at all, just text")
        assert result is None

    def test_already_balanced_returns_none(self) -> None:
        result = _try_repair_truncated('{"complete": true}  ')
        assert result is None

    def test_nested_array_truncation(self) -> None:
        text = '{"items": ["a", "b", "c"], "count": 3'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["items"] == ["a", "b", "c"]

    def test_deeply_nested_truncation(self) -> None:
        # raw_decode finds the first complete inner object; repair handles
        # the case where nothing is complete.
        text = '{"outer": {"inner": {"value": 42}'
        result = extract_json_from_text(text)
        assert result is not None
        # Inner object {"value": 42} is found by raw_decode
        assert result["value"] == 42

    def test_repair_where_no_inner_object_complete(self) -> None:
        """When no inner object is self-contained, repair closes all braces."""
        text = '{"diagnosis": "tumor", "nested": {"score": 0.85'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["diagnosis"] == "tumor"
