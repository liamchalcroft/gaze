"""Extended tests for radiant_harness.utils.json_coerce — covering uncovered branches.

Targets lines 31-34 (integer from string), 37-39 (boolean from string),
42 (array from string), 51-54 (array items integer coercion), 58 (non-coercible
array item), and 62-64 (array items object recursion).
"""

from __future__ import annotations

from typing import Any

from radiant_harness.utils.json_coerce import _coerce_value, coerce_json_types


# ---------------------------------------------------------------------------
# _coerce_value: integer from string (lines 31-34)
# ---------------------------------------------------------------------------


class TestCoerceIntegerFromString:
    def test_integer_from_clean_string(self) -> None:
        assert _coerce_value("42", {"type": "integer"}) == 42

    def test_integer_from_float_string(self) -> None:
        """int(float("3.7")) → 3  — truncates toward zero."""
        assert _coerce_value("3.7", {"type": "integer"}) == 3

    def test_integer_from_negative_string(self) -> None:
        assert _coerce_value("-5", {"type": "integer"}) == -5

    def test_integer_from_invalid_string_returns_original(self) -> None:
        assert _coerce_value("not_a_number", {"type": "integer"}) == "not_a_number"

    def test_integer_from_empty_string_returns_original(self) -> None:
        assert _coerce_value("", {"type": "integer"}) == ""


# ---------------------------------------------------------------------------
# _coerce_value: boolean from string (lines 37-39)
# ---------------------------------------------------------------------------


class TestCoerceBooleanFromString:
    def test_true_lowercase(self) -> None:
        assert _coerce_value("true", {"type": "boolean"}) is True

    def test_false_lowercase(self) -> None:
        assert _coerce_value("false", {"type": "boolean"}) is False

    def test_true_mixed_case(self) -> None:
        assert _coerce_value("True", {"type": "boolean"}) is True

    def test_false_mixed_case(self) -> None:
        assert _coerce_value("FALSE", {"type": "boolean"}) is False

    def test_non_boolean_string_returns_original(self) -> None:
        assert _coerce_value("yes", {"type": "boolean"}) == "yes"

    def test_empty_string_returns_original(self) -> None:
        assert _coerce_value("", {"type": "boolean"}) == ""


# ---------------------------------------------------------------------------
# _coerce_value: array from string (line 42)
# ---------------------------------------------------------------------------


class TestCoerceArrayFromString:
    def test_non_empty_string_wrapped_in_list(self) -> None:
        assert _coerce_value("glioblastoma", {"type": "array"}) == ["glioblastoma"]

    def test_empty_string_becomes_empty_list(self) -> None:
        assert _coerce_value("", {"type": "array"}) == []


# ---------------------------------------------------------------------------
# _coerce_value: array items integer coercion (lines 51-54, 58)
# ---------------------------------------------------------------------------


class TestCoerceArrayItemsInteger:
    def test_string_items_to_integers(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        assert _coerce_value(["1", "2", "3"], schema) == [1, 2, 3]

    def test_float_items_to_integers(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        assert _coerce_value([1.5, 2.9, 3.0], schema) == [1, 2, 3]

    def test_mixed_string_and_number_items(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        assert _coerce_value(["10", 20, 30.5], schema) == [10, 20, 30]

    def test_invalid_string_item_returns_original(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        original = ["1", "bad", "3"]
        result = _coerce_value(original, schema)
        assert result is original  # returns unchanged

    def test_non_coercible_item_type_returns_original(self) -> None:
        """Items that are not str/int/float (e.g. None) return original list."""
        schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        original = [1, None, 3]
        result = _coerce_value(original, schema)
        assert result is original


# ---------------------------------------------------------------------------
# _coerce_value: array items object recursion (lines 62-64)
# ---------------------------------------------------------------------------


class TestCoerceArrayItemsObjectRecursion:
    def test_recurses_into_object_items(self) -> None:
        """Array of objects with numeric string fields should be coerced."""
        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "label": {"type": "string"},
                },
            },
        }
        data = [
            {"x": "10.5", "y": "20.3", "label": "lesion"},
            {"x": "5.0", "y": "15.0", "label": "normal"},
        ]
        _coerce_value(data, schema)
        assert data[0]["x"] == 10.5
        assert data[0]["y"] == 20.3
        assert data[0]["label"] == "lesion"  # strings stay strings
        assert data[1]["x"] == 5.0

    def test_non_dict_items_in_object_array_skipped(self) -> None:
        """Non-dict items in an object array are silently skipped."""
        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        }
        data = [{"x": "1"}, "not_a_dict", {"x": "2"}]
        _coerce_value(data, schema)
        assert data[0]["x"] == 1.0
        assert data[1] == "not_a_dict"
        assert data[2]["x"] == 2.0


# ---------------------------------------------------------------------------
# coerce_json_types: top-level integration
# ---------------------------------------------------------------------------


class TestCoerceJsonTypesIntegration:
    def test_nested_schema_unwrapping(self) -> None:
        """Handles json_schema.schema wrapper around the actual schema."""
        response: dict[str, Any] = {"confidence": "0.85", "found": "true"}
        schema = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "confidence": {"type": "number"},
                        "found": {"type": "boolean"},
                    }
                }
            }
        }
        coerce_json_types(response, schema)
        assert response["confidence"] == 0.85
        assert response["found"] is True

    def test_deeply_nested_object(self) -> None:
        """Coercion recurses into nested objects."""
        response: dict[str, Any] = {
            "result": {
                "bbox": {"x": "10", "y": "20", "w": "30", "h": "40"},
            }
        }
        schema = {
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                                "w": {"type": "integer"},
                                "h": {"type": "integer"},
                            },
                        }
                    },
                }
            }
        }
        coerce_json_types(response, schema)
        assert response["result"]["bbox"] == {"x": 10, "y": 20, "w": 30, "h": 40}

    def test_array_of_objects_with_nested_bbox(self) -> None:
        """Matches NOVA localizations[].bounding_box pattern."""
        response: dict[str, Any] = {
            "localizations": [
                {"bounding_box": {"x": "1", "y": "2", "w": "3", "h": "4"}, "label": "tumor"}
            ]
        }
        schema = {
            "properties": {
                "localizations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bounding_box": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                    "w": {"type": "integer"},
                                    "h": {"type": "integer"},
                                },
                            },
                            "label": {"type": "string"},
                        },
                    },
                }
            }
        }
        coerce_json_types(response, schema)
        bb = response["localizations"][0]["bounding_box"]
        assert bb == {"x": 1, "y": 2, "w": 3, "h": 4}
        assert response["localizations"][0]["label"] == "tumor"
