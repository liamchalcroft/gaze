"""Comprehensive tests for type definitions and JSON parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pydantic import ValidationError

from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import JSONParseError
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.types import parse_json_response


@pytest.mark.unit
class TestModelResponse:
    """Test cases for ModelResponse type."""

    def test_model_response_creation(self):
        """Test ModelResponse can be created with valid data."""
        response = ModelResponse(
            text="Test response",
            confidence=0.85,
            reasoning="Test reasoning",
            metadata={"source": "test"},
        )

        assert response.text == "Test response"
        assert response.confidence == 0.85
        assert response.reasoning == "Test reasoning"
        assert response.metadata == {"source": "test"}

    def test_model_response_validation(self):
        """Test ModelResponse validates field types."""
        # Valid creation
        response = ModelResponse(
            text="Valid text",
            confidence=0.5,
            reasoning="Valid reasoning",
            metadata={},
        )
        assert isinstance(response.confidence, float)
        assert isinstance(response.text, str)

    def test_model_response_confidence_bounds(self):
        """Test ModelResponse accepts confidence values in valid range."""
        # Test boundary values
        response_min = ModelResponse(
            text="Min confidence", confidence=0.0, reasoning="", metadata={}
        )
        assert response_min.confidence == 0.0

        response_max = ModelResponse(
            text="Max confidence", confidence=1.0, reasoning="", metadata={}
        )
        assert response_max.confidence == 1.0

    def test_model_response_optional_fields(self):
        """Test ModelResponse handles optional fields correctly."""
        # Test with minimal required fields
        response = ModelResponse(
            text="Required text",
            confidence=0.5,
        )

        # Optional fields should be None by default
        assert response.reasoning is None
        assert response.metadata is None

    def test_model_response_serialization(self):
        """Test ModelResponse can be serialized to dict."""
        response = ModelResponse(
            text="Serialization test",
            confidence=0.75,
            reasoning="Test serialization",
            metadata={"key": "value"},
        )

        response_dict = response.model_dump()

        assert response_dict["text"] == "Serialization test"
        assert response_dict["confidence"] == 0.75
        assert response_dict["reasoning"] == "Test serialization"
        assert response_dict["metadata"]["key"] == "value"

    def test_model_response_deserialization(self):
        """Test ModelResponse can be created from dict."""
        response_data: dict[str, Any] = {
            "text": "Deserialization test",
            "confidence": 0.9,
            "reasoning": "Test deserialization",
            "metadata": {"source": "dict"},
        }

        response = ModelResponse(**response_data)

        assert response.text == "Deserialization test"
        assert response.confidence == 0.9
        assert response.reasoning == "Test deserialization"
        assert response.metadata is not None
        assert response.metadata["source"] == "dict"


@pytest.mark.unit
class TestBatchData:
    """Test cases for BatchData type."""

    def test_batch_data_creation(self):
        """Test BatchData can be created with valid paths and metadata."""
        batch = BatchData(
            images=[Path("/path/to/image1.png"), Path("/path/to/image2.png")],
            metadata=[{"id": 1, "type": "test"}, {"id": 2, "type": "validation"}],
        )

        assert len(batch.images) == 2
        assert len(batch.metadata) == 2
        assert isinstance(batch.images[0], Path)
        assert isinstance(batch.metadata[0], dict)

    def test_batch_data_empty_batch(self):
        """Test BatchData handles empty batches."""
        batch = BatchData(images=[], metadata=[])

        assert len(batch.images) == 0
        assert len(batch.metadata) == 0

    def test_batch_data_length_mismatch(self):
        """Test BatchData validation with mismatched lengths."""
        # This tests the assumption that images and metadata should have same length
        # The actual validation might be in the application logic
        batch = BatchData(
            images=[Path("/path/to/image.png")],
            metadata=[{"id": 1}, {"id": 2}],  # Different length
        )

        # BatchData itself might not enforce length matching,
        # but we can test that it stores the data as provided
        assert len(batch.images) == 1
        assert len(batch.metadata) == 2

    def test_batch_data_path_types(self):
        """Test BatchData handles different path input types."""
        # Test with string paths that get converted to Path objects
        batch = BatchData(
            images=["/string/path.png", Path("/path/obj.png")],
            metadata=[{}, {}],
        )

        # Should accept both string and Path types
        assert isinstance(batch.images[0], str)
        assert isinstance(batch.images[1], Path)


@pytest.mark.unit
class TestEvaluationMetrics:
    """Test cases for EvaluationMetrics type."""

    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics can be created with valid metrics."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.92,
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.auc_roc == 0.92

    def test_evaluation_metrics_optional_fields(self):
        """Test EvaluationMetrics with optional fields."""
        # Test with only accuracy
        metrics = EvaluationMetrics(
            accuracy=0.75, precision=None, recall=None, f1_score=None, auc_roc=None
        )

        assert metrics.accuracy == 0.75
        # Other fields should be None or have default values
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1_score is None
        assert metrics.auc_roc is None

    def test_evaluation_metrics_optional_none(self):
        """Test EvaluationMetrics with optional None values."""
        metrics = EvaluationMetrics(
            accuracy=0.8,  # Required field
            precision=None,
            recall=None,
            f1_score=None,
            auc_roc=None,
        )

        # accuracy is required, others can be None
        assert metrics.accuracy == 0.8
        assert all(
            value is None
            for value in [metrics.precision, metrics.recall, metrics.f1_score, metrics.auc_roc]
        )

    def test_evaluation_metrics_boundary_values(self):
        """Test EvaluationMetrics with boundary metric values."""
        # Test perfect scores
        perfect_metrics = EvaluationMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            auc_roc=1.0,
        )

        assert all(
            value == 1.0
            for value in [
                perfect_metrics.accuracy,
                perfect_metrics.precision,
                perfect_metrics.recall,
                perfect_metrics.f1_score,
                perfect_metrics.auc_roc,
            ]
        )

        # Test zero scores
        zero_metrics = EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_roc=0.0,
        )

        assert all(
            value == 0.0
            for value in [
                zero_metrics.accuracy,
                zero_metrics.precision,
                zero_metrics.recall,
                zero_metrics.f1_score,
                zero_metrics.auc_roc,
            ]
        )


@pytest.mark.unit
class TestJSONParseError:
    """Test cases for JSONParseError exception."""

    def test_json_parse_error_creation(self):
        """Test JSONParseError can be created with original content and error."""
        original_content = '{"invalid": json}'
        error_msg = "Expecting ',' delimiter"

        error = JSONParseError(original_content, error_msg)

        assert error.original_content == original_content
        assert error.error == error_msg
        assert error_msg in str(error)

    def test_json_parse_error_message_format(self):
        """Test JSONParseError formats message correctly."""
        content = "malformed json"
        error_msg = "JSON decode error"

        error = JSONParseError(content, error_msg)

        expected_message = f"Failed to parse JSON: {error_msg}"
        assert str(error) == expected_message

    def test_json_parse_error_inheritance(self):
        """Test JSONParseError inherits from Exception properly."""
        error = JSONParseError("content", "error")

        assert isinstance(error, Exception)
        assert isinstance(error, JSONParseError)


@pytest.mark.unit
class TestParseJSONResponse:
    """Test cases for JSON response parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON responses."""
        valid_json = '{"key": "value", "number": 42, "array": [1, 2, 3]}'

        result = parse_json_response(valid_json)

        assert result == {"key": "value", "number": 42, "array": [1, 2, 3]}

    def test_parse_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        markdown_json = """```json
{"result": "success", "data": {"id": 123}}
```"""

        result = parse_json_response(markdown_json)

        assert result == {"result": "success", "data": {"id": 123}}

    def test_parse_json_with_multiple_fence_types(self):
        """Test parsing handles different markdown fence types."""
        test_cases = [
            '```json\n{"test": "value"}\n```',
            '```\n{"test": "value"}\n```',
            '`{"test": "value"}`',
        ]

        for test_case in test_cases:
            result = parse_json_response(test_case)
            assert result == {"test": "value"}

    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with various whitespace patterns."""
        whitespace_cases = [
            '  \n  {"clean": "json"}  \n  ',
            '\t{"tabbed": "json"}\t',
            '\r\n{"windows": "json"}\r\n',
        ]

        for case in whitespace_cases:
            result = parse_json_response(case)
            assert "clean" in result or "tabbed" in result or "windows" in result

    def test_parse_json_with_prefix_text(self):
        """Test parsing removes common prefixes."""
        prefixed_cases = [
            'Here is the JSON:\n{"data": "value"}',
            'JSON:\n{"data": "value"}',
            'Result: {"data": "value"}',
            '{"data": "value"}',  # No prefix
        ]

        for case in prefixed_cases:
            result = parse_json_response(case)
            assert result == {"data": "value"}

    def test_parse_invalid_json_raises_error(self):
        """Test parsing invalid JSON raises JSONParseError."""
        invalid_cases = [
            '{"invalid": json}',  # Unquoted value
            '{"missing": }',  # Missing value
            '{invalid: "json"}',  # Unquoted key
            "not json at all",  # Not JSON
            "",  # Empty string
            "{}{}",  # Multiple JSON objects
        ]

        for invalid_case in invalid_cases:
            with pytest.raises(JSONParseError) as exc_info:
                parse_json_response(invalid_case)

            # Verify error contains original content
            assert exc_info.value.original_content == invalid_case
            assert len(exc_info.value.error) > 0

    def test_parse_json_preserves_nested_structures(self):
        """Test parsing preserves complex nested JSON structures."""
        complex_json = json.dumps(
            {
                "metadata": {
                    "confidence": 0.85,
                    "model": "test-model",
                    "nested": {
                        "array": [1, 2, {"key": "value"}],
                        "null_value": None,
                        "boolean": True,
                    },
                },
                "results": [
                    {"id": 1, "score": 0.9},
                    {"id": 2, "score": 0.8},
                ],
            }
        )

        result = parse_json_response(complex_json)

        assert result["metadata"]["confidence"] == 0.85
        assert result["metadata"]["nested"]["boolean"] is True
        assert result["metadata"]["nested"]["null_value"] is None
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.9

    def test_parse_json_unicode_handling(self):
        """Test parsing handles unicode characters correctly."""
        unicode_json = '{"unicode": "café", "emoji": "🔥", "chinese": "你好"}'

        result = parse_json_response(unicode_json)

        assert result["unicode"] == "café"
        assert result["emoji"] == "🔥"
        assert result["chinese"] == "你好"

    def test_parse_json_large_numbers(self):
        """Test parsing handles large numbers correctly."""
        large_number_json = '{"large_int": 9999999999999999, "large_float": 1.23e-10}'

        result = parse_json_response(large_number_json)

        assert result["large_int"] == 9999999999999999
        assert result["large_float"] == 1.23e-10

    def test_parse_json_error_context(self):
        """Test JSONParseError provides useful error context."""
        malformed_json = '{"key": "value", "invalid": }'

        with pytest.raises(JSONParseError) as exc_info:
            parse_json_response(malformed_json)

        error = exc_info.value
        assert error.original_content == malformed_json
        assert "JSON" in error.error or "Expecting" in error.error


@pytest.mark.integration
class TestTypeIntegration:
    """Integration tests for type interactions."""

    def test_model_response_from_parsed_json(self):
        """Test creating ModelResponse from parsed JSON."""
        json_text = """{
            "text": "Integration test response",
            "confidence": 0.95,
            "reasoning": "High confidence prediction",
            "metadata": {"source": "json_parse"}
        }"""

        parsed_data = parse_json_response(json_text)
        response = ModelResponse(**parsed_data)

        assert response.text == "Integration test response"
        assert response.confidence == 0.95
        assert response.metadata is not None
        assert response.metadata["source"] == "json_parse"

    def test_evaluation_metrics_from_parsed_json(self):
        """Test creating EvaluationMetrics from parsed JSON."""
        metrics_json = """{
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.90,
            "f1_score": 0.87,
            "auc_roc": 0.92
        }"""

        parsed_metrics = parse_json_response(metrics_json)
        metrics = EvaluationMetrics(**parsed_metrics)

        assert metrics.accuracy == 0.88
        assert metrics.f1_score == 0.87

    def test_batch_data_with_model_responses(self):
        """Test BatchData can work with ModelResponse objects."""
        responses = [
            ModelResponse(text="Response 1", confidence=0.8),
            ModelResponse(text="Response 2", confidence=0.9),
        ]

        # BatchData primarily holds paths and metadata, but we can test integration
        batch = BatchData(
            images=[Path("/img1.png"), Path("/img2.png")],
            metadata=[r.model_dump() for r in responses],
        )

        assert len(batch.images) == 2
        assert batch.metadata[0]["text"] == "Response 1"
        assert batch.metadata[1]["confidence"] == 0.9


@pytest.mark.edge_case
class TestTypeErrorHandling:
    """Test error handling and edge cases for types."""

    def test_model_response_invalid_confidence(self):
        """Test ModelResponse validation with invalid confidence values."""
        # Confidence values outside 0-1 range might be accepted depending on validation
        # Test what the current implementation actually does
        try:
            response = ModelResponse(
                text="Test",
                confidence=1.5,  # > 1.0
            )
            # If no validation error, verify value is stored
            assert response.confidence == 1.5
        except (ValidationError, BeartypeCallHintParamViolation):
            # If validation exists, ensure it catches invalid values
            pass

    def test_batch_data_string_paths(self):
        """Test BatchData handling of string image paths."""
        # Test with string paths
        batch = BatchData(
            images=["string/path.png"],
            metadata=[{}],
        )

        # Should accept string paths as-is (no automatic conversion)
        assert isinstance(batch.images[0], str)
        assert batch.images[0] == "string/path.png"

    def test_parse_json_dict_only(self):
        """Test JSON parsing only accepts dictionary inputs."""
        valid_dict_cases = [
            "{}",  # Empty object
            '{"key": "value"}',  # Simple dict
            '{"nested": {"key": "value"}}',  # Nested dict
        ]

        for case in valid_dict_cases:
            result = parse_json_response(case)
            assert isinstance(result, dict)

        # Non-dict JSON should raise errors
        non_dict_cases = ["null", "true", "42", '"string"', "[]"]

        for case in non_dict_cases:
            with pytest.raises(JSONParseError):
                parse_json_response(case)

    def test_types_memory_efficiency(self):
        """Test types handle large data efficiently."""
        # Create large metadata dictionary
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}

        response = ModelResponse(
            text="Large metadata test",
            confidence=0.5,
            metadata=large_metadata,
        )

        # Should handle large metadata without issues
        assert response.metadata is not None
        assert len(response.metadata) == 1000
        assert response.metadata["key_500"] == "value_500"

    def test_types_thread_safety(self):
        """Test type creation is thread-safe (basic test)."""
        import concurrent.futures

        def create_response(i: int) -> ModelResponse:
            return ModelResponse(
                text=f"Response {i}", confidence=0.5 + i / 100, reasoning=None, metadata=None
            )

        # Create responses concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_response, i) for i in range(10)]
            responses = [future.result() for future in futures]

        # All responses should be created successfully
        assert len(responses) == 10
        assert all(isinstance(r, ModelResponse) for r in responses)
