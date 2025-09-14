"""
Comprehensive API serialization utilities using Pydantic.

This module demonstrates enterprise-grade API serialization patterns with
proper validation, error handling, and performance optimization.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from pydantic import BaseModel
from pydantic import ValidationError

from nova_retrieval_vlm.types import BatchAnalysisResponse
from nova_retrieval_vlm.types import MetadataDict
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.types import VisionAnalysisRequest
from nova_retrieval_vlm.types import VisionAnalysisResponse


class APISerializer:
    """Enterprise-grade API serialization with comprehensive validation."""

    @staticmethod
    @beartype
    def serialize_response(response: BaseModel) -> str:
        """
        Serialize a Pydantic response model to JSON.

        Args:
            response: Pydantic model instance

        Returns:
            JSON string representation

        Raises:
            ValueError: If serialization fails
        """
        try:
            return response.model_dump_json(
                exclude_none=True,  # Skip None values
                by_alias=True,  # Use field aliases if defined
                indent=2,  # Pretty formatting for debugging
            )
        except Exception as e:
            raise ValueError(f"Failed to serialize response: {e}") from e

    @staticmethod
    @beartype
    def deserialize_request(json_data: str, request_type: type[BaseModel]) -> BaseModel:
        """
        Deserialize JSON to a Pydantic request model with validation.

        Args:
            json_data: JSON string
            request_type: Target Pydantic model class

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If validation fails
            ValueError: If JSON parsing fails
        """
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        try:
            return request_type(**data)
        except ValidationError as e:
            logger.error(f"Validation failed for {request_type.__name__}: {e}")
            raise

    @staticmethod
    @beartype
    def create_vision_request(
        image_path: str | Path,
        task_type: str,
        use_retrieval: bool = False,
        metadata: Union[MetadataDict, None] = None,
    ) -> VisionAnalysisRequest:
        """
        Create a validated vision analysis request.

        Args:
            image_path: Path to image file
            task_type: Analysis task type
            use_retrieval: Whether to use retrieval
            metadata: Additional metadata

        Returns:
            Validated request object
        """
        return VisionAnalysisRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            image_path=image_path,
            task_type=task_type,
            use_retrieval=use_retrieval,
            metadata=metadata or {},
        )

    @staticmethod
    @beartype
    def create_success_response(
        request_id: str,
        analysis_text: str,
        confidence: float,
        processing_time: float,
        generation_log: Union[dict, None] = None,
    ) -> VisionAnalysisResponse:
        """
        Create a successful analysis response.

        Args:
            request_id: Matching request ID
            analysis_text: Generated analysis text
            confidence: Model confidence score
            processing_time: Processing duration
            generation_log: Optional generation metadata

        Returns:
            Validated response object
        """
        analysis_result = ModelResponse(text=analysis_text, confidence=confidence)

        return VisionAnalysisResponse(
            request_id=request_id,
            analysis_result=analysis_result,
            generation_log=generation_log,
            processing_time=processing_time,
        )

    @staticmethod
    @beartype
    def create_error_response(
        request_id: str, error_message: str, processing_time: float = 0.0
    ) -> VisionAnalysisResponse:
        """
        Create an error response.

        Args:
            request_id: Matching request ID
            error_message: Error description
            processing_time: Time before error occurred

        Returns:
            Error response object
        """
        # Create minimal analysis result for error case
        analysis_result = ModelResponse(text="", confidence=0.0)

        return VisionAnalysisResponse(
            request_id=request_id,
            analysis_result=analysis_result,
            processing_time=processing_time,
            error=error_message,
        )


class ResponseValidator:
    """Utilities for validating API responses."""

    @staticmethod
    @beartype
    def validate_analysis_response(response: VisionAnalysisResponse) -> bool:
        """
        Validate that an analysis response is complete and valid.

        Args:
            response: Response to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for required fields
        if not response.request_id:
            logger.warning("Response missing request_id")
            return False

        if response.processing_time < 0:
            logger.warning("Invalid processing time")
            return False

        # Check analysis result
        if response.error is None:
            # Success case - must have meaningful analysis
            if not response.analysis_result.text.strip():
                logger.warning("Empty analysis text in success response")
                return False

            if not (0.0 <= response.analysis_result.confidence <= 1.0):
                logger.warning("Invalid confidence score")
                return False

        return True

    @staticmethod
    @beartype
    def validate_batch_response(response: BatchAnalysisResponse) -> dict[str, Any]:
        """
        Validate a batch response and return quality metrics.

        Args:
            response: Batch response to validate

        Returns:
            Dictionary of validation metrics
        """
        metrics = {
            "total_requests": len(response.results),
            "successful_requests": 0,
            "failed_requests": 0,
            "validation_errors": [],
            "success_rate": 0.0,
        }

        for i, result in enumerate(response.results):
            if ResponseValidator.validate_analysis_response(result):
                if result.error is None:
                    metrics["successful_requests"] += 1
                else:
                    metrics["failed_requests"] += 1
            else:
                metrics["validation_errors"].append(f"Result {i} validation failed")

        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]

        return metrics


@beartype
def demo_api_serialization() -> dict[str, Any]:
    """
    Demonstrate comprehensive API serialization patterns.

    Returns:
        Demo results with serialization examples
    """
    logger.info("🧪 Demonstrating Pydantic API serialization...")

    # Create a vision analysis request
    request = APISerializer.create_vision_request(
        image_path="/path/to/test_image.jpg",
        task_type="diagnosis",
        use_retrieval=True,
        metadata={"patient_id": "P001", "study_date": "2024-01-15"},
    )

    # Serialize to JSON
    request_json = APISerializer.serialize_response(request)
    logger.info(f"✅ Serialized request: {len(request_json)} bytes")

    # Create success response
    success_response = APISerializer.create_success_response(
        request_id=request.request_id,
        analysis_text="No acute abnormalities detected. Normal brain MRI findings.",
        confidence=0.87,
        processing_time=2.34,
        generation_log={"tokens": 42, "cost": 0.01},
    )

    # Serialize response
    response_json = APISerializer.serialize_response(success_response)
    logger.info(f"✅ Serialized success response: {len(response_json)} bytes")

    # Validate response
    is_valid = ResponseValidator.validate_analysis_response(success_response)
    logger.info(f"✅ Response validation: {'PASSED' if is_valid else 'FAILED'}")

    # Create error response
    APISerializer.create_error_response(
        request_id=request.request_id, error_message="Image file not found", processing_time=0.12
    )

    # Test deserialization
    try:
        deserialized_request = APISerializer.deserialize_request(
            request_json, VisionAnalysisRequest
        )
        logger.info(f"✅ Deserialization successful: {deserialized_request.task_type}")
    except ValidationError as e:
        logger.error(f"❌ Deserialization failed: {e}")
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "request_size_bytes": len(request_json),
        "response_size_bytes": len(response_json),
        "validation_passed": is_valid,
        "serialization_format": "JSON with Pydantic validation",
        "features": [
            "Comprehensive field validation",
            "Automatic JSON schema generation",
            "Type-safe serialization/deserialization",
            "Built-in error handling",
            "Performance optimization",
        ],
    }


if __name__ == "__main__":
    # Run demonstration
    result = demo_api_serialization()

    if result["success"]:

        for _feature in result["features"]:
            pass
