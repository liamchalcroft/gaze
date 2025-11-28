"""Type definitions for the NOVA retrieval VLM project.

Provides tensor type annotations with jaxtyping and Pydantic models for
structured validation. All dependencies are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import UInt8
from jaxtyping import jaxtyped
from pydantic import BaseModel
from pydantic import Field

# Vision tensor types with jaxtyping shape specifications
ImageTensor = Float[torch.Tensor, "batch channels height width"]
ImageTensorSingle = Float[torch.Tensor, "channels height width"]
GrayscaleImage = Float[torch.Tensor, "height width"]

FeatureTensor = Float[torch.Tensor, "batch features"]
EmbeddingTensor = Float[torch.Tensor, "batch embed_dim"]
AttentionWeights = Float[torch.Tensor, "batch seq_len seq_len"]

LogitsTensor = Float[torch.Tensor, "batch classes"]
ProbabilityTensor = Float[torch.Tensor, "batch classes"]
ConfidenceScores = Float[torch.Tensor, "batch"]

BatchIndices = Int[torch.Tensor, "batch"]
SequenceTensor = Int[torch.Tensor, "batch seq_len"]
TokenIds = Int[torch.Tensor, "seq_len"]

BoundingBoxes = Float[torch.Tensor, "num_boxes 4"]
BatchedBoundingBoxes = Float[torch.Tensor, "batch max_boxes 4"]

MedicalImageBatch = Float[torch.Tensor, "batch 1 height width"]
MultiModalFeatures = Float[torch.Tensor, "batch modalities features"]

# NumPy array types for image processing
ImageArray = Float[np.ndarray[Any, Any], "height width channels"]
GrayscaleArray = Float[np.ndarray[Any, Any], "height width"]
FeatureArray = Float[np.ndarray[Any, Any], "features"]
MaskArray = Bool[np.ndarray[Any, Any], "height width"]
IntensityArray = UInt8[np.ndarray[Any, Any], "height width"]


def tensor_validated(func: Any) -> Any:
    """Decorator combining jaxtyped shape validation with beartype runtime checks."""
    return jaxtyped(typechecker=beartype)(func)


# Path types
ImagePath = str | Path
ModelPath = str | Path

# Core data structures
MetadataDict = dict[str, Any]
ConfigDict = dict[str, Any]

# Medical imaging specific
DicomMetadata = dict[str, str | int | float]
NiftiMetadata = dict[str, str | int | float | list[float]]


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for model performance."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float | None = Field(None, ge=0.0, le=1.0)
    recall: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(None, ge=0.0, le=1.0)


class ModelResponse(BaseModel):
    """Structured response from a vision-language model."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None
    metadata: MetadataDict | None = None


class RetrievalResult(BaseModel):
    """Result from document retrieval."""

    text: str
    score: float = Field(ge=0.0)
    source: str
    reasoning_type: str | None = None
    metadata: MetadataDict | None = None


class BatchData(BaseModel):
    """Batch of data for processing."""

    images: list[ImagePath]
    metadata: list[MetadataDict]
    labels: list[str] | None = None


class APIRequest(BaseModel):
    """Base class for API requests with comprehensive validation."""

    request_id: str = Field(description="Unique identifier for the request")
    timestamp: float = Field(description="Request timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {Path: str}


class VisionAnalysisRequest(APIRequest):
    """Request for vision-language model analysis."""

    image_path: ImagePath = Field(description="Path to the image file")
    task_type: str = Field(
        description="Type of analysis task", pattern="^(localization|caption|diagnosis|detection)$"
    )
    system_prompt: str | None = Field(None, description="Optional system prompt override")
    use_retrieval: bool = Field(False, description="Whether to use retrieval augmentation")
    retrieval_passages: list[str] = Field(
        default_factory=list, description="Pre-retrieved passages"
    )
    metadata: MetadataDict = Field(default_factory=dict, description="Additional metadata")

    class Config(APIRequest.Config):
        """Enhanced configuration."""

        json_schema_extra = {
            "example": {
                "request_id": "req_123456",
                "timestamp": 1609459200.0,
                "image_path": "/path/to/image.jpg",
                "task_type": "diagnosis",
                "system_prompt": None,
                "use_retrieval": True,
                "retrieval_passages": ["Medical context passage..."],
                "metadata": {"patient_id": "P001"},
            }
        }


class VisionAnalysisResponse(BaseModel):
    """Response from vision-language model analysis."""

    request_id: str = Field(description="Matching request identifier")
    analysis_result: ModelResponse = Field(description="Main analysis result")
    generation_log: dict[str, Any] | None = Field(None, description="Generation metadata")
    processing_time: float = Field(ge=0.0, description="Total processing time in seconds")
    error: str | None = Field(None, description="Error message if analysis failed")

    class Config:
        """Pydantic configuration."""

        json_encoders = {Path: str}


class BatchAnalysisRequest(APIRequest):
    """Request for batch analysis of multiple images."""

    batch_data: BatchData = Field(description="Batch of images and metadata")
    task_type: str = Field(description="Type of analysis task")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum parallel workers")


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis."""

    request_id: str = Field(description="Matching request identifier")
    results: list[VisionAnalysisResponse] = Field(description="Individual analysis results")
    batch_summary: dict = Field(description="Summary statistics for the batch")
    total_processing_time: float = Field(ge=0.0, description="Total batch processing time")

    def success_rate(self) -> float:
        """Calculate success rate of the batch."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.error is None)
        return successful / len(self.results)


class JSONParseError(Exception):
    """Raised when JSON parsing fails definitively."""

    def __init__(self, original_content: str, error: str) -> None:
        self.original_content = original_content
        self.error = error
        super().__init__(f"Failed to parse JSON: {error}")


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class APIError(ModelError):
    """Raised when API calls fail."""

    pass


class ValidationError(ModelError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


@beartype
def parse_json_response(payload: str) -> dict[str, Any]:
    """Parse JSON response from model with minimal preprocessing.

    Args:
        payload: Raw response string from model

    Returns:
        Parsed dictionary

    Raises:
        JSONParseError: If parsing fails after basic cleanup
    """
    # Minimal cleanup - remove markdown fences and common prefixes
    cleaned = payload.strip()

    # Remove markdown fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    # Handle single backticks
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1]

    # Remove common prefixes
    prefixes = ["answer:", "json:", "result:", "here is the json:", "output:"]
    cleaned_lower = cleaned.lower().strip()
    for prefix in prefixes:
        if cleaned_lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    # Try to extract JSON from text that contains other content
    cleaned = cleaned.strip()
    if not cleaned.startswith("{") and "{" in cleaned:
        # Find the first { and extract from there
        json_start = cleaned.find("{")
        cleaned = cleaned[json_start:]

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise JSONParseError(payload, f"Expected dict, got {type(result)}")
        return result
    except json.JSONDecodeError as e:
        raise JSONParseError(payload, f"JSON decode error: {e}") from e


@beartype
def validate_image_path(path: ImagePath) -> Path:
    """Validate and convert image path to Path object.

    Args:
        path: Image path as string or Path

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValidationError(f"Image path does not exist: {path}")
    if not path_obj.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    return path_obj


@beartype
def validate_batch_data(batch: BatchData) -> BatchData:
    """Validate batch data structure.

    Args:
        batch: Batch data to validate

    Returns:
        Validated batch data

    Raises:
        ValidationError: If batch data is invalid
    """
    if len(batch.images) != len(batch.metadata):
        raise ValidationError(
            f"Images and metadata length mismatch: {len(batch.images)} vs {len(batch.metadata)}"
        )

    if batch.labels is not None and len(batch.labels) != len(batch.images):
        raise ValidationError(
            f"Labels and images length mismatch: {len(batch.labels)} vs {len(batch.images)}"
        )

    return batch
