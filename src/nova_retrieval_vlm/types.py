"""Type definitions for the NOVA retrieval VLM project.

Provides tensor type annotations with jaxtyping and Pydantic models for
structured validation. All dependencies are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import UInt8
from jaxtyping import jaxtyped
from pydantic import BaseModel
from pydantic import ConfigDict
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
# Note: jaxtyping uses np.ndarray directly, shape checking is via the string annotation
# Using npt.NDArray for pyright compatibility
ImageArray = Float[npt.NDArray[np.floating[Any]], "height width channels"]
GrayscaleArray = Float[npt.NDArray[np.floating[Any]], "height width"]
FeatureArray = Float[npt.NDArray[np.floating[Any]], "features"]
MaskArray = Bool[npt.NDArray[np.bool_], "height width"]
IntensityArray = UInt8[npt.NDArray[np.uint8], "height width"]


def tensor_validated(func: Any) -> Any:
    """Decorator combining jaxtyped shape validation with beartype runtime checks."""
    return jaxtyped(typechecker=beartype)(func)


# Path types
ImagePath = str | Path
ModelPath = str | Path

# Core data structures
MetadataDict = dict[str, Any]

# Medical imaging specific
DicomMetadata = dict[str, str | int | float]
NiftiMetadata = dict[str, str | int | float | list[float]]


class EvaluationMetrics(BaseModel):
    """Base evaluation metrics - prefer task-specific metrics classes."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float | None = Field(None, ge=0.0, le=1.0)
    recall: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(None, ge=0.0, le=1.0)


class CaptionMetrics(BaseModel):
    """Evaluation metrics for caption generation tasks.

    All metrics are normalized to 0-1 range.
    """

    bleu: float = Field(ge=0.0, le=1.0, description="BLEU score (primary metric)")
    bert_f1: float = Field(ge=0.0, le=1.0, description="BERTScore F1")
    meteor: float = Field(ge=0.0, le=1.0, description="METEOR score")
    modality_f1: float = Field(ge=0.0, le=1.0, description="Modality keyword F1")
    clinical_f1: float = Field(ge=0.0, le=1.0, description="Clinical keyword F1")
    binary_accuracy: float = Field(ge=0.0, le=1.0, description="Normal/abnormal accuracy")
    binary_f1: float = Field(ge=0.0, le=1.0, description="Normal/abnormal F1")
    radgraph_f1: float | None = Field(
        None, ge=0.0, le=1.0, description="RadGraph F1 (optional dependency)"
    )


class DetectionMetrics(BaseModel):
    """Evaluation metrics for object detection/localization tasks.

    Follows NOVA benchmark protocol with multiple IoU thresholds.
    """

    map30: float = Field(ge=0.0, le=1.0, description="mAP at IoU threshold 0.3")
    map50: float = Field(ge=0.0, le=1.0, description="mAP at IoU threshold 0.5 (primary)")
    map50_95: float = Field(ge=0.0, le=1.0, description="mAP averaged across IoU 0.5-0.95")
    acc50: float = Field(ge=0.0, le=1.0, description="Detection accuracy at IoU 0.5")
    tp30: int = Field(ge=0, description="True positives at IoU 0.3")
    fp30: int = Field(ge=0, description="False positives at IoU 0.3")


class DiagnosisMetrics(BaseModel):
    """Evaluation metrics for diagnosis classification tasks.

    Follows NOVA benchmark protocol with LLM semantic matching.
    """

    top1: float = Field(ge=0.0, le=1.0, description="Top-1 accuracy (primary)")
    top5: float = Field(ge=0.0, le=1.0, description="Top-5 accuracy")
    coverage: float = Field(ge=0.0, description="Unique predictions / unique references")
    entropy: float = Field(ge=0.0, description="Shannon entropy of prediction distribution")


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

    model_config = ConfigDict(ser_json_bytes="utf8")

    request_id: str = Field(description="Unique identifier for the request")
    timestamp: float = Field(description="Request timestamp")


class VisionAnalysisRequest(APIRequest):
    """Request for vision-language model analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_123456",
                "timestamp": 1609459200.0,
                "image_path": "/path/to/image.jpg",
                "task_type": "diagnosis",
                "system_prompt": None,
                "metadata": {"patient_id": "P001"},
            }
        }
    )

    image_path: ImagePath = Field(description="Path to the image file")
    task_type: str = Field(
        description="Type of analysis task", pattern="^(localization|caption|diagnosis|detection)$"
    )
    system_prompt: str | None = Field(None, description="Optional system prompt override")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional metadata")


class VisionAnalysisResponse(BaseModel):
    """Response from vision-language model analysis."""

    request_id: str = Field(description="Matching request identifier")
    analysis_result: ModelResponse = Field(description="Main analysis result")
    generation_log: dict[str, Any] | None = Field(None, description="Generation metadata")
    processing_time: float = Field(ge=0.0, description="Total processing time in seconds")
    error: str | None = Field(None, description="Error message if analysis failed")


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
    batch_summary: dict[str, Any] = Field(description="Summary statistics for the batch")
    total_processing_time: float = Field(ge=0.0, description="Total batch processing time")

    def success_rate(self) -> float:
        """Calculate success rate of the batch."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.error is None)
        return successful / len(self.results)


class JSONParseError(Exception):
    """Raised when JSON parsing fails definitively."""

    def __init__(
        self, original_content: str, error: str, attempts: int | None = None
    ) -> None:
        self.original_content = original_content
        self.error = error
        self.attempts = attempts
        msg = f"Failed to parse JSON: {error}"
        if attempts is not None:
            msg += f" (after {attempts} attempts)"
        super().__init__(msg)


class ModelError(Exception):
    """Base exception for model-related errors."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.model_name = model_name


class APIError(ModelError):
    """Raised when API calls fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message, model_name)
        self.status_code = status_code
        self.response_body = response_body


class ValidationError(ModelError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        invalid_value: object = None,
    ) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.invalid_value = invalid_value


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None) -> None:
        super().__init__(message)
        self.config_key = config_key


@beartype
def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text.

    Handles:
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    - Single backtick wrapping

    Args:
        text: Text potentially wrapped in markdown fences

    Returns:
        Text with fences removed
    """
    cleaned = text.strip()

    # Remove triple-backtick fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    # Handle single backticks
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1]

    return cleaned.strip()


# Common prefixes that models add before JSON output
JSON_PREFIXES = frozenset(["answer:", "json:", "result:", "here is the json:", "output:"])


@beartype
def strip_json_prefixes(text: str) -> str:
    """Remove common prefixes that models add before JSON.

    Args:
        text: Text potentially starting with a prefix

    Returns:
        Text with prefix removed
    """
    cleaned_lower = text.lower().strip()
    for prefix in JSON_PREFIXES:
        if cleaned_lower.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


@beartype
def extract_json_object(text: str) -> str:
    """Extract JSON object from text that may contain other content.

    Finds the first '{' and extracts from there.

    Args:
        text: Text containing a JSON object

    Returns:
        Extracted JSON string starting from first '{'

    Raises:
        JSONParseError: If no JSON object found in text
    """
    text = text.strip()
    if text.startswith("{"):
        return text

    json_start = text.find("{")
    if json_start == -1:
        raise JSONParseError(text, "No JSON object found in response")

    return text[json_start:]


@beartype
def parse_json_response(payload: str) -> dict[str, Any]:
    """Parse JSON response from model with explicit preprocessing steps.

    Preprocessing pipeline:
    1. Strip markdown fences (```json, ```, single backticks)
    2. Strip common prefixes ("answer:", "json:", etc.)
    3. Extract JSON object (find first '{')
    4. Parse JSON

    Args:
        payload: Raw response string from model

    Returns:
        Parsed dictionary

    Raises:
        JSONParseError: If parsing fails after preprocessing
    """
    # Explicit preprocessing pipeline
    cleaned = strip_markdown_fences(payload)
    cleaned = strip_json_prefixes(cleaned)
    cleaned = extract_json_object(cleaned)

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise JSONParseError(payload, f"Expected dict, got {type(result).__name__}")
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
