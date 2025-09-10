"""Type definitions for the NOVA retrieval VLM project.

This module provides modern type annotations using jaxtyping for tensor shapes
and beartype for runtime validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from pydantic import BaseModel
from pydantic import Field

# Try to import torch and jaxtyping, but make them optional
torch = None
Float = None
Int = None
jaxtyped = None
TORCH_AVAILABLE = False
JAXTYPING_AVAILABLE = False

try:
    import torch
    if torch.__spec__ is not None:  # Check if torch is properly initialized
        TORCH_AVAILABLE = True
except (ImportError, AttributeError, ValueError):
    pass

try:
    from jaxtyping import Float
    from jaxtyping import Int
    from jaxtyping import jaxtyped
    if TORCH_AVAILABLE:  # Only use jaxtyping if torch is available
        JAXTYPING_AVAILABLE = True
except (ImportError, AttributeError):
    pass

# Tensor type aliases with shape annotations (only if torch/jaxtyping available)
if TORCH_AVAILABLE and JAXTYPING_AVAILABLE:
    try:
        ImageTensor = Float[torch.Tensor, "batch channels height width"]
        FeatureTensor = Float[torch.Tensor, "batch features"]
        LogitsTensor = Float[torch.Tensor, "batch classes"]
        BatchIndices = Int[torch.Tensor, "batch"]
    except (TypeError, AttributeError):
        # Fallback if jaxtyping fails with torch
        ImageTensor = Any
        FeatureTensor = Any
        LogitsTensor = Any
        BatchIndices = Any
else:
    # Fallback type aliases when torch/jaxtyping not available
    ImageTensor = Any
    FeatureTensor = Any
    LogitsTensor = Any
    BatchIndices = Any

# NumPy equivalents
if JAXTYPING_AVAILABLE:
    try:
        ImageArray = Float[np.ndarray, "height width channels"]
        FeatureArray = Float[np.ndarray, "features"]
    except (TypeError, AttributeError):
        ImageArray = np.ndarray
        FeatureArray = np.ndarray
else:
    ImageArray = np.ndarray
    FeatureArray = np.ndarray

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


# Decorators for type checking (only if jaxtyping available)
if JAXTYPING_AVAILABLE:
    jax_beartype = jaxtyped(typechecker=beartype)
else:
    jax_beartype = beartype


@jax_beartype
def validate_image_tensor(tensor: ImageTensor) -> ImageTensor:
    """Validate that a tensor has the correct image shape."""
    if TORCH_AVAILABLE and hasattr(tensor, 'dim'):
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (BCHW), got {tensor.dim()}D")
    return tensor


@jax_beartype
def validate_feature_tensor(tensor: FeatureTensor) -> FeatureTensor:
    """Validate that a tensor has the correct feature shape."""
    if TORCH_AVAILABLE and hasattr(tensor, 'dim'):
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor (batch, features), got {tensor.dim()}D")
    return tensor


class JSONParseError(Exception):
    """Raised when JSON parsing fails definitively."""

    def __init__(self, original_content: str, error: str) -> None:
        self.original_content = original_content
        self.error = error
        super().__init__(f"Failed to parse JSON: {error}")


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
            cleaned = cleaned[len(prefix):].strip()
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
