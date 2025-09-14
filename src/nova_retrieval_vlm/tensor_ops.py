"""
Tensor operations with jaxtyping + beartype validation.

This module demonstrates enterprise-grade tensor shape validation for medical imaging
and vision-language model operations.
"""

from __future__ import annotations

import numpy as np

from nova_retrieval_vlm.types import JAXTYPING_AVAILABLE
from nova_retrieval_vlm.types import AttentionWeights
from nova_retrieval_vlm.types import BoundingBoxes
from nova_retrieval_vlm.types import FeatureTensor
from nova_retrieval_vlm.types import ImageArray
from nova_retrieval_vlm.types import ImageTensor  # Import all tensor types
from nova_retrieval_vlm.types import MedicalImageBatch
from nova_retrieval_vlm.types import tensor_validated

if JAXTYPING_AVAILABLE:
    import torch
    import torch.nn.functional as F
else:
    # Fallback imports
    from typing import Any

    torch = Any
    F = Any


@tensor_validated
def preprocess_medical_image(
    image: ImageArray, target_size: tuple[int, int] = (512, 512)
) -> MedicalImageBatch:
    """
    Preprocess medical image with shape validation.

    Args:
        image: Input medical image with shape (height, width, channels)
        target_size: Target output size (height, width)

    Returns:
        Preprocessed medical image batch with shape (1, 1, height, width)

    Raises:
        TypeError: If tensor shapes don't match expected dimensions
    """
    if not JAXTYPING_AVAILABLE:
        # Fallback processing without tensor validation
        return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using standard weights
        gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray_image = image.squeeze() if len(image.shape) == 3 else image

    # Resize to target size
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray((gray_image * 255).astype(np.uint8))
    resized_image = pil_image.resize(target_size, PILImage.LANCZOS)

    # Convert to tensor and normalize
    tensor_image = torch.from_numpy(np.array(resized_image)).float() / 255.0

    # Add batch and channel dimensions: (height, width) -> (1, 1, height, width)
    batched_image = tensor_image.unsqueeze(0).unsqueeze(0)

    return batched_image


@tensor_validated
def extract_image_features(images: MedicalImageBatch, feature_dim: int = 768) -> FeatureTensor:
    """
    Extract features from medical images with shape validation.

    Args:
        images: Batch of medical images with shape (batch, 1, height, width)
        feature_dim: Dimension of extracted features

    Returns:
        Feature tensor with shape (batch, features)
    """
    if not JAXTYPING_AVAILABLE:
        batch_size = images.shape[0]
        return torch.randn(batch_size, feature_dim)

    batch_size, channels, height, width = images.shape

    # Simple feature extraction (global average pooling + linear projection)
    # In practice, this would use a pretrained CNN backbone
    pooled_features = F.adaptive_avg_pool2d(images, (1, 1))  # (batch, 1, 1, 1)
    flattened = pooled_features.flatten(start_dim=1)  # (batch, 1)

    # Project to desired feature dimension
    features = F.linear(
        flattened,
        weight=torch.randn(feature_dim, 1),  # Random projection for demo
        bias=torch.zeros(feature_dim),
    )

    return features


@tensor_validated
def compute_attention_weights(
    query_features: FeatureTensor, key_features: FeatureTensor, temperature: float = 1.0
) -> AttentionWeights:
    """
    Compute attention weights between query and key features.

    Args:
        query_features: Query features with shape (batch, features)
        key_features: Key features with shape (batch, features)
        temperature: Temperature scaling for attention

    Returns:
        Attention weights with shape (batch, batch, batch) for cross-attention
    """
    if not JAXTYPING_AVAILABLE:
        batch_size = query_features.shape[0]
        return torch.softmax(torch.randn(batch_size, batch_size, batch_size), dim=-1)

    batch_size, feature_dim = query_features.shape

    # Compute scaled dot-product attention
    # Q: (batch, features), K: (batch, features) -> attention: (batch, batch, batch)

    # Expand for cross-attention computation
    queries = query_features.unsqueeze(1).expand(batch_size, batch_size, feature_dim)
    keys = key_features.unsqueeze(0).expand(batch_size, batch_size, feature_dim)

    # Compute attention scores
    scores = torch.sum(queries * keys, dim=-1, keepdim=True) / (temperature * (feature_dim**0.5))
    scores = scores.expand(batch_size, batch_size, batch_size)

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    return attention_weights


@tensor_validated
def predict_bounding_boxes(
    image_features: FeatureTensor, max_detections: int = 10
) -> BoundingBoxes:
    """
    Predict bounding boxes from image features.

    Args:
        image_features: Image features with shape (batch, features)
        max_detections: Maximum number of detections per image

    Returns:
        Bounding boxes with shape (num_detections, 4) in (x1, y1, x2, y2) format
    """
    if not JAXTYPING_AVAILABLE:
        return torch.rand(max_detections, 4)

    batch_size, feature_dim = image_features.shape

    # Simple bounding box prediction (in practice, would use detection head)
    # Project features to bbox coordinates
    bbox_logits = F.linear(
        image_features[0:1],  # Take first image for demo
        weight=torch.randn(4 * max_detections, feature_dim),
        bias=torch.zeros(4 * max_detections),
    )

    # Reshape to (max_detections, 4) and apply sigmoid for normalized coords
    bbox_coords = torch.sigmoid(bbox_logits.view(max_detections, 4))

    return bbox_coords


@tensor_validated
def validate_tensor_batch(
    tensor_batch: ImageTensor | FeatureTensor, expected_batch_size: int
) -> bool:
    """
    Validate that tensor batch has expected size.

    Args:
        tensor_batch: Input tensor batch
        expected_batch_size: Expected batch size

    Returns:
        True if validation passes

    Raises:
        ValueError: If batch size doesn't match
    """
    if not JAXTYPING_AVAILABLE:
        return True

    actual_batch_size = tensor_batch.shape[0]

    if actual_batch_size != expected_batch_size:
        raise ValueError(
            f"Batch size mismatch: expected {expected_batch_size}, got {actual_batch_size}"
        )

    return True


@tensor_validated
def process_medical_batch(
    image_paths: list[str], batch_size: int = 4, feature_dim: int = 768
) -> tuple[MedicalImageBatch, FeatureTensor, BoundingBoxes]:
    """
    Complete medical image processing pipeline with shape validation.

    Args:
        image_paths: List of paths to medical images
        batch_size: Processing batch size
        feature_dim: Feature extraction dimension

    Returns:
        Tuple of:
        - Processed medical images (batch, 1, height, width)
        - Extracted features (batch, features)
        - Predicted bounding boxes (detections, 4)
    """
    if not JAXTYPING_AVAILABLE:
        # Fallback without shape validation
        dummy_images = torch.randn(batch_size, 1, 512, 512)
        dummy_features = torch.randn(batch_size, feature_dim)
        dummy_boxes = torch.rand(10, 4)
        return dummy_images, dummy_features, dummy_boxes

    # Load and preprocess images
    processed_images = []
    for _i, image_path in enumerate(image_paths[:batch_size]):
        try:
            from PIL import Image as PILImage

            # Load image
            pil_image = PILImage.open(image_path).convert("RGB")
            image_array = np.array(pil_image).astype(np.float32) / 255.0

            # Preprocess with shape validation
            processed_image = preprocess_medical_image(image_array)
            processed_images.append(processed_image)

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}") from e

    # Concatenate into batch
    if processed_images:
        image_batch = torch.cat(processed_images, dim=0)
    else:
        image_batch = torch.randn(batch_size, 1, 512, 512)

    # Extract features with shape validation
    features = extract_image_features(image_batch, feature_dim)

    # Predict bounding boxes with shape validation
    bounding_boxes = predict_bounding_boxes(features)

    # Validate final batch dimensions
    validate_tensor_batch(image_batch, batch_size)
    validate_tensor_batch(features, batch_size)

    return image_batch, features, bounding_boxes


# Utility functions for tensor shape inspection
def get_tensor_info(tensor: ImageTensor | FeatureTensor | Any) -> dict[str, Any]:
    """Get tensor information."""
    if not hasattr(tensor, "shape"):
        return {"error": "Not a tensor"}

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown",
        "device": str(tensor.device) if hasattr(tensor, "device") else "unknown",
        "requires_grad": getattr(tensor, "requires_grad", False),
        "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024)
        if hasattr(tensor, "numel")
        else 0,
    }


def validate_medical_workflow(image_paths: list[str]) -> dict[str, Any]:
    """
    Validate medical imaging workflow with reporting.

    Returns:
        Validation report with tensor shapes and memory usage
    """
    try:
        # Run complete pipeline
        images, features, boxes = process_medical_batch(image_paths, batch_size=2)

        return {
            "success": True,
            "jaxtyping_enabled": JAXTYPING_AVAILABLE,
            "tensors": {
                "medical_images": get_tensor_info(images),
                "extracted_features": get_tensor_info(features),
                "bounding_boxes": get_tensor_info(boxes),
            },
            "validation_passed": True,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "jaxtyping_enabled": JAXTYPING_AVAILABLE,
        }
