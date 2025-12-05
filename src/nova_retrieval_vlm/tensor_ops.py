"""Tensor operations with jaxtyping + beartype validation for medical imaging."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from nova_retrieval_vlm.types import AttentionWeights
from nova_retrieval_vlm.types import BoundingBoxes
from nova_retrieval_vlm.types import FeatureTensor
from nova_retrieval_vlm.types import ImageArray
from nova_retrieval_vlm.types import ImageTensor
from nova_retrieval_vlm.types import MedicalImageBatch
from nova_retrieval_vlm.types import tensor_validated

# Module-level caches for neural network components
_feature_backbone: torch.nn.Module | None = None
_projection_cache: dict[tuple[int, int, str], torch.nn.Module] = {}
_detection_heads_cache: dict[tuple[int, int], tuple[torch.nn.Module, torch.nn.Module]] = {}

# Constants for image processing
RGB_TO_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_feature_backbone(device: torch.device | None = None) -> torch.nn.Module:
    """Get cached feature extraction backbone (ResNet18 without classifier).

    Args:
        device: Target device for the backbone. If None, uses CPU.

    Returns:
        Cached backbone module on the specified device.
    """
    global _feature_backbone
    if _feature_backbone is None:
        from torchvision import models

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final classifier layer to get features
        _feature_backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        _feature_backbone.eval()
        _feature_backbone.requires_grad_(False)

    # Move to device if specified and not already there
    if device is not None:
        _feature_backbone = _feature_backbone.to(device)

    return _feature_backbone


def _get_cached_projection(
    input_dim: int, output_dim: int, device: torch.device
) -> torch.nn.Module:
    """Get cached projection layer for feature dimension mapping.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        device: Target device

    Returns:
        Cached projection layer on the specified device.
    """
    cache_key = (input_dim, output_dim, str(device))
    if cache_key not in _projection_cache:
        projection = torch.nn.Linear(input_dim, output_dim, device=device)
        projection.eval()
        projection.requires_grad_(False)
        _projection_cache[cache_key] = projection
    return _projection_cache[cache_key]


def _get_cached_detection_heads(
    feature_dim: int, max_detections: int
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Get cached detection heads for bounding box prediction.

    Args:
        feature_dim: Input feature dimension
        max_detections: Maximum number of detections

    Returns:
        Tuple of (detection_head, objectness_head) modules.
    """
    cache_key = (feature_dim, max_detections)
    if cache_key not in _detection_heads_cache:
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, max_detections * 4),
        )
        detection_head.eval()
        detection_head.requires_grad_(False)

        objectness_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, max_detections),
            torch.nn.Sigmoid(),
        )
        objectness_head.eval()
        objectness_head.requires_grad_(False)

        _detection_heads_cache[cache_key] = (detection_head, objectness_head)

    return _detection_heads_cache[cache_key]


def clear_tensor_caches() -> None:
    """Clear all cached neural network components to free memory."""
    global _feature_backbone, _projection_cache, _detection_heads_cache
    _feature_backbone = None
    _projection_cache.clear()
    _detection_heads_cache.clear()


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
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using standard weights
        gray_image = np.dot(image[..., :3], RGB_TO_GRAY_WEIGHTS)
    else:
        gray_image = image.squeeze() if len(image.shape) == 3 else image

    # Resize to target size
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
    from torchvision import transforms

    # Verify expected shape (batch, channels, height, width)
    if len(images.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape {images.shape}")

    # Get device and cached backbone
    device = images.device
    backbone = _get_feature_backbone(device)

    # Convert single-channel medical images to RGB for pretrained network
    if images.shape[1] == 1:  # grayscale to RGB
        images = images.repeat(1, 3, 1, 1)

    # Normalize for ImageNet pretrained weights
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    images = normalize(images)

    # Extract features using backbone
    with torch.no_grad():
        # Squeeze only the spatial dimensions (last two), keep batch and channel
        raw_features = backbone(images).squeeze(dim=-1).squeeze(dim=-1)  # (batch, 512)

    # Project to desired feature dimension if different (using cached layer)
    if raw_features.shape[-1] != feature_dim:
        projection = _get_cached_projection(raw_features.shape[-1], feature_dim, device)
        features = projection(raw_features)
    else:
        features = raw_features

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
    _, feature_dim = image_features.shape
    device = image_features.device

    # Get cached detection heads
    detection_head, objectness_head = _get_cached_detection_heads(feature_dim, max_detections)
    detection_head = detection_head.to(device)
    objectness_head = objectness_head.to(device)

    with torch.no_grad():
        # Process first image features
        first_image_features = image_features[0:1]  # (1, feature_dim)

        # Predict bounding box coordinates
        bbox_logits = detection_head(first_image_features)  # (1, max_detections * 4)
        bbox_coords = torch.sigmoid(bbox_logits.view(max_detections, 4))  # (max_det, 4)

        # Predict objectness scores
        objectness_scores = objectness_head(first_image_features).squeeze()  # (max_detections,)

        # Filter detections by confidence threshold (> 0.5)
        valid_detections = objectness_scores > 0.5
        bbox_coords = bbox_coords[valid_detections] if valid_detections.any() else torch.zeros(0, 4)

    return bbox_coords


@tensor_validated
def validate_tensor_batch(
    tensor_batch: ImageTensor | FeatureTensor, expected_batch_size: int
) -> bool:
    """Validate that tensor batch has expected size.

    Args:
        tensor_batch: Input tensor batch
        expected_batch_size: Expected batch size

    Returns:
        True if validation passes

    Raises:
        ValueError: If batch size doesn't match
    """
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
    # Load and preprocess images
    processed_images = []
    for image_path in image_paths[:batch_size]:
        try:
            pil_image = PILImage.open(image_path).convert("RGB")
            image_array = np.array(pil_image).astype(np.float32) / 255.0

            # Preprocess with shape validation
            processed_image = preprocess_medical_image(image_array)
            processed_images.append(processed_image)

        except FileNotFoundError as e:
            raise ValueError(f"Image file not found: {image_path}") from e
        except UnidentifiedImageError as e:
            raise ValueError(f"Cannot identify image file: {image_path}") from e

    if not processed_images:
        raise ValueError("No images could be processed from provided paths")
    image_batch = torch.cat(processed_images, dim=0)

    # Extract features with shape validation
    features = extract_image_features(image_batch, feature_dim)

    # Predict bounding boxes with shape validation
    bounding_boxes = predict_bounding_boxes(features)

    # Validate final batch dimensions
    validate_tensor_batch(image_batch, batch_size)
    validate_tensor_batch(features, batch_size)

    return image_batch, features, bounding_boxes


# Utility functions for tensor shape inspection
@beartype
def get_tensor_info(tensor: torch.Tensor) -> dict[str, Any]:
    """Get tensor information.

    Args:
        tensor: PyTorch tensor to inspect

    Returns:
        Dictionary with tensor metadata (shape, dtype, device, memory usage)
    """
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024),
    }


@beartype
def validate_medical_workflow(image_paths: list[str]) -> dict[str, Any]:
    """Validate medical imaging workflow with reporting.

    Args:
        image_paths: List of paths to medical images

    Returns:
        Validation report with tensor shapes and memory usage

    Raises:
        ValueError: If no images provided or image processing fails
        FileNotFoundError: If image files don't exist
    """
    if not image_paths:
        raise ValueError("No image paths provided for validation")

    images, features, boxes = process_medical_batch(image_paths, batch_size=2)
    return {
        "success": True,
        "tensors": {
            "medical_images": get_tensor_info(images),
            "extracted_features": get_tensor_info(features),
            "bounding_boxes": get_tensor_info(boxes),
        },
        "validation_passed": True,
    }
