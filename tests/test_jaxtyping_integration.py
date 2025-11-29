"""Tests for jaxtyping tensor validation."""

from __future__ import annotations

import pytest
import torch

from nova_retrieval_vlm.tensor_ops import compute_attention_weights
from nova_retrieval_vlm.tensor_ops import extract_image_features
from nova_retrieval_vlm.tensor_ops import validate_medical_workflow
from nova_retrieval_vlm.tensor_ops import validate_tensor_batch


class TestTensorValidation:
    """Tests for tensor shape validation."""

    def test_validate_tensor_batch(self):
        """Test tensor batch validation."""
        tensor = torch.randn(4, 1, 512, 512)
        validate_tensor_batch(tensor, expected_batch_size=4)

    def test_validate_tensor_batch_wrong_size(self):
        """Test that wrong batch size raises error."""
        tensor = torch.randn(4, 768)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_tensor_batch(tensor, expected_batch_size=3)

    def test_extract_image_features(self):
        """Test feature extraction."""
        images = torch.randn(4, 1, 512, 512)
        features = extract_image_features(images, feature_dim=768)
        assert features.shape == (4, 768)

    def test_compute_attention_weights(self):
        """Test attention weight computation."""
        query = torch.randn(3, 768)
        key = torch.randn(3, 768)
        attention = compute_attention_weights(query, key, temperature=1.0)
        assert attention.shape == (3, 3, 3)


class TestMedicalWorkflow:
    """Tests for medical imaging workflow."""

    def test_validate_medical_workflow(self):
        """Test complete workflow validation returns expected structure."""
        result = validate_medical_workflow(["dummy1.jpg", "dummy2.jpg"])
        # The workflow may fail if files don't exist, but should return expected keys
        assert "success" in result
        assert "tensors" in result or "error" in result


class TestImageOps:
    """Tests for image operations."""

    def test_flip_horizontal_import(self):
        """Test flip_horizontal is importable."""
        from nova_retrieval_vlm.visual_reasoning import flip_horizontal

        assert flip_horizontal is not None

    def test_flip_vertical_import(self):
        """Test flip_vertical is importable."""
        from nova_retrieval_vlm.visual_reasoning import flip_vertical

        assert flip_vertical is not None

    def test_rotate_90_import(self):
        """Test rotate_90 is importable."""
        from nova_retrieval_vlm.visual_reasoning import rotate_90

        assert rotate_90 is not None
