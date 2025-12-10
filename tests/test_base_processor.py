"""Tests for base AgenticProcessorBase functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness import AgenticProcessorBase


@pytest.fixture
def temp_image_path(tmp_path: Path) -> Path:
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path)
    return image_path


@pytest.fixture
def temp_image_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create multiple temporary image files for testing."""
    path1 = tmp_path / "test1.jpg"
    path2 = tmp_path / "test2.png"
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    img1.save(path1)
    img2.save(path2)
    return path1, path2


class TestAgenticProcessorBase:
    """Test core agentic processor functionality."""

    def test_normalize_image_inputs_single_image(self, temp_image_path: Path):
        """Test normalizing single image input."""
        processor = _TestProcessor()
        result = processor._normalize_image_inputs(temp_image_path, None)
        assert len(result) == 1
        assert result[0].path == temp_image_path
        assert result[0].label is None

    def test_normalize_image_inputs_single_image_with_label(self, temp_image_path: Path):
        """Test normalizing single image with label."""
        processor = _TestProcessor()
        result = processor._normalize_image_inputs(temp_image_path, ["T1-weighted"])
        assert len(result) == 1
        assert result[0].path == temp_image_path
        assert result[0].label == "T1-weighted"

    def test_normalize_image_inputs_multiple_images(self, temp_image_paths: tuple[Path, Path]):
        """Test normalizing multiple image inputs."""
        processor = _TestProcessor()
        path1, path2 = temp_image_paths
        result = processor._normalize_image_inputs(
            [path1, path2], ["pre-contrast", "post-contrast"]
        )
        assert len(result) == 2
        assert result[0].path == path1
        assert result[0].label == "pre-contrast"
        assert result[1].path == path2
        assert result[1].label == "post-contrast"

    def test_normalize_image_inputs_label_mismatch(self, temp_image_paths: tuple[Path, Path]):
        """Test error when label count doesn't match image count."""
        processor = _TestProcessor()
        path1, path2 = temp_image_paths
        with pytest.raises(ValueError, match="Number of labels .* must match"):
            processor._normalize_image_inputs([path1, path2], ["only one label"])

    def test_normalize_image_inputs_file_not_found(self, tmp_path: Path):
        """Test error when image file doesn't exist."""
        processor = _TestProcessor()
        non_existent_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            processor._normalize_image_inputs(non_existent_path, None)

    def test_normalize_image_inputs_invalid_format(self, tmp_path: Path):
        """Test error with unsupported image format."""
        processor = _TestProcessor()
        invalid_path = tmp_path / "test.txt"
        invalid_path.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported image format"):
            processor._normalize_image_inputs(invalid_path, None)


class _TestProcessor(AgenticProcessorBase):
    """Test implementation of AgenticProcessorBase."""

    def get_system_prompt(self, images, metadata):  # noqa: ARG002
        return "Test system prompt"

    def get_user_message(self, images, metadata):  # noqa: ARG002
        return "Test user message"

    def get_response_schema(self):
        return {"type": "object", "properties": {"continue": {"type": "boolean"}}}

    def validate_response(self, response):
        return "continue" in response and isinstance(response["continue"], bool)
