"""Tests for base AgenticProcessorBase functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from radiant_harness import AgenticProcessorBase
from radiant_harness.base import ImageInput


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

    def test_normalize_image_inputs_single_image_label_mismatch(self, temp_image_path: Path):
        """Test error when label count doesn't match single image."""
        processor = _TestProcessor()
        with pytest.raises(ValueError, match="Number of labels .* must match"):
            processor._normalize_image_inputs(temp_image_path, ["t1", "t2"])

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

    def test_invalid_reasoning_effort_raises(self):
        with pytest.raises(ValueError, match="reasoning_effort"):
            _TestProcessor(reasoning_effort="extreme")


class _TestProcessor(AgenticProcessorBase):
    """Test implementation of AgenticProcessorBase."""

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "Test system prompt"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "Test user message"

    def get_response_schema(self) -> dict[str, Any] | None:
        return {"type": "object", "properties": {"continue": {"type": "boolean"}}}

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "continue" in response and isinstance(response["continue"], bool)


class TestImageInputLoad:
    """Test ImageInput.load() behavior."""

    def test_load_rejects_oversized_image(self, tmp_path: Path) -> None:
        """load() raises ValueError when image exceeds max_image_dimension."""
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import ImageProcessingConfig
        from radiant_harness.config import get_config
        from radiant_harness.config import set_config

        original = get_config()
        try:
            set_config(HarnessConfig(image=ImageProcessingConfig(max_image_dimension=50)))

            image_path = tmp_path / "large.png"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(image_path)

            image_input = ImageInput(path=image_path)
            with pytest.raises(ValueError, match="exceed maximum"):
                image_input.load()
        finally:
            set_config(original)

    def test_load_accepts_image_within_limits(self, tmp_path: Path) -> None:
        """load() succeeds for images within max_image_dimension."""
        image_path = tmp_path / "ok.png"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(image_path)

        image_input = ImageInput(path=image_path)
        image_input.load()

        assert image_input.width == 100
        assert image_input.height == 100
        assert image_input.encoded is not None
