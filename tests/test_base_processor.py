"""Tests for base AgenticProcessorBase functionality."""

from __future__ import annotations

import asyncio
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


class TestImageInputImmutability:
    """ImageInput must be frozen (immutable)."""

    def test_cannot_assign_to_fields(self) -> None:
        """Assigning to any field on a frozen dataclass raises FrozenInstanceError."""
        inp = ImageInput(path=Path("/fake/img.png"))
        with pytest.raises(AttributeError):
            inp.width = 42  # type: ignore[misc]

    def test_load_returns_new_instance(self, tmp_path: Path) -> None:
        """load() returns a distinct ImageInput; original is untouched."""
        image_path = tmp_path / "test.png"
        Image.new("RGB", (50, 50), color="red").save(image_path)

        original = ImageInput(path=image_path)
        loaded = original.load()

        assert loaded is not original
        assert loaded.width == 50
        assert original.width == 0

    def test_load_returns_self_when_already_loaded(self) -> None:
        """load() on an already-loaded instance returns the same object."""
        inp = ImageInput.from_pil(Image.new("RGB", (10, 10)))
        assert inp.load() is inp


class TestImageInputLoad:
    """Test ImageInput.load() behavior."""

    def test_load_rejects_oversized_image(self, tmp_path: Path) -> None:
        """load() raises ValueError when image exceeds max_image_dimension."""
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import ImageProcessingConfig
        from radiant_harness.config import config_context

        with config_context(HarnessConfig(image=ImageProcessingConfig(max_image_dimension=50))):
            image_path = tmp_path / "large.png"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(image_path)

            image_input = ImageInput(path=image_path)
            with pytest.raises(ValueError, match="exceed maximum"):
                image_input.load()

    def test_load_accepts_image_within_limits(self, tmp_path: Path) -> None:
        """load() succeeds for images within max_image_dimension."""
        image_path = tmp_path / "ok.png"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(image_path)

        image_input = ImageInput(path=image_path)
        loaded = image_input.load()

        assert loaded.width == 100
        assert loaded.height == 100
        assert loaded.encoded is not None
        # Original is unchanged — frozen dataclass
        assert image_input.width == 0
        assert image_input.encoded is None


class TestImageInputAload:
    """Test ImageInput.aload() async behavior."""

    @pytest.mark.asyncio
    async def test_aload_populates_fields(self, tmp_path: Path) -> None:
        """aload() returns a new ImageInput with dimensions and encoding."""
        image_path = tmp_path / "test.png"
        Image.new("RGB", (80, 60), color="blue").save(image_path)

        image_input = ImageInput(path=image_path)
        loaded = await image_input.aload()

        assert loaded.width == 80
        assert loaded.height == 60
        assert loaded.encoded is not None
        assert loaded.pil_image is not None
        # Original is unchanged
        assert image_input.width == 0

    @pytest.mark.asyncio
    async def test_aload_is_noop_when_already_loaded(self) -> None:
        """aload() returns self for images created via from_pil."""
        pil = Image.new("RGB", (32, 32), color="red")
        image_input = ImageInput.from_pil(pil)
        loaded = await image_input.aload()
        assert loaded is image_input
        assert loaded.width == 32

    @pytest.mark.asyncio
    async def test_aload_concurrent_multiple_images(self, tmp_path: Path) -> None:
        """Multiple aload() calls can run concurrently via gather."""
        inputs = []
        for i in range(4):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (64, 64), color="green").save(p)
            inputs.append(ImageInput(path=p))

        loaded = list(await asyncio.gather(*(inp.aload() for inp in inputs)))

        for inp in loaded:
            assert inp.width == 64
            assert inp.encoded is not None

    @pytest.mark.asyncio
    async def test_aload_rejects_oversized_image(self, tmp_path: Path) -> None:
        """aload() propagates ValueError for oversized images."""
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import ImageProcessingConfig
        from radiant_harness.config import config_context

        with config_context(HarnessConfig(image=ImageProcessingConfig(max_image_dimension=50))):
            image_path = tmp_path / "big.png"
            Image.new("RGB", (100, 100), color="red").save(image_path)

            image_input = ImageInput(path=image_path)
            with pytest.raises(ValueError, match="exceed maximum"):
                await image_input.aload()


class TestImageInputCorruptFile:
    """Test ImageInput.load() with corrupt/unreadable files."""

    def test_load_corrupt_file_raises_valueerror(self, tmp_path: Path) -> None:
        """load() wraps PIL errors into ValueError with the file path."""
        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_bytes(b"this is not a valid image file")

        image_input = ImageInput(path=corrupt_path)
        with pytest.raises(ValueError, match="Failed to load image"):
            image_input.load()

    def test_load_truncated_image_raises_valueerror(self, tmp_path: Path) -> None:
        """load() wraps PIL decode errors for truncated images."""
        # Write a valid PNG header but truncate the data
        valid_path = tmp_path / "valid.png"
        Image.new("RGB", (50, 50), color="red").save(valid_path)
        data = valid_path.read_bytes()
        truncated_path = tmp_path / "truncated.png"
        truncated_path.write_bytes(data[:20])  # Only header, no pixel data

        image_input = ImageInput(path=truncated_path)
        with pytest.raises(ValueError, match="Failed to load image"):
            image_input.load()

    @pytest.mark.asyncio
    async def test_aload_corrupt_file_raises_valueerror(self, tmp_path: Path) -> None:
        """aload() propagates wrapped ValueError for corrupt files."""
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"\x00\x00\x00")

        image_input = ImageInput(path=corrupt_path)
        with pytest.raises(ValueError, match="Failed to load image"):
            await image_input.aload()
