"""Tests for ImageManager image loading, transformation, and cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools.image_manager import ImageManager


def _create_image(tmp_path: Path, name: str = "test.png", size: tuple[int, int] = (50, 50)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


class TestImageManagerSetImage:
    def test_set_image_loads_correctly(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)

        assert mgr.has_image
        assert mgr.current_image is not None
        assert mgr.current_image.size == (50, 50)
        assert mgr.image_path == path

    def test_set_image_missing_file_raises(self, tmp_path: Path) -> None:
        mgr = ImageManager()
        with pytest.raises(ToolExecutionError, match="not found"):
            mgr.set_image(tmp_path / "missing.png")

    def test_set_image_invalid_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")
        mgr = ImageManager()
        with pytest.raises(ToolExecutionError, match="not a valid image"):
            mgr.set_image(bad)

    def test_set_image_replaces_previous(self, tmp_path: Path) -> None:
        img1 = _create_image(tmp_path, "a.png", (30, 30))
        img2 = _create_image(tmp_path, "b.png", (60, 60))
        mgr = ImageManager()
        mgr.set_image(img1)
        assert mgr.current_image is not None
        assert mgr.current_image.size == (30, 30)

        mgr.set_image(img2)
        assert mgr.current_image is not None
        assert mgr.current_image.size == (60, 60)
        assert mgr.image_path == img2


class TestImageManagerTransform:
    def test_transform_applies_operation(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (50, 50)

    def test_transform_without_image_raises(self) -> None:
        mgr = ImageManager()
        with pytest.raises(ToolExecutionError, match="No image loaded"):
            mgr.transform_image(lambda img: img)

    def test_multiple_transforms_chain(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))
        mgr.transform_image(lambda img: img.resize((25, 25)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (25, 25)


class TestImageManagerReset:
    def test_reset_restores_original(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((10, 10)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (10, 10)

        mgr.reset_to_original()
        assert mgr.current_image is not None
        assert mgr.current_image.size == (100, 100)

    def test_reset_without_original_raises(self) -> None:
        mgr = ImageManager()
        with pytest.raises(ToolExecutionError, match="No original image"):
            mgr.reset_to_original()


class TestImageManagerClose:
    def test_close_releases_resources(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.has_image

        mgr.close()
        assert not mgr.has_image
        assert mgr.current_image is None
        assert mgr.image_path is None

    def test_close_idempotent(self) -> None:
        mgr = ImageManager()
        mgr.close()  # Should not raise
        mgr.close()  # Still should not raise


class TestImageManagerEnsureLoaded:
    @pytest.mark.asyncio
    async def test_ensure_loaded_without_path_raises(self) -> None:
        mgr = ImageManager()
        with pytest.raises(ToolExecutionError, match="No image path set"):
            await mgr.ensure_loaded()

    @pytest.mark.asyncio
    async def test_ensure_loaded_is_idempotent(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)

        # Already loaded → fast path, no error
        await mgr.ensure_loaded()
        assert mgr.has_image


class TestImageManagerInitialState:
    def test_initial_state_empty(self) -> None:
        mgr = ImageManager()
        assert not mgr.has_image
        assert mgr.current_image is None
        assert mgr.image_path is None
