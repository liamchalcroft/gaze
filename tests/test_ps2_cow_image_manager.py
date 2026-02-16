"""Tests for copy-on-write ImageManager."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.tools.image_manager import ImageManager


def _create_image(tmp_path: Path, name: str = "test.png", size: tuple[int, int] = (50, 50)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


class TestCOWSetImage:
    def test_current_is_original_after_set_image(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is mgr._original_image

    def test_current_diverges_after_transform(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))

        assert mgr.current_image is not mgr._original_image
        assert mgr.current_image is not None
        assert mgr.current_image.size == (50, 50)
        assert mgr._original_image is not None
        assert mgr._original_image.size == (100, 100)


class TestCOWSetPreloaded:
    def test_current_is_original_after_preload(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        img = Image.open(path)
        img.load()

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path)

        assert mgr.current_image is mgr._original_image
        assert mgr.current_image is img

    def test_preload_then_transform_preserves_original(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(80, 80))
        img = Image.open(path)
        img.load()

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path)

        mgr.transform_image(lambda img: img.resize((20, 20)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (20, 20)
        assert mgr._original_image is not None
        assert mgr._original_image.size == (80, 80)


class TestCOWEnsureLoaded:
    @pytest.mark.asyncio
    async def test_ensure_loaded_cow(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr._image_path = path.resolve()

        await mgr.ensure_loaded()
        assert mgr.current_image is mgr._original_image


class TestCOWClose:
    def test_close_cow_state_no_crash(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.close()
        assert mgr.current_image is None
        assert mgr._original_image is None

    def test_close_after_transform_no_crash(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)
        mgr.transform_image(lambda img: img.resize((50, 50)))

        mgr.close()
        assert mgr.current_image is None
        assert mgr._original_image is None

    def test_close_idempotent(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        mgr.close()
        mgr.close()


class TestCOWReset:
    def test_reset_noop_in_cow_state(self, tmp_path: Path) -> None:
        """When no transform has happened, reset is a no-op."""
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        original_ref = mgr._original_image
        mgr.reset_to_original()

        assert mgr.current_image is original_ref
        assert mgr._original_image is original_ref

    def test_reset_after_transform_copies_original(self, tmp_path: Path) -> None:
        """After a transform, reset creates a new copy of the original."""
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((10, 10)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (10, 10)

        mgr.reset_to_original()
        assert mgr.current_image is not None
        assert mgr.current_image.size == (100, 100)
        assert mgr.current_image is not mgr._original_image

    def test_transform_after_reset_works(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))
        mgr.reset_to_original()
        mgr.transform_image(lambda img: img.resize((25, 25)))

        assert mgr.current_image is not None
        assert mgr.current_image.size == (25, 25)
        assert mgr._original_image is not None
        assert mgr._original_image.size == (100, 100)


class TestCOWTransformPreservesOriginal:
    def test_original_usable_after_cow_transform(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))

        assert mgr._original_image is not None
        assert mgr._original_image.size == (100, 100)
        _ = mgr._original_image.getpixel((0, 0))

    def test_multiple_transforms_from_cow(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((50, 50)))
        mgr.transform_image(lambda img: img.resize((25, 25)))

        assert mgr.current_image is not None
        assert mgr.current_image.size == (25, 25)
        assert mgr._original_image is not None
        assert mgr._original_image.size == (100, 100)
