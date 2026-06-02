"""Tests for ImageManager image loading, transformation, and cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from gaze.exceptions import ToolExecutionError
from gaze.tools.image_manager import ImageManager


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


class TestImageManagerCopyIsolation:
    """Verify that original and current images are independent copies."""

    def test_set_image_creates_independent_copies(self, tmp_path: Path) -> None:
        """After set_image, current and original must be distinct objects."""
        path = _create_image(tmp_path, size=(40, 40))
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not mgr._original_image  # noqa: SLF001

    def test_transform_does_not_corrupt_original(self, tmp_path: Path) -> None:
        """Transforming current must leave original pixel data untouched."""
        path = _create_image(tmp_path, size=(80, 80))
        mgr = ImageManager()
        mgr.set_image(path)

        original_size = mgr._original_image.size  # noqa: SLF001
        mgr.transform_image(lambda img: img.resize((20, 20)))

        assert mgr._original_image.size == original_size  # noqa: SLF001
        assert mgr.current_image is not None
        assert mgr.current_image.size == (20, 20)

    def test_set_preloaded_copies_incoming_image(self, tmp_path: Path) -> None:
        """set_preloaded_image must copy the input so caller retains ownership."""
        path = _create_image(tmp_path, size=(60, 60))
        caller_img = Image.new("RGB", (60, 60), color=(255, 0, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(caller_img, path)

        # Caller's image should be independent from manager's copies
        assert mgr._original_image is not caller_img  # noqa: SLF001
        assert mgr.current_image is not caller_img
        assert mgr.current_image is not mgr._original_image  # noqa: SLF001

    def test_close_does_not_affect_caller_image(self, tmp_path: Path) -> None:
        """Closing the manager after set_preloaded must not close caller's image."""
        path = _create_image(tmp_path, size=(60, 60))
        caller_img = Image.new("RGB", (60, 60), color=(0, 255, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(caller_img, path)
        mgr.close()

        # Caller's image should still be usable
        assert caller_img.size == (60, 60)
        _ = caller_img.getpixel((0, 0))  # Should not raise

    @pytest.mark.asyncio
    async def test_ensure_loaded_creates_independent_copies(self, tmp_path: Path) -> None:
        """ensure_loaded must also create independent current/original copies."""
        path = _create_image(tmp_path, size=(50, 50))
        mgr = ImageManager()
        mgr._image_path = path  # noqa: SLF001
        await mgr.ensure_loaded()
        assert mgr.current_image is not mgr._original_image  # noqa: SLF001

    def test_reset_after_transform_preserves_original(self, tmp_path: Path) -> None:
        """Transform → reset cycle must produce a fresh copy of the original."""
        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)

        mgr.transform_image(lambda img: img.resize((10, 10)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (10, 10)

        mgr.reset_to_original()
        assert mgr.current_image is not None
        assert mgr.current_image.size == (100, 100)
        assert mgr.current_image is not mgr._original_image  # noqa: SLF001


class TestImageManagerOSErrors:
    """Cover set_image generic OSError (lines 126-127)."""

    def test_set_image_generic_oserror_is_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _create_image(tmp_path)
        mgr = ImageManager()

        def _raise_oserror(*_args: object, **_kwargs: object) -> None:
            raise OSError("Simulated disk failure")

        monkeypatch.setattr("PIL.Image.open", _raise_oserror)
        with pytest.raises(ToolExecutionError, match="Failed to read"):
            mgr.set_image(path)


class TestEnsureLoadedErrorPaths:
    """Cover ensure_loaded async error handlers (lines 150, 164-169)."""

    @pytest.mark.asyncio
    async def test_ensure_loaded_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError from async load is wrapped (line 164-165)."""
        ghost = tmp_path / "ghost.png"
        mgr = ImageManager()
        mgr._image_path = ghost  # noqa: SLF001
        with pytest.raises(ToolExecutionError, match="not found"):
            await mgr.ensure_loaded()

    @pytest.mark.asyncio
    async def test_ensure_loaded_invalid_image(self, tmp_path: Path) -> None:
        """UnidentifiedImageError from async load is wrapped (line 166-167)."""
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")
        mgr = ImageManager()
        mgr._image_path = bad  # noqa: SLF001
        with pytest.raises(ToolExecutionError, match="not a valid image"):
            await mgr.ensure_loaded()

    @pytest.mark.asyncio
    async def test_ensure_loaded_generic_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic OSError from async load is wrapped (lines 168-169)."""
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr._image_path = path  # noqa: SLF001

        def _raise_oserror(*_args: object, **_kwargs: object) -> None:
            raise OSError("Simulated disk failure")

        monkeypatch.setattr("PIL.Image.open", _raise_oserror)
        with pytest.raises(ToolExecutionError, match="Failed to read"):
            await mgr.ensure_loaded()

    @pytest.mark.asyncio
    async def test_ensure_loaded_double_check_via_concurrent_calls(self, tmp_path: Path) -> None:
        """Two concurrent ensure_loaded calls; second hits double-check (line 150)."""
        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr._image_path = path  # noqa: SLF001

        await asyncio.gather(mgr.ensure_loaded(), mgr.ensure_loaded())
        assert mgr.has_image
        assert mgr.current_image is not None
        assert mgr.current_image.size == (50, 50)


class TestResetToOriginalNoop:
    """Cover reset_to_original no-op fast path (line 252)."""

    def test_reset_noop_when_current_is_original(self) -> None:
        """When _current_image is _original_image, reset returns immediately."""
        img = Image.new("RGB", (30, 30), color=(100, 100, 100))
        mgr = ImageManager()
        mgr._original_image = img  # noqa: SLF001
        mgr._current_image = img  # noqa: SLF001

        mgr.reset_to_original()
        assert mgr.current_image is img
        assert mgr.current_image.size == (30, 30)


# ---------------------------------------------------------------------------
# Original encoding cache
# ---------------------------------------------------------------------------


class TestOriginalEncodingCache:
    def test_encoding_can_be_set_and_read(self, tmp_path: Path) -> None:
        from gaze.tools.registry import encode_image

        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None

        encoded = encode_image(mgr.current_image)
        mgr.original_encoding = encoded
        assert mgr.original_encoding is encoded

    def test_close_clears_encoding(self, tmp_path: Path) -> None:
        from gaze.tools.registry import encode_image

        path = _create_image(tmp_path)
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        mgr.close()
        assert mgr.original_encoding is None

    def test_set_image_clears_encoding(self, tmp_path: Path) -> None:
        from gaze.tools.registry import encode_image

        img1 = _create_image(tmp_path, "a.png", (30, 30))
        img2 = _create_image(tmp_path, "b.png", (60, 60))
        mgr = ImageManager()
        mgr.set_image(img1)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        mgr.set_image(img2)
        assert mgr.original_encoding is None

    def test_reset_preserves_encoding(self, tmp_path: Path) -> None:
        from gaze.tools.registry import encode_image

        path = _create_image(tmp_path, size=(100, 100))
        mgr = ImageManager()
        mgr.set_image(path)
        assert mgr.current_image is not None
        cached = encode_image(mgr.current_image)
        mgr.original_encoding = cached

        mgr.transform_image(lambda img: img.resize((10, 10)))
        mgr.reset_to_original()
        assert mgr.original_encoding is cached

    def test_set_preloaded_image_clears_encoding(self, tmp_path: Path) -> None:
        from gaze.tools.registry import encode_image

        path1 = _create_image(tmp_path, "a.png")
        mgr = ImageManager()
        mgr.set_image(path1)
        assert mgr.current_image is not None
        mgr.original_encoding = encode_image(mgr.current_image)

        new_img = Image.new("RGB", (80, 80), color=(0, 0, 0))
        path2 = _create_image(tmp_path, "b.png", (80, 80))
        mgr.set_preloaded_image(new_img, path2)
        assert mgr.original_encoding is None


# ---------------------------------------------------------------------------
# transfer_ownership in set_preloaded_image
# ---------------------------------------------------------------------------


class TestTransferOwnership:
    """Verify transfer_ownership=True avoids a redundant PIL copy."""

    def test_transfer_ownership_true_reuses_input(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(60, 60))
        img = Image.new("RGB", (60, 60), color=(255, 0, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=True)

        assert mgr._original_image is img  # noqa: SLF001
        assert mgr.current_image is not img
        assert mgr.current_image is not None
        assert mgr.current_image.size == (60, 60)

    def test_transfer_ownership_false_copies_input(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(60, 60))
        img = Image.new("RGB", (60, 60), color=(0, 255, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=False)

        assert mgr._original_image is not img  # noqa: SLF001

    def test_transfer_ownership_default_is_false(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(40, 40))
        img = Image.new("RGB", (40, 40), color=(0, 0, 255))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path)

        assert mgr._original_image is not img  # noqa: SLF001

    def test_transfer_ownership_reset_still_works(self, tmp_path: Path) -> None:
        path = _create_image(tmp_path, size=(80, 80))
        img = Image.new("RGB", (80, 80), color=(128, 128, 128))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=True)

        mgr.transform_image(lambda i: i.resize((20, 20)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (20, 20)

        mgr.reset_to_original()
        assert mgr.current_image is not None
        assert mgr.current_image.size == (80, 80)


# ---------------------------------------------------------------------------
# Reset skips encode when cached
# ---------------------------------------------------------------------------


class TestResetUsesCache:
    @pytest.mark.asyncio
    async def test_reset_skips_encode_when_cached(self, tmp_path: Path) -> None:
        from unittest.mock import patch as mock_patch

        from gaze.tools.registry import ToolRegistry
        from gaze.tools.registry import encode_image
        from gaze.tools.visual import _execute_reset

        path = _create_image(tmp_path, size=(100, 100))
        registry = ToolRegistry(image_path=path, tools=[])

        mgr = registry.get_image_manager()
        await mgr.ensure_loaded()
        assert mgr.current_image is not None

        cached = encode_image(mgr.current_image)
        mgr.original_encoding = cached
        mgr.transform_image(lambda img: img.resize((10, 10)))

        encode_call_count = 0
        _real_encode = encode_image

        def _counting_encode(*args, **kwargs):
            nonlocal encode_call_count
            encode_call_count += 1
            return _real_encode(*args, **kwargs)

        with mock_patch("gaze.tools.visual.encode_image", _counting_encode):
            result = await _execute_reset(registry)

        assert encode_call_count == 0
        assert result.image_base64 == cached.data
