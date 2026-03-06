"""Image management for the radiology VLM agent harness."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from PIL import Image
from PIL import UnidentifiedImageError

from radiant_harness.exceptions import ToolExecutionError


@beartype
class ImageManager:
    """Manages image loading, transformation, and state.

    Example:
        manager = ImageManager()
        manager.set_image(Path("scan.png"))
        manager.transform_image(lambda img: zoom_image(img, 2.0))
        current = manager.current_image
    """

    @beartype
    def __init__(self) -> None:
        """Initialize image manager."""
        self._image_path: Path | None = None
        self._current_image: Image.Image | None = None
        self._original_image: Image.Image | None = None
        self._original_encoding: Any = None
        self._image_lock = asyncio.Lock()
        self._modified: bool = False

    @property
    def current_image(self) -> Image.Image | None:
        """Get the currently loaded image (may be None if not yet loaded)."""
        return self._current_image

    @property
    def image_path(self) -> Path | None:
        """Get the path to the source image."""
        return self._image_path

    @property
    def has_image(self) -> bool:
        """Check if an image is currently loaded."""
        return self._current_image is not None

    @property
    def is_modified(self) -> bool:
        """Whether the current image has been modified from the original.

        Set ``True`` by :meth:`transform_image`, cleared by
        :meth:`reset_to_original`, :meth:`set_image`,
        :meth:`set_preloaded_image`, and :meth:`close`.
        """
        return self._modified

    @property
    def original_encoding(self) -> Any:
        """Cached base64 encoding of the *original* (unmodified) image.

        Set once after the first encode and reused by reset to skip
        redundant JPEG→base64 work.

        Typed as ``Any`` at runtime to avoid circular-import forward-reference
        issues with beartype.  Static checkers see ``EncodedImage | None`` via
        the TYPE_CHECKING guard.
        """
        return self._original_encoding

    @original_encoding.setter
    def original_encoding(self, value: Any) -> None:
        self._original_encoding = value

    @staticmethod
    def _validate_image_path(image_path: Path) -> Path:
        """Resolve and validate an image path against traversal attacks.

        Returns the resolved (absolute, symlink-free) path.

        Raises:
            ToolExecutionError: If the path escapes allowed directories or
                uses dangerous patterns like ``..``.
        """
        resolved = image_path.resolve()
        # Block paths that contain '..' components before resolution
        # (e.g. symlink tricks where resolved path looks safe but intent is malicious)
        if ".." in image_path.parts:
            raise ToolExecutionError(f"Path traversal detected: {image_path}")
        # Block device files and other non-regular files (if they already exist)
        if resolved.exists() and not resolved.is_file():
            raise ToolExecutionError(f"Path is not a regular file: {image_path}")
        return resolved

    @beartype
    def set_image(self, image_path: Path) -> None:
        """Set the source image for operations.

        Args:
            image_path: Path to the image file

        Raises:
            ToolExecutionError: If image cannot be loaded or path is invalid
        """
        resolved = self._validate_image_path(image_path)
        self.close()

        try:
            with Image.open(resolved) as img:
                self._original_image = img.copy()
                self._current_image = self._original_image.copy()
        except FileNotFoundError as e:
            raise ToolExecutionError(f"Image file not found: {image_path}") from e
        except UnidentifiedImageError as e:
            raise ToolExecutionError(f"File is not a valid image: {image_path}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to read image file {image_path}: {e}") from e

        self._image_path = resolved
        self._modified = False
        logger.debug(f"Loaded image: {resolved} ({self._current_image.size})")

    @beartype
    async def ensure_loaded(self) -> None:
        """Ensure image is loaded, loading from path if necessary.

        This is thread-safe and can be called multiple times.

        Raises:
            ToolExecutionError: If no image path is set or image cannot be loaded.
        """
        # Fast path - if already loaded, return immediately
        if self._current_image is not None:
            return

        # Acquire lock before checking again to prevent race conditions
        async with self._image_lock:
            # Double-checked locking pattern - check again after acquiring lock
            if self._current_image is not None:
                return

            if self._image_path is None:
                raise ToolExecutionError("No image path set - call set_image() first")

            # Re-validate in case the path was set before validation was added
            image_path = self._validate_image_path(self._image_path)

            def _load_image() -> Image.Image:
                with Image.open(image_path) as img:
                    return img.copy()

            try:
                loaded = await asyncio.to_thread(_load_image)
            except FileNotFoundError as e:
                raise ToolExecutionError(f"Image file not found: {self._image_path}") from e
            except UnidentifiedImageError as e:
                raise ToolExecutionError(f"File is not a valid image: {self._image_path}") from e
            except OSError as e:
                raise ToolExecutionError(
                    f"Failed to read image file {self._image_path}: {e}"
                ) from e

            self._original_image = loaded
            self._current_image = loaded.copy()
            self._modified = False

    @beartype
    def transform_image(self, operation: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a transformation to the current image with automatic cleanup.

        The previous ``_current_image`` is always closed after the operation
        since current and original are always independent copies.

        This method is synchronous and does **not** acquire ``_image_lock``.
        It is safe to call from the single-threaded asyncio event loop (the
        normal execution path via ``ToolRegistry.execute``), but callers must
        not invoke it concurrently from multiple coroutines on the same
        ``ImageManager`` instance.

        Args:
            operation: Function that takes current image and returns new image

        Raises:
            ToolExecutionError: If no image is loaded

        Example:
            manager.transform_image(lambda img: zoom_image(img, 2.0))
        """
        if self._current_image is None:
            raise ToolExecutionError("No image loaded to transform")

        old_image = self._current_image
        self._current_image = operation(old_image)
        self._modified = True
        # Safe to always close: current is never aliased to original.
        if old_image is not self._current_image:
            old_image.close()

    @beartype
    def set_preloaded_image(
        self,
        image: Image.Image,
        image_path: Path,
        *,
        transfer_ownership: bool = False,
    ) -> None:
        """Set a pre-loaded PIL Image, avoiding a redundant disk read.

        Args:
            image: Already-loaded PIL Image (must have pixel data in memory).
            image_path: Path to the source file (stored for reset/logging).
            transfer_ownership: If True, the manager takes ownership of
                *image* directly (used as ``_original_image`` without
                copying).  The caller **must not** use or close *image*
                after this call.  When False (default), *image* is copied
                so the caller retains ownership.

        Raises:
            ToolExecutionError: If image_path contains traversal patterns.
        """
        resolved = self._validate_image_path(image_path)
        self.close()
        if transfer_ownership:
            self._original_image = image
        else:
            self._original_image = image.copy()
        self._current_image = self._original_image.copy()
        self._image_path = resolved
        self._modified = False

    @beartype
    def reset_to_original(self) -> None:
        """Reset the current image to the originally loaded state.

        Raises:
            ToolExecutionError: If no image is loaded
        """
        if self._original_image is None:
            raise ToolExecutionError("No original image to reset to")

        if self._current_image is self._original_image:
            return

        old_image = self._current_image
        self._current_image = self._original_image.copy()
        self._modified = False
        if old_image is not None:
            old_image.close()
        logger.debug("Reset image to original")

    @beartype
    def close(self) -> None:
        """Close and release all image resources."""
        if self._current_image is not None:
            self._current_image.close()
        self._current_image = None
        if self._original_image is not None:
            self._original_image.close()
        self._original_image = None
        self._image_path = None
        self._original_encoding = None
        self._modified = False
