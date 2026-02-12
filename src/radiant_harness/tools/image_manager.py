"""Image management for the radiology VLM agent harness."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

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
        self._image_lock = asyncio.Lock()

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

    @beartype
    def set_image(self, image_path: Path) -> None:
        """Set the source image for operations.

        Args:
            image_path: Path to the image file

        Raises:
            ToolExecutionError: If image cannot be loaded
        """
        self.close()

        try:
            with Image.open(image_path) as img:
                self._original_image = img.copy()
                self._current_image = img.copy()
        except FileNotFoundError as e:
            raise ToolExecutionError(f"Image file not found: {image_path}") from e
        except UnidentifiedImageError as e:
            raise ToolExecutionError(f"File is not a valid image: {image_path}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to read image file {image_path}: {e}") from e

        self._image_path = image_path
        logger.debug(f"Loaded image: {image_path} ({self._current_image.size})")

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

            image_path = self._image_path  # capture for thread closure

            def _load_image() -> tuple[Image.Image, Image.Image]:
                with Image.open(image_path) as img:
                    return img.copy(), img.copy()

            try:
                original, current = await asyncio.to_thread(_load_image)
            except FileNotFoundError as e:
                raise ToolExecutionError(f"Image file not found: {self._image_path}") from e
            except UnidentifiedImageError as e:
                raise ToolExecutionError(f"File is not a valid image: {self._image_path}") from e
            except OSError as e:
                raise ToolExecutionError(
                    f"Failed to read image file {self._image_path}: {e}"
                ) from e

            self._original_image = original
            self._current_image = current

    @beartype
    def transform_image(self, operation: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a transformation to the current image with automatic cleanup.

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
        old_image.close()

    @beartype
    def set_preloaded_image(self, image: Image.Image, image_path: Path) -> None:
        """Set a pre-loaded PIL Image, avoiding a redundant disk read.

        The caller transfers ownership of *image* — it must not be modified
        or closed afterwards.  One internal copy is made for the working
        ("current") image; the original reference is kept as-is for resets.

        Args:
            image: Already-loaded PIL Image (must have pixel data in memory).
            image_path: Path to the source file (stored for reset/logging).
        """
        self.close()
        self._original_image = image
        self._current_image = image.copy()
        self._image_path = image_path

    @beartype
    def reset_to_original(self) -> None:
        """Reset the current image to the originally loaded state.

        Raises:
            ToolExecutionError: If no image is loaded
        """
        if self._original_image is None:
            raise ToolExecutionError("No original image to reset to")

        old_image = self._current_image
        self._current_image = self._original_image.copy()
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
