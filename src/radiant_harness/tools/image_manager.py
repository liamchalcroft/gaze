"""Image management for the radiology VLM agent harness.

Provides image loading, transformation, and state management separate from tool execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from beartype import beartype
from loguru import logger
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError

if TYPE_CHECKING:
    pass


@beartype
class ImageManager:
    """Manages image loading, transformation, and state.

    Handles:
    - Loading images from disk
    - Managing current image state
    - Applying transformations with proper cleanup
    - Image history tracking
    - Thread-safe operations via async locks

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
        try:
            # Close existing images
            self.close()

            # Load and store image
            with Image.open(image_path) as img:
                self._original_image = img.copy()
                self._current_image = img.copy()
            self._image_path = image_path
            logger.debug(f"Loaded image: {image_path} ({self._current_image.size})")
        except Exception as e:
            raise ToolExecutionError(f"Failed to load image {image_path}: {e}") from e

    @beartype
    async def ensure_loaded(self) -> None:
        """Ensure image is loaded, loading from path if necessary.

        This is thread-safe and can be called multiple times.
        """
        if self._current_image is None and self._image_path is not None:
            async with self._image_lock:
                if self._current_image is None and self._image_path is not None:
                    try:
                        with Image.open(self._image_path) as img:
                            self._original_image = img.copy()
                            self._current_image = img.copy()
                    except Exception as e:
                        raise ToolExecutionError(
                            f"Failed to load image {self._image_path}: {e}"
                        ) from e

    @beartype
    def transform_image(self, operation: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a transformation to the current image with automatic cleanup.

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
        try:
            self._current_image = operation(old_image)
        except Exception as e:
            # Restore original image on failure
            self._current_image = old_image
            raise ToolExecutionError(f"Image transformation failed: {e}") from e
        finally:
            old_image.close()

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

    @beartype
    def get_size(self) -> tuple[int, int] | None:
        """Get the current image size as (width, height).

        Returns:
            Tuple of (width, height) or None if no image loaded
        """
        if self._current_image is not None:
            return self._current_image.size
        return None

    @beartype
    def copy_current(self) -> Image.Image | None:
        """Create a copy of the current image.

        Returns:
            Copy of current image or None if no image loaded
        """
        if self._current_image is not None:
            return self._current_image.copy()
        return None

    async def __aenter__(self) -> ImageManager:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        """Async context manager exit with cleanup."""
        self.close()
