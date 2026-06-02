"""Image input handling: loading, decompression-bomb guarding, downscaling.

Split out of base.py; ImageInput is re-exported from gaze.base and gaze.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from loguru import logger
from PIL import Image

from gaze.config import get_config
from gaze.tools.registry import EncodedImage
from gaze.tools.registry import encode_image


@dataclass(frozen=True)
class ImageInput:
    """Represents a single image input with optional label.

    Immutable after construction.  Use :meth:`load` / :meth:`aload` to
    obtain a *new* ``ImageInput`` with populated pixel data — the
    original unloaded instance is never mutated.

    Attributes:
        path: Path to the image file
        label: Optional label for the image (e.g., "T1-weighted", "Pre-contrast")
        width: Image width in pixels (populated after loading)
        height: Image height in pixels (populated after loading)
        encoded: Base64-encoded image (populated after loading)
        pil_image: Loaded PIL Image kept in memory to avoid re-reading from disk
    """

    path: Path
    label: str | None = None
    width: int = 0
    height: int = 0
    encoded: EncodedImage | None = None
    pil_image: Image.Image | None = None

    @staticmethod
    @beartype
    def from_pil(
        image: Image.Image,
        *,
        label: str | None = None,
        path: Path | None = None,
    ) -> ImageInput:
        """Create an ImageInput directly from a PIL Image, skipping disk I/O.

        Args:
            image: PIL Image with pixel data already in memory.
            label: Optional label for the image.
            path: Optional source path (for logging only). Defaults to
                  a synthetic ``<in-memory>`` path.
        """
        max_dim = get_config().image.max_image_dimension
        if image.width > max_dim or image.height > max_dim:
            raise ValueError(
                f"Image dimensions {image.width}x{image.height} exceed "
                f"maximum allowed dimension of {max_dim}px"
            )
        return ImageInput(
            path=path or Path("<in-memory>"),
            label=label,
            width=image.width,
            height=image.height,
            encoded=encode_image(image),
            pil_image=image,
        )

    @beartype
    def load(self) -> ImageInput:
        """Load image and return a new ImageInput with populated fields.

        Returns ``self`` if already loaded (e.g. via :meth:`from_pil`).
        """
        if self.pil_image is not None:
            return self

        max_dim = get_config().image.max_image_dimension
        try:
            img = Image.open(self.path)
            # Gate on header dimensions BEFORE forcing a full decode so an
            # oversized image (a decompression bomb) is rejected without
            # allocating its pixel buffer. For PNG/JPEG the header dimensions
            # equal the decoded dimensions, so this bounds the decode.
            if img.width > max_dim or img.height > max_dim:
                img.close()
                raise ValueError(
                    f"Image dimensions {img.width}x{img.height} exceed "
                    f"maximum allowed dimension of {max_dim}px"
                )
            img.load()  # Force full pixel decode into memory
        except (Image.UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Failed to load image '{self.path}': {e}") from e
        return ImageInput(
            path=self.path,
            label=self.label,
            width=img.width,
            height=img.height,
            encoded=encode_image(img),
            pil_image=img,
        )

    async def aload(self) -> ImageInput:
        """Async version of :meth:`load` — offloads blocking I/O to a thread.

        Returns ``self`` if already loaded.
        Use this instead of ``load()`` when calling from an async context
        to avoid blocking the event loop during image decoding and encoding.
        """
        if self.pil_image is not None:
            return self
        return await asyncio.to_thread(self.load)

    def _load_pil_only(self) -> ImageInput:
        """Load PIL pixel data without base64 encoding.

        Used internally when encoding will be deferred (e.g. before
        downscaling) to avoid a wasted JPEG+base64 cycle at full resolution.
        """
        if self.pil_image is not None:
            return self

        max_dim = get_config().image.max_image_dimension
        try:
            img = Image.open(self.path)
            # Gate on header dimensions before the full decode (see load()).
            if img.width > max_dim or img.height > max_dim:
                img.close()
                raise ValueError(
                    f"Image dimensions {img.width}x{img.height} exceed "
                    f"maximum allowed dimension of {max_dim}px"
                )
            img.load()
        except (Image.UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Failed to load image '{self.path}': {e}") from e
        return ImageInput(
            path=self.path,
            label=self.label,
            width=img.width,
            height=img.height,
            encoded=None,
            pil_image=img,
        )

    async def _aload_pil_only(self) -> ImageInput:
        """Async version of :meth:`_load_pil_only`."""
        if self.pil_image is not None:
            return self
        return await asyncio.to_thread(self._load_pil_only)


def _downscale_image(img: ImageInput, max_dim: int) -> ImageInput:
    """Return a new ImageInput downscaled so neither side exceeds *max_dim*.

    If the image already fits, returns it with encoding (encoding lazily if
    the caller used ``_load_pil_only`` to defer it).
    Uses Lanczos resampling for quality.
    """
    if img.pil_image is None:
        return img
    if img.width <= max_dim and img.height <= max_dim:
        # Already fits — encode now if deferred by _load_pil_only.
        if img.encoded is not None:
            return img
        return ImageInput(
            path=img.path,
            label=img.label,
            width=img.width,
            height=img.height,
            encoded=encode_image(img.pil_image),
            pil_image=img.pil_image,
        )
    pil = img.pil_image.copy()
    pil.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    logger.info(
        f"Downscaled {img.path.name} from {img.width}x{img.height} "
        f"to {pil.width}x{pil.height} (max_encode_dimension={max_dim})"
    )
    return ImageInput(
        path=img.path,
        label=img.label,
        width=pil.width,
        height=pil.height,
        encoded=encode_image(pil),
        pil_image=pil,
    )
