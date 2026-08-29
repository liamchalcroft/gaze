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


def _open_and_decode(path: Path) -> Image.Image:
    """Open *path* and force a full decode, closing the handle on any failure.

    Gates on the header dimensions BEFORE decoding so an oversized image (a
    decompression bomb) is rejected without allocating its pixel buffer. For
    PNG/JPEG the header dimensions equal the decoded dimensions, so this bounds
    the decode.
    """
    max_dim = get_config().image.max_image_dimension
    try:
        img = Image.open(path)
    except (Image.UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Failed to load image '{path}': {e}") from e

    try:
        if img.width > max_dim or img.height > max_dim:
            raise ValueError(
                f"Image dimensions {img.width}x{img.height} exceed "
                f"maximum allowed dimension of {max_dim}px"
            )
        img.load()  # Force full pixel decode into memory
    except (Image.UnidentifiedImageError, OSError, SyntaxError) as e:
        # A truncated file raises here, after the header check passed. Closing
        # is what the oversize branch already did; omitting it leaked the file
        # handle on every failed decode.
        img.close()
        raise ValueError(f"Failed to load image '{path}': {e}") from e
    except BaseException:
        img.close()
        raise
    return img


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
        owns_pil_image: True when ``pil_image`` was decoded by this library and
            may therefore be closed by it. False for a caller-supplied image
            (see :meth:`from_pil`), which the caller keeps using afterwards.
    """

    path: Path
    label: str | None = None
    width: int = 0
    height: int = 0
    encoded: EncodedImage | None = None
    pil_image: Image.Image | None = None
    owns_pil_image: bool = False

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
        # Encoding is deferred to load()/aload(). Encoding here would be
        # wasted work whenever max_encode_dimension is set, because the caller
        # then downscales and re-encodes at the smaller size.
        return ImageInput(
            path=path or Path("<in-memory>"),
            label=label,
            width=image.width,
            height=image.height,
            encoded=None,
            pil_image=image,
            owns_pil_image=False,
        )

    @beartype
    def load(self) -> ImageInput:
        """Load image and return a new ImageInput with populated fields.

        When pixels are already in memory (e.g. via :meth:`from_pil`) only the
        base64 encoding is filled in; the image is not re-read from disk.
        """
        if self.pil_image is not None:
            if self.encoded is not None:
                return self
            return ImageInput(
                path=self.path,
                label=self.label,
                width=self.width,
                height=self.height,
                encoded=encode_image(self.pil_image),
                pil_image=self.pil_image,
                owns_pil_image=self.owns_pil_image,
            )

        img = _open_and_decode(self.path)
        return ImageInput(
            path=self.path,
            label=self.label,
            width=img.width,
            height=img.height,
            encoded=encode_image(img),
            pil_image=img,
            owns_pil_image=True,
        )

    @beartype
    async def aload(self) -> ImageInput:
        """Async version of :meth:`load` that offloads blocking I/O to a thread.

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

        img = _open_and_decode(self.path)
        return ImageInput(
            path=self.path,
            label=self.label,
            width=img.width,
            height=img.height,
            encoded=None,
            pil_image=img,
            owns_pil_image=True,
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
            owns_pil_image=img.owns_pil_image,
        )
    pil = img.pil_image.copy()
    pil.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    logger.info(
        f"Downscaled {img.path.name} from {img.width}x{img.height} "
        f"to {pil.width}x{pil.height} (max_encode_dimension={max_dim})"
    )
    # The full-resolution source is superseded by the thumbnail and nothing
    # else refers to it, so release its buffer now rather than at GC time.
    # A caller-supplied image is never ours to close.
    if img.owns_pil_image:
        img.pil_image.close()
    return ImageInput(
        path=img.path,
        label=img.label,
        width=pil.width,
        height=pil.height,
        encoded=encode_image(pil),
        pil_image=pil,
        owns_pil_image=True,
    )
