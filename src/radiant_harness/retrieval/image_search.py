"""Web image search tool for medical image reference.

Provides image search capabilities for finding reference medical images from
trusted sources like NIH Open-i.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urljoin

import aiohttp
from beartype import beartype
from loguru import logger

from radiant_harness.cache import TTLCache
from radiant_harness.config import CacheConfig
from radiant_harness.config import SearchConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import HarnessError

_MODALITY_KEYWORDS = {
    "mri": "MRI",
    "magnetic resonance": "MRI",
    "ct scan": "CT",
    "computed tomography": "CT",
    "x-ray": "X-ray",
    "radiograph": "X-ray",
    "ultrasound": "Ultrasound",
    "sonography": "Ultrasound",
    "pet": "PET",
    "positron emission": "PET",
    "flair": "MRI",
    "t1-weighted": "MRI",
    "t2-weighted": "MRI",
    "dwi": "MRI",
    "mammograph": "Mammography",
}

_BODY_PART_KEYWORDS = {
    "brain": "brain",
    "cerebral": "brain",
    "head": "head",
    "chest": "chest",
    "thorax": "chest",
    "lung": "chest",
    "pulmonary": "chest",
    "abdomen": "abdomen",
    "abdominal": "abdomen",
    "liver": "abdomen",
    "kidney": "abdomen",
    "spine": "spine",
    "spinal": "spine",
    "vertebr": "spine",
    "pelvis": "pelvis",
    "hip": "pelvis",
    "knee": "knee",
    "shoulder": "shoulder",
    "heart": "cardiac",
    "cardiac": "cardiac",
}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

_CONTENT_TYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}


@dataclass
class ImageSearchResult:
    """Result from a medical image search."""

    title: str
    image_url: str
    thumbnail_url: str | None
    source_url: str  # URL to the source article/case
    source: str  # e.g., "openi"
    modality: str | None = None  # e.g., "MRI", "CT", "X-ray"
    body_part: str | None = None  # e.g., "brain", "chest"
    diagnosis: str | None = None
    caption: str | None = None
    article_title: str | None = None
    authors: str | None = None
    publication_date: str | None = None
    license: str | None = None
    reliability_score: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageSearchError(HarnessError):
    """Raised when an image search operation fails."""

    def __init__(self, engine_name: str, message: str, original_error: Exception | None = None):
        self.engine_name = engine_name
        self.original_error = original_error
        super().__init__(f"{engine_name}: {message}")


class ImageDownloadError(HarnessError):
    """Raised when an image download operation fails."""

    def __init__(self, url: str, message: str, original_error: Exception | None = None):
        self.url = url
        self.original_error = original_error
        super().__init__(f"Failed to download {url}: {message}")


class ImageSearchEngine:
    """Base class for image search engines.

    Provides common functionality for searching medical image databases.
    """

    @beartype
    def __init__(
        self,
        name: str,
        config: SearchConfig | None = None,
    ) -> None:
        """Initialize image search engine.

        Args:
            name: Engine identifier
            config: Search configuration. If None, uses global default.
        """
        self._config = config or get_config().search
        self.name = name
        self.timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        self.max_retries = self._config.max_retries
        self.headers = self._get_headers()
        self._session: aiohttp.ClientSession | None = None

    @property
    def config(self) -> SearchConfig:
        """Get the search configuration."""
        return self._config

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session for connection pooling."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        """Close the session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> dict[str, str]:
        """Get standard headers for web requests."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @beartype
    async def search(self, query: str, max_results: int = 5) -> list[ImageSearchResult]:
        """Search with retry logic.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of image search results

        Raises:
            ImageSearchError: If all retry attempts fail
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                results = await self._search_impl(query, max_results)
                return results
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e
                logger.warning(f"Image search attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise ImageSearchError(self.name, "All search attempts failed", last_error)

    @beartype
    async def _search_impl(self, query: str, max_results: int) -> list[ImageSearchResult]:
        """Implement actual search logic in subclasses.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of image search results
        """
        raise NotImplementedError


class OpenISearchEngine(ImageSearchEngine):
    """NIH Open-i Biomedical Image Search Engine.

    Provides access to the NIH Open-i database of biomedical images,
    including MRI, CT, X-ray, and other medical imaging modalities.
    """

    @beartype
    def __init__(self, config: SearchConfig | None = None) -> None:
        """Initialize Open-i search engine.

        Args:
            config: Search configuration. If None, uses global default.
        """
        super().__init__("Open-i", config=config)
        self.base_url = self._config.openi_base_url
        self.openi_base_url = "https://openi.nlm.nih.gov/"  # For URL joining

    async def _search_impl(self, query: str, max_results: int) -> list[ImageSearchResult]:
        params = {
            "query": query,
            "m": max_results,
            "it": "x,p,m,ct",
        }

        session = await self._get_session()
        async with session.get(self.base_url, params=params) as response:
            if response.status != 200:
                raise ImageSearchError(self.name, f"Open-i API returned status {response.status}")

            try:
                data = await response.json()
            except (aiohttp.ContentTypeError, json.JSONDecodeError) as e:
                text = await response.text()
                raise ImageSearchError(
                    self.name,
                    f"Open-i returned invalid JSON response: {text[:200]}",
                ) from e

        return self._parse_results(data)

    @beartype
    def _parse_results(self, data: dict[str, Any]) -> list[ImageSearchResult]:
        """Parse Open-i API response into ImageSearchResult objects.

        Args:
            data: JSON response from Open-i API

        Returns:
            List of parsed image search results
        """
        results: list[ImageSearchResult] = []
        items = data.get("list", [])
        skipped_no_image = 0

        for item in items:
            # Get image URL - require at least one
            image_url = item.get("imgLarge") or item.get("imgThumb")
            if not image_url:
                skipped_no_image += 1
                continue

            thumbnail_url = item.get("imgThumb", "")

            # Ensure absolute URLs
            if not image_url.startswith("http"):
                image_url = urljoin(self.openi_base_url, image_url)
            if thumbnail_url and not thumbnail_url.startswith("http"):
                thumbnail_url = urljoin(self.openi_base_url, thumbnail_url)

            title = item.get("title") or "Medical Image"
            caption = item.get("caption", "")
            article_title = item.get("articleTitle", "")
            combined_text = f"{caption} {title}"
            modality = self._extract_modality(combined_text)
            body_part = self._extract_body_part(combined_text)

            pmcid = item.get("pmcid", "")
            source_url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                if pmcid
                else item.get("detailedURL", "")
            )

            result = ImageSearchResult(
                title=title[:200],
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                source_url=source_url,
                source="openi",
                modality=modality,
                body_part=body_part,
                caption=caption,
                article_title=article_title,
                authors=item.get("authors", ""),
                publication_date=item.get("pubDate", ""),
                license=item.get("license", "Open Access"),
                reliability_score=0.90,
                metadata={
                    "pmcid": pmcid,
                    "mesh_terms": item.get("meshMajor", []),
                    "image_type": item.get("imgType", ""),
                },
            )
            results.append(result)

        if skipped_no_image > 0:
            logger.debug(f"Skipped {skipped_no_image} Open-i results without image URLs")
        return results

    @beartype
    def _extract_modality(self, text: str) -> str | None:
        text_lower = text.lower()
        # Check longer keywords first so "ct scan" matches before "ct" substring
        for keyword, modality in sorted(
            _MODALITY_KEYWORDS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if keyword in text_lower:
                return modality
        return None

    @beartype
    def _extract_body_part(self, text: str) -> str | None:
        text_lower = text.lower()
        # Check longer keywords first for more specific matches
        for keyword, part in sorted(
            _BODY_PART_KEYWORDS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if keyword in text_lower:
                return part
        return None


class MedicalImageSearchManager:
    """Manager for medical image search operations.

    Provides a unified interface for searching multiple medical image
    databases with caching, rate limiting, and result filtering.

    Example:
        async with MedicalImageSearchManager() as manager:
            results = await manager.search("brain MRI glioblastoma")
            for result in results:
                print(f"Found: {result.title} ({result.modality})")
    """

    @beartype
    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int | None = None,
        download_dir: Path | None = None,
        rate_limit_delay: float | None = None,
        search_config: SearchConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None:
        """Initialize medical image search manager.

        Args:
            engines: List of search engines to use (default: ["openi"])
            max_results_per_engine: Results per engine (overrides config)
            download_dir: Directory for downloaded images
            rate_limit_delay: Delay between API calls (overrides config)
            search_config: Search configuration. If None, uses global default.
            cache_config: Cache configuration. If None, uses global default.

        Raises:
            ValueError: If no valid engines are specified
        """
        config = get_config()
        self._search_config = search_config or config.search
        self._cache_config = cache_config or config.cache

        self.max_results_per_engine = (
            self._search_config.max_results_per_engine
            if max_results_per_engine is None
            else max_results_per_engine
        )
        self.rate_limit_delay = (
            self._search_config.rate_limit_delay_seconds
            if rate_limit_delay is None
            else rate_limit_delay
        )

        if self.max_results_per_engine < 1:
            raise ValueError(
                f"max_results_per_engine must be >= 1, got {self.max_results_per_engine}"
            )
        if self.rate_limit_delay < 0:
            raise ValueError(f"rate_limit_delay must be >= 0, got {self.rate_limit_delay}")

        # Use shared TTLCache instead of manual cache management
        self._cache: TTLCache[list[ImageSearchResult]] = TTLCache(self._cache_config)

        # Track whether we created a temp directory (for cleanup)
        self._created_temp_dir = False

        # Use secure temporary directory with proper permissions
        if download_dir:
            self.download_dir = download_dir
            self.download_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            self.download_dir = Path(tempfile.mkdtemp(prefix="rh_images_"))
            # Ensure directory has restricted permissions
            self.download_dir.chmod(0o700)
            self._created_temp_dir = True
            # Register cleanup on exit
            atexit.register(self._cleanup_temp_dir)

        self.engines: list[ImageSearchEngine] = []
        engines = engines or ["openi"]
        supported_engines = {"openi"}
        for engine in engines:
            if engine == "openi":
                self.engines.append(OpenISearchEngine(config=self._search_config))
            elif engine not in supported_engines:
                raise ValueError(
                    f"Unknown image search engine: '{engine}'. "
                    f"Supported engines: {', '.join(sorted(supported_engines))}"
                )

        if not self.engines:
            raise ValueError("No valid image search engines configured")

    async def __aenter__(self) -> MedicalImageSearchManager:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        for engine in self.engines:
            await engine.close()

        # Clean up temporary directory if we created it
        if self._created_temp_dir:
            self._cleanup_temp_dir()

    def _cleanup_temp_dir(self) -> None:
        """Clean up temporary directory and its contents."""
        try:
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                logger.debug(f"Cleaned up temporary directory: {self.download_dir}")
        except OSError as e:
            logger.warning(f"Failed to clean up temporary directory {self.download_dir}: {e}")

    @beartype
    async def search(
        self,
        query: str,
        modality: str | None = None,
        body_part: str | None = None,
    ) -> list[ImageSearchResult]:
        """Search for medical images.

        Args:
            query: Search query string
            modality: Optional imaging modality filter (e.g., "MRI", "CT")
            body_part: Optional anatomical body part filter (e.g., "brain", "chest")

        Returns:
            List of image search results with metadata

        Raises:
            ValueError: If query is empty
            ImageSearchError: If all search engines fail
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        enhanced_query = query
        if modality:
            enhanced_query += f" {modality}"
        if body_part:
            enhanced_query += f" {body_part}"

        query_hash = hashlib.sha256(enhanced_query.encode()).hexdigest()[:8]
        cache_key = f"img:{query_hash}|mod={modality}|part={body_part}"

        # Check cache using TTLCache (handles expiration automatically)
        cached_results = self._cache.get(cache_key)
        if cached_results is not None:
            logger.debug(f"Using cached image results for: {query}")
            return cached_results

        logger.info(f"Searching for medical images: '{enhanced_query}'")

        all_results: list[ImageSearchResult] = []
        errors: list[ImageSearchError] = []

        for i, engine in enumerate(self.engines):
            try:
                results = await engine.search(enhanced_query, self.max_results_per_engine)
                all_results.extend(results)
                if i < len(self.engines) - 1:
                    await asyncio.sleep(self.rate_limit_delay)
            except ImageSearchError as e:
                errors.append(e)
                logger.error(f"Image search engine {engine.name} failed: {e}")

        if errors and not all_results:
            raise ImageSearchError(
                "MedicalImageSearchManager",
                f"All image search engines failed: {[str(e) for e in errors]}",
            )

        modality_filter = modality.lower() if modality else None
        body_part_filter = body_part.lower() if body_part else None

        if modality_filter:
            all_results = [
                r for r in all_results if r.modality and modality_filter in r.modality.lower()
            ]

        if body_part_filter:
            all_results = [
                r for r in all_results if r.body_part and body_part_filter in r.body_part.lower()
            ]

        seen_urls: set[str] = set()
        unique_results: list[ImageSearchResult] = []
        for result in all_results:
            if result.image_url not in seen_urls:
                seen_urls.add(result.image_url)
                unique_results.append(result)

        # Cache results using TTLCache (handles expiration automatically)
        self._cache.set(cache_key, unique_results)

        logger.info(f"Image search complete: {len(unique_results)} unique results")
        return unique_results

    @beartype
    async def download_image(self, result: ImageSearchResult) -> Path:
        """Download an image from search results.

        Args:
            result: Image search result to download

        Returns:
            Path to the downloaded image file

        Raises:
            ImageDownloadError: If download fails
        """
        extension = self._get_extension_from_url(result.image_url)
        url_hash = hashlib.sha256(result.image_url.encode()).hexdigest()[:20]

        if extension:
            filepath = self.download_dir / f"{url_hash}{extension}"
            if filepath.exists():
                logger.debug(f"Image already cached: {filepath}")
                return filepath

        try:
            # Always create a dedicated session for downloads to avoid
            # session reuse issues and ensure proper resource cleanup
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                return await self._do_download(session, result, url_hash, extension)

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            raise ImageDownloadError(result.image_url, str(e), e) from e

    # Magic byte signatures for common image formats.
    _IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"\xff\xd8\xff", "JPEG"),
        (b"GIF87a", "GIF"),
        (b"GIF89a", "GIF"),
        (b"RIFF", "WEBP"),  # WEBP starts with RIFF....WEBP
        (b"BM", "BMP"),
    )

    # Maximum download size (10 MB) to prevent resource exhaustion.
    _MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

    @staticmethod
    def _validate_image_magic(content: bytes, url: str) -> None:
        """Verify that *content* starts with a known image magic signature.

        Raises:
            ImageDownloadError: If the content doesn't match any known format.
        """
        for magic, _fmt in MedicalImageSearchManager._IMAGE_MAGIC:
            if content[:len(magic)] == magic:
                return
        raise ImageDownloadError(url, "Downloaded content does not match any known image format")

    async def _do_download(
        self,
        session: aiohttp.ClientSession,
        result: ImageSearchResult,
        url_hash: str,
        extension: str | None,
    ) -> Path:
        """Perform the actual image download using the provided session.

        Raises:
            ImageDownloadError: If download fails due to HTTP error, invalid
                content type, failed magic-byte check, or oversized response.
        """
        async with session.get(
            result.image_url,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                raise ImageDownloadError(
                    result.image_url,
                    f"HTTP {response.status}",
                )

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ImageDownloadError(
                    result.image_url,
                    f"Response is not an image: {content_type}",
                )

            # Enforce size limit before reading the full body.
            content_length = response.headers.get("Content-Length")
            try:
                declared_size = int(content_length) if content_length else 0
            except ValueError:
                declared_size = 0
            if declared_size > self._MAX_DOWNLOAD_BYTES:
                raise ImageDownloadError(
                    result.image_url,
                    f"Image too large: {declared_size} bytes "
                    f"(max {self._MAX_DOWNLOAD_BYTES})",
                )

            content = await response.read()

            if len(content) > self._MAX_DOWNLOAD_BYTES:
                raise ImageDownloadError(
                    result.image_url,
                    f"Image too large: {len(content)} bytes "
                    f"(max {self._MAX_DOWNLOAD_BYTES})",
                )

            # Validate actual file content matches an image format.
            self._validate_image_magic(content, result.image_url)

            if extension is None:
                extension = self._get_extension_from_content_type(content_type)

            filepath = self.download_dir / f"{url_hash}{extension}"

            if filepath.exists():
                logger.debug(f"Image already cached: {filepath}")
                return filepath

            filepath.write_bytes(content)
            logger.info(f"Downloaded image: {filepath}")
            return filepath

    def _get_extension_from_url(self, url: str) -> str | None:
        """Extract image extension from URL path suffix (not substring)."""
        import posixpath
        from urllib.parse import urlparse as _urlparse

        path = _urlparse(url).path
        _, ext = posixpath.splitext(path)
        ext = ext.lower()
        return ext if ext in _IMAGE_EXTENSIONS else None

    def _get_extension_from_content_type(self, content_type: str) -> str:
        main_type = content_type.split(";")[0].strip().lower()
        return _CONTENT_TYPE_EXTENSION_MAP.get(main_type, ".jpg")


async def search_medical_images(
    query: str,
    max_results: int = 5,
    modality: str | None = None,
    body_part: str | None = None,
) -> list[ImageSearchResult]:
    async with MedicalImageSearchManager(max_results_per_engine=max_results) as manager:
        return await manager.search(query, modality=modality, body_part=body_part)
