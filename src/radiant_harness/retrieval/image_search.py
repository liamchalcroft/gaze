"""Web image search tool for medical image reference.

Provides image search capabilities for finding reference medical images from
trusted sources like NIH Open-i.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import ipaddress
import json
import re
import shutil
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from types import TracebackType
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse

import aiohttp
from beartype import beartype
from loguru import logger

from radiant_harness._frozen import deep_freeze
from radiant_harness.cache import TTLCache
from radiant_harness.config import CacheConfig
from radiant_harness.config import SearchConfig
from radiant_harness.config import get_config
from radiant_harness.retrieval.base import BaseSearchEngine
from radiant_harness.retrieval.base import SearchEngineError

# Pre-sorted by keyword length (longest first) so longer, more specific
# keywords are matched before shorter substrings (e.g. "ct scan" before "ct").
# Sorted once at import time to avoid re-sorting on every extraction call.
_MODALITY_KEYWORDS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
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
        }.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)

_BODY_PART_KEYWORDS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
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
        }.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)

# ---------------------------------------------------------------------------
# Module-level temp-dir tracking for atexit cleanup
# ---------------------------------------------------------------------------
# Instead of registering a bound method per MedicalImageSearchManager
# instance (which holds a strong reference to *self* and prevents GC),
# we track just the Path objects and register a single atexit handler.
_temp_dirs_lock = threading.Lock()
_temp_dirs: set[Path] = set()


def _atexit_cleanup_temp_dirs() -> None:
    """Remove all tracked temporary directories at interpreter shutdown."""
    with _temp_dirs_lock:
        for dir_path in list(_temp_dirs):
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
            except OSError:
                pass  # best-effort at shutdown
        _temp_dirs.clear()


atexit.register(_atexit_cleanup_temp_dirs)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_api_field(value: str, *, max_length: int = 500) -> str:
    """Sanitize a text field from an external API response.

    Strips control characters and truncates to *max_length* to reduce
    prompt-injection surface when these values later appear in LLM
    conversations.
    """
    value = _CONTROL_CHAR_RE.sub("", value)
    if len(value) > max_length:
        value = value[:max_length]
    return value


_PMCID_RE = re.compile(r"^PMC\d{1,10}$")

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

_CONTENT_TYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
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
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        frozen = deep_freeze(self.metadata)
        if not isinstance(frozen, MappingProxyType):
            raise TypeError("metadata must freeze to a mapping proxy")
        object.__setattr__(self, "metadata", frozen)


class ImageSearchError(SearchEngineError):
    """Raised when an image search operation fails."""


class ImageDownloadError(SearchEngineError):
    """Raised when an image download operation fails."""

    def __init__(self, url: str, message: str, original_error: Exception | None = None):
        self.url = url
        super().__init__("ImageDownload", f"Failed to download {url}: {message}", original_error)


# Hostnames from which image downloads are permitted.  Open-i returns
# image URLs on these domains; any other hostname is rejected to prevent
# SSRF via crafted API responses.
_ALLOWED_DOWNLOAD_HOSTNAMES: frozenset[str] = frozenset(
    {
        "openi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    }
)


def _validate_download_url(
    url: str,
    *,
    allowed_hostnames: frozenset[str] = _ALLOWED_DOWNLOAD_HOSTNAMES,
) -> None:
    """Validate a download URL against SSRF attacks.

    Enforces HTTPS scheme, rejects private/loopback/link-local addresses,
    resolves the hostname via DNS, and validates resolved IPs to close the
    DNS-rebinding gap.  Also enforces an explicit hostname allowlist.

    Raises:
        ImageDownloadError: If the URL fails validation.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ImageDownloadError(
            url,
            f"Only HTTPS URLs are allowed for image downloads, got {parsed.scheme!r}",
        )

    hostname = parsed.hostname
    if not hostname:
        raise ImageDownloadError(url, "URL has no hostname")

    # Reject well-known loopback hostnames
    if hostname in ("localhost", "0.0.0.0"):  # noqa: S104
        raise ImageDownloadError(
            url,
            f"Downloads from loopback addresses are not allowed: {hostname}",
        )

    # Enforce hostname allowlist (primary SSRF defence)
    if hostname not in allowed_hostnames:
        raise ImageDownloadError(
            url,
            f"Hostname {hostname!r} is not in the allowed download set: "
            f"{sorted(allowed_hostnames)}",
        )

    # Reject bare-IP private/loopback/link-local addresses
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        pass  # hostname is a regular DNS name — resolved below
    else:
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise ImageDownloadError(
                url,
                f"Downloads from private/loopback addresses are not allowed: {hostname}",
            )

    # Resolve DNS and validate all resulting IPs to close DNS-rebinding gap
    import socket

    try:
        addrinfos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ImageDownloadError(url, f"DNS resolution failed for {hostname}: {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfos:
        ip_str = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if resolved.is_private or resolved.is_loopback or resolved.is_link_local:
            raise ImageDownloadError(
                url,
                f"Hostname {hostname} resolves to private/loopback address {ip_str}",
            )


class ImageSearchEngine(BaseSearchEngine[ImageSearchResult, ImageSearchError]):
    """Base class for image search engines.

    Inherits session management, retry logic, and honest bot User-Agent
    from :class:`BaseSearchEngine`.
    """

    def _make_error(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> ImageSearchError:
        return ImageSearchError(self.name, message, original_error)


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
        # Derive the origin from the configured API URL for resolving
        # relative image paths, instead of hardcoding a separate URL.
        parsed = urlparse(self._config.openi_base_url)
        self.openi_base_url = f"{parsed.scheme}://{parsed.netloc}/"

    async def _search_impl(self, query: str, max_results: int) -> list[ImageSearchResult]:
        params = {
            "query": query,
            "m": max_results,
            "it": "x,p,m,ct",
        }

        session = await self._get_session()
        async with session.get(self.base_url, params=params) as response:
            # Let aiohttp raise ClientResponseError so transient HTTP errors
            # (429, 5xx) are retried by the base-class retry wrapper.
            response.raise_for_status()

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
        skipped_non_https = 0

        for item in items:
            # Get image URL - require at least one
            image_url = item.get("imgLarge") or item.get("imgThumb")
            if not image_url:
                skipped_no_image += 1
                continue

            thumbnail_url = item.get("imgThumb") or None

            # Ensure absolute URLs
            if not image_url.startswith("http"):
                image_url = urljoin(self.openi_base_url, image_url)
            if thumbnail_url and not thumbnail_url.startswith("http"):
                thumbnail_url = urljoin(self.openi_base_url, thumbnail_url)

            # Reject non-HTTPS image URLs from untrusted API responses
            if not image_url.startswith("https://"):
                skipped_non_https += 1
                continue

            # Enforce HTTPS on thumbnail URLs as well
            if thumbnail_url and not thumbnail_url.startswith("https://"):
                thumbnail_url = None

            title = _sanitize_api_field(item.get("title") or "Medical Image", max_length=200)
            caption = _sanitize_api_field(item.get("caption", ""))
            article_title = _sanitize_api_field(item.get("articleTitle", ""))
            combined_text = f"{caption} {title}"
            modality = self._extract_modality(combined_text)
            body_part = self._extract_body_part(combined_text)

            pmcid = _sanitize_api_field(item.get("pmcid", ""), max_length=30)
            if pmcid and not _PMCID_RE.match(pmcid):
                pmcid = ""
            source_url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                if pmcid
                else _sanitize_api_field(item.get("detailedURL", ""), max_length=500)
            )

            result = ImageSearchResult(
                title=title,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                source_url=source_url,
                source="openi",
                modality=modality,
                body_part=body_part,
                caption=caption,
                article_title=article_title,
                authors=_sanitize_api_field(item.get("authors", "")),
                publication_date=_sanitize_api_field(item.get("pubDate", ""), max_length=30),
                license=_sanitize_api_field(item.get("license", ""), max_length=100) or None,
                reliability_score=0.90,
                metadata={
                    "pmcid": pmcid,
                    "mesh_terms": [
                        _sanitize_api_field(term, max_length=100)
                        for term in item.get("meshMajor", [])
                        if isinstance(term, str)
                    ],
                    "image_type": _sanitize_api_field(str(item.get("imgType", "")), max_length=50),
                },
            )
            results.append(result)

        if skipped_no_image > 0:
            logger.debug(f"Skipped {skipped_no_image} Open-i results without image URLs")
        if skipped_non_https > 0:
            logger.warning(f"Skipped {skipped_non_https} Open-i results with non-HTTPS image URLs")
        return results

    @beartype
    def _extract_modality(self, text: str) -> str | None:
        text_lower = text.lower()
        # _MODALITY_KEYWORDS is pre-sorted longest-first at module level.
        for keyword, modality in _MODALITY_KEYWORDS:
            if keyword in text_lower:
                return modality
        return None

    @beartype
    def _extract_body_part(self, text: str) -> str | None:
        text_lower = text.lower()
        # _BODY_PART_KEYWORDS is pre-sorted longest-first at module level.
        for keyword, part in _BODY_PART_KEYWORDS:
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
            # Track for module-level atexit cleanup (no strong ref to self)
            with _temp_dirs_lock:
                _temp_dirs.add(self.download_dir)

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

        self._download_session: aiohttp.ClientSession | None = None

    async def _get_download_session(self) -> aiohttp.ClientSession:
        if self._download_session is None or self._download_session.closed:
            import radiant_harness

            self._download_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": f"radiant_harness/{radiant_harness.__version__}",
                },
            )
        return self._download_session

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
        if self._download_session is not None and not self._download_session.closed:
            await self._download_session.close()
            self._download_session = None
        for engine in self.engines:
            await engine.close()
        self._cache.clear()

        # Clean up temporary directory if we created it
        if self._created_temp_dir:
            self._cleanup_temp_dir()
            # Remove from module-level tracker so atexit won't double-clean
            with _temp_dirs_lock:
                _temp_dirs.discard(self.download_dir)

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

        query_hash = hashlib.sha256(enhanced_query.encode()).hexdigest()[:16]
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
            session = await self._get_download_session()
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
            if content[: len(magic)] == magic:
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
            ImageDownloadError: If download fails due to SSRF validation,
                HTTP error, invalid content type, failed magic-byte check,
                or oversized response.
        """
        # SSRF gate: reject non-HTTPS, private, and loopback URLs before
        # making any network request.  The URL originates from an untrusted
        # external API response (e.g. Open-i).
        await asyncio.to_thread(_validate_download_url, result.image_url)

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
                    f"Image too large: {declared_size} bytes (max {self._MAX_DOWNLOAD_BYTES})",
                )

            # Stream in chunks to enforce the size limit even when the
            # server lies about Content-Length or omits it entirely.
            # This prevents OOM from unbounded response.read().
            chunks: list[bytes] = []
            total_read = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total_read += len(chunk)
                if total_read > self._MAX_DOWNLOAD_BYTES:
                    raise ImageDownloadError(
                        result.image_url,
                        f"Image too large: >{self._MAX_DOWNLOAD_BYTES} bytes "
                        f"(streaming limit exceeded)",
                    )
                chunks.append(chunk)
            content = b"".join(chunks)

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
