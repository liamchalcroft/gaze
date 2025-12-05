"""
Web Image Search Tool for Medical Image Reference

This module provides image search capabilities for finding reference medical images
from trusted sources like NIH Open-i and Radiopaedia.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp
from beartype import beartype
from loguru import logger


@dataclass
class ImageSearchResult:
    """Result from a medical image search."""

    title: str
    image_url: str
    thumbnail_url: str | None
    source_url: str  # URL to the source article/case
    source: str  # e.g., "openi", "radiopaedia"
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            "title": self.title,
            "image_url": self.image_url,
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "source": self.source,
            "modality": self.modality,
            "body_part": self.body_part,
            "diagnosis": self.diagnosis,
            "caption": self.caption[:500] + "..." if self.caption and len(self.caption) > 500 else self.caption,
            "article_title": self.article_title,
            "reliability": f"{self.reliability_score:.2f}",
        }


class ImageSearchError(Exception):
    """Raised when an image search operation fails."""

    def __init__(self, engine_name: str, message: str, original_error: Exception | None = None):
        self.engine_name = engine_name
        self.original_error = original_error
        super().__init__(f"{engine_name}: {message}")


class ImageSearchEngine:
    """Base class for image search engines."""

    def __init__(self, name: str, timeout: int = 30, max_retries: int = 3):
        self.name = name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.headers = self._get_headers()
        # Reusable session for connection pooling (lazy-initialized)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session for connection pooling."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
            )
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

    async def search(self, query: str, max_results: int = 5) -> list[ImageSearchResult]:
        """Search with retry logic."""
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

    async def _search_impl(self, query: str, max_results: int) -> list[ImageSearchResult]:
        """Implement actual search logic in subclasses."""
        raise NotImplementedError


class OpenISearchEngine(ImageSearchEngine):
    """
    NIH Open-i Biomedical Image Search Engine.

    Open-i provides access to 1.6M+ images from PubMed Central articles.
    API documentation: https://openi.nlm.nih.gov/services
    """

    def __init__(self, **kwargs):
        super().__init__("Open-i", **kwargs)
        self.base_url = "https://openi.nlm.nih.gov/api/search"

    async def _search_impl(self, query: str, max_results: int) -> list[ImageSearchResult]:
        """Search Open-i for medical images."""
        # Build query parameters
        # Open-i API accepts query parameters directly
        params = {
            "query": query,
            "m": max_results,  # Number of results to return
            "it": "x,p,m,ct",  # Image types: x-ray, photograph, microscopy, CT
        }

        session = await self._get_session()
        async with session.get(self.base_url, params=params) as response:
            if response.status != 200:
                raise ImageSearchError(
                    self.name,
                    f"Open-i API returned status {response.status}",
                )

            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                text = await response.text()
                logger.warning(f"Open-i returned non-JSON response: {text[:200]}")
                return []

        return self._parse_results(data)

    def _parse_results(self, data: dict[str, Any]) -> list[ImageSearchResult]:
        """Parse Open-i API response."""
        results = []

        # Open-i returns results in a 'list' field
        items = data.get("list", [])

        for item in items:
            try:
                # Extract image URLs
                image_url = item.get("imgLarge", "") or item.get("imgThumb", "")
                thumbnail_url = item.get("imgThumb", "")

                if not image_url:
                    continue

                # Make URLs absolute if needed
                if image_url and not image_url.startswith("http"):
                    image_url = urljoin("https://openi.nlm.nih.gov/", image_url)
                if thumbnail_url and not thumbnail_url.startswith("http"):
                    thumbnail_url = urljoin("https://openi.nlm.nih.gov/", thumbnail_url)

                # Extract metadata
                title = item.get("title", "Medical Image")
                caption = item.get("caption", "")
                article_title = item.get("articleTitle", "")

                # Determine modality from caption or image type
                modality = self._extract_modality(caption + " " + title)

                # Extract body part from caption
                body_part = self._extract_body_part(caption + " " + title)

                # Build source URL
                pmcid = item.get("pmcid", "")
                if pmcid:
                    source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                else:
                    source_url = item.get("detailedURL", "")

                result = ImageSearchResult(
                    title=title[:200] if title else "Medical Image",
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
                    reliability_score=0.90,  # NIH/PubMed Central is high reliability
                    metadata={
                        "pmcid": pmcid,
                        "mesh_terms": item.get("meshMajor", []),
                        "image_type": item.get("imgType", ""),
                    },
                )
                results.append(result)

            except KeyError as e:
                # Missing expected field in API response - skip this result
                logger.debug(f"Skipping Open-i result with missing field: {e}")
                continue

        return results

    @beartype
    def _extract_modality(self, text: str) -> str | None:
        """Extract imaging modality from text."""
        text_lower = text.lower()
        modalities = {
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

        for keyword, modality in modalities.items():
            if keyword in text_lower:
                return modality

        return None

    @beartype
    def _extract_body_part(self, text: str) -> str | None:
        """Extract body part from text."""
        text_lower = text.lower()
        body_parts = {
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

        for keyword, part in body_parts.items():
            if keyword in text_lower:
                return part

        return None


class MedicalImageSearchManager:
    """Manager for medical image search operations."""

    MAX_CACHE_SIZE = 500  # Maximum number of cached queries

    def __init__(
        self,
        engines: list[str] | None = None,
        max_results_per_engine: int = 5,
        timeout: int = 30,
        cache_duration: int = 300,
        download_dir: Path | None = None,
        rate_limit_delay: float = 0.5,  # seconds between engine calls
    ):
        """
        Initialize medical image search manager.

        Args:
            engines: List of search engines to use (default: ["openi"])
            max_results_per_engine: Results to fetch per engine
            timeout: Request timeout in seconds
            cache_duration: Cache duration in seconds
            download_dir: Directory for downloaded images (default: temp dir)
            rate_limit_delay: Delay between search engine calls in seconds
        """
        self.max_results_per_engine = max_results_per_engine
        self.cache_duration = cache_duration
        self.rate_limit_delay = rate_limit_delay
        self._cache: dict[str, tuple[float, list[ImageSearchResult]]] = {}
        self.download_dir = download_dir or Path(tempfile.gettempdir()) / "nova_images"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Initialize search engines
        self.engines: list[ImageSearchEngine] = []
        engines = engines or ["openi"]

        supported_engines = {"openi"}
        for engine in engines:
            if engine == "openi":
                self.engines.append(OpenISearchEngine(timeout=timeout))
            elif engine not in supported_engines:
                raise ValueError(
                    f"Unknown image search engine: '{engine}'. "
                    f"Supported engines: {', '.join(sorted(supported_engines))}"
                )

        if not self.engines:
            raise ValueError("No valid image search engines configured")

    async def __aenter__(self) -> MedicalImageSearchManager:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close all engine sessions and release resources."""
        for engine in self.engines:
            await engine.close()

    def _evict_stale_cache(self) -> None:
        """Evict stale entries and enforce maximum cache size."""
        current_time = time.time()

        # First, remove expired entries
        expired_keys = [
            key for key, (timestamp, _) in self._cache.items()
            if current_time - timestamp > self.cache_duration
        ]
        for key in expired_keys:
            del self._cache[key]

        # If still over limit, evict oldest entries
        if len(self._cache) > self.MAX_CACHE_SIZE:
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])
            keys_to_remove = sorted_keys[: len(self._cache) - self.MAX_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del self._cache[key]

    @beartype
    async def search(
        self,
        query: str,
        modality: str | None = None,
        body_part: str | None = None,
    ) -> list[ImageSearchResult]:
        """
        Search for medical reference images.

        Args:
            query: Search query (e.g., "glioblastoma MRI T1 contrast")
            modality: Filter by imaging modality (MRI, CT, X-ray, etc.)
            body_part: Filter by body part (brain, chest, etc.)

        Returns:
            List of image search results
        """
        # Evict stale cache entries periodically
        self._evict_stale_cache()

        # Build enhanced query
        enhanced_query = query
        if modality:
            enhanced_query += f" {modality}"
        if body_part:
            enhanced_query += f" {body_part}"

        # Check cache
        cache_key = f"img:{enhanced_query}"
        if cache_key in self._cache:
            timestamp, cached_results = self._cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                logger.debug(f"Using cached image results for: {query}")
                return cached_results

        logger.info(f"Searching for medical images: '{enhanced_query}'")

        # Search across all engines
        all_results: list[ImageSearchResult] = []
        errors: list[ImageSearchError] = []

        for i, engine in enumerate(self.engines):
            try:
                results = await engine.search(enhanced_query, self.max_results_per_engine)
                all_results.extend(results)
                # Rate limit between engines (skip after last engine)
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

        # Filter by modality and body part if specified
        if modality:
            all_results = [
                r for r in all_results if r.modality and modality.lower() in r.modality.lower()
            ]

        if body_part:
            all_results = [
                r for r in all_results if r.body_part and body_part.lower() in r.body_part.lower()
            ]

        # Deduplicate by image URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.image_url not in seen_urls:
                seen_urls.add(result.image_url)
                unique_results.append(result)

        # Cache results
        self._cache[cache_key] = (time.time(), unique_results)

        logger.info(f"Image search complete: {len(unique_results)} unique results")
        return unique_results

    @beartype
    async def download_image(self, result: ImageSearchResult) -> Path | None:
        """
        Download an image to local storage.

        Args:
            result: Image search result to download

        Returns:
            Path to downloaded image, or None if download failed
        """
        # Try to get extension from URL first
        extension = self._get_extension_from_url(result.image_url)

        # Create filename from URL hash (md5 is fine for caching, not security)
        # Use 20 chars for lower collision probability in large image collections
        url_hash = hashlib.md5(result.image_url.encode(), usedforsecurity=False).hexdigest()[:20]

        # If we got extension from URL, check cache first
        if extension:
            filepath = self.download_dir / f"{url_hash}{extension}"
            if filepath.exists():
                logger.debug(f"Image already cached: {filepath}")
                return filepath

        # Download image
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    result.image_url,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                if response.status != 200:
                    logger.warning(f"Failed to download image: HTTP {response.status}")
                    return None

                # Validate it's actually an image
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.warning(f"Response is not an image: {content_type}")
                    return None

                content = await response.read()

                # Get extension from content-type if not determined from URL
                if extension is None:
                    extension = self._get_extension_from_content_type(content_type)

                filepath = self.download_dir / f"{url_hash}{extension}"

                # Check cache again (might have been downloaded while we were fetching)
                if filepath.exists():
                    logger.debug(f"Image already cached: {filepath}")
                    return filepath

                # Save to disk
                filepath.write_bytes(content)
                logger.info(f"Downloaded image: {filepath}")
                return filepath

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"Failed to download image: {e}")
            return None

    def _get_extension_from_url(self, url: str) -> str | None:
        """Get file extension from URL, or None if not determinable."""
        url_lower = url.lower()
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            if ext in url_lower:
                return ext
        return None

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from HTTP Content-Type header."""
        content_type_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
        }
        # Extract main type (ignore parameters like charset)
        main_type = content_type.split(";")[0].strip().lower()
        return content_type_map.get(main_type, ".jpg")

    def format_for_llm(self, results: list[ImageSearchResult]) -> str:
        """Format results for LLM consumption."""
        if not results:
            return "No reference images found."

        formatted = ["## Reference Medical Images\n"]

        for i, result in enumerate(results, 1):
            entry = f"""
### Image {i}

**Title:** {result.title}
**Source:** {result.source} (Reliability: {result.reliability_score:.2f})
**Modality:** {result.modality or 'Unknown'}
**Body Part:** {result.body_part or 'Unknown'}

**Caption:** {result.caption[:300] + '...' if result.caption and len(result.caption) > 300 else result.caption or 'No caption'}

**Image URL:** {result.image_url}
**Source Article:** {result.source_url}
"""
            formatted.append(entry)

        return "\n".join(formatted)


# Convenience functions
async def search_medical_images(
    query: str,
    max_results: int = 5,
    modality: str | None = None,
    body_part: str | None = None,
) -> list[ImageSearchResult]:
    """
    Search for medical reference images.

    Args:
        query: Search query
        max_results: Maximum results to return
        modality: Filter by imaging modality
        body_part: Filter by body part

    Returns:
        List of image search results
    """
    async with MedicalImageSearchManager(max_results_per_engine=max_results) as manager:
        return await manager.search(query, modality=modality, body_part=body_part)
