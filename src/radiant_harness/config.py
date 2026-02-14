"""Configuration classes for the radiology VLM agent harness.

Provides centralized configuration for constants, limits, and tunable parameters.
All previously hardcoded values are now configurable via these dataclasses.
"""

from __future__ import annotations

import ipaddress
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from urllib.parse import urlparse


@dataclass(frozen=True)
class ImageProcessingConfig:
    """Configuration for image processing operations.

    Attributes:
        min_image_size: Minimum dimension for images and crops (pixels)
        max_image_dimension: Maximum allowed width or height in pixels
        min_zoom_factor: Minimum allowed zoom factor
        max_zoom_factor: Maximum allowed zoom factor
        min_contrast_factor: Minimum allowed contrast factor
        max_contrast_factor: Maximum allowed contrast factor
        min_threshold_window: Minimum intensity window width for threshold tool.
            Prevents destructive narrow windowing that destroys diagnostic info.
        default_jpeg_quality: Default JPEG quality for encoding (1-100)
    """

    min_image_size: int = 10
    max_image_dimension: int = 16384
    min_zoom_factor: float = 0.5
    max_zoom_factor: float = 4.0
    min_contrast_factor: float = 0.5
    max_contrast_factor: float = 3.0
    min_threshold_window: int = 30
    default_jpeg_quality: int = 85
    min_brightness_factor: float = 0.5
    max_brightness_factor: float = 3.0
    min_sharpness_factor: float = 0.0
    max_sharpness_factor: float = 3.0
    max_grid_divisions: int = 8
    min_gaussian_sigma: float = 0.5
    max_gaussian_sigma: float = 5.0
    max_morphological_iterations: int = 5
    min_clahe_clip_limit: float = 1.0
    max_clahe_clip_limit: float = 10.0

    def __post_init__(self) -> None:
        if self.min_image_size < 1:
            raise ValueError(f"min_image_size must be >= 1, got {self.min_image_size}")
        if self.max_image_dimension < self.min_image_size:
            raise ValueError(
                f"max_image_dimension ({self.max_image_dimension}) "
                f"must be >= min_image_size ({self.min_image_size})"
            )
        if self.min_zoom_factor >= self.max_zoom_factor:
            raise ValueError(
                f"min_zoom_factor ({self.min_zoom_factor}) "
                f"must be < max_zoom_factor ({self.max_zoom_factor})"
            )
        if self.min_contrast_factor >= self.max_contrast_factor:
            raise ValueError(
                f"min_contrast_factor ({self.min_contrast_factor}) "
                f"must be < max_contrast_factor ({self.max_contrast_factor})"
            )
        if not 1 <= self.min_threshold_window <= 255:
            raise ValueError(
                f"min_threshold_window must be between 1 and 255, got {self.min_threshold_window}"
            )
        if not 1 <= self.default_jpeg_quality <= 100:
            raise ValueError(
                f"default_jpeg_quality must be between 1 and 100, got {self.default_jpeg_quality}"
            )
        if self.min_brightness_factor >= self.max_brightness_factor:
            raise ValueError(
                f"min_brightness_factor ({self.min_brightness_factor}) "
                f"must be < max_brightness_factor ({self.max_brightness_factor})"
            )
        if self.min_sharpness_factor < 0:
            raise ValueError(f"min_sharpness_factor must be >= 0, got {self.min_sharpness_factor}")
        if self.min_sharpness_factor >= self.max_sharpness_factor:
            raise ValueError(
                f"min_sharpness_factor ({self.min_sharpness_factor}) "
                f"must be < max_sharpness_factor ({self.max_sharpness_factor})"
            )
        if not 2 <= self.max_grid_divisions <= 20:
            raise ValueError(
                f"max_grid_divisions must be between 2 and 20, got {self.max_grid_divisions}"
            )
        if self.min_gaussian_sigma <= 0:
            raise ValueError(f"min_gaussian_sigma must be > 0, got {self.min_gaussian_sigma}")
        if self.min_gaussian_sigma >= self.max_gaussian_sigma:
            raise ValueError(
                f"min_gaussian_sigma ({self.min_gaussian_sigma}) "
                f"must be < max_gaussian_sigma ({self.max_gaussian_sigma})"
            )
        if not 1 <= self.max_morphological_iterations <= 20:
            raise ValueError(
                f"max_morphological_iterations must be between 1 and 20, "
                f"got {self.max_morphological_iterations}"
            )
        if self.min_clahe_clip_limit >= self.max_clahe_clip_limit:
            raise ValueError(
                f"min_clahe_clip_limit ({self.min_clahe_clip_limit}) "
                f"must be < max_clahe_clip_limit ({self.max_clahe_clip_limit})"
            )


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for caching behavior.

    Attributes:
        max_cache_size: Maximum number of cached entries
        cache_duration_seconds: Time-to-live for cache entries in seconds
        evict_ratio: Fraction of cache to evict when over limit (0.0-1.0)
    """

    max_cache_size: int = 500
    cache_duration_seconds: int = 300  # 5 minutes
    evict_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.max_cache_size < 1:
            raise ValueError(f"max_cache_size must be >= 1, got {self.max_cache_size}")
        if self.cache_duration_seconds <= 0:
            raise ValueError(
                f"cache_duration_seconds must be > 0, got {self.cache_duration_seconds}"
            )
        if not 0.0 < self.evict_ratio < 1.0:
            raise ValueError(
                f"evict_ratio must be between 0.0 and 1.0 exclusive, got {self.evict_ratio}"
            )


def _validate_base_url(url: str, field_name: str) -> None:
    """Validate a base URL for SSRF protection.

    Enforces HTTPS and rejects private/loopback addresses given as bare IPs
    or well-known loopback hostnames.  Does NOT perform DNS resolution — that
    would block the event loop when configs are constructed in async contexts
    and is unreliable at import time (network may be unavailable).
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"{field_name} must use HTTPS scheme, got {parsed.scheme!r} in {url!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"{field_name} has no hostname: {url!r}")

    # Reject well-known loopback hostnames
    if hostname in ("localhost", "0.0.0.0"):  # noqa: S104
        raise ValueError(f"{field_name} must not point to a loopback address: {hostname}")

    # Reject bare-IP private/loopback/link-local addresses
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return  # hostname is a regular DNS name — fine
    if addr.is_private or addr.is_loopback or addr.is_link_local:
        raise ValueError(
            f"{field_name} must not point to a private/loopback address: {hostname}"
        )


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for search operations.

    Attributes:
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        rate_limit_delay_seconds: Delay between API calls in seconds
        max_results_per_engine: Default results to fetch per engine
        max_total_results: Maximum total results to return
        max_content_preview_length: Maximum characters for content preview
        max_snippet_length: Maximum characters for snippet extraction
        max_content_for_llm: Maximum characters for LLM formatting
        ncbi_base_url: Base URL for NCBI E-utilities API
        openi_base_url: Base URL for OpenI API
    """

    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_delay_seconds: float = 1.0
    max_results_per_engine: int = 5
    max_total_results: int = 10
    max_content_preview_length: int = 2000
    max_snippet_length: int = 100
    max_content_for_llm: int = 5000
    ncbi_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    openi_base_url: str = "https://openi.nlm.nih.gov/api/search"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if self.rate_limit_delay_seconds < 0:
            raise ValueError(
                f"rate_limit_delay_seconds must be >= 0, got {self.rate_limit_delay_seconds}"
            )
        if self.max_results_per_engine < 1:
            raise ValueError(
                f"max_results_per_engine must be >= 1, got {self.max_results_per_engine}"
            )
        if self.max_total_results < 1:
            raise ValueError(f"max_total_results must be >= 1, got {self.max_total_results}")
        for attr in ("ncbi_base_url", "openi_base_url"):
            _validate_base_url(getattr(self, attr), attr)


@dataclass(frozen=True)
class RankingWeights:
    """Weights for search result ranking.

    All weights should be positive floats. The final score is computed as:
    score = reliability + (medical_relevance * medical_relevance_weight) + ...

    Attributes:
        medical_relevance_weight: Weight for medical relevance score
        recency_max_boost: Maximum boost for recent publications
        recency_decay_years: Number of years over which recency boost decays
        open_access_boost: Boost for open access articles
        title_match_weight: Weight per query term match in title
        content_match_weight: Weight per query term match in content
        entity_match_weight: Weight per medical entity match
        content_type_boosts: Boosts by content type per search type
    """

    medical_relevance_weight: float = 0.3
    recency_max_boost: float = 0.15
    recency_decay_years: int = 15
    open_access_boost: float = 0.1
    title_match_weight: float = 0.2
    content_match_weight: float = 0.05
    entity_match_weight: float = 0.1
    content_type_boosts: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "diagnosis": {"case_report": 0.2, "article": 0.1},
            "guidelines": {"guidelines": 0.3, "review": 0.2},
            "research": {"article": 0.2, "review": 0.1},
            "anatomy": {"review": 0.2, "article": 0.1},
        }
    )

    def __post_init__(self) -> None:
        # Deep-freeze the nested dict to enforce immutability
        if not isinstance(self.content_type_boosts, MappingProxyType):
            frozen = MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.content_type_boosts.items()}
            )
            object.__setattr__(self, "content_type_boosts", frozen)


@dataclass(frozen=True)
class AgenticConfig:
    """Configuration for agentic processing.

    Attributes:
        max_turns_limit: Absolute maximum turns allowed (hard limit)
        default_max_turns: Default max turns if not specified
        default_max_tokens: Default max tokens per generation
        default_temperature: Default temperature for generation
    """

    max_turns_limit: int = 20
    default_max_turns: int = 10
    default_max_tokens: int = 16384
    default_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_turns_limit < 1:
            raise ValueError(f"max_turns_limit must be >= 1, got {self.max_turns_limit}")
        if self.default_max_turns < 1:
            raise ValueError(f"default_max_turns must be >= 1, got {self.default_max_turns}")
        if self.default_max_turns > self.max_turns_limit:
            raise ValueError(
                f"default_max_turns ({self.default_max_turns}) "
                f"must be <= max_turns_limit ({self.max_turns_limit})"
            )
        if self.default_max_tokens < 1:
            raise ValueError(f"default_max_tokens must be >= 1, got {self.default_max_tokens}")
        if self.default_temperature < 0:
            raise ValueError(f"default_temperature must be >= 0, got {self.default_temperature}")


@dataclass(frozen=True)
class HarnessConfig:
    """Root configuration for the radiant harness.

    Provides access to all sub-configurations. Can be customized by
    passing individual config objects or by modifying the defaults.

    Example:
        # Use defaults
        config = HarnessConfig()

        # Customize specific settings
        config = HarnessConfig(
            cache=CacheConfig(max_cache_size=1000),
            search=SearchConfig(timeout_seconds=60),
        )

        # Access settings
        print(config.image.max_zoom_factor)  # 4.0
        print(config.cache.cache_duration_seconds)  # 300
    """

    image: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)


class _ConfigHolder:
    """Thread-safe container for global configuration.

    Uses a class to avoid module-level global statement while maintaining
    singleton pattern for configuration management.
    """

    _lock = threading.Lock()
    _config: HarnessConfig = HarnessConfig()

    @classmethod
    def get(cls) -> HarnessConfig:
        """Get the current configuration.

        No lock needed: Python's GIL makes reference reads atomic.
        The write-side lock on set() is sufficient for correctness.
        """
        return cls._config

    @classmethod
    def set(cls, config: HarnessConfig) -> None:
        """Set the configuration."""
        with cls._lock:
            cls._config = config


def get_config() -> HarnessConfig:
    """Get the current default configuration.

    Thread-safe access to the global configuration instance.

    Returns:
        The global default HarnessConfig instance
    """
    return _ConfigHolder.get()


def set_config(config: HarnessConfig) -> None:
    """Set the global default configuration.

    Thread-safe replacement of the global configuration.

    Args:
        config: New configuration to use as default

    Example:
        from radiant_harness.config import set_config, HarnessConfig, CacheConfig

        custom_config = HarnessConfig(
            cache=CacheConfig(max_cache_size=1000)
        )
        set_config(custom_config)
    """
    _ConfigHolder.set(config)
