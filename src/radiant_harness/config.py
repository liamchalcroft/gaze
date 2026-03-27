"""Configuration classes for the radiology VLM agent harness.

Provides centralized configuration for constants, limits, and tunable parameters.
All previously hardcoded values are now configurable via these dataclasses.
"""

from __future__ import annotations

import contextlib
import ipaddress
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from dataclasses import field
from urllib.parse import urlparse

from beartype import beartype


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
    min_threshold_window: int = 50
    default_jpeg_quality: int = 85
    min_brightness_factor: float = 0.5
    max_brightness_factor: float = 2.0
    min_sharpness_factor: float = 0.1
    max_sharpness_factor: float = 3.0
    max_grid_divisions: int = 8
    min_gaussian_sigma: float = 0.5
    max_gaussian_sigma: float = 5.0
    max_morphological_iterations: int = 5
    min_clahe_clip_limit: float = 1.0
    max_clahe_clip_limit: float = 4.0
    max_clahe_tile_size: int = 32
    min_window_width: int = 50

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
        if not 2 <= self.max_clahe_tile_size <= 64:
            raise ValueError(
                f"max_clahe_tile_size must be between 2 and 64, got {self.max_clahe_tile_size}"
            )
        if self.min_window_width < 2:
            raise ValueError(f"min_window_width must be >= 2, got {self.min_window_width}")


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


# Hostnames known to be safe destinations for search API base URLs.
# Used by _validate_base_url to reject arbitrary DNS names that could
# resolve to internal services (DNS rebinding / SSRF).
_ALLOWED_SEARCH_HOSTNAMES: frozenset[str] = frozenset(
    {
        "eutils.ncbi.nlm.nih.gov",
        "openi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    }
)


def _validate_base_url(
    url: str,
    field_name: str,
    *,
    allowed_hostnames: frozenset[str] | None = None,
) -> None:
    """Validate a base URL for SSRF protection.

    Enforces HTTPS, rejects private/loopback addresses given as bare IPs
    or well-known loopback hostnames, and optionally enforces an explicit
    hostname allowlist to prevent DNS-rebinding attacks.

    Args:
        url: The URL to validate.
        field_name: Config field name (for error messages).
        allowed_hostnames: If provided, the URL's hostname must appear in
            this set.  This closes the DNS-rebinding gap where an attacker
            registers a domain that resolves to an internal IP.
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
        pass  # hostname is a regular DNS name — check allowlist below
    else:
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise ValueError(
                f"{field_name} must not point to a private/loopback address: {hostname}"
            )

    # Enforce hostname allowlist (closes DNS-rebinding gap)
    if allowed_hostnames is not None and hostname not in allowed_hostnames:
        raise ValueError(
            f"{field_name} hostname {hostname!r} is not in the allowed set: "
            f"{sorted(allowed_hostnames)}"
        )


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for search operations.

    Attributes:
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        rate_limit_delay_seconds: Delay between API calls in seconds
        max_content_preview_length: Maximum characters for content preview
        max_snippet_length: Maximum characters for snippet extraction
        max_content_for_llm: Maximum characters for LLM formatting
        ncbi_base_url: Base URL for NCBI E-utilities API
        openi_base_url: Base URL for OpenI API
    """

    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_delay_seconds: float = 1.0
    max_content_preview_length: int = 500
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
        for attr in ("ncbi_base_url", "openi_base_url"):
            _validate_base_url(
                getattr(self, attr),
                attr,
                allowed_hostnames=_ALLOWED_SEARCH_HOSTNAMES,
            )


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


class _ConfigHolder:
    """Container for global defaults plus context-local overrides.

    ``set()`` updates the process-wide default config. ``get()`` first checks
    for an active context override so concurrent threads/tasks can isolate
    temporary config changes without stomping on each other.
    """

    _lock = threading.Lock()
    _config: HarnessConfig = HarnessConfig()
    _context_config: ContextVar[HarnessConfig | None] = ContextVar(
        "radiant_harness_context_config",
        default=None,
    )

    @classmethod
    def get(cls) -> HarnessConfig:
        """Get the effective configuration for the current thread/task."""
        context_config = cls._context_config.get()
        return context_config if context_config is not None else cls._config

    @classmethod
    def set(cls, config: HarnessConfig) -> None:
        """Set the global default configuration."""
        with cls._lock:
            cls._config = config

    @classmethod
    def set_context(cls, config: HarnessConfig) -> Token[HarnessConfig | None]:
        """Set a context-local configuration override."""
        return cls._context_config.set(config)

    @classmethod
    def reset_context(cls, token: Token[HarnessConfig | None]) -> None:
        """Reset the current context-local override."""
        cls._context_config.reset(token)


@beartype
def get_config() -> HarnessConfig:
    """Get the current default configuration.

    Returns the context-local override when one is active, otherwise the
    process-wide default configuration.

    Returns:
        The global default HarnessConfig instance
    """
    return _ConfigHolder.get()


@beartype
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


@beartype
def reset_config() -> None:
    """Reset the global configuration to defaults.

    Convenience wrapper that restores a fresh ``HarnessConfig()``.
    Useful in test teardown to prevent config leakage between tests.
    """
    _ConfigHolder.set(HarnessConfig())


@contextlib.contextmanager
def config_context(config: HarnessConfig) -> Iterator[HarnessConfig]:
    """Temporarily override configuration for the current thread/task only.

    This does not mutate the process-wide default set by :func:`set_config`.
    Nested contexts restore correctly, and concurrent threads/tasks keep their
    own overrides.

    Args:
        config: Temporary configuration to use inside the block.

    Yields:
        The temporary configuration (same object as *config*).

    Example:
        from radiant_harness.config import config_context, HarnessConfig, CacheConfig

        with config_context(HarnessConfig(cache=CacheConfig(max_cache_size=1000))):
            # get_config() returns the temporary config here
            ...
        # original config is restored here
    """
    token = _ConfigHolder.set_context(config)
    try:
        yield config
    finally:
        _ConfigHolder.reset_context(token)
