"""Shared caching utilities for the radiology VLM agent harness.

Provides a generic TTL cache implementation used by search managers.
"""

from __future__ import annotations

import time
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from beartype import beartype
from loguru import logger

from radiant_harness.config import CacheConfig
from radiant_harness.config import get_config

T = TypeVar("T")


@runtime_checkable
class _SupportsClose(Protocol):
    def close(self) -> Any: ...


class TTLCache(Generic[T]):
    """Time-to-live cache with automatic eviction.

    A generic cache that stores values with timestamps and automatically
    evicts stale entries. Supports configurable size limits and eviction ratios.

    Type Parameters:
        T: The type of values stored in the cache

    Example:
        from radiant_harness.cache import TTLCache

        # Create cache with default config
        cache: TTLCache[list[str]] = TTLCache()

        # Store a value
        cache.set("my_key", ["result1", "result2"])

        # Retrieve if not expired
        results = cache.get("my_key")  # Returns list or None

        # Check if key exists and is valid
        if cache.has("my_key"):
            print("Cache hit!")

        # Check hit/miss stats
        print(cache.stats())  # {'hits': 5, 'misses': 2, 'hit_rate': 0.71, ...}
    """

    @beartype
    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize TTL cache.

        Args:
            config: Cache configuration. If None, uses global default config.
        """
        self._config = config or get_config().cache
        self._cache: dict[str, tuple[float, T]] = {}
        self._hits: int = 0
        self._misses: int = 0

    @property
    def config(self) -> CacheConfig:
        """Get the cache configuration."""
        return self._config

    @beartype
    def get(self, key: str) -> T | None:
        """Get a value from cache if it exists and is not expired.

        Args:
            key: Cache key to look up

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self._cache:
            self._misses += 1
            return None

        timestamp, value = self._cache[key]
        if time.time() - timestamp > self._config.cache_duration_seconds:
            # Expired - remove and return None
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    @beartype
    def has(self, key: str) -> bool:
        """Check if a key exists in cache and is not expired.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired
        """
        return self.get(key) is not None

    @beartype
    def set(self, key: str, value: T) -> None:
        """Store a value in the cache.

        Automatically triggers eviction if cache is over size limit.

        Args:
            key: Cache key
            value: Value to store
        """
        # Evict before adding to ensure space
        self._evict_stale()

        self._cache[key] = (time.time(), value)

    @beartype
    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def _close_value(self, key: str, value: object, reason: str) -> None:
        """Close a cached value if it exposes a close method."""
        if isinstance(value, _SupportsClose):
            try:
                value.close()
            except OSError as e:
                logger.warning(f"Error closing {reason} cached value for key {key}: {e}")

    @beartype
    def clear(self, reset_stats: bool = False) -> None:
        """Remove all entries from the cache with cleanup.

        Args:
            reset_stats: If True, also reset hit/miss counters
        """
        for key, (_, value) in self._cache.items():
            self._close_value(key, value, "cached")
        self._cache.clear()
        if reset_stats:
            self._hits = 0
            self._misses = 0

    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing cache entries."""
        self._hits = 0
        self._misses = 0

    @beartype
    def _evict_stale(self) -> None:
        """Evict expired entries and enforce size limit.

        Called automatically before adding new entries.
        """
        current_time = time.time()

        # First, remove expired entries with cleanup
        expired_keys = [
            key
            for key, (timestamp, _) in self._cache.items()
            if current_time - timestamp > self._config.cache_duration_seconds
        ]
        for key in expired_keys:
            _, value = self._cache[key]
            self._close_value(key, value, "expired")
            del self._cache[key]

        # If still over limit, evict oldest entries
        if len(self._cache) > self._config.max_cache_size:
            # Sort by timestamp (oldest first)
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])

            # Calculate how many to remove based on evict_ratio
            target_size = int(self._config.max_cache_size * (1 - self._config.evict_ratio))
            keys_to_remove = sorted_keys[: len(self._cache) - target_size]

            for key in keys_to_remove:
                _, value = self._cache[key]
                self._close_value(key, value, "evicted")
                del self._cache[key]

    @property
    def size(self) -> int:
        """Get the current number of entries in the cache."""
        return len(self._cache)

    @beartype
    def stats(self) -> dict[str, int | float]:
        """Get cache statistics including hit/miss rates.

        Returns:
            Dictionary with size, max_size, duration, hits, misses, and hit_rate
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._config.max_cache_size,
            "duration_seconds": self._config.cache_duration_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }
