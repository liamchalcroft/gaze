"""Tests targeting uncovered lines in cache.py.

Covers:
- TTLCache.config property (L74)
- TTLCache.get() with expired entry — close + miss (L94-97)
- TTLCache.clear(reset_stats=True) — counter reset (L184-185)
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from gaze.cache import TTLCache
from gaze.config import CacheConfig

# ---------------------------------------------------------------------------
# TTLCache.config property (L74)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTTLCacheConfig:
    def test_config_property_returns_config(self) -> None:
        cfg = CacheConfig(cache_duration_seconds=42, max_cache_size=5)
        cache: TTLCache[str] = TTLCache(config=cfg)
        assert cache.config is cfg
        assert cache.config.cache_duration_seconds == 42
        assert cache.config.max_cache_size == 5


# ---------------------------------------------------------------------------
# TTLCache.get() expired entry (L94-97)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTTLCacheExpiry:
    def test_expired_entry_returns_none_and_counts_miss(self) -> None:
        """An expired entry is deleted and counted as a miss (L94-97)."""
        cfg = CacheConfig(cache_duration_seconds=1, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config=cfg)
        cache.set("k", "v")

        # Verify it's there first
        assert cache.get("k") == "v"
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

        # Expire the entry by patching time
        with patch.object(time, "time", return_value=time.time() + 100):
            result = cache.get("k")

        assert result is None
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        # Entry should have been deleted
        assert cache.size == 0

    def test_expired_entry_with_closeable_calls_close(self) -> None:
        """Expired closeable values get close() called."""

        class FakeCloseable:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        cfg = CacheConfig(cache_duration_seconds=1, max_cache_size=10)
        cache: TTLCache[FakeCloseable] = TTLCache(config=cfg)
        obj = FakeCloseable()
        cache.set("k", obj)

        with patch.object(time, "time", return_value=time.time() + 100):
            result = cache.get("k")

        assert result is None
        assert obj.closed is True


# ---------------------------------------------------------------------------
# TTLCache.clear with reset_stats=True (L184-185)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTTLCacheClearResetStats:
    def test_clear_with_reset_stats_zeroes_counters(self) -> None:
        cfg = CacheConfig(cache_duration_seconds=300, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config=cfg)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # hit
        cache.get("missing")  # miss

        stats_before = cache.stats()
        assert stats_before["hits"] == 1
        assert stats_before["misses"] == 1
        assert stats_before["size"] == 2

        cache.clear(reset_stats=True)

        stats_after = cache.stats()
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0
        assert stats_after["size"] == 0

    def test_clear_without_reset_stats_keeps_counters(self) -> None:
        cfg = CacheConfig(cache_duration_seconds=300, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config=cfg)
        cache.set("a", "1")
        cache.get("a")  # hit
        cache.get("x")  # miss

        cache.clear(reset_stats=False)

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 0
