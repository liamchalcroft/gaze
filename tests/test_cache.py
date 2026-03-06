"""Tests for TTLCache improvements."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from radiant_harness.cache import TTLCache
from radiant_harness.config import CacheConfig


class TestTTLCache:
    """Test TTLCache functionality with cleanup."""

    def test_cache_cleanup_on_clear(self):
        """Test that cached objects are cleaned up on clear."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=10)
        cache = TTLCache(config)

        # Create mock objects with close method
        mock_obj1 = MagicMock()
        mock_obj2 = MagicMock()
        mock_obj3 = MagicMock()  # No close method

        # Add objects to cache
        cache.set("key1", mock_obj1)
        cache.set("key2", mock_obj2)
        cache.set("key3", mock_obj3)

        # Clear cache
        cache.clear()

        # Verify close was called on objects that have it
        mock_obj1.close.assert_called_once()
        mock_obj2.close.assert_called_once()
        # mock_obj3 has no close method, so no assertion

    def test_cache_cleanup_on_eviction(self):
        """Test that expired objects are cleaned up on eviction."""
        config = CacheConfig(cache_duration_seconds=0.1, max_cache_size=2)  # Short TTL
        cache = TTLCache(config)

        mock_obj1 = MagicMock()
        mock_obj2 = MagicMock()
        mock_obj3 = MagicMock()

        # Add objects
        cache.set("key1", mock_obj1)
        cache.set("key2", mock_obj2)

        # Wait for expiration
        time.sleep(0.2)

        # Add another object to trigger eviction
        cache.set("key3", mock_obj3)

        # Try to get expired objects - should trigger cleanup
        assert cache.get("key1") is None
        assert cache.get("key2") is None

        # Verify cleanup was called
        mock_obj1.close.assert_called_once()
        mock_obj2.close.assert_called_once()

    def test_cache_size_limit_eviction(self):
        """Test that oldest objects are evicted when size limit is reached."""
        # max_cache_size=2 means when we try to add a 3rd item, size exceeds limit
        # With evict_ratio=0.5, target_size = 2 * (1-0.5) = 1
        # So we need to go from 3 items to 1 item (evict 2)
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=2, evict_ratio=0.5)
        cache = TTLCache(config)

        mock_obj1 = MagicMock()
        mock_obj2 = MagicMock()
        mock_obj3 = MagicMock()
        mock_obj4 = MagicMock()

        # Fill cache
        cache.set("key1", mock_obj1)  # len=1 (no eviction check)
        time.sleep(0.01)
        cache.set("key2", mock_obj2)  # len=2 (no eviction, not > 2)
        time.sleep(0.01)
        cache.set("key3", mock_obj3)  # len=3 (eviction checks when > 2)
        time.sleep(0.01)
        # Now with 3 items, adding a 4th keeps size within limit after eviction
        cache.set("key4", mock_obj4)

        # After eviction, cache size should be at most max_cache_size
        assert cache.size <= config.max_cache_size

    def test_cache_error_handling_during_cleanup(self):
        """Test that cleanup errors (OSError/IOError) are handled gracefully."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=2)
        cache = TTLCache(config)

        # Create object that raises OSError on close (the type we catch)
        mock_obj = MagicMock()
        mock_obj.close.side_effect = OSError("Cleanup failed")

        cache.set("key", mock_obj)
        cache.clear()  # Should not raise exception

        # Error should be logged but not raised
        mock_obj.close.assert_called_once()

    def test_cache_closes_values_on_size_limit_eviction(self):
        """Test that closeable values are properly closed on size limit eviction."""
        # max_cache_size=2, evict_ratio=0.5 means:
        # - eviction triggers when cache size > 2 (i.e., at 3 items)
        # - target after eviction = 2 * (1 - 0.5) = 1 item
        config = CacheConfig(max_cache_size=2, evict_ratio=0.5, cache_duration_seconds=3600)
        cache: TTLCache[MagicMock] = TTLCache(config)

        mock1 = MagicMock()
        mock2 = MagicMock()
        mock3 = MagicMock()

        cache.set("key1", mock1)  # size=1
        time.sleep(0.01)  # Ensure ordering
        cache.set("key2", mock2)  # size=2
        time.sleep(0.01)
        cache.set("key3", mock3)  # size=3, but eviction checks BEFORE add (size=2, not > 2)
        time.sleep(0.01)

        # Eviction happens on key3; adding key4 should not evict further
        mock4 = MagicMock()
        cache.set("key4", mock4)

        # mock1 should have been closed during eviction (it's the oldest)
        mock1.close.assert_called_once()
        # mock2 should also have been closed (target_size=1, need to remove 2 items)
        mock2.close.assert_called_once()

    def test_cache_get_returns_none_for_missing_key(self):
        """Test that get returns None for non-existent keys."""
        config = CacheConfig(cache_duration_seconds=60)
        cache: TTLCache[str] = TTLCache(config)

        assert cache.get("nonexistent") is None

    def test_cache_delete_returns_correct_status(self):
        """Test that delete returns True for existing keys, False for missing."""
        config = CacheConfig(cache_duration_seconds=60)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("key", "value")

        assert cache.delete("key") is True
        assert cache.delete("key") is False
        assert cache.delete("nonexistent") is False

    def test_replacing_existing_key_closes_old_value(self):
        """Overwriting a key must clean up the previous closeable value."""
        config = CacheConfig(cache_duration_seconds=60)
        cache: TTLCache[MagicMock] = TTLCache(config)

        first = MagicMock()
        second = MagicMock()

        cache.set("key", first)
        cache.set("key", second)

        first.close.assert_called_once()
        second.close.assert_not_called()

    def test_cache_has_respects_expiration(self):
        """Test that has() returns False for expired entries."""
        config = CacheConfig(cache_duration_seconds=0.1)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("key", "value")
        assert cache.has("key") is True

        time.sleep(0.2)
        assert cache.has("key") is False

    def test_cache_stats(self):
        """Test that stats returns correct values including hit/miss rates."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Generate some hits and misses
        cache.get("key1")  # hit
        cache.get("key2")  # hit
        cache.get("key3")  # miss (doesn't exist)
        cache.get("key1")  # hit

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["duration_seconds"] == 60
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75  # 3 / 4 = 0.75

    def test_cache_stats_reset(self):
        """Test that stats can be reset."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        # Reset stats
        cache.reset_stats()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Cache entry should still exist
        assert cache.get("key1") == "value1"
        stats = cache.stats()
        assert stats["hits"] == 1

    def test_cache_config_validation_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            CacheConfig(max_cache_size=0)
        with pytest.raises(ValueError):
            CacheConfig(cache_duration_seconds=0)
        with pytest.raises(ValueError):
            CacheConfig(evict_ratio=-0.1)
        with pytest.raises(ValueError):
            CacheConfig(evict_ratio=1.1)
        with pytest.raises(ValueError):
            CacheConfig(evict_ratio=0.0)
        with pytest.raises(ValueError):
            CacheConfig(evict_ratio=1.0)


class TestTTLCacheStress:
    """Stress and concurrency tests for TTLCache."""

    def test_concurrent_threads_no_corruption(self):
        """Many threads doing get/set simultaneously must not corrupt state."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=50, evict_ratio=0.3)
        cache: TTLCache[int] = TTLCache(config)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    cache.set(f"key-{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    cache.get(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=writer, args=(200,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(100,)),
            threading.Thread(target=reader, args=(200,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised exceptions: {errors}"
        assert cache.size <= config.max_cache_size
        stats = cache.stats()
        assert stats["hits"] + stats["misses"] > 0

    def test_eviction_under_pressure(self):
        """Filling cache far beyond max_size stays bounded after each set."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=20, evict_ratio=0.5)
        cache: TTLCache[int] = TTLCache(config)

        for i in range(500):
            cache.set(f"key-{i}", i)

        assert cache.size <= config.max_cache_size

    def test_mixed_ttl_expiry_and_size_eviction(self):
        """Entries expire by TTL while new entries trigger size eviction."""
        config = CacheConfig(cache_duration_seconds=0.1, max_cache_size=10, evict_ratio=0.3)
        cache: TTLCache[int] = TTLCache(config)

        # Fill with entries that will expire
        for i in range(10):
            cache.set(f"old-{i}", i)

        time.sleep(0.15)

        # Insert new entries; stale ones should be cleaned first
        for i in range(15):
            cache.set(f"new-{i}", i)

        assert cache.size <= config.max_cache_size

        # Old entries must all be gone
        for i in range(10):
            assert cache.get(f"old-{i}") is None

        # Most recent entries should be retrievable
        assert cache.get(f"new-{14}") == 14

    def test_has_does_not_affect_stats(self):
        """has() must never change hit/miss counters."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=10)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("a", "val")

        cache.has("a")  # exists
        cache.has("missing")  # does not exist

        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_concurrent_delete_and_get(self):
        """Concurrent delete and get on the same keys must not raise."""
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=100)
        cache: TTLCache[int] = TTLCache(config)
        errors: list[Exception] = []

        for i in range(100):
            cache.set(f"key-{i}", i)

        def deleter() -> None:
            try:
                for i in range(100):
                    cache.delete(f"key-{i}")
            except Exception as e:
                errors.append(e)

        def getter() -> None:
            try:
                for i in range(100):
                    cache.get(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=deleter),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised exceptions: {errors}"
