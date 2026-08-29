"""Tests for TTLCache improvements."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

import gaze.cache
from gaze.cache import TTLCache
from gaze.config import CacheConfig


class _FakeClock:
    """Deterministic stand-in for the module-level time source.

    ``TTLCache`` reads the current time via ``gaze.cache.time.time()``. Tests
    monkeypatch that reference with an instance of this class so TTL expiry and
    insert ordering can be driven explicitly, with no real ``time.sleep``.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeClock]:
    """Patch the cache's time source with a controllable fake clock."""
    fake = _FakeClock()
    monkeypatch.setattr(gaze.cache, "time", fake)
    yield fake


class TestTTLCache:
    """Test TTLCache functionality with cleanup."""

    # mock_obj3 has no close method, so no assertion

    def test_cache_size_limit_eviction(self, clock: _FakeClock):
        """Test that oldest objects are evicted when size limit is reached."""
        # max_cache_size=2 means when we try to add a 3rd item, size exceeds limit
        # With evict_ratio=0.5, target_size = 2 * (1-0.5) = 1
        # So we need to go from 3 items to 1 item (evict 2)
        config = CacheConfig(cache_duration_seconds=60, max_cache_size=2, evict_ratio=0.5)
        cache: TTLCache[str] = TTLCache(config)

        # Fill cache. Advance the clock between sets so each entry has a distinct
        # timestamp, making oldest-first eviction deterministic.
        for i in range(1, 5):
            cache.set(f"key{i}", f"value{i}")
            clock.advance(1)

        # After eviction, cache size should be at most max_cache_size
        assert cache.size <= config.max_cache_size

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

    def test_cache_has_respects_expiration(self, clock: _FakeClock):
        """Test that has() returns False for expired entries."""
        config = CacheConfig(cache_duration_seconds=0.1)
        cache: TTLCache[str] = TTLCache(config)

        cache.set("key", "value")
        assert cache.has("key") is True

        # Not yet expired: still within the TTL window.
        clock.advance(0.05)
        assert cache.has("key") is True

        # Past the TTL: entry is now expired.
        clock.advance(0.06)
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

    def test_mixed_ttl_expiry_and_size_eviction(self, clock: _FakeClock):
        """Entries expire by TTL while new entries trigger size eviction."""
        config = CacheConfig(cache_duration_seconds=0.1, max_cache_size=10, evict_ratio=0.3)
        cache: TTLCache[int] = TTLCache(config)

        # Fill with entries that will expire
        for i in range(10):
            cache.set(f"old-{i}", i)

        # Advance past the TTL so every "old-*" entry is now stale.
        clock.advance(0.15)

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


class TestTTLCacheConfigAndStats:
    """Config exposure and hit/miss accounting."""

    def test_config_property_returns_the_instance_it_was_built_with(self) -> None:
        cfg = CacheConfig(cache_duration_seconds=42, max_cache_size=5)
        cache: TTLCache[str] = TTLCache(config=cfg)
        assert cache.config is cfg

    def test_expired_entry_is_deleted_and_counted_as_a_miss(self, clock: _FakeClock) -> None:
        cache: TTLCache[str] = TTLCache(CacheConfig(cache_duration_seconds=1, max_cache_size=10))
        cache.set("k", "v")
        assert cache.get("k") == "v"

        clock.advance(100)
        assert cache.get("k") is None

        stats = cache.stats()
        assert (stats["hits"], stats["misses"]) == (1, 1)
        assert cache.size == 0

    def test_clear_with_reset_stats_zeroes_counters(self) -> None:
        cache: TTLCache[str] = TTLCache(CacheConfig(cache_duration_seconds=300, max_cache_size=10))
        cache.set("a", "1")
        cache.get("a")
        cache.get("missing")

        cache.clear(reset_stats=True)

        stats = cache.stats()
        assert (stats["hits"], stats["misses"], stats["size"]) == (0, 0, 0)

    def test_clear_without_reset_stats_keeps_counters(self) -> None:
        cache: TTLCache[str] = TTLCache(CacheConfig(cache_duration_seconds=300, max_cache_size=10))
        cache.set("a", "1")
        cache.get("a")
        cache.get("x")

        cache.clear(reset_stats=False)

        stats = cache.stats()
        assert (stats["hits"], stats["misses"], stats["size"]) == (1, 1, 0)
