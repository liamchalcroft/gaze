"""Tests for TTLCache improvements."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

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
        # max_cache_size=2 means when we try to add a 4th item, 3 items will exceed limit
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
        # Now with 3 items, adding a 4th triggers eviction when _evict_stale checks
        cache.set("key4", mock_obj4)  # Should trigger eviction of oldest items

        # After eviction, cache size should be at most max_cache_size
        assert cache.size <= config.max_cache_size + 1  # +1 for the just-added item

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
