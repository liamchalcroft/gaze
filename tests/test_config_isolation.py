"""Tests for config isolation: reset_config, config_context, and the autouse fixture."""

from __future__ import annotations

import pytest

from radiant_harness.config import CacheConfig
from radiant_harness.config import HarnessConfig
from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import config_context
from radiant_harness.config import get_config
from radiant_harness.config import reset_config
from radiant_harness.config import set_config


class TestResetConfig:
    """Tests for reset_config()."""

    def test_restores_defaults(self) -> None:
        custom = HarnessConfig(cache=CacheConfig(max_cache_size=999))
        set_config(custom)
        assert get_config().cache.max_cache_size == 999

        reset_config()
        assert get_config().cache.max_cache_size == CacheConfig().max_cache_size

    def test_idempotent(self) -> None:
        reset_config()
        first = get_config()
        reset_config()
        second = get_config()
        assert first == second


class TestConfigContext:
    """Tests for config_context() context manager."""

    def test_applies_temporary_config(self) -> None:
        original_max = get_config().cache.max_cache_size
        temporary = HarnessConfig(cache=CacheConfig(max_cache_size=42))

        with config_context(temporary):
            assert get_config().cache.max_cache_size == 42

        assert get_config().cache.max_cache_size == original_max

    def test_restores_on_exception(self) -> None:
        original = get_config()
        temporary = HarnessConfig(cache=CacheConfig(max_cache_size=1))

        with pytest.raises(RuntimeError, match="boom"), config_context(temporary):
            assert get_config().cache.max_cache_size == 1
            raise RuntimeError("boom")

        assert get_config() is original

    def test_yields_the_temporary_config(self) -> None:
        temporary = HarnessConfig(image=ImageProcessingConfig(max_image_dimension=256))

        with config_context(temporary) as cfg:
            assert cfg is temporary
            assert cfg.image.max_image_dimension == 256

    def test_nesting(self) -> None:
        outer = HarnessConfig(cache=CacheConfig(max_cache_size=10))
        inner = HarnessConfig(cache=CacheConfig(max_cache_size=20))
        original_max = get_config().cache.max_cache_size

        with config_context(outer):
            assert get_config().cache.max_cache_size == 10
            with config_context(inner):
                assert get_config().cache.max_cache_size == 20
            assert get_config().cache.max_cache_size == 10

        assert get_config().cache.max_cache_size == original_max


class TestAutouseFixture:
    """Verify the conftest autouse fixture prevents config leakage.

    These tests MUST run in order (test_a then test_b) to demonstrate
    that mutations in test_a don't leak to test_b.
    """

    def test_a_mutate_config(self) -> None:
        """Mutate the global config — the fixture should clean up after."""
        set_config(HarnessConfig(cache=CacheConfig(max_cache_size=777)))
        assert get_config().cache.max_cache_size == 777

    def test_b_config_is_default(self) -> None:
        """Config must be back to defaults — proves the fixture ran."""
        assert get_config().cache.max_cache_size == CacheConfig().max_cache_size
