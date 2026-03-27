"""Tests for config isolation: reset_config, config_context, and the autouse fixture."""

from __future__ import annotations

import asyncio
import threading

import pytest

from radiant_harness.config import CacheConfig
from radiant_harness.config import HarnessConfig
from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import _validate_base_url
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

    def test_thread_contexts_do_not_leak_into_each_other(self) -> None:
        first_seen: list[int] = []
        second_seen: list[int] = []
        entered = threading.Barrier(2)

        def first_worker() -> None:
            with config_context(HarnessConfig(cache=CacheConfig(max_cache_size=111))):
                entered.wait()
                first_seen.append(get_config().cache.max_cache_size)

        def second_worker() -> None:
            with config_context(HarnessConfig(cache=CacheConfig(max_cache_size=222))):
                entered.wait()
                second_seen.append(get_config().cache.max_cache_size)

        first = threading.Thread(target=first_worker)
        second = threading.Thread(target=second_worker)
        first.start()
        second.start()
        first.join()
        second.join()

        assert first_seen == [111]
        assert second_seen == [222]
        assert get_config().cache.max_cache_size == CacheConfig().max_cache_size

    @pytest.mark.asyncio
    async def test_async_task_contexts_do_not_leak_into_each_other(self) -> None:
        first_ready = asyncio.Event()
        second_ready = asyncio.Event()

        async def first_task() -> int:
            with config_context(HarnessConfig(cache=CacheConfig(max_cache_size=111))):
                first_ready.set()
                await second_ready.wait()
                return get_config().cache.max_cache_size

        async def second_task() -> int:
            await first_ready.wait()
            with config_context(HarnessConfig(cache=CacheConfig(max_cache_size=222))):
                second_ready.set()
                return get_config().cache.max_cache_size

        first_result, second_result = await asyncio.gather(first_task(), second_task())
        assert first_result == 111
        assert second_result == 222
        assert get_config().cache.max_cache_size == CacheConfig().max_cache_size


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


class TestImageProcessingConfigValidation:
    """Test all __post_init__ validation branches (config.py lines 61-114)."""

    def test_min_image_size_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_image_size must be >= 1"):
            ImageProcessingConfig(min_image_size=0)

    def test_max_dimension_below_min_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_image_dimension.*must be >= min_image_size"):
            ImageProcessingConfig(min_image_size=100, max_image_dimension=50)

    def test_min_zoom_ge_max_zoom_raises(self) -> None:
        with pytest.raises(ValueError, match="min_zoom_factor.*must be < max_zoom_factor"):
            ImageProcessingConfig(min_zoom_factor=5.0, max_zoom_factor=2.0)

    def test_min_contrast_ge_max_contrast_raises(self) -> None:
        with pytest.raises(ValueError, match="min_contrast_factor.*must be < max_contrast_factor"):
            ImageProcessingConfig(min_contrast_factor=3.0, max_contrast_factor=0.5)

    def test_jpeg_quality_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="default_jpeg_quality must be between 1 and 100"):
            ImageProcessingConfig(default_jpeg_quality=0)

    def test_jpeg_quality_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="default_jpeg_quality must be between 1 and 100"):
            ImageProcessingConfig(default_jpeg_quality=101)

    def test_min_brightness_ge_max_brightness_raises(self) -> None:
        with pytest.raises(
            ValueError, match="min_brightness_factor.*must be < max_brightness_factor"
        ):
            ImageProcessingConfig(min_brightness_factor=3.0, max_brightness_factor=0.5)

    def test_min_sharpness_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_sharpness_factor must be >= 0"):
            ImageProcessingConfig(min_sharpness_factor=-1.0)

    def test_min_sharpness_ge_max_sharpness_raises(self) -> None:
        with pytest.raises(
            ValueError, match="min_sharpness_factor.*must be < max_sharpness_factor"
        ):
            ImageProcessingConfig(min_sharpness_factor=5.0, max_sharpness_factor=1.0)

    def test_grid_divisions_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="max_grid_divisions must be between 2 and 20"):
            ImageProcessingConfig(max_grid_divisions=1)

    def test_grid_divisions_above_twenty_raises(self) -> None:
        with pytest.raises(ValueError, match="max_grid_divisions must be between 2 and 20"):
            ImageProcessingConfig(max_grid_divisions=21)

    def test_gaussian_sigma_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_gaussian_sigma must be > 0"):
            ImageProcessingConfig(min_gaussian_sigma=0.0)

    def test_gaussian_sigma_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_gaussian_sigma.*must be < max_gaussian_sigma"):
            ImageProcessingConfig(min_gaussian_sigma=6.0, max_gaussian_sigma=1.0)

    def test_morphological_iterations_zero_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_morphological_iterations must be between 1 and 20"
        ):
            ImageProcessingConfig(max_morphological_iterations=0)

    def test_morphological_iterations_above_twenty_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_morphological_iterations must be between 1 and 20"
        ):
            ImageProcessingConfig(max_morphological_iterations=21)

    def test_clahe_clip_min_ge_max_raises(self) -> None:
        with pytest.raises(
            ValueError, match="min_clahe_clip_limit.*must be < max_clahe_clip_limit"
        ):
            ImageProcessingConfig(min_clahe_clip_limit=10.0, max_clahe_clip_limit=1.0)


class TestValidateBaseUrl:
    """Test _validate_base_url edge cases (config.py line 190)."""

    def test_missing_hostname_raises(self) -> None:
        with pytest.raises(ValueError, match="has no hostname"):
            _validate_base_url("https:///path/only", "test_field")
