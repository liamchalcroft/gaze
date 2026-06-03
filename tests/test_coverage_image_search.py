"""Coverage tests for retrieval/image_search.py uncovered paths.

Targets: MedicalImageSearchManager init validation (498-505),
search() caching/filtering/dedup (577-640), context manager (521-530),
close() cleanup, ImageSearchResult metadata.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from gaze.retrieval.image_search import ImageSearchError
from gaze.retrieval.image_search import ImageSearchResult
from gaze.retrieval.image_search import MedicalImageSearchManager
from gaze.retrieval.image_search import OpenISearchEngine

# ---------------------------------------------------------------------------
# ImageSearchResult edge cases
# ---------------------------------------------------------------------------


class TestImageSearchResultMetadata:
    def test_dict_metadata_frozen(self) -> None:
        result = ImageSearchResult(
            title="Brain MRI",
            image_url="https://example.com/img.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
            metadata={"key": "value", "nested": [1, 2]},
        )
        assert isinstance(result.metadata, MappingProxyType)
        assert result.metadata["key"] == "value"
        assert result.metadata["nested"] == (1, 2)  # list frozen to tuple

    def test_empty_metadata_default(self) -> None:
        result = ImageSearchResult(
            title="CT Scan",
            image_url="https://example.com/ct.jpg",
            thumbnail_url=None,
            source_url="https://example.com",
            source="openi",
        )
        assert isinstance(result.metadata, MappingProxyType)
        assert len(result.metadata) == 0


# ---------------------------------------------------------------------------
# MedicalImageSearchManager.__init__ validation (lines 498-505)
# ---------------------------------------------------------------------------


class TestManagerInitValidation:
    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown image search engine: 'bing'"):
            MedicalImageSearchManager(engines=["bing"])

    def test_default_engine_is_openi(self) -> None:
        mgr = MedicalImageSearchManager()
        assert len(mgr.engines) == 1
        assert isinstance(mgr.engines[0], OpenISearchEngine)

    def test_negative_max_results_raises(self) -> None:
        with pytest.raises(ValueError, match="max_results_per_engine must be >= 1"):
            MedicalImageSearchManager(max_results_per_engine=0)

    def test_negative_rate_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_limit_delay must be >= 0"):
            MedicalImageSearchManager(rate_limit_delay=-1.0)


# ---------------------------------------------------------------------------
# Context manager protocol (lines 521-530)
# ---------------------------------------------------------------------------


class TestContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        mgr = MedicalImageSearchManager()
        async with mgr as ctx:
            assert ctx is mgr

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self) -> None:
        mgr = MedicalImageSearchManager()
        mgr.close = AsyncMock()
        async with mgr:
            pass
        mgr.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# search() — caching, filtering, dedup (lines 577-640)
# ---------------------------------------------------------------------------


def _make_result(
    title: str,
    url: str,
    modality: str | None = None,
    body_part: str | None = None,
) -> ImageSearchResult:
    return ImageSearchResult(
        title=title,
        image_url=url,
        thumbnail_url=None,
        source_url="https://example.com",
        source="openi",
        modality=modality,
        body_part=body_part,
    )


class TestSearch:
    @pytest.mark.asyncio
    async def test_empty_query_raises(self) -> None:
        mgr = MedicalImageSearchManager()
        with pytest.raises(ValueError, match="non-empty string"):
            await mgr.search("")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises(self) -> None:
        mgr = MedicalImageSearchManager()
        with pytest.raises(ValueError, match="non-empty string"):
            await mgr.search("   ")

    @pytest.mark.asyncio
    async def test_basic_search_returns_results(self) -> None:
        mgr = MedicalImageSearchManager()
        results = [
            _make_result("Brain MRI", "https://img1.com/a.jpg", "MRI", "brain"),
            _make_result("Chest CT", "https://img2.com/b.jpg", "CT", "chest"),
        ]
        mgr.engines[0].search = AsyncMock(return_value=results)

        found = await mgr.search("brain tumor")
        assert len(found) == 2
        assert found[0].title == "Brain MRI"
        assert found[1].title == "Chest CT"

    @pytest.mark.asyncio
    async def test_modality_filter(self) -> None:
        mgr = MedicalImageSearchManager()
        results = [
            _make_result("Brain MRI", "https://a.com/1.jpg", "MRI", "brain"),
            _make_result("Chest CT", "https://a.com/2.jpg", "CT", "chest"),
            _make_result("Hand X-ray", "https://a.com/3.jpg", "X-ray", "hand"),
        ]
        mgr.engines[0].search = AsyncMock(return_value=results)

        found = await mgr.search("scan", modality="MRI")
        assert len(found) == 1
        assert found[0].modality == "MRI"

    @pytest.mark.asyncio
    async def test_body_part_filter(self) -> None:
        mgr = MedicalImageSearchManager()
        results = [
            _make_result("Brain MRI", "https://a.com/1.jpg", "MRI", "brain"),
            _make_result("Chest CT", "https://a.com/2.jpg", "CT", "chest"),
        ]
        mgr.engines[0].search = AsyncMock(return_value=results)

        found = await mgr.search("scan", body_part="chest")
        assert len(found) == 1
        assert found[0].body_part == "chest"

    @pytest.mark.asyncio
    async def test_dedup_by_url(self) -> None:
        mgr = MedicalImageSearchManager()
        dup_url = "https://a.com/same.jpg"
        results = [
            _make_result("First", dup_url, "MRI"),
            _make_result("Second", dup_url, "MRI"),
            _make_result("Third", "https://a.com/other.jpg", "CT"),
        ]
        mgr.engines[0].search = AsyncMock(return_value=results)

        found = await mgr.search("brain")
        assert len(found) == 2
        assert found[0].title == "First"
        assert found[1].title == "Third"

    @pytest.mark.asyncio
    async def test_cache_returns_same_results(self) -> None:
        mgr = MedicalImageSearchManager()
        results = [_make_result("Cached", "https://a.com/c.jpg")]
        mgr.engines[0].search = AsyncMock(return_value=results)

        first = await mgr.search("test query")
        second = await mgr.search("test query")

        assert first == second
        mgr.engines[0].search.assert_awaited_once()  # only 1 real call

    @pytest.mark.asyncio
    async def test_all_engines_fail_raises(self) -> None:
        mgr = MedicalImageSearchManager()
        mgr.engines[0].search = AsyncMock(
            side_effect=ImageSearchError("openi", "connection failed")
        )

        with pytest.raises(ImageSearchError, match="All image search engines failed"):
            await mgr.search("brain MRI")

    @pytest.mark.asyncio
    async def test_partial_engine_failure_returns_results(self) -> None:
        """If one engine fails but another succeeds, results are returned."""
        mgr = MedicalImageSearchManager()
        good_results = [_make_result("Good", "https://a.com/ok.jpg")]

        # Add a second mock engine that fails
        failing_engine = MagicMock()
        failing_engine.name = "failing"
        failing_engine.search = AsyncMock(side_effect=ImageSearchError("failing", "oops"))
        mgr.engines.append(failing_engine)

        # First engine succeeds
        mgr.engines[0].search = AsyncMock(return_value=good_results)

        found = await mgr.search("test")
        assert len(found) == 1
        assert found[0].title == "Good"

    @pytest.mark.asyncio
    async def test_combined_modality_and_body_part_filter(self) -> None:
        mgr = MedicalImageSearchManager()
        results = [
            _make_result("Brain MRI", "https://a.com/1.jpg", "MRI", "brain"),
            _make_result("Chest MRI", "https://a.com/2.jpg", "MRI", "chest"),
            _make_result("Brain CT", "https://a.com/3.jpg", "CT", "brain"),
        ]
        mgr.engines[0].search = AsyncMock(return_value=results)

        found = await mgr.search("scan", modality="MRI", body_part="brain")
        assert len(found) == 1
        assert found[0].title == "Brain MRI"


# ---------------------------------------------------------------------------
# close() cleanup
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up_resources(self) -> None:
        mgr = MedicalImageSearchManager()
        mock_engine = AsyncMock()
        mgr.engines = [mock_engine]

        await mgr.close()
        mock_engine.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_temp_dir_cleaned(self, tmp_path: Path) -> None:
        mgr = MedicalImageSearchManager()
        assert mgr._created_temp_dir is True
        temp_dir = mgr.download_dir
        assert temp_dir.exists()

        await mgr.close()
        assert not temp_dir.exists()

    @pytest.mark.asyncio
    async def test_close_custom_dir_not_removed(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "images"
        custom_dir.mkdir()
        mgr = MedicalImageSearchManager(download_dir=custom_dir)
        assert mgr._created_temp_dir is False

        await mgr.close()
        assert custom_dir.exists()  # custom dir NOT cleaned up
