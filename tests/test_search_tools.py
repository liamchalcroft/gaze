from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import ImageSearchResult
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_search_tools


def _make_registry_with_mock_web(fake_search):
    """Create a registry whose web search manager uses *fake_search*."""
    registry = ToolRegistry(tools=create_search_tools())
    mock_manager = AsyncMock()
    mock_manager.search = fake_search
    registry._web_search_manager = mock_manager
    return registry


def _make_registry_with_mock_images(fake_search):
    """Create a registry whose image search manager uses *fake_search*."""
    registry = ToolRegistry(tools=create_search_tools())
    mock_manager = AsyncMock()
    mock_manager.search = fake_search
    registry._image_search_manager = mock_manager
    return registry


@pytest.mark.asyncio
async def test_search_web_tool_formats_results() -> None:
    async def fake_search(query, *, search_type="general"):
        return [
            SearchResult(
                title="Glioblastoma MRI",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                content="c" * 600,
                snippet="A short summary.",
                source="pubmed",
                reliability_score=0.95,
                publication_date="2024",
                content_type="article",
                medical_relevance=0.9,
                extracted_entities=("glioblastoma",),
                open_access=True,
            )
        ]

    registry = _make_registry_with_mock_web(fake_search)
    result = await registry.execute("search_web", query="glioblastoma MRI", search_type="diagnosis")

    assert result.success
    assert result.metadata["results_count"] == 1
    assert result.formatted_results is not None
    assert "## PubMed Search Results" in result.formatted_results
    assert "Glioblastoma MRI" in result.formatted_results
    assert "Content:" in result.formatted_results
    assert "..." in result.formatted_results


@pytest.mark.asyncio
async def test_search_web_tool_error_propagates() -> None:
    async def fake_search(query, *, search_type="general"):
        raise SearchError("PubMed", "down")

    registry = _make_registry_with_mock_web(fake_search)
    result = await registry.execute("search_web", query="glioblastoma MRI", search_type="diagnosis")

    assert not result.success
    assert result.error is not None
    assert "PubMed" in result.error
    assert result.metadata["query"] == "glioblastoma MRI"
    assert result.metadata["search_type"] == "diagnosis"


@pytest.mark.asyncio
async def test_search_images_tool_formats_results() -> None:
    async def fake_search(query, *, modality=None, body_part=None):
        return [
            ImageSearchResult(
                title="Reference MRI",
                image_url="https://openi.nlm.nih.gov/img.png",
                thumbnail_url=None,
                source_url="https://openi.nlm.nih.gov/source",
                source="openi",
                modality="MRI",
                body_part="brain",
                caption="caption" * 200,
                article_title="Example Article",
                reliability_score=0.9,
            )
        ]

    registry = _make_registry_with_mock_images(fake_search)
    result = await registry.execute(
        "search_images", query="meningioma", modality="MRI", body_part="brain"
    )

    assert result.success
    assert result.metadata["results_count"] == 1
    assert result.formatted_results is not None
    assert "## Reference Medical Images" in result.formatted_results
    assert "Reference MRI" in result.formatted_results


@pytest.mark.asyncio
async def test_search_images_tool_error_propagates() -> None:
    async def fake_search(query, *, modality=None, body_part=None):
        raise ImageSearchError("Open-i", "down")

    registry = _make_registry_with_mock_images(fake_search)
    result = await registry.execute(
        "search_images", query="meningioma", modality="MRI", body_part="brain"
    )

    assert not result.success
    assert result.error is not None
    assert "Open-i" in result.error
    assert result.metadata["query"] == "meningioma"
    assert result.metadata["modality"] == "MRI"
    assert result.metadata["body_part"] == "brain"


def test_search_web_schema_enum_matches_backend() -> None:
    """Schema search_type enum must include all types accepted by WebSearchManager."""
    from radiant_harness.retrieval.web_search import WebSearchManager
    from radiant_harness.tools.registry import ToolDocumenter

    tools = create_search_tools()
    doc = ToolDocumenter(tools)
    schemas = doc.get_tool_schemas()

    search_web_schema = next(s for s in schemas if s["function"]["name"] == "search_web")
    schema_enum = set(
        search_web_schema["function"]["parameters"]["properties"]["search_type"]["enum"]
    )
    backend_types = WebSearchManager.ALLOWED_SEARCH_TYPES

    assert schema_enum == backend_types, (
        f"Schema enum {sorted(schema_enum)} != backend types {sorted(backend_types)}"
    )


class TestSessionReuse:
    """Verify search managers are reused across multiple tool calls."""

    @pytest.mark.asyncio
    async def test_web_search_manager_reused_across_calls(self) -> None:
        """Two search_web calls should use the same manager instance."""
        call_count = 0

        async def fake_search(query, *, search_type="general"):
            nonlocal call_count
            call_count += 1
            return []

        registry = ToolRegistry(tools=create_search_tools())
        mock_manager = AsyncMock()
        mock_manager.search = fake_search
        mock_manager.close = AsyncMock()
        registry._web_search_manager = mock_manager

        await registry.execute("search_web", query="test1")
        await registry.execute("search_web", query="test2")

        assert call_count == 2
        # Manager instance should still be the same object
        assert registry._web_search_manager is mock_manager

    @pytest.mark.asyncio
    async def test_image_search_manager_reused_across_calls(self) -> None:
        """Two search_images calls should use the same manager instance."""
        call_count = 0

        async def fake_search(query, *, modality=None, body_part=None):
            nonlocal call_count
            call_count += 1
            return []

        registry = ToolRegistry(tools=create_search_tools())
        mock_manager = AsyncMock()
        mock_manager.search = fake_search
        mock_manager.close = AsyncMock()
        registry._image_search_manager = mock_manager

        await registry.execute("search_images", query="test1")
        await registry.execute("search_images", query="test2")

        assert call_count == 2
        assert registry._image_search_manager is mock_manager

    @pytest.mark.asyncio
    async def test_aclose_cleans_up_managers(self) -> None:
        """aclose() should close all search managers."""
        registry = ToolRegistry(tools=create_search_tools())

        mock_web = AsyncMock()
        mock_web.close = AsyncMock()
        mock_img = AsyncMock()
        mock_img.close = AsyncMock()
        registry._web_search_manager = mock_web
        registry._image_search_manager = mock_img

        await registry.aclose()

        mock_web.close.assert_awaited_once()
        mock_img.close.assert_awaited_once()
        assert registry._web_search_manager is None
        assert registry._image_search_manager is None

    @pytest.mark.asyncio
    async def test_aclose_skips_injected_shared_managers(self) -> None:
        """Injected shared managers stay alive until their owner closes them."""
        mock_web = AsyncMock()
        mock_web.close = AsyncMock()
        mock_img = AsyncMock()
        mock_img.close = AsyncMock()

        registry = ToolRegistry(
            tools=create_search_tools(),
            web_search_manager=mock_web,
            image_search_manager=mock_img,
        )

        await registry.aclose()

        mock_web.close.assert_not_awaited()
        mock_img.close.assert_not_awaited()
        assert registry._web_search_manager is None
        assert registry._image_search_manager is None


class TestValueErrorHandling:
    """ValueError from manager must be caught, not crash the agentic loop."""

    @pytest.mark.asyncio
    async def test_search_web_invalid_search_type_returns_error(self) -> None:
        async def fake_search(query, *, search_type="general"):
            raise ValueError(f"search_type must be one of ..., got '{search_type}'")

        registry = _make_registry_with_mock_web(fake_search)
        result = await registry.execute("search_web", query="test", search_type="invalid")

        assert not result.success
        assert result.error is not None
        assert "search_type" in result.error

    @pytest.mark.asyncio
    async def test_search_web_empty_query_returns_error(self) -> None:
        async def fake_search(query, *, search_type="general"):
            raise ValueError("query must be a non-empty string")

        registry = _make_registry_with_mock_web(fake_search)
        result = await registry.execute("search_web", query="")

        assert not result.success
        assert result.error is not None
        assert "query" in result.error

    @pytest.mark.asyncio
    async def test_search_images_value_error_returns_error(self) -> None:
        async def fake_search(query, *, modality=None, body_part=None):
            raise ValueError("query must be a non-empty string")

        registry = _make_registry_with_mock_images(fake_search)
        result = await registry.execute("search_images", query="")

        assert not result.success
        assert result.error is not None
        assert "query" in result.error


class TestFormattedOutputSizeLimit:
    """Formatted output must respect max_content_for_llm."""

    @pytest.mark.asyncio
    async def test_web_search_output_capped(self) -> None:
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import SearchConfig
        from radiant_harness.config import config_context

        async def fake_search(query, *, search_type="general"):
            return [
                SearchResult(
                    title=f"Article {i} with a sufficiently long title",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
                    content="x" * 1000,
                    snippet="summary",
                    source="pubmed",
                    reliability_score=0.9,
                    publication_date="2024",
                    content_type="article",
                    medical_relevance=0.8,
                )
                for i in range(10)
            ]

        with config_context(HarnessConfig(search=SearchConfig(max_content_for_llm=500))):
            registry = _make_registry_with_mock_web(fake_search)
            result = await registry.execute("search_web", query="test")

        assert result.success
        assert result.formatted_results is not None
        assert "results truncated]" in result.formatted_results

    @pytest.mark.asyncio
    async def test_image_search_output_capped(self) -> None:
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import SearchConfig
        from radiant_harness.config import config_context

        async def fake_search(query, *, modality=None, body_part=None):
            return [
                ImageSearchResult(
                    title=f"Image {i}",
                    image_url=f"https://openi.nlm.nih.gov/img{i}.png",
                    thumbnail_url=None,
                    source_url=f"https://openi.nlm.nih.gov/source{i}",
                    source="openi",
                    caption="y" * 1000,
                )
                for i in range(10)
            ]

        with config_context(HarnessConfig(search=SearchConfig(max_content_for_llm=500))):
            registry = _make_registry_with_mock_images(fake_search)
            result = await registry.execute("search_images", query="test")

        assert result.success
        assert result.formatted_results is not None
        assert "results truncated]" in result.formatted_results

    @pytest.mark.asyncio
    async def test_content_preview_uses_config(self) -> None:
        from radiant_harness.config import HarnessConfig
        from radiant_harness.config import SearchConfig
        from radiant_harness.config import config_context

        async def fake_search(query, *, search_type="general"):
            return [
                SearchResult(
                    title="Short title",
                    url="https://pubmed.ncbi.nlm.nih.gov/1/",
                    content="A" * 200,
                    snippet="summary",
                    source="pubmed",
                    reliability_score=0.9,
                )
            ]

        with config_context(HarnessConfig(search=SearchConfig(max_content_preview_length=50))):
            registry = _make_registry_with_mock_web(fake_search)
            result = await registry.execute("search_web", query="test")

        assert result.success
        assert result.formatted_results is not None
        # Content should be truncated to 50 chars + "..."
        assert "..." in result.formatted_results
        # The full 200-char content should NOT appear
        assert "A" * 200 not in result.formatted_results
