from __future__ import annotations

import pytest

import radiant_harness.tools.search as search_tools
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import ImageSearchResult
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_search_tools


@pytest.mark.asyncio
async def test_search_web_tool_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str, max_results: int = 5, search_type: str = "general"
    ) -> list[SearchResult]:
        _ = query, max_results, search_type
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
                extracted_entities=["glioblastoma"],
                open_access=True,
            )
        ]

    monkeypatch.setattr(search_tools, "search_medical_literature", fake_search)

    registry = ToolRegistry(tools=create_search_tools())
    result = await registry.execute("search_web", query="glioblastoma MRI", search_type="diagnosis")

    assert result.success
    assert result.metadata["results_count"] == 1
    assert result.formatted_results is not None
    assert "## PubMed Search Results" in result.formatted_results
    assert "Glioblastoma MRI" in result.formatted_results
    assert "Content:" in result.formatted_results
    assert "..." in result.formatted_results


@pytest.mark.asyncio
async def test_search_web_tool_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str, max_results: int = 5, search_type: str = "general"
    ) -> list[SearchResult]:
        _ = query, max_results, search_type
        raise SearchError("PubMed", "down")

    monkeypatch.setattr(search_tools, "search_medical_literature", fake_search)

    registry = ToolRegistry(tools=create_search_tools())
    result = await registry.execute("search_web", query="glioblastoma MRI", search_type="diagnosis")

    assert not result.success
    assert result.error is not None
    assert "PubMed" in result.error
    assert result.metadata["query"] == "glioblastoma MRI"
    assert result.metadata["search_type"] == "diagnosis"


@pytest.mark.asyncio
async def test_search_images_tool_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str,
        max_results: int = 5,
        modality: str | None = None,
        body_part: str | None = None,
    ) -> list[ImageSearchResult]:
        _ = query, max_results, modality, body_part
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

    monkeypatch.setattr(search_tools, "search_medical_images", fake_search)

    registry = ToolRegistry(tools=create_search_tools())
    result = await registry.execute(
        "search_images", query="meningioma", modality="MRI", body_part="brain"
    )

    assert result.success
    assert result.metadata["results_count"] == 1
    assert result.formatted_results is not None
    assert "## Reference Medical Images" in result.formatted_results
    assert "Reference MRI" in result.formatted_results


@pytest.mark.asyncio
async def test_search_images_tool_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str,
        max_results: int = 5,
        modality: str | None = None,
        body_part: str | None = None,
    ) -> list[ImageSearchResult]:
        _ = query, max_results, modality, body_part
        raise ImageSearchError("Open-i", "down")

    monkeypatch.setattr(search_tools, "search_medical_images", fake_search)

    registry = ToolRegistry(tools=create_search_tools())
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
