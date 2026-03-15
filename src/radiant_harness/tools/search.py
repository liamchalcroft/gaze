"""Search tools for evidence-based radiology analysis.

Provides web search and image search tools for retrieving medical literature
and reference images during VLM analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beartype import beartype
from loguru import logger

from radiant_harness.config import get_config
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.web_search import SearchError
from radiant_harness.tools.registry import Tool
from radiant_harness.types import ToolResult

if TYPE_CHECKING:
    from radiant_harness.tools.registry import ToolRegistry


# Note: Private tool execute functions don't use @beartype because they're
# only called internally by ToolRegistry which validates inputs at the public API.
# registry is unused but required by tool executor interface.


async def _execute_search_web(
    registry: ToolRegistry,
    query: str,
    search_type: str = "general",
) -> ToolResult:
    """Search PubMed for medical literature."""
    logger.info(f"Searching PubMed: '{query}' (type: {search_type})")

    try:
        manager = registry.get_web_search_manager()
        search_results = await manager.search(query, search_type=search_type)
    except (SearchError, ValueError) as e:
        return ToolResult(
            tool_name="search_web",
            description=f"PubMed search failed for '{query}'",
            error=str(e),
            metadata={"query": query, "search_type": search_type},
        )

    if not search_results:
        return ToolResult(
            tool_name="search_web",
            description=f"No results found for '{query}'",
            metadata={"query": query, "search_type": search_type, "results_count": 0},
        )

    search_config = get_config().search
    max_preview = search_config.max_content_preview_length
    max_total = search_config.max_content_for_llm

    formatted_results: list[str] = []
    sources: set[str] = set()
    total_reliability = 0.0
    content_types: set[str] = set()
    total_length = len("\n## PubMed Search Results\n\n")

    for i, result in enumerate(search_results, 1):
        sources.add(result.source)
        total_reliability += result.reliability_score
        content_types.add(result.content_type)

        lines = [
            f"{i}. **{result.title}**",
            f"   **Source:** {result.source} (Reliability: {result.reliability_score:.2f})",
            f"   **Type:** {result.content_type}"
            f" | **Open Access:** {'Yes' if result.open_access else 'No'}",
        ]
        if result.publication_date:
            lines.append(f"   **Date:** {result.publication_date}")
        if result.journal:
            lines.append(f"   **Journal:** {result.journal}")
        if result.content and result.content != result.title:
            content_preview = result.content[:max_preview] + (
                "..." if len(result.content) > max_preview else ""
            )
            lines.append(f"   **Content:** {content_preview}")
        if result.extracted_entities:
            lines.append(f"   **Key terms:** {', '.join(result.extracted_entities[:5])}")
        lines.append(f"   **URL:** {result.url}")

        entry = "\n".join(lines)
        entry_cost = len(entry) + 2  # +2 for "\n\n" separator
        if total_length + entry_cost > max_total and formatted_results:
            remaining = len(search_results) - i + 1
            formatted_results.append(f"[{remaining} more results truncated]")
            break
        formatted_results.append(entry)
        total_length += entry_cost

    formatted_summary = "\n## PubMed Search Results\n\n" + "\n\n".join(formatted_results)
    avg_reliability = total_reliability / len(search_results)

    return ToolResult(
        tool_name="search_web",
        description=f"Found {len(search_results)} PubMed articles",
        metadata={
            "query": query,
            "search_type": search_type,
            "results_count": len(search_results),
            "sources": list(sources),
            "avg_reliability": round(avg_reliability, 2),
            "content_types": list(content_types),
            "open_access_count": sum(1 for r in search_results if r.open_access),
            "formatted_results": formatted_summary,
        },
    )


async def _execute_search_images(
    registry: ToolRegistry,
    query: str,
    modality: str = "any",
    body_part: str = "any",
) -> ToolResult:
    """Search NIH Open-i for reference medical images."""
    modality_filter = None if modality == "any" else modality
    body_part_filter = None if body_part == "any" else body_part
    logger.info(
        f"Searching images: '{query}' (modality: {modality_filter}, body: {body_part_filter})"
    )

    try:
        manager = registry.get_image_search_manager()
        search_results = await manager.search(
            query=query,
            modality=modality_filter,
            body_part=body_part_filter,
        )
    except (ImageSearchError, ValueError) as e:
        return ToolResult(
            tool_name="search_images",
            description=f"Image search failed for '{query}'",
            error=str(e),
            metadata={"query": query, "modality": modality_filter, "body_part": body_part_filter},
        )

    if not search_results:
        return ToolResult(
            tool_name="search_images",
            description=f"No images found for '{query}'",
            metadata={
                "query": query,
                "modality": modality_filter,
                "body_part": body_part_filter,
                "results_count": 0,
            },
        )

    search_config = get_config().search
    max_preview = search_config.max_content_preview_length
    max_total = search_config.max_content_for_llm

    formatted_results: list[str] = []
    modalities_found: set[str] = set()
    body_parts_found: set[str] = set()
    total_length = len("\n## Reference Medical Images\n\n")

    for i, result in enumerate(search_results, 1):
        if result.modality:
            modalities_found.add(result.modality)
        if result.body_part:
            body_parts_found.add(result.body_part)

        lines = [
            f"{i}. **{result.title}**",
            f"   **Source:** {result.source} (Reliability: {result.reliability_score:.2f})",
            f"   **Modality:** {result.modality or 'Unknown'}"
            f" | **Body Part:** {result.body_part or 'Unknown'}",
        ]
        if result.caption:
            caption_preview = result.caption[:max_preview] + (
                "..." if len(result.caption) > max_preview else ""
            )
            lines.append(f"   **Caption:** {caption_preview}")
        if result.article_title:
            lines.append(f"   **Article:** {result.article_title[:100]}")
        lines.append(f"   **Image URL:** {result.image_url}")
        lines.append(f"   **Source Article:** {result.source_url}")

        entry = "\n".join(lines)
        entry_cost = len(entry) + 2
        if total_length + entry_cost > max_total and formatted_results:
            remaining = len(search_results) - i + 1
            formatted_results.append(f"[{remaining} more results truncated]")
            break
        formatted_results.append(entry)
        total_length += entry_cost

    formatted_summary = "\n## Reference Medical Images\n\n" + "\n\n".join(formatted_results)

    return ToolResult(
        tool_name="search_images",
        description=f"Found {len(search_results)} reference images",
        metadata={
            "query": query,
            "modality": modality_filter,
            "body_part": body_part_filter,
            "results_count": len(search_results),
            "modalities_found": list(modalities_found),
            "body_parts_found": list(body_parts_found),
            "image_urls": [r.image_url for r in search_results],
            "formatted_results": formatted_summary,
        },
    )


# Prompt documentation for search tools
SEARCH_WEB_PROMPT_DOC = (
    "**search_web** - Search PubMed for medical literature and evidence\n"
    "  - Parameter `query` (string): Medical search terms "
    '(e.g., "glioblastoma MRI imaging characteristics")\n'
    "  - Parameter `search_type` (string): One of "
    '"diagnosis", "guidelines", "research", "anatomy", '
    '"treatment", "differential", "general"\n'
    "  - Use for: Verifying findings, accessing diagnostic "
    "criteria, researching conditions\n"
    "  - Returns: Article titles, abstracts, publication info, "
    "and reliability scores\n"
    "  - Tip: Include condition name, imaging modality, "
    "and specific findings in query"
)

SEARCH_IMAGES_PROMPT_DOC = (
    "**search_images** - Search NIH Open-i for reference "
    "medical images\n"
    "  - Parameter `query` (string): Image search terms "
    '(e.g., "glioblastoma MRI T1 contrast")\n'
    '  - Parameter `modality` (optional): Filter by "MRI", '
    '"CT", "X-ray", "Ultrasound", "PET", "Mammography"\n'
    '  - Parameter `body_part` (optional): Filter by "brain", '
    '"head", "chest", "abdomen", "spine", "pelvis", "cardiac"\n'
    "  - Use for: Finding similar cases, comparison images, "
    "visual references\n"
    "  - Returns: Image URLs with captions, source articles, "
    "and reliability scores"
)


@beartype
def create_search_tools(disabled_tools: set[str] | None = None) -> list[Tool]:
    """Create the standard set of search tools for evidence-based analysis.

    These tools provide access to medical literature and reference images
    for supporting evidence-based analysis during multi-turn reasoning.

    Args:
        disabled_tools: Set of tool names to exclude from the returned list

    Returns:
        List of Tool objects ready for registration with ToolRegistry

    Example:
        # Create all search tools
        tools = create_search_tools()

        # Create tools excluding image search
        tools = create_search_tools(disabled_tools={"search_images"})
    """
    disabled = disabled_tools or set()
    tools: list[Tool] = []

    if "search_web" not in disabled:
        tools.append(
            Tool(
                name="search_web",
                description=(
                    "Search PubMed for medical literature, guidelines, "
                    "research papers, and reference cases."
                ),
                parameters={
                    "query": {
                        "type": "string",
                        "description": (
                            "Medical search query. Include condition, imaging "
                            "modality, and key findings."
                        ),
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search.",
                        "enum": [
                            "diagnosis",
                            "research",
                            "guidelines",
                            "anatomy",
                            "treatment",
                            "differential",
                            "general",
                        ],
                        "default": "general",
                    },
                },
                execute=_execute_search_web,
                requires_image=False,
                prompt_documentation=SEARCH_WEB_PROMPT_DOC,
                category="search",
            )
        )

    if "search_images" not in disabled:
        tools.append(
            Tool(
                name="search_images",
                description=(
                    "Search NIH Open-i for reference medical images. "
                    "Returns image URLs with captions and metadata."
                ),
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Medical image search query.",
                    },
                    "modality": {
                        "type": "string",
                        "description": "Filter by imaging modality.",
                        "enum": ["MRI", "CT", "X-ray", "Ultrasound", "PET", "Mammography", "any"],
                        "default": "any",
                    },
                    "body_part": {
                        "type": "string",
                        "description": "Filter by body part.",
                        "enum": [
                            "brain",
                            "head",
                            "chest",
                            "abdomen",
                            "spine",
                            "pelvis",
                            "cardiac",
                            "any",
                        ],
                        "default": "any",
                    },
                },
                execute=_execute_search_images,
                requires_image=False,
                prompt_documentation=SEARCH_IMAGES_PROMPT_DOC,
                category="search",
            )
        )

    return tools
