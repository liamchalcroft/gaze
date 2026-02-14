from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import GenerationLog
from radiant_harness.retrieval.web_search import SearchResult
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.types import ToolResult


class RecordingAdapter(AdapterProtocol):
    def __init__(self, tool_calls: list[dict[str, Any]]) -> None:
        self.tool_calls = tool_calls
        self.messages_history: list[list[dict[str, Any]]] = []
        self.calls = 0

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format
        self.messages_history.append(messages or [])
        self.calls += 1
        if self.calls == 1:
            return (
                "",
                self.tool_calls,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


async def _formatted_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="format",
        description="formatted tool",
        metadata={"formatted_results": "FORMATTED_RESULT"},
    )


async def _image_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="image",
        description="image tool",
        image_base64="AAA",
        image_mime_type="image/png",
    )


class InlineToolProcessor(AgenticProcessorBase):
    def __init__(self, adapter: AdapterProtocol, tool: Tool) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=False,
            use_web_search=False,
            max_turns=2,
            adapter_factory=lambda: adapter,
        )
        self._tool = tool

    def get_system_prompt(self, images=None, metadata=None) -> str:  # type: ignore[override]
        _ = images, metadata
        return "system"

    def get_user_message(self, images=None, metadata=None) -> str:  # type: ignore[override]
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "fake",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "continue": {"type": "boolean"},
                        "result": {"type": "string"},
                    },
                    "required": ["continue", "result"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_response(self, response: dict[str, Any]) -> bool:
        return {"continue", "result"} <= set(response)

    def _create_tool_registry(self, images=None) -> ToolRegistry | None:  # type: ignore[override]
        _ = images
        return ToolRegistry(image_path=None, tools=[self._tool])


class SearchToolProcessor(AgenticProcessorBase):
    def __init__(self, adapter: AdapterProtocol) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=False,
            use_web_search=True,
            max_turns=2,
            adapter_factory=lambda: adapter,
        )

    def get_system_prompt(self, images=None, metadata=None) -> str:  # type: ignore[override]
        _ = images, metadata
        return "system"

    def get_user_message(self, images=None, metadata=None) -> str:  # type: ignore[override]
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "fake",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "continue": {"type": "boolean"},
                        "result": {"type": "string"},
                    },
                    "required": ["continue", "result"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_response(self, response: dict[str, Any]) -> bool:
        return {"continue", "result"} <= set(response)


@pytest.mark.asyncio
async def test_tool_message_includes_formatted_results() -> None:
    adapter = RecordingAdapter(tool_calls=[{"id": "call-1", "name": "format", "arguments": {}}])
    tool = Tool(
        name="format",
        description="formats",
        parameters={},
        execute=_formatted_tool,
        requires_image=False,
    )
    processor = InlineToolProcessor(adapter=adapter, tool=tool)

    await processor.analyze(images=None, metadata={"history": "hx"})

    tool_message = next(msg for msg in adapter.messages_history[1] if msg.get("role") == "tool")
    assert isinstance(tool_message["content"], str)
    assert "formatted tool" in tool_message["content"]
    assert "FORMATTED_RESULT" in tool_message["content"]


@pytest.mark.asyncio
async def test_tool_message_includes_image_payload() -> None:
    adapter = RecordingAdapter(tool_calls=[{"id": "call-1", "name": "image", "arguments": {}}])
    tool = Tool(
        name="image",
        description="image",
        parameters={},
        execute=_image_tool,
        requires_image=False,
    )
    processor = InlineToolProcessor(adapter=adapter, tool=tool)

    await processor.analyze(images=None, metadata={"history": "hx"})

    tool_message = next(msg for msg in adapter.messages_history[1] if msg.get("role") == "tool")
    assert isinstance(tool_message["content"], list)
    assert tool_message["content"][0]["type"] == "text"
    assert tool_message["content"][1]["type"] == "image_url"
    assert tool_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_search_web_tool_message_includes_formatted_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search(query, *, search_type="general", **kwargs):
        return [
            SearchResult(
                title="Glioblastoma MRI",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                content="c" * 50,
                snippet="A short summary.",
                source="pubmed",
                reliability_score=0.95,
                content_type="article",
                medical_relevance=0.9,
                extracted_entities=["glioblastoma"],
                open_access=True,
            )
        ]

    mock_manager = AsyncMock()
    mock_manager.search = fake_search
    mock_manager.close = AsyncMock()
    monkeypatch.setattr(ToolRegistry, "get_web_search_manager", lambda _self: mock_manager)

    adapter = RecordingAdapter(
        tool_calls=[
            {
                "id": "call-1",
                "name": "search_web",
                "arguments": {"query": "glioblastoma", "search_type": "diagnosis"},
            }
        ]
    )
    processor = SearchToolProcessor(adapter=adapter)

    await processor.analyze(images=None, metadata={"history": "hx"})

    tool_message = next(msg for msg in adapter.messages_history[1] if msg.get("role") == "tool")
    assert isinstance(tool_message["content"], str)
    assert "## PubMed Search Results" in tool_message["content"]
