from __future__ import annotations

from typing import Any

import pytest

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import ImageInput
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import GenerationLog
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.types import AgenticResult
from radiant_harness.types import ToolResult


class FakeAdapter(AdapterProtocol):
    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format  # Unused
        self.calls += 1
        if self.calls == 1:
            return (
                "",
                [{"id": "call-1", "name": "echo", "arguments": {"value": 5}}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


class FailingAdapter(AdapterProtocol):
    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            "",
            [{"id": "call-1", "name": "boom", "arguments": {}}],
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
        )


async def _echo_tool(registry: ToolRegistry, value: int) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="echo", description=f"echo {value}", metadata={"value": value})


async def _boom_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    raise ValueError("boom")


class FakeProcessor(AgenticProcessorBase):
    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=3,
            adapter_factory=FakeAdapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "system"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
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

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        tool = Tool(
            name="echo",
            description="echo back",
            parameters={"value": {"type": "integer", "description": "value to echo"}},
            execute=_echo_tool,
            requires_image=False,
        )
        return ToolRegistry(image_path=None, tools=[tool])


class FailingToolProcessor(AgenticProcessorBase):
    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=3,
            adapter_factory=FailingAdapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "system"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        _ = response
        return True

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        tool = Tool(
            name="boom",
            description="fails",
            parameters={},
            execute=_boom_tool,
            requires_image=False,
        )
        return ToolRegistry(image_path=None, tools=[tool])


@pytest.mark.asyncio
async def test_agentic_processor_runs_tool_and_finalizes() -> None:
    processor = FakeProcessor()
    result: AgenticResult = await processor.analyze(images=None, metadata={"history": "hx"})

    assert result.final_response["result"] == "done"
    assert result.tool_call_count == 1
    assert any(turn.tool_results for turn in result.turns if turn.role == "tool_result")


@pytest.mark.asyncio
async def test_agentic_processor_surfaces_unexpected_tool_errors() -> None:
    processor = FailingToolProcessor()

    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={"history": "hx"})

    assert "unexpected error" in str(exc.value)
    assert exc.value.partial_response is not None
    assert exc.value.partial_response["error"] == "tool_unexpected_error"


# --- Adapters and processors for new edge-case tests ---


class AlwaysToolAdapter(AdapterProtocol):
    """Adapter that always returns tool calls, even when tools=None."""

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            "",
            [{"id": "call-1", "name": "echo", "arguments": {"value": 1}}],
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
        )


class MarkdownJsonAdapter(AdapterProtocol):
    """Adapter that returns JSON wrapped in a markdown code block."""

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            '```json\n{"continue": false, "result": "wrapped"}\n```',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


class LoopExhaustionProcessor(AgenticProcessorBase):
    """Processor whose adapter always returns tool calls, exhausting the loop."""

    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=2,
            adapter_factory=AlwaysToolAdapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "system"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        _ = response
        return True

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        tool = Tool(
            name="echo",
            description="echo back",
            parameters={"value": {"type": "integer", "description": "value to echo"}},
            execute=_echo_tool,
            requires_image=False,
        )
        return ToolRegistry(image_path=None, tools=[tool])


class MarkdownJsonProcessor(AgenticProcessorBase):
    """Processor whose adapter returns JSON in a markdown code block."""

    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=False,
            use_web_search=False,
            max_turns=1,
            adapter_factory=MarkdownJsonAdapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "system"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "result" in response

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        return None


# --- New edge-case tests ---


@pytest.mark.asyncio
async def test_loop_exhaustion_raises_when_all_turns_use_tools() -> None:
    """Loop guard fires when model returns tool calls on every turn."""
    processor = LoopExhaustionProcessor()

    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={})

    assert "exhausted" in str(exc.value).lower()
    assert exc.value.turns_completed > 0


@pytest.mark.asyncio
async def test_markdown_wrapped_json_parsed_correctly() -> None:
    """JSON wrapped in ```json ... ``` is parsed via fallback."""
    processor = MarkdownJsonProcessor()
    result: AgenticResult = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "wrapped"
    assert result.final_response["continue"] is False
