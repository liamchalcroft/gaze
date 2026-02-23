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
    supports_multipart_tool_content: bool = True

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
    supports_multipart_tool_content: bool = True

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
async def test_agentic_processor_surfaces_tool_errors_gracefully() -> None:
    """Tool failures are returned to the model, not fatal.

    The FailingAdapter always returns tool calls, so the graceful error
    flows back on turns 1 and 2, and the last-turn guard fires on turn 3.
    """
    processor = FailingToolProcessor()

    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={"history": "hx"})

    # The final error is the last-turn guard (not a tool crash)
    assert "final turn" in str(exc.value).lower()
    assert exc.value.partial_response is not None
    assert exc.value.partial_response["error"] == "tools_unavailable"


# --- Adapters and processors for new edge-case tests ---


class AlwaysToolAdapter(AdapterProtocol):
    """Adapter that always returns tool calls, even when tools=None."""

    supports_multipart_tool_content: bool = True

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

    supports_multipart_tool_content: bool = True

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
    """Last-turn guard fires when model returns tool calls on final turn."""
    processor = LoopExhaustionProcessor()

    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={})

    # With the last-turn guard, the error now fires on the final turn
    # rather than via the for/else exhaustion path.
    assert "final turn" in str(exc.value).lower() or "exhausted" in str(exc.value).lower()
    assert exc.value.turns_completed > 0
    assert exc.value.partial_response is not None
    assert exc.value.partial_response["error"] == "tools_unavailable"


@pytest.mark.asyncio
async def test_markdown_wrapped_json_parsed_correctly() -> None:
    """JSON wrapped in ```json ... ``` is parsed via fallback."""
    processor = MarkdownJsonProcessor()
    result: AgenticResult = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "wrapped"
    assert result.final_response["continue"] is False


# --- Patch Set #1: last-turn guard, penultimate warning, max_turns=1 ---


class LastTurnToolAdapter(AdapterProtocol):
    """Adapter that returns tool calls on every turn, ignoring tools=None."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0
        self.tools_per_call: list[list[dict[str, Any]] | None] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, response_format
        self.calls += 1
        self.tools_per_call.append(tools)
        # Always return tool calls regardless of whether tools were offered
        return (
            "",
            [{"id": f"call-{self.calls}", "name": "echo", "arguments": {"value": 1}}],
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
        )


@pytest.mark.asyncio
async def test_last_turn_tool_calls_rejected_without_execution() -> None:
    """Tool calls on the last turn are rejected immediately, not executed."""
    adapter = LastTurnToolAdapter()

    class SingleTurnToolProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=1,
                adapter_factory=lambda: adapter,
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
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = SingleTurnToolProcessor()

    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={})

    assert "final turn" in str(exc.value).lower()
    assert exc.value.partial_response is not None
    assert exc.value.partial_response["error"] == "tools_unavailable"
    # Adapter was called exactly once (the single turn) — tools were NOT executed
    assert adapter.calls == 1


class PenultimateWarningAdapter(AdapterProtocol):
    """Adapter that continues for 2 turns then finalizes on turn 3."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_history: list[list[dict[str, Any]]] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format
        self.messages_history.append(list(messages or []))
        self.calls += 1
        if self.calls <= 2:
            return (
                '{"continue": true, "result": "thinking"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_penultimate_turn_warning_injected() -> None:
    """On the penultimate turn, a system warning is injected into messages."""
    adapter = PenultimateWarningAdapter()

    class ThreeTurnProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
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

    processor = ThreeTurnProcessor()
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "done"
    assert adapter.calls == 3

    # The 3rd call's messages should contain the penultimate warning
    # (injected after turn 2, which is turn_idx=1, the penultimate of 3)
    third_call_messages = adapter.messages_history[2]
    user_messages = [m for m in third_call_messages if m.get("role") == "user"]
    warning_messages = [m for m in user_messages if "final turn" in str(m.get("content", "")).lower()]
    assert len(warning_messages) == 1, (
        f"Expected exactly 1 penultimate warning, found {len(warning_messages)}"
    )


class MaxTurns1ToolAdapter(AdapterProtocol):
    """Records whether tools were offered on each call."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.tools_offered: list[list[dict[str, Any]] | None] = []
        self.messages_history: list[list[dict[str, Any]]] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, response_format
        self.messages_history.append(list(messages or []))
        self.tools_offered.append(tools)
        return (
            '{"continue": false, "result": "immediate"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_max_turns_1_with_tools_enabled_passes_no_tools() -> None:
    """With max_turns=1, the single turn is the last turn so tools=None."""
    adapter = MaxTurns1ToolAdapter()

    class OneTurnToolProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=1,
                adapter_factory=lambda: adapter,
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
            tool = Tool(
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = OneTurnToolProcessor()
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "immediate"
    # Model was called once and tools=None was passed (last turn)
    assert len(adapter.tools_offered) == 1
    assert adapter.tools_offered[0] is None


# --- Patch Set #2: graceful tool failure, multi-image prompt, parallel execution ---


class RecoveringAdapter(AdapterProtocol):
    """Adapter that requests a failing tool, then recovers with a final response."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_history: list[list[dict[str, Any]]] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format
        self.messages_history.append(list(messages or []))
        self.calls += 1
        if self.calls == 1:
            # First call: request the failing tool
            return (
                "",
                [{"id": "call-1", "name": "boom", "arguments": {}}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        # Second call: recover with a final response
        return (
            '{"continue": false, "result": "recovered"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_graceful_tool_failure_lets_model_recover() -> None:
    """Tool crash returns error to model; model recovers on next turn."""
    adapter = RecoveringAdapter()

    class RecoveringProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
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
            tool = Tool(
                name="boom",
                description="fails",
                parameters={},
                execute=_boom_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = RecoveringProcessor()
    result = await processor.analyze(images=None, metadata={})

    # Model recovered after tool failure
    assert result.final_response["result"] == "recovered"
    assert adapter.calls == 2

    # The tool error message was surfaced in conversation to the model
    second_call_messages = adapter.messages_history[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    tool_content = tool_messages[0]["content"]
    assert "boom" in tool_content.lower() or "error" in tool_content.lower()

    # The tool_result turn recorded the error
    tool_result_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_result_turns) == 1
    assert tool_result_turns[0].tool_results[0].error is not None


@pytest.mark.asyncio
async def test_multi_image_tool_limitation_in_prompt() -> None:
    """When >1 images are provided, system prompt notes tools only affect first."""
    from PIL import Image as PILImage

    adapter = MaxTurns1ToolAdapter()

    class MultiImageProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=1,
                adapter_factory=lambda: adapter,
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

    processor = MultiImageProcessor()

    # Create two PIL images
    img1 = PILImage.new("RGB", (100, 100), color="red")
    img2 = PILImage.new("RGB", (100, 100), color="blue")
    result = await processor.analyze(
        images=[img1, img2],
        image_labels=["T1-weighted", "T2-FLAIR"],
    )

    assert result.final_response["result"] == "immediate"

    # Check that the system prompt in the first call mentions the limitation
    first_call_messages = adapter.messages_history[0]
    system_message = next(m for m in first_call_messages if m.get("role") == "system")
    system_content = system_message["content"]
    assert "first image" in system_content.lower()
    assert "T1-weighted" in system_content


class TimingAdapter(AdapterProtocol):
    """Adapter that records timestamps and requests both image and search tools."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            # Request both an image tool and an independent tool
            return (
                "",
                [
                    {"id": "call-img", "name": "img_tool", "arguments": {}},
                    {"id": "call-ind", "name": "ind_tool", "arguments": {}},
                ],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


async def _slow_image_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    import asyncio

    await asyncio.sleep(0.05)
    return ToolResult(tool_name="img_tool", description="image op done")


_independent_tool_started_at: float = 0.0


async def _independent_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    import time

    global _independent_tool_started_at  # noqa: PLW0603
    _independent_tool_started_at = time.monotonic()
    return ToolResult(tool_name="ind_tool", description="independent done")


@pytest.mark.asyncio
async def test_parallel_execution_of_image_and_independent_tools() -> None:
    """Independent tools start while image tools are still running."""
    import time

    global _independent_tool_started_at  # noqa: PLW0603
    _independent_tool_started_at = 0.0

    adapter = TimingAdapter()

    img_tool = Tool(
        name="img_tool",
        description="image tool",
        parameters={},
        execute=_slow_image_tool,
        requires_image=True,
    )
    ind_tool = Tool(
        name="ind_tool",
        description="independent tool",
        parameters={},
        execute=_independent_tool,
        requires_image=False,
    )

    class ParallelToolProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
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
            return ToolRegistry(image_path=None, tools=[img_tool, ind_tool])

    processor = ParallelToolProcessor()
    start = time.monotonic()
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "done"
    # Both tools completed successfully
    tool_result_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_result_turns) == 1
    assert len(tool_result_turns[0].tool_results) == 2

    # The independent tool should have started before the image tool finished
    # (image tool sleeps 50ms). If sequential, total would be >= 50ms + epsilon.
    # With parallel execution, independent tool starts at roughly the same time.
    assert _independent_tool_started_at > 0
    assert _independent_tool_started_at - start < 0.04, (
        "Independent tool should start concurrently with image tool, not after"
    )


# --- Patch Set #3: string continue coercion, idle-tool escalation, last-turn salvage ---


class StringContinueAdapter(AdapterProtocol):
    """Adapter that returns 'continue' as a string instead of bool."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            # Return string "true" — model wants to continue
            return (
                '{"continue": "true", "result": "thinking"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )
        # Return string "false" — model is done
        return (
            '{"continue": "false", "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_string_continue_coerced_to_bool() -> None:
    """String 'true'/'false' in 'continue' field is coerced to bool."""
    adapter = StringContinueAdapter()

    class StringContinueProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
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

    processor = StringContinueProcessor()
    result = await processor.analyze(images=None, metadata={})

    # Model returned string "false" on turn 2 — coerced and accepted
    assert result.final_response["result"] == "done"
    assert adapter.calls == 2


class IdleToolAdapter(AdapterProtocol):
    """Adapter that returns continue:true JSON without ever requesting tools."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        # Always return continue:true with no tool calls
        return (
            '{"continue": true, "result": "still thinking"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_idle_tool_escalation_force_accepts_after_nudge() -> None:
    """After idle-tool nudge, the next idle turn force-accepts the response."""
    adapter = IdleToolAdapter()

    class IdleToolProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=10,  # plenty of turns — should terminate early
                adapter_factory=lambda: adapter,
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
            tool = Tool(
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = IdleToolProcessor()
    result = await processor.analyze(images=None, metadata={})

    # _IDLE_TOOL_TURNS_LIMIT=3, so:
    # Turn 1,2: continue:true, no tools — under threshold
    # Turn 3: idle threshold hit, nudge injected
    # Turn 4: still no tools, force-accept
    assert adapter.calls == 4
    assert result.final_response["result"] == "still thinking"
    assert result.final_response["continue"] is False


class LastTurnSalvageAdapter(AdapterProtocol):
    """Adapter that returns both tool calls AND valid JSON text on the last turn."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        # Return BOTH valid JSON text AND tool calls
        return (
            '{"continue": false, "result": "salvaged"}',
            [{"id": "call-1", "name": "echo", "arguments": {"value": 1}}],
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_last_turn_salvages_valid_text_alongside_tool_calls() -> None:
    """When model returns tool calls + valid JSON on last turn, salvage the text."""
    adapter = LastTurnSalvageAdapter()

    class SalvageProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=1,
                adapter_factory=lambda: adapter,
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
            tool = Tool(
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = SalvageProcessor()
    # Previously this would raise AgenticProcessingError;
    # now it salvages the valid text response.
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "salvaged"
    assert result.final_response["continue"] is False
    assert adapter.calls == 1


class LastTurnNoSalvageAdapter(AdapterProtocol):
    """Adapter that returns tool calls + invalid text on the last turn."""

    supports_multipart_tool_content: bool = True

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        # Tool calls + empty text — nothing to salvage
        return (
            "",
            [{"id": "call-1", "name": "echo", "arguments": {"value": 1}}],
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
        )


@pytest.mark.asyncio
async def test_last_turn_still_raises_when_text_not_salvageable() -> None:
    """When model returns tool calls + no valid JSON text, still raises."""

    class NoSalvageProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=1,
                adapter_factory=LastTurnNoSalvageAdapter,
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
            tool = Tool(
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = NoSalvageProcessor()
    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={})

    assert "final turn" in str(exc.value).lower()
    assert exc.value.partial_response is not None
    assert exc.value.partial_response["error"] == "tools_unavailable"


# ---------------------------------------------------------------------------
# Unknown tool recovery
# ---------------------------------------------------------------------------


class UnknownToolAdapter(AdapterProtocol):
    """Adapter that calls a non-existent tool on turn 1, then recovers on turn 2."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_history: list[list[dict[str, Any]]] = []

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format
        self.messages_history.append(list(messages or []))
        self.calls += 1

        if self.calls == 1:
            # Call a tool that doesn't exist
            return (
                "",
                [{"id": "call-1", "name": "nonexistent_tool", "arguments": "{}"}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        # Turn 2: recover with a valid final response
        return (
            '{"continue": false, "result": "recovered after unknown tool"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


class UnknownToolProcessor(AgenticProcessorBase):
    def __init__(self, adapter: UnknownToolAdapter) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=3,
            adapter_factory=lambda: adapter,
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
            parameters={"value": {"type": "integer", "description": "value"}},
            execute=_echo_tool,
            requires_image=False,
        )
        return ToolRegistry(image_path=None, tools=[tool])


@pytest.mark.asyncio
async def test_unknown_tool_is_recoverable() -> None:
    """Model calling a non-existent tool should get an error ToolResult, not crash.

    The model can then self-correct on the next turn instead of losing
    all progress in the agentic loop.
    """
    adapter = UnknownToolAdapter()
    processor = UnknownToolProcessor(adapter)
    result: AgenticResult = await processor.analyze(images=None, metadata={})

    # Model recovered and produced a valid final response
    assert result.final_response["result"] == "recovered after unknown tool"

    # Verify the error message was sent back to the model (in tool_result turn)
    tool_result_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_result_turns) == 1
    tr = tool_result_turns[0].tool_results[0]
    assert tr.error is not None
    assert "nonexistent_tool" in tr.error
    assert "echo" in tr.error  # should list available tools


@pytest.mark.asyncio
async def test_unknown_tool_error_lists_available_tools() -> None:
    """The error ToolResult for an unknown tool should list what tools ARE available."""
    adapter = UnknownToolAdapter()
    processor = UnknownToolProcessor(adapter)
    result: AgenticResult = await processor.analyze(images=None, metadata={})

    tool_result_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_result_turns) == 1
    error_msg = tool_result_turns[0].tool_results[0].error
    assert error_msg is not None
    assert "Available tools:" in error_msg
    assert "echo" in error_msg


# ---------------------------------------------------------------------------
# Patch Set: _COORD_MODIFYING_TOOLS matches registered tool names
# ---------------------------------------------------------------------------


def test_coord_modifying_tools_match_registered_names() -> None:
    """Every name in _COORD_MODIFYING_TOOLS must match an actual registered tool."""
    from radiant_harness.base import _COORD_MODIFYING_TOOLS
    from radiant_harness.tools import create_visual_tools

    tools = create_visual_tools(disabled_tools=set())
    registered_names = {t.name for t in tools}

    for coord_tool in _COORD_MODIFYING_TOOLS:
        assert coord_tool in registered_names, (
            f"_COORD_MODIFYING_TOOLS contains '{coord_tool}' but no tool with that name "
            f"is registered. Registered: {sorted(registered_names)}"
        )


def test_flip_tools_in_coord_modifying_set() -> None:
    """flip_horizontal and flip_vertical must be in _COORD_MODIFYING_TOOLS.

    These tools mirror coordinate axes, so bounding box coordinates from
    a flipped image are invalid in the original coordinate space.
    """
    from radiant_harness.base import _COORD_MODIFYING_TOOLS

    assert "flip_horizontal" in _COORD_MODIFYING_TOOLS
    assert "flip_vertical" in _COORD_MODIFYING_TOOLS


# ---------------------------------------------------------------------------
# Patch Set: Non-dict JSON on intermediate turn nudges instead of crashing
# ---------------------------------------------------------------------------


class NonDictJsonAdapter(AdapterProtocol):
    """Adapter that returns a JSON array on turn 1, then a valid object on turn 2."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            # Return a JSON array — valid JSON but not a dict
            return (
                '[1, 2, 3]',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )
        return (
            '{"continue": false, "result": "recovered"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_non_dict_json_intermediate_turn_nudges() -> None:
    """JSON array on intermediate turn should nudge, not crash."""
    adapter = NonDictJsonAdapter()

    class NonDictProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
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

    processor = NonDictProcessor()
    result = await processor.analyze(images=None, metadata={})

    # Model recovered after being nudged
    assert result.final_response["result"] == "recovered"
    assert adapter.calls == 2


class NonDictJsonLastTurnAdapter(AdapterProtocol):
    """Adapter that returns a JSON array on the only (last) turn."""

    supports_multipart_tool_content: bool = True

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
            '[1, 2, 3]',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_non_dict_json_last_turn_still_crashes() -> None:
    """JSON array on the last turn should still raise AgenticProcessingError."""

    class NonDictLastTurnProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=1,
                adapter_factory=NonDictJsonLastTurnAdapter,
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

    processor = NonDictLastTurnProcessor()
    with pytest.raises(AgenticProcessingError) as exc:
        await processor.analyze(images=None, metadata={})

    assert "JSON object" in str(exc.value)
