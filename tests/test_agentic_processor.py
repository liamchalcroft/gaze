from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gaze.base import AgenticProcessorBase
from gaze.base import ImageInput
from gaze.exceptions import AgenticProcessingError
from gaze.exceptions import SchemaValidationError
from gaze.models import AdapterProtocol
from gaze.models import GenerationLog
from gaze.tools import Tool
from gaze.tools import ToolRegistry
from gaze.types import AgenticResult
from gaze.types import ToolResult


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
        **kwargs,
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
        **kwargs,
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


class InvalidFinalResponseAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            '{"continue": false}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


class InvalidFinalResponseProcessor(AgenticProcessorBase):
    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=False,
            use_web_search=False,
            max_turns=1,
            adapter_factory=InvalidFinalResponseAdapter,
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
                "name": "invalid-final",
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


@pytest.mark.asyncio
async def test_invalid_final_response_raises_schema_validation_error() -> None:
    processor = InvalidFinalResponseProcessor()

    with pytest.raises(SchemaValidationError) as exc:
        await processor.analyze(images=None, metadata={})

    assert exc.value.turns_completed == 1
    assert exc.value.missing_fields == ["result"]
    assert exc.value.response == {"continue": False}


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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
    # (injected after turn 2, which is turn_idx=1, the penultimate of 3).
    # Match the specific "Next turn" phrasing to distinguish from the
    # always-present "FINAL turn" notice injected on the last turn itself.
    third_call_messages = adapter.messages_history[2]
    user_messages = [m for m in third_call_messages if m.get("role") == "user"]
    penultimate_warnings = [
        m for m in user_messages if "next turn" in str(m.get("content", "")).lower()
    ]
    assert len(penultimate_warnings) == 1, (
        f"Expected exactly 1 penultimate warning, found {len(penultimate_warnings)}"
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
        **kwargs: Any,
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
        **kwargs: Any,
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
    assert "boom" in tool_result_turns[0].tool_results[0].error.lower()


@pytest.mark.asyncio
async def test_multi_image_tool_limitation_in_prompt() -> None:
    """When >1 images are provided, system prompt notes tools only affect first.

    Uses max_turns=2 because single-turn mode skips tool registry creation
    (tools are never offered on the single/last turn), making the warning moot.
    """
    from PIL import Image as PILImage

    adapter = MaxTurns1ToolAdapter()

    class MultiImageProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=2,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs,
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
    from gaze.base import _COORD_MODIFYING_TOOLS
    from gaze.tools import create_visual_tools

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
    from gaze.base import _COORD_MODIFYING_TOOLS

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
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            # Return a JSON array — valid JSON but not a dict
            return (
                "[1, 2, 3]",
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
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            "[1, 2, 3]",
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


# ---------------------------------------------------------------------------
# asyncio.TimeoutError in tool execution is caught gracefully
# ---------------------------------------------------------------------------


async def _timeout_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    raise asyncio.TimeoutError("search timed out")


class TimeoutToolAdapter(AdapterProtocol):
    """Adapter that calls a tool which raises asyncio.TimeoutError, then recovers."""

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
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            return (
                "",
                [{"id": "call-1", "name": "timeout_tool", "arguments": "{}"}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "recovered after timeout"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_asyncio_timeout_in_tool_is_caught_gracefully() -> None:
    """asyncio.TimeoutError in a tool returns an error result, not a crash."""
    adapter = TimeoutToolAdapter()

    class TimeoutToolProcessor(AgenticProcessorBase):
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
                name="timeout_tool",
                description="always times out",
                parameters={},
                execute=_timeout_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = TimeoutToolProcessor()
    result = await processor.analyze(images=None, metadata={})

    # Model recovered after the timeout error was returned as a ToolResult
    assert result.final_response["result"] == "recovered after timeout"
    assert adapter.calls == 2

    # The tool_result turn recorded the timeout error
    tool_result_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_result_turns) == 1
    assert tool_result_turns[0].tool_results[0].error is not None
    assert "timed out" in tool_result_turns[0].tool_results[0].error.lower()


@pytest.mark.asyncio
async def test_create_tool_registry_reuses_shared_search_managers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSharedManager:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    fake_web = _FakeSharedManager()
    fake_img = _FakeSharedManager()

    import gaze.retrieval.image_search as image_search_module
    import gaze.retrieval.web_search as web_search_module

    monkeypatch.setattr(web_search_module, "WebSearchManager", lambda: fake_web)
    monkeypatch.setattr(image_search_module, "MedicalImageSearchManager", lambda: fake_img)

    class SearchReuseProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=True,
                max_turns=1,
                adapter_factory=InvalidFinalResponseAdapter,
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

    processor = SearchReuseProcessor()
    registry1 = processor._create_tool_registry([])
    registry2 = processor._create_tool_registry([])
    assert registry1 is not None
    assert registry2 is not None

    assert registry1._web_search_manager is fake_web
    assert registry2._web_search_manager is fake_web
    assert registry1._image_search_manager is fake_img
    assert registry2._image_search_manager is fake_img

    await registry1.aclose()
    await registry2.aclose()
    assert fake_web.close_calls == 0
    assert fake_img.close_calls == 0

    await processor.aclose()
    assert fake_web.close_calls == 1
    assert fake_img.close_calls == 1


# ---------------------------------------------------------------------------
# Malformed tool arguments return error ToolResult, not crash (Finding 1)
# ---------------------------------------------------------------------------


class MalformedArgsAdapter(AdapterProtocol):
    """Adapter: turn 1 returns tool call with invalid JSON args, turn 2 finalizes."""

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
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            return (
                "",
                [{"id": "call-1", "name": "echo", "arguments": "NOT-VALID-JSON!!!"}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "recovered"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_malformed_tool_args_returns_error_not_crash() -> None:
    """Malformed JSON in tool arguments must produce an error ToolResult,
    not crash the analysis with AgenticProcessingError."""
    adapter = MalformedArgsAdapter()

    class MalformedArgsProcessor(AgenticProcessorBase):
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
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = MalformedArgsProcessor()
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "recovered"
    assert adapter.calls == 2

    # The malformed args error was reported as a ToolResult, not swallowed
    tool_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_turns) == 1
    tr = tool_turns[0].tool_results[0]
    assert tr.error is not None
    assert "json" in tr.error.lower() or "malformed" in tr.error.lower()


@pytest.mark.asyncio
async def test_non_dict_tool_args_returns_error_not_crash() -> None:
    """Tool arguments that are valid JSON but not an object (e.g. a list)
    must return an error ToolResult, not crash."""

    class ArrayArgsAdapter(AdapterProtocol):
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
            **kwargs: Any,
        ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
            _ = messages, max_tokens, temperature, tools, response_format
            self.calls += 1
            if self.calls == 1:
                return (
                    "",
                    [{"id": "call-1", "name": "echo", "arguments": "[1, 2, 3]"}],
                    GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
                )
            return (
                '{"continue": false, "result": "ok"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )

    adapter = ArrayArgsAdapter()

    class ArrayArgsProcessor(AgenticProcessorBase):
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
                name="echo",
                description="echo back",
                parameters={"value": {"type": "integer", "description": "v"}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    processor = ArrayArgsProcessor()
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "ok"
    tool_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_turns) == 1
    assert tool_turns[0].tool_results[0].error is not None
    err = tool_turns[0].tool_results[0].error.lower()
    assert "json" in err or "malformed" in err or "type" in err


# ---------------------------------------------------------------------------
# coord_space_modified only set for successful tool calls (Finding 2)
# ---------------------------------------------------------------------------


async def _failing_crop(registry: ToolRegistry, **kwargs: Any) -> ToolResult:  # noqa: ARG001
    raise ValueError("invalid crop region")


@pytest.mark.asyncio
async def test_failed_coord_tool_does_not_set_coord_space_modified() -> None:
    """A failed crop/zoom/rotate should NOT trigger the coordinate warning."""

    class FailingCropAdapter(AdapterProtocol):
        supports_multipart_tool_content: bool = True

        def __init__(self) -> None:
            self.calls = 0
            self.last_messages: list[dict[str, Any]] = []

        async def generate_chat(
            self,
            messages: list[dict[str, Any]] | None = None,
            max_tokens: int | None = None,
            temperature: float | None = None,
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
            _ = max_tokens, temperature, tools, response_format
            self.last_messages = list(messages or [])
            self.calls += 1
            if self.calls == 1:
                return (
                    "",
                    [{"id": "c1", "name": "crop", "arguments": '{"x1":0,"y1":0,"x2":-1,"y2":-1}'}],
                    GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
                )
            return (
                '{"continue": false, "result": "done"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )

    adapter = FailingCropAdapter()

    class CropTestProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=True,
                use_web_search=False,
                max_turns=2,  # Force turn 2 to be the last turn
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
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "continue": {"type": "boolean"},
                            "result": {"type": "string"},
                        },
                        "required": ["continue", "result"],
                    },
                },
            }

        def validate_response(self, response: dict[str, Any]) -> bool:
            return "result" in response

        def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
            _ = images
            crop_tool = Tool(
                name="crop",
                description="crop image",
                parameters={
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                },
                execute=_failing_crop,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[crop_tool])

    processor = CropTestProcessor()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "done"

    # The final-turn message should NOT contain the coordinate warning
    # because the crop failed.
    final_user_msgs = [
        m
        for m in adapter.last_messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any("FINAL turn" in str(p.get("text", "")) for p in m["content"])
    ]
    assert len(final_user_msgs) == 1
    final_text = str(final_user_msgs[0]["content"])
    assert "WARNING" not in final_text
    assert "changed the coordinate space" not in final_text


# ---------------------------------------------------------------------------
# Coordinate warning emitted even when response_schema is None (Finding 3)
# ---------------------------------------------------------------------------


async def _noop_crop(registry: ToolRegistry, **kwargs: Any) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="crop", description="Cropped image to region")


@pytest.mark.asyncio
async def test_coord_warning_emitted_without_response_schema() -> None:
    """The coordinate-space warning must appear on the final turn even when
    get_response_schema() returns None (free-form responses)."""

    class CoordWarningAdapter(AdapterProtocol):
        supports_multipart_tool_content: bool = True

        def __init__(self) -> None:
            self.calls = 0
            self.last_messages: list[dict[str, Any]] = []

        async def generate_chat(
            self,
            messages: list[dict[str, Any]] | None = None,
            max_tokens: int | None = None,
            temperature: float | None = None,
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
            _ = max_tokens, temperature, tools, response_format
            self.last_messages = list(messages or [])
            self.calls += 1
            if self.calls == 1:
                return (
                    "",
                    [{"id": "c1", "name": "crop", "arguments": '{"x1":0,"y1":0,"x2":10,"y2":10}'}],
                    GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
                )
            # Second turn: continue to use up a turn so we get a final-turn injection
            if self.calls == 2:
                return (
                    '{"continue": true, "result": "partial"}',
                    None,
                    GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
                )
            return (
                '{"continue": false, "result": "done"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )

    adapter = CoordWarningAdapter()

    class NoSchemaCoordProcessor(AgenticProcessorBase):
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
            crop_tool = Tool(
                name="crop",
                description="crop image",
                parameters={
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                },
                execute=_noop_crop,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[crop_tool])

    processor = NoSchemaCoordProcessor()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "done"

    # Even with response_schema=None, the final turn message must exist
    final_user_msgs = [
        m
        for m in adapter.last_messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any("FINAL turn" in str(p.get("text", "")) for p in m["content"])
    ]
    assert len(final_user_msgs) == 1
    final_text = str(final_user_msgs[0]["content"])
    assert "FINAL turn" in final_text
    assert "Do NOT attempt tool calls" in final_text


# ── Regression tests for agentic loop audit fixes ────────────────────────


class EmptyResponseAdapter(AdapterProtocol):
    """Returns empty text with no tool calls, then valid JSON."""

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
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = max_tokens, temperature, tools, response_format
        self.messages_history.append(list(messages or []))
        self.calls += 1
        if self.calls == 1:
            # Empty response — triggers nudge
            return (
                "",
                None,
                GenerationLog(
                    prompt_tokens=1,
                    completion_tokens=0,
                    finish_reason="stop",
                ),
            )
        # Valid response after nudge
        return (
            '{"continue": false, "result": "recovered"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_empty_response_assistant_message_has_content_field() -> None:
    """Fix #1: Assistant messages always include 'content' even when text is empty."""
    adapter = EmptyResponseAdapter()

    class SimpleProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "result" in response

    processor = SimpleProcessor()
    result = await processor.analyze()
    assert result.final_response["result"] == "recovered"

    # Check that the assistant message from the empty turn has 'content'
    call2_messages = adapter.messages_history[1]
    assistant_msgs = [m for m in call2_messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    # Every assistant message must have 'content' key per OpenAI spec
    for msg in assistant_msgs:
        assert "content" in msg, f"Assistant message missing 'content': {msg}"


class RepeatedGarbageAdapter(AdapterProtocol):
    """Returns non-JSON on every turn to test force-finalize escalation."""

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
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        return (
            "I cannot produce JSON",
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=5, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_force_finalize_escalation_raises() -> None:
    """Fix #3: After _MAX_RECOVERY_NUDGES consecutive failures, raises instead
    of burning more turns with the same force-finalize message."""
    adapter = RepeatedGarbageAdapter()

    class GarbageProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=False,
                max_turns=10,  # Plenty of turns
                adapter_factory=lambda: adapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "result" in response

    processor = GarbageProcessor()
    with pytest.raises(AgenticProcessingError, match="consecutive recovery attempts"):
        await processor.analyze()

    # Should give up well before using all 10 turns
    assert adapter.calls < 10


@pytest.mark.asyncio
async def test_single_turn_skips_tool_registry() -> None:
    """Fix #4: Single-turn mode does not create tool registry."""
    adapter = MaxTurns1ToolAdapter()
    registry_created = False

    class TrackingProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=True,
                max_turns=1,
                adapter_factory=lambda: adapter,
            )

        def _create_tool_registry(self, images):
            nonlocal registry_created
            registry_created = True
            return super()._create_tool_registry(images)

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "result" in response

    processor = TrackingProcessor()
    result = await processor.analyze()
    assert result.final_response["result"] == "immediate"
    assert not registry_created, "Single-turn mode should not create tool registry"


class CoerceAdapter(AdapterProtocol):
    """Returns response with string-typed numbers to test central coercion."""

    supports_multipart_tool_content: bool = True

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        # Return score as string — coerce_json_types should fix this
        return (
            '{"continue": false, "result": "ok", "score": "42"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
async def test_central_coerce_json_types_before_validation() -> None:
    """Fix #5: base.py calls coerce_json_types centrally before validation."""

    class CoerceProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=False,
                max_turns=1,
                adapter_factory=CoerceAdapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "continue": {"type": "boolean"},
                            "result": {"type": "string"},
                            "score": {"type": "integer"},
                        },
                        "required": ["continue", "result", "score"],
                    },
                },
            }

        def validate_response(self, response):
            return (
                "result" in response and "score" in response and isinstance(response["score"], int)
            )

    processor = CoerceProcessor()
    result = await processor.analyze()
    # Score should have been coerced from "42" to 42 centrally
    assert result.final_response["score"] == 42
    assert isinstance(result.final_response["score"], int)


@pytest.mark.asyncio
async def test_confidence_penalizes_nudge_turns() -> None:
    """Fix #6: Confidence decreases when non-tool assistant turns exceed 1."""
    from gaze.types import Turn

    class PenaltyProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(model_name="test", use_tools=False, max_turns=1)

        def get_system_prompt(self, images, metadata):
            return ""

        def get_user_message(self, images, metadata):
            return ""

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return True

    processor = PenaltyProcessor()

    # 1 non-tool assistant turn (normal) → no penalty
    turns_1 = [Turn(role="assistant", content="answer")]
    conf_1 = processor.calculate_confidence({}, turns_1)

    # 3 non-tool assistant turns (2 nudges) → penalty
    turns_3 = [
        Turn(role="assistant", content="garbage"),
        Turn(role="assistant", content="garbage"),
        Turn(role="assistant", content="answer"),
    ]
    conf_3 = processor.calculate_confidence({}, turns_3)

    assert conf_1 == 0.5  # base, no tool bonus, no penalty
    assert conf_3 < conf_1  # penalty applied
    assert conf_3 == pytest.approx(0.5 - 0.05 * 2)  # 2 extra non-tool turns


# =====================================================================
# n_ctx detection in truncation error
# =====================================================================


class TruncatedAdapter(AdapterProtocol):
    """Adapter that simulates a context-limited server (n_ctx < max_tokens)."""

    supports_multipart_tool_content: bool = True

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    async def generate_chat(
        self,
        messages=None,
        max_tokens=None,
        temperature=None,
        tools=None,
        response_format=None,
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        return (
            "",
            None,
            GenerationLog(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                finish_reason="length",
            ),
        )

    async def aclose(self) -> None:
        pass


class TruncatedProcessor(AgenticProcessorBase):
    """Single-turn processor that always gets a truncated empty response."""

    def __init__(self, prompt_tok: int, comp_tok: int) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=False,
            use_web_search=False,
            max_turns=1,
            adapter_factory=lambda: TruncatedAdapter(prompt_tok, comp_tok),
            max_tokens=8192,
        )

    def get_system_prompt(self, images, metadata):
        return "test"

    def get_user_message(self, images, metadata):
        return "test"

    def get_response_schema(self):
        return None

    def validate_response(self, response):
        return True


@pytest.mark.asyncio
async def test_truncation_error_detects_nctx_limit() -> None:
    """When completion_tokens << max_tokens, error message should mention n_ctx."""
    processor = TruncatedProcessor(prompt_tok=2685, comp_tok=1411)
    with pytest.raises(AgenticProcessingError, match="Server context window"):
        await processor.analyze(images=None, metadata={})
    await processor.aclose()


@pytest.mark.asyncio
async def test_truncation_error_max_tokens_hint() -> None:
    """When completion_tokens ~ max_tokens, error should suggest increasing max_tokens."""
    processor = TruncatedProcessor(prompt_tok=100, comp_tok=8000)
    with pytest.raises(AgenticProcessingError, match="Increase max_tokens"):
        await processor.analyze(images=None, metadata={})
    await processor.aclose()


# =====================================================================
# continue field coercion: None, int, bool-like values
# =====================================================================


class NullContinueAdapter(AdapterProtocol):
    """Adapter that returns 'continue' as null (None)."""

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
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        return (
            '{"continue": null, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )

    async def aclose(self) -> None:
        pass


class IntContinueAdapter(AdapterProtocol):
    """Adapter returning 'continue' as int 1 then 0."""

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
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        if self.calls == 1:
            return (
                '{"continue": 1, "result": "thinking"}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )
        return (
            '{"continue": 0, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_null_continue_coerced_to_false() -> None:
    """null/None in 'continue' field is coerced to False (model is done)."""
    adapter = NullContinueAdapter()

    class NullContinueProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "result" in response

    processor = NullContinueProcessor()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "done"
    assert adapter.calls == 1  # Accepted on first turn (null → false → done)


@pytest.mark.asyncio
async def test_int_continue_coerced_to_bool() -> None:
    """Integer 0/1 in 'continue' field is coerced to bool."""
    adapter = IntContinueAdapter()

    class IntContinueProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=3,
                adapter_factory=lambda: adapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "result" in response

    processor = IntContinueProcessor()
    result = await processor.analyze(images=None, metadata={})
    # Turn 1: continue=1 → True (keep going), turn 2: continue=0 → False (done)
    assert result.final_response["result"] == "done"
    assert adapter.calls == 2


# =====================================================================
# AgenticResult confidence bounds validation
# =====================================================================

from gaze.types import Turn  # noqa: E402


class TestAgenticResultConfidenceBounds:
    """AgenticResult rejects confidence outside [0.0, 1.0]."""

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            AgenticResult(
                final_response={"continue": False},
                turns=(Turn(role="assistant", content="ok"),),
                total_tokens=10,
                confidence=1.5,
            )

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            AgenticResult(
                final_response={"continue": False},
                turns=(Turn(role="assistant", content="ok"),),
                total_tokens=10,
                confidence=-0.1,
            )

    def test_confidence_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            AgenticResult(
                final_response={"continue": False},
                turns=(Turn(role="assistant", content="ok"),),
                total_tokens=10,
                confidence=float("nan"),
            )

    def test_confidence_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            AgenticResult(
                final_response={"continue": False},
                turns=(Turn(role="assistant", content="ok"),),
                total_tokens=10,
                confidence=float("inf"),
            )

    def test_confidence_boundary_zero_ok(self) -> None:
        result = AgenticResult(
            final_response={"continue": False},
            turns=(Turn(role="assistant", content="ok"),),
            total_tokens=10,
            confidence=0.0,
        )
        assert result.confidence == 0.0

    def test_confidence_boundary_one_ok(self) -> None:
        result = AgenticResult(
            final_response={"continue": False},
            turns=(Turn(role="assistant", content="ok"),),
            total_tokens=10,
            confidence=1.0,
        )
        assert result.confidence == 1.0


# =====================================================================
# Post-loop inner-schema wrapping re-runs coerce_json_types
# =====================================================================


class InnerSchemaAdapter(AdapterProtocol):
    """Adapter that returns a sub-object (inner schema keys) instead of full schema.

    Simulates a local model that outputs the inner caption object fields
    at the top level, with string types that need coercion.
    """

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
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format
        self.calls += 1
        # Return inner-object keys at top level, with string score (needs coercion)
        return (
            '{"text": "A brain MRI", "score": "0.85", "continue": false}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_post_loop_wrapping_recoerces_types() -> None:
    """After inner-schema wrapping, coerce_json_types re-runs on wrapped dict."""

    class InnerSchemaProcessor(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test-model",
                use_tools=False,
                use_web_search=False,
                max_turns=1,
                adapter_factory=InnerSchemaAdapter,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "caption": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                            },
                            "tags": {"type": "array"},
                            "continue": {"type": "boolean"},
                        },
                        "required": ["caption", "tags", "continue"],
                    },
                },
            }

        def validate_response(self, response):
            if "caption" not in response:
                return False
            cap = response["caption"]
            if not isinstance(cap, dict):
                return False
            return "text" in cap and isinstance(cap.get("score"), int | float)

    processor = InnerSchemaProcessor()
    result = await processor.analyze(images=None, metadata={})
    # The model returned {text, score, continue} — inner keys of "caption".
    # _try_wrap_inner_schema wraps under "caption", then coerce_json_types
    # converts score from string "0.85" to float 0.85.
    assert "caption" in result.final_response
    cap = result.final_response["caption"]
    assert cap["text"] == "A brain MRI"
    assert isinstance(cap["score"], float)
    assert cap["score"] == pytest.approx(0.85)
