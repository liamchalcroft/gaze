from __future__ import annotations

from typing import Any

import pytest

from radiant_harness.base import AgenticProcessorBase
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
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
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


async def _echo_tool(registry: ToolRegistry, value: int) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="echo", description=f"echo {value}", metadata={"value": value})


class FakeProcessor(AgenticProcessorBase):
    def __init__(self) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=3,
            adapter_factory=FakeAdapter,
        )

    def get_system_prompt(self, images, metadata) -> str:  # type: ignore[override]
        return "system"

    def get_user_message(self, images, metadata) -> str:  # type: ignore[override]
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

    def _create_tool_registry(self, images, active_image_index: int = 0) -> ToolRegistry | None:  # type: ignore[override]
        tool = Tool(
            name="echo",
            description="echo back",
            parameters={"value": {"type": "integer", "description": "value to echo"}},
            execute=_echo_tool,
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

