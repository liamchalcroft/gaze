"""Tests for radiant_harness.verifiers.adapter — RadiantHarnessAdapter.

Covers _extract_user_prompt, _convert_response_to_messages,
_collect_tool_calls, _collect_tool_results, and create_environment_class.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

try:
    from radiant_harness.verifiers.adapter import RadiantHarnessAdapter
    from radiant_harness.verifiers.base import BaseMultiTurnEnv

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

from radiant_harness.base import AgenticProcessorBase, ImageInput
from radiant_harness.types import AgenticResult, ToolCall, ToolResult, Turn

pytestmark = pytest.mark.skipif(not _HAS_VERIFIERS, reason="verifiers not installed")


class _StubProcessor(AgenticProcessorBase):
    """Minimal concrete processor for adapter tests."""

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        return "System prompt"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        return metadata.get("user_prompt", "Hello")

    def get_response_schema(self) -> dict[str, Any] | None:
        return {"type": "object", "properties": {"answer": {"type": "string"}}}

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "answer" in response


def _make_processor() -> _StubProcessor:
    return _StubProcessor(use_tools=False)


# ---------------------------------------------------------------------------
# _extract_user_prompt
# ---------------------------------------------------------------------------


class TestExtractUserPrompt:
    def setup_method(self) -> None:
        self.adapter = RadiantHarnessAdapter(processor=_make_processor())

    def test_simple_string_content(self) -> None:
        messages = [{"role": "user", "content": "What is this?"}]
        assert self.adapter._extract_user_prompt(messages) == "What is this?"

    def test_multimodal_content(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        assert self.adapter._extract_user_prompt(messages) == "Describe the image."

    def test_multimodal_multiple_text_parts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1."},
                    {"type": "text", "text": "Part 2."},
                ],
            }
        ]
        assert self.adapter._extract_user_prompt(messages) == "Part 1.\nPart 2."

    def test_returns_last_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert self.adapter._extract_user_prompt(messages) == "second"

    def test_returns_empty_for_no_user(self) -> None:
        messages = [{"role": "assistant", "content": "only assistant"}]
        assert self.adapter._extract_user_prompt(messages) == ""

    def test_returns_empty_for_empty_messages(self) -> None:
        assert self.adapter._extract_user_prompt([]) == ""


# ---------------------------------------------------------------------------
# _convert_response_to_messages
# ---------------------------------------------------------------------------


class TestConvertResponseToMessages:
    def setup_method(self) -> None:
        self.adapter = RadiantHarnessAdapter(processor=_make_processor())

    def test_response_only(self) -> None:
        messages = self.adapter._convert_response_to_messages('{"answer": "yes"}', [], [])
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == '{"answer": "yes"}'

    def test_with_tool_results(self) -> None:
        tool_calls = [{"id": "call_1", "name": "zoom", "arguments": "{}"}]
        tool_results = [
            {"tool_name": "zoom", "description": "Zoomed", "error": None, "metadata": {}}
        ]
        messages = self.adapter._convert_response_to_messages(
            '{"answer": "yes"}', tool_calls, tool_results
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_1"
        parsed = json.loads(messages[1]["content"])
        assert parsed["tool_name"] == "zoom"

    def test_tool_results_exceed_tool_calls_uses_index(self) -> None:
        """When tool_results > tool_calls, fallback to string index for tool_call_id."""
        tool_calls = [{"id": "call_1", "name": "zoom", "arguments": "{}"}]
        tool_results = [
            {"tool_name": "zoom", "description": "r1", "error": None, "metadata": {}},
            {"tool_name": "crop", "description": "r2", "error": None, "metadata": {}},
        ]
        messages = self.adapter._convert_response_to_messages("resp", tool_calls, tool_results)
        assert messages[1]["tool_call_id"] == "call_1"
        assert messages[2]["tool_call_id"] == "1"  # fallback index

    def test_empty_response_text_no_assistant_message(self) -> None:
        messages = self.adapter._convert_response_to_messages("", [], [])
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# _collect_tool_calls / _collect_tool_results
# ---------------------------------------------------------------------------


class TestCollectToolCallsAndResults:
    def setup_method(self) -> None:
        self.adapter = RadiantHarnessAdapter(processor=_make_processor())

    def _make_result(self) -> AgenticResult:
        tc = ToolCall(id="tc_1", name="zoom", arguments={"level": 2})
        tr = ToolResult(tool_name="zoom", description="Zoomed in", metadata={"key": "val"})
        turn = Turn(
            role="assistant",
            content="Let me zoom in.",
            tool_calls=[tc],
            tool_results=[tr],
        )
        return AgenticResult(
            final_response={"answer": "tumor", "continue": False},
            turns=[turn],
            total_tokens=100,
            confidence=0.9,
        )

    def test_collect_tool_calls_extracts_all(self) -> None:
        result = self._make_result()
        calls = self.adapter._collect_tool_calls(result)
        assert len(calls) == 1
        assert calls[0]["id"] == "tc_1"
        assert calls[0]["name"] == "zoom"
        assert calls[0]["arguments"] == {"level": 2}

    def test_collect_tool_results_extracts_all(self) -> None:
        result = self._make_result()
        results = self.adapter._collect_tool_results(result)
        assert len(results) == 1
        assert results[0]["tool_name"] == "zoom"
        assert results[0]["description"] == "Zoomed in"
        assert results[0]["error"] is None
        assert results[0]["metadata"] == {"key": "val"}

    def test_collect_across_multiple_turns(self) -> None:
        tc1 = ToolCall(id="tc_1", name="zoom", arguments={"level": 1})
        tc2 = ToolCall(id="tc_2", name="crop", arguments='{"x": 10}')
        turn1 = Turn(role="assistant", content="t1", tool_calls=[tc1], tool_results=[])
        turn2 = Turn(role="assistant", content="t2", tool_calls=[tc2], tool_results=[])
        result = AgenticResult(
            final_response={"answer": "x"},
            turns=[turn1, turn2],
            total_tokens=200,
            confidence=0.8,
        )
        calls = self.adapter._collect_tool_calls(result)
        assert len(calls) == 2
        assert calls[0]["name"] == "zoom"
        assert calls[1]["name"] == "crop"
        # String arguments stay as strings
        assert calls[1]["arguments"] == '{"x": 10}'


# ---------------------------------------------------------------------------
# create_environment_class
# ---------------------------------------------------------------------------


class TestCreateEnvironmentClass:
    def test_creates_subclass_of_base(self) -> None:
        adapter = RadiantHarnessAdapter(processor=_make_processor())
        env_cls = adapter.create_environment_class()
        assert issubclass(env_cls, BaseMultiTurnEnv)

    def test_created_env_has_adapter(self) -> None:
        adapter = RadiantHarnessAdapter(processor=_make_processor())
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[])
        assert hasattr(env, "_adapter")
        assert isinstance(env._adapter, RadiantHarnessAdapter)
