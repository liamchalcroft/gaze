"""Tests for gaze.verifiers.adapter — GazeAdapter.

Covers _extract_user_prompt, _convert_response_to_messages,
_collect_tool_calls, _collect_tool_results, create_environment_class,
AdapterEnv.env_response, AdapterEnv.is_completed, and Path image_path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

try:
    from gaze.verifiers.adapter import GazeAdapter
    from gaze.verifiers.base import BaseMultiTurnEnv

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

from gaze.base import AgenticProcessorBase
from gaze.base import ImageInput
from gaze.types import AgenticResult
from gaze.types import ToolCall
from gaze.types import ToolResult
from gaze.types import Turn

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
        self.adapter = GazeAdapter(processor=_make_processor())

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
        self.adapter = GazeAdapter(processor=_make_processor())

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
        self.adapter = GazeAdapter(processor=_make_processor())

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
        adapter = GazeAdapter(processor=_make_processor())
        env_cls = adapter.create_environment_class()
        assert issubclass(env_cls, BaseMultiTurnEnv)

    def test_created_env_has_adapter(self) -> None:
        adapter = GazeAdapter(processor=_make_processor())
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[])
        assert hasattr(env, "_adapter")
        assert isinstance(env._adapter, GazeAdapter)


# ---------------------------------------------------------------------------
# AdapterEnv.env_response  (adapter.py lines 205-214)
# ---------------------------------------------------------------------------


class TestAdapterEnvResponse:
    @pytest.mark.asyncio
    async def test_env_response_updates_state(self) -> None:
        """env_response increments turn, counts tool_calls, sets is_complete."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[])

        # Monkey-patch process_verifiers_messages to avoid real model call
        async def fake_process(
            messages: Any,
            info: Any,  # noqa: ARG001
        ) -> dict[str, Any]:
            return {
                "response": {"answer": "tumor"},
                "messages": [{"role": "assistant", "content": '{"answer": "tumor"}'}],
                "tool_calls": [{"id": "tc_1", "name": "zoom", "arguments": "{}"}],
                "turns": 1,
                "is_complete": True,
            }

        env._adapter.process_verifiers_messages = fake_process  # type: ignore[assignment]

        messages = [{"role": "user", "content": "Analyze scan"}]
        state: dict[str, Any] = {"turn": 0, "tool_uses": 0}

        new_messages = await env.env_response(messages, state)

        # State is mutated in-place
        assert state["turn"] == 1
        assert state["tool_uses"] == 1
        assert state["is_complete"] is True
        assert len(new_messages) == 1
        assert new_messages[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_env_response_with_no_tool_calls(self) -> None:
        """When no tools are used, tool_uses stays at zero."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[])

        async def fake_process(
            messages: Any,
            info: Any,  # noqa: ARG001
        ) -> dict[str, Any]:
            return {
                "response": {"answer": "normal"},
                "messages": [{"role": "assistant", "content": '{"answer": "normal"}'}],
                "tool_calls": [],
                "turns": 1,
                "is_complete": False,
            }

        env._adapter.process_verifiers_messages = fake_process  # type: ignore[assignment]

        messages = [{"role": "user", "content": "Test"}]
        state: dict[str, Any] = {"turn": 2, "tool_uses": 5}

        await env.env_response(messages, state)

        # State is mutated in-place
        assert state["turn"] == 3
        assert state["tool_uses"] == 5
        assert state["is_complete"] is False


# ---------------------------------------------------------------------------
# AdapterEnv.is_completed  (adapter.py lines 223-225)
# ---------------------------------------------------------------------------


def _test_state(**overrides: Any) -> dict[str, Any]:
    """Create a minimal verifiers-compatible State dict for testing."""
    base: dict[str, Any] = {
        "timing": {"start_time": time.time()},
        "trajectory": [{"prompt": [], "completion": []}],
        "prompt": [],
        "completion": [],
    }
    base.update(overrides)
    return base


class TestAdapterEnvIsCompleted:
    @pytest.mark.asyncio
    async def test_completed_via_max_turns(self) -> None:
        """@vf.stop _turn_limit_reached fires when turn >= max_turns."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[], max_turns=3)

        state = _test_state(turn=3, is_complete=False)
        assert await env.is_completed(state) is True

    @pytest.mark.asyncio
    async def test_completed_via_state_flag(self) -> None:
        """@vf.stop _adapter_complete fires when is_complete=True."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[], max_turns=100)

        state = _test_state(turn=1, is_complete=True)
        assert await env.is_completed(state) is True

    @pytest.mark.asyncio
    async def test_not_completed(self) -> None:
        """Neither max_turns nor state flag → not complete."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)
        env_cls = adapter.create_environment_class()
        env = env_cls(cases=[], max_turns=100)

        state = _test_state(turn=1, is_complete=False)
        assert await env.is_completed(state) is False


# ---------------------------------------------------------------------------
# process_verifiers_messages with Path image_path  (adapter.py line 69)
# ---------------------------------------------------------------------------


class TestProcessWithPathImagePath:
    @pytest.mark.asyncio
    async def test_path_object_image_path_passed_through(self) -> None:
        """When info['image_path'] is a Path, it is used directly (not str→Path)."""
        processor = _make_processor()
        adapter = GazeAdapter(processor=processor)

        captured: dict[str, Any] = {}

        async def fake_analyze(*, images: Any = None, metadata: Any = None) -> AgenticResult:
            captured["images"] = images
            return AgenticResult(
                final_response={"answer": "ok", "continue": False},
                turns=[],
                total_tokens=10,
                confidence=0.5,
            )

        adapter.processor.analyze = fake_analyze  # type: ignore[assignment]

        messages = [{"role": "user", "content": "test"}]
        info: dict[str, Any] = {"image_path": Path("/test_data/test.png")}

        result = await adapter.process_verifiers_messages(messages, info)

        assert isinstance(captured["images"], Path)
        assert captured["images"] == Path("/test_data/test.png")
        assert result["is_complete"] is True
