"""Tests for the Phase-7 public surface.

Covers the high-level ``analyze()`` convenience function, ``SimpleProcessor``,
settable ``temperature``, the direct ``adapter=`` argument, the
``should_continue`` hook, and ``AgenticConfig`` wiring (constructor + context).
"""

from __future__ import annotations

from typing import Any

import pytest

from gaze import AgenticConfig
from gaze import GazeConfig
from gaze import SimpleProcessor
from gaze import analyze
from gaze import config_context
from gaze.models import AdapterProtocol
from gaze.models import GenerationLog


class ScriptedAdapter(AdapterProtocol):
    """Returns scripted ``(content, tool_calls, finish_reason)`` turns in order.

    Records the ``temperature`` and ``max_tokens`` seen on each call so tests
    can assert they were threaded through from the processor.
    """

    supports_multipart_tool_content: bool = True

    def __init__(self, turns: list[tuple[str, list[dict[str, Any]] | None, str]]) -> None:
        self._turns = turns
        self.calls = 0
        self.temperatures: list[float] = []
        self.max_tokens_seen: list[int] = []

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, tools, response_format, stream, seed
        self.temperatures.append(temperature)
        self.max_tokens_seen.append(max_tokens)
        content, tool_calls, finish = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return content, tool_calls, GenerationLog(10, 10, finish)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_analyze_runs_without_a_subclass() -> None:
    adapter = ScriptedAdapter([('{"continue": false, "answer": "ok"}', None, "stop")])
    result = await analyze(
        images=None,
        system="sys",
        user="usr",
        adapter=adapter,
        max_turns=1,
        use_tools=False,
    )
    assert result.final_response["answer"] == "ok"
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_analyze_applies_validator() -> None:
    adapter = ScriptedAdapter([('{"continue": false, "answer": "ok"}', None, "stop")])
    seen: list[dict[str, Any]] = []

    def _validate(resp: dict[str, Any]) -> bool:
        seen.append(resp)
        return "answer" in resp

    result = await analyze(
        images=None,
        adapter=adapter,
        validate=_validate,
        max_turns=1,
        use_tools=False,
    )
    assert result.final_response["answer"] == "ok"
    assert seen  # the validator was actually consulted


@pytest.mark.asyncio
async def test_temperature_is_threaded_to_adapter_and_runconfig() -> None:
    adapter = ScriptedAdapter([('{"continue": false, "x": 1}', None, "stop")])
    result = await analyze(
        images=None,
        adapter=adapter,
        temperature=0.7,
        max_turns=1,
        use_tools=False,
    )
    assert adapter.temperatures == [0.7]
    assert result.run_config is not None
    assert result.run_config.temperature == 0.7


@pytest.mark.asyncio
async def test_temperature_defaults_to_zero() -> None:
    adapter = ScriptedAdapter([('{"continue": false, "x": 1}', None, "stop")])
    await analyze(images=None, adapter=adapter, max_turns=1, use_tools=False)
    assert adapter.temperatures == [0.0]


def test_simple_processor_uses_explicit_adapter_without_factory() -> None:
    adapter = ScriptedAdapter([("{}", None, "stop")])
    processor = SimpleProcessor(
        system_prompt="s",
        user_message="u",
        adapter=adapter,
        use_tools=False,
    )
    processor._ensure_initialized()
    assert processor._model_adapter is adapter


@pytest.mark.asyncio
async def test_should_continue_override_stops_immediately() -> None:
    # The model asks to continue every turn, but the override forces a stop,
    # so only a single generation happens.
    class StopNowProcessor(SimpleProcessor):
        def should_continue(self, response: dict[str, Any]) -> bool:
            return False

    adapter = ScriptedAdapter([('{"continue": true, "answer": "done"}', None, "stop")])
    processor = StopNowProcessor(
        system_prompt="s",
        user_message="u",
        adapter=adapter,
        max_turns=5,
        use_tools=False,
    )
    result = await processor.analyze(images=None)
    assert result.final_response["answer"] == "done"
    assert adapter.calls == 1


def test_agentic_config_validation() -> None:
    AgenticConfig()  # defaults are valid
    with pytest.raises(ValueError, match="default_max_turns"):
        AgenticConfig(default_max_turns=0)
    with pytest.raises(ValueError, match="default_max_turns"):
        AgenticConfig(default_max_turns=50, max_turns_limit=30)
    with pytest.raises(ValueError, match="default_temperature"):
        AgenticConfig(default_temperature=3.0)
    with pytest.raises(ValueError, match="max_consecutive_nudges"):
        AgenticConfig(max_consecutive_nudges=0)


def test_agentic_config_constructor_argument_overrides_defaults() -> None:
    cfg = AgenticConfig(default_max_turns=3, default_temperature=0.5)
    processor = SimpleProcessor(
        system_prompt="s",
        user_message="u",
        adapter=ScriptedAdapter([("{}", None, "stop")]),
        agentic_config=cfg,
        use_tools=False,
    )
    assert processor.max_turns == 3
    assert processor.temperature == 0.5


def test_agentic_config_via_context_changes_defaults() -> None:
    cfg = GazeConfig(agentic=AgenticConfig(default_max_turns=4, default_temperature=0.25))
    with config_context(cfg):
        processor = SimpleProcessor(
            system_prompt="s",
            user_message="u",
            adapter=ScriptedAdapter([("{}", None, "stop")]),
            use_tools=False,
        )
        assert processor.max_turns == 4
        assert processor.temperature == 0.25


def test_explicit_temperature_beats_config_default() -> None:
    cfg = GazeConfig(agentic=AgenticConfig(default_temperature=0.9))
    with config_context(cfg):
        processor = SimpleProcessor(
            system_prompt="s",
            user_message="u",
            adapter=ScriptedAdapter([("{}", None, "stop")]),
            temperature=0.1,
            use_tools=False,
        )
        assert processor.temperature == 0.1
