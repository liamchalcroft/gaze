"""Coverage tests for verifiers/mixin.py — VerifiableProcessorMixin.as_verifiers_env.

Targets lines 89, 126-253: the dynamically generated _VerifiableEnv class
including init, get_system_prompt, _build_user_message, build_initial_state,
env_response, is_completed, and get_reward_function.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

try:
    import verifiers as vf  # noqa: F401

    from gaze.base import AgenticProcessorBase
    from gaze.verifiers.base import BaseMultiTurnEnv
    from gaze.verifiers.mixin import VerifiableProcessorMixin
    from gaze.verifiers.rewards import BaseRewardFunction

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

pytestmark = pytest.mark.skipif(not _HAS_VERIFIERS, reason="verifiers not installed")


# ---------------------------------------------------------------------------
# Minimal concrete processor for testing
# ---------------------------------------------------------------------------


class _DummyReward(BaseRewardFunction):
    """Minimal reward function for testing."""

    def __call__(self, prompt, completion, info=None, **kwargs):
        return 1.0


class _DummyProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    """Concrete processor that satisfies both mixin and base class contracts."""

    def __init__(self, **kwargs: Any) -> None:
        # Store extra kwargs for testing forwarding
        self._extra = kwargs
        # Call parent with defaults — use adapter_factory to avoid real API keys
        super().__init__(
            model_name="test-model",
            use_tools=False,
            adapter_factory=MagicMock,
        )

    def get_system_prompt(self, images=None, metadata=None) -> str:
        return "You are a test system."

    def get_user_message(self, images=None, metadata=None) -> str:
        question = metadata.get("question", "default question") if metadata else "default"
        return f"Analyze: {question}"

    def get_response_schema(self) -> dict:
        return {"type": "object", "properties": {"answer": {"type": "string"}}}

    def validate_response(self, response):
        return response

    def get_reward_function(self) -> BaseRewardFunction:
        return _DummyReward()


# ---------------------------------------------------------------------------
# as_verifiers_env — factory and generated class (lines 126-253)
# ---------------------------------------------------------------------------


class TestAsVerifiersEnv:
    def test_returns_a_class(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(
            max_turns=3,
            cases=[{"question": "What is this?"}],
        )
        assert isinstance(env_class, type)
        assert issubclass(env_class, BaseMultiTurnEnv)

    def test_env_instantiates_with_cases(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(
            max_turns=3,
            cases=[{"question": "Q"}],
        )
        env = env_class()
        assert env._max_turns == 3
        assert len(env._cases) == 1

    def test_max_turns_propagated(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(
            max_turns=7,
            cases=[],
        )
        env = env_class()
        assert env._max_turns == 7

    def test_processor_kwargs_forwarded(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            custom_param="abc",
        )
        env = env_class()
        assert env._processor._extra == {"custom_param": "abc"}


class TestEnvGetSystemPrompt:
    def test_returns_processor_system_prompt(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()
        prompt = env.get_system_prompt()
        assert prompt == "You are a test system."


class TestEnvBuildUserMessage:
    def test_text_only_case(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        msg = env._build_user_message({"question": "What is the diagnosis?"})
        assert isinstance(msg, str)
        assert "What is the diagnosis?" in msg

    def test_image_case_with_base_path(self, tmp_path: Path) -> None:
        """With image_path and image_base_path, returns multimodal content."""
        img = tmp_path / "scan.png"
        from PIL import Image

        Image.new("RGB", (2, 2), "red").save(img)

        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            image_base_path=tmp_path,
        )
        env = env_class()

        msg = env._build_user_message({"question": "Q", "image_path": "scan.png"})
        assert isinstance(msg, list)
        assert len(msg) == 2
        assert msg[0]["type"] == "text"
        assert "Q" in msg[0]["text"]
        assert msg[1]["type"] == "image_url"
        assert msg[1]["image_url"]["url"].startswith("data:image/")

    def test_absolute_image_path_not_resolved(self, tmp_path: Path) -> None:
        """Absolute image paths bypass _safe_resolve_image_path."""
        img = tmp_path / "abs.png"
        from PIL import Image

        Image.new("RGB", (2, 2), "blue").save(img)

        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            image_base_path=tmp_path,
        )
        env = env_class()

        msg = env._build_user_message({"question": "Q", "image_path": str(img)})
        assert isinstance(msg, list)
        assert msg[1]["type"] == "image_url"

    def test_image_key_fallback(self, tmp_path: Path) -> None:
        """Falls back to 'image' key when 'image_path' is absent."""
        img = tmp_path / "fallback.png"
        from PIL import Image

        Image.new("RGB", (2, 2), "green").save(img)

        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            image_base_path=tmp_path,
        )
        env = env_class()

        msg = env._build_user_message({"question": "Q", "image": "fallback.png"})
        assert isinstance(msg, list)
        assert msg[1]["type"] == "image_url"


class TestEnvSetupState:
    @pytest.mark.asyncio
    async def test_text_only_state(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        state: dict[str, Any] = {"info": {"question": "Q"}}
        state = await env.setup_state(state)
        assert "image_path" not in state
        assert state["turn"] == 0

    @pytest.mark.asyncio
    async def test_image_path_added_to_state(self, tmp_path: Path) -> None:
        img = tmp_path / "state.png"
        img.touch()

        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            image_base_path=tmp_path,
        )
        env = env_class()

        state: dict[str, Any] = {"info": {"image_path": "state.png"}}
        state = await env.setup_state(state)
        assert "image_path" in state
        assert state["image_path"] == str((tmp_path / "state.png").resolve())

    @pytest.mark.asyncio
    async def test_absolute_image_path_preserved(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(
            cases=[],
            image_base_path=Path("/base"),
        )
        env = env_class()

        state: dict[str, Any] = {"info": {"image_path": "/abs/img.png"}}
        state = await env.setup_state(state)
        assert state["image_path"] == "/abs/img.png"


class TestEnvResponse:
    @pytest.mark.asyncio
    async def test_updates_state_correctly(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        mock_result = {
            "messages": [{"role": "assistant", "content": "answer"}],
            "tool_calls": [{"id": "c1", "name": "zoom"}],
            "is_complete": False,
        }
        env._adapter = MagicMock()
        env._adapter.process_verifiers_messages = AsyncMock(return_value=mock_result)

        state: dict[str, Any] = {"turn": 0, "tool_uses": 0}
        messages = [{"role": "user", "content": "test"}]

        result_msgs = await env.env_response(messages, state)

        assert result_msgs == [{"role": "assistant", "content": "answer"}]
        # State is mutated in-place
        assert state["turn"] == 1
        assert state["tool_uses"] == 1
        assert state["is_complete"] is False

    @pytest.mark.asyncio
    async def test_passes_image_path_from_state(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        mock_result = {
            "messages": [],
            "tool_calls": [],
            "is_complete": True,
        }
        env._adapter = MagicMock()
        env._adapter.process_verifiers_messages = AsyncMock(return_value=mock_result)

        state: dict[str, Any] = {"turn": 0, "tool_uses": 0, "image_path": "/img/scan.png"}
        await env.env_response([], state)

        call_info = env._adapter.process_verifiers_messages.call_args[1]["info"]
        assert call_info["image_path"] == "/img/scan.png"

    @pytest.mark.asyncio
    async def test_info_from_state_defaults_to_empty(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        mock_result = {"messages": [], "tool_calls": [], "is_complete": True}
        env._adapter = MagicMock()
        env._adapter.process_verifiers_messages = AsyncMock(return_value=mock_result)

        state: dict[str, Any] = {"turn": 0, "tool_uses": 0}
        await env.env_response([], state)

        call_info = env._adapter.process_verifiers_messages.call_args[1]["info"]
        assert "image_path" in call_info  # from state.get("image_path") → None


class TestEnvIsCompleted:
    @staticmethod
    def _state(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "timing": {"start_time": time.time()},
            "trajectory": [{"prompt": [], "completion": []}],
            "prompt": [],
            "completion": [],
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_max_turns_reached(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(max_turns=2, cases=[])
        env = env_class()

        assert await env.is_completed(self._state(turn=2, is_complete=False)) is True

    @pytest.mark.asyncio
    async def test_is_complete_flag_in_state(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(max_turns=10, cases=[])
        env = env_class()

        assert await env.is_completed(self._state(turn=1, is_complete=True)) is True

    @pytest.mark.asyncio
    async def test_not_complete(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(max_turns=10, cases=[])
        env = env_class()

        assert await env.is_completed(self._state(turn=1, is_complete=False)) is False


class TestEnvGetRewardFunction:
    def test_delegates_to_processor(self) -> None:
        env_class = _DummyProcessor.as_verifiers_env(cases=[])
        env = env_class()

        reward_fn = env.get_reward_function()
        assert isinstance(reward_fn, _DummyReward)
        assert reward_fn("", [], {}) == 1.0


class TestEnvClassLevelDefaults:
    def test_env_cases_override_class_cases(self) -> None:
        """env_cases parameter overrides class-level cases."""
        class_cases = [{"question": "class Q"}]
        env_cases = [{"question": "env Q1"}, {"question": "env Q2"}]

        env_class = _DummyProcessor.as_verifiers_env(cases=class_cases)
        env = env_class(env_cases=env_cases)
        assert len(env._cases) == 2

    def test_class_cases_used_as_default(self) -> None:
        class_cases = [{"question": "Q1"}]
        env_class = _DummyProcessor.as_verifiers_env(cases=class_cases)
        env = env_class()
        assert len(env._cases) == 1
