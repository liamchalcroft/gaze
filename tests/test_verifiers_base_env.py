"""Tests for radiant_harness.verifiers.base — BaseMultiTurnEnv.

Covers __init__, case loading, prompt building, state management,
turn tracking, debug logging, text extraction, and tool request parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    from radiant_harness.verifiers.base import BaseMultiTurnEnv

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

pytestmark = pytest.mark.skipif(not _HAS_VERIFIERS, reason="verifiers not installed")


# ---------------------------------------------------------------------------
# __init__ with in-memory cases
# ---------------------------------------------------------------------------


class TestBaseMultiTurnEnvInit:
    def test_init_with_cases(self) -> None:
        cases = [
            {"question": "What is the diagnosis?"},
            {"question": "Describe the lesion."},
        ]
        env = BaseMultiTurnEnv(cases=cases, max_turns=5, name="TestEnv")
        assert env._max_turns == 5
        assert len(env._cases) == 2
        assert env._cases[0]["question"] == "What is the diagnosis?"

    def test_init_empty_cases_default(self) -> None:
        env = BaseMultiTurnEnv(cases=None, max_turns=3, name="EmptyEnv")
        assert env._cases == []

    def test_init_log_dir_custom(self, tmp_path: Path) -> None:
        env = BaseMultiTurnEnv(cases=[], log_dir=tmp_path, name="LogEnv")
        assert env._log_dir == tmp_path
        assert env._log_path == tmp_path / "logenv_debug.log"

    def test_init_log_dir_default(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="DefaultLog")
        assert env._log_dir.name == "logs"


# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "cases.jsonl"
        rows = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        loaded = BaseMultiTurnEnv._load_jsonl(str(jsonl))
        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"
        assert loaded[1]["answer"] == "A2"

    def test_load_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "sparse.jsonl"
        jsonl.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        loaded = BaseMultiTurnEnv._load_jsonl(str(jsonl))
        assert len(loaded) == 2

    def test_init_with_dataset_path(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "ds.jsonl"
        jsonl.write_text('{"question": "What?"}\n')
        env = BaseMultiTurnEnv(dataset_path=str(jsonl), name="FromFile")
        assert len(env._cases) == 1
        assert env._cases[0]["question"] == "What?"


# ---------------------------------------------------------------------------
# _prepare_cases / _build_prompt / _build_user_message
# ---------------------------------------------------------------------------


class TestCaseProcessing:
    def test_prepare_cases_builds_prompts_and_infos(self) -> None:
        cases = [{"question": "Q1"}, {"question": "Q2"}]
        env = BaseMultiTurnEnv(cases=cases, name="Prep")
        prompts, infos = env._prepare_cases(cases)

        assert len(prompts) == 2
        assert len(infos) == 2

        # Each prompt starts with system + user messages
        assert prompts[0][0]["role"] == "system"
        assert prompts[0][1]["role"] == "user"

        # Info includes case_index and original fields
        assert infos[0]["case_index"] == 0
        assert infos[0]["question"] == "Q1"
        assert infos[1]["case_index"] == 1

    def test_build_prompt_includes_system_and_user(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="Prompt")
        prompt = env._build_prompt({"question": "Is there a lesion?"})

        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert "helpful assistant" in prompt[0]["content"].lower()
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "Is there a lesion?"

    def test_build_user_message_uses_question_key(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="Msg")
        assert env._build_user_message({"question": "Hello"}) == "Hello"

    def test_build_user_message_falls_back_to_str(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="Msg")
        result = env._build_user_message({"data": 42})
        assert "42" in result  # str(case) includes the value


# ---------------------------------------------------------------------------
# get_system_prompt / build_initial_state
# ---------------------------------------------------------------------------


class TestDefaultBehavior:
    def test_get_system_prompt_returns_string(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="Sys")
        prompt = env.get_system_prompt()
        assert "helpful assistant" in prompt.lower()
        assert "accurately" in prompt.lower()

    def test_build_initial_state_structure(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="State")
        state = env.build_initial_state(prompt=[], info={"case_index": 0})
        assert state["turn"] == 0
        assert state["tool_uses"] == 0
        assert state["info"]["case_index"] == 0


# ---------------------------------------------------------------------------
# is_completed / env_response (async)
# ---------------------------------------------------------------------------


class TestTurnManagement:
    @pytest.mark.asyncio
    async def test_is_completed_false_below_max_turns(self) -> None:
        env = BaseMultiTurnEnv(cases=[], max_turns=5, name="Comp")
        assert await env.is_completed([], {"turn": 4}) is False

    @pytest.mark.asyncio
    async def test_is_completed_true_at_max_turns(self) -> None:
        env = BaseMultiTurnEnv(cases=[], max_turns=5, name="Comp")
        assert await env.is_completed([], {"turn": 5}) is True

    @pytest.mark.asyncio
    async def test_is_completed_true_above_max_turns(self) -> None:
        env = BaseMultiTurnEnv(cases=[], max_turns=3, name="Comp")
        assert await env.is_completed([], {"turn": 10}) is True

    @pytest.mark.asyncio
    async def test_env_response_increments_turn(self) -> None:
        env = BaseMultiTurnEnv(cases=[], max_turns=10, name="Resp")
        messages, new_state = await env.env_response([], {"turn": 2})
        assert messages == []
        assert new_state["turn"] == 3

    @pytest.mark.asyncio
    async def test_env_response_from_zero(self) -> None:
        env = BaseMultiTurnEnv(cases=[], name="Resp")
        _, new_state = await env.env_response([], {})
        assert new_state["turn"] == 1


# ---------------------------------------------------------------------------
# _log_debug
# ---------------------------------------------------------------------------


class TestLogDebug:
    def test_log_debug_creates_file(self, tmp_path: Path) -> None:
        env = BaseMultiTurnEnv(cases=[], log_dir=tmp_path, name="Logger")
        env._log_debug("test entry 1")
        env._log_debug("test entry 2")

        log_file = tmp_path / "logger_debug.log"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert lines == ["test entry 1", "test entry 2"]


# ---------------------------------------------------------------------------
# _last_assistant_text
# ---------------------------------------------------------------------------


class TestLastAssistantText:
    def setup_method(self) -> None:
        self.env = BaseMultiTurnEnv(cases=[], name="Ast")

    def test_extracts_string_content(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "The lesion is in the frontal lobe."},
        ]
        assert self.env._last_assistant_text(messages) == "The lesion is in the frontal lobe."

    def test_extracts_from_multimodal_content(self) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "Multimodal text."},
                ],
            }
        ]
        assert self.env._last_assistant_text(messages) == "Multimodal text."

    def test_returns_empty_for_no_assistant(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        assert self.env._last_assistant_text(messages) == ""

    def test_returns_last_assistant_not_first(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "second"},
        ]
        assert self.env._last_assistant_text(messages) == "second"

    def test_returns_empty_for_empty_list(self) -> None:
        assert self.env._last_assistant_text([]) == ""


# ---------------------------------------------------------------------------
# _extract_tool_request
# ---------------------------------------------------------------------------


class TestExtractToolRequest:
    def setup_method(self) -> None:
        self.env = BaseMultiTurnEnv(cases=[], name="Tool")

    def test_parses_numeric_args(self) -> None:
        result = self.env._extract_tool_request("ZOOM [100, 200, 2]", ["zoom", "crop"])
        assert result is not None
        name, args = result
        assert name == "zoom"
        assert args == [100.0, 200.0, 2.0]

    def test_parses_string_args(self) -> None:
        # Note: parser uppercases text before matching, so string args are uppercase
        result = self.env._extract_tool_request("CROP [left, right]", ["zoom", "crop"])
        assert result is not None
        name, args = result
        assert name == "crop"
        assert args == ["LEFT", "RIGHT"]

    def test_case_insensitive_matching(self) -> None:
        result = self.env._extract_tool_request("zoom [10, 20]", ["ZOOM"])
        assert result is not None
        assert result[0] == "zoom"

    def test_returns_none_for_no_match(self) -> None:
        result = self.env._extract_tool_request("No tool here", ["zoom", "crop"])
        assert result is None

    def test_returns_none_for_unknown_tool(self) -> None:
        result = self.env._extract_tool_request("ROTATE [90]", ["zoom", "crop"])
        assert result is None

    def test_first_matching_tool_wins(self) -> None:
        text = "ZOOM [1] then CROP [2, 3]"
        result = self.env._extract_tool_request(text, ["zoom", "crop"])
        assert result is not None
        assert result[0] == "zoom"
        assert result[1] == [1.0]
