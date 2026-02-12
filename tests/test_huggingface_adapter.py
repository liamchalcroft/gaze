"""Tests for HuggingFace adapter behaviors without optional deps.

Tests cover:
- Streaming rejection (raises ModelError before touching torch)
- response_format handling (warns instead of raising)
- Tool schema injection into messages (_inject_tool_docs)
- Tool call parsing from model output (_parse_tool_calls)
- Tool prompt formatting (_format_tools_for_prompt)
"""

from __future__ import annotations

import pytest

from radiant_harness.exceptions import ModelError
from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter
from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter
from radiant_harness.models.huggingface_adapter import _format_tools_for_prompt
from radiant_harness.models.huggingface_adapter import _inject_tool_docs

# ---------------------------------------------------------------------------
# Streaming still raises before torch is needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_huggingface_adapter_rejects_streaming() -> None:
    adapter = HuggingFaceAdapter(model_name="dummy")
    with pytest.raises(ModelError, match="Streaming is not supported"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            stream=True,
        )


@pytest.mark.asyncio
async def test_huggingface_vlm_adapter_rejects_streaming() -> None:
    adapter = HuggingFaceVLMAdapter(model_name="dummy")
    with pytest.raises(ModelError, match="Streaming is not supported"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            stream=True,
        )


# ---------------------------------------------------------------------------
# response_format: warn, don't raise (parity fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_huggingface_adapter_warns_on_response_format() -> None:
    """response_format should not raise; it should warn and proceed.

    The base agentic loop always passes response_format, so raising breaks
    all HF adapter usage. We verify it no longer raises ModelError, then
    expect ImportError (no torch in CI) from the rest of generate_chat.
    """
    adapter = HuggingFaceAdapter(model_name="dummy")
    # Should NOT raise ModelError for response_format.
    # It will raise ImportError because torch isn't installed in test env.
    with pytest.raises(ImportError, match="torch"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            response_format={"type": "json_object"},
        )


@pytest.mark.asyncio
async def test_huggingface_vlm_adapter_warns_on_response_format() -> None:
    adapter = HuggingFaceVLMAdapter(model_name="dummy")
    with pytest.raises(ImportError, match="torch"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            response_format={"type": "json_object"},
        )


# ---------------------------------------------------------------------------
# Tool prompt formatting (pure functions, no torch needed)
# ---------------------------------------------------------------------------

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": "Zoom into a region of the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "level": {"type": "integer", "description": "Zoom level"},
                },
                "required": ["x", "y", "level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contrast",
            "description": "Adjust image contrast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {"type": "number", "description": "Contrast factor"},
                },
                "required": ["factor"],
            },
        },
    },
]


class TestFormatToolsForPrompt:
    def test_contains_tool_names(self) -> None:
        text = _format_tools_for_prompt(_SAMPLE_TOOLS)
        assert "zoom" in text
        assert "contrast" in text

    def test_contains_param_descriptions(self) -> None:
        text = _format_tools_for_prompt(_SAMPLE_TOOLS)
        assert "X coordinate" in text
        assert "Contrast factor" in text

    def test_marks_required_params(self) -> None:
        text = _format_tools_for_prompt(_SAMPLE_TOOLS)
        assert "(required)" in text

    def test_contains_example_block(self) -> None:
        text = _format_tools_for_prompt(_SAMPLE_TOOLS)
        assert "```tool" in text

    def test_empty_tools_produces_preamble_only(self) -> None:
        text = _format_tools_for_prompt([])
        assert "Available Tools" in text


# ---------------------------------------------------------------------------
# Tool doc injection into messages
# ---------------------------------------------------------------------------


class TestInjectToolDocs:
    def test_appends_to_existing_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "You are a radiologist."},
            {"role": "user", "content": "Analyze this scan."},
        ]
        result = _inject_tool_docs(messages, _SAMPLE_TOOLS)

        # Original list unmodified
        assert messages[0]["content"] == "You are a radiologist."

        # System message now includes tool docs
        assert result[0]["role"] == "system"
        assert "You are a radiologist." in result[0]["content"]
        assert "zoom" in result[0]["content"]
        assert "contrast" in result[0]["content"]

    def test_inserts_system_message_when_missing(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_tool_docs(messages, _SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "zoom" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_does_not_mutate_original(self) -> None:
        messages = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "Hello"},
        ]
        _inject_tool_docs(messages, _SAMPLE_TOOLS)
        assert messages[0]["content"] == "Original."


# ---------------------------------------------------------------------------
# Tool call parsing (unit test the regex parser)
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def setup_method(self) -> None:
        self.adapter = HuggingFaceAdapter(model_name="dummy")

    def test_single_tool_call(self) -> None:
        content = 'I will zoom in.\n```tool\n{"name": "zoom", "arguments": {"x": 10}}\n```\nDone.'
        tool_calls, remaining = self.adapter._parse_tool_calls(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "zoom"
        assert '"x": 10' in tool_calls[0]["arguments"]
        assert "```tool" not in remaining
        assert "I will zoom in." in remaining

    def test_multiple_tool_calls(self) -> None:
        content = (
            '```tool\n{"name": "zoom", "arguments": {"x": 1}}\n```\n'
            '```tool\n{"name": "crop", "arguments": {"w": 50}}\n```'
        )
        tool_calls, _ = self.adapter._parse_tool_calls(content)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "call_0"
        assert tool_calls[1]["id"] == "call_1"

    def test_no_tool_calls(self) -> None:
        content = "Just some text without any tool calls."
        tool_calls, remaining = self.adapter._parse_tool_calls(content)
        assert tool_calls is None
        assert remaining == content

    def test_malformed_json_skipped(self) -> None:
        content = "```tool\n{bad json}\n```\nSome text."
        tool_calls, remaining = self.adapter._parse_tool_calls(content)
        assert tool_calls is None
        assert "Some text." in remaining
