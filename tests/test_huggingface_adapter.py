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
from radiant_harness.models.huggingface_adapter import _inject_json_mode
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
    """response_format should not raise ModelError; it should warn and proceed.

    The base agentic loop always passes response_format, so raising breaks
    all HF adapter usage. We verify it does NOT raise ModelError, then
    expect either ImportError (no torch in CI) or OSError (torch installed
    but model "dummy" not found) from the rest of generate_chat.
    """
    adapter = HuggingFaceAdapter(model_name="dummy")
    # Should NOT raise ModelError for response_format.
    with pytest.raises((ImportError, OSError)):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            response_format={"type": "json_object"},
        )


@pytest.mark.asyncio
async def test_huggingface_vlm_adapter_warns_on_response_format() -> None:
    """response_format should not raise ModelError for VLM adapter either."""
    adapter = HuggingFaceVLMAdapter(model_name="dummy")
    with pytest.raises((ImportError, OSError)):
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
        assert tool_calls[0]["id"].startswith("call_")
        assert tool_calls[1]["id"].startswith("call_")
        assert tool_calls[0]["id"] != tool_calls[1]["id"]

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


# ---------------------------------------------------------------------------
# max_input_length configuration
# ---------------------------------------------------------------------------


class TestMaxInputLength:
    def test_default_max_input_length_is_none(self) -> None:
        adapter = HuggingFaceAdapter(model_name="dummy")
        assert adapter._max_input_length is None

    def test_custom_max_input_length(self) -> None:
        adapter = HuggingFaceAdapter(model_name="dummy", max_input_length=4096)
        assert adapter._max_input_length == 4096

    def test_vlm_adapter_accepts_max_input_length(self) -> None:
        adapter = HuggingFaceVLMAdapter(model_name="dummy", max_input_length=8192)
        assert adapter._max_input_length == 8192


# ---------------------------------------------------------------------------
# Parity: tool_call dict shape matches OpenAI adapter format
# ---------------------------------------------------------------------------


class TestToolCallParity:
    """Verify HuggingFace tool_call dicts match OpenAI adapter shape."""

    REQUIRED_KEYS = {"id", "name", "arguments"}

    def test_single_tool_call_has_required_keys(self) -> None:
        adapter = HuggingFaceAdapter(model_name="dummy")
        content = '```tool\n{"name": "zoom", "arguments": {"x": 10}}\n```'
        tool_calls, _ = adapter._parse_tool_calls(content)

        assert tool_calls is not None
        for tc in tool_calls:
            assert set(tc.keys()) == self.REQUIRED_KEYS

    def test_arguments_is_json_string(self) -> None:
        """OpenAI returns arguments as a JSON string, not a dict. HF must match."""
        adapter = HuggingFaceAdapter(model_name="dummy")
        content = '```tool\n{"name": "zoom", "arguments": {"x": 10}}\n```'
        tool_calls, _ = adapter._parse_tool_calls(content)

        assert tool_calls is not None
        assert isinstance(tool_calls[0]["arguments"], str)
        # Must be valid JSON
        import json

        parsed = json.loads(tool_calls[0]["arguments"])
        assert isinstance(parsed, dict)

    def test_id_is_string(self) -> None:
        adapter = HuggingFaceAdapter(model_name="dummy")
        content = '```tool\n{"name": "zoom", "arguments": {}}\n```'
        tool_calls, _ = adapter._parse_tool_calls(content)

        assert tool_calls is not None
        assert isinstance(tool_calls[0]["id"], str)
        assert tool_calls[0]["id"].startswith("call_")


# ---------------------------------------------------------------------------
# Protocol signature parity
# ---------------------------------------------------------------------------


class TestInjectJsonMode:
    """Tests for _inject_json_mode prompt emulation."""

    def test_appends_to_existing_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "You are a radiologist."},
            {"role": "user", "content": "Analyze this."},
        ]
        result = _inject_json_mode(messages)
        assert "valid JSON" in result[0]["content"]
        assert result[0]["content"].startswith("You are a radiologist.")

    def test_inserts_system_message_when_missing(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_mode(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "valid JSON" in result[0]["content"]

    def test_does_not_mutate_original(self) -> None:
        messages = [{"role": "system", "content": "Original."}]
        _inject_json_mode(messages)
        assert messages[0]["content"] == "Original."


class TestProtocolSignatureParity:
    """Verify HF adapters match the AdapterProtocol signature."""

    def test_generate_chat_signature_matches_protocol(self) -> None:
        """Both HF adapters should accept the same params as the protocol."""
        import inspect

        from radiant_harness.models.adapter_protocol import AdapterProtocol

        proto_sig = inspect.signature(AdapterProtocol.generate_chat)
        hf_sig = inspect.signature(HuggingFaceAdapter.generate_chat)
        vlm_sig = inspect.signature(HuggingFaceVLMAdapter.generate_chat)

        proto_params = set(proto_sig.parameters.keys()) - {"self"}
        hf_params = set(hf_sig.parameters.keys()) - {"self"}
        vlm_params = set(vlm_sig.parameters.keys()) - {"self"}

        assert hf_params == proto_params, f"HF params {hf_params} != protocol {proto_params}"
        assert vlm_params == proto_params, f"VLM params {vlm_params} != protocol {proto_params}"

    def test_generate_chat_return_annotation_includes_async_iterator(self) -> None:
        """Return annotation should include AsyncIterator to match protocol."""
        import inspect

        hf_sig = inspect.signature(HuggingFaceAdapter.generate_chat)
        vlm_sig = inspect.signature(HuggingFaceVLMAdapter.generate_chat)

        hf_return = str(hf_sig.return_annotation)
        vlm_return = str(vlm_sig.return_annotation)

        assert "AsyncIterator" in hf_return, f"HF return {hf_return} missing AsyncIterator"
        assert "AsyncIterator" in vlm_return, f"VLM return {vlm_return} missing AsyncIterator"


# ---------------------------------------------------------------------------
# Patch Set #1 regression tests
# ---------------------------------------------------------------------------


class TestToolCallIdUniqueness:
    """Verify tool-call IDs are globally unique across multiple parse calls."""

    def test_ids_unique_across_two_parses(self) -> None:
        """Two separate _parse_tool_calls invocations must never produce the same ID."""
        adapter = HuggingFaceAdapter(model_name="dummy")

        content_a = '```tool\n{"name": "zoom", "arguments": {"x": 1}}\n```'
        content_b = '```tool\n{"name": "crop", "arguments": {"w": 50}}\n```'

        calls_a, _ = adapter._parse_tool_calls(content_a)
        calls_b, _ = adapter._parse_tool_calls(content_b)

        assert calls_a is not None and calls_b is not None
        all_ids = [tc["id"] for tc in calls_a] + [tc["id"] for tc in calls_b]
        assert len(all_ids) == len(set(all_ids)), f"Duplicate IDs found: {all_ids}"

    def test_ids_unique_across_many_parses(self) -> None:
        """Stress test: 20 sequential parses should produce 20 unique IDs."""
        adapter = HuggingFaceAdapter(model_name="dummy")
        all_ids: list[str] = []

        for _ in range(20):
            content = '```tool\n{"name": "zoom", "arguments": {}}\n```'
            calls, _ = adapter._parse_tool_calls(content)
            assert calls is not None
            all_ids.append(calls[0]["id"])

        assert len(all_ids) == len(set(all_ids)), f"Duplicate IDs in {all_ids}"


class TestVLMGenKwargsParity:
    """Verify VLM generate_chat gen_kwargs match base HuggingFaceAdapter.

    These tests inspect the source code structure to confirm the VLM path
    has the same optimizations as the base path (use_cache, non_blocking,
    autocast). They do not require torch to be installed.
    """

    def test_vlm_generate_chat_has_use_cache(self) -> None:
        """VLM gen_kwargs must include use_cache: True (parity with base)."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert '"use_cache"' in source or "'use_cache'" in source, (
            "VLM generate_chat is missing 'use_cache' in gen_kwargs"
        )

    def test_vlm_uses_non_blocking_transfer(self) -> None:
        """VLM input tensor transfer must use non_blocking=True."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert "non_blocking=True" in source, (
            "VLM generate_chat is missing non_blocking=True in .to() call"
        )

    def test_vlm_uses_autocast(self) -> None:
        """VLM _run_vlm_generate must use torch.amp.autocast."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert "torch.amp.autocast" in source, (
            "VLM generate_chat is missing torch.amp.autocast"
        )

    def test_base_uses_non_deprecated_autocast(self) -> None:
        """Base adapter must use torch.amp.autocast, not deprecated torch.cuda.amp.autocast."""
        import inspect

        source = inspect.getsource(HuggingFaceAdapter.generate_chat)
        assert "torch.cuda.amp.autocast" not in source, (
            "Base adapter still uses deprecated torch.cuda.amp.autocast"
        )
        assert "torch.amp.autocast" in source, (
            "Base adapter is missing torch.amp.autocast"
        )
