"""Tests for HuggingFace adapter behaviors without optional deps.

Tests cover:
- Streaming rejection (raises ModelError before touching torch)
- response_format handling (warns instead of raising)
- Tool schema injection into messages (_inject_tool_docs)
- Tool call parsing from model output (_parse_tool_calls)
- Tool prompt formatting (_format_tools_for_prompt)
"""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from radiant_harness.exceptions import ModelError
from radiant_harness.models import huggingface_adapter as hf_module
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


class TestExtractImagesCache:
    def test_reuses_decoded_data_url_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buffer = BytesIO()
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="PNG")
        data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

        decode_calls = 0
        real_b64decode = hf_module.base64.b64decode

        def _counting_b64decode(data: str) -> bytes:
            nonlocal decode_calls
            decode_calls += 1
            return real_b64decode(data)

        monkeypatch.setattr(hf_module.base64, "b64decode", _counting_b64decode)

        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url}}],
            }
        ]

        first = adapter._extract_images(messages)
        second = adapter._extract_images(messages)

        assert decode_calls == 1
        assert len(first) == 1
        assert len(second) == 1
        assert first[0] is not second[0]
        assert first[0].size == second[0].size == (8, 8)

        first[0].close()
        second[0].close()


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
        assert parsed == {"x": 10}

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

    def test_separate_adapters_have_isolated_counters(self) -> None:
        """Two adapter instances must start their counters independently.

        This verifies the fix for the global _tool_call_counter leak where
        all adapters shared a single monotonic counter, causing IDs from
        one session to influence another.
        """
        adapter_a = HuggingFaceAdapter(model_name="dummy-a")
        adapter_b = HuggingFaceAdapter(model_name="dummy-b")

        content = '```tool\n{"name": "zoom", "arguments": {}}\n```'

        calls_a, _ = adapter_a._parse_tool_calls(content)
        calls_b, _ = adapter_b._parse_tool_calls(content)

        assert calls_a is not None and calls_b is not None
        # Both adapters should start at call_1 since they have independent counters
        assert calls_a[0]["id"] == "call_1"
        assert calls_b[0]["id"] == "call_1"

    def test_vlm_adapter_has_isolated_counter(self) -> None:
        """VLM adapter inherits per-instance counter from base adapter."""
        adapter_base = HuggingFaceAdapter(model_name="dummy-base")
        adapter_vlm = HuggingFaceVLMAdapter(model_name="dummy-vlm")

        content = '```tool\n{"name": "zoom", "arguments": {}}\n```'

        # Use base adapter first
        calls_base, _ = adapter_base._parse_tool_calls(content)
        assert calls_base is not None
        assert calls_base[0]["id"] == "call_1"

        # VLM adapter should also start at call_1 (not call_2)
        calls_vlm, _ = adapter_vlm._parse_tool_calls(content)
        assert calls_vlm is not None
        assert calls_vlm[0]["id"] == "call_1"

    def test_counter_increments_within_instance(self) -> None:
        """Counter increments correctly within a single adapter instance."""
        adapter = HuggingFaceAdapter(model_name="dummy")
        content = '```tool\n{"name": "zoom", "arguments": {}}\n```'

        calls_1, _ = adapter._parse_tool_calls(content)
        calls_2, _ = adapter._parse_tool_calls(content)
        calls_3, _ = adapter._parse_tool_calls(content)

        assert calls_1 is not None and calls_2 is not None and calls_3 is not None
        assert calls_1[0]["id"] == "call_1"
        assert calls_2[0]["id"] == "call_2"
        assert calls_3[0]["id"] == "call_3"


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
        assert "torch.amp.autocast" in source, "VLM generate_chat is missing torch.amp.autocast"

    def test_base_uses_non_deprecated_autocast(self) -> None:
        """Base adapter must use torch.amp.autocast, not deprecated torch.cuda.amp.autocast."""
        import inspect

        source = inspect.getsource(HuggingFaceAdapter.generate_chat)
        assert "torch.cuda.amp.autocast" not in source, (
            "Base adapter still uses deprecated torch.cuda.amp.autocast"
        )
        assert "torch.amp.autocast" in source, "Base adapter is missing torch.amp.autocast"


# ---------------------------------------------------------------------------
# Patch Set #5: temperature=0 greedy decoding parity
# ---------------------------------------------------------------------------


class TestTemperatureZeroGreedyParity:
    """Verify temperature=0 produces greedy (do_sample=False) gen_kwargs.

    OpenAI treats temperature=0 as deterministic/greedy.  The HuggingFace
    adapters must match: do_sample=False and a safe temperature placeholder
    (1.0) rather than clamping to 0.01 which enables sampling.
    """

    def test_base_adapter_greedy_source(self) -> None:
        """Base generate_chat must set do_sample=False when temperature <= 0."""
        import inspect

        source = inspect.getsource(HuggingFaceAdapter.generate_chat)
        # The old broken pattern: max(temperature, 0.01)
        assert "max(temperature, 0.01)" not in source, (
            "Base adapter still clamps temperature to 0.01 instead of greedy decoding"
        )
        assert "do_sample" in source

    def test_vlm_adapter_greedy_source(self) -> None:
        """VLM generate_chat must set do_sample=False when temperature <= 0."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert "max(temperature, 0.01)" not in source, (
            "VLM adapter still clamps temperature to 0.01 instead of greedy decoding"
        )
        assert "do_sample" in source

    def test_base_greedy_flag_logic(self) -> None:
        """Verify the greedy flag: temperature <= 0 → do_sample = not greedy = False."""
        import inspect

        source = inspect.getsource(HuggingFaceAdapter.generate_chat)
        assert "greedy = temperature <= 0" in source
        assert '"do_sample": not greedy' in source

    def test_vlm_greedy_flag_logic(self) -> None:
        """Verify VLM uses the same greedy logic."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert "greedy = temperature <= 0" in source
        assert '"do_sample": not greedy' in source


# ---------------------------------------------------------------------------
# Patch Set #5: data URI MIME type validation in _extract_images
# ---------------------------------------------------------------------------


class TestExtractImagesMimeValidation:
    """Verify _extract_images rejects non-image data URIs."""

    def test_rejects_text_html_data_uri(self) -> None:
        """A data:text/html;base64,... URI must be skipped, not decoded."""
        import base64

        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        html_b64 = base64.b64encode(b"<h1>hello</h1>").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:text/html;base64,{html_b64}"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert images == [], "Non-image data URI should have been rejected"

    def test_accepts_image_png_data_uri(self) -> None:
        """A valid data:image/png;base64,... URI must be accepted."""
        import base64
        from io import BytesIO

        from PIL import Image as PILImage

        # Create a tiny valid PNG
        img = PILImage.new("RGB", (2, 2), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert len(images) == 1
        assert images[0].size == (2, 2)

    def test_rejects_application_json_data_uri(self) -> None:
        """data:application/json must be rejected."""
        import base64

        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        json_b64 = base64.b64encode(b'{"key": "value"}').decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:application/json;base64,{json_b64}"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert images == []


# ---------------------------------------------------------------------------
# Patch Set: VLM .to() guard for non-tensor processor outputs
# ---------------------------------------------------------------------------


class TestVLMInputToDeviceGuard:
    """Verify VLM generate_chat guards .to() calls for non-tensor values.

    Some HuggingFace processors return metadata (lists, ints) alongside
    tensors.  Calling .to(device) on a plain list raises AttributeError.
    The guard ``hasattr(v, 'to')`` must be present.
    """

    def test_vlm_generate_chat_has_hasattr_guard(self) -> None:
        """VLM generate_chat must guard .to() with hasattr check."""
        import inspect

        source = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)
        assert 'hasattr(v, "to")' in source, (
            "VLM generate_chat is missing hasattr(v, 'to') guard on inputs dict"
        )

    def test_base_adapter_has_no_guard_needed(self) -> None:
        """Base adapter uses tokenizer (always returns tensors), no guard needed."""
        import inspect

        source = inspect.getsource(HuggingFaceAdapter.generate_chat)
        # Base adapter uses self.tokenizer() which always returns tensors,
        # so the guard is not required there. Verify it's using .to() directly.
        assert ".to(self.device" in source


# ---------------------------------------------------------------------------
# Patch Set: @beartype on VLMAdapter.__init__
# ---------------------------------------------------------------------------


class TestVLMAdapterBeartype:
    """Verify VLM adapter __init__ is decorated with @beartype."""

    def test_vlm_init_rejects_invalid_model_name_type(self) -> None:
        """@beartype should reject non-str model_name."""
        from beartype.roar import BeartypeException

        with pytest.raises((BeartypeException, TypeError)):
            HuggingFaceVLMAdapter(model_name=12345)  # type: ignore[arg-type]

    def test_vlm_init_rejects_invalid_device_type(self) -> None:
        """@beartype should reject non-str device."""
        from beartype.roar import BeartypeException

        with pytest.raises((BeartypeException, TypeError)):
            HuggingFaceVLMAdapter(model_name="dummy", device=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PR #2: Adapter security hardening
# ---------------------------------------------------------------------------


class TestTrustRemoteCodeWarning:
    """trust_remote_code=True must emit a warning."""

    def test_warning_emitted_when_true(self) -> None:
        from unittest.mock import patch

        with patch("radiant_harness.models.huggingface_adapter.logger") as mock_logger:
            HuggingFaceAdapter(model_name="dummy", trust_remote_code=True)
            mock_logger.warning.assert_called_once()
            assert "trust_remote_code" in mock_logger.warning.call_args[0][0]
            assert "arbitrary code" in mock_logger.warning.call_args[0][0]

    def test_no_warning_when_false(self) -> None:
        from unittest.mock import patch

        with patch("radiant_harness.models.huggingface_adapter.logger") as mock_logger:
            HuggingFaceAdapter(model_name="dummy", trust_remote_code=False)
            mock_logger.warning.assert_not_called()

    def test_vlm_adapter_inherits_warning(self) -> None:
        from unittest.mock import patch

        with patch("radiant_harness.models.huggingface_adapter.logger") as mock_logger:
            HuggingFaceVLMAdapter(model_name="dummy", trust_remote_code=True)
            mock_logger.warning.assert_called_once()
            assert "trust_remote_code" in mock_logger.warning.call_args[0][0]


class TestExtractImagesRejectsLocalPaths:
    """_extract_images must reject arbitrary local file paths."""

    def test_local_path_rejected(self) -> None:
        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "/etc/passwd"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert images == [], "Local file paths must be rejected"

    def test_relative_path_rejected(self) -> None:
        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "../../sensitive/data.png"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert images == [], "Relative file paths must be rejected"

    def test_data_uri_still_accepted(self) -> None:
        """Data URIs should still work (regression check)."""
        import base64
        from io import BytesIO

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (2, 2), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        adapter = HuggingFaceVLMAdapter(model_name="dummy")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                ],
            },
        ]
        images = adapter._extract_images(messages)
        assert len(images) == 1


# ---------------------------------------------------------------------------
# GenerationLog.reasoning_content parity
# ---------------------------------------------------------------------------


class TestReasoningContentParity:
    """HF adapters never populate reasoning_content — document the gap."""

    def test_hf_generation_log_has_no_reasoning_content(self) -> None:
        """HF GenerationLog always has reasoning_content=None."""
        from radiant_harness.models.adapter_protocol import GenerationLog

        # Simulate what HF adapters produce
        gen_log = GenerationLog(
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
        )
        assert gen_log.reasoning_content is None

    def test_hf_generate_chat_source_omits_reasoning_content(self) -> None:
        """HF adapters must not set reasoning_content (intentional gap)."""
        import inspect

        base_src = inspect.getsource(HuggingFaceAdapter.generate_chat)
        vlm_src = inspect.getsource(HuggingFaceVLMAdapter.generate_chat)

        assert "reasoning_content" not in base_src, (
            "HF base adapter unexpectedly sets reasoning_content"
        )
        assert "reasoning_content" not in vlm_src, (
            "HF VLM adapter unexpectedly sets reasoning_content"
        )


# ---------------------------------------------------------------------------
# Error type parity: HF raises ModelError, not APIError
# ---------------------------------------------------------------------------


class TestErrorTypeParity:
    """HF adapters raise ModelError, never APIError — document the divergence."""

    def test_hf_does_not_import_api_error(self) -> None:
        """HF adapter module must not import APIError (intentional divergence)."""
        import inspect

        source = inspect.getsource(hf_module)
        assert "from radiant_harness.exceptions import APIError" not in source
        assert "APIError" not in source

    @pytest.mark.asyncio
    async def test_streaming_raises_model_error_not_api_error(self) -> None:
        """Streaming rejection must raise ModelError specifically."""
        from radiant_harness.exceptions import APIError
        from radiant_harness.exceptions import ModelError

        adapter = HuggingFaceAdapter(model_name="dummy")
        with pytest.raises(ModelError) as exc_info:
            await adapter.generate_chat(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
                temperature=0.0,
                stream=True,
            )
        assert not isinstance(exc_info.value, APIError)


# ---------------------------------------------------------------------------
# _inject_json_mode: schema inclusion
# ---------------------------------------------------------------------------


class TestInjectJsonModeSchema:
    """Verify _inject_json_mode includes schema content when provided."""

    def test_json_object_format_no_schema(self) -> None:
        """Plain json_object format produces generic instruction only."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _inject_json_mode(messages, {"type": "json_object"})
        assert "valid JSON" in result[0]["content"]
        assert "json_schema" not in result[0]["content"]

    def test_json_schema_format_includes_schema(self) -> None:
        """json_schema format must include the schema in the prompt."""
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "confidence"],
                },
            },
        }
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _inject_json_mode(messages, schema_format)
        content = result[0]["content"]
        assert "valid JSON" in content
        assert '"answer"' in content
        assert '"confidence"' in content
        assert '"required"' in content

    def test_none_response_format_produces_generic_instruction(self) -> None:
        """None response_format (backward compat) produces generic instruction."""
        messages = [{"role": "system", "content": "Base."}]
        result = _inject_json_mode(messages)
        assert "valid JSON" in result[0]["content"]

    def test_does_not_mutate_original(self) -> None:
        messages = [{"role": "system", "content": "Original."}]
        schema_format = {
            "type": "json_schema",
            "json_schema": {"name": "r", "schema": {"type": "object"}},
        }
        _inject_json_mode(messages, schema_format)
        assert messages[0]["content"] == "Original."
