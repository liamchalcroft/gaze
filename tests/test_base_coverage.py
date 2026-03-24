"""Tests targeting uncovered lines in base.py."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import ImageInput
from radiant_harness.base import _downscale_image
from radiant_harness.base import _try_wrap_inner_schema
from radiant_harness.config import HarnessConfig
from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import config_context
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import GenerationLog
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import encode_image
from radiant_harness.types import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _echo_tool(registry: ToolRegistry, value: int = 0) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="echo", description=f"echo {value}", metadata={"value": value})


def _make_processor_cls(
    adapter_cls: type,
    *,
    max_turns: int = 3,
    use_tools: bool = True,
    schema: dict[str, Any] | None = None,
    validator: Any = None,
) -> type:
    """Build a concrete AgenticProcessorBase subclass with the given adapter."""
    _schema = schema or {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
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
    _validator = validator or (lambda self, r: {"continue", "result"} <= set(r))

    class _Proc(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=use_tools,
                max_turns=max_turns,
                adapter_factory=adapter_cls,
            )

        def get_system_prompt(self, images, metadata):
            return "system"

        def get_user_message(self, images, metadata):
            return "user"

        def get_response_schema(self):
            return _schema

        def validate_response(self, response):
            return _validator(self, response)

        def _create_tool_registry(self, images):
            tool = Tool(
                name="echo",
                description="echo",
                parameters={"value": {"type": "integer", "description": "v", "default": 0}},
                execute=_echo_tool,
                requires_image=False,
            )
            return ToolRegistry(image_path=None, tools=[tool])

    return _Proc


# ---------------------------------------------------------------------------
# ImageInput._load_pil_only / _aload_pil_only  (lines 225-253)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPilOnly:
    def test_already_loaded_returns_self(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        pil = Image.new("RGB", (10, 10), "red")
        pil.save(p)
        inp = ImageInput(path=p, pil_image=pil, width=10, height=10)
        result = inp._load_pil_only()
        assert result is inp

    def test_loads_from_disk_without_encoding(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        Image.new("RGB", (50, 50), "blue").save(p)
        inp = ImageInput(path=p)
        loaded = inp._load_pil_only()
        assert loaded.pil_image is not None
        assert loaded.pil_image.size == (50, 50)
        assert loaded.encoded is None
        assert loaded.width == 50
        assert loaded.height == 50

    def test_oversized_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "big.png"
        Image.new("RGB", (80, 80), "green").save(p)
        with config_context(HarnessConfig(image=ImageProcessingConfig(max_image_dimension=50))):
            inp = ImageInput(path=p)
            with pytest.raises(ValueError, match="exceed maximum"):
                inp._load_pil_only()

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.png"
        p.write_text("not an image")
        inp = ImageInput(path=p)
        with pytest.raises(ValueError, match="Failed to load"):
            inp._load_pil_only()

    @pytest.mark.asyncio
    async def test_aload_pil_only_already_loaded(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        pil = Image.new("RGB", (10, 10), "red")
        pil.save(p)
        inp = ImageInput(path=p, pil_image=pil, width=10, height=10)
        result = await inp._aload_pil_only()
        assert result is inp

    @pytest.mark.asyncio
    async def test_aload_pil_only_from_disk(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        Image.new("RGB", (30, 30), "green").save(p)
        inp = ImageInput(path=p)
        loaded = await inp._aload_pil_only()
        assert loaded.pil_image is not None
        assert loaded.pil_image.size == (30, 30)
        assert loaded.encoded is None


# ---------------------------------------------------------------------------
# _try_wrap_inner_schema  (lines 256-304)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTryWrapInnerSchema:
    SCHEMA: dict[str, Any] = {
        "json_schema": {
            "schema": {
                "properties": {
                    "caption": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "score": {"type": "number"},
                        },
                    },
                    "detection": {"type": "array"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "notes": {"type": "string"},
                    "continue": {"type": "boolean"},
                }
            }
        }
    }

    def test_wraps_matching_inner_keys(self) -> None:
        salvaged = {"text": "hello", "score": 0.9, "continue": False}
        result = _try_wrap_inner_schema(salvaged, self.SCHEMA)
        assert result["caption"] == {"text": "hello", "score": 0.9}
        assert result["detection"] == []
        assert result["count"] == 0
        assert result["flag"] is False
        assert result["notes"] == ""
        assert result["continue"] is False

    def test_no_match_returns_unchanged(self) -> None:
        salvaged = {"unknown_key": "value", "continue": False}
        result = _try_wrap_inner_schema(salvaged, self.SCHEMA)
        assert result is salvaged

    def test_empty_inner_keys_skipped(self) -> None:
        schema = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "empty_obj": {"type": "object", "properties": {}},
                        "continue": {"type": "boolean"},
                    }
                }
            }
        }
        salvaged = {"continue": False}
        result = _try_wrap_inner_schema(salvaged, schema)
        assert result is salvaged

    def test_non_object_properties_skipped(self) -> None:
        schema = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "items": {"type": "array"},
                        "continue": {"type": "boolean"},
                    }
                }
            }
        }
        salvaged = {"items": [1, 2], "continue": False}
        result = _try_wrap_inner_schema(salvaged, schema)
        assert result is salvaged


# ---------------------------------------------------------------------------
# _downscale_image  (lines 314-341)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownscaleImage:
    def test_no_pil_image_returns_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        Image.new("RGB", (10, 10)).save(p)
        inp = ImageInput(path=p)
        result = _downscale_image(inp, 100)
        assert result is inp

    def test_fits_already_encoded_returns_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        pil = Image.new("RGB", (50, 50))
        pil.save(p)
        enc = encode_image(pil)
        inp = ImageInput(path=p, width=50, height=50, encoded=enc, pil_image=pil)
        result = _downscale_image(inp, 100)
        assert result is inp

    def test_fits_needs_encoding(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        pil = Image.new("RGB", (50, 50), "red")
        pil.save(p)
        inp = ImageInput(path=p, width=50, height=50, encoded=None, pil_image=pil)
        result = _downscale_image(inp, 100)
        assert result.encoded is not None
        assert len(result.encoded.data) > 0
        assert result.width == 50
        assert result.height == 50

    def test_too_large_downscales(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        pil = Image.new("RGB", (200, 100), "blue")
        pil.save(p)
        inp = ImageInput(path=p, width=200, height=100, encoded=None, pil_image=pil)
        result = _downscale_image(inp, 50)
        assert result.width <= 50
        assert result.height <= 50
        assert result.encoded is not None
        assert len(result.encoded.data) > 0


# ---------------------------------------------------------------------------
# __init__ validation  (lines 432, 449)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitValidation:
    def test_max_turns_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_turns must be >= 1"):

            class P(AgenticProcessorBase):
                def get_system_prompt(self, images, metadata):
                    return ""

                def get_user_message(self, images, metadata):
                    return ""

                def get_response_schema(self):
                    return None

                def validate_response(self, response):
                    return True

            P.__init__ = lambda self: AgenticProcessorBase.__init__(self, max_turns=0)
            P()

    def test_max_turns_clamped_to_limit(self) -> None:
        class P(AgenticProcessorBase):
            def __init__(self):
                super().__init__(max_turns=100)

            def get_system_prompt(self, images, metadata):
                return ""

            def get_user_message(self, images, metadata):
                return ""

            def get_response_schema(self):
                return None

            def validate_response(self, response):
                return True

        p = P()
        assert p.max_turns == 30  # _MAX_TURNS_LIMIT


# ---------------------------------------------------------------------------
# Tool name sanitization  (lines 1034-1058)
# ---------------------------------------------------------------------------


class SanitizationAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(self, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return (
                "",
                [
                    {"id": "c1", "name": None, "arguments": '{"value": 1}'},
                    {"id": "c2", "name": "<|end|>", "arguments": '{"value": 2}'},
                    {"id": "c3", "name": "echo()", "arguments": None},
                ],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        return (
            '{"continue": false, "result": "ok"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_name_sanitization() -> None:
    """None names and empty-after-sanitization names are skipped; trailing parens stripped."""
    Proc = _make_processor_cls(SanitizationAdapter)
    processor = Proc()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "ok"
    # Only the echo() call (cleaned to "echo") should have produced a result
    tool_turns = [t for t in result.turns if t.role == "tool_result"]
    assert len(tool_turns) == 1
    # The one valid tool call should have succeeded
    assert tool_turns[0].tool_results[0].tool_name == "echo"
    assert tool_turns[0].tool_results[0].success is True


# ---------------------------------------------------------------------------
# Truncation on intermediate turn  (lines 1209-1230)
# ---------------------------------------------------------------------------


class TruncationAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(self, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return (
                "partial text that was cut off...",
                None,
                GenerationLog(
                    prompt_tokens=100, completion_tokens=100, finish_reason="length"
                ),
            )
        return (
            '{"continue": false, "result": "recovered"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_truncation_intermediate_turn_nudges() -> None:
    """Truncated intermediate turn adds nudge message; model recovers on next turn."""
    Proc = _make_processor_cls(TruncationAdapter, max_turns=5)
    processor = Proc()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "recovered"
    assert len(result.turns) >= 2


# ---------------------------------------------------------------------------
# Incomplete response nudge  (lines 1419-1457)
# ---------------------------------------------------------------------------


class IncompleteAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(self, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # Missing "result" → fails validation
            return (
                '{"continue": false}',
                None,
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
            )
        return (
            '{"continue": false, "result": "fixed"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_incomplete_response_nudge() -> None:
    """Incomplete JSON (fails validation) with turns left → nudge, then recover."""
    Proc = _make_processor_cls(IncompleteAdapter, max_turns=5)
    processor = Proc()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "fixed"
    assert len(result.turns) >= 2


# ---------------------------------------------------------------------------
# Force completion on last turn  (lines 1463-1466)
# ---------------------------------------------------------------------------


class AlwaysContinueAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(self, messages=None, **kwargs):
        return (
            '{"continue": true, "result": "forced"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_force_completion_on_last_turn() -> None:
    """Model always says continue=true → forced to false on last turn."""
    Proc = _make_processor_cls(AlwaysContinueAdapter, max_turns=1, use_tools=False)
    processor = Proc()
    result = await processor.analyze(images=None, metadata={})
    assert result.final_response["result"] == "forced"
    assert result.final_response["continue"] is False


# ---------------------------------------------------------------------------
# Non-boolean continue flag  (line 1408)
# ---------------------------------------------------------------------------


class BadContinueAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(self, **kwargs):
        # Use a list for "continue" — not bool, int, or str → hits the else branch
        return (
            '{"continue": [1, 2], "result": "bad"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_boolean_continue_raises() -> None:
    """Non-bool, non-string 'continue' value raises AgenticProcessingError."""
    Proc = _make_processor_cls(BadContinueAdapter, max_turns=1, use_tools=False)
    processor = Proc()
    with pytest.raises(AgenticProcessingError, match="continue.*must be boolean"):
        await processor.analyze(images=None, metadata={})


# ---------------------------------------------------------------------------
# _format_tool_results empty  (line 1737)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_tool_results_empty() -> None:
    Proc = _make_processor_cls(AlwaysContinueAdapter, use_tools=False)
    processor = Proc()
    result = processor._format_tool_results([])
    assert result == "No tools were executed."


# ---------------------------------------------------------------------------
# Streaming response  (line 1014)
# ---------------------------------------------------------------------------


class StreamingAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(self, **kwargs):
        async def _stream():
            yield "chunk"

        return _stream()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_response_raises() -> None:
    """Streaming responses raise AgenticProcessingError."""
    Proc = _make_processor_cls(StreamingAdapter, max_turns=1, use_tools=False)
    processor = Proc()
    with pytest.raises(AgenticProcessingError, match="Streaming"):
        await processor.analyze(images=None, metadata={})


# ---------------------------------------------------------------------------
# Downscale path in analyze()  (lines 728-731)
# ---------------------------------------------------------------------------


class SingleTurnAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(self, **kwargs):
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_analyze_with_max_encode_dimension(tmp_path: Path) -> None:
    """analyze() with max_encode_dimension downscales images before processing."""

    class DownscaleProc(AgenticProcessorBase):
        def __init__(self) -> None:
            super().__init__(
                model_name="test",
                use_tools=False,
                max_turns=1,
                adapter_factory=SingleTurnAdapter,
                max_encode_dimension=32,
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

        def validate_response(self, response):
            return "result" in response

    p = tmp_path / "large.png"
    Image.new("RGB", (200, 200), "red").save(p)
    processor = DownscaleProc()
    result = await processor.analyze(images=p, metadata={})
    assert result.final_response["result"] == "done"


# ---------------------------------------------------------------------------
# Truncation salvage on last turn with _try_wrap_inner_schema (lines 1234-1251)
# ---------------------------------------------------------------------------


class LastTurnTruncationAdapter(AdapterProtocol):
    supports_multipart_tool_content: bool = True

    async def generate_chat(self, **kwargs):
        # Return truncated JSON on the only turn (last turn)
        return (
            '{"text": "salvaged caption", "score": 0.8}',
            None,
            GenerationLog(prompt_tokens=100, completion_tokens=500, finish_reason="length"),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_truncation_salvage_on_last_turn() -> None:
    """Truncated last turn salvages partial JSON and wraps via _try_wrap_inner_schema."""
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "strict": True,
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
                    "result": {"type": "string"},
                    "continue": {"type": "boolean"},
                },
                "required": ["caption", "result", "continue"],
                "additionalProperties": False,
            },
        },
    }

    def _validate(self, r):
        return "caption" in r and isinstance(r.get("caption"), dict)

    Proc = _make_processor_cls(
        LastTurnTruncationAdapter, max_turns=1, use_tools=False,
        schema=schema, validator=_validate,
    )
    processor = Proc()
    result = await processor.analyze(images=None, metadata={})
    # Should have salvaged and wrapped inner keys under "caption"
    assert result.final_response["caption"]["text"] == "salvaged caption"
    assert result.final_response["caption"]["score"] == 0.8
    assert result.final_response["continue"] is False
