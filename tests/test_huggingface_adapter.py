"""Tests for HuggingFace adapter behaviors without optional deps."""

from __future__ import annotations

import pytest

from radiant_harness.exceptions import ModelError
from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter
from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter


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
async def test_huggingface_adapter_rejects_response_format() -> None:
    adapter = HuggingFaceAdapter(model_name="dummy")
    with pytest.raises(ModelError, match="response_format"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            response_format={"type": "json_object"},
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


@pytest.mark.asyncio
async def test_huggingface_vlm_adapter_rejects_response_format() -> None:
    adapter = HuggingFaceVLMAdapter(model_name="dummy")
    with pytest.raises(ModelError, match="response_format"):
        await adapter.generate_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
