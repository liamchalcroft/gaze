"""Public model interfaces for the VLM harness."""

from __future__ import annotations

from radiant_harness.models._types import GenerationLog
from radiant_harness.models.adapter_protocol import AdapterProtocol
from radiant_harness.models.openai_adapter import OpenAIAdapter

__all__ = ["AdapterProtocol", "GenerationLog", "OpenAIAdapter"]
