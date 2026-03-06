"""Adapter utilities for integrating Radiant Harness with verifiers.

Provides utilities to convert between Radiant Harness formats
and verifiers package formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import verifiers as vf
from beartype import beartype
from verifiers.types import Messages
from verifiers.types import State

from radiant_harness._frozen import deep_thaw

from .. import AgenticProcessorBase
from ..types import AgenticResult


class RadiantHarnessAdapter:
    """Adapter for using Radiant Harness processors with verifiers.

    Bridges the gap between the two packages:
    - Converts messages between formats
    - Collects tool calls and results from processor runs
    - Manages image metadata
    """

    @beartype
    def __init__(
        self,
        processor: AgenticProcessorBase,
    ) -> None:
        """Initialize adapter.

        Args:
            processor: Radiant Harness processor
        """
        self.processor = processor

    @beartype
    async def process_verifiers_messages(
        self,
        messages: Messages,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Process messages using Radiant Harness.

        Args:
            messages: verifiers format messages
            info: Additional information (may include 'image_path')

        Returns:
            Processed result with response and metadata
        """
        user_prompt = self._extract_user_prompt(messages)
        metadata = dict(info)
        if user_prompt:
            metadata.setdefault("user_prompt", user_prompt)

        # Extract image path from info if provided
        image_path = info.get("image_path")
        images: Path | None = None
        if image_path:
            images = Path(image_path) if isinstance(image_path, str) else image_path

        agentic_result = await self.processor.analyze(images=images, metadata=metadata)
        tool_calls = self._collect_tool_calls(agentic_result)
        tool_results = self._collect_tool_results(agentic_result)
        response_payload = deep_thaw(agentic_result.final_response)
        response_text = json.dumps(response_payload)
        should_continue = bool(agentic_result.final_response.get("continue"))

        return {
            "response": response_payload,
            "messages": self._convert_response_to_messages(response_text, tool_calls, tool_results),
            "tool_calls": tool_calls,
            "turns": agentic_result.num_turns,
            "is_complete": not should_continue,
        }

    @beartype
    def _collect_tool_calls(self, result: AgenticResult) -> list[dict[str, Any]]:
        """Flatten tool calls from agentic turns."""
        return [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
                if isinstance(tool_call.arguments, str)
                else deep_thaw(tool_call.arguments),
            }
            for turn in result.turns
            for tool_call in turn.tool_calls
        ]

    @beartype
    def _collect_tool_results(self, result: AgenticResult) -> list[dict[str, Any]]:
        """Convert tool results into serializable dictionaries."""
        return [
            {
                "tool_name": tool_result.tool_name,
                "description": tool_result.description,
                "error": tool_result.error,
                "metadata": deep_thaw(tool_result.metadata),
            }
            for turn in result.turns
            for tool_result in turn.tool_results
        ]

    @beartype
    def _convert_response_to_messages(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> Messages:
        """Convert a harness response to verifiers messages format."""
        messages: Messages = []

        if response_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

        if tool_results:
            for idx, tool_result in enumerate(tool_results):
                # Use actual tool call ID when available, fall back to index
                tool_call_id = tool_calls[idx]["id"] if idx < len(tool_calls) else str(idx)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result),
                        "tool_call_id": tool_call_id,
                    }
                )

        return messages

    @beartype
    def _extract_user_prompt(self, messages: Messages) -> str:
        """Get the most recent user text from verifiers messages."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                combined = "\n".join(part for part in text_parts if part)
                if combined:
                    return combined
        return ""

    @beartype
    def create_environment_class(
        self,
        base_class: type[vf.MultiTurnEnv] | None = None,
        **env_kwargs: Any,
    ) -> type[vf.MultiTurnEnv]:
        """Create a verifiers MultiTurnEnv class that uses this adapter.

        Args:
            base_class: Base environment class to inherit from
            **env_kwargs: Additional arguments for environment

        Returns:
            Environment class
        """
        _ = env_kwargs
        if base_class is None:
            from .base import BaseMultiTurnEnv

            base_class = BaseMultiTurnEnv

        # Capture adapter config in closure
        captured_processor = self.processor

        class AdapterEnv(base_class):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                self._adapter = RadiantHarnessAdapter(
                    processor=captured_processor,
                )

            async def env_response(
                self,
                messages: Messages,
                state: State,
                info: dict[str, Any] | None = None,
            ) -> tuple[vf.Messages, vf.State]:
                """Generate response using Radiant Harness."""
                result = await self._adapter.process_verifiers_messages(
                    messages, info if info is not None else {}
                )

                new_state = dict(state)
                new_state["turn"] = state.get("turn", 0) + 1
                new_state["tool_uses"] = state.get("tool_uses", 0) + len(result["tool_calls"])
                new_state["is_complete"] = result["is_complete"]

                return result["messages"], new_state

            async def is_completed(
                self,
                messages: Messages,
                state: State,
                info: dict[str, Any] | None = None,
            ) -> bool:
                """Check if complete using adapter result."""
                if await super().is_completed(messages, state, info):
                    return True
                return state.get("is_complete", False)

        return AdapterEnv
