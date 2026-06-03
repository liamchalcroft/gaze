"""VerifiableProcessor mixin for tight verifiers integration.

Provides seamless integration between AgenticProcessorBase subclasses
and the verifiers package for RL training and evaluation.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

import verifiers as vf
from beartype import beartype
from PIL import Image

from gaze.tools import encode_image
from gaze.verifiers.rewards import BaseRewardFunction


def _safe_resolve_image_path(base: Path, relative: str) -> str:
    """Resolve a relative image path, ensuring it stays within *base*.

    Prevents path traversal attacks where a malicious ``image_path`` like
    ``../../etc/passwd`` could escape the intended image directory.

    Args:
        base: The trusted base directory for images.
        relative: The untrusted relative path from user/dataset input.

    Returns:
        The resolved absolute path as a string.

    Raises:
        ValueError: If the resolved path escapes *base*.
    """
    resolved = (base / relative).resolve()
    base_resolved = base.resolve()
    # The resolved path must either *be* the base directory or start with
    # it followed by a separator (to avoid prefix confusion like
    # /data/images vs /data/images_evil).
    # Use os.sep for cross-platform compatibility.
    if resolved != base_resolved and not str(resolved).startswith(str(base_resolved) + os.sep):
        raise ValueError(
            f"Image path traversal blocked: '{relative}' resolves outside base directory"
        )
    return str(resolved)


def _image_file_to_data_url(image_path: str) -> str:
    """Encode a local image file as a data URL for OpenAI-compatible APIs."""
    with Image.open(image_path) as image:
        return encode_image(image).to_data_url()


class VerifiableProcessorMixin:
    """Mixin that adds verifiers integration to AgenticProcessorBase subclasses.

    Provides methods for:
    - Creating verifiers environments from processors
    - Defining task-specific reward functions
    - Converting between GAZE and verifiers formats

    Usage:
        class MyProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
            def get_reward_function(self) -> BaseRewardFunction:
                return ExactMatchReward()

            # ... implement other abstract methods ...

        # Create verifiers environment
        env_class = MyProcessor.as_verifiers_env()
    """

    @abstractmethod
    def get_reward_function(self) -> BaseRewardFunction:
        """Return the reward function for this task.

        Must be implemented by subclasses to provide task-specific rewards.

        Returns:
            Reward function instance compatible with verifiers
        """
        ...

    @classmethod
    @beartype
    def as_verifiers_env(
        cls,
        *,
        max_turns: int = 10,
        cases: list[dict[str, Any]] | None = None,
        dataset_path: str | None = None,
        image_base_path: Path | None = None,
        **processor_kwargs: Any,
    ) -> type:
        """Create a verifiers MultiTurnEnv class from this processor.

        The returned environment class can be used directly with verifiers
        for training or evaluation.

        Args:
            max_turns: Maximum conversation turns per episode
            cases: Pre-loaded cases (optional, alternative to dataset_path)
            dataset_path: Path to JSONL dataset file
            image_base_path: Base path for resolving relative image paths
            **processor_kwargs: Arguments passed to processor __init__

        Returns:
            MultiTurnEnv subclass configured with this processor

        Example:
            EnvClass = NOVAProcessor.as_verifiers_env(
                max_turns=10,
                dataset_path="data/train.jsonl",
                model_name="openai/gpt-4o",
            )
            env = EnvClass()
        """

        from gaze.verifiers.adapter import GazeAdapter
        from gaze.verifiers.base import BaseMultiTurnEnv

        processor_cls = cls  # Capture for closure

        class _VerifiableEnv(BaseMultiTurnEnv):
            """Dynamically generated verifiers environment."""

            def __init__(
                self,
                env_cases: list[dict[str, Any]] | None = None,
                env_dataset_path: str | None = None,
                **env_kwargs: Any,
            ) -> None:
                # Create processor and adapter *before* super().__init__
                # because BaseMultiTurnEnv.__init__ calls _prepare_cases →
                # _build_prompt → get_system_prompt which needs _processor.
                self._processor = processor_cls(**processor_kwargs)
                self._adapter = GazeAdapter(
                    processor=self._processor,
                )
                self._image_base_path = image_base_path

                # Use provided cases/path or fall back to class-level defaults
                actual_cases = env_cases or cases
                actual_path = env_dataset_path or dataset_path

                super().__init__(
                    cases=actual_cases,
                    dataset_path=actual_path,
                    max_turns=max_turns,
                    name=f"{processor_cls.__name__}Env",
                    **env_kwargs,
                )

            def get_system_prompt(self) -> str:
                """Get system prompt from processor."""
                # Get a minimal system prompt for verifiers
                # Full prompt is built during processing
                return self._processor.get_system_prompt(images=[], metadata={})

            def _build_user_message(
                self,
                case: dict[str, Any],
            ) -> str | list[dict[str, Any]]:
                """Build user message with image support."""
                # Extract image path from case
                image_path = case.get("image_path") or case.get("image")

                if image_path:
                    # Resolve relative paths safely (prevent traversal)
                    if self._image_base_path and not Path(image_path).is_absolute():
                        image_path = _safe_resolve_image_path(self._image_base_path, image_path)

                    # Build multimodal message
                    text_content = self._processor.get_user_message(
                        images=[],  # Images handled separately
                        metadata=case,
                    )

                    return [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_file_to_data_url(image_path)},
                        },
                    ]

                # Text-only case
                return self._processor.get_user_message(images=[], metadata=case)

            async def setup_state(self, state: vf.State) -> vf.State:
                """Initialize state with image info."""
                state = await super().setup_state(state)

                # Store image path for tool execution
                info = state.get("info") or {}
                image_path = info.get("image_path") or info.get("image")
                if image_path:
                    if self._image_base_path and not Path(image_path).is_absolute():
                        image_path = _safe_resolve_image_path(self._image_base_path, image_path)
                    state["image_path"] = image_path

                return state

            async def env_response(
                self,
                messages: vf.Messages,
                state: vf.State,
                **kwargs: Any,  # noqa: ARG002 - Required by vf.MultiTurnEnv interface
            ) -> vf.Messages:
                """Generate environment response using processor."""
                info = state.get("info") or {}

                # Get image path from state if available
                image_path = state.get("image_path")

                # Process using adapter with image context
                result = await self._adapter.process_verifiers_messages(
                    messages=messages,
                    info={**info, "image_path": image_path},
                )

                # Update state in-place
                state["turn"] = state.get("turn", 0) + 1
                state["tool_uses"] = state.get("tool_uses", 0) + len(result["tool_calls"])
                state["is_complete"] = result["is_complete"]

                return result["messages"]

            @vf.stop
            async def _processor_complete(self, state: vf.State) -> bool:
                """Stop when processor signals completion."""
                return state.get("is_complete", False)

            def get_reward_function(self) -> BaseRewardFunction:
                """Get reward function from processor."""
                return self._processor.get_reward_function()

        return _VerifiableEnv
