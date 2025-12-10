"""VerifiableProcessor mixin for tight verifiers integration.

Provides seamless integration between AgenticProcessorBase subclasses
and the verifiers package for RL training and evaluation.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype

if TYPE_CHECKING:
    import verifiers as vf

    from radiant_harness.verifiers.rewards import BaseRewardFunction


class VerifiableProcessorMixin:
    """Mixin that adds verifiers integration to AgenticProcessorBase subclasses.

    Provides methods for:
    - Creating verifiers environments from processors
    - Defining task-specific reward functions
    - Converting between harness and verifiers formats

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
    ) -> type[vf.MultiTurnEnv]:
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

        from radiant_harness.verifiers.adapter import RadiantHarnessAdapter
        from radiant_harness.verifiers.base import BaseMultiTurnEnv

        processor_cls = cls  # Capture for closure

        class _VerifiableEnv(BaseMultiTurnEnv):
            """Dynamically generated verifiers environment."""

            def __init__(
                self,
                env_cases: list[dict[str, Any]] | None = None,
                env_dataset_path: str | None = None,
                **env_kwargs: Any,
            ) -> None:
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

                # Create processor instance
                self._processor = processor_cls(**processor_kwargs)
                self._adapter = RadiantHarnessAdapter(
                    processor=self._processor,
                    registry=None,  # Registry created per-episode
                )
                self._image_base_path = image_base_path

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
                    # Resolve relative paths
                    if self._image_base_path and not Path(image_path).is_absolute():
                        image_path = str(self._image_base_path / image_path)

                    # Build multimodal message
                    text_content = self._processor.get_user_message(
                        images=[],  # Images handled separately
                        metadata=case,
                    )

                    return [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_path}"},
                        },
                    ]

                # Text-only case
                return self._processor.get_user_message(images=[], metadata=case)

            def build_initial_state(
                self,
                prompt: vf.Messages,
                info: dict[str, Any],
            ) -> vf.State:
                """Build initial state with image info."""
                state = super().build_initial_state(prompt, info)

                # Store image path for tool execution
                image_path = info.get("image_path") or info.get("image")
                if image_path:
                    if self._image_base_path and not Path(image_path).is_absolute():
                        image_path = str(self._image_base_path / image_path)
                    state["image_path"] = image_path

                return state

            async def env_response(
                self,
                messages: vf.Messages,
                state: vf.State,
                info: dict[str, Any] | None = None,
            ) -> tuple[vf.Messages, vf.State]:
                """Generate environment response using processor."""
                info = info or {}

                # Get image path from state if available
                image_path = state.get("image_path")

                # Process using adapter with image context
                result = await self._adapter.process_verifiers_messages(
                    messages=messages,
                    info={**info, "image_path": image_path},
                )

                # Update state
                new_state = dict(state)
                new_state["turn"] = state.get("turn", 0) + 1
                new_state["tool_uses"] = state.get("tool_uses", 0) + len(result["tool_calls"])
                new_state["is_complete"] = result["is_complete"]

                return result["messages"], new_state

            async def is_completed(
                self,
                messages: vf.Messages,
                state: vf.State,
                info: dict[str, Any] | None = None,
            ) -> bool:
                """Check completion using both base and processor logic."""
                if await super().is_completed(messages, state, info):
                    return True
                return state.get("is_complete", False)

            def get_reward_function(self) -> BaseRewardFunction:
                """Get reward function from processor."""
                return self._processor.get_reward_function()

        return _VerifiableEnv


@beartype
def create_verifiable_processor(
    base_processor_cls: type,
    reward_fn: BaseRewardFunction,
) -> type:
    """Create a VerifiableProcessor from a base processor class and reward function.

    Utility for adding verifiers support to existing processors without
    modifying the original class.

    Args:
        base_processor_cls: AgenticProcessorBase subclass
        reward_fn: Reward function to use

    Returns:
        New class with VerifiableProcessorMixin

    Example:
        VerifiableNOVA = create_verifiable_processor(
            NOVAProcessor,
            CombinedReward([ExactMatchReward(), IoUReward()]),
        )
    """

    class _VerifiableProcessor(VerifiableProcessorMixin, base_processor_cls):
        """Dynamically generated verifiable processor."""

        _reward_fn = reward_fn

        def get_reward_function(self) -> BaseRewardFunction:
            return self._reward_fn

    _VerifiableProcessor.__name__ = f"Verifiable{base_processor_cls.__name__}"
    _VerifiableProcessor.__qualname__ = _VerifiableProcessor.__name__

    return _VerifiableProcessor
