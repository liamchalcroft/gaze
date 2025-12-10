"""Base agentic processor for radiology VLM analysis.

Provides multi-turn agentic loop with tool calling support.
Task-specific details are provided via dependency injection.
"""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from PIL import Image

from radiant_harness.config import AgenticConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import OpenAIAdapter
from radiant_harness.tools import EncodedImage
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_search_tools
from radiant_harness.tools import create_visual_tools
from radiant_harness.tools import encode_image
from radiant_harness.types import AgenticResult
from radiant_harness.types import ToolCall
from radiant_harness.types import ToolResult
from radiant_harness.types import Turn


@dataclass
class ImageInput:
    """Represents a single image input with optional label.

    Attributes:
        path: Path to the image file
        label: Optional label for the image (e.g., "T1-weighted", "Pre-contrast")
        width: Image width in pixels (set after loading)
        height: Image height in pixels (set after loading)
        encoded: Base64-encoded image (set after loading)
    """

    path: Path
    label: str | None = None
    width: int = 0
    height: int = 0
    encoded: EncodedImage | None = None

    @beartype
    def load(self) -> None:
        """Load image and populate dimensions and encoding."""
        with Image.open(self.path) as img:
            self.width = img.width
            self.height = img.height
            self.encoded = encode_image(img)


class AgenticProcessorBase(ABC):
    """Abstract base class for multi-turn agentic analysis.

    Provides core agentic loop with tool calling support. Subclasses
    must implement methods to provide task-specific prompts and schemas.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_tools: bool = True,
        use_web_search: bool = False,
        max_turns: int | None = None,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
        disabled_tools: list[str] | None = None,
        adapter_factory: Callable[[], AdapterProtocol] | None = None,
        config: AgenticConfig | None = None,
    ) -> None:
        """Initialize agentic processor.

        Args:
            model_name: Model name for analysis
            use_tools: Enable visual tools
            use_web_search: Enable web/image search tools
            max_turns: Maximum turns before forced completion (uses config default if None)
            reasoning_enabled: Enable model's internal reasoning
            reasoning_effort: Reasoning effort level ("high", "medium", "low")
            enable_caching: Enable prompt caching
            disabled_tools: Specific tool names to disable
            adapter_factory: Optional factory for custom model adapter
            config: Agentic configuration. If None, uses global default.

        Raises:
            ValueError: If max_turns < 1
        """
        self._config = config or get_config().agentic

        if max_turns is None:
            max_turns = self._config.default_max_turns
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self.model_name = model_name
        self.use_tools = use_tools
        self.use_web_search = use_web_search

        if max_turns > self._config.max_turns_limit:
            logger.warning(
                f"max_turns={max_turns} exceeds max_turns_limit={self._config.max_turns_limit}, clamping"
            )
        self.max_turns = min(max_turns, self._config.max_turns_limit)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching
        self._adapter_factory = adapter_factory

        self._disabled_tools: set[str] = set(disabled_tools or [])
        if not use_web_search:
            self._disabled_tools.update(["search_web", "search_images"])

        self._model_adapter: AdapterProtocol | None = None

    @beartype
    def _ensure_initialized(self) -> None:
        """Ensure model adapter is initialized."""
        if self._model_adapter is None:
            if self._adapter_factory:
                self._model_adapter = self._adapter_factory()
            else:
                self._model_adapter = OpenAIAdapter(
                    model_name=self.model_name,
                    reasoning_enabled=self.reasoning_enabled,
                    reasoning_effort=self.reasoning_effort,
                    enable_caching=self.enable_caching,
                )

    @beartype
    def _create_tool_registry(
        self,
        images: list[ImageInput],
        active_image_index: int = 0,
    ) -> ToolRegistry | None:
        """Create a tool registry with appropriate tools.

        Args:
            images: List of loaded image inputs (may be empty for text-only)
            active_image_index: Index of the initially active image for tools

        Returns:
            ToolRegistry if images are present and tools enabled, None otherwise

        Subclasses can override to customize available tools.
        """
        if not images:
            if self.use_web_search:
                tools = create_search_tools(self._disabled_tools)
                return ToolRegistry(image_path=None, tools=tools)
            return None

        tools: list[Tool] = []

        if self.use_tools:
            tools.extend(create_visual_tools(self._disabled_tools))

        if self.use_web_search:
            tools.extend(create_search_tools(self._disabled_tools))

        if not tools:
            return None

        active_image = images[active_image_index]
        return ToolRegistry(image_path=active_image.path, tools=tools)

    @abstractmethod
    def get_system_prompt(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build the system prompt for this task.

        Args:
            images: List of image inputs (may be empty for text-only tasks)
            metadata: Task and context metadata

        Returns:
            System prompt string
        """
        ...

    @abstractmethod
    def get_user_message(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build the initial user message for this task.

        Args:
            images: List of image inputs (may be empty for text-only tasks)
            metadata: Task and context metadata

        Returns:
            User message string
        """
        ...

    @abstractmethod
    def get_response_schema(self) -> dict[str, Any] | None:
        """Get the JSON schema for structured outputs.

        Returns:
            OpenAI-compatible JSON schema dict, or None for free-form responses
        """
        ...

    @abstractmethod
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate that a response has required fields.

        Args:
            response: Parsed JSON response from model

        Returns:
            True if valid, False otherwise
        """
        ...

    @beartype
    def calculate_confidence(
        self,
        response: dict[str, Any],  # noqa: ARG002 - Available for subclass override
        turns: list[Turn],
    ) -> float:
        """Calculate confidence score for the result.

        Default implementation provides a base score. Subclasses can
        override for task-specific confidence calculation.

        Args:
            response: Final parsed response
            turns: All conversation turns

        Returns:
            Confidence score in range [0.0, 1.0]
        """
        confidence = 0.5

        tool_turns = sum(1 for t in turns if t.tool_calls)
        tool_bonus = min(tool_turns * 0.1, 0.2)
        confidence += tool_bonus

        return min(1.0, confidence)

    @beartype
    async def analyze(
        self,
        images: list[Path] | Path | None = None,
        metadata: dict[str, Any] | None = None,
        image_labels: list[str] | None = None,
    ) -> AgenticResult:
        """Run agentic analysis on images and/or text.

        Args:
            images: Image path(s) - None for text-only, Path for single, list for multi
            metadata: Context metadata (clinical history, patient info, etc.)
            image_labels: Optional labels for each image (e.g., ["T1", "T2-FLAIR"])

        Returns:
            AgenticResult with final response and conversation history
        """
        self._ensure_initialized()

        metadata = metadata or {}

        image_inputs = self._normalize_image_inputs(images, image_labels)

        for img in image_inputs:
            img.load()

        tool_registry = self._create_tool_registry(image_inputs)

        try:
            return await self._run_analysis(
                images=image_inputs,
                metadata=metadata,
                tool_registry=tool_registry,
            )
        finally:
            if tool_registry is not None:
                await tool_registry.aclose()

    @beartype
    def _normalize_image_inputs(
        self,
        images: list[Path] | Path | None,
        labels: list[str] | None,
    ) -> list[ImageInput]:
        """Normalize image input formats to list of ImageInput."""
        if images is None:
            return []

        # Validate inputs
        if isinstance(images, list) and labels and len(images) != len(labels):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of images ({len(images)})"
            )

        if isinstance(images, Path):
            # Validate single image exists
            if not images.exists():
                raise FileNotFoundError(f"Image file not found: {images}")
            # Validate file extension
            if images.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                raise ValueError(f"Unsupported image format: {images.suffix}")
            label = labels[0] if labels else None
            return [ImageInput(path=images, label=label)]

        # List of paths
        result: list[ImageInput] = []
        for i, path in enumerate(images):
            # Validate each image exists
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            # Validate file extension
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                raise ValueError(f"Unsupported image format: {path.suffix}")

            label = labels[i] if labels and i < len(labels) else None
            result.append(ImageInput(path=path, label=label))
        return result

    @beartype
    async def _run_analysis(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
        tool_registry: ToolRegistry | None,
    ) -> AgenticResult:
        """Run the analysis loop."""
        system_prompt = self.get_system_prompt(images=images, metadata=metadata)
        policy_lines = [
            f"Multi-turn session with a maximum of {self.max_turns} turns.",
            'Return JSON every turn with a boolean field "continue".',
            'Set "continue": true when you need more tools/analysis; false when final.',
            "Final response must satisfy the provided response schema.",
        ]
        system_prompt = f"{system_prompt}\n\nPOLICY:\n- " + "\n- ".join(policy_lines)

        if tool_registry:
            tool_docs = tool_registry.get_documenter().generate_prompt_documentation()
            if tool_docs:
                system_prompt = f"{system_prompt}\n\nAvailable tools:\n{tool_docs}"
        user_message = self.get_user_message(images=images, metadata=metadata)

        turns: list[Turn] = []
        final_response: dict[str, Any] = {}

        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_message}]
        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": img.encoded.to_data_url()},
            }
            for img in images
            if img.encoded is not None
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        model_adapter = self._model_adapter
        if model_adapter is None:
            raise RuntimeError("Model adapter not initialized after _ensure_initialized()")

        tool_schemas = tool_registry.get_tool_schemas() if tool_registry else None
        response_schema = self.get_response_schema()

        total_tokens: int = 0
        for turn_idx in range(self.max_turns):
            is_last_turn = turn_idx == self.max_turns - 1
            current_tools = None if is_last_turn else tool_schemas
            current_schema = response_schema
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            chat_result = await model_adapter.generate_chat(
                messages=messages,
                max_tokens=self._config.default_max_tokens,
                temperature=self._config.default_temperature,
                tools=current_tools,
                response_format=current_schema,
            )
            if isinstance(chat_result, AsyncIterator):
                raise AgenticProcessingError(
                    "Streaming responses are not supported in _run_analysis()",
                    turns_completed=turn_idx,
                )
            response_text, tool_calls, gen_log = chat_result

            total_tokens += gen_log.tokens

            typed_tool_calls: list[ToolCall] = []
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    missing_fields = [f for f in ("id", "name", "arguments") if f not in tc]
                    if missing_fields:
                        raise AgenticProcessingError(
                            f"Tool call {i} missing required fields: {missing_fields}. Got: {tc}",
                            turns_completed=turn_idx + 1,
                        )
                    typed_tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc["arguments"],
                        )
                    )

            if typed_tool_calls and tool_registry is None:
                raise AgenticProcessingError(
                    "Model requested tool calls but tools are disabled or unavailable",
                    turns_completed=turn_idx + 1,
                    partial_response={
                        "error": "tools_unavailable",
                        "tools": [tc.name for tc in typed_tool_calls],
                    },
                )

            turn = Turn(
                role="assistant",
                content=response_text,
                tool_calls=typed_tool_calls,
            )
            turns.append(turn)

            assistant_message: dict[str, Any] = {"role": "assistant"}
            if response_text:
                assistant_message["content"] = response_text
            if typed_tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in typed_tool_calls
                ]
            messages.append(assistant_message)

            if typed_tool_calls and tool_registry is not None:
                logger.info(f"Executing {len(typed_tool_calls)} tool calls")
                tool_results = await self._execute_tools(
                    tool_calls=typed_tool_calls,
                    tool_registry=tool_registry,
                    turn_idx=turn_idx,
                )

                for i, result in enumerate(tool_results):
                    tool_call_id = typed_tool_calls[i].id
                    image_data_url = result.get_image_data_url()

                    if image_data_url:
                        tool_content: str | list[dict[str, Any]] = [
                            {"type": "text", "text": result.description},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]
                    else:
                        tool_content: str | list[dict[str, Any]] = result.description
                        if result.error:
                            tool_content = f"{tool_content}\nError: {result.error}"
                        if formatted := result.formatted_results:
                            tool_content = f"{tool_content}\n{formatted}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_content,
                        }
                    )

                tool_turn = Turn(
                    role="tool_result",
                    content=self._format_tool_results(tool_results),
                    tool_results=tool_results,
                )
                turns.append(tool_turn)
                continue
            try:
                parsed_obj: dict[str, Any] | list | str | int | float | bool | None = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise AgenticProcessingError(
                    f"Invalid JSON on turn {turn_idx + 1}: {e}. Response: {response_text[:200]}",
                    turns_completed=turn_idx + 1,
                ) from e

            if not isinstance(parsed_obj, dict):
                raise AgenticProcessingError(
                    "Model response must be a JSON object",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "invalid_response_type"},
                )

            parsed: dict[str, Any] = parsed_obj

            wants_continue = parsed.get("continue", False)
            if not isinstance(wants_continue, bool):
                raise AgenticProcessingError(
                    "Response field 'continue' must be boolean",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "invalid_continue_flag"},
                )

            if not wants_continue:
                # Model is done
                final_response = parsed
                break

            if is_last_turn:
                # Force completion on last turn
                parsed["continue"] = False
                final_response = parsed
                logger.info("Max turns reached, forcing completion")
                break

            # Model wants to continue - add warning on penultimate turn
            if turn_idx == self.max_turns - 2:
                turn_warning = (
                    f"[System: Next turn ({turn_idx + 2}/{self.max_turns}) is your "
                    f"final turn. You must provide complete analysis with 'continue': false]"
                )
                messages.append({"role": "user", "content": turn_warning})

        if not self.validate_response(final_response):
            raise AgenticProcessingError(
                "Final response failed schema validation",
                turns_completed=len(turns),
                partial_response=final_response,
            )

        confidence = self.calculate_confidence(final_response, turns)

        return AgenticResult(
            final_response=final_response,
            turns=turns,
            total_tokens=total_tokens,
            confidence=confidence,
        )

    @beartype
    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        turn_idx: int,
    ) -> list[ToolResult]:
        """Execute a list of tool calls."""
        results: list[ToolResult] = []

        for tool_call in tool_calls:
            # Parse arguments: either already a dict or a JSON string
            if isinstance(tool_call.arguments, str):
                try:
                    parsed_args: dict[str, Any] | list | str | int | float | bool | None = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    raise AgenticProcessingError(
                        f"Malformed JSON in tool arguments for '{tool_call.name}': {e}",
                        turns_completed=turn_idx + 1,
                        partial_response={"error": "malformed_tool_args", "tool": tool_call.name},
                    ) from e

                if not isinstance(parsed_args, dict):
                    raise AgenticProcessingError(
                        f"Tool arguments for '{tool_call.name}' must be a JSON object, got {type(parsed_args).__name__}",
                        turns_completed=turn_idx + 1,
                        partial_response={"error": "invalid_tool_args", "tool": tool_call.name},
                    )
                tool_args: dict[str, Any] = parsed_args
            else:
                if not isinstance(tool_call.arguments, dict):
                    raise AgenticProcessingError(
                        f"Tool arguments for '{tool_call.name}' must be a dict, got {type(tool_call.arguments).__name__}",
                        turns_completed=turn_idx + 1,
                        partial_response={"error": "invalid_tool_args", "tool": tool_call.name},
                    )
                tool_args = dict(tool_call.arguments)

            logger.debug(f"Executing: {tool_call.name}({tool_args})")

            try:
                result = await tool_registry.execute(tool_call.name, **tool_args)
            except UnknownToolError as e:
                logger.error(f"Unknown tool requested on turn {turn_idx + 1}: {tool_call.name}")
                raise AgenticProcessingError(
                    f"Tool '{tool_call.name}' is not registered",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "unknown_tool", "tool": tool_call.name},
                ) from e
            except ToolExecutionError as e:
                logger.error(
                    f"Tool '{tool_call.name}' failed during execution on turn {turn_idx + 1}: {e}"
                )
                raise AgenticProcessingError(
                    f"Tool '{tool_call.name}' failed during execution: {e}",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "tool_execution_failed", "tool": tool_call.name},
                ) from e

            results.append(result)

        return results

    @beartype
    def _format_tool_results(self, tool_results: list[ToolResult]) -> str:
        """Format tool results for logging."""
        if not tool_results:
            return "No tools were executed."

        result_strings: list[str] = []
        for result in tool_results:
            if result.success:
                result_str = f"[OK] {result.tool_name}: {result.description}"
            else:
                result_str = f"[FAIL] {result.tool_name}: {result.error}"
            result_strings.append(result_str)

        return "\n".join(result_strings)
