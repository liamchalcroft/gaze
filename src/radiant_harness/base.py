"""Base agentic processor for radiology VLM analysis.

Provides multi-turn agentic loop with tool calling support.
Task-specific details are provided via dependency injection.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype
from beartype.roar import BeartypeException
from loguru import logger
from PIL import Image

from radiant_harness.config import AgenticConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.models import AdapterProtocol
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
from radiant_harness.utils.json_extract import extract_json_from_text

# Maximum characters allowed in a single tool result message.
# Limits prompt-injection surface from external data (PubMed abstracts, etc.).
_MAX_TOOL_CONTENT_CHARS = 8_000

# Regex for ASCII/Unicode control characters (except newline/tab).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_tool_content(text: str, *, max_chars: int = _MAX_TOOL_CONTENT_CHARS) -> str:
    """Sanitize tool result text before injecting into the LLM conversation.

    1. Strip control characters that could confuse tokenizers.
    2. Truncate to *max_chars* to limit prompt-injection surface.
    3. Wrap in an untrusted-content marker so the model can distinguish
       tool output from system/user instructions.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return f"[Tool Result - External Data]\n{text}\n[End Tool Result]"


@dataclass
class ImageInput:
    """Represents a single image input with optional label.

    Attributes:
        path: Path to the image file
        label: Optional label for the image (e.g., "T1-weighted", "Pre-contrast")
        width: Image width in pixels (set after loading)
        height: Image height in pixels (set after loading)
        encoded: Base64-encoded image (set after loading)
        pil_image: Loaded PIL Image kept in memory to avoid re-reading from disk
    """

    path: Path
    label: str | None = None
    width: int = 0
    height: int = 0
    encoded: EncodedImage | None = None
    pil_image: Image.Image | None = None

    @staticmethod
    @beartype
    def from_pil(
        image: Image.Image,
        *,
        label: str | None = None,
        path: Path | None = None,
    ) -> ImageInput:
        """Create an ImageInput directly from a PIL Image, skipping disk I/O.

        Args:
            image: PIL Image with pixel data already in memory.
            label: Optional label for the image.
            path: Optional source path (for logging only). Defaults to
                  a synthetic ``<in-memory>`` path.
        """
        max_dim = get_config().image.max_image_dimension
        if image.width > max_dim or image.height > max_dim:
            raise ValueError(
                f"Image dimensions {image.width}x{image.height} exceed "
                f"maximum allowed dimension of {max_dim}px"
            )
        inp = ImageInput(
            path=path or Path("<in-memory>"),
            label=label,
            width=image.width,
            height=image.height,
            encoded=encode_image(image),
            pil_image=image,
        )
        return inp

    @beartype
    def load(self) -> None:
        """Load image and populate dimensions, encoding, and PIL reference.

        No-op if the image was already loaded (e.g. via ``from_pil``).
        """
        if self.pil_image is not None:
            return

        img = Image.open(self.path)
        img.load()  # Force full pixel decode into memory
        max_dim = get_config().image.max_image_dimension
        if img.width > max_dim or img.height > max_dim:
            img.close()
            raise ValueError(
                f"Image dimensions {img.width}x{img.height} exceed "
                f"maximum allowed dimension of {max_dim}px"
            )
        self.width = img.width
        self.height = img.height
        self.encoded = encode_image(img)
        self.pil_image = img

    async def aload(self) -> None:
        """Async version of :meth:`load` — offloads blocking I/O to a thread.

        Use this instead of ``load()`` when calling from an async context
        to avoid blocking the event loop during image decoding and encoding.
        """
        if self.pil_image is not None:
            return
        await asyncio.to_thread(self.load)


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

        allowed_reasoning_efforts = {"low", "medium", "high"}
        if reasoning_effort not in allowed_reasoning_efforts:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(allowed_reasoning_efforts)}, got {reasoning_effort}"
            )

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
                # Import here to avoid coupling the abstract base to a concrete adapter
                from radiant_harness.models import OpenAIAdapter

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
    ) -> ToolRegistry | None:
        """Create a tool registry with appropriate tools.

        Args:
            images: List of loaded image inputs (may be empty for text-only)

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

        if len(images) > 1:
            logger.warning(
                f"Tool registry only supports a single active image; "
                f"using first of {len(images)} images, rest will be ignored"
            )
        active_image = images[0]

        # If the PIL Image was kept from load(), hand it directly to the
        # ImageManager to avoid re-reading the file from disk.
        if active_image.pil_image is not None:
            registry = ToolRegistry(tools=tools)
            registry.get_image_manager().set_preloaded_image(
                active_image.pil_image, active_image.path
            )
            return registry

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
        images: list[Path] | list[Image.Image] | Path | Image.Image | None = None,
        metadata: dict[str, Any] | None = None,
        image_labels: list[str] | None = None,
    ) -> AgenticResult:
        """Run agentic analysis on images and/or text.

        Args:
            images: Image input(s). Accepts ``Path``, ``PIL.Image.Image``,
                    lists of either, or ``None`` for text-only analysis.
                    Passing PIL Images directly avoids a temp-file round-trip.
            metadata: Context metadata (clinical history, patient info, etc.)
            image_labels: Optional labels for each image (e.g., ["T1", "T2-FLAIR"])

        Returns:
            AgenticResult with final response and conversation history
        """
        self._ensure_initialized()

        metadata = metadata or {}

        image_inputs = self._normalize_image_inputs(images, image_labels)

        await asyncio.gather(*(img.aload() for img in image_inputs))

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
        images: list[Path] | list[Image.Image] | Path | Image.Image | None,
        labels: list[str] | None,
    ) -> list[ImageInput]:
        """Normalize image input formats to list of ImageInput."""
        if images is None:
            return []

        # --- Single PIL Image ---
        if isinstance(images, Image.Image):
            if labels is not None and len(labels) != 1:
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of images (1)"
                )
            label = labels[0] if labels else None
            return [ImageInput.from_pil(images, label=label)]

        # --- Single Path ---
        if isinstance(images, Path):
            if labels is not None and len(labels) != 1:
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of images (1)"
                )
            if not images.exists():
                raise FileNotFoundError(f"Image file not found: {images}")
            if images.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                raise ValueError(f"Unsupported image format: {images.suffix}")
            label = labels[0] if labels else None
            return [ImageInput(path=images, label=label)]

        # --- List inputs ---
        if labels is not None and len(images) != len(labels):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of images ({len(images)})"
            )

        result: list[ImageInput] = []
        for i, item in enumerate(images):
            label = labels[i] if labels and i < len(labels) else None
            if isinstance(item, Image.Image):
                result.append(ImageInput.from_pil(item, label=label))
            else:
                # Path
                if not item.exists():
                    raise FileNotFoundError(f"Image file not found: {item}")
                if item.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                    raise ValueError(f"Unsupported image format: {item.suffix}")
                result.append(ImageInput(path=item, label=label))
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

        if tool_registry and len(images) > 1:
            first_label = images[0].label or images[0].path.name
            system_prompt = (
                f"{system_prompt}\n\nIMPORTANT: Visual tools (zoom, crop, "
                f"contrast, etc.) operate only on the first image "
                f"({first_label}). All images are visible in the "
                f"conversation, but tool manipulations apply to the first "
                f"image only."
            )
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
        nudge_count: int = 0

        def _force_finalize_message() -> str:
            """Build a force-finalize message with the response JSON skeleton."""
            schema_obj = {}
            if response_schema is not None:
                schema_obj = response_schema.get("json_schema", {}).get("schema", {})
            props = schema_obj.get("properties", {})
            skeleton: dict[str, str] = {}
            for key, prop in props.items():
                ptype = prop.get("type", "string")
                if ptype == "boolean":
                    skeleton[key] = "true/false"
                elif ptype == "array":
                    skeleton[key] = "[...]"
                elif ptype == "object":
                    skeleton[key] = "{...}"
                elif ptype in ("number", "integer"):
                    skeleton[key] = "0"
                else:
                    skeleton[key] = "..."
            skeleton["continue"] = "false"
            skeleton_str = json.dumps(skeleton, indent=2)
            return (
                "[System: You have failed to produce valid JSON for multiple consecutive turns. "
                "You MUST respond with ONLY a JSON object NOW. Fill in this template with your "
                f"best analysis based on what you have observed so far:\n{skeleton_str}\n"
                "Replace placeholder values with your actual analysis. Respond with ONLY the "
                "JSON object, no other text.]"
            )

        for turn_idx in range(self.max_turns):
            is_last_turn = turn_idx == self.max_turns - 1
            current_tools = None if is_last_turn else tool_schemas
            # Only enforce response_format when tools are NOT being offered.
            # Many providers cannot handle tools + response_format together,
            # returning empty responses.  On the last turn (tools stripped)
            # we enforce the schema so the final answer is well-formed.
            current_format = response_schema if current_tools is None else None
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            # On the last turn, inject a schema reminder so models that don't
            # fully support response_format still know the expected output.
            if is_last_turn and turn_idx > 0:
                # Reset image to original so model sees untransformed view
                if tool_registry is not None:
                    try:
                        tool_registry.get_image_manager().reset_to_original()
                    except ToolExecutionError:
                        logger.warning("Failed to reset image on final turn")

                # Build final-turn message with original image + schema reminder
                final_parts: list[dict[str, Any]] = []

                if response_schema is not None:
                    schema_obj = response_schema.get("json_schema", {}).get("schema", {})
                    top_keys = list(schema_obj.get("properties", {}).keys())
                    if top_keys:
                        coord_note = ""
                        if images:
                            coord_note = (
                                " The ORIGINAL (untransformed) image is re-attached below. "
                                "All spatial coordinates (bounding boxes) MUST reference "
                                "this original image, NOT any zoomed or cropped version."
                            )
                        final_parts.append(
                            {
                                "type": "text",
                                "text": (
                                    f"[System: This is your FINAL turn. Tools are no longer available."
                                    f"{coord_note} "
                                    f"You MUST respond with a complete JSON object containing these "
                                    f"required top-level keys: {top_keys}. "
                                    f"Set 'continue': false. Do NOT attempt tool calls.]"
                                ),
                            }
                        )

                # Re-inject original images for coordinate reference
                final_parts.extend(
                    {"type": "image_url", "image_url": {"url": img.encoded.to_data_url()}}
                    for img in images
                    if img.encoded is not None
                )

                if final_parts:
                    messages.append({"role": "user", "content": final_parts})

            # Strip stale base64 images from earlier rounds (including the
            # initial user message) to reduce payload on subsequent API calls.
            if turn_idx > 0:
                self._strip_stale_images(messages)

            chat_result = await model_adapter.generate_chat(
                messages=messages,
                max_tokens=self._config.default_max_tokens,
                temperature=self._config.default_temperature,
                tools=current_tools,
                response_format=current_format,
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
                    # Sanitize tool name: some models return None (GLM-4.6V)
                    # or append special tokens like <|end_of_box|> (Qwen3-VL).
                    raw_name = tc["name"]
                    if raw_name is None:
                        logger.warning(
                            "Tool call {} on turn {} has name=None, skipping",
                            i,
                            turn_idx + 1,
                        )
                        continue
                    clean_name = re.sub(r"<\|[^|]*\|>", "", raw_name).strip()
                    # Strip trailing parentheses — some models (GLM-4.6V) include
                    # "()" or "(args)" in the tool name field.
                    clean_name = re.sub(r"\(.*\)\s*$", "", clean_name).strip()
                    if not clean_name:
                        logger.warning(
                            "Tool call {} on turn {} has empty name after sanitization (raw={!r}), skipping",
                            i,
                            turn_idx + 1,
                            raw_name,
                        )
                        continue
                    # Normalize missing/empty arguments to "{}" — some models send
                    # None or "" for tools that take no parameters.
                    raw_args = tc["arguments"]
                    if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
                        raw_args = "{}"
                    typed_tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            name=clean_name,
                            arguments=raw_args,
                        )
                    )

            if typed_tool_calls and (tool_registry is None or is_last_turn):
                reason = (
                    "tools were withheld on final turn"
                    if is_last_turn
                    else "tools are disabled or unavailable"
                )
                raise AgenticProcessingError(
                    f"Model requested tool calls but {reason}",
                    turns_completed=turn_idx + 1,
                    partial_response={
                        "error": "tools_unavailable",
                        "tools": [tc.name for tc in typed_tool_calls],
                    },
                )

            turn = Turn(
                role="assistant",
                content=response_text,
                tool_calls=tuple(typed_tool_calls),
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
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                            if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments),
                        },
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

                multipart_ok = model_adapter.supports_multipart_tool_content

                for i, result in enumerate(tool_results):
                    tool_call_id = typed_tool_calls[i].id
                    image_data_url = result.get_image_data_url()

                    tool_content: str | list[dict[str, Any]]
                    if image_data_url and multipart_ok:
                        tool_content = [
                            {
                                "type": "text",
                                "text": _sanitize_tool_content(result.description),
                            },
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]
                    else:
                        raw = result.description
                        if result.error:
                            raw = f"{raw}\nError: {result.error}"
                        if formatted := result.formatted_results:
                            raw = f"{raw}\n{formatted}"
                        tool_content = _sanitize_tool_content(raw)

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
                    tool_results=tuple(tool_results),
                )
                turns.append(tool_turn)
                nudge_count = 0  # Successful tool calls reset nudge counter
                continue
            # Detect truncated responses before attempting JSON parsing
            if gen_log.finish_reason == "length":
                if not is_last_turn:
                    nudge_count += 1
                    # Truncated on intermediate turn — nudge model to use tools
                    # or produce concise JSON on the next turn.
                    logger.warning(
                        f"Turn {turn_idx + 1} truncated (completion_tokens="
                        f"{gen_log.completion_tokens}). Nudge {nudge_count}/"
                        f"{self._config.max_consecutive_nudges}."
                    )
                    if nudge_count >= self._config.max_consecutive_nudges:
                        messages.append({"role": "user", "content": _force_finalize_message()})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Your previous response was too long and was "
                                    "truncated. Please use tools or provide a concise JSON "
                                    "response with 'continue' field.]"
                                ),
                            }
                        )
                    continue
                raise AgenticProcessingError(
                    f"Response truncated on turn {turn_idx + 1} "
                    f"(finish_reason='length', completion_tokens={gen_log.completion_tokens}, "
                    f"max_tokens={self._config.default_max_tokens}). "
                    f"Increase default_max_tokens or simplify the response schema.",
                    turns_completed=turn_idx + 1,
                )

            # Handle empty or non-JSON responses on intermediate turns:
            # nudge the model instead of crashing.
            if not response_text.strip() and not is_last_turn:
                nudge_count += 1
                logger.warning(
                    f"Turn {turn_idx + 1} returned empty response with no tool calls. "
                    f"Nudge {nudge_count}/{self._config.max_consecutive_nudges}."
                )
                if nudge_count >= self._config.max_consecutive_nudges:
                    messages.append({"role": "user", "content": _force_finalize_message()})
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: You returned an empty response. Please provide "
                                "your analysis as a JSON object. Use tools if you need "
                                "more information, or set 'continue': false to finalize.]"
                            ),
                        }
                    )
                continue

            try:
                parsed_obj: dict[str, Any] | list | str | int | float | bool | None = json.loads(
                    response_text
                )
            except json.JSONDecodeError:
                # Fallback: extract JSON from markdown code blocks or embedded text
                fallback = extract_json_from_text(response_text)
                if fallback is None:
                    if not is_last_turn:
                        nudge_count += 1
                        # On intermediate turns, nudge instead of crashing
                        logger.warning(
                            f"Turn {turn_idx + 1} returned non-JSON text. "
                            f"Nudge {nudge_count}/{self._config.max_consecutive_nudges}."
                        )
                        if nudge_count >= self._config.max_consecutive_nudges:
                            messages.append({"role": "user", "content": _force_finalize_message()})
                        else:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[System: Your response was not valid JSON. "
                                        "Please respond with a JSON object including "
                                        "a 'continue' field.]"
                                    ),
                                }
                            )
                        continue
                    raise AgenticProcessingError(
                        f"No valid JSON found on turn {turn_idx + 1}. "
                        f"Response: {response_text[:200]}",
                        turns_completed=turn_idx + 1,
                    ) from None
                parsed_obj = fallback

            if not isinstance(parsed_obj, dict):
                raise AgenticProcessingError(
                    "Model response must be a JSON object",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "invalid_response_type"},
                )

            parsed: dict[str, Any] = parsed_obj
            nudge_count = 0  # Valid JSON resets nudge counter

            wants_continue = parsed.get("continue", False)
            if not isinstance(wants_continue, bool):
                raise AgenticProcessingError(
                    "Response field 'continue' must be boolean",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "invalid_continue_flag"},
                )

            if not wants_continue:
                # Model says it's done — but if the response is incomplete
                # (fails validation) and we have turns left, nudge instead
                # of accepting a garbage final response.
                if not is_last_turn and not self.validate_response(parsed):
                    nudge_count += 1
                    logger.warning(
                        f"Turn {turn_idx + 1} returned incomplete response "
                        f"(keys: {list(parsed.keys())[:10]}). "
                        f"Nudge {nudge_count}/{self._config.max_consecutive_nudges}."
                    )
                    if nudge_count >= self._config.max_consecutive_nudges:
                        messages.append({"role": "user", "content": _force_finalize_message()})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Your response is incomplete — it's missing "
                                    "required fields. Please continue your analysis and "
                                    "provide a complete response with all required sections. "
                                    "Set 'continue': true if you need more turns.]"
                                ),
                            }
                        )
                    continue
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
                    f"final turn. You must provide complete analysis with 'continue': false. "
                    f"If you used zoom or crop, call reset() NOW to return to the "
                    f"original image before your final response.]"
                )
                messages.append({"role": "user", "content": turn_warning})
        else:
            raise AgenticProcessingError(
                f"Analysis loop exhausted all {self.max_turns} turns without "
                f"producing a final response "
                f"(model may have returned tool calls on every turn)",
                turns_completed=len(turns),
            )

        if not self.validate_response(final_response):
            # Log the actual keys for debugging
            top_keys = list(final_response.keys())[:10]
            raise AgenticProcessingError(
                f"Final response failed schema validation. Top-level keys: {top_keys}",
                turns_completed=len(turns),
                partial_response=final_response,
            )

        confidence = self.calculate_confidence(final_response, turns)

        return AgenticResult(
            final_response=final_response,
            turns=tuple(turns),
            total_tokens=total_tokens,
            confidence=confidence,
        )

    def _parse_tool_args(self, tool_call: ToolCall, turn_idx: int) -> dict[str, Any]:
        """Parse tool call arguments, raising on malformed JSON."""
        if isinstance(tool_call.arguments, str):
            try:
                parsed_args: dict[str, Any] | list | str | int | float | bool | None = json.loads(
                    tool_call.arguments
                )
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
            return parsed_args

        if not isinstance(tool_call.arguments, dict):
            raise AgenticProcessingError(
                f"Tool arguments for '{tool_call.name}' must be a dict, got {type(tool_call.arguments).__name__}",
                turns_completed=turn_idx + 1,
                partial_response={"error": "invalid_tool_args", "tool": tool_call.name},
            )
        return dict(tool_call.arguments)

    @beartype
    async def _run_single_tool(
        self,
        tool_call: ToolCall,
        tool_registry: ToolRegistry,
        turn_idx: int,
    ) -> ToolResult:
        """Execute a single tool call with error handling.

        Recoverable errors (tool execution failures, unexpected crashes)
        return a ``ToolResult`` with an error description so the model can
        adapt.  Only structural errors (unknown tool name) are fatal.
        """
        tool_args = self._parse_tool_args(tool_call, turn_idx)
        logger.debug(f"Executing: {tool_call.name}({tool_args})")

        try:
            return await tool_registry.execute(tool_call.name, **tool_args)
        except UnknownToolError as e:
            # Structural error: model hallucinated a tool name — fatal.
            logger.error(f"Unknown tool requested on turn {turn_idx + 1}: {tool_call.name}")
            raise AgenticProcessingError(
                f"Tool '{tool_call.name}' is not registered",
                turns_completed=turn_idx + 1,
                partial_response={"error": "unknown_tool", "tool": tool_call.name},
            ) from e
        except ToolExecutionError as e:
            logger.warning(f"Tool '{tool_call.name}' failed on turn {turn_idx + 1}: {e}")
            return ToolResult(
                tool_name=tool_call.name,
                description=f"Tool '{tool_call.name}' failed",
                error=str(e),
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OSError,
            RuntimeError,
            BeartypeException,
        ) as e:
            logger.warning(
                f"Tool '{tool_call.name}' crashed on turn {turn_idx + 1}: {e}",
            )
            return ToolResult(
                tool_name=tool_call.name,
                description=f"Tool '{tool_call.name}' encountered an error",
                error=str(e),
            )

    @beartype
    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        turn_idx: int,
    ) -> list[ToolResult]:
        """Execute a list of tool calls, parallelizing when safe.

        Image-mutating tools (requires_image=True) run sequentially to
        preserve shared ImageManager state.  Independent tools (search, etc.)
        run concurrently *alongside* the sequential image tools so that
        e.g. a PubMed search overlaps with zoom/crop operations.
        """
        import asyncio

        if len(tool_calls) <= 1:
            return [await self._run_single_tool(tc, tool_registry, turn_idx) for tc in tool_calls]

        def _is_image_tool(tc: ToolCall) -> bool:
            tool = tool_registry.get_documenter().get_tool(tc.name)
            return tool is not None and tool.requires_image

        # Partition into image-mutating (must be sequential) and independent tools
        image_indices = [i for i, tc in enumerate(tool_calls) if _is_image_tool(tc)]
        other_indices = [i for i, tc in enumerate(tool_calls) if not _is_image_tool(tc)]

        results: list[ToolResult | None] = [None] * len(tool_calls)

        async def _run_image_tools() -> None:
            """Run image-mutating tools sequentially."""
            for i in image_indices:
                results[i] = await self._run_single_tool(tool_calls[i], tool_registry, turn_idx)

        async def _run_other_tools() -> None:
            """Run independent tools concurrently."""
            if not other_indices:
                return
            other_results = await asyncio.gather(
                *(
                    self._run_single_tool(tool_calls[i], tool_registry, turn_idx)
                    for i in other_indices
                )
            )
            for i, result in zip(other_indices, other_results, strict=True):
                results[i] = result

        # Run both groups in parallel — image tools are sequential within
        # their group, but the group itself overlaps with independent tools.
        await asyncio.gather(_run_image_tools(), _run_other_tools())

        # All slots must be filled — a None here means a tool silently
        # returned nothing, which should never happen.
        for i, r in enumerate(results):
            if r is None:
                tc_name = tool_calls[i].name
                raise AgenticProcessingError(
                    f"Tool '{tc_name}' produced no result (slot {i} is None)",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "tool_no_result", "tool": tc_name},
                )

        # mypy/pyright: after the None check above, all elements are ToolResult
        return [r for r in results if r is not None]

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

    @staticmethod
    @beartype
    def _strip_stale_images(messages: list[dict[str, Any]]) -> None:
        """Replace base64 image data URLs in older messages with text placeholders.

        Strips images from two sources:

        1. **Initial user message** — the original input images have already
           been seen by the model on turn 0.  Subsequent turns get updated
           images from tool results, so the originals are redundant payload.
        2. **Older tool result messages** — keeps images only in tool messages
           that follow the *last* assistant message (the most recent round).

        This dramatically reduces the payload sent on subsequent API calls.
        """
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        for i in range(last_assistant_idx):
            msg = messages[i]
            role = msg.get("role")
            if role not in ("tool", "user"):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            placeholder = (
                "[original image omitted]" if role == "user" else "[previous tool image omitted]"
            )
            new_content: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "image_url":
                    new_content.append({"type": "text", "text": placeholder})
                else:
                    new_content.append(part)
            msg["content"] = new_content
