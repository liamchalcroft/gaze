"""Base agentic processor for radiology VLM analysis.

Provides multi-turn agentic loop with tool calling support.
Task-specific details are provided via dependency injection.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import cast

from beartype import beartype
from loguru import logger
from PIL import Image

from gaze._frozen import deep_thaw
from gaze._image import ImageInput
from gaze._image import _downscale_image
from gaze._schema_skeleton import _build_schema_skeleton
from gaze._schema_skeleton import _try_wrap_inner_schema
from gaze.exceptions import AgenticProcessingError
from gaze.exceptions import GazeError
from gaze.exceptions import SchemaValidationError
from gaze.exceptions import ToolExecutionError
from gaze.exceptions import UnknownToolError
from gaze.models import AdapterProtocol
from gaze.retrieval.base import _sanitize_exception_message
from gaze.tools import Tool
from gaze.tools import ToolRegistry
from gaze.tools import create_search_tools
from gaze.tools import create_visual_tools
from gaze.types import AgenticResult
from gaze.types import RunConfig
from gaze.types import ToolCall
from gaze.types import ToolResult
from gaze.types import Turn
from gaze.utils.json_coerce import coerce_json_types
from gaze.utils.json_extract import extract_json_from_text

# Maximum characters allowed in a single tool result message.
# Limits prompt-injection surface from external data (PubMed abstracts, etc.).
_MAX_TOOL_CONTENT_CHARS = 8_000

# Agentic processing defaults (previously AgenticConfig dataclass).
_MAX_TURNS_LIMIT = 30
_DEFAULT_MAX_TURNS = 10
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_TEMPERATURE = 0.0
_MAX_CONSECUTIVE_NUDGES = 2
_MAX_RECOVERY_NUDGES = _MAX_CONSECUTIVE_NUDGES + 2  # Total nudges before giving up

# Regex for ASCII/Unicode control characters (except newline/tab).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Pre-compiled regexes for tool name sanitization (used per tool call).
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_TRAILING_PARENS_RE = re.compile(r"\(.*\)\s*$")

# Turns with zero tool calls before force-finalizing in agentic mode.
# Prevents wasting tokens when the model ignores available tools.
_IDLE_TOOL_TURNS_LIMIT = 3

# Tools that modify the image coordinate space. After these, bounding box
# coordinates no longer correspond to the original image.
# - crop/zoom: viewport changes to a subregion/magnified area
# - rotate: pixel coordinates shift (90/270° also swaps dimensions)
# - flip_horizontal/flip_vertical: mirrors coordinate axes
_COORD_MODIFYING_TOOLS = frozenset({"crop", "zoom", "rotate", "flip_horizontal", "flip_vertical"})

# Tools that modify the image intensity space. After these, pixel values
# no longer represent original tissue intensities — quantitative
# measurements (get_intensity_stats, intensity_profile) reflect the
# transformed data.
_INTENSITY_MODIFYING_TOOLS = frozenset(
    {
        "threshold",
        "window_level",
        "equalize_histogram",
        "adaptive_equalize",
        "invert",
        "detect_edges",
        "symmetry_diff",
        "morphological",
        "denoise",
        "adjust_contrast",
        "adjust_brightness",
        "adjust_sharpness",
    }
)


def _sanitize_tool_content(text: str, *, max_chars: int = _MAX_TOOL_CONTENT_CHARS) -> str:
    """Sanitize tool result text before injecting into the LLM conversation.

    1. Strip control characters that could confuse tokenizers.
    2. Truncate to *max_chars* to limit prompt-injection surface.
    3. Wrap in a unique, per-call untrusted-content marker so the model
       can distinguish tool output from system/user instructions.
       The boundary is randomized to prevent adversarial content from
       closing the marker prematurely.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    boundary = secrets.token_hex(8)
    return f"[Tool Result - External Data - {boundary}]\n{text}\n[End Tool Result - {boundary}]"


class AgenticProcessorBase(ABC):
    """Abstract base class for multi-turn agentic analysis.

    Provides core agentic loop with tool calling support. Subclasses
    must implement methods to provide task-specific prompts and schemas.
    """

    @beartype
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
        max_encode_dimension: int | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize agentic processor.

        Args:
            model_name: Model name for analysis
            use_tools: Enable visual tools
            use_web_search: Enable web/image search tools
            max_turns: Maximum turns before forced completion (default 10)
            reasoning_enabled: Enable model's internal reasoning
            reasoning_effort: Reasoning effort level ("high", "medium", "low")
            enable_caching: Enable prompt caching
            disabled_tools: Specific tool names to disable
            adapter_factory: Optional factory for custom model adapter
            max_encode_dimension: If set, downscale images so neither side
                exceeds this many pixels before base64 encoding.
            seed: Random seed for model API calls (reproducibility).
            max_tokens: Max completion tokens per turn. If None, uses
                the module default (16384).

        Raises:
            ValueError: If max_turns < 1
        """
        if max_turns is None:
            max_turns = _DEFAULT_MAX_TURNS
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        allowed_reasoning_efforts = {"low", "medium", "high"}
        if reasoning_effort not in allowed_reasoning_efforts:
            raise ValueError(
                f"reasoning_effort must be one of "
                f"{sorted(allowed_reasoning_efforts)}, got {reasoning_effort}"
            )

        self.model_name = model_name
        self.use_tools = use_tools
        self.use_web_search = use_web_search
        self.max_encode_dimension = max_encode_dimension
        self.seed = seed
        self.max_tokens = max_tokens

        if max_turns > _MAX_TURNS_LIMIT:
            logger.warning(
                f"max_turns={max_turns} exceeds max_turns_limit={_MAX_TURNS_LIMIT}, clamping"
            )
        self.max_turns = min(max_turns, _MAX_TURNS_LIMIT)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching
        self._adapter_factory = adapter_factory

        self._disabled_tools: set[str] = set(disabled_tools or [])
        if not use_web_search:
            self._disabled_tools.update(["search_web", "search_images"])

        self._model_adapter: AdapterProtocol | None = None
        self._shared_web_search_manager = None
        self._shared_image_search_manager = None

        # Per-processor caches — tools, schemas, and docs are invariant across
        # analyze() calls because use_tools/use_web_search/_disabled_tools are
        # fixed at construction time.
        self._visual_tools_cache: list[Tool] | None = None
        self._search_tools_cache: list[Tool] | None = None
        self._tool_schemas_cache: list[dict[str, Any]] | None = None
        self._tool_docs_cache: str | None = None

    @beartype
    def _ensure_initialized(self) -> None:
        """Ensure model adapter is initialized."""
        if self._model_adapter is None:
            if self._adapter_factory:
                self._model_adapter = self._adapter_factory()
            else:
                # Import here to avoid coupling the abstract base to a concrete adapter
                from gaze.models import OpenAIAdapter

                self._model_adapter = OpenAIAdapter(
                    model_name=self.model_name,
                    reasoning_enabled=self.reasoning_enabled,
                    reasoning_effort=self.reasoning_effort,
                    enable_caching=self.enable_caching,
                )

    def _get_shared_web_search_manager(self):
        if self._shared_web_search_manager is None:
            from gaze.retrieval.web_search import WebSearchManager

            self._shared_web_search_manager = WebSearchManager()
        return self._shared_web_search_manager

    def _get_shared_image_search_manager(self):
        if self._shared_image_search_manager is None:
            from gaze.retrieval.image_search import MedicalImageSearchManager

            self._shared_image_search_manager = MedicalImageSearchManager()
        return self._shared_image_search_manager

    async def __aenter__(self) -> AgenticProcessorBase:
        """Enter an async context, returning self. Use ``async with``."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context, releasing processor-owned resources."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close processor-owned resources that are reused across analyses."""
        close_tasks: list[asyncio.Task[None]] = []
        if self._shared_web_search_manager is not None:
            close_tasks.append(asyncio.create_task(self._shared_web_search_manager.close()))
            self._shared_web_search_manager = None
        if self._shared_image_search_manager is not None:
            close_tasks.append(asyncio.create_task(self._shared_image_search_manager.close()))
            self._shared_image_search_manager = None
        if close_tasks:
            await asyncio.gather(*close_tasks)

    def _get_visual_tools(self) -> list[Tool]:
        """Return cached visual tools, creating them on first call."""
        if self._visual_tools_cache is None:
            self._visual_tools_cache = create_visual_tools(self._disabled_tools)
        return self._visual_tools_cache

    def _get_search_tools(self) -> list[Tool]:
        """Return cached search tools, creating them on first call."""
        if self._search_tools_cache is None:
            self._search_tools_cache = create_search_tools(self._disabled_tools)
        return self._search_tools_cache

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
        web_search_manager = None
        image_search_manager = None
        if not images:
            if self.use_web_search:
                tools = list(self._get_search_tools())
                tool_names = {tool.name for tool in tools}
                if "search_web" in tool_names:
                    web_search_manager = self._get_shared_web_search_manager()
                if "search_images" in tool_names:
                    image_search_manager = self._get_shared_image_search_manager()
                return ToolRegistry(
                    image_path=None,
                    tools=tools,
                    web_search_manager=web_search_manager,
                    image_search_manager=image_search_manager,
                )
            return None

        tools: list[Tool] = []

        if self.use_tools:
            tools.extend(self._get_visual_tools())

        if self.use_web_search:
            tools.extend(self._get_search_tools())

        if not tools:
            return None

        tool_names = {tool.name for tool in tools}
        if "search_web" in tool_names:
            web_search_manager = self._get_shared_web_search_manager()
        if "search_images" in tool_names:
            image_search_manager = self._get_shared_image_search_manager()

        if len(images) > 1:
            logger.warning(
                f"Tool registry only supports a single active image; "
                f"using first of {len(images)} images, rest will be ignored"
            )
        active_image = images[0]

        # If the PIL Image was kept from load(), hand it directly to the
        # ImageManager to avoid re-reading the file from disk.
        # transfer_ownership=True avoids an extra ~12MB copy; the
        # ImageInput.pil_image reference is not used after this point.
        if active_image.pil_image is not None:
            registry = ToolRegistry(
                tools=tools,
                web_search_manager=web_search_manager,
                image_search_manager=image_search_manager,
            )
            mgr = registry.get_image_manager()
            mgr.set_preloaded_image(
                active_image.pil_image,
                active_image.path,
                transfer_ownership=True,
            )
            if active_image.encoded is not None:
                mgr.original_encoding = active_image.encoded
            return registry

        return ToolRegistry(
            image_path=active_image.path,
            tools=tools,
            web_search_manager=web_search_manager,
            image_search_manager=image_search_manager,
        )

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

        # Penalize for non-tool assistant turns beyond the final answer.
        # Extra such turns indicate recovery nudges were needed.
        non_tool_assistant = sum(1 for t in turns if t.role == "assistant" and not t.tool_calls)
        if non_tool_assistant > 1:
            confidence -= 0.05 * (non_tool_assistant - 1)

        return max(0.0, min(1.0, confidence))

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

        # When downscaling is needed, defer JPEG+base64 encoding until after
        # the resize to avoid encoding at full resolution then discarding it.
        if self.max_encode_dimension is not None:
            image_inputs = list(
                await asyncio.gather(*(img._aload_pil_only() for img in image_inputs))
            )
            image_inputs = [
                _downscale_image(img, self.max_encode_dimension) for img in image_inputs
            ]
        else:
            image_inputs = list(await asyncio.gather(*(img.aload() for img in image_inputs)))

        # Single-turn mode never offers tools (last turn withholds them),
        # so skip registry creation to avoid wasted I/O and memory.
        tool_registry = self._create_tool_registry(image_inputs) if self.max_turns > 1 else None

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
            if ".." in images.parts:
                raise ValueError(f"Path traversal detected: {images}")
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
                if ".." in item.parts:
                    raise ValueError(f"Path traversal detected: {item}")
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
        response_schema = self.get_response_schema()
        system_prompt = self.get_system_prompt(images=images, metadata=metadata)
        if self.max_turns > 1:
            policy_lines = [
                f"Multi-turn session with a maximum of {self.max_turns} turns.",
                'Return JSON every turn with a boolean field "continue".',
                'Set "continue": true when you need more tools/analysis; false when final.',
                "Final response must satisfy the provided response schema.",
            ]
            system_prompt = f"{system_prompt}\n\nPOLICY:\n- " + "\n- ".join(policy_lines)

        # Build schema skeleton once; reused for single-turn prompt injection
        # and force-finalize nudges (avoids redundant schema traversal).
        skeleton, field_hints = _build_schema_skeleton(response_schema)
        skeleton_str = json.dumps(skeleton, indent=2)

        # Single-turn: inject JSON skeleton so models that ignore response_format
        # (e.g. local models via LM Studio) still know the expected output shape.
        if self.max_turns == 1 and response_schema is not None:
            hints_block = "\n".join(field_hints)
            system_prompt = (
                f"{system_prompt}\n\n"
                f"OUTPUT FORMAT: You MUST respond with ONLY a valid JSON object. "
                f"No other text, no markdown, no explanation outside the JSON. "
                f"Keep your reasoning concise — the JSON output is what matters.\n"
                f"Required structure:\n{skeleton_str}"
            )
            if hints_block:
                system_prompt += f"\n\nField descriptions:\n{hints_block}"

        if tool_registry and self.max_turns > 1:
            # Reuse cached docs across analyze() calls — tools are invariant.
            if self._tool_docs_cache is None:
                self._tool_docs_cache = (
                    tool_registry.get_documenter().generate_prompt_documentation()
                )
            tool_docs = self._tool_docs_cache
            if tool_docs:
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"AVAILABLE TOOLS:\n"
                    f"You have access to the following tools through function calling. "
                    f"Call these tools to gather information, manipulate the image, "
                    f"or retrieve evidence.\n\n"
                    f"{tool_docs}"
                )

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

        # Reuse cached schemas across analyze() calls — tools are invariant.
        if tool_registry is not None:
            if self._tool_schemas_cache is None:
                self._tool_schemas_cache = tool_registry.get_tool_schemas()
            tool_schemas = self._tool_schemas_cache
        else:
            tool_schemas = None

        total_tokens: int = 0
        nudge_count: int = 0
        total_tool_calls: int = 0
        coord_space_modified: bool = False
        intensity_modified: bool = False
        idle_tool_nudged: bool = False
        strip_watermark: int = 0

        def _force_finalize_message() -> str:
            """Build a force-finalize message using the pre-built skeleton."""
            return (
                "[System: Your previous responses were not valid JSON. "
                "You MUST respond with ONLY a JSON object NOW — no other text. "
                f"Copy this template and fill in your analysis:\n{skeleton_str}\n"
                "Replace every placeholder with your actual findings. "
                'Set "continue": false. Output ONLY the JSON.]'
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
                # Build final-turn message with original image + schema reminder.
                # No reset_to_original() — original images are re-injected from
                # ImageInput.encoded; the ImageManager is closed in analyze().
                final_parts: list[dict[str, Any]] = []

                coord_note = ""
                if images:
                    w, h = images[0].width, images[0].height
                    if coord_space_modified:
                        coord_note = (
                            f" WARNING: You used crop/zoom/rotate/flip which changed the "
                            f"coordinate space. The original {w}x{h} image is "
                            f"re-attached below. "
                            f"Any bounding boxes from your transformed analysis are "
                            f"INVALID. Re-examine this original image and provide ALL "
                            f"coordinates in the original pixel space "
                            f"[0, {w - 1}] x [0, {h - 1}]."
                        )
                    else:
                        coord_note = (
                            " The ORIGINAL (untransformed) image is re-attached below. "
                            "All spatial coordinates (bounding boxes) MUST reference "
                            "this original image, NOT any zoomed or cropped version."
                        )
                    if intensity_modified:
                        coord_note += (
                            " NOTE: You used intensity-modifying tools (threshold, "
                            "window_level, equalize, etc.) during this session. "
                            "Any intensity measurements from modified images do NOT "
                            "reflect original tissue characteristics."
                        )

                schema_note = ""
                if response_schema is not None:
                    schema_obj = response_schema.get("json_schema", {}).get("schema", {})
                    top_keys = list(schema_obj.get("properties", {}).keys())
                    if top_keys:
                        schema_note = (
                            f" You MUST respond with a complete JSON object containing "
                            f"these required top-level keys: {top_keys}."
                        )

                final_parts.append(
                    {
                        "type": "text",
                        "text": (
                            f"[System: This is your FINAL turn. Tools are no longer available."
                            f"{coord_note}{schema_note} "
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

                messages.append({"role": "user", "content": final_parts})

            # Circuit-breaker: if nudges have been exhausted without recovery,
            # stop burning turns with the same force-finalize message.
            if nudge_count > _MAX_RECOVERY_NUDGES:
                raise AgenticProcessingError(
                    f"Model failed to produce valid output after {nudge_count} "
                    f"consecutive recovery attempts",
                    turns_completed=len(turns),
                )

            # Strip stale base64 images from earlier rounds (including the
            # initial user message) to reduce payload on subsequent API calls.
            if turn_idx > 0 and images:
                strip_watermark = self._strip_stale_images(messages, strip_watermark)

            chat_result = await model_adapter.generate_chat(
                messages=messages,
                max_tokens=self.max_tokens or _DEFAULT_MAX_TOKENS,
                temperature=_DEFAULT_TEMPERATURE,
                tools=current_tools,
                response_format=current_format,
                seed=self.seed,
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
                    clean_name = _SPECIAL_TOKEN_RE.sub("", raw_name).strip()
                    # Strip trailing parentheses — some models (GLM-4.6V) include
                    # "()" or "(args)" in the tool name field.
                    clean_name = _TRAILING_PARENS_RE.sub("", clean_name).strip()
                    if not clean_name:
                        logger.warning(
                            "Tool call {} on turn {} has empty name "
                            "after sanitization (raw={!r}), skipping",
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
                # Before crashing, check if the model also returned valid JSON
                # text alongside the tool calls — some models do both.  If the
                # text is a valid, complete response we can salvage it.
                if response_text.strip():
                    salvaged = extract_json_from_text(response_text)
                    if salvaged is not None and response_schema is not None:
                        coerce_json_types(salvaged, response_schema)
                    if (
                        salvaged is not None
                        and isinstance(salvaged.get("continue"), bool)
                        and self.validate_response(salvaged)
                    ):
                        logger.warning(
                            f"Turn {turn_idx + 1}: Model returned tool calls on "
                            f"{'final turn' if is_last_turn else 'tools-unavailable turn'} "
                            f"alongside a valid JSON response — salvaging text response."
                        )
                        salvaged["continue"] = False
                        # Record the turn (without executing the spurious tool calls)
                        turns.append(Turn(role="assistant", content=response_text))
                        final_response = salvaged
                        break

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

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response_text,
            }
            if typed_tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                            if isinstance(tc.arguments, str)
                            else json.dumps(deep_thaw(tc.arguments)),
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
                        if image_data_url and not multipart_ok:
                            raw = (
                                f"{raw}\n[Image produced but cannot be displayed "
                                f"in this adapter's tool messages. Use the visual "
                                f"information from the text description above.]"
                            )
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
                total_tool_calls += len(typed_tool_calls)
                # Inject turn counter so the model can budget remaining turns
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Turn {turn_idx + 1}/{self.max_turns}]",
                    }
                )
                succeeded = frozenset(
                    tc.name
                    for tc, tr in zip(typed_tool_calls, tool_results, strict=True)
                    if tr.success
                )
                if "reset" in succeeded:
                    coord_space_modified = False
                    intensity_modified = False
                else:
                    if succeeded & _COORD_MODIFYING_TOOLS:
                        coord_space_modified = True
                    if succeeded & _INTENSITY_MODIFYING_TOOLS:
                        intensity_modified = True
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
                        f"{_MAX_CONSECUTIVE_NUDGES}."
                    )
                    if nudge_count >= _MAX_CONSECUTIVE_NUDGES:
                        messages.append({"role": "user", "content": _force_finalize_message()})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Your previous response was too long and got "
                                    "cut off. Be more concise. Respond with ONLY a short "
                                    "JSON object — no explanations outside the JSON. "
                                    f"Required structure:\n{skeleton_str}]"
                                ),
                            }
                        )
                    continue
                # Last turn truncated — try to salvage partial JSON before failing.
                # Thinking models often consume most of the token budget on
                # reasoning, leaving the visible output truncated mid-JSON.
                if response_text.strip():
                    salvaged = extract_json_from_text(response_text)
                    if salvaged is not None and isinstance(salvaged, dict):
                        salvaged["continue"] = False
                        if response_schema is not None:
                            coerce_json_types(salvaged, response_schema)
                        # If salvaged keys don't match top-level schema but DO
                        # match a sub-schema property, wrap them.  This handles
                        # truncated output where the model started generating an
                        # inner object (e.g. caption fields) before being cut off.
                        if response_schema is not None and not self.validate_response(salvaged):
                            salvaged = _try_wrap_inner_schema(salvaged, response_schema)
                        logger.warning(
                            f"Turn {turn_idx + 1} truncated but salvaged partial JSON "
                            f"(keys: {list(salvaged.keys())[:10]})"
                        )
                        final_response = salvaged
                        break
                effective_max = self.max_tokens or _DEFAULT_MAX_TOKENS
                total = gen_log.prompt_tokens + gen_log.completion_tokens
                # When completion_tokens < max_tokens but the model still hit
                # finish_reason=length, the server's context window (n_ctx) is
                # the binding constraint, not our max_tokens parameter.
                if gen_log.completion_tokens < effective_max * 0.9:
                    hint = (
                        f"Server context window appears to be ~{total} tokens "
                        f"(prompt={gen_log.prompt_tokens} + "
                        f"completion={gen_log.completion_tokens}). "
                        f"Increase n_ctx in LM Studio or use a model with a "
                        f"larger context window."
                    )
                else:
                    hint = "Increase max_tokens or simplify the response schema."
                raise AgenticProcessingError(
                    f"Response truncated on turn {turn_idx + 1} "
                    f"(finish_reason='length', completion_tokens="
                    f"{gen_log.completion_tokens}, "
                    f"max_tokens={effective_max}). {hint}",
                    turns_completed=turn_idx + 1,
                )

            # Handle empty or non-JSON responses on intermediate turns:
            # nudge the model instead of crashing.
            if not response_text.strip() and not is_last_turn:
                nudge_count += 1
                logger.warning(
                    f"Turn {turn_idx + 1} returned empty response with no tool calls. "
                    f"Nudge {nudge_count}/{_MAX_CONSECUTIVE_NUDGES}."
                )
                if nudge_count >= _MAX_CONSECUTIVE_NUDGES:
                    messages.append({"role": "user", "content": _force_finalize_message()})
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: You returned an empty response. Respond with "
                                "ONLY a JSON object. Use tools if you need more "
                                "information, or set 'continue': false to finalize. "
                                f"Required structure:\n{skeleton_str}]"
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
                            f"Nudge {nudge_count}/{_MAX_CONSECUTIVE_NUDGES}."
                        )
                        if nudge_count >= _MAX_CONSECUTIVE_NUDGES:
                            messages.append({"role": "user", "content": _force_finalize_message()})
                        else:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[System: Your response was not valid JSON. "
                                        "Respond with ONLY a JSON object — no markdown, "
                                        "no explanation outside the JSON. Required "
                                        f"structure:\n{skeleton_str}]"
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
                if not is_last_turn:
                    nudge_count += 1
                    logger.warning(
                        f"Turn {turn_idx + 1} returned non-object JSON "
                        f"(got {type(parsed_obj).__name__}). "
                        f"Nudge {nudge_count}/{_MAX_CONSECUTIVE_NUDGES}."
                    )
                    if nudge_count >= _MAX_CONSECUTIVE_NUDGES:
                        messages.append({"role": "user", "content": _force_finalize_message()})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Your response was a JSON "
                                    f"{type(parsed_obj).__name__}, not an object. "
                                    "Respond with ONLY a JSON object matching this "
                                    f"structure:\n{skeleton_str}]"
                                ),
                            }
                        )
                    continue
                raise AgenticProcessingError(
                    "Model response must be a JSON object",
                    turns_completed=turn_idx + 1,
                    partial_response={"error": "invalid_response_type"},
                )

            parsed: dict[str, Any] = parsed_obj
            nudge_count = 0  # Valid JSON resets nudge counter

            # Coerce types centrally so processors don't need to call it
            # individually.  Runs before validate_response() to prevent
            # unnecessary nudges from string-vs-number mismatches.
            if response_schema is not None:
                coerce_json_types(parsed, response_schema)

            wants_continue = parsed.get("continue", False)
            # Inject missing "continue" — local models often omit it when
            # the answer is "false" (they treat absence as false).  Without
            # this, validate_response() rejects the response because the
            # key is literally absent from the dict.
            if "continue" not in parsed:
                parsed["continue"] = False
            if not isinstance(wants_continue, bool):
                # Local models frequently return non-bool values for "continue":
                #   None / null  → treat as false (model is done)
                #   0 / 1        → coerce to bool
                #   "true"/"false" strings → coerce to bool
                if wants_continue is None:
                    wants_continue = False
                    parsed["continue"] = False
                    logger.warning(f"Turn {turn_idx + 1}: coerced null 'continue' to False")
                elif isinstance(wants_continue, int):
                    original_value = wants_continue
                    wants_continue = bool(wants_continue)
                    parsed["continue"] = wants_continue
                    logger.warning(
                        f"Turn {turn_idx + 1}: coerced int 'continue' "
                        f"value {original_value!r} to bool {wants_continue!r}"
                    )
                elif isinstance(wants_continue, str) and wants_continue.strip().lower() in (
                    "true",
                    "false",
                    "yes",
                    "no",
                ):
                    original_value = wants_continue
                    wants_continue = wants_continue.strip().lower() in ("true", "yes")
                    parsed["continue"] = wants_continue
                    logger.warning(
                        f"Turn {turn_idx + 1}: coerced string 'continue' "
                        f"value {original_value!r} to bool {wants_continue!r}"
                    )
                else:
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
                    # Identify missing required fields from schema to guide the model
                    missing_hint = ""
                    if response_schema is not None:
                        schema_props = (
                            response_schema.get("json_schema", {})
                            .get("schema", {})
                            .get("required", [])
                        )
                        missing = [k for k in schema_props if k not in parsed]
                        if missing:
                            missing_hint = f" Missing top-level fields: {missing}."
                        else:
                            # All top-level keys present — check for type/value issues
                            missing_hint = (
                                " All top-level keys are present but some have"
                                " invalid types or values (check numbers, booleans,"
                                " nested objects)."
                            )
                    logger.warning(
                        f"Turn {turn_idx + 1} returned incomplete response "
                        f"(keys: {list(parsed.keys())[:10]}).{missing_hint} "
                        f"Nudge {nudge_count}/{_MAX_CONSECUTIVE_NUDGES}."
                    )
                    if nudge_count >= _MAX_CONSECUTIVE_NUDGES:
                        messages.append({"role": "user", "content": _force_finalize_message()})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Your response is incomplete — it failed "
                                    f"validation.{missing_hint} Respond with a complete "
                                    "JSON object matching this structure:\n"
                                    f"{skeleton_str}\n"
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

            # Early termination: if tools are available but the model hasn't
            # used any after several turns, force finalize to avoid token waste.
            # After the first nudge, if the model *still* doesn't use tools,
            # force-accept whatever it returned rather than burning more turns.
            if (
                tool_registry is not None
                and total_tool_calls == 0
                and turn_idx >= _IDLE_TOOL_TURNS_LIMIT - 1
            ):
                if idle_tool_nudged:
                    # Already nudged once — accept this response as final.
                    logger.warning(
                        f"Turn {turn_idx + 1}: Still no tools after idle-tool nudge. "
                        f"Force-accepting current response."
                    )
                    parsed["continue"] = False
                    final_response = parsed
                    break
                logger.warning(
                    f"Turn {turn_idx + 1}: No tools used after {turn_idx + 1} turns "
                    f"in agentic mode. Force-finalizing to avoid token waste."
                )
                messages.append({"role": "user", "content": _force_finalize_message()})
                idle_tool_nudged = True
                continue

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
            # Defensive: on the last turn, every code path breaks or raises.
            # If a future change makes this reachable, fail loudly.
            raise AgenticProcessingError(
                f"Internal error: agentic loop completed {self.max_turns} turns "
                f"without producing a final response (this should be unreachable)",
                turns_completed=len(turns),
            )

        # Coerce types on the final response (catches salvaged paths that
        # bypassed the per-turn coerce above).
        if response_schema is not None:
            coerce_json_types(final_response, response_schema)

        if not self.validate_response(final_response):
            # Small/local models sometimes return a sub-object (e.g. the
            # inner keys of ``region_of_interest``) instead of the full
            # top-level schema.  The same wrapping logic used for truncated
            # responses can recover these cases.
            if response_schema is not None:
                wrapped = _try_wrap_inner_schema(final_response, response_schema)
                if wrapped is not final_response:
                    # Re-run coerce on the wrapped dict — the pre-wrapping coerce
                    # at line 1516 operated on the un-wrapped structure where field
                    # names didn't match the top-level schema, so nested fields
                    # went uncoerced.  Now that wrapping placed them under the
                    # correct parent key, coercion can find and fix them.
                    coerce_json_types(wrapped, response_schema)
                    if self.validate_response(wrapped):
                        logger.warning(
                            f"Recovered response via inner-schema wrapping "
                            f"(original keys: {list(final_response.keys())[:10]})"
                        )
                        final_response = wrapped

            if not self.validate_response(final_response):
                top_keys = list(final_response.keys())[:10]
                missing_fields: list[str] = []
                if response_schema is not None:
                    schema_obj = response_schema.get("json_schema", {}).get("schema", {})
                    required_raw = schema_obj.get("required", [])
                    if isinstance(required_raw, list):
                        required_list = cast("list[str]", required_raw)
                        missing_fields = [
                            field for field in required_list if field not in final_response
                        ]
                raise SchemaValidationError(
                    f"Final response failed schema validation. Top-level keys: {top_keys}",
                    turns_completed=len(turns),
                    missing_fields=missing_fields,
                    response=final_response,
                )

        confidence = self.calculate_confidence(final_response, turns)

        run_config = RunConfig(
            model_name=self.model_name,
            temperature=_DEFAULT_TEMPERATURE,
            seed=self.seed,
            max_tokens=self.max_tokens or _DEFAULT_MAX_TOKENS,
            max_turns=self.max_turns,
        )

        return AgenticResult(
            final_response=final_response,
            turns=tuple(turns),
            total_tokens=total_tokens,
            confidence=confidence,
            run_config=run_config,
        )

    def _parse_tool_args(self, tool_call: ToolCall) -> dict[str, Any]:
        """Parse tool call arguments, raising on malformed JSON.

        Raises ToolExecutionError (not AgenticProcessingError) so that
        callers in _run_single_tool can catch it and return an error
        ToolResult, letting the model self-correct.
        """
        if isinstance(tool_call.arguments, str):
            try:
                parsed_args: dict[str, Any] | list | str | int | float | bool | None = json.loads(
                    tool_call.arguments
                )
            except json.JSONDecodeError as e:
                raise ToolExecutionError(
                    f"Malformed JSON in tool arguments: {e}",
                    tool_name=tool_call.name,
                ) from e

            if not isinstance(parsed_args, dict):
                raise ToolExecutionError(
                    f"Tool arguments must be a JSON object, got {type(parsed_args).__name__}",
                    tool_name=tool_call.name,
                )
            return parsed_args

        if not isinstance(tool_call.arguments, Mapping):
            raise ToolExecutionError(
                f"Tool arguments must be a JSON object, got {type(tool_call.arguments).__name__}",
                tool_name=tool_call.name,
            )
        thawed_args = deep_thaw(tool_call.arguments)
        if not isinstance(thawed_args, dict):
            raise ToolExecutionError(
                f"Tool arguments must be a JSON object, got {type(thawed_args).__name__}",
                tool_name=tool_call.name,
            )
        return thawed_args

    @beartype
    async def _run_single_tool(
        self,
        tool_call: ToolCall,
        tool_registry: ToolRegistry,
        turn_idx: int,
    ) -> ToolResult:
        """Execute a single tool call with error handling.

        All tool errors — including unknown tool names — return a
        ``ToolResult`` with an error description so the model can
        self-correct on subsequent turns.
        """
        try:
            tool_args = self._parse_tool_args(tool_call)
            logger.debug(f"Executing: {tool_call.name}({tool_args})")
            return await tool_registry.execute(tool_call.name, **tool_args)
        except UnknownToolError as e:
            # Model hallucinated a tool name. Return an error ToolResult so it
            # can self-correct on the next turn instead of crashing the loop.
            available = ", ".join(sorted(e.available_tools)) if e.available_tools else "none"
            logger.warning(
                f"Unknown tool '{tool_call.name}' on turn {turn_idx + 1}. Available: {available}",
            )
            return ToolResult(
                tool_name=tool_call.name,
                description=f"Tool '{tool_call.name}' does not exist",
                error=(f"Unknown tool '{tool_call.name}'. Available tools: {available}"),
            )
        except ToolExecutionError as e:
            logger.warning(f"Tool '{tool_call.name}' failed on turn {turn_idx + 1}: {e}")
            return ToolResult(
                tool_name=tool_call.name,
                description=f"Tool '{tool_call.name}' failed",
                error=_sanitize_exception_message(e),
            )
        except (
            TypeError,
            ValueError,
            RuntimeError,
            OSError,
            LookupError,
            AttributeError,
            ArithmeticError,
            asyncio.TimeoutError,
            GazeError,
        ) as e:
            logger.warning(
                "Tool '%s' crashed on turn %d (%s): %s",
                tool_call.name,
                turn_idx + 1,
                type(e).__name__,
                _sanitize_exception_message(e),
            )
            return ToolResult(
                tool_name=tool_call.name,
                description=f"Tool '{tool_call.name}' encountered an error",
                error=f"{type(e).__name__}: {_sanitize_exception_message(e)}",
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
        if len(tool_calls) <= 1:
            return [await self._run_single_tool(tc, tool_registry, turn_idx) for tc in tool_calls]

        # Single-pass partition avoids double documenter lookup per tool.
        documenter = tool_registry.get_documenter()
        image_indices: list[int] = []
        other_indices: list[int] = []
        for i, tc in enumerate(tool_calls):
            tool = documenter.get_tool(tc.name)
            if tool is not None and tool.requires_image:
                image_indices.append(i)
            else:
                other_indices.append(i)

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
    def _strip_stale_images(messages: list[dict[str, Any]], start_index: int = 0) -> int:
        """Replace base64 image data URLs in older messages with text placeholders.

        Strips images from two sources:

        1. **Initial user message** — the original input images have already
           been seen by the model on turn 0.  Subsequent turns get updated
           images from tool results, so the originals are redundant payload.
        2. **Older tool result messages** — keeps images only in tool messages
           that follow the *last* assistant message (the most recent round).

        This dramatically reduces the payload sent on subsequent API calls.

        Args:
            messages: The conversation message list (mutated in place).
            start_index: Skip messages before this index — they have already
                been stripped by a prior call.

        Returns:
            The ``last_assistant_idx`` up to which stripping was performed.
            Pass this value as ``start_index`` on the next call to avoid
            re-scanning already-processed messages.
        """
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        for i in range(start_index, last_assistant_idx):
            msg = messages[i]
            role = msg.get("role")
            if role not in ("tool", "user"):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # Fast path: skip messages already stripped (no image_url parts left).
            parts = cast("list[dict[str, Any]]", content)
            if not any(part.get("type") == "image_url" for part in parts):
                continue
            placeholder = (
                "[original image omitted]" if role == "user" else "[previous tool image omitted]"
            )
            new_content: list[dict[str, Any]] = []
            for part in parts:
                if part.get("type") == "image_url":
                    new_content.append({"type": "text", "text": placeholder})
                else:
                    new_content.append(part)
            msg["content"] = new_content
        return last_assistant_idx
