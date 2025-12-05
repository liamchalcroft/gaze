"""Agentic processor for medical image analysis.

Provides clean, coherent multi-turn analysis with:
- Model-controlled continuation via 'continue' field
- Optional tool calling for visual analysis
- Optional reasoning capabilities
- Structured JSON output with caption, diagnosis, and localization
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from PIL import Image

from nova_retrieval_vlm.agentic.tools import ToolExecutionError
from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.agentic.tools import UnknownToolError
from nova_retrieval_vlm.agentic.tools import encode_image
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA


@dataclass
class ToolCall:
    """Represents a tool call request from the model."""

    id: str
    name: str
    arguments: str  # JSON string of arguments

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Create ToolCall from OpenAI-format dict."""
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            arguments=data["arguments"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-format dict."""
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Turn:
    """Represents a single turn in the agentic conversation."""

    role: str  # 'user', 'assistant', 'tool_result'
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    image_base64: str | None = None


@dataclass
class AgenticResult:
    """Result of an agentic analysis session.

    Attributes:
        final_response: Complete JSON response with caption, diagnosis, localization
        turns: All conversation turns in the analysis
        total_tokens: Total tokens consumed across all turns
        confidence: Overall confidence in the analysis
        retrieval_passages: Retrieved context passages (future use)
    """

    final_response: dict[str, Any]
    turns: list[Turn]
    total_tokens: int
    confidence: float
    retrieval_passages: list[str] = field(default_factory=list)


class AgenticProcessingError(Exception):
    """Raised when agentic processing fails."""

    def __init__(
        self, message: str, turns_completed: int, partial_response: dict[str, Any] | None = None
    ):
        self.turns_completed = turns_completed
        self.partial_response = partial_response
        super().__init__(message)


class AgenticProcessor:
    """Multi-turn agentic processor for medical image analysis.

    Architecture:
    - Model-controlled continuation via 'continue' field
    - Optional visual tools (zoom, crop, contrast, threshold)
    - Optional reasoning capabilities
    - Turn limit enforcement with clear errors
    - Structured JSON output for caption, diagnosis, localization
    """

    MAX_TURNS = 20
    CONFIDENCE_THRESHOLD = 0.7

    # Confidence calculation weights (sum to ~1.0 for normalized confidence)
    # Base confidence when response is generated
    CONFIDENCE_BASE = 0.5
    # Weight for response completeness (0-0.3 based on having caption/diagnosis/localization)
    CONFIDENCE_COMPLETENESS_WEIGHT = 0.3
    # Bonus per tool-using turn (capped at CONFIDENCE_TOOL_MAX)
    CONFIDENCE_TOOL_BONUS = 0.1
    CONFIDENCE_TOOL_MAX = 0.2

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_tools: bool = True,
        use_web_search: bool = False,
        max_turns: int = 5,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
        disabled_tools: list[str] | None = None,
    ):
        """Initialize agentic processor.

        Args:
            model_name: Model name for analysis
            use_tools: Enable visual tools (zoom, crop, contrast, threshold, flip, rotate)
            use_web_search: Enable web search for evidence-based analysis
            max_turns: Maximum allowed turns before forced completion (clamped to MAX_TURNS=20)
            reasoning_enabled: Enable model's internal reasoning capabilities
            reasoning_effort: Reasoning effort level ("high", "medium", "low")
            enable_caching: Enable prompt caching for performance
            disabled_tools: List of tool names to disable

        Raises:
            ValueError: If max_turns < 1
        """
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self.model_name = model_name
        self.use_tools = use_tools
        self.use_web_search = use_web_search

        if max_turns > self.MAX_TURNS:
            logger.warning(
                f"max_turns={max_turns} exceeds MAX_TURNS={self.MAX_TURNS}, clamping to {self.MAX_TURNS}"
            )
        self.max_turns = min(max_turns, self.MAX_TURNS)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching

        # Build disabled tools list from config
        self._disabled_tools: set[str] = set(disabled_tools or [])
        if not use_web_search:
            self._disabled_tools.update(["search_web", "search_images"])

        # Lazily initialized components
        self._model_adapter: OpenAIAdapter | None = None

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if self._model_adapter is None:
            self._model_adapter = OpenAIAdapter(
                model_name=self.model_name,
                reasoning_enabled=self.reasoning_enabled,
                reasoning_effort=self.reasoning_effort,
                enable_caching=self.enable_caching,
            )

    @beartype
    async def analyze(
        self,
        image_path: Path,
        metadata: dict[str, Any] | None = None,
        retrieval_passages: list[str] | None = None,
    ) -> AgenticResult:
        """Run agentic analysis on a medical image.

        Uses the unified 'all_tasks' template for comprehensive analysis
        including localization, captioning, and diagnosis in a single pass.

        Args:
            image_path: Path to the medical image
            metadata: Image metadata including clinical history
            retrieval_passages: Optional context passages (future use)

        Returns:
            AgenticResult with final JSON response and conversation history
        """
        # Ensure model adapter is initialized
        self._ensure_initialized()

        metadata = metadata or {}
        retrieval_passages = retrieval_passages or []

        # Initialize tool registry for this image
        tool_registry = ToolRegistry(image_path, disabled_tools=list(self._disabled_tools))

        try:
            return await self._run_analysis(
                image_path=image_path,
                metadata=metadata,
                retrieval_passages=retrieval_passages,
                tool_registry=tool_registry,
            )
        finally:
            # Ensure tool registry resources are cleaned up
            tool_registry.close()

    @beartype
    async def _run_analysis(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        retrieval_passages: list[str],
        tool_registry: ToolRegistry,
    ) -> AgenticResult:
        """Run the actual analysis loop. Separated for resource management."""
        # Load image for dimensions and base64 encoding
        with Image.open(image_path) as image:
            image_width = image.width
            image_height = image.height
            encoded_image = encode_image(image)

        # Use unified all_tasks template for consistent behavior
        system_prompt = create_enhanced_prompt(
            template_name="all_tasks.jinja",
            image_path=image_path,
            passages=retrieval_passages,
            metadata={
                **metadata,
                "width": image_width,
                "height": image_height,
                "image_id": image_path.name,
                "enable_visual_tools": self.use_tools,
                "enable_web_search": self.use_web_search,
            },
            mode="agentic",
        )

        # Initialize conversation
        turns: list[Turn] = []
        total_tokens = 0
        final_response: dict[str, Any] = {}

        # Build conversation messages for agentic loop
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the brain MRI image comprehensively. "
                            "Provide captioning, diagnosis, and localization."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": encoded_image.to_data_url()},
                    },
                ],
            },
        ]

        # Multi-turn agentic loop with proper OpenRouter tool calling
        # _ensure_initialized() guarantees _model_adapter is set
        # Use assert for type narrowing (mypy/pyright)
        assert self._model_adapter is not None
        model_adapter = self._model_adapter

        # Cache tool schemas - they don't change during the conversation
        tool_schemas = tool_registry.get_tool_schemas() if self.use_tools else None
        # IMPORTANT: Can't use both tools and response_format simultaneously
        response_format_when_no_tools = NOVA_UNIFIED_SCHEMA

        for turn_idx in range(self.max_turns):
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            # Use structured output only when NOT using tools
            response_format = None if tool_schemas else response_format_when_no_tools

            # Generate response with tool calling support
            response_text, tool_calls, gen_log = await model_adapter.generate_chat(
                messages=messages,
                max_tokens=8192,
                temperature=0.0,
                tools=tool_schemas,
                response_format=response_format,
            )

            total_tokens += gen_log.tokens if gen_log else 0

            # Parse raw tool calls into typed ToolCall objects
            typed_tool_calls: list[ToolCall] = []
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    if "name" not in tc:
                        raise AgenticProcessingError(
                            f"Tool call {i} missing required 'name' field: {tc}",
                            turns_completed=turn_idx + 1,
                        )
                    if "arguments" not in tc:
                        raise AgenticProcessingError(
                            f"Tool call '{tc['name']}' missing required 'arguments' field",
                            turns_completed=turn_idx + 1,
                        )
                    typed_tool_calls.append(
                        ToolCall(
                            id=tc.get("id", f"nova_{uuid.uuid4().hex[:12]}"),
                            name=tc["name"],
                            arguments=tc["arguments"],
                        )
                    )

            # Create turn record
            turn = Turn(
                role="assistant",
                content=response_text,
                tool_calls=typed_tool_calls,
            )
            turns.append(turn)

            # Add assistant message to conversation - MUST include tool_calls if present
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if response_text:
                assistant_message["content"] = response_text
            if typed_tool_calls:
                # Format tool calls for OpenAI API format
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in typed_tool_calls
                ]
            messages.append(assistant_message)

            # Check if model wants to use tools
            if typed_tool_calls and self.use_tools and turn_idx < self.max_turns - 1:
                logger.info(f"Model requested {len(typed_tool_calls)} tool calls")

                # Execute tools and collect results
                tool_results: list[ToolResult] = []
                for tool_call in typed_tool_calls:
                    # Parse arguments from JSON string
                    try:
                        tool_args = json.loads(tool_call.arguments)
                    except json.JSONDecodeError as e:
                        raise AgenticProcessingError(
                            f"Malformed JSON in tool arguments for '{tool_call.name}': {e}",
                            turns_completed=turn_idx + 1,
                            partial_response={"error": "malformed_tool_args", "tool": tool_call.name},
                        ) from e

                    logger.debug(f"Executing tool: {tool_call.name} with args: {tool_args}")
                    try:
                        result = await tool_registry.execute(tool_call.name, **tool_args)
                    except UnknownToolError as e:
                        logger.warning(f"Model requested unknown tool: {e}")
                        result = ToolResult(
                            tool_name=tool_call.name,
                            description=f"Unknown tool: {tool_call.name}",
                            error=str(e),
                        )
                    except ToolExecutionError as e:
                        logger.warning(f"Tool execution failed: {e}")
                        result = ToolResult(
                            tool_name=tool_call.name,
                            description="Tool execution failed",
                            error=str(e),
                        )
                    tool_results.append(result)

                # Add tool results to conversation with proper format
                for i, result in enumerate(tool_results):
                    tool_call_id = typed_tool_calls[i].id

                    # Build tool result content - include image if produced
                    image_data_url = result.get_image_data_url()
                    if image_data_url:
                        # For tools that produce images, include image in result
                        tool_content: str | list[dict[str, Any]] = [
                            {"type": "text", "text": result.description},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]
                    else:
                        # Text-only result (e.g., web search)
                        tool_content = result.description
                        if result.error:
                            tool_content = f"{tool_content}\nError: {result.error}"
                        if formatted := result.metadata.get("formatted_results"):
                            tool_content = f"{tool_content}\n{formatted}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_content,
                        }
                    )

                # Create tool result turn for logging
                tool_turn = Turn(
                    role="tool_result",
                    content=self._format_tool_results(tool_results),
                    tool_results=tool_results,
                )
                turns.append(tool_turn)

                continue

            # Check if we should continue or if we have a final answer
            try:
                parsed = json.loads(response_text)

                # Check if the model wants to continue
                if parsed.get("continue", False):
                    # Continue the conversation
                    if turn_idx == self.max_turns - 1:
                        # Last turn reached, force final response
                        parsed["continue"] = False
                        final_response = parsed
                        logger.info("Max turns reached, accepting final response")
                        break

                    # On penultimate turn, inject a system reminder about turn limit.
                    # Note: OpenAI API requires alternating user/assistant messages,
                    # so we use a "user" role message. This is a limitation of the API.
                    if turn_idx == self.max_turns - 2:
                        turn_warning = (
                            f"[System: Turn {turn_idx + 2}/{self.max_turns} - "
                            f"Next response must be your final analysis with 'continue': false]"
                        )
                        messages.append({"role": "user", "content": turn_warning})
                    # Continue to next turn
                else:
                    # Model is done, this is the final response
                    final_response = parsed
                    break

            except json.JSONDecodeError as e:
                # Not a JSON response - fail immediately
                raise AgenticProcessingError(
                    f"Invalid JSON response on turn {turn_idx + 1}: {e}. "
                    f"Response: {response_text[:200]}...",
                    turns_completed=turn_idx + 1,
                ) from e

        # Calculate confidence based on analysis
        confidence = self._calculate_confidence(final_response, turns)

        return AgenticResult(
            final_response=final_response,
            turns=turns,
            retrieval_passages=retrieval_passages,
            total_tokens=total_tokens,
            confidence=confidence,
        )

    @beartype
    def _calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate overall confidence for the result.

        Confidence is based on:
        - Base confidence (CONFIDENCE_BASE = 0.5)
        - Response completeness: presence of caption, diagnosis, localization
          (weighted by CONFIDENCE_COMPLETENESS_WEIGHT = 0.3)
        - Tool usage: bonus for each turn with tool calls
          (CONFIDENCE_TOOL_BONUS = 0.1 per turn, capped at CONFIDENCE_TOOL_MAX = 0.2)

        Returns:
            Confidence score in range [0.5, 1.0]
        """
        confidence = self.CONFIDENCE_BASE

        # Response completeness for unified NOVA format
        has_caption = bool("caption" in response and response["caption"])
        has_diagnosis = bool("diagnosis" in response and response["diagnosis"])
        has_localization = bool("localization" in response and response["localization"])

        completeness_score = sum([has_caption, has_diagnosis, has_localization]) / 3.0
        confidence += completeness_score * self.CONFIDENCE_COMPLETENESS_WEIGHT

        # Multi-turn refinement bonus (tools used)
        tool_turns = sum(1 for t in turns if t.tool_calls)
        tool_usage_bonus = min(tool_turns * self.CONFIDENCE_TOOL_BONUS, self.CONFIDENCE_TOOL_MAX)
        confidence += tool_usage_bonus

        return min(1.0, confidence)

    @beartype
    def _format_tool_results(self, tool_results: list[ToolResult]) -> str:
        """Format tool results for conversation context."""
        if not tool_results:
            return "No tools were executed."

        result_strings = []
        for result in tool_results:
            if result.success:
                result_str = f"[OK] {result.tool_name}: {result.description}"
                if result.metadata:
                    result_str += f" (Details: {result.metadata})"
            else:
                result_str = f"[FAIL] {result.tool_name}: {result.error}"
            result_strings.append(result_str)

        return "\n".join(result_strings)

