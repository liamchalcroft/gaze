"""Agentic processor for medical image analysis.

Provides clean, coherent multi-turn analysis with:
- Model-controlled continuation via 'continue' field
- Optional tool calling for visual analysis
- Optional reasoning capabilities
- Structured JSON output with caption, diagnosis, and localization
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from PIL import Image

from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA


@dataclass
class Turn:
    """Represents a single turn in the agentic conversation."""

    role: str  # 'user', 'assistant', 'tool_result'
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
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


class AgenticProcessor:
    """Multi-turn agentic processor for medical image analysis.

    Clean architecture with:
    - Model-controlled continuation via 'continue' field
    - Optional visual tools (zoom, crop, contrast, threshold)
    - Optional reasoning capabilities
    - Graceful turn limit handling with warnings
    - Structured JSON output for caption, diagnosis, localization
    """

    MAX_TURNS = 5
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_tools: bool = True,
        use_web_search: bool = False,
        max_turns: int = 5,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        enable_caching: bool = True,
    ):
        """Initialize agentic processor.

        Args:
            model_name: Model name for analysis
            use_tools: Enable visual tools (zoom, crop, contrast, threshold, flip, rotate)
            use_web_search: Enable web search for evidence-based analysis
            max_turns: Maximum allowed turns before forced completion
            reasoning_enabled: Enable model's internal reasoning capabilities
            reasoning_effort: Reasoning effort level ("high", "medium", "low")
            enable_caching: Enable prompt caching for performance
        """
        self.model_name = model_name
        self.use_tools = use_tools
        self.use_web_search = use_web_search
        self.max_turns = min(max_turns, self.MAX_TURNS)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.enable_caching = enable_caching

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
        _task: str,
        metadata: dict[str, Any] | None = None,
        retrieval_passages: list[str] | None = None,
    ) -> AgenticResult:
        """Run agentic analysis on a medical image.

        Args:
            image_path: Path to the medical image
            _task: Analysis task (always 'all_tasks' for unified analysis)
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
        tool_registry = ToolRegistry(image_path)

        # Build system prompt with capabilities flags
        from PIL import Image

        image = Image.open(image_path)

        # Use unified all_tasks template for consistent behavior
        system_prompt = create_enhanced_prompt(
            template_name="all_tasks.jinja",
            image_path=image_path,
            passages=retrieval_passages,
            metadata={
                **metadata,
                "width": image.width,
                "height": image.height,
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(image)}"
                        },
                    },
                ],
            },
        ]

        # Multi-turn agentic loop with proper OpenRouter tool calling
        assert self._model_adapter is not None  # Type narrowing for mypy
        for turn_idx in range(self.max_turns):
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            # Get tool schemas for this turn
            tool_schemas = tool_registry.get_tool_schemas() if self.use_tools else None

            # Use structured outputs for consistent JSON parsing
            response_format = NOVA_UNIFIED_SCHEMA

            # Generate response with tool calling support
            response_text, tool_calls, gen_log = await self._model_adapter.generate_chat(
                messages=messages,
                max_tokens=8192,
                temperature=0.0,
                tools=tool_schemas,
                response_format=response_format,  # Use structured outputs when no tools
            )

            total_tokens += gen_log.tokens if gen_log else 0

            # Create turn record
            turn = Turn(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls or [],
            )
            turns.append(turn)

            # Add assistant message to conversation for next turn
            messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

            # Check if model wants to use tools
            if tool_calls and self.use_tools and turn_idx < self.max_turns - 1:
                logger.info(f"Model requested {len(tool_calls)} tool calls")

                # Execute tools
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args_str = tool_call.get("arguments", "{}")

                    # Parse arguments from string (json imported at top of file)
                    tool_args = {}
                    if isinstance(tool_args_str, str):
                        try:
                            tool_args = json.loads(tool_args_str)
                        except ValueError:
                            tool_args = {}
                    else:
                        tool_args = tool_args_str

                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = tool_registry.execute(tool_name, **tool_args)
                    tool_results.append(result)

                # Add tool results to conversation
                for i, result in enumerate(tool_results):
                    tool_content = f"Tool {result.tool_name}: {result.description}"
                    if result.error:
                        tool_content += f" Error: {result.error}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[i].get("id", f"tool_{len(messages)}"),
                            "content": tool_content,
                        }
                    )

                # Create tool result turn
                tool_turn = Turn(
                    role="tool_result",
                    content=self._format_tool_results(tool_results),
                    tool_results=tool_results,
                )
                turns.append(tool_turn)

                # Update image if tools produced a new one
                if tool_results and any(r.image_base64 for r in tool_results):
                    latest_image_result = next(r for r in tool_results if r.image_base64)
                    # Update the image in conversation for next turn
                    if (
                        len(messages) > 1
                        and isinstance(messages[1], dict)
                        and "content" in messages[1]
                        and isinstance(messages[1]["content"], list)
                        and len(messages[1]["content"]) > 1
                        and isinstance(messages[1]["content"][1], dict)
                        and "image_url" in messages[1]["content"][1]
                        and isinstance(messages[1]["content"][1]["image_url"], dict)
                    ):
                        messages[1]["content"][1]["image_url"]["url"] = (
                            f"data:image/jpeg;base64,{latest_image_result.image_base64}"
                        )

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
                        break
                    elif turn_idx == self.max_turns - 2:
                        # Penultimate turn - add final turn warning to messages
                        final_warning = (
                            f"FINAL TURN WARNING: Turn {turn_idx + 1}/{self.max_turns}. "
                            f"Set 'continue': false and complete your analysis."
                        )
                        messages.append({"role": "assistant", "content": final_warning})
                    else:
                        # Continue to next turn normally
                        pass
                else:
                    # Model is done, this is the final response
                    final_response = parsed
                    break

            except json.JSONDecodeError:
                # Not a JSON response, continue for another turn
                if turn_idx == self.max_turns - 1:
                    # Last turn, create a fallback response
                    final_response = {
                        "caption": {"description": response_text, "confidence": 0.5},
                        "diagnosis": {
                            "primary_diagnosis": "Analysis incomplete",
                            "confidence": 0.3,
                        },
                        "localization": {"localizations": [], "confidence": 0.1},
                    }
                    break
                continue

        # Calculate confidence based on analysis
        confidence = self._calculate_confidence(final_response, turns)

        return AgenticResult(
            final_response=final_response,
            turns=turns,
            retrieval_passages=retrieval_passages,
            total_tokens=total_tokens,
            confidence=confidence,
        )

    def _calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate overall confidence for the result."""
        confidence = 0.5

        # Response completeness for unified NOVA format
        has_caption = bool("caption" in response and response["caption"])
        has_diagnosis = bool("diagnosis" in response and response["diagnosis"])
        has_localization = bool("localization" in response and response["localization"])

        completeness_score = sum([has_caption, has_diagnosis, has_localization]) / 3.0
        confidence += completeness_score * 0.3

        # Multi-turn refinement bonus (tools used)
        tool_usage_bonus = min(len([t for t in turns if t.tool_calls]) * 0.1, 0.2)
        confidence += tool_usage_bonus

        return min(1.0, confidence)

    def _format_tool_results(self, tool_results: list[ToolResult]) -> str:
        """Format tool results for conversation context."""
        if not tool_results:
            return "No tools were executed."

        result_strings = []
        for result in tool_results:
            if result.success:
                result_str = f"✓ {result.tool_name}: {result.description}"
                if result.metadata:
                    result_str += f" (Details: {result.metadata})"
            else:
                result_str = f"✗ {result.tool_name}: {result.error}"
            result_strings.append(result_str)

        return "\n".join(result_strings)

    def _image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to base64 JPEG string."""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
