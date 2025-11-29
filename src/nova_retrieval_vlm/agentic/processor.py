"""Agentic processor for medical image analysis.

Provides multi-turn analysis with tool calling, visual reasoning integration,
and retrieval augmentation capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt


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
    """Result of an agentic analysis session."""

    final_response: dict[str, Any]
    turns: list[Turn]
    retrieval_passages: list[str]
    total_tokens: int
    confidence: float


class AgenticProcessor:
    """Multi-turn agentic processor for medical image analysis.

    Integrates:
    - Visual reasoning (structure detection, symmetry analysis)
    - Tool calling (zoom, crop, contrast, threshold)
    - Retrieval augmentation (optional)
    - Multi-turn refinement
    """

    MAX_TURNS = 5
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_visual_reasoning: bool = True,
        use_tools: bool = True,
        max_turns: int = 5,
    ):
        """Initialize agentic processor.

        Args:
            model_name: Model to use for analysis (OpenRouter format)
            use_visual_reasoning: Whether to run visual analysis and inject into prompts
            use_tools: Whether to enable tool calling
            max_turns: Maximum number of turns before forcing completion
        """
        self.model_name = model_name
        self.use_visual_reasoning = use_visual_reasoning
        self.use_tools = use_tools
        self.max_turns = min(max_turns, self.MAX_TURNS)

        # Lazily initialized components
        self._model_adapter: OpenAIAdapter | None = None

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if self._model_adapter is None:
            self._model_adapter = OpenAIAdapter(model_name=self.model_name)

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
            task: Analysis task ('localization', 'diagnosis', 'caption')
            metadata: Optional metadata about the image/patient
            retrieval_passages: Optional retrieved context passages

        Returns:
            AgenticResult with final response and conversation history
        """
        # Ensure model adapter is initialized
        self._ensure_initialized()

        metadata = metadata or {}
        retrieval_passages = retrieval_passages or []

        # Initialize tool registry for this image
        tool_registry = ToolRegistry(image_path)

        # Build unified system prompt with optional tool flags
        from PIL import Image

        image = Image.open(image_path)
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

        # Multi-turn loop
        assert self._model_adapter is not None  # Type narrowing for mypy
        for turn_idx in range(self.max_turns):
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            # Generate response
            response_text, gen_log = await self._model_adapter.generate(
                image_path=image_path,
                passages=retrieval_passages,
                system_prompt=system_prompt if turn_idx == 0 else "",
                max_tokens=1024,
                temperature=0.0,
            )

            total_tokens += gen_log.total_tokens if gen_log else 0

            # Parse response for tool calls or final answer
            parsed = self._parse_response(response_text)

            turn = Turn(
                role="assistant",
                content=response_text,
                tool_calls=parsed.get("tool_calls", []),
            )
            turns.append(turn)

            # Check if model wants to use tools
            tool_calls = parsed.get("tool_calls", [])
            if tool_calls and self.use_tools and turn_idx < self.max_turns - 1:
                # Execute tools
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    result = tool_registry.execute(tool_name, **tool_args)
                    tool_results.append(result)

                # Add tool results to conversation
                tool_turn = Turn(
                    role="tool_result",
                    content=self._format_tool_results(tool_results),
                    tool_results=tool_results,
                )
                turns.append(tool_turn)

                # Update image path if we have a modified image
                # (for next turn's model call, we'd need to handle this differently)
                continue

            # Check if we have a final answer
            if "boxes" in parsed or "diagnosis" in parsed or "caption" in parsed:
                final_response = parsed
                break

            # Try to extract answer from raw response
            final_response = parsed

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

        # Response completeness
        has_answer = (
            ("boxes" in response and response["boxes"])
            or ("diagnosis" in response and response["diagnosis"])
            or ("caption" in response and response["caption"])
        )
        if has_answer:
            confidence += 0.2

        if "reasoning" in response and len(response.get("reasoning", "")) > 50:
            confidence += 0.1

        # Multi-turn refinement bonus
        if len(turns) > 1:
            confidence += 0.05 * min(len(turns) - 1, 3)

        return min(1.0, confidence)
