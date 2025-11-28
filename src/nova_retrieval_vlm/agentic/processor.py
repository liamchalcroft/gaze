"""Agentic processor for medical image analysis.

Provides multi-turn analysis with tool calling, visual reasoning integration,
and retrieval augmentation capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.types import parse_json_response
from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import BilateralSymmetryAnalyzer
from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import BrainStructureDetector
from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import MedicalReasoningChain
from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import RadiologyAnalysis
from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import VisualFeature


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
    visual_analysis: RadiologyAnalysis | None
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
        self._structure_detector: BrainStructureDetector | None = None
        self._symmetry_analyzer: BilateralSymmetryAnalyzer | None = None
        self._reasoning_chain: MedicalReasoningChain | None = None

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if self._model_adapter is None:
            self._model_adapter = OpenAIAdapter(model_name=self.model_name)
        if self._structure_detector is None:
            self._structure_detector = BrainStructureDetector()
        if self._symmetry_analyzer is None:
            self._symmetry_analyzer = BilateralSymmetryAnalyzer()
        if self._reasoning_chain is None:
            self._reasoning_chain = MedicalReasoningChain()

    @beartype
    async def analyze(
        self,
        image_path: Path,
        task: str,
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

        # Run visual reasoning if enabled
        visual_analysis = None
        if self.use_visual_reasoning:
            visual_analysis = self._run_visual_analysis(image_path)

        # Build initial prompt with visual analysis context
        system_prompt = self._build_system_prompt(task, visual_analysis, retrieval_passages)

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
        confidence = self._calculate_confidence(final_response, visual_analysis, turns)

        return AgenticResult(
            final_response=final_response,
            turns=turns,
            visual_analysis=visual_analysis,
            retrieval_passages=retrieval_passages,
            total_tokens=total_tokens,
            confidence=confidence,
        )

    def _run_visual_analysis(self, image_path: Path) -> RadiologyAnalysis | None:
        """Run visual reasoning pipeline on the image."""
        # Ensure visual reasoning components are initialized
        self._ensure_initialized()

        try:
            # Load image as numpy array
            image = cv2.imread(str(image_path))
            if image is None:  # type: ignore[reportUnnecessaryComparison]
                logger.warning(f"Could not load image: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_array = image_rgb.astype(np.float32) / 255.0

            # Type assertions for mypy
            assert self._structure_detector is not None
            assert self._symmetry_analyzer is not None
            assert self._reasoning_chain is not None

            # Detect structures
            features = self._structure_detector.detect_structures(image_array)

            # Analyze symmetry
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            symmetry = self._symmetry_analyzer.analyze_symmetry(gray)

            # Generate reasoning chain
            reasoning_steps = self._reasoning_chain.generate_reasoning(
                features, symmetry, image_array
            )

            # Build overall assessment
            assessment = self._build_assessment(features, symmetry)

            return RadiologyAnalysis(
                visual_features=features,
                symmetry_analysis=symmetry,
                reasoning_chain=reasoning_steps,
                overall_assessment=assessment,
                confidence_score=self._compute_analysis_confidence(features, symmetry),
                recommended_followup=[],
            )

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return None

    def _build_system_prompt(
        self,
        task: str,
        visual_analysis: RadiologyAnalysis | None,
        retrieval_passages: list[str],
    ) -> str:
        """Build system prompt with visual analysis context."""
        base_prompts = {
            "localization": """You are an expert radiologist analyzing medical images.
Your task is to identify and localize abnormalities or regions of interest.

Provide bounding boxes in [x1, y1, x2, y2] pixel format.
Include reasoning for each detection.""",
            "diagnosis": """You are an expert radiologist providing diagnostic assessments.
Analyze the image and provide a primary diagnosis with supporting observations.

Consider differential diagnoses and confidence levels.""",
            "caption": """You are an expert radiologist generating detailed image descriptions.
Describe the imaging modality, anatomical structures, and any findings.

Use precise medical terminology.""",
        }

        prompt = base_prompts.get(task, base_prompts["localization"])

        # Add visual analysis context
        if visual_analysis:
            prompt += "\n\n## Pre-computed Visual Analysis\n"
            prompt += f"Symmetry Score: {visual_analysis.symmetry_analysis.symmetry_score:.2f}\n"
            prompt += (
                f"Midline Shift: {visual_analysis.symmetry_analysis.midline_shift:.1f} pixels\n"
            )

            if visual_analysis.visual_features:
                prompt += "\nDetected Structures:\n"
                for feat in visual_analysis.visual_features[:5]:  # Limit to top 5
                    prompt += f"- {feat.name} at {feat.location}: {feat.description} "
                    prompt += f"(confidence: {feat.confidence:.2f}, type: {feat.feature_type})\n"

            if visual_analysis.reasoning_chain:
                prompt += "\nPreliminary Reasoning:\n"
                for step in visual_analysis.reasoning_chain[:3]:  # Limit to top 3
                    prompt += f"{step.step_number}. {step.observation}\n"

            prompt += f"\nOverall Assessment: {visual_analysis.overall_assessment}\n"

        # Add retrieval context
        if retrieval_passages:
            prompt += "\n\n## Retrieved Medical Context\n"
            for i, passage in enumerate(retrieval_passages[:3], 1):
                prompt += f"{i}. {passage[:500]}...\n"

        # Add tool instructions if enabled
        if self.use_tools:
            prompt += """

## Available Tools
You can request visual tools to examine the image more closely:
- zoom(factor): Zoom into the image (factor 0.5-4.0)
- crop(box): Crop region [x1, y1, x2, y2] in normalized 0-1 coords
- adjust_contrast(factor): Enhance contrast (factor 0.5-3.0)
- threshold(lower, upper): Apply intensity threshold (0-255)
- reset(): Reset to original image

To use a tool, include in your response:
```json
{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}
```

After examining, provide your final answer."""

        # Add output format instructions
        prompt += f"""

## Response Format
Respond with JSON containing:
{self._get_output_format(task)}
"""
        return prompt

    def _get_output_format(self, task: str) -> str:
        """Get expected output format for task."""
        formats = {
            "localization": """{
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["label1", ...],
    "reasoning": "explanation"
}""",
            "diagnosis": """{
    "diagnosis": "primary diagnosis",
    "confidence": 0.0-1.0,
    "findings": ["finding1", ...],
    "differential": ["alt1", ...],
    "reasoning": "explanation"
}""",
            "caption": """{
    "caption": "detailed description",
    "modality": "imaging modality",
    "structures": ["structure1", ...],
    "findings": ["finding1", ...]
}""",
        }
        return formats.get(task, formats["localization"])

    def _build_user_prompt(self, task: str, metadata: dict[str, Any]) -> str:
        """Build user prompt with metadata context."""
        prompt = f"Analyze this medical image for {task}.\n"

        if metadata:
            if "patient_info" in metadata:
                prompt += f"Patient info: {metadata['patient_info']}\n"
            if "modality" in metadata:
                prompt += f"Modality: {metadata['modality']}\n"
            if "clinical_history" in metadata:
                prompt += f"Clinical history: {metadata['clinical_history']}\n"

        return prompt

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse model response for JSON content and tool calls."""
        try:
            return parse_json_response(response_text)
        except Exception:
            # Try to find JSON in response
            import re

            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_response": response_text}

    def _format_tool_results(self, results: list[ToolResult]) -> str:
        """Format tool results for conversation context."""
        lines = ["Tool execution results:"]
        for result in results:
            status = "SUCCESS" if result.success else "FAILED"
            lines.append(f"- {result.tool_name}: {status} - {result.description}")
            if result.error:
                lines.append(f"  Error: {result.error}")
        return "\n".join(lines)

    def _build_assessment(self, features: list[VisualFeature], symmetry: Any) -> str:
        """Build overall assessment from visual analysis."""
        abnormal = [f for f in features if f.feature_type == "abnormal"]
        uncertain = [f for f in features if f.feature_type == "uncertain"]

        if abnormal:
            return f"Abnormal findings detected: {len(abnormal)} regions of concern"
        if uncertain:
            return f"Uncertain findings: {len(uncertain)} regions require attention"
        if symmetry.symmetry_score < 0.85:
            return f"Asymmetry detected (score: {symmetry.symmetry_score:.2f})"
        return "No obvious abnormalities detected"

    def _compute_analysis_confidence(self, features: list[VisualFeature], symmetry: Any) -> float:
        """Compute confidence score from visual analysis."""
        if not features:
            return 0.5

        confidences = [f.confidence for f in features]
        avg_confidence = sum(confidences) / len(confidences)

        # Penalize low symmetry
        if symmetry.symmetry_score < 0.7:
            avg_confidence *= 0.9

        return min(1.0, max(0.0, avg_confidence))

    def _calculate_confidence(
        self,
        response: dict[str, Any],
        visual_analysis: RadiologyAnalysis | None,
        turns: list[Turn],
    ) -> float:
        """Calculate overall confidence for the result."""
        confidence = 0.5

        # Response completeness
        if "boxes" in response and response["boxes"]:
            confidence += 0.2
        if "reasoning" in response and len(response.get("reasoning", "")) > 50:
            confidence += 0.1

        # Visual analysis agreement
        if visual_analysis and visual_analysis.confidence_score > 0.7:
            confidence += 0.1

        # Multi-turn refinement bonus
        if len(turns) > 1:
            confidence += 0.05 * min(len(turns) - 1, 3)

        return min(1.0, confidence)
