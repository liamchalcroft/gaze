"""GEMeX-ThinkVG agentic processor for medical visual grounding.

Extends AgenticProcessorBase for chest X-ray analysis with:
- Visual reasoning chain generation (ThinkVG)
- Anatomical region grounding
- Bounding box localization

Supports verifiers integration for RL training via VerifiableProcessorMixin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from beartype import beartype

from gaze import AgenticProcessorBase
from gaze import ImageInput
from gaze import Turn
from gaze.models import AdapterProtocol
from gaze.verifiers import BaseRewardFunction
from gaze.verifiers import VerifiableProcessorMixin

from .rewards import GEMeXVerifiersReward
from .rewards import RewardWeights
from .schemas import GEMEX_SCHEMA
from .schemas import validate_gemex_response


class GEMeXProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    """Agentic processor for GEMeX-ThinkVG visual grounding task.

    Generates structured responses with:
    1. Chain-of-thought reasoning with region analysis
    2. Medical finding/diagnosis answer
    3. Anatomical location reference
    4. Bounding box coordinates for visual grounding

    Designed for RL fine-tuning with verifiable rewards on all three outputs.

    Example:
        processor = GEMeXProcessor(
            model_name="openai/gpt-4o",
            use_tools=True,
            use_web_search=True,
            max_turns=8,
        )
        result = await processor.analyze(
            images=Path("cxr_image.jpg"),
            metadata={
                "question": "What abnormalities are present?",
                "question_type": "open_ended",
            },
        )
        answer = result.final_response["answer"]
        bbox = result.final_response["location"]["bbox"]
    """

    DOMAIN = "Chest X-ray"
    MODALITY = "CXR"
    BODY_REGION = "Chest"

    # Image dimensions after GEMeX preprocessing
    IMAGE_SIZE = 336

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_tools: bool = True,
        use_web_search: bool = True,
        max_turns: int = 8,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
        reward_weights: RewardWeights | None = None,
        adapter_factory: Callable[[], AdapterProtocol] | None = None,
        max_encode_dimension: int | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            use_tools=use_tools,
            use_web_search=use_web_search,
            max_turns=max_turns,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
            adapter_factory=adapter_factory,
            max_encode_dimension=max_encode_dimension,
            seed=seed,
            max_tokens=max_tokens,
        )
        self._reward_weights = reward_weights

    def get_reward_function(self) -> BaseRewardFunction:
        """Return GEMeX reward function for verifiers integration.

        Returns:
            GEMeXVerifiersReward wrapping the combined answer/location/bbox rewards
        """
        return GEMeXVerifiersReward(weights=self._reward_weights)

    def get_system_prompt(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build GEMeX system prompt with ThinkVG instructions."""
        question_type = metadata.get("question_type", "open_ended")

        prompt_parts = [
            "You are an expert radiologist analyzing chest X-ray images.",
            "Your task is to answer medical questions with visual grounding.",
            "",
            "## Response Format",
            "",
            "You must provide:",
            "1. **Reasoning**: Detailed chain-of-thought analysis of the image",
            "   - Systematically examine relevant anatomical regions",
            "   - Describe what you observe in each region",
            "   - Reference specific coordinates [x1, y1, x2, y2] when discussing regions",
            "",
            "2. **Answer**: Clear, concise answer to the question",
            "",
            "3. **Location**: Visual grounding with:",
            "   - `reference`: Anatomical region name (e.g., 'right lower lobe', 'bilateral lung')",
            "   - `bbox`: Bounding box [x1, y1, x2, y2] in pixel coordinates",
            "",
            f"Image dimensions: {self.IMAGE_SIZE} x {self.IMAGE_SIZE} pixels",
            "Coordinates should be integers in range [0, 336].",
            "",
        ]

        # Add question-type specific guidance
        if question_type == "closed_ended":
            prompt_parts.extend(
                [
                    "## Question Type: Closed-ended (Yes/No)",
                    "Answer with 'Yes' or 'No' based on visual evidence.",
                    "",
                ]
            )
        elif question_type == "single_choice":
            prompt_parts.extend(
                [
                    "## Question Type: Single Choice",
                    "Select the single best answer from the provided options.",
                    "",
                ]
            )
        elif question_type == "multi_choice":
            prompt_parts.extend(
                [
                    "## Question Type: Multiple Choice",
                    "Select all correct answers from the provided options.",
                    "",
                ]
            )
        else:  # open_ended
            prompt_parts.extend(
                [
                    "## Question Type: Open-ended",
                    "Provide a descriptive answer based on your analysis.",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Analysis Guidelines",
                "1. Start by examining the overall image quality and orientation",
                "2. Systematically scan: mediastinum, bilateral lungs, costophrenic angles",
                "3. Identify any abnormalities: consolidation, effusion, nodules, etc.",
                "4. Provide precise bounding box around the most relevant finding",
                "5. Base your answer ONLY on visible evidence",
                "",
            ]
        )

        return "\n".join(prompt_parts)

    def get_user_message(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build GEMeX user message with question."""
        question = metadata.get("question", "")
        question_type = metadata.get("question_type", "open_ended")

        message_parts = []

        if images:
            message_parts.append("Analyze this chest X-ray image.")
            message_parts.append("")

        message_parts.append(f"**Question ({question_type}):** {question}")
        message_parts.append("")

        # Add any provided options for choice questions
        if options := metadata.get("options"):
            message_parts.append("**Options:**")
            for i, opt in enumerate(options):
                message_parts.append(f"  {chr(65 + i)}. {opt}")
            message_parts.append("")

        message_parts.append("Provide your reasoning, answer, and visual grounding location.")

        return "\n".join(message_parts)

    def get_response_schema(self) -> dict[str, Any] | None:
        """Return GEMeX schema for structured outputs."""
        return GEMEX_SCHEMA

    @beartype
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate response has required GEMeX fields."""
        return validate_gemex_response(response)

    def calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate confidence based on response quality and tool usage."""
        # Base confidence from model
        base_confidence = response.get("confidence", 0.5)

        # Bonus for thorough reasoning
        reasoning = response.get("reasoning", "")
        reasoning_bonus = 0.0
        if len(reasoning) > 200:
            reasoning_bonus += 0.05
        if len(reasoning) > 500:
            reasoning_bonus += 0.05

        # Bonus for using visual tools
        visual_tools = {
            "zoom",
            "crop",
            "adjust_contrast",
            "adjust_brightness",
            "adjust_sharpness",
            "threshold",
            "window_level",
            "equalize_histogram",
            "adaptive_equalize",
            "detect_edges",
            "get_intensity_stats",
            "measure",
            "show_grid",
            "symmetry_diff",
            "annotate_region",
            "intensity_profile",
            "denoise",
            "morphological",
            "invert",
        }
        tool_turns = sum(
            1 for t in turns if t.tool_calls and any(tc.name in visual_tools for tc in t.tool_calls)
        )
        tool_bonus = min(tool_turns * 0.05, 0.15)

        # Bonus for valid bbox
        location = response.get("location", {})
        bbox = location.get("bbox", [])
        bbox_bonus = 0.0
        if len(bbox) == 4 and all(0 <= x <= self.IMAGE_SIZE for x in bbox):
            # Valid bbox coordinates
            bbox_bonus = 0.05
            # Additional bonus for reasonable bbox size
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area_ratio = (width * height) / (self.IMAGE_SIZE**2)
                if 0.01 < area_ratio < 0.8:  # Reasonable size
                    bbox_bonus += 0.05

        return min(1.0, base_confidence + reasoning_bonus + tool_bonus + bbox_bonus)
