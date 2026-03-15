"""VQA-RAD agentic processor for visual question answering.

Extends AgenticProcessorBase for radiology VQA with visual tools and web search.

Supports verifiers integration for RL training via VerifiableProcessorMixin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import Literal

from beartype import beartype

from radiant_harness import AgenticProcessorBase
from radiant_harness import ImageInput
from radiant_harness import Turn
from radiant_harness.models import AdapterProtocol
from radiant_harness.utils import extract_json_from_text
from radiant_harness.verifiers import BaseRewardFunction
from radiant_harness.verifiers import VerifiableProcessorMixin
from radiant_harness.verifiers import extract_completion_text

from .evaluation import normalize_answer
from .evaluation import normalize_binary
from .schemas import VQA_RAD_SCHEMA
from .schemas import validate_vqa_rad_response

VQARadQuestionType = Literal["closed", "open"]


class VQARadVerifiersReward(BaseRewardFunction):
    """Verifiers-compatible reward function for VQA-RAD.

    Uses different reward strategies based on question type:
    - Closed (yes/no): Exact match reward
    - Open: Token F1 reward
    """

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute reward for VQA-RAD answer.

        Args:
            prompt: The input prompt (unused)
            completion: Model completion (string or message list)
            info: Case information with ground truth

        Returns:
            Reward in [0.0, 1.0] based on question type
        """
        # Extract completion text
        comp_text = self._extract_text(completion)

        # Parse response
        response = self._extract_json_response(comp_text)
        if response is None:
            return 0.0

        # Get predicted and gold answers
        pred_answer = str(response.get("answer", ""))
        gold_answer = str(info.get("answer", info.get("gold_answer", "")))

        # Determine question type — dataset uses "answer_type" key
        question_type = info.get("answer_type", info.get("question_type", "open"))

        if question_type == "closed":
            return self._compute_closed_reward(pred_answer, gold_answer)
        return self._compute_open_reward(pred_answer, gold_answer)

    def _compute_closed_reward(self, prediction: str, reference: str) -> float:
        """Compute exact match reward for closed questions."""
        pred_norm = normalize_binary(prediction)
        ref_norm = normalize_binary(reference)

        if pred_norm is None or ref_norm is None:
            return 0.0

        return 1.0 if pred_norm == ref_norm else 0.0

    def _compute_open_reward(self, prediction: str, reference: str) -> float:
        """Compute token F1 reward for open questions.

        Uses the same normalize_answer() as evaluation to ensure the RL
        reward signal is consistent with reported metrics.
        """
        if not prediction or not reference:
            return 0.0

        pred_tokens = set(normalize_answer(prediction).split())
        ref_tokens = set(normalize_answer(reference).split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        intersection = pred_tokens & ref_tokens
        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)

        return 2 * precision * recall / (precision + recall)

    def _extract_text(self, completion: Any) -> str:
        """Extract text from completion."""
        return extract_completion_text(completion)

    def _extract_json_response(self, text: str) -> dict[str, Any] | None:
        """Extract JSON response from text.

        Returns None when no valid JSON is found, so malformed completions
        receive a 0.0 reward rather than being silently accepted.
        """
        return extract_json_from_text(text)


class VQARadProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    """Agentic processor for VQA-RAD visual question answering.

    Handles radiology image analysis with visual tools (zoom, crop, contrast)
    and optional web search for medical reference.

    Example:
        processor = VQARadProcessor(
            model_name="openai/gpt-4o",
            use_tools=True,
            use_web_search=True,
            max_turns=5,
        )
        result = await processor.analyze(
            images=Path("xray.jpg"),
            metadata={"question": "Is there a fracture?"},
        )
        print(result.final_response["answer"])  # "yes" or detailed answer
    """

    DOMAIN = "Radiology VQA"
    MODALITY = "Multi-modality"  # X-ray, CT, MRI, etc.

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_tools: bool = True,
        use_web_search: bool = False,
        max_turns: int = 5,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
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

    def get_reward_function(self) -> BaseRewardFunction:
        """Return VQA-RAD reward function for verifiers integration.

        Returns:
            VQARadVerifiersReward for closed/open question handling
        """
        return VQARadVerifiersReward()

    def get_system_prompt(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],  # noqa: ARG002
    ) -> str:
        """Build VQA-RAD system prompt."""
        prompt_parts = [
            "You are an expert radiologist answering questions about medical images.",
            "Analyze the provided radiology image and answer the question accurately.",
            "",
            "Guidelines:",
            "- For yes/no questions, answer clearly with 'yes' or 'no'",
            "- For open-ended questions, provide concise, specific answers",
            "- Base your answer ONLY on what you observe in the image",
            "- Describe the visual evidence supporting your answer",
            "- Identify the anatomical region relevant to the question",
            "",
        ]

        # Add image dimensions if available
        if images:
            img = images[0]
            prompt_parts.append(f"Image dimensions: {img.width} x {img.height} pixels")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def get_user_message(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build VQA-RAD user message with question."""
        question = metadata.get("question", "")

        message_parts = []

        if images:
            message_parts.append("Here is the radiology image for analysis.")
            message_parts.append("")

        answer_type = metadata.get("answer_type", "")
        if answer_type:
            message_parts.append(f"**Question type:** {answer_type}")

        message_parts.append(f"**Question:** {question}")
        message_parts.append("")
        message_parts.append(
            "Analyze the image carefully and provide your answer. "
            "Include your visual reasoning and identify the relevant region."
        )

        return "\n".join(message_parts)

    def get_response_schema(self) -> dict[str, Any] | None:
        """Return VQA-RAD schema for structured outputs."""
        return VQA_RAD_SCHEMA

    @beartype
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate response has required VQA-RAD fields."""
        return validate_vqa_rad_response(response)

    def calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate confidence based on response and tool usage."""
        # Use the model's self-reported confidence as base
        base_confidence = response.get("confidence", 0.5)

        # Bonus for using visual tools (shows thorough examination)
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

        # Bonus for providing image observations
        observations = response.get("image_observations", [])
        obs_bonus = min(len(observations) * 0.02, 0.1)

        # Bonus for identifying region of interest
        roi_bonus = 0.05 if response.get("region_of_interest") else 0.0

        return min(1.0, base_confidence + tool_bonus + obs_bonus + roi_bonus)
