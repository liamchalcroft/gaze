"""PubmedQA agentic processor for biomedical question answering.

Extends AgenticProcessorBase for text-only Q&A with optional web search.
Demonstrates the harness's text-only analysis capability.

Supports verifiers integration for RL training via VerifiableProcessorMixin.
"""

from __future__ import annotations

from typing import Any

from beartype import beartype

from radiant_harness import AgenticProcessorBase
from radiant_harness import ImageInput
from radiant_harness import Turn
from radiant_harness.utils import extract_json_from_text
from radiant_harness.verifiers import BaseRewardFunction
from radiant_harness.verifiers import VerifiableProcessorMixin
from radiant_harness.verifiers import extract_completion_text

from .schemas import PUBMEDQA_SCHEMA
from .schemas import validate_pubmedqa_response


class PubmedQAVerifiersReward(BaseRewardFunction):
    """Verifiers-compatible reward function for PubmedQA.

    Computes exact match reward for yes/no/maybe answers.
    """

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute reward for PubmedQA answer.

        Args:
            prompt: The input prompt (unused)
            completion: Model completion (string or message list)
            info: Case information with ground truth

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Extract completion text
        comp_text = self._extract_text(completion)

        # Parse response
        response = self._extract_json_response(comp_text)
        if response is None:
            return 0.0

        # Get predicted and gold answers
        pred_answer = str(response.get("answer", "")).lower().strip()
        gold_answer = str(info.get("answer", info.get("gold_answer", ""))).lower().strip()

        # Normalize common variations
        pred_normalized = self._normalize_answer(pred_answer)
        gold_normalized = self._normalize_answer(gold_answer)

        return 1.0 if pred_normalized == gold_normalized else 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize PubmedQA answer."""
        answer = answer.lower().strip()
        # Handle common variations
        if answer in {"yes", "y", "true", "positive"}:
            return "yes"
        if answer in {"no", "n", "false", "negative"}:
            return "no"
        if answer in {"maybe", "uncertain", "unclear", "unknown"}:
            return "maybe"
        return answer

    def _extract_text(self, completion: Any) -> str:
        """Extract text from completion."""
        return extract_completion_text(completion)

    def _extract_json_response(self, text: str) -> dict[str, Any] | None:
        """Extract JSON response from text."""
        result = extract_json_from_text(text)
        if result is not None:
            return result

        # Fallback: check for direct yes/no/maybe
        text_lower = text.lower()
        for answer in ["yes", "no", "maybe"]:
            if answer in text_lower:
                return {"answer": answer}

        return None


class PubmedQAProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    """Agentic processor for PubmedQA biomedical question answering.

    This processor handles text-only analysis (no images) and demonstrates
    using web search tools to supplement context with additional literature.

    Example:
        processor = PubmedQAProcessor(
            model_name="openai/gpt-4o",
            use_web_search=True,
            max_turns=5,
        )
        result = await processor.analyze(
            images=None,  # Text-only
            metadata={
                "question": "Is X effective for Y?",
                "context": ["Background...", "Methods...", "Results..."],
                "meshes": ["Term1", "Term2"],
            },
        )
        print(result.final_response["answer"])  # "yes", "no", or "maybe"
    """

    DOMAIN = "Biomedical Literature"

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_web_search: bool = True,
        max_turns: int = 5,
        reasoning_enabled: bool = False,
        reasoning_effort: str = "high",
    ) -> None:
        """Initialize PubmedQA processor.

        Args:
            model_name: Model to use for analysis
            use_web_search: Enable PubMed search for additional context
            max_turns: Maximum conversation turns
            reasoning_enabled: Enable model reasoning mode
            reasoning_effort: Reasoning effort level
        """
        super().__init__(
            model_name=model_name,
            use_tools=False,  # No visual tools for text-only
            use_web_search=use_web_search,
            max_turns=max_turns,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
        )

    def get_reward_function(self) -> BaseRewardFunction:
        """Return PubmedQA reward function for verifiers integration.

        Returns:
            PubmedQAVerifiersReward for exact match on yes/no/maybe
        """
        return PubmedQAVerifiersReward()

    def get_system_prompt(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build PubmedQA system prompt."""
        _ = images  # Text-only task

        prompt_parts = [
            "You are an expert biomedical researcher analyzing PubMed abstracts.",
            "Your task is to answer yes/no/maybe questions based on scientific evidence.",
            "",
            "Guidelines:",
            "- Answer 'yes' if the evidence clearly supports an affirmative answer",
            "- Answer 'no' if the evidence clearly contradicts or negates the claim",
            "- Answer 'maybe' if the evidence is inconclusive, mixed, or insufficient",
            "- Base your answer ONLY on the provided context and any search results",
            "- Provide clear reasoning linking evidence to your conclusion",
            "- Extract key phrases that directly support your answer",
            "",
        ]

        if self.use_web_search:
            prompt_parts.extend([
                "You have access to PubMed search to find additional supporting evidence.",
                "Use search_web tool if the provided context is insufficient.",
                "",
            ])

        # Add MeSH terms if available
        meshes = metadata.get("meshes", [])
        if meshes:
            prompt_parts.append(f"Relevant MeSH terms: {', '.join(meshes[:10])}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def get_user_message(
        self,
        images: list[ImageInput],
        metadata: dict[str, Any],
    ) -> str:
        """Build PubmedQA user message with question and context."""
        _ = images  # Text-only task

        question = metadata.get("question", "")
        context_sections = metadata.get("context", [])
        labels = metadata.get("labels", [])

        message_parts = [f"**Question:** {question}", ""]

        if context_sections:
            message_parts.append("**Abstract Context:**")
            for i, section in enumerate(context_sections):
                label = labels[i] if i < len(labels) else f"Section {i + 1}"
                message_parts.append(f"\n[{label}]\n{section}")
            message_parts.append("")

        message_parts.append(
            "Based on this context, answer the question with 'yes', 'no', or 'maybe'. "
            "Provide your reasoning and identify key supporting evidence."
        )

        return "\n".join(message_parts)

    def get_response_schema(self) -> dict[str, Any] | None:
        """Return PubmedQA schema for structured outputs."""
        return PUBMEDQA_SCHEMA

    @beartype
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate response has required PubmedQA fields."""
        return validate_pubmedqa_response(response)

    def calculate_confidence(
        self,
        response: dict[str, Any],
        turns: list[Turn],
    ) -> float:
        """Calculate confidence based on response and search usage."""
        # Use the model's self-reported confidence as base
        base_confidence = response.get("confidence", 0.5)

        # Bonus for using search tools (shows thorough analysis)
        search_turns = sum(
            1 for t in turns
            if t.tool_calls and any(tc.name == "search_web" for tc in t.tool_calls)
        )
        search_bonus = min(search_turns * 0.05, 0.1)

        # Bonus for providing key evidence
        evidence = response.get("key_evidence", [])
        evidence_bonus = min(len(evidence) * 0.02, 0.1)

        return min(1.0, base_confidence + search_bonus + evidence_bonus)
