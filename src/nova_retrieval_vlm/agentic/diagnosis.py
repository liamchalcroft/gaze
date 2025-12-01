"""Agentic diagnosis processor.

Enhanced diagnosis with tool calling, retrieval augmentation,
and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticDiagnosisProcessor(BaseProcessor):
    """Agentic processor for diagnosis tasks.

    Extends base diagnosis with:
    - Tool calling for interactive analysis (zoom, crop, contrast, flip, rotate)
    - Retrieval augmentation (medical guidelines, similar cases)
    - Multi-turn refinement for differential diagnosis
    - Visual comparison with retrieved examples (planned)
    """

    def __init__(
        self,
        config: Any,
        use_tools: bool = True,
        max_turns: int = 10,
    ):
        """Initialize agentic diagnosis processor.

        Args:
            config: ProcessorConfig with model_name, output_dir, etc.
            use_tools: Enable tool calling (zoom, crop, web search, etc.)
            max_turns: Max turns for multi-turn refinement (default: 10)
        """
        super().__init__(config)
        self.use_tools = use_tools
        self.max_turns = max_turns

        self._agentic_processor = AgenticProcessor(
            model_name=config.model_name,
            use_tools=use_tools,
            use_web_search=use_tools,  # Diagnosis benefits from web search when tools are enabled
            max_turns=max_turns,
            reasoning_enabled=config.reasoning_enabled,
            reasoning_effort=config.reasoning_effort,
            enable_caching=config.enable_caching,
        )

        logger.info(
            f"Initialized AgenticDiagnosisProcessor with {max_turns} max turns, "
            f"tools={'enabled' if use_tools else 'disabled'}"
        )

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process diagnosis batch with fully agentic analysis."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Run agentic analysis - model can search web independently
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                _task="diagnosis",
                metadata=metadata,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    # NOTE: Retrieval method removed - model uses search_web tool instead

    def _convert_result(
        self,
        result: AgenticResult,
        image_path: Path,
        batch_idx: int,
        sample_idx: int,
    ) -> ModelResponse:
        """Convert AgenticResult to ModelResponse."""
        response = result.final_response

        # Extract diagnosis information
        diagnosis = response.get("diagnosis", "")
        confidence = response.get("confidence", result.confidence)
        findings = response.get("findings", [])
        differential = response.get("differential", [])
        reasoning = response.get("reasoning", "")

        # Build structured response text
        response_text = {
            "diagnosis": diagnosis,
            "findings": findings,
            "differential": differential,
        }

        return ModelResponse(
            text=json.dumps(response_text),
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "image_path": str(image_path),
                "diagnosis": diagnosis,
                "num_findings": len(findings),
                "num_differential": len(differential),
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "num_turns": len(result.turns),
                "total_tokens": result.total_tokens,
                "tool_calls": sum(len(t.tool_calls) for t in result.turns),
                "web_searches_used": sum(
                    1
                    for t in result.turns
                    if any(tc.get("name") == "search_web" for tc in t.tool_calls)
                ),
            },
        )

    def _log_analysis(self, result: AgenticResult, sample_idx: int) -> None:
        """Log analysis details for debugging."""
        diagnosis = result.final_response.get("diagnosis", "N/A")
        tool_count = sum(len(t.tool_calls) for t in result.turns)
        logger.info(
            f"Sample {sample_idx}: "
            f"diagnosis='{diagnosis[:50]}...', "
            f"confidence={result.confidence:.2f}, "
            f"turns={len(result.turns)}, "
            f"tools={tool_count}, "
            f"tokens={result.total_tokens}"
        )

        web_searches = sum(
            1 for t in result.turns if any(tc.get("name") == "search_web" for tc in t.tool_calls)
        )
        if web_searches > 0:
            logger.debug(f"  Web searches: {web_searches} performed")

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> EvaluationMetrics:
        """Evaluate diagnosis responses using NOVA protocol."""
        # Extract predicted diagnoses
        predicted_diagnoses = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
                diagnosis = parsed.get("diagnosis", "")
                differential = parsed.get("differential", [])

                # Include differential diagnoses for top-5 evaluation
                all_diagnoses = [diagnosis] + differential if differential else [diagnosis]
                predicted_diagnoses.append(all_diagnoses)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from model: {response.text}") from e

        # Parse ground truth
        ground_truth_diagnoses = []
        for gt in ground_truth:
            gt_parsed = json.loads(gt)
            diagnosis = gt_parsed.get("diagnosis", gt)
            ground_truth_diagnoses.append(diagnosis)

        # Use NOVA official evaluation (with GPT-4o semantic matching)
        # Note: For testing, we use exact matching to avoid API costs
        results = evaluate_diagnosis_nova_official(
            preds=predicted_diagnoses,
            refs=ground_truth_diagnoses,
            use_gpt4o_matching=False,  # Use exact matching by default
        )

        return EvaluationMetrics(
            accuracy=results.get("top1", 0.0),
            precision=results.get("top5"),  # Top-5 as secondary metric
            recall=results.get("coverage"),
            f1_score=None,
            auc_roc=results.get("entropy"),
        )
