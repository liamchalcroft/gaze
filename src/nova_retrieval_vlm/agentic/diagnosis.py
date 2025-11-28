"""Agentic diagnosis processor.

Enhanced diagnosis with visual reasoning integration,
retrieval augmentation, and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.agentic.retrieval_manager import RetrievalManager
from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticDiagnosisProcessor(BaseProcessor):
    """Agentic processor for diagnosis tasks.

    Extends base diagnosis with:
    - Pre-analysis visual reasoning (structure detection, symmetry)
    - Retrieval augmentation (medical guidelines, similar cases)
    - Multi-turn refinement for differential diagnosis
    - Confidence calibration based on findings
    """

    def __init__(
        self,
        config: Any,
        use_visual_reasoning: bool = True,
        use_tools: bool = True,
        max_turns: int = 3,
        index_dir: Path | str | None = None,
    ):
        """Initialize agentic diagnosis processor.

        Args:
            config: ProcessorConfig with model_name, output_dir, etc.
            use_visual_reasoning: Enable visual pre-analysis
            use_tools: Enable tool calling (zoom, crop, etc.)
            max_turns: Max turns for multi-turn refinement
            index_dir: Directory containing retrieval indexes
        """
        super().__init__(config)
        self.use_visual_reasoning = use_visual_reasoning
        self.use_tools = use_tools
        self.max_turns = max_turns

        self._agentic_processor = AgenticProcessor(
            model_name=config.model_name,
            use_visual_reasoning=use_visual_reasoning,
            use_tools=use_tools,
            max_turns=max_turns,
        )

        # Initialize retrieval manager if enabled
        self._retrieval_manager = None
        if config.use_retrieval and index_dir:
            self._retrieval_manager = RetrievalManager(
                index_dir=index_dir,
                retrieval_type=config.retrieval_type,
                top_k=5,
            )

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process diagnosis batch with agentic analysis."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Get retrieval passages if configured
            retrieval_passages = self._get_retrieval_passages(metadata)

            # Run agentic analysis
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                task="diagnosis",
                metadata=metadata,
                retrieval_passages=retrieval_passages,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    def _get_retrieval_passages(
        self,
        metadata: dict[str, Any],
        visual_analysis: Any = None,
    ) -> list[str]:
        """Get retrieval passages for diagnosis task.

        Args:
            metadata: Image metadata containing patient info, modality, etc.
            visual_analysis: Optional visual analysis results

        Returns:
            List of relevant medical guidelines/cases for context.
        """
        if not self.config.use_retrieval or self._retrieval_manager is None:
            return []

        return self._retrieval_manager.retrieve_for_task(
            task="diagnosis",
            metadata=metadata,
            visual_analysis=visual_analysis,
        )

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

        # Add visual analysis context to reasoning if available
        if result.visual_analysis:
            va = result.visual_analysis
            reasoning += f"\n\nVisual pre-analysis: {va.overall_assessment}"
            reasoning += f" (symmetry: {va.symmetry_analysis.symmetry_score:.2f})"

            # Note any detected abnormalities
            abnormal = [f for f in va.visual_features if f.feature_type == "abnormal"]
            if abnormal:
                reasoning += f"\nDetected abnormalities: {len(abnormal)}"

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
                "used_visual_reasoning": result.visual_analysis is not None,
                "retrieval_passages_used": len(result.retrieval_passages),
            },
        )

    def _log_analysis(self, result: AgenticResult, sample_idx: int) -> None:
        """Log analysis details for debugging."""
        diagnosis = result.final_response.get("diagnosis", "N/A")
        logger.info(
            f"Sample {sample_idx}: "
            f"diagnosis='{diagnosis[:50]}...', "
            f"confidence={result.confidence:.2f}, "
            f"turns={len(result.turns)}, "
            f"tokens={result.total_tokens}"
        )

        if result.visual_analysis:
            va = result.visual_analysis
            logger.debug(
                f"  Visual analysis: "
                f"symmetry={va.symmetry_analysis.symmetry_score:.2f}, "
                f"features={len(va.visual_features)}"
            )

        if result.retrieval_passages:
            logger.debug(f"  Retrieval: {len(result.retrieval_passages)} passages used")

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
            except json.JSONDecodeError:
                # Try to extract raw text as diagnosis
                predicted_diagnoses.append([response.text.strip()])

        # Parse ground truth
        ground_truth_diagnoses = []
        for gt in ground_truth:
            try:
                gt_parsed = json.loads(gt)
                diagnosis = gt_parsed.get("diagnosis", gt)
            except json.JSONDecodeError:
                diagnosis = gt
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
