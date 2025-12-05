"""Agentic diagnosis processor.

Enhanced diagnosis with tool calling, retrieval augmentation,
and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.processors.base import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import DiagnosisMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticDiagnosisProcessor(BaseProcessor):
    """Agentic processor for diagnosis tasks.

    Extends base diagnosis with:
    - Tool calling for interactive analysis (zoom, crop, contrast, flip, rotate)
    - Retrieval augmentation (medical guidelines, similar cases)
    - Multi-turn refinement for differential diagnosis
    """

    @beartype
    def __init__(
        self,
        config: ProcessorConfig,
        use_tools: bool = True,
        max_turns: int = 10,
    ) -> None:
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

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=True)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Run agentic analysis - model can search web independently
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                metadata=metadata,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    @beartype
    def _convert_result(
        self,
        result: AgenticResult,
        image_path: str | Path,
        batch_idx: int,
        sample_idx: int,
    ) -> ModelResponse:
        """Convert AgenticResult to ModelResponse.

        Raises:
            KeyError: If required 'diagnosis' field is missing from response.
        """
        response = result.final_response

        # Validate required field - fail fast on missing diagnosis
        if "diagnosis" not in response:
            raise KeyError("Missing required 'diagnosis' in agentic response")

        diagnosis = response["diagnosis"]
        # Optional fields - documented in NOVA agentic schema:
        # - confidence: Model-provided confidence, falls back to AgenticResult.confidence
        # - findings: List of clinical findings, defaults to empty
        # - differential: List of differential diagnoses for top-5 eval, defaults to empty
        # - reasoning: Chain-of-thought reasoning text, defaults to empty
        confidence = response.get("confidence", result.confidence)
        findings: list[str] = response.get("findings", [])
        differential: list[str] = response.get("differential", [])
        reasoning: str = response.get("reasoning", "")

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

    @beartype
    def _log_analysis(self, result: AgenticResult, sample_idx: int) -> None:
        """Log analysis details for debugging."""
        # Diagnosis is guaranteed to exist - validated in _convert_result
        diagnosis = result.final_response["diagnosis"]
        diagnosis_preview = diagnosis[:50] if len(diagnosis) > 50 else diagnosis
        tool_count = sum(len(t.tool_calls) for t in result.turns)
        logger.info(
            f"Sample {sample_idx}: "
            f"diagnosis='{diagnosis_preview}...', "
            f"confidence={result.confidence:.2f}, "
            f"turns={len(result.turns)}, "
            f"tools={tool_count}, "
            f"tokens={result.total_tokens}"
        )

        web_searches = sum(
            1 for t in result.turns if any(tc["name"] == "search_web" for tc in t.tool_calls if "name" in tc)
        )
        if web_searches > 0:
            logger.debug(f"  Web searches: {web_searches} performed")

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> DiagnosisMetrics:
        """Evaluate diagnosis responses using NOVA protocol.

        Raises:
            ValueError: If JSON parsing fails for any response.
            json.JSONDecodeError: If ground truth JSON is malformed.
        """
        # Extract predicted diagnoses
        predicted_diagnoses = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from model: {response.text}") from e

            if "diagnosis" not in parsed:
                raise ValueError(f"Missing 'diagnosis' in parsed response: {parsed}")

            diagnosis = parsed["diagnosis"]
            # differential is optional - used for top-5 evaluation
            differential = parsed.get("differential", [])

            # Include differential diagnoses for top-5 evaluation
            all_diagnoses = [diagnosis] + differential if differential else [diagnosis]
            predicted_diagnoses.append(all_diagnoses)

        # Parse ground truth - fail fast on malformed data
        ground_truth_diagnoses = []
        for gt in ground_truth:
            gt_parsed = json.loads(gt)
            if "diagnosis" not in gt_parsed:
                raise ValueError(f"Missing 'diagnosis' in ground truth: {gt}")
            ground_truth_diagnoses.append(gt_parsed["diagnosis"])

        # Use NOVA official evaluation (with LLM semantic matching per NOVA protocol)
        results = evaluate_diagnosis_nova_official(
            preds=predicted_diagnoses,
            refs=ground_truth_diagnoses,
        )

        return DiagnosisMetrics(
            top1=results["top1"],
            top5=results["top5"],
            coverage=results["coverage"],
            entropy=results["entropy"],
        )
