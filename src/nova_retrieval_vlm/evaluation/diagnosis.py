from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from loguru import logger


def gpt4o_semantic_match(pred: str, ref: str, model_name: str = "openai/gpt-4o") -> bool:
    """
    Use GPT-4o to perform semantic matching between prediction and reference diagnosis.

    This follows the official NOVA evaluation protocol for diagnosis task:
    "GPT-4o is used to perform semantic matching between predictions and ground truth labels"
    """
    try:
        from nova_retrieval_vlm.models import get_model_client

        # Create the semantic matching prompt
        prompt = f"""You are a medical expert evaluating diagnostic predictions.

Your task is to determine if two diagnostic labels refer to the same medical condition,
even if expressed differently.

Consider these diagnostically equivalent:
- Different terminology for the same condition (e.g., "heart attack" = "myocardial infarction")
- Abbreviations vs full terms (e.g., "MI" = "myocardial infarction")
- Different word orders for the same diagnosis
- Synonymous medical terms
- Different levels of specificity that refer to the same core condition

PREDICTION: "{pred}"
REFERENCE: "{ref}"

Respond with ONLY "YES" if they refer to the same medical condition,
or "NO" if they refer to different conditions.
"""

        # Get model client and make the comparison
        import asyncio

        client = get_model_client(model_name)

        # Use the adapter's generate_text method for text-only requests
        async def make_request():
            response, _ = await client.generate_text(
                prompt_text=prompt,
                system_prompt="You are a medical expert. Respond only with YES or NO.",
            )
            return response

        # Run async function in sync context
        result = asyncio.run(make_request()).strip().upper()
        return result == "YES"

    except Exception as e:
        logger.error(f"Semantic matching failed for '{pred}' vs '{ref}': {e}")
        raise ValueError(f"Model evaluation failed: {e}") from e


def evaluate_diagnosis_nova_official(
    preds: Sequence[Any | list[Any]],
    refs: Sequence[Any],
    use_gpt4o_matching: bool = True,
    model_name: str = "openai/gpt-4o",
) -> dict[str, float]:
    """
    Official NOVA diagnosis evaluation using GPT-4o semantic matching.

    This implements the exact protocol described in the NOVA paper:
    "GPT-4o is used to perform semantic matching between predictions and ground truth labels"

    Args:
        preds: List of predicted diagnosis or list of predictions (for top-5).
        refs: List of reference diagnoses.
        use_gpt4o_matching: Whether to use LLM semantic matching (default: True).
        model_name: Model to use for semantic matching.

    Returns:
        Dictionary with keys 'top1', 'top5', 'coverage', 'entropy'.
    """
    n = len(refs)
    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "coverage": 0.0, "entropy": 0.0}

    # Track semantic and exact matches
    top1_count = 0
    top5_count = 0
    all_preds = []

    for i, (p, r) in enumerate(zip(preds, refs, strict=False)):
        if i % 10 == 0:  # Progress indicator
            pass

        if isinstance(p, list):
            # Handle list predictions (top-5)
            top1_pred = p[0] if p else None

            # Top-1 evaluation
            if top1_pred:
                if use_gpt4o_matching:
                    if gpt4o_semantic_match(str(top1_pred), str(r), model_name):
                        top1_count += 1
                elif str(top1_pred).strip().lower() == str(r).strip().lower():
                    top1_count += 1

            # Top-5 evaluation
            top5_match = False
            for pred in p:
                if use_gpt4o_matching:
                    if gpt4o_semantic_match(str(pred), str(r), model_name):
                        top5_match = True
                        break
                elif str(pred).strip().lower() == str(r).strip().lower():
                    top5_match = True
                    break

            if top5_match:
                top5_count += 1

            all_preds.extend(p)
        else:
            # Handle single predictions
            if use_gpt4o_matching:
                if gpt4o_semantic_match(str(p), str(r), model_name):
                    top1_count += 1
                    top5_count += 1
            elif str(p).strip().lower() == str(r).strip().lower():
                top1_count += 1
                top5_count += 1

            all_preds.append(p)

    # Calculate metrics
    results = {
        "top1": top1_count / n,
        "top5": top5_count / n,
    }

    # Coverage: unique predictions vs unique references
    uniq_preds = len({str(p).strip().lower() for p in all_preds})
    uniq_refs = len({str(r).strip().lower() for r in refs})
    results["coverage"] = uniq_preds / uniq_refs if uniq_refs > 0 else 0.0

    # Entropy of prediction distribution
    pred_counts = Counter(str(p).strip().lower() for p in all_preds)
    entropy = 0.0
    total_preds = len(all_preds)
    for count in pred_counts.values():
        p_i = count / total_preds
        entropy -= p_i * math.log(p_i + 1e-12, 2)

    results["entropy"] = entropy

    return results
