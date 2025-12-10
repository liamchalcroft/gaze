from __future__ import annotations

import asyncio
import math

# Default model for semantic matching - cost-efficient SOTA model via OpenRouter
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from beartype import beartype
from loguru import logger

DEFAULT_SEMANTIC_MATCH_MODEL = os.getenv(
    "NOVA_SEMANTIC_MATCH_MODEL",
    "x-ai/grok-4.1-fast:free"
)

# Pre-compiled regex patterns for better performance
_DASH_PATTERN = re.compile(r"\s*–\s*")
_DOUBLE_SPACE_PATTERN = re.compile(r"  +")

# Common medical abbreviation mappings - use frozenset for faster lookups
_ABBREVIATION_MAPPING = {
    "sod": "septo-optic dysplasia",
    "acc": "agenesis of corpus callosum",
    "cpa": "cerebellopontine angle",
    "avm": "arteriovenous malformation",
    "pnet": "primitive neuroectodermal tumor",
    "gbm": "glioblastoma multiforme",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "dwi": "diffusion weighted imaging",
    "flair": "fluid attenuated inversion recovery",
    "dc": "dermoid cyst",
    "ec": "epidermoid cyst",
    "ac": "arachnoid cyst",
    "cm": "cavernous malformation",
    "vs": "vestibular schwannoma",
    "an": "acoustic neuroma",
    "da": "diffuse axonal injury",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
}


@beartype
def normalize_diagnosis_string(diag: str) -> str:
    """
    Normalize diagnosis strings for better matching.

    Handles common variations in medical terminology:
    - Different spacing patterns
    - En-dash vs hyphen
    - Common abbreviations
    - Plural/singular variations

    Optimized with pre-compiled regex patterns for better performance.
    """
    if not diag:
        return ""

    # Use pre-compiled regex patterns for better performance
    normalized = diag.lower().strip()
    normalized = _DASH_PATTERN.sub("-", normalized)
    normalized = _DOUBLE_SPACE_PATTERN.sub(" ", normalized)

    # Expand common abbreviations using the pre-defined mapping
    for abbrev, full in _ABBREVIATION_MAPPING.items():
        if normalized == abbrev:
            normalized = full
        elif normalized.startswith(abbrev + " "):
            normalized = full + normalized[len(abbrev) :]

    return normalized.strip()


@beartype
def exact_diagnosis_match(pred: str, ref: str) -> bool:
    """
    Perform enhanced exact matching for medical diagnoses.

    Returns True if diagnoses are the same after normalization and semantic equivalence.
    """
    pred_norm = normalize_diagnosis_string(pred)
    ref_norm = normalize_diagnosis_string(ref)

    # Direct exact match
    if pred_norm == ref_norm:
        return True

    # Semantic equivalence patterns for medical terminology
    semantic_patterns = [
        # Tumor equivalents
        (r"\bglioblastoma\b", r"\bglioblastoma multiforme\b"),
        (r"\bmedulloblastoma\b", r"\bpnet\b"),  # Primitive neuroectodermal tumor
        (r"\bacoustic neuroma\b", r"\bvestibular schwannoma\b"),
        (r"\bcavernoma\b", r"\bcavernous malformation\b"),
        (r"\bcavernous malformation\b", r"\bcavernoma\b"),
        # Hydrocephalus patterns
        (r"\bhydrocephalus\b.*\babnormalities\b", r"\bhydrocephalus\b"),
        (r"\babnormal.*hydrocephalus\b", r"\bhydrocephalus\b"),
        (r"\bcommunicating hydrocephalus\b", r"\bhydrocephalus\b"),
        (r"\bobstructive hydrocephalus\b", r"\bhydrocephalus\b"),
        # Developmental anomalies
        (r"\bagenesis.*corpus callosum\b", r"\bacc\b"),
        (r"\bcorpus callosum.*agenesis\b", r"\bacc\b"),
        (r"\bsepto-optic dysplasia\b", r"\bsod\b"),
        # Vascular conditions
        (r"\bcerebral infarction\b", r"\bstroke\b"),
        (r"\bischemic stroke\b", r"\bcerebral infarction\b"),
        (r"\bbrain hemorrhage\b", r"\bintracerebral hemorrhage\b"),
        (r"\bsubarachnoid hemorrhage\b", r"\bsah\b"),
        # Cyst patterns
        (r"\barachnoid cyst\b", r"\bcyst\b"),
        (r"\bepidermoid\b.*\bcyst\b", r"\bepidermoid cyst\b"),
        (r"\bdermoid\b.*\bcyst\b", r"\bdermoid cyst\b"),
        # Inflammation/infection
        (r"\bencephalitis\b", r"\bbrain inflammation\b"),
        (r"\bmeningitis\b", r"\bbrain inflammation\b"),
        # Trauma
        (r"\bcontusion\b", r"\bbrain injury\b"),
        (r"\bshearing injury\b", r"\bdiffuse axonal injury\b"),
    ]

    # Check semantic patterns
    for pred_pattern, ref_pattern in semantic_patterns:
        if re.search(pred_pattern, pred_norm, re.IGNORECASE) and re.search(
            ref_pattern, ref_norm, re.IGNORECASE
        ):
            return True
        if re.search(ref_pattern, pred_norm, re.IGNORECASE) and re.search(
            pred_pattern, ref_norm, re.IGNORECASE
        ):
            return True

    # Check if one contains the other (subset relationship)
    if pred_norm in ref_norm or ref_norm in pred_norm:
        # Only count as match if the shorter term is at least 3 words long
        shorter = pred_norm if len(pred_norm) < len(ref_norm) else ref_norm
        if len(shorter.split()) >= 3:
            return True

    return False


@beartype
async def llm_semantic_match_async(
    pred: str, ref: str, model_name: str = DEFAULT_SEMANTIC_MATCH_MODEL
) -> bool:
    """
    Use LLM semantic matching between prediction and reference diagnosis (async).

    This follows the NOVA evaluation protocol for diagnosis task which uses LLM-based
    semantic matching. The default uses a cost-efficient SOTA model via OpenRouter,
    but any capable model (e.g., GPT-4o, Claude) can be substituted.

    Args:
        pred: Predicted diagnosis string.
        ref: Reference/ground truth diagnosis string.
        model_name: Model to use for semantic matching. Default uses the best
            available free model on OpenRouter for cost efficiency.

    Raises:
        ValueError: If LLM API call fails (semantic matching is required for NOVA evaluation).
    """
    from ..models import get_model_client

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

    client = get_model_client(model_name)
    response, _ = await client.generate_text(
        prompt_text=prompt,
        system_prompt="You are a medical expert. Respond only with YES or NO.",
    )
    return response.strip().upper() == "YES"


@beartype
def llm_semantic_match(pred: str, ref: str, model_name: str = DEFAULT_SEMANTIC_MATCH_MODEL) -> bool:
    """
    Synchronous wrapper for LLM semantic matching.

    Note: This function cannot be called from within an async context.
    Use llm_semantic_match_async directly when in async context.

    Raises:
        RuntimeError: If called from within an async context
    """
    # Check if we're in an async context without string-matching exceptions
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop - safe to use asyncio.run()
        loop = None

    if loop is not None:
        raise RuntimeError(
            "llm_semantic_match cannot be called from an async context. "
            "Use llm_semantic_match_async instead."
        )

    return asyncio.run(llm_semantic_match_async(pred, ref, model_name))


@beartype
async def evaluate_diagnosis_nova_official(
    preds: Sequence[Any | list[Any]],
    refs: Sequence[Any],
    model_name: str = DEFAULT_SEMANTIC_MATCH_MODEL,
) -> dict[str, float]:
    """
    Official NOVA diagnosis evaluation using LLM semantic matching (async).

    This implements the NOVA evaluation protocol which uses LLM-based semantic
    matching between predictions and ground truth labels. The default model
    uses a cost-efficient SOTA model via OpenRouter.

    Args:
        preds: List of predicted diagnosis or list of predictions (for top-5).
        refs: List of reference diagnoses.
        model_name: Model to use for semantic matching. Default uses the best
            available free model on OpenRouter for cost efficiency.

    Returns:
        Dictionary with keys 'top1', 'top5', 'coverage', 'entropy'.

    Raises:
        ValueError: If preds and refs have different lengths.
    """
    n = len(refs)
    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "coverage": 0.0, "entropy": 0.0}

    if len(preds) != n:
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {n}")

    # Track semantic and exact matches
    top1_count = 0
    top5_count = 0
    all_preds: list[Any] = []

    for i, (p, r) in enumerate(zip(preds, refs, strict=True)):
        if i % 10 == 0 and i > 0:
            logger.debug(f"Processed {i}/{n} diagnosis comparisons")

        if isinstance(p, list):
            # Handle list predictions (top-5)
            top1_pred = p[0] if p else None

            # Top-1 evaluation - use fast exact match first, then LLM semantic matching
            if top1_pred and (
                exact_diagnosis_match(str(top1_pred), str(r))
                or await llm_semantic_match_async(str(top1_pred), str(r), model_name)
            ):
                top1_count += 1

            # Top-5 evaluation - use fast exact match first, then LLM for remaining
            top5_match = False
            for pred in p:
                if exact_diagnosis_match(str(pred), str(r)):
                    top5_match = True
                    break
                if await llm_semantic_match_async(str(pred), str(r), model_name):
                    top5_match = True
                    break

            if top5_match:
                top5_count += 1

            all_preds.extend(p)
        else:
            # Handle single predictions - use fast exact match first, then LLM
            if exact_diagnosis_match(str(p), str(r)) or await llm_semantic_match_async(str(p), str(r), model_name):
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

    # Entropy of prediction distribution (Shannon entropy in bits)
    pred_counts = Counter(str(p).strip().lower() for p in all_preds)
    entropy = 0.0
    total_preds = len(all_preds)
    for count in pred_counts.values():
        # p_i is always > 0 since count >= 1, no epsilon needed
        p_i = count / total_preds
        entropy -= p_i * math.log2(p_i)

    results["entropy"] = entropy

    return results
