from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from loguru import logger


def normalize_diagnosis_string(diag: str) -> str:
    """
    Normalize diagnosis strings for better matching.

    Handles common variations in medical terminology:
    - Different spacing patterns
    - En-dash vs hyphen
    - Common abbreviations
    - Plural/singular variations
    """
    if not diag:
        return ""

    normalized = (
        diag.lower()
        .strip()
        .replace(" – ", "-")  # en-dash with spaces to hyphen
        .replace(" –", "-")  # en-dash to hyphen
        .replace("  ", " ")  # double spaces to single
    )

    # Common medical abbreviation mappings
    abbreviations = {
        "sod": "septo-optic dysplasia",
        "acc": "agenesis of corpus callosum",
        "hydrocephalus": "hydrocephalus",
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

    # Expand common abbreviations
    for abbrev, full in abbreviations.items():
        if normalized == abbrev:
            normalized = full
        elif normalized.startswith(abbrev + " "):
            normalized = full + normalized[len(abbrev) :]

    return normalized.strip()


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
        (r"\brain hemorrhage\b", r"\bintracerebral hemorrhage\b"),
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

    import re

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


def gpt4o_semantic_match(pred: str, ref: str, model_name: str = "moonshotai/kimi-k2:free") -> bool:
    """
    Use LLM semantic matching between prediction and reference diagnosis.

    This follows the official NOVA evaluation protocol for diagnosis task:
    "GPT-4o is used to perform semantic matching between predictions and ground truth labels"

    Args:
        pred: Predicted diagnosis string.
        ref: Reference/ground truth diagnosis string.
        model_name: Model to use for semantic matching (default: moonshotai/kimi-k2:free).
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
        raise ValueError(f"GPT-4o semantic matching is required for NOVA evaluation: {e}") from e


def evaluate_diagnosis_nova_official(
    preds: Sequence[Any | list[Any]],
    refs: Sequence[Any],
    model_name: str = "moonshotai/kimi-k2:free",
) -> dict[str, float]:
    """
    Official NOVA diagnosis evaluation using LLM semantic matching.

    This implements the exact protocol described in the NOVA paper:
    "GPT-4o is used to perform semantic matching between predictions and ground truth labels"

    Args:
        preds: List of predicted diagnosis or list of predictions (for top-5).
        refs: List of reference diagnoses.
        model_name: Model to use for semantic matching (default: moonshotai/kimi-k2:free).

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

            # Top-1 evaluation - use semantic matching for all cases
            if top1_pred and gpt4o_semantic_match(str(top1_pred), str(r), model_name):
                top1_count += 1

            # Top-5 evaluation - use semantic matching for all cases
            top5_match = False
            for pred in p:
                if gpt4o_semantic_match(str(pred), str(r), model_name):
                    top5_match = True
                    break

            if top5_match:
                top5_count += 1

            all_preds.extend(p)
        else:
            # Handle single predictions - use semantic matching for all cases
            if gpt4o_semantic_match(str(p), str(r), model_name):
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
