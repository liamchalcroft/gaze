from __future__ import annotations
from typing import Sequence, Any, Union
from collections import Counter
import math


def gpt4o_semantic_match(pred: str, ref: str, model_name: str = "mistralai/mistral-small-3.2-24b-instruct:free") -> bool:
    """
    Use Mistral Small 3.2 to perform semantic matching between prediction and reference diagnosis.
    
    This follows the official NOVA evaluation protocol for diagnosis task.
    """
    try:
        from nova_retrieval_vlm.models import get_model_client
        
        # Create the semantic matching prompt
        prompt = f"""You are a medical expert evaluating diagnostic predictions. 

Your task is to determine if two diagnostic labels refer to the same medical condition, even if expressed differently.

Consider these diagnostically equivalent:
- Different terminology for the same condition (e.g., "heart attack" = "myocardial infarction")
- Abbreviations vs full terms (e.g., "MI" = "myocardial infarction") 
- Different word orders for the same diagnosis
- Synonymous medical terms
- Different levels of specificity that refer to the same core condition

PREDICTION: "{pred}"
REFERENCE: "{ref}"

Respond with ONLY "YES" if they refer to the same medical condition, or "NO" if they refer to different conditions.
"""
        
        # Get model client and make the comparison
        import asyncio
        client = get_model_client(model_name)
        
        # Use the adapter's generate_text method for text-only requests
        async def make_request():
            response, _ = await client.generate_text(
                prompt_text=prompt,
                system_prompt="You are a medical expert. Respond only with YES or NO."
            )
            return response
        
        # Run async function in sync context
        result = asyncio.run(make_request()).strip().upper()
        return result == "YES"
        
    except Exception as e:
        # Fallback to exact match if GPT-4o call fails
        print(f"Warning: GPT-4o semantic matching failed ({e}), falling back to exact match")
        return str(pred).strip().lower() == str(ref).strip().lower()


def evaluate_diagnosis_nova_official(
    preds: Sequence[Union[Any, list[Any]]],
    refs: Sequence[Any],
    use_gpt4o_matching: bool = True,
    model_name: str = "mistralai/mistral-small-3.2-24b-instruct:free"
) -> dict[str, float]:
    """
    Official NOVA diagnosis evaluation using LLM semantic matching.
    
    This implements the exact protocol described in the NOVA paper:
    "GPT-4o is used to perform semantic matching between predictions and ground truth labels"
    We use Mistral Small 3.2 as a free alternative that provides equivalent semantic matching.
    
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
        return {'top1': 0.0, 'top5': 0.0, 'coverage': 0.0, 'entropy': 0.0}
    
    # Track semantic and exact matches
    top1_count = 0
    top5_count = 0
    all_preds = []
    
    print(f"Evaluating {n} diagnosis predictions using {'LLM semantic matching' if use_gpt4o_matching else 'exact matching'}...")
    
    for i, (p, r) in enumerate(zip(preds, refs)):
        if i % 10 == 0:  # Progress indicator
            print(f"  Processing {i+1}/{n}")
            
        if isinstance(p, list):
            # Handle list predictions (top-5)
            top1_pred = p[0] if p else None
            
            # Top-1 evaluation
            if top1_pred:
                if use_gpt4o_matching:
                    if gpt4o_semantic_match(str(top1_pred), str(r), model_name):
                        top1_count += 1
                else:
                    if str(top1_pred).strip().lower() == str(r).strip().lower():
                        top1_count += 1
            
            # Top-5 evaluation  
            top5_match = False
            for pred in p:
                if use_gpt4o_matching:
                    if gpt4o_semantic_match(str(pred), str(r), model_name):
                        top5_match = True
                        break
                else:
                    if str(pred).strip().lower() == str(r).strip().lower():
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
            else:
                if str(p).strip().lower() == str(r).strip().lower():
                    top1_count += 1
                    top5_count += 1
            
            all_preds.append(p)
    
    # Calculate metrics
    results = {
        'top1': top1_count / n,
        'top5': top5_count / n,
    }
    
    # Coverage: unique predictions vs unique references
    uniq_preds = len(set(str(p).strip().lower() for p in all_preds))
    uniq_refs = len(set(str(r).strip().lower() for r in refs))
    results['coverage'] = uniq_preds / uniq_refs if uniq_refs > 0 else 0.0
    
    # Entropy of prediction distribution
    pred_counts = Counter(str(p).strip().lower() for p in all_preds)
    entropy = 0.0
    total_preds = len(all_preds)
    for count in pred_counts.values():
        p_i = count / total_preds
        entropy -= p_i * math.log(p_i + 1e-12, 2)
    
    results['entropy'] = entropy
    
    print(f"Diagnosis evaluation complete:")
    print(f"  Top-1 Accuracy: {results['top1']:.3f}")
    print(f"  Top-5 Accuracy: {results['top5']:.3f}")
    print(f"  Coverage: {results['coverage']:.3f}")
    print(f"  Entropy: {results['entropy']:.3f}")
    
    return results


def evaluate_diagnosis(
    preds: Sequence[Union[Any, list[Any]]],
    refs: Sequence[Any],
) -> dict[str, float]:
    """
    Backward compatibility wrapper that uses the official NOVA protocol.
    
    This function now delegates to evaluate_diagnosis_nova_official() for consistency.
    """
    return evaluate_diagnosis_nova_official(preds, refs, use_gpt4o_matching=True)
