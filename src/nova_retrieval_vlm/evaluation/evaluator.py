from __future__ import annotations

"""Light-weight fallback evaluation utilities.

This module provides a very simple text-based evaluator that is **dependency-free**
and therefore guaranteed to work in constrained environments where heavy NLP
metric packages (e.g. *sacrebleu*, *bert-score*) might not be available.
It is **not** intended to replace the richer task-specific evaluation logic
implemented elsewhere in the code-base, but rather to offer a default that
prevents runtime import errors when a more advanced evaluator is not
supplied.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Any

import numpy as np

__all__ = ["EvaluationResult", "Evaluator"]


@dataclass
class EvaluationResult:
    """Container for basic evaluation outputs."""

    score: float
    details: Dict[str, Any]


class Evaluator:  # pylint: disable=too-few-public-methods
    """Extremely simple baseline evaluator.

    The implementation relies on *difflib.SequenceMatcher* to compute a token
    level similarity ratio between *prediction* and *reference* strings.  The
    resulting value is returned as a **score** in the interval \[0, 1\].  For
    structured tasks such as *localization* this evaluator obviously cannot
    capture spatial accuracy - callers are expected to provide a specialised
    evaluator instead.  Nevertheless, this minimal version keeps the overall
    pipeline functional.
    """

    def evaluate_prediction(self, prediction: str, reference: str, task: str = "general") -> Dict[str, float]:
        """Return a single float score for *prediction* against *reference*.

        Parameters
        ----------
        prediction: str
            Model output.
        reference: str
            Ground-truth answer.
        task: str, default "general"
            Task descriptor. Currently ignored but kept for API compatibility.
        """
        if not reference:
            # If we do not have a ground-truth reference we cannot meaningfully
            # evaluate - default to 0.0 so that downstream aggregation works.
            return {"score": 0.0}

        # *SequenceMatcher* operates at character level.  To obtain a slightly
        # more robust estimate we perform a whitespace split first and feed the
        # token lists to the matcher.
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        ratio = SequenceMatcher(None, pred_tokens, ref_tokens).ratio()
        # Guard against possible NaN (should not happen but keep defensive).
        ratio = float(ratio) if not np.isnan(ratio) else 0.0

        return {"score": ratio} 