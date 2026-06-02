"""PubmedQA JSON schemas for structured outputs.

Defines the response schema for yes/no/maybe question answering,
plus canonical answer normalization shared by evaluation and reward code.
"""

from __future__ import annotations

import re
from typing import Any

from gaze.utils import clamp_confidence
from gaze.utils import coerce_json_types

_YES_ALIASES = {"yes", "y", "true", "positive"}
_NO_ALIASES = {"no", "n", "false", "negative"}
_MAYBE_ALIASES = {"maybe", "uncertain", "unclear", "unknown"}


def normalize_pubmedqa_answer(answer: str) -> str:
    """Normalize a PubmedQA answer to canonical form.

    Maps common variations to yes/no/maybe. Also extracts the answer
    word from sentence-form responses that local models sometimes produce
    (e.g. "Yes, based on the evidence..." → "yes").

    This is the single source of truth used by both evaluation metrics
    and the RL reward function.
    """
    answer = answer.lower().strip().rstrip(".,;:")
    if answer in _YES_ALIASES:
        return "yes"
    if answer in _NO_ALIASES:
        return "no"
    if answer in _MAYBE_ALIASES:
        return "maybe"
    # Try extracting a leading yes/no/maybe from longer text
    m = re.match(r"^(yes|no|maybe)\b", answer)
    if m:
        return m.group(1)
    return answer


PUBMEDQA_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "pubmedqa_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["yes", "no", "maybe"],
                    "description": "The answer to the research question",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the answer (0.0-1.0)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of how the answer was derived from the context",
                },
                "key_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key phrases from the context supporting the answer",
                },
                "continue": {
                    "type": "boolean",
                    "description": "true if more analysis needed, false when complete",
                },
            },
            "required": ["answer", "confidence", "reasoning", "key_evidence", "continue"],
            "additionalProperties": False,
        },
    },
}


def validate_pubmedqa_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required PubmedQA fields."""
    required = ["answer", "confidence", "reasoning", "key_evidence", "continue"]
    if not all(field in response for field in required):
        return False

    coerce_json_types(response, PUBMEDQA_SCHEMA)

    # Normalize answer to canonical form (local models may return "Yes" etc.)
    raw_answer = str(response.get("answer", "")).lower().strip()
    normalized = normalize_pubmedqa_answer(raw_answer)
    if normalized not in ("yes", "no", "maybe"):
        return False
    response["answer"] = normalized

    clamped = clamp_confidence(response.get("confidence"))
    if clamped is None:
        return False
    response["confidence"] = clamped

    return isinstance(response.get("continue"), bool)
