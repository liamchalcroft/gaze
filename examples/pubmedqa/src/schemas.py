"""PubmedQA JSON schemas for structured outputs.

Defines the response schema for yes/no/maybe question answering,
plus canonical answer normalization shared by evaluation and reward code.
"""

from __future__ import annotations

from typing import Any


def normalize_pubmedqa_answer(answer: str) -> str:
    """Normalize a PubmedQA answer to canonical form.

    Maps common variations to yes/no/maybe. This is the single source
    of truth used by both evaluation metrics and the RL reward function.
    """
    answer = answer.lower().strip()
    if answer in {"yes", "y", "true", "positive"}:
        return "yes"
    if answer in {"no", "n", "false", "negative"}:
        return "no"
    if answer in {"maybe", "uncertain", "unclear", "unknown"}:
        return "maybe"
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
                    "minimum": 0.0,
                    "maximum": 1.0,
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
    """Validate that a response has required PubmedQA fields.

    Args:
        response: Parsed JSON response

    Returns:
        True if all required fields present and valid
    """
    required = ["answer", "confidence", "reasoning", "key_evidence", "continue"]
    if not all(field in response for field in required):
        return False

    # Validate answer is one of the allowed values
    if response.get("answer") not in ("yes", "no", "maybe"):
        return False

    # Validate confidence is in valid range
    confidence = response.get("confidence")
    return isinstance(confidence, int | float) and 0 <= confidence <= 1
