"""VQA-RAD JSON schemas for structured outputs.

Defines the response schema for visual question answering on radiology images.
"""

from __future__ import annotations

from typing import Any

VQA_RAD_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "vqa_rad_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the visual question",
                },
                "answer_type": {
                    "type": "string",
                    "enum": ["closed", "open"],
                    "description": "Whether the answer is closed (yes/no) or open-ended",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the answer (0.0-1.0)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Visual reasoning explaining how the answer was derived",
                },
                "image_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key observations from the image supporting the answer",
                },
                "region_of_interest": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the relevant region",
                        },
                        "location": {
                            "type": "string",
                            "description": "Anatomical or spatial location",
                        },
                    },
                    "required": ["description", "location"],
                    "description": "Region of the image most relevant to the answer",
                },
                "continue": {
                    "type": "boolean",
                    "description": "true if more analysis needed, false when complete",
                },
            },
            "required": ["answer", "answer_type", "confidence", "reasoning", "continue"],
            "additionalProperties": False,
        },
    },
}


def validate_vqa_rad_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required VQA-RAD fields.

    Args:
        response: Parsed JSON response

    Returns:
        True if all required fields present and valid
    """
    required = ["answer", "answer_type", "confidence", "reasoning", "continue"]
    if not all(field in response for field in required):
        return False

    # Validate answer_type
    if response.get("answer_type") not in ("closed", "open"):
        return False

    # Validate confidence range
    confidence = response.get("confidence")
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        return False

    # Validate answer is not empty
    return bool(response.get("answer"))
