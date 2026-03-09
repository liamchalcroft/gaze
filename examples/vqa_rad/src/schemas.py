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
                    "additionalProperties": False,
                    "description": "Region of the image most relevant to the answer",
                },
                "continue": {
                    "type": "boolean",
                    "description": "true if more analysis needed, false when complete",
                },
            },
            "required": [
                "answer",
                "answer_type",
                "confidence",
                "reasoning",
                "image_observations",
                "region_of_interest",
                "continue",
            ],
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
    required = [
        "answer",
        "answer_type",
        "confidence",
        "reasoning",
        "image_observations",
        "region_of_interest",
        "continue",
    ]
    if not all(field in response for field in required):
        return False

    # Validate answer_type (coerce common alternatives from local models)
    answer_type = response.get("answer_type", "")
    if isinstance(answer_type, str):
        answer_type = answer_type.lower().strip()
    if answer_type in ("yes/no", "binary", "boolean", "closed-ended", "closed"):
        response["answer_type"] = "closed"
    elif answer_type in ("free-form", "open-ended", "open"):
        response["answer_type"] = "open"
    if response.get("answer_type") not in ("closed", "open"):
        return False

    # Validate confidence range (coerce strings from local models)
    confidence = response.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
            response["confidence"] = confidence
        except ValueError:
            return False
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        return False

    # Coerce image_observations from string to list if needed
    obs = response.get("image_observations")
    if isinstance(obs, str):
        response["image_observations"] = [obs] if obs else []

    # Coerce region_of_interest from string to dict if needed
    roi = response.get("region_of_interest")
    if isinstance(roi, str):
        response["region_of_interest"] = {"description": roi, "location": "unknown"}

    # Validate answer is not empty
    return bool(response.get("answer"))
