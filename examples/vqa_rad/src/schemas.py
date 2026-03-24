"""VQA-RAD JSON schemas for structured outputs.

Defines the response schema for visual question answering on radiology images.
"""

from __future__ import annotations

from typing import Any

from radiant_harness.utils import clamp_confidence
from radiant_harness.utils import coerce_json_types

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

# answer_type aliases produced by local models
_CLOSED_ALIASES = {"yes/no", "binary", "boolean", "closed-ended", "closed"}
_OPEN_ALIASES = {"free-form", "open-ended", "open"}


def validate_vqa_rad_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required VQA-RAD fields."""
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

    coerce_json_types(response, VQA_RAD_SCHEMA)

    # Normalize answer_type aliases from local models
    answer_type = str(response.get("answer_type", "")).lower().strip()
    if answer_type in _CLOSED_ALIASES:
        response["answer_type"] = "closed"
    elif answer_type in _OPEN_ALIASES:
        response["answer_type"] = "open"
    if response.get("answer_type") not in ("closed", "open"):
        return False

    clamped = clamp_confidence(response.get("confidence"))
    if clamped is None:
        return False
    response["confidence"] = clamped

    # Coerce region_of_interest: ensure it has the required inner keys
    roi = response.get("region_of_interest")
    if isinstance(roi, str):
        response["region_of_interest"] = {"description": roi, "location": "unknown"}
    elif isinstance(roi, dict):
        if "description" not in roi:
            # Local models often use alternative keys like "anatomical_structure"
            roi["description"] = roi.pop("anatomical_structure", roi.pop("region", "unknown"))
        if "location" not in roi:
            roi["location"] = roi.pop("area", "unknown")

    if not isinstance(response.get("continue"), bool):
        return False

    return bool(response.get("answer"))
