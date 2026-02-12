"""GEMeX-ThinkVG JSON schemas for structured outputs.

Defines the ThinkVG response schema with three verifiable components:
1. Answer - the medical finding/diagnosis
2. Location - anatomical region reference
3. BBox - bounding box coordinates for visual grounding
"""

from __future__ import annotations

import re
from typing import Any

GEMEX_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "gemex_thinkvg_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Chain-of-thought reasoning with region analysis",
                },
                "answer": {
                    "type": "string",
                    "description": "Medical finding or diagnosis answer",
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "string",
                            "description": "Anatomical region name (e.g., 'bilateral lung', 'right lower lobe')",
                        },
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "Bounding box [x1, y1, x2, y2] in pixels",
                        },
                    },
                    "required": ["reference", "bbox"],
                    "description": "Visual grounding location",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the answer (0.0-1.0)",
                },
                "question_type": {
                    "type": "string",
                    "enum": ["open_ended", "closed_ended", "single_choice", "multi_choice"],
                    "description": "Type of question being answered",
                },
                "continue": {
                    "type": "boolean",
                    "description": "true if more analysis needed, false when complete",
                },
            },
            "required": ["reasoning", "answer", "location", "confidence"],
            "additionalProperties": False,
        },
    },
}


def validate_gemex_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required GEMeX fields.

    Args:
        response: Parsed JSON response

    Returns:
        True if all required fields present and valid
    """
    required = ["reasoning", "answer", "location", "confidence"]
    if not all(field in response for field in required):
        return False

    # Validate location structure
    location = response.get("location", {})
    if not isinstance(location, dict):
        return False
    if "reference" not in location or "bbox" not in location:
        return False

    # Validate bbox format
    bbox = location.get("bbox", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(x, int | float) for x in bbox):
        return False

    # Validate confidence range
    confidence = response.get("confidence")
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        return False

    return bool(response.get("answer"))


def parse_thinkvg_response(response_text: str) -> dict[str, Any] | None:
    """Parse ThinkVG XML-style response format.

    The GEMeX dataset uses XML-style responses:
    <response>
        <answer>...</answer>
        <location>
            <ref>...</ref>
            <box>[x1, y1, x2, y2]</box>
        </location>
    </response>

    Args:
        response_text: Raw response text possibly containing XML

    Returns:
        Parsed response dict or None if parsing fails
    """
    # Try to extract XML response block
    response_match = re.search(r"<response>(.*?)</response>", response_text, re.DOTALL)
    if not response_match:
        return None

    content = response_match.group(1)

    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    # Extract location
    location_match = re.search(r"<location>(.*?)</location>", content, re.DOTALL)
    location = {}
    if location_match:
        loc_content = location_match.group(1)

        ref_match = re.search(r"<ref>(.*?)</ref>", loc_content, re.DOTALL)
        location["reference"] = ref_match.group(1).strip() if ref_match else ""

        box_match = re.search(r"<box>\s*\[(.*?)\]\s*</box>", loc_content, re.DOTALL)
        if box_match:
            try:
                coords = [int(x.strip()) for x in box_match.group(1).split(",")]
                location["bbox"] = coords
            except ValueError:
                location["bbox"] = [0, 0, 0, 0]
        else:
            location["bbox"] = [0, 0, 0, 0]

    return {
        "answer": answer,
        "location": location,
    }
