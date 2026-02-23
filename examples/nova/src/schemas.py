"""NOVA benchmark JSON schemas for structured outputs.

Defines the unified response schema for NOVA brain-MRI analysis tasks
including localization, diagnosis, and captioning.

NOTE: This schema uses OpenAI strict mode ("strict": true), which requires:
  - Every object must have "additionalProperties": false
  - Every property in an object must be listed in "required"
  - No optional properties (use nullable types instead if needed)

OpenAI strict mode does NOT support: minItems, maxItems, minimum, maximum,
minLength, maxLength, pattern, format, uniqueItems.  Semantic constraints
(e.g. bounding_box has exactly 4 elements, confidence in [0,1]) are enforced
by validate_nova_response() instead.
"""

from __future__ import annotations

import math
from typing import Any

# NOVA Unified Response Schema for all three tasks
# Fully compliant with OpenAI strict structured outputs:
# - All objects have additionalProperties: false
# - All properties are required (strict mode mandate)
NOVA_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "nova_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "object",
                    "description": "Radiological caption and findings",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Radiological description of visible structures",
                        },
                        "sequence_characteristics": {
                            "type": "string",
                            "description": "Imaging sequence (T1W, T2W, FLAIR, DWI, etc.)",
                        },
                        "orientation": {
                            "type": "string",
                            "description": "Image plane (axial, sagittal, coronal)",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in caption analysis (0.0-1.0)",
                        },
                        "findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific radiological findings",
                        },
                        "anatomical_regions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Anatomical regions examined",
                        },
                    },
                    "required": [
                        "description",
                        "sequence_characteristics",
                        "orientation",
                        "confidence",
                        "findings",
                        "anatomical_regions",
                    ],
                    "additionalProperties": False,
                },
                "diagnosis": {
                    "type": "object",
                    "description": "Primary and differential diagnoses",
                    "properties": {
                        "primary_diagnosis": {
                            "type": "string",
                            "description": "Most likely diagnosis",
                        },
                        "differential_diagnoses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "diagnosis": {"type": "string"},
                                    "confidence": {
                                        "type": "number",
                                    },
                                },
                                "required": ["diagnosis", "confidence"],
                                "additionalProperties": False,
                            },
                            "description": "Alternative diagnoses with confidence",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in primary diagnosis (0.0-1.0)",
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key imaging evidence",
                        },
                        "clinical_recommendations": {
                            "type": "string",
                            "description": "Recommended next steps",
                        },
                    },
                    "required": [
                        "primary_diagnosis",
                        "differential_diagnoses",
                        "confidence",
                        "evidence",
                        "clinical_recommendations",
                    ],
                    "additionalProperties": False,
                },
                "localization": {
                    "type": "object",
                    "description": "Localization of abnormalities with bounding boxes",
                    "properties": {
                        "localizations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding": {
                                        "type": "string",
                                        "description": "Description of the abnormality",
                                    },
                                    "bounding_box": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "[x1, y1, x2, y2] in pixels",
                                    },
                                    "anatomical_location": {
                                        "type": "string",
                                        "description": "Anatomical location",
                                    },
                                    "confidence": {
                                        "type": "number",
                                    },
                                },
                                "required": [
                                    "finding",
                                    "bounding_box",
                                    "anatomical_location",
                                    "confidence",
                                ],
                                "additionalProperties": False,
                            },
                            "description": "List of localized abnormalities",
                        },
                        "image_dimensions": {
                            "type": "object",
                            "properties": {
                                "width": {"type": "integer"},
                                "height": {"type": "integer"},
                            },
                            "required": ["width", "height"],
                            "additionalProperties": False,
                        },
                        "coordinate_system": {
                            "type": "string",
                            "enum": ["absolute_pixels"],
                        },
                    },
                    "required": ["localizations", "image_dimensions", "coordinate_system"],
                    "additionalProperties": False,
                },
                "continue": {
                    "type": "boolean",
                    "description": (
                        "true if more analysis is needed (tools to use), "
                        "false when analysis is complete"
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Chain-of-thought reasoning for the analysis",
                },
            },
            "required": [
                "caption",
                "diagnosis",
                "localization",
                "continue",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}


def get_required_fields() -> list[str]:
    """Get the list of required top-level fields in NOVA schema."""
    return ["caption", "diagnosis", "localization", "continue", "reasoning"]


def _is_valid_confidence(value: object) -> bool:
    """Check that a confidence value is a finite number in [0.0, 1.0]."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, int | float):
        return False
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return False
    return 0.0 <= f <= 1.0


def _is_valid_bbox(value: object) -> bool:
    """Check that a bounding box is a list of exactly 4 finite numbers."""
    if not isinstance(value, list) or len(value) != 4:
        return False
    for coord in value:
        if isinstance(coord, bool):
            return False
        if not isinstance(coord, int | float):
            return False
        f = float(coord)
        if math.isnan(f) or math.isinf(f):
            return False
    return True


def validate_nova_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required NOVA fields with correct types.

    Checks top-level presence, nested structure, and semantic constraints
    that OpenAI strict mode cannot enforce (element counts, numeric bounds).
    This catches malformed outputs that would silently produce garbage
    evaluation scores.

    Args:
        response: Parsed JSON response

    Returns:
        True if all required fields present with correct types and values
    """
    required = get_required_fields()
    if not all(field in response for field in required):
        return False

    # --- caption ---
    caption = response.get("caption")
    if not isinstance(caption, dict):
        return False
    if not isinstance(caption.get("description"), str):
        return False
    # Schema requires confidence; reject None and invalid values
    if not _is_valid_confidence(caption.get("confidence")):
        return False

    # --- diagnosis ---
    diagnosis = response.get("diagnosis")
    if not isinstance(diagnosis, dict):
        return False
    if not isinstance(diagnosis.get("primary_diagnosis"), str):
        return False
    # Schema requires confidence; reject None and invalid values
    if not _is_valid_confidence(diagnosis.get("confidence")):
        return False
    # evidence must be a list of strings when present
    evidence = diagnosis.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, list):
            return False
        if not all(isinstance(e, str) for e in evidence):
            return False
    # differential_diagnoses element validation
    diffs = diagnosis.get("differential_diagnoses")
    if diffs is not None:
        if not isinstance(diffs, list):
            return False
        for diff in diffs:
            if not isinstance(diff, dict):
                return False
            if not isinstance(diff.get("diagnosis"), str):
                return False
            if not _is_valid_confidence(diff.get("confidence")):
                return False

    # --- localization ---
    localization = response.get("localization")
    if not isinstance(localization, dict):
        return False
    localizations_list = localization.get("localizations")
    if not isinstance(localizations_list, list):
        return False
    # Validate each localization element
    for loc in localizations_list:
        if not isinstance(loc, dict):
            return False
        if not isinstance(loc.get("finding"), str):
            return False
        if not _is_valid_bbox(loc.get("bounding_box")):
            return False
        if not isinstance(loc.get("anatomical_location"), str):
            return False
        if not _is_valid_confidence(loc.get("confidence")):
            return False

    # --- continue ---
    return isinstance(response.get("continue"), bool)
