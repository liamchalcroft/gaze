"""NOVA benchmark JSON schemas for structured outputs.

Defines the unified response schema for NOVA brain-MRI analysis tasks
including localization, diagnosis, and captioning.
"""

from __future__ import annotations

from typing import Any

# NOVA Unified Response Schema for all three tasks
NOVA_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "nova_unified_response",
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
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in caption analysis",
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
                    ],
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
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                },
                                "required": ["diagnosis", "confidence"],
                            },
                            "description": "Alternative diagnoses with confidence",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in primary diagnosis",
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
                    "required": ["primary_diagnosis", "confidence", "evidence"],
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
                                        "minItems": 4,
                                        "maxItems": 4,
                                        "description": "[x1, y1, x2, y2] in pixels",
                                    },
                                    "anatomical_location": {
                                        "type": "string",
                                        "description": "Anatomical location",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                },
                                "required": [
                                    "finding",
                                    "bounding_box",
                                    "anatomical_location",
                                    "confidence",
                                ],
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
                        },
                        "coordinate_system": {
                            "type": "string",
                            "enum": ["absolute_pixels"],
                        },
                    },
                    "required": ["localizations", "image_dimensions", "coordinate_system"],
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
            "required": ["caption", "diagnosis", "localization", "continue"],
            "additionalProperties": False,
        },
    },
}


def get_required_fields() -> list[str]:
    """Get the list of required top-level fields in NOVA schema."""
    return ["caption", "diagnosis", "localization", "continue"]


def validate_nova_response(response: dict[str, Any]) -> bool:
    """Validate that a response has required NOVA fields.

    Args:
        response: Parsed JSON response

    Returns:
        True if all required fields present
    """
    required = get_required_fields()
    return all(field in response for field in required)
