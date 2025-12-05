"""JSON schemas for OpenRouter structured outputs.

Provides well-defined schemas for NOVA task outputs to ensure
consistent parsing and validation with structured outputs.
"""

from __future__ import annotations

from typing import Any
from typing import cast

# NOVA Unified Response Schema for all three tasks
NOVA_UNIFIED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nova_unified_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "object",
                    "description": "Comprehensive radiological caption and findings",
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
                            "description": "Image orientation and plane (axial, sagittal, coronal)",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the caption analysis",
                        },
                        "findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific radiological findings observed",
                        },
                        "anatomical_regions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Anatomical regions examined or involved",
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
                            "description": "Most likely diagnosis based on imaging findings",
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
                            "description": "Alternative diagnostic considerations",
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
                            "description": "Key imaging evidence supporting the diagnosis",
                        },
                        "clinical_recommendations": {
                            "type": "string",
                            "description": "Recommended next steps or clinical actions",
                        },
                    },
                    "required": ["primary_diagnosis", "confidence", "evidence"],
                },
                "localization": {
                    "type": "object",
                    "description": "Precise localization of abnormalities with bounding boxes",
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
                                        "description": "Box coordinates [x1, y1, x2, y2] in pixels",
                                    },
                                    "anatomical_location": {
                                        "type": "string",
                                        "description": "Precise anatomical location of the finding",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Confidence in the localization",
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
                            "description": "Image dimensions in pixels",
                        },
                        "coordinate_system": {
                            "type": "string",
                            "enum": ["absolute_pixels"],
                            "description": "Coordinate system used for bounding boxes",
                        },
                    },
                    "required": ["localizations", "image_dimensions", "coordinate_system"],
                },
                "continue": {
                    "type": "boolean",
                    "description": (
                        "Set to true if more analysis is needed (e.g., need to use tools). "
                        "Set to false when analysis is complete and final."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Chain-of-thought reasoning explaining the analysis process.",
                },
            },
            "required": ["caption", "diagnosis", "localization", "continue"],
            "additionalProperties": False,
        },
    },
}

# Extract sub-schemas with proper typing for individual task use
_json_schema = cast(dict[str, Any], NOVA_UNIFIED_SCHEMA["json_schema"])
_unified_schema = cast(dict[str, Any], _json_schema["schema"])
_properties = cast(dict[str, dict[str, Any]], _unified_schema["properties"])

# Helper to construct a complete individual task schema
def _make_task_schema(name: str, task_key: str) -> dict[str, Any]:
    """Create a complete JSON Schema wrapper for an individual task."""
    task_schema = _properties[task_key]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "description": task_schema.get("description", ""),
                "properties": task_schema.get("properties", {}),
                "required": task_schema.get("required", []),
                "additionalProperties": False,
            },
        },
    }


# Individual task schemas for when only one task is needed
CAPTION_SCHEMA = _make_task_schema("nova_caption_response", "caption")
DIAGNOSIS_SCHEMA = _make_task_schema("nova_diagnosis_response", "diagnosis")
LOCALIZATION_SCHEMA = _make_task_schema("nova_localization_response", "localization")
