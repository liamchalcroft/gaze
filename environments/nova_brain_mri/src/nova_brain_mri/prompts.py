"""NOVA system prompt and JSON response schema, in the GAZE style.

The schema and prompt describe the JSON the policy model under test must
emit. The environment scores that JSON with the NOVA rubric; GAZE is not
run as an agent here. Bounding boxes are absolute pixels ``[x1, y1, x2, y2]``.
"""

from __future__ import annotations

from typing import Any

NOVATask = str

# Expected JSON structure for a NOVA brain-MRI analysis. Field names match the
# keys read by the reward functions in ``rewards.py``.
NOVA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": (
                "Radiological description of the brain MRI, including sequence "
                "characteristics and orientation when identifiable."
            ),
        },
        "diagnosis": {
            "type": "object",
            "properties": {
                "primary_diagnosis": {
                    "type": "string",
                    "description": "Most likely diagnosis.",
                },
                "differential_diagnoses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative diagnoses worth considering.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the primary diagnosis, in [0, 1].",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Imaging findings supporting the primary diagnosis.",
                },
            },
            "required": ["primary_diagnosis"],
        },
        "localization": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bounding_box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Pixel box [x1, y1, x2, y2].",
                    },
                    "label": {
                        "type": "string",
                        "description": "Finding the box localizes.",
                    },
                },
                "required": ["bounding_box"],
            },
            "description": "Bounding boxes for abnormalities, in absolute pixels.",
        },
        "continue": {
            "type": "boolean",
            "description": "true to keep analyzing, false when the answer is final.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief chain of reasoning behind the analysis.",
        },
    },
    "required": ["caption", "diagnosis", "localization", "continue"],
}

_TASK_FOCUS: dict[str, str] = {
    "caption": (
        "Focus on the radiological caption: describe the sequence, orientation, "
        "and the salient findings."
    ),
    "diagnosis": (
        "Focus on diagnosis: give the most likely primary diagnosis and a short "
        "list of differentials with supporting evidence."
    ),
    "localization": (
        "Focus on localization: place a bounding box [x1, y1, x2, y2] in pixels "
        "around each abnormality you identify."
    ),
    "all": (
        "Provide a complete analysis: caption, diagnosis, and localization of every abnormality."
    ),
}


def build_system_prompt(task: NOVATask) -> str:
    """Build the system prompt for the requested NOVA task.

    Args:
        task: One of ``caption``, ``diagnosis``, ``localization``, ``all``.

    Returns:
        System prompt instructing the model to emit only NOVA JSON.
    """
    focus = _TASK_FOCUS.get(task, _TASK_FOCUS["all"])

    return (
        "You are an expert neuroradiologist analyzing a brain MRI image.\n\n"
        f"{focus}\n\n"
        "Respond with a single valid JSON object and nothing else. The object "
        "uses these fields:\n"
        "- caption: a radiological description string.\n"
        "- diagnosis: an object with primary_diagnosis (string), "
        "differential_diagnoses (list of strings), confidence (number in [0, 1]), "
        "and evidence (list of strings).\n"
        "- localization: a list of objects, each with bounding_box [x1, y1, x2, y2] "
        "in absolute pixels and a label string.\n"
        "- continue: a boolean. Set it to true if you need another turn to keep "
        "analyzing, and false once your final answer is ready.\n"
        "- reasoning: a short string explaining your analysis.\n\n"
        "Bounding boxes are [x1, y1, x2, y2] with the top-left corner first, in "
        "image pixels. Examine the whole image before concluding."
    )


def build_user_message(case: dict[str, Any]) -> str:
    """Build the textual user message for a case.

    The environment adds the image part separately, so this returns only the
    clinical context and the instruction.

    Args:
        case: Case dict with ``clinical_history`` and ``modality``.

    Returns:
        User message text.
    """
    history = case.get("clinical_history", "")
    modality = case.get("modality", "MRI")

    parts = ["Analyze this brain MRI image."]
    if history:
        parts.append(f"Clinical history: {history}")
    if modality:
        parts.append(f"Modality: {modality}")
    parts.append("Return your final answer as a single JSON object with continue set to false.")

    return "\n".join(parts)
