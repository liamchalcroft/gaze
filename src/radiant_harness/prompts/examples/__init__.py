"""Example task templates for common image analysis use cases.

These templates demonstrate how to define specific tasks using the harness.
Users can copy and adapt these for their own datasets and tasks.

All templates support flexible input modes:
- Text-only: Provide context via metadata (e.g., clinical history)
- Single image: Standard single-image analysis
- Multi-image: Analysis of multiple images with labels

Available examples:
    - lesion_localization.jinja: Detect and localize abnormalities with bounding boxes
    - diagnosis.jinja: Primary diagnosis with differential diagnoses (supports text-only)
    - sequence_classification.jinja: Identify imaging modality/sequence type
    - captioning.jinja: Open-ended radiological description
    - structure_localization.jinja: Localize specific anatomical structures
    - prognosis.jinja: Disease staging and prognosis assessment (supports text-only)
    - longitudinal_comparison.jinja: Compare multiple images over time or across modalities

Usage:
    from pathlib import Path
    from radiant_harness.prompts import load_task_prompt
    from radiant_harness.prompts.examples import EXAMPLES_DIR

    # Load an example template with single image
    prompt = load_task_prompt(
        prompts_dir=EXAMPLES_DIR.parent,
        mode="agentic",
        context={
            "images": [ImageInput(path=Path("/path/to/image.png"), label="T1-weighted")],
            "clinical_history": "Patient with headache",
            "has_images": True,
        },
        template_name="examples/lesion_localization.jinja",
    )

    # Load for text-only diagnosis (no images)
    prompt = load_task_prompt(
        prompts_dir=EXAMPLES_DIR.parent,
        mode="agentic",
        context={
            "images": [],
            "clinical_history": "45yo male with fever, cough, and dyspnea",
            "patient_info": "Smoker, diabetic",
            "has_images": False,
        },
        template_name="examples/diagnosis.jinja",
    )

    # Load for multi-image comparison
    prompt = load_task_prompt(
        prompts_dir=EXAMPLES_DIR.parent,
        mode="agentic",
        context={
            "images": [
                ImageInput(path=Path("/path/to/baseline.png"), label="Baseline"),
                ImageInput(path=Path("/path/to/followup.png"), label="3-month follow-up"),
            ],
            "clinical_history": "Glioblastoma post-resection",
            "time_interval": "3 months",
            "treatment": "Temozolomide + radiation",
            "comparison_type": "longitudinal",
            "has_images": True,
        },
        template_name="examples/longitudinal_comparison.jinja",
    )
"""

from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent

# List of available example templates
AVAILABLE_EXAMPLES = [
    "lesion_localization.jinja",
    "diagnosis.jinja",
    "sequence_classification.jinja",
    "captioning.jinja",
    "structure_localization.jinja",
    "prognosis.jinja",
    "longitudinal_comparison.jinja",
]

__all__ = ["EXAMPLES_DIR", "AVAILABLE_EXAMPLES"]
