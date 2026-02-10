"""Regression tests for NOVA reward/schema alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import examples/nova/src as package "src"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NOVA_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_NOVA_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_NOVA_ROOT))

from src.rewards import NOVAVerifiersReward
from src.rewards import compute_localization_reward


def _schema_completion() -> str:
    return json.dumps(
        {
            "caption": {
                "description": "left temporal lobe lesion",
                "sequence_characteristics": "T2",
                "orientation": "axial",
                "confidence": 0.9,
            },
            "diagnosis": {
                "primary_diagnosis": "glioma",
                "confidence": 0.8,
                "evidence": ["left temporal lesion"],
                "differential_diagnoses": [],
            },
            "localization": {
                "localizations": [
                    {
                        "finding": "lesion",
                        "bounding_box": [10, 10, 20, 20],
                        "anatomical_location": "left temporal lobe",
                        "confidence": 0.8,
                    }
                ],
                "image_dimensions": {"width": 64, "height": 64},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
        }
    )


def test_nova_rewards_use_schema_keys() -> None:
    completion = _schema_completion()
    info = {
        "caption": "left temporal lobe lesion",
        "diagnosis": "glioma",
        "boxes": [[10, 10, 20, 20]],
    }

    assert NOVAVerifiersReward(task="caption")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="diagnosis")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="localization")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="all")("", completion, info) == 1.0


def test_localization_reward_penalizes_false_positive_without_reference() -> None:
    # No lesions in reference: spurious predictions should not get credit.
    assert compute_localization_reward([[0, 0, 10, 10]], []) == 0.0
    # Empty prediction with empty reference should still be treated as correct.
    assert compute_localization_reward([], []) == 1.0
