"""Hermetic smoke tests for the GEMeX-ThinkVG example.

No network, no dataset, no model, no API keys. Exercises the ThinkVG
response schema, the XML response parser, the validator on well-formed and
malformed inputs, and the IoU / bounding-box reward on synthetic boxes.

The package ``__init__`` pulls in the ``verifiers`` integration, so these
tests import the schema and reward submodules directly to stay dependency
light. A separate guarded test confirms the verifiers-backed reward loads
when that optional dependency is present.
"""

from __future__ import annotations

import pytest

from examples.gemex_thinkvg.src.rewards.bbox import IMAGE_SIZE
from examples.gemex_thinkvg.src.rewards.bbox import compute_bbox_reward
from examples.gemex_thinkvg.src.rewards.bbox import compute_iou
from examples.gemex_thinkvg.src.rewards.bbox import validate_bbox
from examples.gemex_thinkvg.src.schemas import GEMEX_SCHEMA
from examples.gemex_thinkvg.src.schemas import parse_thinkvg_response
from examples.gemex_thinkvg.src.schemas import validate_gemex_response


def test_schema_shape() -> None:
    """The response schema declares the three verifiable components."""
    inner = GEMEX_SCHEMA["json_schema"]["schema"]
    assert inner["type"] == "object"
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {
        "reasoning",
        "answer",
        "location",
        "confidence",
        "continue",
    }
    location = inner["properties"]["location"]
    assert set(location["required"]) == {"reference", "bbox"}


def test_parse_thinkvg_xml_response() -> None:
    """The XML-style ThinkVG response is parsed into answer + location."""
    text = (
        "<response>"
        "<answer>pleural effusion</answer>"
        "<location><ref>right lower lobe</ref><box>[10, 20, 110, 120]</box></location>"
        "</response>"
    )
    parsed = parse_thinkvg_response(text)
    assert parsed is not None
    assert parsed["answer"] == "pleural effusion"
    assert parsed["location"]["reference"] == "right lower lobe"
    assert parsed["location"]["bbox"] == [10, 20, 110, 120]
    # No <response> wrapper -> None.
    assert parse_thinkvg_response("no tags here") is None


def test_validate_accepts_well_formed() -> None:
    """A well-formed response with an ordered bbox validates."""
    resp = {
        "reasoning": "The opacity sits in the right lower lobe.",
        "answer": "pleural effusion",
        "location": {"reference": "right lower lobe", "bbox": [10, 20, 110, 120]},
        "confidence": 0.85,
        "continue": False,
    }
    assert validate_gemex_response(resp) is True


def test_validate_rejects_degenerate_bbox() -> None:
    """A bbox with non-increasing coordinates is rejected."""
    resp = {
        "reasoning": "x",
        "answer": "mass",
        # x2 <= x1 -> invalid.
        "location": {"reference": "lung", "bbox": [100, 20, 50, 120]},
        "confidence": 0.5,
        "continue": False,
    }
    assert validate_gemex_response(resp) is False


def test_validate_rejects_missing_field() -> None:
    """Missing a required top-level field is rejected."""
    resp = {
        "answer": "mass",
        "location": {"reference": "lung", "bbox": [10, 20, 110, 120]},
        "confidence": 0.5,
        "continue": False,
        # "reasoning" missing.
    }
    assert validate_gemex_response(resp) is False


def test_compute_iou_known_values() -> None:
    """IoU is exact for identical, disjoint, and half-overlapping boxes."""
    assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)
    assert compute_iou([0, 0, 10, 10], [200, 200, 300, 300]) == pytest.approx(0.0)
    # Two 100x100 boxes overlapping in a 50x100 strip:
    # intersection 5000, union 15000 -> IoU 1/3.
    assert compute_iou([0, 0, 100, 100], [50, 0, 150, 100]) == pytest.approx(1 / 3)


def test_validate_bbox_bounds() -> None:
    """Boxes outside the image (beyond margin) are rejected."""
    assert validate_bbox([10, 20, 110, 120]) is True
    assert validate_bbox([0, 0, IMAGE_SIZE + 100, IMAGE_SIZE + 100]) is False
    assert validate_bbox([1, 2, 3]) is False


def test_bbox_reward_perfect_and_invalid() -> None:
    """A perfect match scores high IoU; an invalid prediction scores zero."""
    perfect = compute_bbox_reward([10, 20, 110, 120], [10, 20, 110, 120])
    assert perfect["valid_prediction"] is True
    assert perfect["iou"] == pytest.approx(1.0)
    assert perfect["iou_50"] is True

    # A zero-area (degenerate) box is rejected.
    invalid = compute_bbox_reward([5, 5, 5, 5], [10, 20, 110, 120])
    assert invalid["valid_prediction"] is False
    assert invalid["reward"] == pytest.approx(0.0)


def test_verifiers_environment_loader_optional() -> None:
    """The verifiers-backed reward loads when verifiers + datasets are present."""
    pytest.importorskip("verifiers")
    pytest.importorskip("datasets")

    from examples.gemex_thinkvg.src.rewards import GEMeXVerifiersReward
    from examples.gemex_thinkvg.src.rewards import RewardWeights

    reward = GEMeXVerifiersReward(weights=RewardWeights())
    assert callable(reward)
