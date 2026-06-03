"""Hermetic smoke tests for the VQA-RAD example.

No network, no dataset download, no model, no API keys. Exercises the
response schema, the answer normaliser and token-F1 metric, and the
validator on well-formed and malformed inputs (including the
region-of-interest coercion local models frequently need).
"""

from __future__ import annotations

import pytest

from examples.vqa_rad.src.evaluation import compute_exact_match
from examples.vqa_rad.src.evaluation import compute_token_f1
from examples.vqa_rad.src.evaluation import evaluate_vqa_rad
from examples.vqa_rad.src.evaluation import normalize_binary
from examples.vqa_rad.src.schemas import VQA_RAD_SCHEMA
from examples.vqa_rad.src.schemas import validate_vqa_rad_response


def test_schema_shape() -> None:
    """The response schema declares the expected required fields and answer_type enum."""
    inner = VQA_RAD_SCHEMA["json_schema"]["schema"]
    assert inner["type"] == "object"
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {
        "answer",
        "answer_type",
        "confidence",
        "reasoning",
        "image_observations",
        "region_of_interest",
        "continue",
    }
    assert inner["properties"]["answer_type"]["enum"] == ["closed", "open"]


def test_normalize_binary_and_synonyms() -> None:
    """Binary normalisation handles variants and medical synonyms."""
    assert normalize_binary("Yes, clearly") == "yes"
    assert normalize_binary("No") == "no"
    assert normalize_binary("true") == "yes"
    assert normalize_binary("possibly") is None
    # Medical synonyms normalise so equivalent answers match exactly.
    assert compute_exact_match("haemorrhage", "bleeding") is True


def test_token_f1_partial_overlap() -> None:
    """Token-level F1 gives partial credit for overlapping tokens."""
    assert compute_token_f1("right lung opacity", "right lung opacity") == pytest.approx(1.0)
    assert compute_token_f1("liver", "kidney") == pytest.approx(0.0)
    # 2 shared tokens of 3 each -> precision 2/3, recall 2/3, F1 2/3.
    f1 = compute_token_f1("right lung mass", "right lung opacity")
    assert f1 == pytest.approx(2 / 3)


def test_validate_accepts_well_formed() -> None:
    """A well-formed dict validates without mutation of the answer."""
    resp = {
        "answer": "yes",
        "answer_type": "closed",
        "confidence": 0.8,
        "reasoning": "The opacity is visible in the lower lobe.",
        "image_observations": ["lower-lobe opacity"],
        "region_of_interest": {"description": "right lower lobe", "location": "right lung"},
        "continue": False,
    }
    assert validate_vqa_rad_response(resp) is True
    assert resp["answer"] == "yes"


def test_validate_coerces_answer_type_and_roi() -> None:
    """answer_type aliases and string/alt-key ROIs are coerced into canonical form."""
    resp = {
        "answer": "pneumonia",
        "answer_type": "open-ended",
        "confidence": 0.7,
        "reasoning": "Consolidation pattern.",
        "image_observations": [],
        # ROI as a bare string and using a local-model alternative key.
        "region_of_interest": "the right lung base",
        "continue": False,
    }
    assert validate_vqa_rad_response(resp) is True
    assert resp["answer_type"] == "open"
    assert isinstance(resp["region_of_interest"], dict)
    assert resp["region_of_interest"]["description"] == "the right lung base"


def test_validate_rejects_malformed() -> None:
    """Missing required keys and empty answers are rejected."""
    missing_field = {
        "answer": "yes",
        "answer_type": "closed",
        "confidence": 0.5,
        "reasoning": "x",
        "image_observations": [],
        # region_of_interest and continue missing.
    }
    assert validate_vqa_rad_response(missing_field) is False

    empty_answer = {
        "answer": "",
        "answer_type": "closed",
        "confidence": 0.5,
        "reasoning": "x",
        "image_observations": [],
        "region_of_interest": {"description": "x", "location": "y"},
        "continue": False,
    }
    assert validate_vqa_rad_response(empty_answer) is False


def test_evaluate_with_answer_types() -> None:
    """Per-type metrics split closed and open questions correctly."""
    preds = ["yes", "no", "edema"]
    refs = ["yes", "yes", "edema"]
    types = ["closed", "closed", "open"]
    metrics = evaluate_vqa_rad(preds, refs, types)
    # Closed: 1 of 2 correct.
    assert metrics["closed_accuracy"] == pytest.approx(0.5)
    assert metrics["num_closed"] == pytest.approx(2.0)
    # Open: 1 of 1 correct.
    assert metrics["open_accuracy"] == pytest.approx(1.0)


def test_evaluate_rejects_length_mismatch() -> None:
    """Mismatched prediction/reference lengths raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_vqa_rad(["yes"], ["yes", "no"])
