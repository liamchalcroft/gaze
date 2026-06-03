"""Hermetic smoke tests for the PubMedQA example.

No network, no dataset download, no model, no API keys. Exercises the
response schema, the canonical answer normaliser, the validator on
well-formed and malformed inputs, and the accuracy/F1 metric on synthetic
predictions.
"""

from __future__ import annotations

import pytest

from examples.pubmedqa.src.evaluation import evaluate_pubmedqa
from examples.pubmedqa.src.schemas import PUBMEDQA_SCHEMA
from examples.pubmedqa.src.schemas import normalize_pubmedqa_answer
from examples.pubmedqa.src.schemas import validate_pubmedqa_response


def test_schema_shape() -> None:
    """The response schema declares the expected required fields and answer enum."""
    inner = PUBMEDQA_SCHEMA["json_schema"]["schema"]
    assert inner["type"] == "object"
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {
        "answer",
        "confidence",
        "reasoning",
        "key_evidence",
        "continue",
    }
    assert inner["properties"]["answer"]["enum"] == ["yes", "no", "maybe"]


def test_normalize_answer_canonicalises_variations() -> None:
    """Common phrasings and sentence-form answers collapse to yes/no/maybe."""
    assert normalize_pubmedqa_answer("Yes") == "yes"
    assert normalize_pubmedqa_answer("TRUE") == "yes"
    assert normalize_pubmedqa_answer("no.") == "no"
    assert normalize_pubmedqa_answer("uncertain") == "maybe"
    # Leading-token extraction from a longer response.
    assert normalize_pubmedqa_answer("Yes, based on the evidence presented") == "yes"
    # Unknown text passes through unchanged (caller decides what to do).
    assert normalize_pubmedqa_answer("perhaps not") == "perhaps not"


def test_validate_accepts_well_formed_and_normalises() -> None:
    """A well-formed dict validates and is normalised in place."""
    resp = {
        "answer": "Yes",
        "confidence": 0.9,
        "reasoning": "The study reports a significant association.",
        "key_evidence": ["significant association"],
        "continue": False,
    }
    assert validate_pubmedqa_response(resp) is True
    # Validation normalises the answer to canonical form.
    assert resp["answer"] == "yes"


def test_validate_clamps_out_of_range_confidence() -> None:
    """Out-of-range confidence is clamped to [0, 1] rather than rejected."""
    resp = {
        "answer": "no",
        "confidence": 1.5,
        "reasoning": "No effect was observed.",
        "key_evidence": [],
        "continue": False,
    }
    assert validate_pubmedqa_response(resp) is True
    assert resp["confidence"] == 1.0


def test_validate_rejects_malformed() -> None:
    """Missing required keys and unrecognised answers are rejected."""
    # Missing the required "continue" key.
    missing_field = {
        "answer": "yes",
        "confidence": 0.5,
        "reasoning": "x",
        "key_evidence": [],
    }
    assert validate_pubmedqa_response(missing_field) is False

    # Answer that does not normalise to yes/no/maybe.
    bad_answer = {
        "answer": "definitely not sure at all",
        "confidence": 0.5,
        "reasoning": "x",
        "key_evidence": [],
        "continue": False,
    }
    assert validate_pubmedqa_response(bad_answer) is False


def test_evaluate_metrics_on_synthetic_inputs() -> None:
    """Accuracy and macro-F1 are computed correctly on a tiny synthetic set."""
    preds = ["yes", "no", "maybe", "yes"]
    refs = ["yes", "no", "maybe", "no"]
    metrics = evaluate_pubmedqa(preds, refs)
    # 3 of 4 correct.
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["accuracy_maybe"] == pytest.approx(1.0)


def test_evaluate_rejects_length_mismatch() -> None:
    """Mismatched prediction/reference lengths raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_pubmedqa(["yes"], ["yes", "no"])
