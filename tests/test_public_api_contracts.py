"""Regression tests for public API and docs-facing contracts."""

from __future__ import annotations

from pathlib import Path

from radiant_harness.__main__ import main
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.exceptions import SchemaValidationError


def test_schema_validation_error_is_agentic_processing_error() -> None:
    err = SchemaValidationError(
        "invalid response",
        turns_completed=2,
        missing_fields=["result"],
        response={"continue": False},
    )
    assert isinstance(err, AgenticProcessingError)
    assert err.turns_completed == 2
    assert err.partial_response == {"continue": False}


def test_readme_structured_output_example_matches_runtime_shape() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert '"json_schema": {' in readme
    assert '"type": "json_schema",\n            "schema": {' not in readme
    assert "from pathlib import Path" in readme


def test_cli_usage_message_matches_example_parser(capsys) -> None:
    exit_code = main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "python -m src.cli --task localization --model openai/gpt-4o" in captured.out
    assert "task=localization" not in captured.out


def test_nova_example_lazy_exports_are_listed_in_all() -> None:
    import examples.nova.src as nova

    for name in (
        "evaluate_caption",
        "evaluate_detection",
        "evaluate_diagnosis_nova_official",
    ):
        assert name in nova.__all__


def test_lazy_import_huggingface_adapter() -> None:
    """HuggingFaceAdapter is listed in __all__ and accessible via __getattr__."""
    import radiant_harness

    assert "HuggingFaceAdapter" in radiant_harness.__all__
    import contextlib

    with contextlib.suppress(ImportError):
        _ = radiant_harness.HuggingFaceAdapter


def test_lazy_import_huggingface_vlm_adapter() -> None:
    """HuggingFaceVLMAdapter is listed in __all__ and accessible via __getattr__."""
    import radiant_harness

    assert "HuggingFaceVLMAdapter" in radiant_harness.__all__
    import contextlib

    with contextlib.suppress(ImportError):
        _ = radiant_harness.HuggingFaceVLMAdapter


def test_getattr_raises_attribute_error_for_unknown() -> None:
    """Accessing a non-existent attribute should raise AttributeError, not silently return None."""
    import pytest

    import radiant_harness

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = radiant_harness.NoSuchAdapter
