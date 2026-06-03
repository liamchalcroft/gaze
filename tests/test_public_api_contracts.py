"""Regression tests for public API and docs-facing contracts."""

from __future__ import annotations

from pathlib import Path

from gaze.__main__ import main
from gaze.exceptions import AgenticProcessingError
from gaze.exceptions import SchemaValidationError


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


def test_cli_info_points_to_repo_not_local_examples(capsys) -> None:
    import gaze

    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"GAZE {gaze.__version__}" in captured.out
    # An installed wheel has no examples/ directory, so the CLI must not tell
    # users to cd into one; it points at the source repository instead.
    assert "cd examples" not in captured.out
    assert "github.com/liamchalcroft/gaze" in captured.out


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
    import gaze

    assert "HuggingFaceAdapter" in gaze.__all__
    import contextlib

    with contextlib.suppress(ImportError):
        _ = gaze.HuggingFaceAdapter


def test_lazy_import_huggingface_vlm_adapter() -> None:
    """HuggingFaceVLMAdapter is listed in __all__ and accessible via __getattr__."""
    import gaze

    assert "HuggingFaceVLMAdapter" in gaze.__all__
    import contextlib

    with contextlib.suppress(ImportError):
        _ = gaze.HuggingFaceVLMAdapter


def test_getattr_raises_attribute_error_for_unknown() -> None:
    """Accessing a non-existent attribute should raise AttributeError, not silently return None."""
    import pytest

    import gaze

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = gaze.NoSuchAdapter
