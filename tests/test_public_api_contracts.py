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
    """HuggingFaceAdapter is reachable by name but kept out of __all__.

    A star import resolves every name in __all__ through __getattr__, which
    would import torch and defeat the deferral.
    """
    import contextlib

    import gaze

    assert "HuggingFaceAdapter" not in gaze.__all__
    with contextlib.suppress(ImportError):
        _ = gaze.HuggingFaceAdapter


def test_lazy_import_huggingface_vlm_adapter() -> None:
    """Same contract as the adapter above: importable by name, not via star."""
    import contextlib

    import gaze

    assert "HuggingFaceVLMAdapter" not in gaze.__all__
    with contextlib.suppress(ImportError):
        _ = gaze.HuggingFaceVLMAdapter


def test_star_import_does_not_pull_in_torch() -> None:
    """The whole point of keeping the adapters out of __all__."""
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        exec("from gaze import *")
        assert "torch" not in sys.modules, "star import pulled in torch"
        print("ok")
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no external input
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_getattr_raises_attribute_error_for_unknown() -> None:
    """Accessing a non-existent attribute should raise AttributeError, not silently return None."""
    import pytest

    import gaze

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = gaze.NoSuchAdapter


def test_unknown_gaze_models_attribute_raises_attribute_error() -> None:
    """The lazy __getattr__ must still reject names it does not provide."""
    import pytest

    from gaze import models

    with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
        _ = models.NonExistent


def test_gaze_models_lazy_import_returns_the_real_adapters() -> None:
    """Accessing the names triggers the deferred import of the torch-backed module."""
    import contextlib

    from gaze import models

    with contextlib.suppress(ImportError):
        assert models.HuggingFaceAdapter.__name__ == "HuggingFaceAdapter"
        assert models.HuggingFaceVLMAdapter.__name__ == "HuggingFaceVLMAdapter"
