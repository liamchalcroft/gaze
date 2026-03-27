"""Tests targeting uncovered lines in prompts/__init__.py.

Covers:
- load_template OSError handling (L77-78)
- create_prompt end-to-end (exercises load_prompt internally)
- combine_prompts for both modes with exact expected output
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from radiant_harness.exceptions import TemplateError
from radiant_harness.prompts import combine_prompts
from radiant_harness.prompts import create_prompt
from radiant_harness.prompts import load_template

# ---------------------------------------------------------------------------
# load_template — OSError (L77-78)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadTemplateOSError:
    def test_os_error_raises_template_error(self, tmp_path: Path) -> None:
        """Non-FileNotFoundError OSError wraps into TemplateError (L77-78)."""
        template_path = tmp_path / "test.jinja"
        template_path.write_text("hello {{ name }}")

        with (
            patch("builtins.open", side_effect=PermissionError("denied")),
            pytest.raises(TemplateError, match="Failed to read template file"),
        ):
            load_template(template_path, {"name": "world"})

    def test_file_not_found_raises_template_error(self) -> None:
        with pytest.raises(TemplateError, match="Template file not found"):
            load_template(Path("/nonexistent/template.jinja"), {})


# ---------------------------------------------------------------------------
# create_prompt — exercises load_prompt internally
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreatePrompt:
    def test_creates_single_turn_prompt(self, tmp_path: Path) -> None:
        mode_dir = tmp_path / "single_turn"
        mode_dir.mkdir()
        (mode_dir / "system.jinja").write_text("SYS: {{ domain }}")
        (mode_dir / "task.jinja").write_text("TASK: {{ task }}")

        result = create_prompt(tmp_path, "single_turn", {"domain": "radiology", "task": "diagnose"})
        assert "SYS: radiology" in result
        assert "TASK: diagnose" in result
        assert "<analysis_instructions>" in result

    def test_creates_agentic_prompt(self, tmp_path: Path) -> None:
        mode_dir = tmp_path / "agentic"
        mode_dir.mkdir()
        (mode_dir / "system.jinja").write_text("SYS")
        (mode_dir / "task.jinja").write_text("TASK")

        result = create_prompt(tmp_path, "agentic", {})
        assert "<agentic_analysis_instructions>" in result


# ---------------------------------------------------------------------------
# combine_prompts — exercise both mode branches with exact checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinePrompts:
    def test_single_turn_wraps_with_analysis_instructions(self) -> None:
        result = combine_prompts("SYS", "TASK", "single_turn")
        assert result == "SYS\n\n<analysis_instructions>\nTASK\n</analysis_instructions>"

    def test_agentic_wraps_with_agentic_instructions(self) -> None:
        result = combine_prompts("SYS", "TASK", "agentic")
        expected = "SYS\n\n<agentic_analysis_instructions>\nTASK\n</agentic_analysis_instructions>"
        assert result == expected

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            combine_prompts("SYS", "TASK", "bogus")
