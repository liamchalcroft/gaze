from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.exceptions import TemplateError
from radiant_harness.prompts import combine_prompts
from radiant_harness.prompts import create_prompt
from radiant_harness.prompts import load_template


def _write_template(dir_path: Path, name: str, content: str) -> None:
    (dir_path / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# create_prompt
# ---------------------------------------------------------------------------


def test_create_prompt_renders_templates(tmp_path: Path) -> None:
    agentic_dir = tmp_path / "agentic"
    agentic_dir.mkdir(parents=True)
    _write_template(agentic_dir, "system.jinja", "System {{ role }}")
    _write_template(agentic_dir, "task.jinja", "Task {{ instruction }}")

    prompt = create_prompt(
        prompts_dir=tmp_path,
        mode="agentic",
        context={"role": "radiologist", "instruction": "analyze"},
    )

    assert "System radiologist" in prompt
    assert "Task analyze" in prompt


# ---------------------------------------------------------------------------
# combine_prompts
# ---------------------------------------------------------------------------


def test_combine_prompts_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        combine_prompts("system", "task", "unknown")


def test_combine_prompts_agentic() -> None:
    result = combine_prompts("SYS", "TASK", "agentic")
    assert "SYS" in result
    assert "TASK" in result
    assert "<agentic_analysis_instructions>" in result


def test_combine_prompts_single_turn() -> None:
    result = combine_prompts("SYS", "TASK", "single_turn")
    assert "SYS" in result
    assert "TASK" in result
    assert "<analysis_instructions>" in result


# ---------------------------------------------------------------------------
# Strict undefined variable behavior
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_missing_variable(tmp_path: Path) -> None:
    """Templates using {{ var }} without 'is defined' must fail when var is absent."""
    _write_template(tmp_path, "strict.jinja", "Hello {{ name }}")
    with pytest.raises(TemplateError, match="undefined value"):
        load_template(tmp_path / "strict.jinja", {})


def test_strict_mode_passes_when_variable_provided(tmp_path: Path) -> None:
    _write_template(tmp_path, "strict.jinja", "Hello {{ name }}")
    result = load_template(tmp_path / "strict.jinja", {"name": "World"})
    assert result == "Hello World"


def test_is_defined_guard_works_when_absent(tmp_path: Path) -> None:
    """{% if x is defined %} should silently skip when x is absent."""
    _write_template(
        tmp_path,
        "guarded.jinja",
        "{% if x is defined and x %}{{ x }}{% endif %}done",
    )
    result = load_template(tmp_path / "guarded.jinja", {})
    assert result == "done"


def test_is_defined_guard_works_when_present(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "guarded.jinja",
        "{% if x is defined and x %}{{ x }}{% endif %}done",
    )
    result = load_template(tmp_path / "guarded.jinja", {"x": "hi"})
    assert result == "hidone"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_load_template_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TemplateError, match="Template file not found"):
        load_template(tmp_path / "nonexistent.jinja", {})


def test_load_template_syntax_error(tmp_path: Path) -> None:
    _write_template(tmp_path, "bad.jinja", "{% if %}broken{% endif %}")
    with pytest.raises(TemplateError, match="Failed to render template"):
        load_template(tmp_path / "bad.jinja", {})


def test_load_prompt_rejects_invalid_mode(tmp_path: Path) -> None:
    from radiant_harness.prompts import load_prompt

    with pytest.raises(ValueError, match="Unknown mode"):
        load_prompt(tmp_path, "system.jinja", "nonexistent_mode", {})


def test_load_prompt_missing_mode_dir(tmp_path: Path) -> None:
    from radiant_harness.prompts import load_prompt

    with pytest.raises(ValueError, match="Mode directory not found"):
        load_prompt(tmp_path, "system.jinja", "agentic", {})


# ---------------------------------------------------------------------------
# Base template rendering (smoke tests)
# ---------------------------------------------------------------------------

BASE_PROMPTS = Path("src/radiant_harness/prompts")


@pytest.mark.parametrize(
    "mode,template",
    [
        ("agentic", "system.jinja"),
        ("agentic", "task.jinja"),
        ("single_turn", "system.jinja"),
        ("single_turn", "task.jinja"),
    ],
)
def test_base_templates_render_with_empty_context(mode: str, template: str) -> None:
    """All base templates use 'is defined' guards, so empty context should work."""
    result = load_template(BASE_PROMPTS / mode / template, {})
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("mode", ["agentic", "single_turn"])
def test_base_templates_render_with_full_context(mode: str) -> None:
    """Base templates should render all sections when full context is provided."""
    context = {
        "domain_expertise": "Expert radiologist",
        "tool_documentation": "zoom(), crop()",
        "analysis_workflow": "1. Scan 2. Analyze",
        "task_instructions": "Analyze this image",
        "images": [],
        "image_info": "512x384 brain MRI",
        "context": "Patient has headaches",
        "output_format": "JSON with findings",
        "has_images": True,
        "enable_web_search": True,
    }
    result = create_prompt(
        prompts_dir=BASE_PROMPTS,
        mode=mode,
        context=context,
    )
    assert "Expert radiologist" in result
    assert "Analyze this image" in result


# ---------------------------------------------------------------------------
# NOVA template rendering (smoke tests)
# ---------------------------------------------------------------------------

NOVA_PROMPTS = Path("examples/nova/src/prompts")


def test_nova_agentic_system_renders() -> None:
    result = load_template(NOVA_PROMPTS / "agentic" / "system.jinja", {})
    assert "neuroradiologist" in result


def test_nova_agentic_task_renders() -> None:
    ctx = {
        "width": 512,
        "height": 384,
        "img_path": "/data/brain.png",
        "clinical_history": "Headache for 3 weeks",
        "enable_visual_tools": True,
        "enable_web_search": True,
    }
    result = load_template(NOVA_PROMPTS / "agentic" / "task.jinja", ctx)
    assert "512" in result
    assert "/data/brain.png" in result
    assert "<clinical_history>" in result
    assert "Headache for 3 weeks" in result


def test_nova_agentic_task_requires_width() -> None:
    with pytest.raises(TemplateError):
        load_template(NOVA_PROMPTS / "agentic" / "task.jinja", {})


def test_nova_single_turn_system_renders() -> None:
    result = load_template(NOVA_PROMPTS / "single_turn" / "system.jinja", {})
    assert "neuroradiologist" in result
    assert "Tools are not available in this mode" in result


def test_nova_single_turn_task_renders() -> None:
    ctx = {
        "width": 512,
        "height": 384,
        "image_id": "case_001.png",
        "img_path": "/data/case_001.png",
    }
    result = load_template(NOVA_PROMPTS / "single_turn" / "task.jinja", ctx)
    assert "512" in result
    assert "case_001.png" in result
    assert "No specific clinical history provided" in result


def test_nova_single_turn_task_with_clinical_history() -> None:
    ctx = {
        "width": 512,
        "height": 384,
        "image_id": "case_001.png",
        "img_path": "/data/case_001.png",
        "clinical_history": "Seizures and confusion",
    }
    result = load_template(NOVA_PROMPTS / "single_turn" / "task.jinja", ctx)
    assert "<clinical_history>" in result
    assert "Seizures and confusion" in result


def test_nova_single_turn_task_requires_width() -> None:
    with pytest.raises(TemplateError):
        load_template(NOVA_PROMPTS / "single_turn" / "task.jinja", {})
