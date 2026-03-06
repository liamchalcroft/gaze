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
    }
    result = load_template(NOVA_PROMPTS / "agentic" / "task.jinja", ctx)
    assert "512" in result
    assert "<clinical_history>" in result
    assert "Headache for 3 weeks" in result


def test_nova_agentic_task_requires_width() -> None:
    with pytest.raises(TemplateError):
        load_template(NOVA_PROMPTS / "agentic" / "task.jinja", {})


def test_nova_single_turn_system_renders() -> None:
    result = load_template(NOVA_PROMPTS / "single_turn" / "system.jinja", {})
    assert "neuroradiologist" in result
    # Single-turn template should not hardcode tool availability claims;
    # tool docs are injected by _run_analysis() based on actual config.
    assert "single response" in result


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


# ---------------------------------------------------------------------------
# Integration: combine_prompts no longer appends trailing instructions
# ---------------------------------------------------------------------------


def test_combine_prompts_agentic_no_trailing_begin() -> None:
    """combine_prompts should NOT append its own 'Begin' instruction."""
    result = combine_prompts("SYS", "TASK", "agentic")
    # The result should end with the closing tag, not a redundant instruction
    assert result.rstrip().endswith("</agentic_analysis_instructions>")


def test_combine_prompts_single_turn_no_trailing_instruction() -> None:
    """combine_prompts should NOT append 'Provide your complete analysis'."""
    result = combine_prompts("SYS", "TASK", "single_turn")
    assert result.rstrip().endswith("</analysis_instructions>")


# ---------------------------------------------------------------------------
# Integration: NOVA agentic template does not duplicate POLICY content
# ---------------------------------------------------------------------------


def test_nova_agentic_task_no_duplicate_continue_instructions() -> None:
    """NOVA task template should not have its own multi-turn strategy section.

    The POLICY block injected by _run_analysis() is the canonical source
    for 'continue' field instructions.
    """
    ctx = {"width": 512, "height": 384}
    result = load_template(NOVA_PROMPTS / "agentic" / "task.jinja", ctx)
    # The template should not contain a standalone "Multi-turn Strategy" section
    assert "**Multi-turn Strategy:**" not in result
    # But the output format JSON example should still reference 'continue'
    assert '"continue"' in result


def test_nova_agentic_task_no_hardcoded_search_tool_signatures() -> None:
    """NOVA task template should not hardcode search tool parameter signatures.

    Tool documentation is auto-injected by _run_analysis().
    """
    ctx = {
        "width": 512,
        "height": 384,
        "enable_visual_tools": True,
        "enable_web_search": True,
    }
    result = load_template(NOVA_PROMPTS / "agentic" / "task.jinja", ctx)
    assert "search_web(query, search_type)" not in result
    assert "search_images(query, modality, body_part)" not in result


# ---------------------------------------------------------------------------
# Integration: tool documentation auto-generation
# ---------------------------------------------------------------------------


def test_tool_documenter_generates_nonempty_docs() -> None:
    """ToolDocumenter.generate_prompt_documentation() returns non-empty string
    when tools are registered — this is what _run_analysis() injects.
    """
    from radiant_harness.tools import create_visual_tools

    tools = create_visual_tools()
    from radiant_harness.tools.registry import ToolDocumenter

    documenter = ToolDocumenter(tools)
    docs = documenter.generate_prompt_documentation()
    assert len(docs) > 100
    # Should mention at least a few core tools
    assert "zoom" in docs
    assert "crop" in docs
    assert "reset" in docs


def test_tool_documenter_covers_all_visual_tools() -> None:
    """Auto-generated tool docs should include ALL registered visual tools."""
    from radiant_harness.tools import create_visual_tools
    from radiant_harness.tools.registry import ToolDocumenter

    tools = create_visual_tools()
    documenter = ToolDocumenter(tools)
    docs = documenter.generate_prompt_documentation()
    for tool in tools:
        assert tool.name in docs, f"Tool '{tool.name}' missing from auto-generated docs"


def test_critical_tools_in_nova_template_guidance() -> None:
    """NOVA agentic task template's Tool Strategy should reference safety-critical tools.

    Tools that modify coordinate/intensity space need explicit guidance
    about reset() and coordinate system implications.
    """
    ctx = {"width": 512, "height": 384, "enable_visual_tools": True}
    result = load_template(NOVA_PROMPTS / "agentic" / "task.jinja", ctx)
    # Coordinate-modifying tools should be mentioned with reset warnings
    for tool in ("crop", "zoom", "symmetry_diff", "detect_edges", "threshold"):
        assert tool in result, f"Critical tool '{tool}' missing from NOVA template guidance"
    # Reset instruction must be present
    assert "reset()" in result


# ---------------------------------------------------------------------------
# Integration: NOVA processor context variable contract
# ---------------------------------------------------------------------------


def test_nova_agentic_renders_with_processor_context() -> None:
    """Simulate the full context dict that NOVAAgenticProcessor passes."""
    ctx = {
        "width": 512,
        "height": 384,
        "image_id": "case_001.png",
        "img_path": "/data/case_001.png",
        "images": [],
        "enable_visual_tools": True,
        "enable_web_search": True,
        "clinical_history": "Progressive headache",
        "has_images": True,
        "num_images": 1,
        "policy": {"max_turns": 10, "requires_continue": True},
    }
    result = create_prompt(
        prompts_dir=NOVA_PROMPTS,
        mode="agentic",
        context=ctx,
        template_name="task.jinja",
    )
    assert "neuroradiologist" in result
    assert "512" in result
    assert "Progressive headache" in result
    assert "Coordinate Space Warning" in result


def test_nova_single_turn_renders_with_processor_context() -> None:
    """Simulate the full context dict for single-turn mode."""
    ctx = {
        "width": 256,
        "height": 256,
        "image_id": "case_002.png",
        "img_path": "/data/case_002.png",
        "images": [],
        "enable_visual_tools": False,
        "enable_web_search": False,
        "clinical_history": "",
        "has_images": True,
        "num_images": 1,
        "policy": {"max_turns": 1, "requires_continue": False},
    }
    result = create_prompt(
        prompts_dir=NOVA_PROMPTS,
        mode="single_turn",
        context=ctx,
        template_name="task.jinja",
    )
    assert "neuroradiologist" in result
    assert "256" in result
    assert '"continue": false' in result
