from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.prompts import combine_prompts
from radiant_harness.prompts import create_prompt


def _write_template(dir_path: Path, name: str, content: str) -> None:
    (dir_path / name).write_text(content, encoding="utf-8")


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


def test_combine_prompts_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        combine_prompts("system", "task", "unknown")
