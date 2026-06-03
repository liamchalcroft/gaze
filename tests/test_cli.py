"""Tests for the ``gaze`` console entry point."""

from __future__ import annotations

import pytest

import gaze
from gaze import __main__


def test_bare_invocation_prints_info(capsys: pytest.CaptureFixture[str]) -> None:
    rc = __main__.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"GAZE {gaze.__version__}" in out
    assert "github.com/liamchalcroft/gaze" in out


def test_info_command_prints_info(capsys: pytest.CaptureFixture[str]) -> None:
    rc = __main__.main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "GAZE" in out


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        __main__.main(["--version"])
    assert exc_info.value.code == 0
    assert gaze.__version__ in capsys.readouterr().out


def test_build_info_reports_adapters_and_tools() -> None:
    info = __main__._build_info()
    assert "OpenAIAdapter" in info
    assert "LMStudioAdapter" in info
    assert "visual" in info
    assert "search" in info
