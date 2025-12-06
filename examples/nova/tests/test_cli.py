"""CLI and orchestration level tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova_retrieval_vlm.cli import save_per_subject_results
from nova_retrieval_vlm.config import AgenticConfig
from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.config import ModelConfig
from nova_retrieval_vlm.config import PathsConfig
from nova_retrieval_vlm.config import VisualizationConfig
from nova_retrieval_vlm.types import JSONParseError
from nova_retrieval_vlm.types import ModelResponse


def _make_config(output_dir: Path) -> Config:
    """Create a minimal Config pointing at temp directories."""
    paths = PathsConfig(data_dir=str(output_dir / "data"), output_dir=str(output_dir))
    return Config(
        model=ModelConfig(name="test-model"),
        paths=paths,
        agentic=AgenticConfig(enabled=False),
        visualization=VisualizationConfig(),
        max_iterations=1,
        batch_size=1,
        skip_existing=False,
    )


def test_save_per_subject_results_persists_valid_json(tmp_path: Path) -> None:
    """Valid JSON response is parsed and persisted to per-subject folders."""
    config = _make_config(tmp_path / "out")
    output_dir = Path(config.paths.output_dir)

    response_payload = {
        "caption": {"description": "ok"},
        "diagnosis": {"primary_diagnosis": "normal"},
        "localization": {"localizations": []},
    }
    responses = [ModelResponse(text=json.dumps(response_payload), confidence=0.7)]

    save_per_subject_results(batch_idx=0, responses=responses, config=config, output_dir=output_dir)

    subject_dir = output_dir / "per_subject" / "subject_0000"
    with (subject_dir / "predictions.json").open() as f:
        saved = json.load(f)

    assert saved["subject_id"] == 0
    assert saved["caption"] == response_payload["caption"]
    assert saved["diagnosis"] == response_payload["diagnosis"]
    assert saved["localization"] == response_payload["localization"]


def test_save_per_subject_results_rejects_invalid_json(tmp_path: Path) -> None:
    """Non-JSON responses are rejected immediately to avoid silent slop."""
    config = _make_config(tmp_path / "out_invalid")
    output_dir = Path(config.paths.output_dir)
    responses = [ModelResponse(text="not-json", confidence=0.1)]

    with pytest.raises(JSONParseError):
        save_per_subject_results(batch_idx=0, responses=responses, config=config, output_dir=output_dir)
