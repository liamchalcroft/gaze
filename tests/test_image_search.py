"""Tests for image search validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.retrieval.image_search import MedicalImageSearchManager


def test_invalid_image_search_limits_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_results_per_engine"):
        MedicalImageSearchManager(max_results_per_engine=0, download_dir=tmp_path)
    with pytest.raises(ValueError, match="rate_limit_delay"):
        MedicalImageSearchManager(rate_limit_delay=-0.1, download_dir=tmp_path)
