"""Tests for image search validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.exceptions import HarnessError
from radiant_harness.retrieval.image_search import ImageDownloadError
from radiant_harness.retrieval.image_search import ImageSearchError
from radiant_harness.retrieval.image_search import MedicalImageSearchManager
from radiant_harness.retrieval.image_search import OpenISearchEngine


def test_invalid_image_search_limits_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_results_per_engine"):
        MedicalImageSearchManager(max_results_per_engine=0, download_dir=tmp_path)
    with pytest.raises(ValueError, match="rate_limit_delay"):
        MedicalImageSearchManager(rate_limit_delay=-0.1, download_dir=tmp_path)


class TestImageSearchErrorHierarchy:
    """Image search errors must be part of the HarnessError hierarchy."""

    def test_image_search_error_is_harness_error(self) -> None:
        err = ImageSearchError("Open-i", "test error")
        assert isinstance(err, HarnessError)

    def test_image_download_error_is_harness_error(self) -> None:
        err = ImageDownloadError("https://example.com/img.png", "timeout")
        assert isinstance(err, HarnessError)

    def test_image_search_error_preserves_fields(self) -> None:
        cause = RuntimeError("cause")
        err = ImageSearchError("Open-i", "search failed", original_error=cause)
        assert err.engine_name == "Open-i"
        assert err.original_error is cause

    def test_image_download_error_preserves_fields(self) -> None:
        cause = RuntimeError("cause")
        err = ImageDownloadError("https://example.com/img.png", "fail", original_error=cause)
        assert err.url == "https://example.com/img.png"
        assert err.original_error is cause


class TestOpenIResultParsing:
    """Tests for Open-i result parsing."""

    def test_modality_extraction(self) -> None:
        engine = OpenISearchEngine()
        assert engine._extract_modality("Brain MRI T2-weighted") == "MRI"
        assert engine._extract_modality("Chest CT scan") == "CT"
        assert engine._extract_modality("Lateral X-ray") == "X-ray"
        assert engine._extract_modality("No modality here") is None

    def test_body_part_extraction(self) -> None:
        engine = OpenISearchEngine()
        assert engine._extract_body_part("Brain lesion") == "brain"
        assert engine._extract_body_part("Chest radiograph") == "chest"
        assert engine._extract_body_part("Spinal cord") == "spine"
        assert engine._extract_body_part("No body part here") is None

    def test_parse_results_skips_no_image(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {"title": "No image result"},
                {
                    "title": "With image",
                    "imgLarge": "https://openi.nlm.nih.gov/img.jpg",
                    "imgThumb": "https://openi.nlm.nih.gov/thumb.jpg",
                    "pmcid": "PMC123",
                },
            ]
        }
        results = engine._parse_results(data)
        assert len(results) == 1
        assert results[0].title == "With image"

    def test_parse_results_absolute_urls(self) -> None:
        engine = OpenISearchEngine()
        data = {
            "list": [
                {
                    "title": "Relative URL",
                    "imgLarge": "/images/test.jpg",
                    "imgThumb": "/thumbs/test.jpg",
                    "pmcid": "PMC456",
                },
            ]
        }
        results = engine._parse_results(data)
        assert results[0].image_url.startswith("https://")
        assert results[0].thumbnail_url is not None
        assert results[0].thumbnail_url.startswith("https://")
