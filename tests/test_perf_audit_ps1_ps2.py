"""Tests for performance audit Patch Set #1 (Memory & Encoding) and #2 (Ranking Regex).

Patch Set #1:
- EncodedImage pre-computes _data_url in __post_init__ (no per-call allocation)
- ImageManager.set_preloaded_image transfer_ownership=True skips copy

Patch Set #2:
- MEDICAL_ENTITY_PATTERNS pre-compiled as re.Pattern objects
- _rank_results uses pre-computed entity_in_query set instead of per-entity regex
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.registry import EncodedImage
from radiant_harness.tools.registry import encode_image

# =====================================================================
# Patch Set #1: EncodedImage _data_url pre-computation
# =====================================================================


class TestEncodedImageDataUrlCached:
    """Verify to_data_url() returns the pre-computed string, not a new one."""

    def test_data_url_is_precomputed_in_post_init(self) -> None:
        """_data_url should be set during __post_init__, not lazily."""
        enc = EncodedImage(data="abc123", mime_type="image/jpeg")
        # Access the private field directly to confirm it was pre-set
        assert enc._data_url == "data:image/jpeg;base64,abc123"

    def test_to_data_url_returns_same_object(self) -> None:
        """Repeated calls to to_data_url() must return the exact same str object."""
        enc = EncodedImage(data="xyz", mime_type="image/png")
        url1 = enc.to_data_url()
        url2 = enc.to_data_url()
        # `is` check: same object identity, not just equality
        assert url1 is url2

    def test_encode_image_returns_precomputed_url(self) -> None:
        """encode_image → EncodedImage should have a valid precomputed data URL."""
        img = Image.new("RGB", (8, 8), color=(100, 100, 100))
        result = encode_image(img)
        url = result.to_data_url()
        assert url.startswith("data:image/jpeg;base64,")
        # Confirm it matches the manual construction
        assert url == f"data:{result.mime_type};base64,{result.data}"

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        """EncodedImage is frozen — attributes must not be mutable."""
        enc = EncodedImage(data="abc", mime_type="image/jpeg")
        with pytest.raises(AttributeError):
            enc.data = "mutated"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            enc._data_url = "mutated"  # type: ignore[misc]


# =====================================================================
# Patch Set #1: transfer_ownership in set_preloaded_image
# =====================================================================


def _create_image(tmp_path: Path, name: str = "test.png", size: tuple[int, int] = (50, 50)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


class TestTransferOwnership:
    """Verify transfer_ownership=True avoids a redundant PIL copy."""

    def test_transfer_ownership_true_reuses_input(self, tmp_path: Path) -> None:
        """With transfer_ownership=True, _original_image IS the input image."""
        path = _create_image(tmp_path, size=(60, 60))
        img = Image.new("RGB", (60, 60), color=(255, 0, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=True)

        # Original should be the exact same object (no copy)
        assert mgr._original_image is img  # noqa: SLF001
        # Current should be a copy of original (independent)
        assert mgr.current_image is not img
        assert mgr.current_image is not None
        assert mgr.current_image.size == (60, 60)

    def test_transfer_ownership_false_copies_input(self, tmp_path: Path) -> None:
        """With transfer_ownership=False (default), _original_image is a copy."""
        path = _create_image(tmp_path, size=(60, 60))
        img = Image.new("RGB", (60, 60), color=(0, 255, 0))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=False)

        # Original should NOT be the same object
        assert mgr._original_image is not img  # noqa: SLF001

    def test_transfer_ownership_default_is_false(self, tmp_path: Path) -> None:
        """Default behavior (no kwarg) should copy the input."""
        path = _create_image(tmp_path, size=(40, 40))
        img = Image.new("RGB", (40, 40), color=(0, 0, 255))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path)

        assert mgr._original_image is not img  # noqa: SLF001

    def test_transfer_ownership_reset_still_works(self, tmp_path: Path) -> None:
        """Reset should work correctly after transfer_ownership=True."""
        path = _create_image(tmp_path, size=(80, 80))
        img = Image.new("RGB", (80, 80), color=(128, 128, 128))

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=True)

        mgr.transform_image(lambda i: i.resize((20, 20)))
        assert mgr.current_image is not None
        assert mgr.current_image.size == (20, 20)

        mgr.reset_to_original()
        assert mgr.current_image is not None
        assert mgr.current_image.size == (80, 80)

    def test_transfer_ownership_original_encoding_preserved(self, tmp_path: Path) -> None:
        """original_encoding should be settable after transfer_ownership."""
        path = _create_image(tmp_path, size=(32, 32))
        img = Image.new("RGB", (32, 32))
        enc = encode_image(img)

        mgr = ImageManager()
        mgr.set_preloaded_image(img, path, transfer_ownership=True)
        mgr.original_encoding = enc

        assert mgr.original_encoding is enc


# =====================================================================
# Patch Set #2: Pre-compiled entity patterns
# =====================================================================


class TestPrecompiledEntityPatterns:
    """Verify MEDICAL_ENTITY_PATTERNS are pre-compiled re.Pattern objects."""

    def test_patterns_are_compiled(self) -> None:
        """All entries in MEDICAL_ENTITY_PATTERNS should be re.Pattern."""
        import re

        from radiant_harness.retrieval.web_search import PubMedSearchEngine

        patterns = PubMedSearchEngine.MEDICAL_ENTITY_PATTERNS
        assert len(patterns) > 0
        for pat in patterns:
            assert isinstance(pat, re.Pattern), f"Expected re.Pattern, got {type(pat)}"

    def test_extract_entities_still_correct(self) -> None:
        """Pre-compiled patterns must produce the same entity extraction."""
        from radiant_harness.retrieval.web_search import PubMedSearchEngine

        engine = PubMedSearchEngine()
        entities = engine._extract_medical_entities("Brain MRI shows tumor with edema near cortex")
        assert isinstance(entities, tuple)
        assert "mri" in entities
        assert "tumor" in entities
        assert "edema" in entities
        assert "cortex" in entities


# =====================================================================
# Patch Set #2: entity_in_query set-based lookup correctness
# =====================================================================


class TestEntityInQuerySetLookup:
    """Verify _rank_results entity matching uses set lookup correctly."""

    def test_entity_match_word_boundary_preserved(self) -> None:
        """Entity 'ct' must NOT match query 'duct ectasia' (word boundary)."""
        from radiant_harness.retrieval.web_search import SearchResult
        from radiant_harness.retrieval.web_search import WebSearchManager

        manager = WebSearchManager()

        result_with_ct = SearchResult(
            title="CT imaging study",
            url="https://pubmed.ncbi.nlm.nih.gov/100/",
            content="CT scan findings.",
            snippet="CT",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=("ct",),
        )
        result_no_entity = SearchResult(
            title="Other study",
            url="https://pubmed.ncbi.nlm.nih.gov/101/",
            content="Other study content.",
            snippet="Other",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=(),
        )

        # "duct ectasia" contains "ct" as substring but NOT as whole word
        ranked = manager._rank_results(
            [result_with_ct, result_no_entity],
            query="duct ectasia",
            search_type="general",
        )
        # Entity "ct" should NOT get a match boost
        assert ranked[0].ranking_score == ranked[1].ranking_score

    def test_entity_match_positive_case(self) -> None:
        """Entity 'tumor' should match query containing 'tumor'."""
        from radiant_harness.retrieval.web_search import SearchResult
        from radiant_harness.retrieval.web_search import WebSearchManager

        manager = WebSearchManager()

        with_entity = SearchResult(
            title="Tumor imaging",
            url="https://pubmed.ncbi.nlm.nih.gov/200/",
            content="Tumor analysis.",
            snippet="T",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=("tumor",),
        )
        without_entity = SearchResult(
            title="Tumor imaging",
            url="https://pubmed.ncbi.nlm.nih.gov/201/",
            content="Tumor analysis.",
            snippet="T",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=(),
        )

        ranked = manager._rank_results(
            [without_entity, with_entity],
            query="brain tumor diagnosis",
            search_type="general",
        )

        # Result with entity match should rank higher
        assert ranked[0].url.endswith("/200/")
        assert ranked[0].ranking_score > ranked[1].ranking_score

    def test_multiple_entities_accumulate(self) -> None:
        """Multiple entity matches should produce higher score than single."""
        from radiant_harness.retrieval.web_search import SearchResult
        from radiant_harness.retrieval.web_search import WebSearchManager

        manager = WebSearchManager()

        multi = SearchResult(
            title="Study A",
            url="https://pubmed.ncbi.nlm.nih.gov/300/",
            content="Content A.",
            snippet="A",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=("tumor", "edema", "mri"),
        )
        single = SearchResult(
            title="Study B",
            url="https://pubmed.ncbi.nlm.nih.gov/301/",
            content="Content B.",
            snippet="B",
            source="pubmed",
            reliability_score=0.95,
            extracted_entities=("tumor",),
        )

        ranked = manager._rank_results(
            [single, multi],
            query="tumor edema mri findings",
            search_type="general",
        )

        # Multi-entity result should rank higher
        assert ranked[0].url.endswith("/300/")
        assert ranked[0].ranking_score > ranked[1].ranking_score
