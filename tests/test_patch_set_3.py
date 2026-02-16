"""Tests for Patch Set 3: Prompt safety + keyword F1 tokenization."""

from __future__ import annotations

import pathlib
import re


# ---------------------------------------------------------------------------
# 1. Keyword tokenization helper
# ---------------------------------------------------------------------------
# The caption module imports through evaluation/__init__.py which pulls in
# detection.py → torch. We replicate the helper here to test independently.

_WORD_PATTERN = re.compile(r"\b[\w][\w-]*[\w]\b|\b\w\b")


def _extract_keyword_tokens(text: str) -> set[str]:
    """Local copy of the helper from caption.py for testing without torch."""
    tokens: set[str] = set()
    for match in _WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        tokens.add(word)
        if "-" in word:
            tokens.update(word.split("-"))
    return tokens


class TestExtractKeywordTokens:
    """Tests for _extract_keyword_tokens helper in caption evaluation."""

    def test_simple_words(self) -> None:
        result = _extract_keyword_tokens("axial T2 FLAIR")
        assert "axial" in result
        assert "t2" in result
        assert "flair" in result

    def test_hyphenated_compound_splits(self) -> None:
        """T2-weighted should yield both the compound and its parts."""
        result = _extract_keyword_tokens("T2-weighted")
        assert "t2-weighted" in result
        assert "t2" in result
        assert "weighted" in result

    def test_multiple_hyphenated(self) -> None:
        result = _extract_keyword_tokens("T1-weighted axial FLAIR T2-weighted sagittal")
        # Compounds
        assert "t1-weighted" in result
        assert "t2-weighted" in result
        # Parts
        assert "t1" in result
        assert "t2" in result
        assert "weighted" in result
        # Plain words
        assert "axial" in result
        assert "flair" in result
        assert "sagittal" in result

    def test_case_insensitive(self) -> None:
        result = _extract_keyword_tokens("FLAIR Axial DWI")
        assert "flair" in result
        assert "axial" in result
        assert "dwi" in result

    def test_punctuation_stripped(self) -> None:
        """Punctuation like commas and periods should not be part of tokens."""
        result = _extract_keyword_tokens("lesion, tumor. cyst!")
        assert "lesion" in result
        assert "tumor" in result
        assert "cyst" in result
        # Should NOT contain punctuation-bearing variants
        assert "lesion," not in result
        assert "tumor." not in result

    def test_empty_string(self) -> None:
        result = _extract_keyword_tokens("")
        assert result == set()

    def test_single_char_words(self) -> None:
        """Single character words should be captured by the regex."""
        result = _extract_keyword_tokens("a T2 scan")
        assert "a" in result
        assert "t2" in result
        assert "scan" in result

    def test_clinical_term_matching(self) -> None:
        """Verify that clinical terms from NOVA evaluation are matched."""
        text = "lesion with hemorrhage and edema causing mass effect"
        result = _extract_keyword_tokens(text)
        for term in ["lesion", "hemorrhage", "edema", "mass"]:
            assert term in result, f"Expected '{term}' in tokens"

    def test_modality_term_matching(self) -> None:
        """Verify modality terms from NOVA evaluation are matched, including via hyphen splitting."""
        text = "Axial T2-weighted FLAIR showing coronal DWI"
        result = _extract_keyword_tokens(text)
        expected = {"flair", "t2", "axial", "coronal", "dwi", "weighted"}
        for term in expected:
            assert term in result, f"Expected modality term '{term}' in tokens"


class TestKeywordTokenSourceParity:
    """Verify our local copy matches the real source code."""

    def test_source_contains_identical_regex(self) -> None:
        """The regex and helper in caption.py should match this test's copy."""
        source = (
            pathlib.Path(__file__).resolve().parents[1]
            / "examples"
            / "nova"
            / "src"
            / "evaluation"
            / "caption.py"
        ).read_text()
        assert r'\b[\w][\w-]*[\w]\b|\b\w\b' in source
        assert "def _extract_keyword_tokens(text: str)" in source
        assert 'tokens.update(word.split("-"))' in source


# ---------------------------------------------------------------------------
# 2. Prompt disclaimer and bias checks
# ---------------------------------------------------------------------------

PROMPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "nova" / "src" / "prompts"


class TestPromptDisclaimer:
    """Verify research disclaimer is present in system prompts."""

    def test_single_turn_system_has_disclaimer(self) -> None:
        text = (PROMPT_DIR / "single_turn" / "system.jinja").read_text()
        assert "research benchmark evaluation" in text.lower()
        assert "not a clinical decision-support system" in text.lower()

    def test_agentic_system_has_disclaimer(self) -> None:
        text = (PROMPT_DIR / "agentic" / "system.jinja").read_text()
        assert "research benchmark evaluation" in text.lower()
        assert "not a clinical decision-support system" in text.lower()


class TestPathologyBiasSoftened:
    """Verify prompts use softer language about confirmed pathology."""

    def test_single_turn_task_no_shouting(self) -> None:
        """Task prompt should not use all-caps emphatic statements about pathology."""
        text = (PROMPT_DIR / "single_turn" / "task.jinja").read_text()
        assert "EVERY CASE" not in text
        assert "CONFIRMED PATHOLOGY" not in text
        # Should use softer phrasing
        assert "confirmed abnormalities" in text or "confirmed pathology" in text

    def test_single_turn_task_no_clinical_decision_claim(self) -> None:
        """Should not claim analysis guides clinical decisions."""
        text = (PROMPT_DIR / "single_turn" / "task.jinja").read_text()
        assert "guide clinical decisions" not in text.lower()

    def test_agentic_task_no_shouting(self) -> None:
        text = (PROMPT_DIR / "agentic" / "task.jinja").read_text()
        assert "EVERY CASE" not in text

    def test_agentic_system_benchmark_context(self) -> None:
        """Agentic system prompt should frame as benchmark, not clinical."""
        text = (PROMPT_DIR / "agentic" / "system.jinja").read_text()
        assert "benchmark" in text.lower()
