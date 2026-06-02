"""Tests for negation-aware clinical term detection in caption evaluation.

Covers:
1. _has_clinical_terms() negation window logic
2. Binary classification consistency with abnormal_prevalence
"""

from __future__ import annotations

import pytest

# caption.py has heavy module-level imports (nltk, sacrebleu, bert_score).
# Skip the whole module when those are missing (CI without --extra nova).
nltk = pytest.importorskip("nltk", reason="nltk required for caption evaluation tests")
sacrebleu = pytest.importorskip("sacrebleu", reason="sacrebleu required")

from examples.nova.src.evaluation.caption import _has_clinical_terms  # noqa: E402


class TestHasClinicalTermsNegation:
    """Verify negation-aware clinical term detection."""

    # --- Positive cases (should be classified as abnormal) ---

    def test_bare_clinical_term(self) -> None:
        """A clinical term without negation is abnormal."""
        assert _has_clinical_terms("Large tumor in the right temporal lobe")

    def test_multiple_clinical_terms(self) -> None:
        assert _has_clinical_terms("Hemorrhage with surrounding edema")

    def test_clinical_term_with_distant_negation(self) -> None:
        """Negation word beyond the window should not suppress the term."""
        # Window is 3 tokens. "no" is 5 tokens before "lesion".
        assert _has_clinical_terms("no significant change but new lesion noted")

    def test_one_negated_one_not(self) -> None:
        """If one term is negated but another isn't, still abnormal."""
        assert _has_clinical_terms("no hemorrhage but large tumor present")

    # --- Negative cases (should be classified as normal) ---

    def test_negated_with_no(self) -> None:
        assert not _has_clinical_terms("no evidence of tumor")

    def test_negated_with_not(self) -> None:
        assert not _has_clinical_terms("tumor is not seen")

    def test_negated_with_without(self) -> None:
        assert not _has_clinical_terms("without hemorrhage or edema")

    def test_negated_with_absent(self) -> None:
        assert not _has_clinical_terms("lesion absent on follow-up")

    def test_negated_with_normal(self) -> None:
        assert not _has_clinical_terms("normal brain, no abnormality")

    def test_negated_with_negative(self) -> None:
        assert not _has_clinical_terms("negative for metastasis")

    def test_unremarkable(self) -> None:
        assert not _has_clinical_terms("unremarkable enhancement pattern")

    def test_no_clinical_terms_at_all(self) -> None:
        assert not _has_clinical_terms("Normal brain MRI, axial T2 FLAIR")

    # --- Edge cases ---

    def test_empty_string(self) -> None:
        assert not _has_clinical_terms("")

    def test_punctuation_around_term(self) -> None:
        """Clinical term with surrounding punctuation should still match."""
        assert _has_clinical_terms("findings: tumor, large")

    def test_negation_with_punctuation(self) -> None:
        """Negation word with punctuation should still negate."""
        assert not _has_clinical_terms("no, tumor is not present")

    def test_window_boundary_exactly_at_limit(self) -> None:
        """Negation exactly _NEGATION_WINDOW (3) tokens before the term."""
        # "no" is 3 tokens before "tumor": no [1] evidence [2] of [3] tumor
        assert not _has_clinical_terms("no evidence of tumor")

    def test_window_boundary_one_beyond(self) -> None:
        """Negation 4 tokens before the term — outside window."""
        # "no" is 4 tokens before "lesion": no [1] real [2] clear [3] visible [4] lesion
        assert _has_clinical_terms("no real clear visible lesion")


class TestBinaryClassificationConsistency:
    """Verify binary_accuracy/F1 and abnormal_prevalence use the same definition."""

    @pytest.fixture
    def _mock_heavy_deps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub out heavy dependencies (BERTScore, METEOR, sacrebleu)."""
        import types

        torch = pytest.importorskip("torch")

        def fake_corpus_bleu(preds, refs, **kwargs):  # noqa: ARG001
            result = types.SimpleNamespace()
            result.score = 50.0
            return result

        def fake_bert_score(**kwargs):  # noqa: ARG001
            n = len(kwargs.get("cands", []))
            hash_str = "roberta-large_L17_no-idf_version=0.3.13(hug_trans=4.52.4)-rescaled"
            return ((torch.zeros(n), torch.zeros(n), torch.zeros(n)), hash_str)

        monkeypatch.setattr(
            "examples.nova.src.evaluation.caption.sacrebleu.corpus_bleu", fake_corpus_bleu
        )
        monkeypatch.setattr("examples.nova.src.evaluation.caption.bert_score_fn", fake_bert_score)
        monkeypatch.setattr("examples.nova.src.evaluation.caption._ensure_nltk_data", lambda: True)
        monkeypatch.setattr("examples.nova.src.evaluation.caption.RADGRAPH_AVAILABLE", False)

        def fake_meteor(refs, pred):  # noqa: ARG001
            return 0.5

        monkeypatch.setattr("examples.nova.src.evaluation.caption.nltk_meteor_score", fake_meteor)
        monkeypatch.setattr(
            "examples.nova.src.evaluation.caption._cached_word_tokenize",
            lambda t: tuple(t.lower().split()),
        )

    @pytest.mark.usefixtures("_mock_heavy_deps")
    def test_negated_ref_classified_as_normal(self) -> None:
        """A reference with negated clinical terms should be classified as normal.

        Before the fix, binary_accuracy used raw set intersection (bool(r_clin))
        which would classify "no evidence of tumor" as abnormal. After the fix,
        it uses _has_clinical_terms() which correctly classifies it as normal.
        """
        from examples.nova.src.evaluation.caption import evaluate_caption

        preds = ["Normal brain MRI"]
        refs = ["No evidence of tumor, normal brain"]

        result = evaluate_caption(preds, refs)

        # Both pred and ref should be classified as "normal" (no non-negated clinical terms)
        assert result["binary_accuracy"] == 1.0, (
            f"Expected binary_accuracy=1.0 (both normal), got {result['binary_accuracy']}"
        )
        assert result["abnormal_prevalence"] == 0.0, (
            f"Expected abnormal_prevalence=0.0, got {result['abnormal_prevalence']}"
        )
