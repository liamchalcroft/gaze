"""Audit tests for PubMedQA example.

Covers the findings from the PubMedQA audit:
1. Regex fallback ordering (last-match, not first)
2. Gold-answer key fallback in reward
3. Normalization parity between reward and evaluation
4. CLI failure accounting (no silent maybe defaults)
5. Unlabeled config guard
"""

from __future__ import annotations

import pytest

from examples.pubmedqa.src.evaluation import evaluate_pubmedqa
from examples.pubmedqa.src.processor import PubmedQAVerifiersReward
from examples.pubmedqa.src.schemas import normalize_pubmedqa_answer

# ---------------------------------------------------------------------------
# Finding 4: Regex fallback should take LAST match, not first
# ---------------------------------------------------------------------------


class TestRegexFallbackOrdering:
    """The regex fallback must extract the *last* yes/no/maybe in the text,
    since models typically state reasoning before the answer."""

    def setup_method(self) -> None:
        self.reward = PubmedQAVerifiersReward()

    def test_last_match_wins_yes_then_no(self) -> None:
        text = (
            "The question asks if treatment works. While some studies say yes, "
            "the overall evidence says no."
        )
        result = self.reward._extract_json_response(text)
        assert result is not None
        assert result["answer"] == "no"

    def test_last_match_wins_no_then_yes(self) -> None:
        text = "Initial reports said no, but subsequent trials confirmed yes."
        result = self.reward._extract_json_response(text)
        assert result is not None
        assert result["answer"] == "yes"

    def test_last_match_wins_maybe_after_yes(self) -> None:
        text = "Some say yes, others no, so the answer is maybe."
        result = self.reward._extract_json_response(text)
        assert result is not None
        assert result["answer"] == "maybe"

    def test_json_preferred_over_regex(self) -> None:
        text = '{"answer": "no", "continue": false}'
        result = self.reward._extract_json_response(text)
        assert result is not None
        assert result["answer"] == "no"

    def test_no_match_returns_none(self) -> None:
        result = self.reward._extract_json_response("Inconclusive results were found.")
        assert result is None


# ---------------------------------------------------------------------------
# Finding 2: Reward function gold-answer key fallback
# ---------------------------------------------------------------------------


class TestGoldAnswerKeyFallback:
    """The reward function should find the gold answer under multiple key names."""

    def setup_method(self) -> None:
        self.reward = PubmedQAVerifiersReward()

    def _make_completion(self, answer: str) -> str:
        return f'{{"answer": "{answer}", "confidence": 0.9, "reasoning": "test", "key_evidence": [], "continue": false}}'

    def test_answer_key(self) -> None:
        score = self.reward(
            prompt="",
            completion=self._make_completion("yes"),
            info={"answer": "yes"},
        )
        assert score == 1.0

    def test_gold_answer_key(self) -> None:
        score = self.reward(
            prompt="",
            completion=self._make_completion("yes"),
            info={"gold_answer": "yes"},
        )
        assert score == 1.0

    def test_gold_key(self) -> None:
        score = self.reward(
            prompt="",
            completion=self._make_completion("no"),
            info={"gold": "no"},
        )
        assert score == 1.0

    def test_no_gold_key_returns_zero(self) -> None:
        score = self.reward(
            prompt="",
            completion=self._make_completion("yes"),
            info={},
        )
        assert score == 0.0

    def test_answer_key_takes_priority(self) -> None:
        """When multiple keys exist, 'answer' should be used."""
        score = self.reward(
            prompt="",
            completion=self._make_completion("yes"),
            info={"answer": "yes", "gold_answer": "no"},
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# Finding 5: Normalization parity between reward and evaluation
# ---------------------------------------------------------------------------


ALL_SYNONYMS = [
    ("yes", "yes"),
    ("y", "yes"),
    ("true", "yes"),
    ("positive", "yes"),
    ("no", "no"),
    ("n", "no"),
    ("false", "no"),
    ("negative", "no"),
    ("maybe", "maybe"),
    ("uncertain", "maybe"),
    ("unclear", "maybe"),
    ("unknown", "maybe"),
    # Unknown values should pass through unchanged
    ("something_else", "something_else"),
]


class TestNormalizationParity:
    """evaluation._normalize_answer and reward normalization must agree
    because they both import from schemas.normalize_pubmedqa_answer."""

    def setup_method(self) -> None:
        self.reward = PubmedQAVerifiersReward()

    @pytest.mark.parametrize("raw,expected", ALL_SYNONYMS)
    def test_shared_normalization(self, raw: str, expected: str) -> None:
        assert normalize_pubmedqa_answer(raw) == expected

    @pytest.mark.parametrize("raw,expected", ALL_SYNONYMS)
    def test_evaluation_uses_shared_normalization(self, raw: str, expected: str) -> None:
        """Evaluation normalizes the same way as the canonical function."""
        # Pass raw as both prediction and reference; if normalization
        # maps to the same canonical form, accuracy should be 1.0
        metrics = evaluate_pubmedqa([raw], [expected])
        assert metrics["accuracy"] == 1.0

    @pytest.mark.parametrize("raw,expected", ALL_SYNONYMS)
    def test_reward_normalization_matches(self, raw: str, expected: str) -> None:
        """Reward extracts and normalizes identically to evaluation."""
        completion = f'{{"answer": "{raw}", "confidence": 0.9, "reasoning": "t", "key_evidence": [], "continue": false}}'
        score = self.reward(prompt="", completion=completion, info={"answer": expected})
        assert score == 1.0


# ---------------------------------------------------------------------------
# Finding 3: pqa_unlabeled has no ground truth
# ---------------------------------------------------------------------------


class TestUnlabeledGuard:
    """Samples with None final_decision should produce empty string, not 'none'."""

    @staticmethod
    def _make_stub_dataset() -> PubmedQADataset:
        """Create a PubmedQADataset without hitting the network."""
        from examples.pubmedqa.src.dataset import PubmedQADataset

        obj = PubmedQADataset.__new__(PubmedQADataset)
        obj.config = "pqa_labeled"  # type: ignore[attr-defined]
        return obj

    def test_none_final_decision_becomes_empty_string(self) -> None:
        """Simulate what _transform_sample does with None final_decision."""
        ds = self._make_stub_dataset()
        raw_item = {
            "pubid": 123,
            "question": "Test?",
            "context": {"contexts": [], "labels": [], "meshes": []},
            "long_answer": "",
            "final_decision": None,
        }
        transformed = ds._transform_sample(raw_item)
        assert transformed["answer"] == ""
        assert transformed["answer"] != "none"

    def test_valid_final_decision_preserved(self) -> None:
        ds = self._make_stub_dataset()
        raw_item = {
            "pubid": 456,
            "question": "Test?",
            "context": {"contexts": [], "labels": [], "meshes": []},
            "long_answer": "",
            "final_decision": "yes",
        }
        transformed = ds._transform_sample(raw_item)
        assert transformed["answer"] == "yes"


# ---------------------------------------------------------------------------
# Finding 1 + 7: CLI failure accounting
# ---------------------------------------------------------------------------


class TestCLIFailureAccounting:
    """Failures must be excluded from metrics, not silently defaulted to 'maybe'."""

    def test_evaluate_empty_raises(self) -> None:
        """If all samples fail, evaluate_pubmedqa should raise on empty input."""
        with pytest.raises(ValueError, match="Cannot evaluate empty"):
            evaluate_pubmedqa([], [])

    def test_failures_dont_inflate_maybe_accuracy(self) -> None:
        """If we have predictions without the failed-sample defaults,
        maybe-class accuracy is not inflated by failures."""
        # 3 real predictions, 2 of which are correct
        preds = ["yes", "no", "maybe"]
        refs = ["yes", "yes", "maybe"]
        metrics = evaluate_pubmedqa(preds, refs)
        # 2/3 correct
        assert abs(metrics["accuracy"] - 2 / 3) < 1e-9
        # maybe accuracy should be 1.0 (1 correct out of 1 maybe ref)
        assert metrics["accuracy_maybe"] == 1.0
        # no class has zero support — no inflated maybe
        assert metrics["support_maybe"] == 1.0


# ---------------------------------------------------------------------------
# Integration: reward on message-list completion format
# ---------------------------------------------------------------------------


class TestRewardMessageListFormat:
    """Reward must handle verifiers message-list completions, not just strings."""

    def setup_method(self) -> None:
        self.reward = PubmedQAVerifiersReward()

    def test_message_list_completion(self) -> None:
        completion = [
            {
                "role": "assistant",
                "content": '{"answer": "yes", "confidence": 0.9, "reasoning": "test", "key_evidence": [], "continue": false}',
            }
        ]
        score = self.reward(prompt="", completion=completion, info={"answer": "yes"})
        assert score == 1.0

    def test_multimodal_content_list(self) -> None:
        completion = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking..."},
                    {
                        "type": "text",
                        "text": '{"answer": "no", "confidence": 0.8, "reasoning": "test", "key_evidence": [], "continue": false}',
                    },
                ],
            }
        ]
        score = self.reward(prompt="", completion=completion, info={"answer": "no"})
        assert score == 1.0

    def test_prose_fallback_in_message_list(self) -> None:
        completion = [
            {
                "role": "assistant",
                "content": "After reviewing the evidence, the answer is no.",
            }
        ]
        score = self.reward(prompt="", completion=completion, info={"answer": "no"})
        assert score == 1.0
