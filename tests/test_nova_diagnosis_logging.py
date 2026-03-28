"""Tests for NOVA LLM diagnosis judgment logging and majority vote."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Guard imports that chain through evaluation/__init__.py → detection.py → torch
try:
    from examples.nova.src.evaluation.diagnosis import JudgmentLog
    from examples.nova.src.evaluation.diagnosis import JudgmentRecord
    from examples.nova.src.evaluation.diagnosis import evaluate_diagnosis_nova_official
    from examples.nova.src.evaluation.diagnosis import llm_semantic_match_async

    _HAS_DIAGNOSIS = True
except (ImportError, ModuleNotFoundError):
    _HAS_DIAGNOSIS = False

_skip_no_torch = pytest.mark.skipif(not _HAS_DIAGNOSIS, reason="torch not installed")


# ---------------------------------------------------------------------------
# 1. JudgmentRecord and JudgmentLog
# ---------------------------------------------------------------------------


@_skip_no_torch
class TestJudgmentRecord:
    """Test the JudgmentRecord dataclass."""

    def test_frozen(self) -> None:
        rec = JudgmentRecord(pred="gbm", ref="glioblastoma", method="exact_match", verdict=True)
        with pytest.raises(AttributeError):
            rec.verdict = False  # type: ignore[misc]

    def test_defaults(self) -> None:
        rec = JudgmentRecord(pred="a", ref="b", method="llm", verdict=False)
        assert rec.model == ""
        assert rec.raw_responses == ()
        assert rec.num_votes == 0
        assert rec.vote_counts == (0, 0)

    def test_llm_record_fields(self) -> None:
        rec = JudgmentRecord(
            pred="meningioma",
            ref="meningioma",
            method="llm",
            verdict=True,
            model="gpt-4o",
            raw_responses=("YES",),
            num_votes=1,
            vote_counts=(1, 0),
        )
        assert rec.model == "gpt-4o"
        assert rec.num_votes == 1


@_skip_no_torch
class TestJudgmentLog:
    """Test the JudgmentLog container."""

    def test_add_and_to_dicts(self) -> None:
        log = JudgmentLog()
        log.add(JudgmentRecord(pred="a", ref="b", method="exact_match", verdict=True))
        log.add(
            JudgmentRecord(
                pred="c",
                ref="d",
                method="llm",
                verdict=False,
                model="test-model",
                raw_responses=("NO",),
                num_votes=1,
                vote_counts=(0, 1),
            )
        )
        dicts = log.to_dicts()
        assert len(dicts) == 2
        assert dicts[0]["method"] == "exact_match"
        assert dicts[0]["verdict"] is True
        assert dicts[1]["method"] == "llm"
        assert dicts[1]["verdict"] is False
        assert dicts[1]["model"] == "test-model"
        assert dicts[1]["vote_counts"] == {"yes": 0, "no": 1}

    def test_empty_log(self) -> None:
        log = JudgmentLog()
        assert log.to_dicts() == []


# ---------------------------------------------------------------------------
# 2. llm_semantic_match_async — majority vote
# ---------------------------------------------------------------------------


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Create a mock OpenAI client that returns the given responses in order."""
    client = MagicMock()
    call_idx = 0

    async def _create(**kwargs: object) -> MagicMock:
        nonlocal call_idx
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = responses[call_idx % len(responses)]
        call_idx += 1
        return resp

    client.chat.completions.create = _create
    return client


@_skip_no_torch
class TestLLMSemanticMatchVoting:
    """Test majority vote and judgment record returns."""

    def test_single_vote_returns_record(self) -> None:
        client = _make_mock_client(["YES"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            verdict, record = asyncio.run(
                llm_semantic_match_async("meningioma", "meningioma", num_votes=1)
            )
        assert verdict is True
        assert record.method == "llm"
        assert record.num_votes == 1
        assert record.vote_counts == (1, 0)
        assert record.raw_responses == ("YES",)

    def test_single_vote_no(self) -> None:
        client = _make_mock_client(["NO"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            verdict, record = asyncio.run(llm_semantic_match_async("tumor", "cyst", num_votes=1))
        assert verdict is False
        assert record.vote_counts == (0, 1)

    def test_majority_vote_3_unanimous_yes(self) -> None:
        client = _make_mock_client(["YES", "YES", "YES"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            verdict, record = asyncio.run(
                llm_semantic_match_async("gbm", "glioblastoma multiforme", num_votes=3)
            )
        assert verdict is True
        assert record.num_votes == 3
        assert record.vote_counts == (3, 0)
        assert len(record.raw_responses) == 3

    def test_majority_vote_3_split_2_1(self) -> None:
        client = _make_mock_client(["YES", "NO", "YES"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            verdict, record = asyncio.run(
                llm_semantic_match_async("meningioma", "schwannoma", num_votes=3)
            )
        assert verdict is True  # 2 YES > 1 NO
        assert record.vote_counts == (2, 1)

    def test_majority_vote_3_split_1_2(self) -> None:
        client = _make_mock_client(["YES", "NO", "NO"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            verdict, record = asyncio.run(llm_semantic_match_async("tumor", "cyst", num_votes=3))
        assert verdict is False  # 1 YES < 2 NO
        assert record.vote_counts == (1, 2)

    def test_invalid_num_votes_zero(self) -> None:
        with pytest.raises(ValueError, match="num_votes must be >= 1"):
            asyncio.run(llm_semantic_match_async("a", "b", num_votes=0))

    def test_invalid_num_votes_even(self) -> None:
        with pytest.raises(ValueError, match="num_votes must be odd"):
            asyncio.run(llm_semantic_match_async("a", "b", num_votes=2))


# ---------------------------------------------------------------------------
# 3. evaluate_diagnosis_nova_official — judgment log in results
# ---------------------------------------------------------------------------


@_skip_no_torch
class TestEvaluateDiagnosisJudgmentLog:
    """Test that evaluate_diagnosis_nova_official returns a judgment_log."""

    def test_exact_match_logged(self) -> None:
        """Exact matches should appear in judgment_log with method=exact_match."""
        results = asyncio.run(
            evaluate_diagnosis_nova_official(
                preds=["glioblastoma", "meningioma"],
                refs=["glioblastoma", "meningioma"],
            )
        )
        assert "judgment_log" in results
        log = results["judgment_log"]
        assert len(log) == 2
        assert all(entry["method"] == "exact_match" for entry in log)
        assert all(entry["verdict"] is True for entry in log)

    def test_synonym_match_logged(self) -> None:
        """Synonym matches should appear with method=synonym_match."""
        results = asyncio.run(
            evaluate_diagnosis_nova_official(
                preds=["acoustic neuroma"],
                refs=["vestibular schwannoma"],
            )
        )
        log = results["judgment_log"]
        assert len(log) == 1
        assert log[0]["method"] == "synonym_match"
        assert log[0]["verdict"] is True

    def test_llm_match_logged(self) -> None:
        """LLM matches should appear with method=llm and include model info."""
        client = _make_mock_client(["YES"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            results = asyncio.run(
                evaluate_diagnosis_nova_official(
                    preds=["brain tumor"],
                    refs=["cerebral neoplasm"],
                )
            )
        log = results["judgment_log"]
        assert len(log) == 1
        assert log[0]["method"] == "llm"
        assert log[0]["verdict"] is True
        assert log[0]["model"] != ""

    def test_empty_returns_empty_log(self) -> None:
        results = asyncio.run(evaluate_diagnosis_nova_official(preds=[], refs=[]))
        assert results["judgment_log"] == []

    def test_mixed_methods_logged(self) -> None:
        """Mix of exact, synonym, and LLM matches all appear in log."""
        client = _make_mock_client(["NO"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            results = asyncio.run(
                evaluate_diagnosis_nova_official(
                    preds=["glioblastoma", "acoustic neuroma", "brain tumor"],
                    refs=["glioblastoma", "vestibular schwannoma", "cyst"],
                )
            )
        log = results["judgment_log"]
        assert len(log) == 3
        methods = {entry["method"] for entry in log}
        assert "exact_match" in methods
        assert "synonym_match" in methods
        assert "llm" in methods

    def test_num_votes_passed_through(self) -> None:
        """num_votes parameter is forwarded to LLM calls."""
        client = _make_mock_client(["YES", "YES", "NO"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            results = asyncio.run(
                evaluate_diagnosis_nova_official(
                    preds=["brain tumor"],
                    refs=["cerebral neoplasm"],
                    num_votes=3,
                )
            )
        log = results["judgment_log"]
        assert len(log) == 1
        assert log[0]["num_votes"] == 3
        assert log[0]["vote_counts"]["yes"] + log[0]["vote_counts"]["no"] == 3

    def test_metrics_still_correct(self) -> None:
        """Adding judgment logging doesn't break metric computation."""
        client = _make_mock_client(["NO"])
        with patch(
            "examples.nova.src.evaluation.diagnosis._get_semantic_match_client",
            return_value=client,
        ):
            results = asyncio.run(
                evaluate_diagnosis_nova_official(
                    preds=["glioblastoma", "meningioma", "migraine"],
                    refs=["glioblastoma", "meningioma", "arachnoid cyst"],
                )
            )
        # First two are exact matches, third is LLM → NO
        assert results["top1"] == pytest.approx(2 / 3)
        assert "judgment_log" in results
        assert len(results["judgment_log"]) == 3


# NOTE: Evidence-tier tests are in test_web_search.py
# NOTE: Path traversal tests are in test_security.py
