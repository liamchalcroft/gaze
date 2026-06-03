"""Tests that core data types enforce immutability contracts.

Validates that frozen dataclasses with immutable containers (tuples,
MappingProxyType) reject mutation attempts at runtime.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from gaze.models.adapter_protocol import GenerationLog
from gaze.types import AgenticResult
from gaze.types import ToolCall
from gaze.types import ToolResult
from gaze.types import Turn


class TestGenerationLogFrozen:
    """GenerationLog must be frozen — attribute assignment should raise."""

    def test_cannot_reassign_prompt_tokens(self) -> None:
        log = GenerationLog(prompt_tokens=10, completion_tokens=20, finish_reason="stop")
        with pytest.raises(FrozenInstanceError):
            log.prompt_tokens = 99  # type: ignore[misc]

    def test_cannot_reassign_completion_tokens(self) -> None:
        log = GenerationLog(prompt_tokens=10, completion_tokens=20, finish_reason="stop")
        with pytest.raises(FrozenInstanceError):
            log.completion_tokens = 99  # type: ignore[misc]

    def test_cannot_reassign_finish_reason(self) -> None:
        log = GenerationLog(prompt_tokens=10, completion_tokens=20, finish_reason="stop")
        with pytest.raises(FrozenInstanceError):
            log.finish_reason = "length"  # type: ignore[misc]

    def test_tokens_property(self) -> None:
        log = GenerationLog(prompt_tokens=10, completion_tokens=20, finish_reason="stop")
        assert log.tokens == 30


class TestTurnImmutability:
    """Turn.tool_calls and Turn.tool_results must be immutable tuples."""

    def test_tool_calls_is_tuple(self) -> None:
        tc = ToolCall(id="1", name="zoom", arguments="{}")
        turn = Turn(role="assistant", content="test", tool_calls=[tc])
        assert isinstance(turn.tool_calls, tuple)
        assert turn.tool_calls == (tc,)

    def test_tool_results_is_tuple(self) -> None:
        tr = ToolResult(tool_name="zoom", description="done")
        turn = Turn(role="tool_result", content="test", tool_results=[tr])
        assert isinstance(turn.tool_results, tuple)
        assert turn.tool_results == (tr,)

    def test_tool_calls_already_tuple(self) -> None:
        tc = ToolCall(id="1", name="zoom", arguments="{}")
        turn = Turn(role="assistant", content="test", tool_calls=(tc,))
        assert isinstance(turn.tool_calls, tuple)
        assert turn.tool_calls == (tc,)

    def test_tool_calls_cannot_be_mutated(self) -> None:
        """tuple does not support append — this is the key invariant."""
        turn = Turn(role="assistant", content="test")
        assert turn.tool_calls == ()
        with pytest.raises(AttributeError):
            turn.tool_calls.append(ToolCall(id="1", name="zoom", arguments="{}"))  # type: ignore[attr-defined]

    def test_tool_results_cannot_be_mutated(self) -> None:
        turn = Turn(role="tool_result", content="test")
        assert turn.tool_results == ()
        with pytest.raises(AttributeError):
            turn.tool_results.append(ToolResult(tool_name="zoom", description="done"))  # type: ignore[attr-defined]

    def test_cannot_reassign_role(self) -> None:
        turn = Turn(role="user", content="test")
        with pytest.raises(FrozenInstanceError):
            turn.role = "assistant"  # type: ignore[misc]

    def test_default_empty_tuples(self) -> None:
        turn = Turn(role="user", content="hello")
        assert turn.tool_calls == ()
        assert turn.tool_results == ()


class TestToolCallImmutability:
    """ToolCall.arguments must freeze JSON-object inputs."""

    def test_dict_arguments_are_mapping_proxy(self) -> None:
        tc = ToolCall(id="1", name="zoom", arguments={"x": 1, "y": 2})
        assert isinstance(tc.arguments, MappingProxyType)
        assert tc.arguments["x"] == 1

    def test_dict_arguments_cannot_be_mutated(self) -> None:
        tc = ToolCall(id="1", name="zoom", arguments={"x": 1})
        with pytest.raises(TypeError):
            tc.arguments["x"] = 2  # type: ignore[index]

    def test_nested_arguments_are_deep_frozen(self) -> None:
        raw = {"box": [1, 2, {"x": 3}]}
        tc = ToolCall(id="1", name="crop", arguments=raw)
        raw["box"][2]["x"] = 9
        assert tc.arguments["box"][2]["x"] == 3
        with pytest.raises(TypeError):
            tc.arguments["box"][2]["x"] = 4  # type: ignore[index]


class TestAgenticResultImmutability:
    """AgenticResult must freeze its mutable inputs."""

    def _make_result(self) -> AgenticResult:
        turn = Turn(role="assistant", content="done")
        return AgenticResult(
            final_response={"result": "ok"},
            turns=[turn],
            total_tokens=100,
            confidence=0.9,
        )

    def test_final_response_is_mapping_proxy(self) -> None:
        result = self._make_result()
        assert isinstance(result.final_response, MappingProxyType)

    def test_final_response_read_access(self) -> None:
        result = self._make_result()
        assert result.final_response["result"] == "ok"

    def test_final_response_get_access(self) -> None:
        result = self._make_result()
        assert result.final_response.get("result") == "ok"
        assert result.final_response.get("missing", "default") == "default"

    def test_final_response_cannot_be_mutated(self) -> None:
        result = self._make_result()
        with pytest.raises(TypeError):
            result.final_response["result"] = "hacked"  # type: ignore[index]

    def test_final_response_pre_wrapped_proxy_is_still_deep_frozen(self) -> None:
        nested = {"inner": {"x": 1}}
        result = AgenticResult(
            final_response=MappingProxyType(nested),
            turns=[Turn(role="assistant", content="done")],
            total_tokens=1,
            confidence=1.0,
        )
        nested["inner"]["x"] = 9
        assert result.final_response["inner"]["x"] == 1

    def test_turns_is_tuple(self) -> None:
        result = self._make_result()
        assert isinstance(result.turns, tuple)
        assert len(result.turns) == 1

    def test_turns_coerces_from_list(self) -> None:
        turn1 = Turn(role="assistant", content="a")
        turn2 = Turn(role="user", content="b")
        result = AgenticResult(
            final_response={"x": 1},
            turns=[turn1, turn2],
            total_tokens=50,
            confidence=0.5,
        )
        assert isinstance(result.turns, tuple)
        assert result.turns == (turn1, turn2)

    def test_turns_cannot_be_mutated(self) -> None:
        result = self._make_result()
        with pytest.raises(AttributeError):
            result.turns.append(Turn(role="user", content="x"))  # type: ignore[attr-defined]

    def test_cannot_reassign_total_tokens(self) -> None:
        result = self._make_result()
        with pytest.raises(FrozenInstanceError):
            result.total_tokens = 0  # type: ignore[misc]

    def test_properties_work(self) -> None:
        tc = ToolCall(id="1", name="zoom", arguments="{}")
        turns = [
            Turn(role="assistant", content="let me zoom", tool_calls=[tc]),
            Turn(
                role="tool_result",
                content="zoomed",
                tool_results=[ToolResult(tool_name="zoom", description="ok")],
            ),
        ]
        result = AgenticResult(
            final_response={"done": True},
            turns=turns,
            total_tokens=200,
            confidence=0.8,
        )
        assert result.num_turns == 2
        assert result.tool_call_count == 1
        assert result.get_tools_used() == {"zoom"}


class TestSearchResultFrozen:
    """SearchResult must be frozen with immutable extracted_entities."""

    def test_cannot_reassign_title(self) -> None:
        from gaze.retrieval.web_search import SearchResult

        sr = SearchResult(
            title="Test",
            url="https://x.com",
            content="c",
            snippet="s",
            source="pubmed",
            reliability_score=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            sr.title = "hacked"  # type: ignore[misc]

    def test_cannot_reassign_ranking_score(self) -> None:
        from gaze.retrieval.web_search import SearchResult

        sr = SearchResult(
            title="Test",
            url="https://x.com",
            content="c",
            snippet="s",
            source="pubmed",
            reliability_score=0.9,
            ranking_score=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            sr.ranking_score = 1.0  # type: ignore[misc]

    def test_extracted_entities_is_tuple(self) -> None:
        from gaze.retrieval.web_search import SearchResult

        sr = SearchResult(
            title="Test",
            url="https://x.com",
            content="c",
            snippet="s",
            source="pubmed",
            reliability_score=0.9,
            extracted_entities=["a", "b"],
        )
        assert isinstance(sr.extracted_entities, tuple)
        assert sr.extracted_entities == ("a", "b")

    def test_extracted_entities_cannot_be_mutated(self) -> None:
        from gaze.retrieval.web_search import SearchResult

        sr = SearchResult(
            title="Test",
            url="https://x.com",
            content="c",
            snippet="s",
            source="pubmed",
            reliability_score=0.9,
            extracted_entities=["a"],
        )
        with pytest.raises(AttributeError):
            sr.extracted_entities.append("c")  # type: ignore[attr-defined]

    def test_default_entities_empty_tuple(self) -> None:
        from gaze.retrieval.web_search import SearchResult

        sr = SearchResult(
            title="Test",
            url="https://x.com",
            content="c",
            snippet="s",
            source="pubmed",
            reliability_score=0.9,
        )
        assert sr.extracted_entities == ()


class TestImageSearchResultFrozen:
    """ImageSearchResult must be frozen with immutable metadata."""

    def test_cannot_reassign_title(self) -> None:
        from gaze.retrieval.image_search import ImageSearchResult

        isr = ImageSearchResult(
            title="Test",
            image_url="https://x.com/img.png",
            thumbnail_url=None,
            source_url="https://x.com",
            source="openi",
        )
        with pytest.raises(FrozenInstanceError):
            isr.title = "hacked"  # type: ignore[misc]

    def test_metadata_is_mapping_proxy(self) -> None:
        from gaze.retrieval.image_search import ImageSearchResult

        isr = ImageSearchResult(
            title="Test",
            image_url="https://x.com/img.png",
            thumbnail_url=None,
            source_url="https://x.com",
            source="openi",
            metadata={"key": "val"},
        )
        assert isinstance(isr.metadata, MappingProxyType)
        assert isr.metadata["key"] == "val"

    def test_metadata_cannot_be_mutated(self) -> None:
        from gaze.retrieval.image_search import ImageSearchResult

        isr = ImageSearchResult(
            title="Test",
            image_url="https://x.com/img.png",
            thumbnail_url=None,
            source_url="https://x.com",
            source="openi",
            metadata={"key": "val"},
        )
        with pytest.raises(TypeError):
            isr.metadata["key"] = "new"  # type: ignore[index]

    def test_default_metadata_empty(self) -> None:
        from gaze.retrieval.image_search import ImageSearchResult

        isr = ImageSearchResult(
            title="Test",
            image_url="https://x.com/img.png",
            thumbnail_url=None,
            source_url="https://x.com",
            source="openi",
        )
        assert isinstance(isr.metadata, MappingProxyType)
        assert len(isr.metadata) == 0

    def test_nested_metadata_is_deep_frozen(self) -> None:
        from gaze.retrieval.image_search import ImageSearchResult

        nested = {"outer": {"x": 1}}
        isr = ImageSearchResult(
            title="Test",
            image_url="https://x.com/img.png",
            thumbnail_url=None,
            source_url="https://x.com",
            source="openi",
            metadata=MappingProxyType(nested),
        )
        nested["outer"]["x"] = 9
        assert isr.metadata["outer"]["x"] == 1


class TestToolResultMetadataFrozen:
    """ToolResult.metadata must be a MappingProxyType."""

    def test_metadata_is_frozen(self) -> None:
        tr = ToolResult(tool_name="zoom", description="done", metadata={"key": "val"})
        assert isinstance(tr.metadata, MappingProxyType)
        with pytest.raises(TypeError):
            tr.metadata["key"] = "new"  # type: ignore[index]

    def test_metadata_already_proxy(self) -> None:
        proxy = MappingProxyType({"a": {"x": 1}})
        tr = ToolResult(tool_name="zoom", description="done", metadata=proxy)
        assert isinstance(tr.metadata, MappingProxyType)
        with pytest.raises(TypeError):
            tr.metadata["a"]["x"] = 2  # type: ignore[index]
